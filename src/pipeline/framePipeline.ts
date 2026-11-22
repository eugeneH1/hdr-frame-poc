/**
 * Frame pipeline orchestrator.
 *
 * Responsibility: coordinate deterministic generation of frames `0..frameCount-1`
 * for a single run — evaluate the timeline, render the HDR target, present the
 * preview, read back float pixels, and hand each `FloatFrame` to a consumer and
 * an optional downstream WebCodecs tap. It owns run lifecycle and cancellation
 * only; it contains no shader or math logic.
 *
 * Design assumptions:
 * - **Bounded memory:** frames are rendered ahead by at most
 *   `maxFramesInFlight` (default 2), read back, delivered via `onFrame`, and
 *   released. A 1280x720 CPU frame is ~14.75 MB; the default keeps at most two
 *   staging buffers and two CPU payloads live at once. Enable
 *   `readback.reuseOutputBuffer` to trade that safety for zero per-frame CPU
 *   allocation (the pipeline then forces depth 1).
 * - **Determinism:** frame N depends only on `getSceneAtFrame(N, config)`, never
 *   on wall-clock time, so a slow preview, encoder, or a cancelled run cannot
 *   change the content of frames that *were* produced.
 * - **Overlap:** after frame N's render is submitted, its `copyTextureToBuffer`
 *   is submitted *before* frame N+1 is recorded. The GPU copies execute in queue
 *   order, while mapping/unpacking frame N can overlap rendering frame N+1.
 * - **Back-pressure:** `onFrame` is awaited before another frame is delivered,
 *   and the readback lookahead is capped, keeping memory bounded.
 * - **Cancellation:** cooperative via `AbortSignal`; checked before each new
 *   frame. Readbacks already in flight are drained and delivered, pending
 *   staging jobs are cancelled safely, and the optional encoder is flushed.
 * - **Telemetry:** cheap sampled pixel statistics and per-stage timings are
 *   computed per frame, but `onTelemetry` is throttled (plus forced samples at
 *   start/finish) so the UI is never flooded.
 */

import { destroyGpuContext, initGpuContext } from '../gpu/context.ts';
import type { GpuContext } from '../gpu/context.ts';
import { createHdrRenderer } from '../gpu/hdrRenderer.ts';
import type { HdrRenderer } from '../gpu/hdrRenderer.ts';
import { createPreviewRenderer } from '../gpu/previewRenderer.ts';
import type { PreviewRenderer } from '../gpu/previewRenderer.ts';
import { beginFrameReadback, createReadbackManager } from '../gpu/readback.ts';
import type { ReadbackJob, ReadbackManager, ReadbackOptions } from '../gpu/readback.ts';
import { getSceneAtFrame } from '../timeline/sceneAnimator.ts';
import { encodeFrame, flushAndClose } from '../encoder/webCodecsTap.ts';
import type { VideoEncoderTap } from '../encoder/webCodecsTap.ts';
import type {
  EncoderTelemetry,
  FloatFrame,
  FrameTimingBreakdown,
  FrameValueStats,
  RenderConfig,
  TelemetrySnapshot,
} from '../types/frame.ts';
import { BYTES_PER_PIXEL_F32 } from '../types/frame.ts';

/** Minimum wall-clock gap between throttled telemetry callbacks. */
const TELEMETRY_INTERVAL_MS = 100;
/** Pixel stride for min/max telemetry sampling (keeps the scan cheap). */
const STATS_PIXEL_STRIDE = 16;
/** Number of completion timestamps used to estimate achieved FPS. */
const FPS_WINDOW_SIZE = 30;
/** Maximum warnings retained in a telemetry snapshot. */
const MAX_WARNINGS = 5;
/** Default readback lookahead: one GPU copy overlapping one CPU unpack. */
const DEFAULT_MAX_FRAMES_IN_FLIGHT = 2;

/** Metadata reported when a run starts. */
export interface PipelineStartInfo {
  /** Total number of frames that will be produced. */
  readonly frameCount: number;
  /** Output dimensions in pixels. */
  readonly width: number;
  /** Output dimensions in pixels. */
  readonly height: number;
}

/** Summary returned when a run finishes or is cancelled. */
export interface PipelineSummary {
  /** Frames produced and delivered before completion/cancellation. */
  readonly framesCompleted: number;
  /** Whether the run was stopped by the abort signal. */
  readonly cancelled: boolean;
  /** Total wall-clock milliseconds for the run, including encoder flush. */
  readonly elapsedMs: number;
  /** Final telemetry snapshot. */
  readonly telemetry: TelemetrySnapshot;
}

/**
 * Hooks the pipeline (and the workbench) uses to report progress. All callbacks
 * are optional; the pipeline behaves identically when any are omitted.
 */
export interface PipelineCallbacks {
  /** Called once before the first frame is processed. */
  onStart?: (info: PipelineStartInfo) => void;
  /**
   * Receives each completed frame, in ascending `frameIndex` order. May return a
   * promise; the pipeline awaits it before producing the next frame (back-pressure).
   */
  onFrame?: (frame: FloatFrame) => void | Promise<void>;
  /** Receives a telemetry sample, typically every frame or throttled. */
  onTelemetry?: (snapshot: TelemetrySnapshot) => void;
  /** Non-fatal diagnostics (e.g. skipped preview). Must not abort the run. */
  onWarning?: (message: string) => void;
}

/** Readback tuning for one run. */
export interface PipelineReadbackOptions extends ReadbackOptions {
  /**
   * Maximum frames whose readback may be in flight at once (default `2`).
   * Clamped to `[1, stagingBufferCount]`. Forced to `1` when
   * `reuseOutputBuffer` is enabled, because two live frames would alias the
   * same CPU array.
   */
  readonly maxFramesInFlight?: number;
}

/** Options controlling a single pipeline run. */
export interface RunFramePipelineOptions {
  /** Immutable run configuration (dimensions, fps, frameCount, color spaces). */
  readonly config: RenderConfig;
  /**
   * Canvas used for live preview presentation. Omit to run headless (readback only),
   * which is useful for tests and batch export.
   */
  readonly previewCanvas?: HTMLCanvasElement;
  /** Progress hooks. */
  readonly callbacks?: PipelineCallbacks;
  /** Cooperative cancellation; checked between frames. */
  readonly signal?: AbortSignal;
  /** Staging-ring and CPU-output reuse policy. */
  readonly readback?: PipelineReadbackOptions;
  /**
   * Optional downstream WebCodecs consumer. The pipeline calls `encodeFrame`
   * for every delivered frame and always flushes/closes the tap on completion,
   * cancellation, or error. The caller creates and owns the tap.
   */
  readonly encoderTap?: VideoEncoderTap | null;
}

/** One submitted readback plus the render timings that led to it. */
interface PendingReadback {
  readonly frameIndex: number;
  readonly job: ReadbackJob;
  readonly startedAt: number;
  readonly renderMs: number;
  readonly previewMs: number;
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function toEncoderTelemetry(tap: VideoEncoderTap): EncoderTelemetry {
  const status = tap.status();
  return {
    state: status.state,
    codec: status.codec,
    frameFormat: status.frameFormat,
    colorSpace: status.colorSpace,
    framesSubmitted: status.framesSubmitted,
    framesClosed: status.framesClosed,
    chunksEmitted: status.chunksEmitted,
    chunkBytes: status.chunkBytes,
    keyFrames: status.keyFrames,
    encodeQueueSize: status.encodeQueueSize,
    lastError: status.lastError,
  };
}

/**
 * Samples RGB channels on a fixed pixel stride and reports the min/max plus one
 * full center pixel. Scanning every pixel would cost ~3.7M float reads per frame;
 * this keeps telemetry overhead negligible while still exposing HDR peaks.
 */
function summarizePixels(pixels: Float32Array): FrameValueStats {
  const pixelCount = Math.floor(pixels.length / 4);
  if (pixelCount === 0) {
    return { min: 0, max: 0, sample: [0, 0, 0, 0], sampledPixels: 0 };
  }

  let min = Number.POSITIVE_INFINITY;
  let max = Number.NEGATIVE_INFINITY;
  let sampledPixels = 0;

  for (let pixel = 0; pixel < pixelCount; pixel += STATS_PIXEL_STRIDE) {
    const base = pixel * 4;
    for (let channel = 0; channel < 3; channel++) {
      const value = pixels[base + channel];
      if (value < min) min = value;
      if (value > max) max = value;
    }
    sampledPixels++;
  }

  const center = Math.floor(pixelCount / 2) * 4;

  return {
    min: Number.isFinite(min) ? min : 0,
    max: Number.isFinite(max) ? max : 0,
    sample: [pixels[center], pixels[center + 1], pixels[center + 2], pixels[center + 3]],
    sampledPixels,
  };
}

/**
 * Runs the full deterministic frame pipeline.
 *
 * Per frame `N`:
 * 1. `getSceneAtFrame(N, config)` → linear Rec. 2020 scene (`animationTime` is
 *    exactly `N / config.fps`).
 * 2. Render into the offscreen float HDR target and record the optional preview.
 * 3. Submit scene + preview, then submit frame N's readback copy immediately.
 * 4. Let that copy overlap the next frame's render/unpack; await `finish()`
 *    before delivering frame N.
 * 5. Feed the decoded frame to the optional WebCodecs tap and `onFrame`, then
 *    `release()` it even if a callback throws.
 *
 * On completion or cancellation, resources are disposed, the encoder is
 * flushed, and a `PipelineSummary` is returned. A fatal error rejects after
 * cleanup.
 *
 * @param options - Run configuration, preview target, callbacks, readback
 *   tuning, encoder tap, and abort signal.
 * @returns Summary of the run including final telemetry.
 */
export async function runFramePipeline(
  options: RunFramePipelineOptions,
): Promise<PipelineSummary> {
  const { config, callbacks, signal } = options;
  const encoder = options.encoderTap ?? null;
  const runStart = performance.now();

  const warnings: string[] = [];
  const completionTimes: number[] = [];
  const pending: PendingReadback[] = [];

  let framesCompleted = 0;
  let lastFrameIndex = -1;
  let lastFrameMs = 0;
  let averageFrameMs = 0;
  let measuredFps = 0;
  let readbackBytes = 0;
  let retainedBytes = 0;
  let lastFrameValues: FrameValueStats | undefined;
  let renderFormat: GPUTextureFormat | undefined;
  let lastTelemetryAt = Number.NEGATIVE_INFINITY;
  let cancelled = false;
  let encoderFlushed = false;
  let encoderWarningIssued = false;

  let lastFrameTimings: FrameTimingBreakdown | undefined;
  let averageFrameTimings: FrameTimingBreakdown | undefined;
  const timingTotals: MutableTimings = {
    renderMs: 0,
    previewMs: 0,
    readbackMs: 0,
    unpackMs: 0,
    encodeMs: 0,
    consumerMs: 0,
  };

  function addWarning(message: string): void {
    warnings.push(message);
    if (warnings.length > MAX_WARNINGS) warnings.shift();
    callbacks?.onWarning?.(message);
  }

  function buildTelemetry(): TelemetrySnapshot {
    return {
      framesCompleted,
      framesTotal: config.frameCount,
      measuredFps,
      lastFrameMs,
      averageFrameMs,
      readbackBytes,
      // Released frames are not retained. A shared output buffer is retained by
      // design and reported here so the cost of `reuseOutputBuffer` is visible.
      retainedBytes,
      warnings: warnings.slice(),
      lastFrameIndex,
      elapsedMs: performance.now() - runStart,
      renderFormat,
      lastFrameValues,
      lastFrameTimings,
      averageFrameTimings,
      encoder: encoder ? toEncoderTelemetry(encoder) : undefined,
    };
  }

  function emitTelemetry(force: boolean): void {
    const now = performance.now();
    if (!force && now - lastTelemetryAt < TELEMETRY_INTERVAL_MS) return;
    lastTelemetryAt = now;
    callbacks?.onTelemetry?.(buildTelemetry());
  }

  if (signal?.aborted) {
    await flushAndClose(encoder);
    encoderFlushed = true;
    return {
      framesCompleted: 0,
      cancelled: true,
      elapsedMs: 0,
      telemetry: buildTelemetry(),
    };
  }

  let context: GpuContext;
  try {
    context = await initGpuContext(config);
  } catch (error) {
    await flushAndClose(encoder);
    throw error;
  }
  renderFormat = context.renderFormat;

  let renderer: HdrRenderer | null = null;
  let readbackManager: ReadbackManager | null = null;
  let preview: PreviewRenderer | null = null;
  let detachDeviceLost: (() => void) | null = null;

  function recordTimings(breakdown: FrameTimingBreakdown): void {
    lastFrameTimings = breakdown;
    timingTotals.renderMs += breakdown.renderMs;
    timingTotals.previewMs += breakdown.previewMs;
    timingTotals.readbackMs += breakdown.readbackMs;
    timingTotals.unpackMs += breakdown.unpackMs;
    timingTotals.encodeMs += breakdown.encodeMs;
    timingTotals.consumerMs += breakdown.consumerMs;

    if (framesCompleted > 0) {
      const n = framesCompleted;
      averageFrameTimings = {
        renderMs: timingTotals.renderMs / n,
        previewMs: timingTotals.previewMs / n,
        readbackMs: timingTotals.readbackMs / n,
        unpackMs: timingTotals.unpackMs / n,
        encodeMs: timingTotals.encodeMs / n,
        consumerMs: timingTotals.consumerMs / n,
        // `totalMs` is wall-clock per delivered frame rather than the sum of
        // overlapping per-frame spans, so it stays comparable to the FPS cell.
        totalMs: averageFrameMs,
      };
    }
  }

  async function deliver(item: PendingReadback): Promise<void> {
    const manager = readbackManager;
    if (!manager) {
      throw new Error('pipeline: readback manager is not initialized');
    }

    const rawFrame = await item.job.finish();
    try {
      const readback = rawFrame.readback;
      const readbackMs = readback?.waitMs ?? 0;
      const unpackMs = readback?.unpackMs ?? 0;

      lastFrameValues = summarizePixels(rawFrame.pixels);
      framesCompleted++;
      lastFrameIndex = item.frameIndex;
      readbackBytes = framesCompleted * manager.bufferSize;

      completionTimes.push(performance.now());
      if (completionTimes.length > FPS_WINDOW_SIZE + 1) {
        completionTimes.shift();
      }
      if (completionTimes.length >= 2) {
        const first = completionTimes[0];
        const last = completionTimes[completionTimes.length - 1];
        const spanMs = last - first;
        measuredFps = spanMs > 0 ? ((completionTimes.length - 1) * 1000) / spanMs : 0;
      }

      // The tap may use pixels before `onFrame`; both see the same immutable
      // staging-free CPU payload. `renderTimeMs` is filled in after encoding.
      const frameForEncode: FloatFrame = {
        meta: { ...rawFrame.meta, renderTimeMs: 0 },
        pixels: rawFrame.pixels,
        bytesPerRow: rawFrame.bytesPerRow,
        readback,
        release: () => rawFrame.release(),
      };

      let encodeMs = 0;
      if (encoder) {
        const encodeStart = performance.now();
        const accepted = encodeFrame(encoder, frameForEncode, item.frameIndex === 0);
        encodeMs = performance.now() - encodeStart;

        if (!accepted && !encoderWarningIssued) {
          encoderWarningIssued = true;
          let lastError: string | null = null;
          try {
            lastError = encoder.status().lastError;
          } catch {
            // Status is diagnostics only; never let it break the run.
          }
          addWarning(
            `VideoEncoder tap skipped frame ${item.frameIndex}: ${lastError ?? 'encoder is closed or errored'}`,
          );
        }
      }

      const productionMs =
        item.renderMs + item.previewMs + readbackMs + unpackMs + encodeMs;
      const delivered: FloatFrame = {
        meta: { ...rawFrame.meta, renderTimeMs: productionMs },
        pixels: rawFrame.pixels,
        bytesPerRow: rawFrame.bytesPerRow,
        readback,
        release: () => rawFrame.release(),
      };

      let consumerMs = 0;
      const consumerStart = performance.now();
      try {
        await callbacks?.onFrame?.(delivered);
      } finally {
        consumerMs = performance.now() - consumerStart;
        // Release even if the consumer throws or cancellation arrives during the
        // callback; the next iteration may not run at all.
        delivered.release();
      }

      const totalMs = performance.now() - item.startedAt;
      lastFrameMs = totalMs;
      averageFrameMs = (performance.now() - runStart) / framesCompleted;

      recordTimings({
        renderMs: item.renderMs,
        previewMs: item.previewMs,
        readbackMs,
        unpackMs,
        encodeMs,
        consumerMs,
        totalMs,
      });

      emitTelemetry(item.frameIndex === 0);
    } finally {
      // Safety net for unexpected failures before `delivered.release()` is
      // reached; releasing an already-released frame is idempotent.
      rawFrame.release();
    }
  }

  try {
    detachDeviceLost = context.onDeviceLost((reason) => {
      addWarning(`GPU device lost: ${reason}`);
    });

    // Keep non-null locals for the hot loop while still assigning to the
    // nullable handles that the `finally` block disposes on every exit path.
    const hdrRenderer = createHdrRenderer(context, config);
    renderer = hdrRenderer;

    const manager = createReadbackManager(context, config, options.readback);
    readbackManager = manager;

    const requestedDepth =
      options.readback?.maxFramesInFlight ?? DEFAULT_MAX_FRAMES_IN_FLIGHT;
    const depth = Number.isFinite(requestedDepth)
      ? Math.floor(requestedDepth)
      : DEFAULT_MAX_FRAMES_IN_FLIGHT;
    const maxFramesInFlight = manager.reuseOutputBuffer
      ? 1
      : Math.max(1, Math.min(depth, manager.stagingBufferCount));
    retainedBytes = manager.reuseOutputBuffer
      ? config.width * config.height * BYTES_PER_PIXEL_F32
      : 0;

    if (options.previewCanvas) {
      try {
        preview = createPreviewRenderer(context, options.previewCanvas);
      } catch (error) {
        addWarning(`Preview disabled: ${describeError(error)}`);
        preview = null;
      }
    }

    const hdrTargetView = hdrRenderer.target.createView();
    const { device } = context;

    callbacks?.onStart?.({
      frameCount: config.frameCount,
      width: config.width,
      height: config.height,
    });
    emitTelemetry(true);

    for (let frameIndex = 0; frameIndex < config.frameCount; frameIndex++) {
      if (signal?.aborted) {
        cancelled = true;
        break;
      }

      // Bounded lookahead: deliver the oldest frame before starting another.
      if (pending.length >= maxFramesInFlight) {
        const oldest = pending.shift();
        if (oldest) await deliver(oldest);
      }

      if (signal?.aborted) {
        cancelled = true;
        break;
      }

      const frameStart = performance.now();
      // `scene.animationTime` is exactly `frameIndex / config.fps`; the timeline
      // is a pure function of the index, so preview/readback timing cannot alter
      // the content of any frame that is produced.
      const scene = getSceneAtFrame(frameIndex, config);

      // Scene and preview share one submission and one command encoder, so the
      // swap-chain texture is acquired exactly once per frame.
      const commandEncoder = device.createCommandEncoder({
        label: `frame-${frameIndex}-encoder`,
      });
      hdrRenderer.renderSceneToTexture(commandEncoder, scene);
      const beforePreview = performance.now();

      let previewMs = 0;
      if (preview) {
        const previewStart = performance.now();
        try {
          preview.renderPreview(commandEncoder, hdrTargetView);
        } catch (error) {
          // Preview is best-effort: disable it and keep producing frames.
          addWarning(`Preview disabled: ${describeError(error)}`);
          preview.dispose();
          preview = null;
        }
        previewMs = performance.now() - previewStart;
      }

      const submitStart = performance.now();
      device.queue.submit([commandEncoder.finish()]);
      const submitMs = performance.now() - submitStart;

      // The readback copy is submitted before the next frame can overwrite the
      // target. With depth >= 2, its map/unpack overlaps the next render.
      const job = beginFrameReadback(context, manager, hdrRenderer, frameIndex, config);
      pending.push({
        frameIndex,
        job,
        startedAt: frameStart,
        renderMs: beforePreview - frameStart + submitMs,
        previewMs,
      });
    }

    cancelled = cancelled || (signal?.aborted ?? false);

    // Drain in delivery order. Frames whose copies were already submitted are
    // still valid output even if cancellation arrived while they were in flight.
    while (pending.length > 0) {
      const item = pending.shift();
      if (!item) break;
      await deliver(item);
    }

    await flushAndClose(encoder);
    encoderFlushed = true;

    const finalTelemetry = buildTelemetry();
    callbacks?.onTelemetry?.(finalTelemetry);

    return {
      framesCompleted,
      cancelled,
      elapsedMs: performance.now() - runStart,
      telemetry: finalTelemetry,
    };
  } catch (error) {
    for (const item of pending) {
      item.job.cancel();
    }
    pending.length = 0;
    throw error;
  } finally {
    if (!encoderFlushed) {
      try {
        await flushAndClose(encoder);
      } catch (error) {
        addWarning(`VideoEncoder flush failed: ${describeError(error)}`);
      }
    }
    detachDeviceLost?.();
    preview?.dispose();
    readbackManager?.dispose();
    renderer?.dispose();
    destroyGpuContext(context);
  }
}

interface MutableTimings {
  renderMs: number;
  previewMs: number;
  readbackMs: number;
  unpackMs: number;
  encodeMs: number;
  consumerMs: number;
}
