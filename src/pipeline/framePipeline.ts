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
import { describeLogging, logDebug, logError, logInfo, logWarn } from '../diagnostics/logger.ts';

/** Minimum wall-clock gap between throttled telemetry callbacks. */
const TELEMETRY_INTERVAL_MS = 100;
/** Pixel stride for min/max telemetry sampling (keeps the scan cheap). */
const STATS_PIXEL_STRIDE = 16;
/** Number of completion timestamps used to estimate achieved FPS. */
const FPS_WINDOW_SIZE = 30;
/** Maximum warnings retained in a telemetry snapshot. */
const MAX_WARNINGS = 5;
/** Default readback lookahead: staging copies overlapping CPU unpack. */
const DEFAULT_MAX_FRAMES_IN_FLIGHT = 3;
/** How often an await that should be quick logs its diagnostic state. */
const STALL_LOG_INTERVAL_MS = 5000;
/** Encoder queue depth at which the pipeline reports encoding back-pressure. */
const ENCODER_QUEUE_WARN_THRESHOLD = 48;
/** Minimum gap between encoder back-pressure warnings. */
const ENCODER_QUEUE_WARN_INTERVAL_MS = 2000;
/**
 * Encoder queue depth at which the pipeline pauses before submitting another
 * frame and waits for the encoder to catch up. Set with ample headroom (64)
 * so normal runs never stutter while still bounding memory on long jobs.
 */
const ENCODER_QUEUE_DRAIN_THRESHOLD = 64;
/** Target queue depth before resuming submission after back-pressure. */
const ENCODER_QUEUE_RESUME_THRESHOLD = 32;
/** Upper bound on the final flush; a wedged encoder is force-closed after this. */
const ENCODER_FLUSH_TIMEOUT_MS = 30_000;

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

  let encoderCodec = 'off';
  if (encoder) {
    try {
      encoderCodec = encoder.status().codec;
    } catch {
      encoderCodec = 'unavailable';
    }
  }

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
  let deviceLostReason: string | null = null;
  let lastEncoderQueueWarningAt = Number.NEGATIVE_INFINITY;
  /** Exact pipeline step for failure/stall reports. */
  let currentStage = 'startup';
  /** Frame being produced (may be ahead of `lastFrameIndex` while in flight). */
  let currentFrameIndex = -1;

  let lastFrameTimings: FrameTimingBreakdown | undefined;
  let averageFrameTimings: FrameTimingBreakdown | undefined;
  const timingTotals: MutableTimings = {
    renderMs: 0,
    previewMs: 0,
    readbackMs: 0,
    unpackMs: 0,
    encodeMs: 0,
    consumerMs: 0,
    totalMs: 0,
  };

  function addWarning(message: string): void {
    // Console first: the UI only shows the latest warning and telemetry keeps
    // at most MAX_WARNINGS, so a console line is the only durable record.
    logWarn('pipeline', message);
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

  logInfo('pipeline', 'Run requested', {
    width: config.width,
    height: config.height,
    fps: config.fps,
    frameCount: config.frameCount,
    requestedFormat: config.renderFormat,
    preview: Boolean(options.previewCanvas),
    encoder: encoderCodec,
    maxFramesInFlight:
      options.readback?.maxFramesInFlight ?? DEFAULT_MAX_FRAMES_IN_FLIGHT,
    logging: describeLogging(),
  });

  if (signal?.aborted) {
    logInfo('pipeline', 'Abort signal was already aborted; no frames will be produced');
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
  currentStage = 'gpu-init';
  logInfo('pipeline', 'Initializing WebGPU', {
    requestedFormat: config.renderFormat,
  });
  const initWatchdog = setInterval(() => {
    logWarn('pipeline', 'GPU initialization has not completed yet', {
      waitingMs: Math.round(performance.now() - runStart),
      requestedFormat: config.renderFormat,
      pageVisibility:
        typeof document !== 'undefined' ? document.visibilityState : 'unknown',
    });
  }, STALL_LOG_INTERVAL_MS);
  try {
    context = await initGpuContext(config);
  } catch (error) {
    logError('pipeline', 'GPU context initialization failed', error, {
      requestedFormat: config.renderFormat,
    });
    await flushAndClose(encoder);
    throw error;
  } finally {
    clearInterval(initWatchdog);
  }
  renderFormat = context.renderFormat;

  let renderer: HdrRenderer | null = null;
  let readbackManager: ReadbackManager | null = null;
  let preview: PreviewRenderer | null = null;
  let detachDeviceLost: (() => void) | null = null;
  let detachVisibility: (() => void) | null = null;

  function encoderQueueSize(): number | undefined {
    if (!encoder) return undefined;
    try {
      return encoder.encodeQueueSize();
    } catch {
      return undefined;
    }
  }

  /**
   * Encoder counters for stall diagnostics. `chunksEmitted` advancing between
   * watchdog ticks means the encoder is slow; a frozen value means it is stuck.
   */
  function encoderCounters(): Record<string, number> {
    if (!encoder) return {};
    try {
      const status = encoder.status();
      return {
        encoderFramesSubmitted: status.framesSubmitted,
        encoderFramesClosed: status.framesClosed,
        encoderChunksEmitted: status.chunksEmitted,
      };
    } catch {
      return {};
    }
  }

  /**
   * Flushes and closes the tap, but never waits forever. A flush that has not
   * settled within `ENCODER_FLUSH_TIMEOUT_MS` is abandoned and the encoder is
   * force-closed so the run can still return once frames are all delivered.
   */
  async function flushEncoderBounded(): Promise<void> {
    const tap = encoder;
    if (!tap) {
      await flushAndClose(null);
      return;
    }

    let timer: ReturnType<typeof setTimeout> | null = null;
    const timeout = new Promise<never>((_, reject) => {
      timer = setTimeout(() => {
        try {
          tap.close();
        } catch {
          // Ignore teardown races.
        }
        reject(
          new Error(
            `VideoEncoder flush did not settle within ${ENCODER_FLUSH_TIMEOUT_MS} ms; encoder force-closed`,
          ),
        );
      }, ENCODER_FLUSH_TIMEOUT_MS);
    });

    try {
      // `flushAndClose` swallows flush rejections, so race it against our own
      // hard timeout to distinguish a slow encoder from a wedged one.
      await Promise.race([flushAndClose(tap), timeout]);
    } finally {
      if (timer !== null) clearTimeout(timer);
    }
  }

  /**
   * Awaits `work`, logging the pipeline's state every few seconds until it
   * settles. Readback maps and encoder flushes can stall indefinitely after a
   * device loss; without this, a hung run looks identical to a slow one and
   * produces no diagnostics at all.
   */
  async function watchForStall<T>(work: Promise<T>, label: string): Promise<T> {
    const waitStartedAt = performance.now();
    const timer = setInterval(() => {
      logWarn('pipeline', `${label} has not settled yet`, {
        stage: currentStage,
        currentFrameIndex,
        waitingMs: Math.round(performance.now() - waitStartedAt),
        framesCompleted,
        framesTotal: config.frameCount,
        lastFrameIndex,
        pendingReadbacks: pending.length,
        activeReadbacks: readbackManager?.activeReadbacks,
        deviceLost: deviceLostReason ?? 'no',
        pageVisibility:
          typeof document !== 'undefined' ? document.visibilityState : 'unknown',
        encoderQueueSize: encoderQueueSize() ?? 'n/a',
        runElapsedMs: Math.round(performance.now() - runStart),
        ...encoderCounters(),
      });
    }, STALL_LOG_INTERVAL_MS);

    try {
      return await work;
    } finally {
      clearInterval(timer);
    }
  }

  function recordTimings(breakdown: FrameTimingBreakdown): void {
    lastFrameTimings = breakdown;
    timingTotals.renderMs += breakdown.renderMs;
    timingTotals.previewMs += breakdown.previewMs;
    timingTotals.readbackMs += breakdown.readbackMs;
    timingTotals.unpackMs += breakdown.unpackMs;
    timingTotals.encodeMs += breakdown.encodeMs;
    timingTotals.consumerMs += breakdown.consumerMs;
    timingTotals.totalMs += breakdown.totalMs;

    if (framesCompleted > 0) {
      const n = framesCompleted;
      averageFrameMs = timingTotals.totalMs / n;
      averageFrameTimings = {
        renderMs: timingTotals.renderMs / n,
        previewMs: timingTotals.previewMs / n,
        readbackMs: timingTotals.readbackMs / n,
        unpackMs: timingTotals.unpackMs / n,
        encodeMs: timingTotals.encodeMs / n,
        consumerMs: timingTotals.consumerMs / n,
        totalMs: averageFrameMs,
      };
    }
  }

  async function deliver(item: PendingReadback): Promise<void> {
    const manager = readbackManager;
    if (!manager) {
      throw new Error('pipeline: readback manager is not initialized');
    }

    logDebug('pipeline', `frame ${item.frameIndex} waiting for readback`, {
      stagingBufferIndex: item.job.stagingBufferIndex,
      renderMs: Number(item.renderMs.toFixed(2)),
      previewMs: Number(item.previewMs.toFixed(2)),
      activeReadbacks: manager.activeReadbacks,
      pendingReadbacks: pending.length,
    });

    const rawFrame = await watchForStall(
      item.job.finish(),
      `readback for frame ${item.frameIndex}`,
    );
    try {
      currentStage = `frame ${item.frameIndex} pixel-stats`;
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
        // Back-pressure: if the encoder is behind, wait for it to catch up before
        // queueing another frame. This bounds queue growth and keeps the final
        // flush short instead of draining the whole run's backlog.
        const queued = encoderQueueSize() ?? 0;
        if (queued >= ENCODER_QUEUE_DRAIN_THRESHOLD) {
          currentStage = `frame ${item.frameIndex} encoder-drain`;
          const drainStart = performance.now();
          logDebug('pipeline', 'Encoder back-pressure: waiting for queue to drain', {
            frameIndex: item.frameIndex,
            encodeQueueSize: queued,
            targetQueueSize: ENCODER_QUEUE_RESUME_THRESHOLD,
            framesCompleted,
          });
          await watchForStall(
            encoder.drain(ENCODER_QUEUE_RESUME_THRESHOLD),
            'VideoEncoder back-pressure drain',
          );
          encodeMs += performance.now() - drainStart;
        }

        currentStage = `frame ${item.frameIndex} encode`;
        const encodeStart = performance.now();
        const accepted = encodeFrame(encoder, frameForEncode, item.frameIndex === 0);
        encodeMs += performance.now() - encodeStart;

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

        const queueSize = encoderQueueSize();
        if (
          queueSize !== undefined &&
          queueSize >= ENCODER_QUEUE_WARN_THRESHOLD &&
          performance.now() - lastEncoderQueueWarningAt >=
            ENCODER_QUEUE_WARN_INTERVAL_MS
        ) {
          lastEncoderQueueWarningAt = performance.now();
          logWarn('pipeline', 'VideoEncoder queue is backing up', {
            frameIndex: item.frameIndex,
            encodeQueueSize: queueSize,
            encodeMs: Number(encodeMs.toFixed(2)),
            framesCompleted,
          });
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
        currentStage = `frame ${item.frameIndex} onFrame`;
        const pendingConsumer = callbacks?.onFrame?.(delivered);
        if (pendingConsumer) {
          await watchForStall(
            pendingConsumer,
            `onFrame consumer for frame ${item.frameIndex}`,
          );
        }
      } finally {
        consumerMs = performance.now() - consumerStart;
        // Release even if the consumer throws or cancellation arrives during the
        // callback; the next iteration may not run at all.
        delivered.release();
      }

      const totalMs = performance.now() - item.startedAt;
      lastFrameMs = totalMs;

      recordTimings({
        renderMs: item.renderMs,
        previewMs: item.previewMs,
        readbackMs,
        unpackMs,
        encodeMs,
        consumerMs,
        totalMs,
      });

      if (
        item.frameIndex === 0 ||
        item.frameIndex === config.frameCount - 1 ||
        item.frameIndex % 30 === 0
      ) {
        logDebug(
          'pipeline',
          `frame ${item.frameIndex} delivered (${framesCompleted}/${config.frameCount})`,
          {
            stagingBufferIndex: readback?.stagingBufferIndex,
            mapWaitMs: Number(readbackMs.toFixed(2)),
            unpackMs: Number(unpackMs.toFixed(2)),
            encodeMs: Number(encodeMs.toFixed(2)),
            consumerMs: Number(consumerMs.toFixed(2)),
            totalMs: Number(totalMs.toFixed(2)),
            measuredFps: Number(measuredFps.toFixed(2)),
            encoderQueueSize: encoderQueueSize() ?? 'n/a',
            pendingReadbacks: pending.length,
            deviceLost: deviceLostReason ?? 'no',
          },
        );
      }

      currentStage = `frame ${item.frameIndex} telemetry`;
      emitTelemetry(item.frameIndex === 0);
    } finally {
      // Safety net for unexpected failures before `delivered.release()` is
      // reached; releasing an already-released frame is idempotent.
      rawFrame.release();
    }
  }

  try {
    currentStage = 'device-lost-subscription';
    detachDeviceLost = context.onDeviceLost((reason) => {
      deviceLostReason = reason;
      addWarning(`GPU device lost: ${reason}`);
      // Unblock any in-flight readback whose `mapAsync` may never resolve after
      // device loss. The readback manager races every map against an internal
      // rejection promise; `signalDeviceLost` fires that rejection immediately
      // and also rejects pending staging-buffer waiters.
      readbackManager?.signalDeviceLost(reason);
    });

    if (typeof document !== 'undefined') {
      const onVisibilityChange = (): void => {
        logInfo('pipeline', `Page visibility changed to "${document.visibilityState}"`, {
          framesCompleted,
          lastFrameIndex,
          pendingReadbacks: pending.length,
          runElapsedMs: Math.round(performance.now() - runStart),
        });
      };
      document.addEventListener('visibilitychange', onVisibilityChange);
      detachVisibility = () =>
        document.removeEventListener('visibilitychange', onVisibilityChange);
    }

    // Keep non-null locals for the hot loop while still assigning to the
    // nullable handles that the `finally` block disposes on every exit path.
    currentStage = 'renderer-init';
    const hdrRenderer = createHdrRenderer(context, config);
    renderer = hdrRenderer;

    currentStage = 'readback-manager';
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

    logInfo('pipeline', 'Resources ready', {
      renderFormat,
      stagingBufferCount: manager.stagingBufferCount,
      maxFramesInFlight,
      reuseOutputBuffer: manager.reuseOutputBuffer,
      bufferMiB: Number((manager.bufferSize / (1024 * 1024)).toFixed(2)),
      retainedMiB: Number((retainedBytes / (1024 * 1024)).toFixed(2)),
    });

    if (options.previewCanvas) {
      currentStage = 'preview-init';
      try {
        preview = createPreviewRenderer(context, options.previewCanvas);
        logInfo('pipeline', 'Preview enabled', {
          format: preview.format,
          width: preview.width,
          height: preview.height,
        });
      } catch (error) {
        addWarning(`Preview disabled: ${describeError(error)}`);
        preview = null;
      }
    }

    const hdrTargetView = hdrRenderer.target.createView();
    const { device } = context;

    currentStage = 'onStart';
    callbacks?.onStart?.({
      frameCount: config.frameCount,
      width: config.width,
      height: config.height,
    });
    currentStage = 'initial-telemetry';
    emitTelemetry(true);
    logInfo('pipeline', 'Frame loop starting', {
      frameCount: config.frameCount,
      maxFramesInFlight,
      encoder: encoder ? 'on' : 'off',
    });

    for (let frameIndex = 0; frameIndex < config.frameCount; frameIndex++) {
      currentFrameIndex = frameIndex;
      currentStage = `frame ${frameIndex} begin`;
      if (signal?.aborted) {
        cancelled = true;
        logInfo('pipeline', 'Abort signal observed before frame scheduling', {
          nextFrameIndex: frameIndex,
          framesCompleted,
        });
        break;
      }

      const isMilestone =
        frameIndex === 0 ||
        frameIndex === config.frameCount - 1 ||
        frameIndex % 30 === 0;

      if (isMilestone) {
        logDebug('pipeline', `frame ${frameIndex} begin`, {
          pendingReadbacks: pending.length,
          framesCompleted,
          encoderQueueSize: encoderQueueSize() ?? 'n/a',
          deviceLost: deviceLostReason ?? 'no',
          runElapsedMs: Math.round(performance.now() - runStart),
        });
      }

      // Bounded lookahead: deliver the oldest frame before starting another.
      if (pending.length >= maxFramesInFlight) {
        const oldest = pending.shift();
        if (oldest) await deliver(oldest);
      }

      if (signal?.aborted) {
        cancelled = true;
        logInfo('pipeline', 'Abort signal observed while delivering lookahead', {
          nextFrameIndex: frameIndex,
          framesCompleted,
        });
        break;
      }

      const frameStart = performance.now();
      // `scene.animationTime` is exactly `frameIndex / config.fps`; the timeline
      // is a pure function of the index, so preview/readback timing cannot alter
      // the content of any frame that is produced.
      currentStage = `frame ${frameIndex} scene-evaluation`;
      const scene = getSceneAtFrame(frameIndex, config);

      // Scene and preview share one submission and one command encoder, so the
      // swap-chain texture is acquired exactly once per frame.
      currentStage = `frame ${frameIndex} render-recording`;
      const commandEncoder = device.createCommandEncoder({
        label: `frame-${frameIndex}-encoder`,
      });
      hdrRenderer.renderSceneToTexture(commandEncoder, scene);
      const beforePreview = performance.now();

      let previewMs = 0;
      if (preview) {
        currentStage = `frame ${frameIndex} preview-recording`;
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

      currentStage = `frame ${frameIndex} submit`;
      const submitStart = performance.now();
      device.queue.submit([commandEncoder.finish()]);
      const submitMs = performance.now() - submitStart;

      // The readback copy is submitted before the next frame can overwrite the
      // target. With depth >= 2, its map/unpack overlaps the next render.
      currentStage = `frame ${frameIndex} readback-submit`;
      const job = beginFrameReadback(context, manager, hdrRenderer, frameIndex, config);
      pending.push({
        frameIndex,
        job,
        startedAt: frameStart,
        renderMs: beforePreview - frameStart + submitMs,
        previewMs,
      });

      if (isMilestone) {
        logDebug('pipeline', `frame ${frameIndex} submitted`, {
          renderMs: Number((beforePreview - frameStart + submitMs).toFixed(2)),
          previewMs: Number(previewMs.toFixed(2)),
          stagingBufferIndex: job.stagingBufferIndex,
          pendingReadbacks: pending.length,
        });
      }
    }

    cancelled = cancelled || (signal?.aborted ?? false);

    if (cancelled) {
      logInfo('pipeline', 'Frame scheduling stopped by cancellation', {
        framesScheduled: framesCompleted + pending.length,
        framesCompleted,
        framesTotal: config.frameCount,
      });
    } else {
      logInfo('pipeline', 'All frames scheduled', {
        framesScheduled: config.frameCount,
        framesCompleted,
        pendingReadbacks: pending.length,
      });
    }

    // Drain in delivery order. Frames whose copies were already submitted are
    // still valid output even if cancellation arrived while they were in flight.
    currentStage = 'drain-pending-readbacks';
    logInfo('pipeline', 'Draining pending readbacks', { pendingReadbacks: pending.length });
    while (pending.length > 0) {
      const item = pending.shift();
      if (!item) break;
      await deliver(item);
    }

    if (encoder) {
      logInfo('pipeline', 'Flushing VideoEncoder', {
        framesCompleted,
        framesSubmitted: encoder.status().framesSubmitted,
        chunksEmitted: encoder.status().chunksEmitted,
        encodeQueueSize: encoderQueueSize() ?? 'n/a',
      });
    }
    currentStage = 'encoder-flush';
    try {
      await watchForStall(flushEncoderBounded(), 'VideoEncoder flush');
    } catch (error) {
      // Every frame was already delivered; losing trailing chunks is a warning,
      // not a reason to discard an otherwise complete run.
      addWarning(`VideoEncoder flush timed out: ${describeError(error)}`);
    }
    encoderFlushed = true;

    currentStage = 'finalize';
    const finalTelemetry = buildTelemetry();
    callbacks?.onTelemetry?.(finalTelemetry);

    const summary: PipelineSummary = {
      framesCompleted,
      cancelled,
      elapsedMs: performance.now() - runStart,
      telemetry: finalTelemetry,
    };

    logInfo('pipeline', 'Run finished', {
      framesCompleted,
      framesTotal: config.frameCount,
      cancelled,
      elapsedMs: Math.round(summary.elapsedMs),
      measuredFps: Number(measuredFps.toFixed(2)),
      averageFrameMs: Number(averageFrameMs.toFixed(2)),
      deviceLost: deviceLostReason ?? 'no',
      warnings: warnings.slice(),
      encoder: encoder ? toEncoderTelemetry(encoder) : 'off',
    });

    if (!cancelled && framesCompleted < config.frameCount) {
      logWarn(
        'pipeline',
        `Run finished with ${framesCompleted}/${config.frameCount} frames but no thrown error`,
        {
          lastFrameIndex,
          deviceLost: deviceLostReason ?? 'no',
          warnings: warnings.slice(),
        },
      );
    }

    return summary;
  } catch (error) {
    logError(
      'pipeline',
      `Run failed after ${framesCompleted}/${config.frameCount} frames`,
      error,
      {
        stage: currentStage,
        currentFrameIndex,
        lastFrameIndex,
        pendingReadbacks: pending.length,
        activeReadbacks: readbackManager?.activeReadbacks,
        deviceLost: deviceLostReason ?? 'no',
        encoderQueueSize: encoderQueueSize() ?? 'n/a',
        runElapsedMs: Math.round(performance.now() - runStart),
      },
    );
    for (const item of pending) {
      item.job.cancel();
    }
    pending.length = 0;
    throw error;
  } finally {
    currentStage = 'cleanup';
    if (!encoderFlushed) {
      logDebug('pipeline', 'Flushing VideoEncoder during cleanup');
      try {
        await watchForStall(flushEncoderBounded(), 'VideoEncoder flush during cleanup');
      } catch (error) {
        addWarning(`VideoEncoder flush failed: ${describeError(error)}`);
      }
    }
    detachVisibility?.();
    detachDeviceLost?.();
    preview?.dispose();
    readbackManager?.dispose();
    renderer?.dispose();
    destroyGpuContext(context);
    logDebug('pipeline', 'Cleanup complete', { framesCompleted, cancelled });
  }
}

interface MutableTimings {
  renderMs: number;
  previewMs: number;
  readbackMs: number;
  unpackMs: number;
  encodeMs: number;
  consumerMs: number;
  totalMs: number;
}
