/**
 * Frame pipeline orchestrator.
 *
 * Responsibility: coordinate deterministic generation of frames `0..frameCount-1`
 * for a single run — evaluate the timeline, render the HDR target, present the
 * preview, read back float pixels, and hand each `FloatFrame` to a consumer. It
 * owns run lifecycle and cancellation only; it contains no shader or math logic.
 *
 * Design assumptions:
 * - **Bounded memory:** each frame is rendered, presented, read back, delivered
 *   via `onFrame`, and released before the next frame starts. All 300 frames are
 *   never resident together (~14.75 MB each on the CPU, ~4.4 GB total).
 * - **Determinism:** frame N depends only on `getSceneAtFrame(N, config)`, never
 *   on wall-clock time, so a slow preview or a cancelled run cannot change the
 *   content of frames that *were* produced.
 * - **Back-pressure:** `onFrame` is awaited before the next frame begins, which
 *   keeps memory bounded even when the consumer is slow.
 * - **Cancellation:** cooperative via `AbortSignal`; checked between frames and
 *   after each readback. In-flight GPU work is allowed to settle, then all
 *   resources (preview, readback staging pool, renderer, device) are disposed in
 *   a `finally` block.
 * - **Telemetry:** cheap sampled pixel statistics are computed per frame, but the
 *   `onTelemetry` callback is throttled (plus forced samples at start/finish) so
 *   the UI is never flooded.
 */

import { destroyGpuContext, initGpuContext } from '../gpu/context.ts';
import { createHdrRenderer } from '../gpu/hdrRenderer.ts';
import type { HdrRenderer } from '../gpu/hdrRenderer.ts';
import { createPreviewRenderer } from '../gpu/previewRenderer.ts';
import type { PreviewRenderer } from '../gpu/previewRenderer.ts';
import { createReadbackManager, readFrameFloats } from '../gpu/readback.ts';
import type { ReadbackManager } from '../gpu/readback.ts';
import { getSceneAtFrame } from '../timeline/sceneAnimator.ts';
import type {
  FloatFrame,
  FrameValueStats,
  RenderConfig,
  TelemetrySnapshot,
} from '../types/frame.ts';

/** Minimum wall-clock gap between throttled telemetry callbacks. */
const TELEMETRY_INTERVAL_MS = 100;
/** Pixel stride for min/max telemetry sampling (keeps the scan cheap). */
const STATS_PIXEL_STRIDE = 16;
/** Number of completion timestamps used to estimate achieved FPS. */
const FPS_WINDOW_SIZE = 30;
/** Maximum warnings retained in a telemetry snapshot. */
const MAX_WARNINGS = 5;

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
  /** Total wall-clock milliseconds for the run. */
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
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
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
 * 2. `renderSceneToTexture(...)` → offscreen float HDR target.
 * 3. `renderPreview(...)` on the same command encoder when a canvas is present.
 * 4. Submit the combined scene + preview passes.
 * 5. `readFrameFloats(...)` → `FloatFrame` with row padding stripped.
 * 6. `onFrame(frame)`, then `frame.release()` (even if the callback throws).
 *
 * On completion or cancellation, resources are disposed and a `PipelineSummary`
 * is returned. A fatal error rejects after cleanup.
 *
 * @param options - Run configuration, preview target, callbacks, and abort signal.
 * @returns Summary of the run including final telemetry.
 */
export async function runFramePipeline(
  options: RunFramePipelineOptions,
): Promise<PipelineSummary> {
  const { config, callbacks, signal } = options;
  const runStart = performance.now();

  const warnings: string[] = [];
  const completionTimes: number[] = [];

  let framesCompleted = 0;
  let lastFrameIndex = -1;
  let lastFrameMs = 0;
  let averageFrameMs = 0;
  let measuredFps = 0;
  let readbackBytes = 0;
  let lastFrameValues: FrameValueStats | undefined;
  let renderFormat: GPUTextureFormat | undefined;
  let lastTelemetryAt = Number.NEGATIVE_INFINITY;
  let cancelled = false;

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
      // Frames are released immediately after `onFrame`, so nothing is retained
      // between iterations beyond the one frame currently being delivered.
      retainedBytes: 0,
      warnings: warnings.slice(),
      lastFrameIndex,
      elapsedMs: performance.now() - runStart,
      renderFormat,
      lastFrameValues,
    };
  }

  function emitTelemetry(force: boolean): void {
    const now = performance.now();
    if (!force && now - lastTelemetryAt < TELEMETRY_INTERVAL_MS) return;
    lastTelemetryAt = now;
    callbacks?.onTelemetry?.(buildTelemetry());
  }

  if (signal?.aborted) {
    return {
      framesCompleted: 0,
      cancelled: true,
      elapsedMs: 0,
      telemetry: buildTelemetry(),
    };
  }

  const context = await initGpuContext(config);
  renderFormat = context.renderFormat;

  let renderer: HdrRenderer | null = null;
  let readbackManager: ReadbackManager | null = null;
  let preview: PreviewRenderer | null = null;
  let detachDeviceLost: (() => void) | null = null;

  try {
    detachDeviceLost = context.onDeviceLost((reason) => {
      addWarning(`GPU device lost: ${reason}`);
    });

    // Keep non-null locals for the hot loop while still assigning to the
    // nullable handles that the `finally` block disposes on every exit path.
    const hdrRenderer = createHdrRenderer(context, config);
    renderer = hdrRenderer;

    const manager = createReadbackManager(context, config);
    readbackManager = manager;

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

      const frameStart = performance.now();
      // `scene.animationTime` is exactly `frameIndex / config.fps`; the timeline
      // is a pure function of the index, so preview/readback timing cannot alter
      // the content of any frame that is produced.
      const scene = getSceneAtFrame(frameIndex, config);

      // Scene and preview share one submission and one command encoder, so the
      // swap-chain texture is acquired exactly once per frame.
      const encoder = device.createCommandEncoder({ label: `frame-${frameIndex}-encoder` });
      hdrRenderer.renderSceneToTexture(encoder, scene);

      if (preview) {
        try {
          preview.renderPreview(encoder, hdrTargetView);
        } catch (error) {
          // Preview is best-effort: disable it and keep producing frames.
          addWarning(`Preview disabled: ${describeError(error)}`);
          preview.dispose();
          preview = null;
        }
      }

      device.queue.submit([encoder.finish()]);

      const rawFrame = await readFrameFloats(context, manager, hdrRenderer, frameIndex, config);

      const frameMs = performance.now() - frameStart;
      lastFrameMs = frameMs;
      lastFrameValues = summarizePixels(rawFrame.pixels);
      framesCompleted++;
      lastFrameIndex = frameIndex;
      readbackBytes = framesCompleted * manager.bufferSize;
      averageFrameMs = (performance.now() - runStart) / framesCompleted;

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

      const delivered: FloatFrame = {
        meta: { ...rawFrame.meta, renderTimeMs: frameMs },
        pixels: rawFrame.pixels,
        bytesPerRow: rawFrame.bytesPerRow,
        // Delegate so releasing the wrapper also releases the readback frame.
        release: () => rawFrame.release(),
      };

      try {
        await callbacks?.onFrame?.(delivered);
      } finally {
        // Release even if the consumer throws or a cancellation arrives during
        // the callback; the next iteration may not run at all.
        delivered.release();
      }

      emitTelemetry(frameIndex === 0);
    }

    cancelled = cancelled || (signal?.aborted ?? false);

    const finalTelemetry = buildTelemetry();
    callbacks?.onTelemetry?.(finalTelemetry);

    return {
      framesCompleted,
      cancelled,
      elapsedMs: performance.now() - runStart,
      telemetry: finalTelemetry,
    };
  } finally {
    detachDeviceLost?.();
    preview?.dispose();
    readbackManager?.dispose();
    renderer?.dispose();
    destroyGpuContext(context);
  }
}
