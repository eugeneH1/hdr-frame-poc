/**
 * Frame pipeline orchestrator.
 *
 * Responsibility: coordinate deterministic generation of frames `0..frameCount-1`
 * for a single run — evaluate the timeline, render the HDR target, present the
 * preview, read back float pixels, and hand each `FloatFrame` to a consumer (and
 * optionally the WebCodecs tap). It owns run lifecycle and cancellation only; it
 * contains no shader or math logic.
 *
 * Design assumptions:
 * - **Bounded memory:** frames are processed one at a time and released after
 *   `onFrame`, so all 300 frames are never resident together (~7 MB each). Back-
 *   pressure is applied by awaiting `onFrame` before starting the next frame.
 * - Determinism: frame N depends only on `getSceneAtFrame(N, config)`, never on
 *   wall-clock time, so preview cadence or dropped frames don't alter output.
 * - Cancellation is cooperative via an `AbortSignal`; in-flight GPU work should be
 *   allowed to settle, then resources disposed.
 */

import type { FloatFrame, RenderConfig, TelemetrySnapshot } from '../types/frame.ts';

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
 * are optional; the pipeline must behave identically when any are omitted.
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

/**
 * Runs the full deterministic frame pipeline.
 *
 * Intended orchestration per frame `N`:
 * 1. `getSceneAtFrame(N, config)` → linear Rec. 2020 scene.
 * 2. `renderHdrFrame(...)` → `rgba16float` offscreen target.
 * 3. `renderPreview(...)` when a preview canvas is provided (skippable on lag).
 * 4. `readFrameFloats(...)` → `FloatFrame` (256-byte row-aligned copy, padding stripped).
 * 5. `onFrame(frame)` (and optional `encodeFrame`), then `frame.release()`.
 *
 * On completion it calls `onStart`/`onTelemetry` appropriately, disposes the GPU
 * context and readback manager, and returns a `PipelineSummary`.
 *
 * @param options - Run configuration, preview target, callbacks, and abort signal.
 * @returns Summary of the run including final telemetry.
 */
export async function runFramePipeline(
  _options: RunFramePipelineOptions,
): Promise<PipelineSummary> {
  throw new Error('pipeline.framePipeline.runFramePipeline: not implemented');
}
