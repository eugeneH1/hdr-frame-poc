/**
 * Workbench DOM setup (plain TypeScript + HTML, no framework).
 *
 * Responsibility: build the POC's minimal UI — a preview `<canvas>`, Run / Cancel
 * controls, and a status/telemetry readout — and expose a small API the entry point
 * uses to drive a run. It deliberately contains no rendering logic: the workbench
 * only creates a fresh `RenderConfig`, starts/cancels `runFramePipeline`, and
 * renders the telemetry it receives.
 *
 * Design assumptions:
 * - The canvas is sized to the run's `width`/`height` design dimensions; CSS may
 *   scale it down while the backing store stays at native resolution.
 * - Cancel maps to an `AbortController`; the workbench never touches GPU objects.
 */

import type { RenderConfig, TelemetrySnapshot } from '../types/frame.ts';

/** Options accepted when mounting the workbench. */
export interface WorkbenchOptions {
  /** Element to render the workbench into (typically `#app`). */
  readonly root: HTMLElement;
  /**
   * Builds the pristine `RenderConfig` for a run. Called once per Run press so the
   * session can reset any derived state. Defaults to the POC's 1280x720/30fps/300.
   */
  readonly createConfig?: () => RenderConfig;
}

/** Handles onto the workbench's DOM and run controls. */
export interface Workbench {
  /** The preview canvas the pipeline renders into. */
  readonly canvas: HTMLCanvasElement;
  /** Element used to display status text and telemetry. */
  readonly statusEl: HTMLElement;
  /** Triggers a new render run; no-op if one is already active. */
  run(): Promise<void>;
  /** Cancels the active run, if any. */
  cancel(): void;
  /** Renders a telemetry snapshot into the status display. */
  updateTelemetry(snapshot: TelemetrySnapshot): void;
  /** Removes event listeners and clears the workbench DOM. */
  unmount(): void;
}

/**
 * Mounts the workbench UI into `options.root`.
 *
 * Intended steps: create the canvas, Run/Cancel buttons, and status `<pre>`; wire
 * button handlers to `runFramePipeline` with `callbacks.onTelemetry` feeding
 * `updateTelemetry`; manage the active `AbortController`.
 *
 * @param options - Mount target and optional config factory.
 * @returns A `Workbench` handle for programmatic control.
 */
export function mountWorkbench(_options: WorkbenchOptions): Workbench {
  throw new Error('ui.workbench.mountWorkbench: not implemented');
}
