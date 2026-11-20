/**
 * Deterministic timeline evaluator.
 *
 * Responsibility: turn a `(frameIndex, RenderConfig)` pair into a complete
 * `SceneState` describing an HDR gradient background plus three animated 100x100
 * shapes. There is no stored animation state: frame N is a pure function of N,
 * so frames may be rendered out-of-order or in parallel without drift.
 *
 * Intended scene (to be fleshed out later):
 * - Background: smooth linear-light gradient (e.g. ~0.05 to ~4.0 nits-equivalent)
 *   to exercise the HDR range.
 * - `shape-a` (warm, high chroma), `shape-b` (cool), `shape-c` (near-white
 *   highlight) each with distinct color, opacity, and transform paths.
 *
 * Stub status: throws `not implemented`; inputs/outputs are finalized.
 */

import type { ColorSpaceSpec, RenderConfig, SceneState } from '../types/frame.ts';

/**
 * Evaluates the full scene for a given frame index.
 *
 * @param frameIndex - Zero-based frame index; expected in `[0, config.frameCount)`.
 * @param config - Immutable run configuration (dimensions, fps, color spaces, seed).
 * @returns `SceneState` containing `animationTime`, background colors, and exactly
 *   three `ShapeState` entries (shape ids `shape-a`, `shape-b`, `shape-c`).
 * @throws Implementations should range-check `frameIndex` against `config.frameCount`.
 */
export function getSceneAtFrame(_frameIndex: number, _config: RenderConfig): SceneState {
  throw new Error('sceneAnimator.getSceneAtFrame: not implemented');
}

/**
 * Provides the linear Rec. 2020 working color space spec.
 *
 * Kept here so both timeline and renderers agree on primaries/white point before
 * shader constants are generated.
 *
 * @returns `ColorSpaceSpec` describing linear Rec. 2020 with a 1000-nit reference.
 */
export function getWorkingColorSpace(): ColorSpaceSpec {
  throw new Error('sceneAnimator.getWorkingColorSpace: not implemented');
}
