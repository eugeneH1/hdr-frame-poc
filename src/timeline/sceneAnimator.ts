/**
 * Deterministic timeline evaluator.
 *
 * Responsibility: turn a `(frameIndex, RenderConfig)` pair into a complete
 * `SceneState` describing an HDR gradient background plus three animated 100x100
 * shapes. There is no stored animation state: frame N is a pure function of N,
 * so frames may be rendered out of order or in parallel without drift.
 *
 * Design notes:
 * - All animation frequencies are integer multiples of the run duration, so a
 *   full 10 s / 300 frame loop is seamless (frame 300 would equal frame 0).
 * - Colors are authored directly in **linear Rec. 2020**; values above `1.0` are
 *   deliberate HDR highlights, not clamped. The renderer keeps them intact in an
 *   `rgba16float` target and only tone-maps/encodes at preview/encode time.
 * - All positions, translations, and sizes are in **design-space pixels**
 *   (`0..1280` x, `0..720` y, origin top-left, +Y down), **not NDC**. Convert at
 *   the GPU boundary with `designToClip` / `composeShapeClipMatrix` from
 *   `math/mat3.ts`; `composeShapeMatrix` alone stays in pixel space.
 * - The background keeps a dark, readable base tone and adds a smooth HDR pulse
 *   that periodically pushes a channel above `1.0`.
 */

import type {
  ColorSpaceSpec,
  LinearColor,
  RenderConfig,
  SceneState,
  ShapeState,
  Transform2D,
} from '../types/frame.ts';
import { clamp01, lerp, pingPong } from './easing.ts';

const TAU = Math.PI * 2;

/**
 * Skew is a `tan` shear, so keep the peak angle modest to avoid extreme slants
 * on the 100x100 boxes. `tan(0.32) ~= 0.33` of horizontal shear at worst.
 */
const MAX_SKEW_RAD = 0.32;

/** Scale stays strictly positive, so shapes never collapse or mirror-flip. */
const MIN_SCALE = 0.7;
const MAX_SCALE = 1.5;

/** One full oscillation of `cycles` per normalized run phase, with a phase offset. */
function sinWave(phase: number, cycles: number, offset: number): number {
  return Math.sin(TAU * (phase * cycles + offset));
}

function cosWave(phase: number, cycles: number, offset: number): number {
  return Math.cos(TAU * (phase * cycles + offset));
}

/** Smooth `0 -> 1 -> 0` pulse over `cycles` oscillations of the run. */
function pulse(phase: number, cycles: number, offset: number): number {
  return pingPong(phase * cycles + offset);
}

/** Eased value between `min` and `max`, driven by a smooth pulse. */
function fade(
  min: number,
  max: number,
  phase: number,
  cycles: number,
  offset: number,
): number {
  return lerp(min, max, pulse(phase, cycles, offset));
}

/** Builds a `Transform2D` pivoting around the shape center. */
function makeTransform(
  x: number,
  y: number,
  rotation: number,
  scaleX: number,
  scaleY: number,
  skewX: number,
): Transform2D {
  return {
    translation: { x, y },
    rotation,
    scale: { x: scaleX, y: scaleY },
    skewX,
    pivot: { x: 0, y: 0 },
  };
}

/** Distinct Rec. 2020 wide-gamut HDR colors (each has a channel well above 1.0). */
const SHAPE_A_COLOR: LinearColor = { r: 5.6, g: 0.62, b: 0.16, a: 1 };
const SHAPE_B_COLOR: LinearColor = { r: 0.14, g: 4.4, b: 3.1, a: 1 };
const SHAPE_C_COLOR: LinearColor = { r: 3.6, g: 3.1, b: 2.7, a: 1 };

/**
 * Evaluates the full scene for a given frame index.
 *
 * The result is derived solely from `frameIndex` and `config`; there is no
 * dependence on wall-clock time, prior frames, or mutable module state.
 *
 * @param frameIndex - Zero-based frame index; must be an integer in
 *   `[0, config.frameCount)`.
 * @param config - Immutable run configuration (dimensions, fps, color spaces, seed).
 * @returns `SceneState` containing `animationTime`, background colors, and exactly
 *   three `ShapeState` entries (shape ids `shape-a`, `shape-b`, `shape-c`).
 * @throws {RangeError} If `frameIndex` is not an integer within `[0, frameCount)`.
 */
export function getSceneAtFrame(frameIndex: number, config: RenderConfig): SceneState {
  if (!Number.isInteger(frameIndex) || frameIndex < 0 || frameIndex >= config.frameCount) {
    throw new RangeError(
      `sceneAnimator.getSceneAtFrame: frameIndex ${frameIndex} is outside [0, ${config.frameCount})`,
    );
  }

  const animationTime = frameIndex / config.fps;
  // Normalized run phase in [0, 1); integer cycle counts keep the loop seamless.
  const u = frameIndex / config.frameCount;

  // --- Background: dark readable base plus a smooth HDR highlight pulse -----
  const bgPulse = pulse(u, 2, 0.0);
  const bgAccent = pulse(u, 3, 0.28);

  const background: LinearColor = {
    r: 0.06 + 1.3 * bgPulse,
    g: 0.035 + 0.14 * bgPulse,
    b: 0.1 + 0.42 * bgPulse,
    a: 1,
  };
  const backgroundTo: LinearColor = {
    r: 0.02 + 0.1 * bgAccent,
    g: 0.07 + 1.15 * bgAccent,
    b: 0.28 + 0.7 * bgAccent,
    a: 1,
  };

  // --- Shape A: warm, high-chroma, fast orbit with gentle skew ---------------
  const shapeA: ShapeState = {
    id: 'shape-a',
    size: { x: 100, y: 100 },
    transform: makeTransform(
      420 + 200 * sinWave(u, 1, 0.0),
      330 + 140 * sinWave(u, 2, 0.12),
      0.15 + 0.5 * sinWave(u, 2, 0.05),
      fade(MIN_SCALE, 1.45, u, 1, 0.2),
      fade(MIN_SCALE + 0.05, 1.3, u, 2, 0.45),
      0.95 * MAX_SKEW_RAD * cosWave(u, 2, 0.3),
    ),
    color: SHAPE_A_COLOR,
    opacity: clamp01(fade(0.35, 0.95, u, 1, 0.1)),
  };

  // --- Shape B: cool cyan, counter-phase orbit and opposite skew -------------
  const shapeB: ShapeState = {
    id: 'shape-b',
    size: { x: 100, y: 100 },
    transform: makeTransform(
      880 + 180 * cosWave(u, 1, 0.3),
      320 + 160 * sinWave(u, 2, 0.55),
      -0.3 + 1.2 * cosWave(u, 2, 0.2),
      fade(0.85, MAX_SCALE, u, 2, 0.6),
      fade(0.75, 1.35, u, 1, 0.05),
      MAX_SKEW_RAD * sinWave(u, 1, 0.7),
    ),
    color: SHAPE_B_COLOR,
    opacity: clamp01(fade(0.3, 0.9, u, 2, 0.35)),
  };

  // --- Shape C: near-white highlight sweeping across the lower frame ---------
  const shapeC: ShapeState = {
    id: 'shape-c',
    size: { x: 100, y: 100 },
    transform: makeTransform(
      650 + 260 * cosWave(u, 2, 0.15),
      500 + 95 * sinWave(u, 1, 0.4),
      0.25 + 0.9 * sinWave(u, 3, 0.1),
      fade(MIN_SCALE, 1.25, u, 1, 0.5),
      fade(MIN_SCALE, 1.25, u, 2, 0.8),
      0.9 * MAX_SKEW_RAD * cosWave(u, 3, 0.25),
    ),
    color: SHAPE_C_COLOR,
    opacity: clamp01(fade(0.25, 0.85, u, 1, 0.65)),
  };

  const shapes: readonly [ShapeState, ShapeState, ShapeState] = [shapeA, shapeB, shapeC];

  return {
    frameIndex,
    animationTime,
    background,
    backgroundTo,
    shapes,
  };
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
  return {
    name: 'Rec.2020-linear',
    redPrimary: [0.708, 0.292],
    greenPrimary: [0.17, 0.797],
    bluePrimary: [0.131, 0.046],
    whitePoint: [0.3127, 0.329],
    transfer: 'linear',
    peakLuminanceNits: 1000,
  };
}
