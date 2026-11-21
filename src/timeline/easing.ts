/**
 * Pure easing and interpolation helpers for the timeline.
 *
 * Responsibility: small, deterministic scalar functions used by `sceneAnimator`
 * to drive motion, fades, and looping. Every function here is **pure** and
 * independent of `frameIndex`/wall-clock, so frame N always looks the same.
 *
 * Conventions:
 * - Inputs that are documented as "normalized" are expected in `[0, 1]`.
 * - Functions do not allocate, mutate, or read external state.
 */

/**
 * Clamps a value into the inclusive range `[0, 1]`.
 *
 * `NaN` propagates unchanged (rather than being silently coerced to `0`), which
 * keeps upstream math bugs visible instead of hiding them in the animation.
 *
 * @param value - Arbitrary input value.
 * @returns `value` clamped to `[0, 1]`.
 */
export function clamp01(value: number): number {
  if (value < 0) return 0;
  if (value > 1) return 1;
  return value;
}

/**
 * Linearly interpolates between `a` and `b`.
 *
 * `t` is intentionally *not* clamped: callers pass pre-normalized values, and
 * unclamped extrapolation is occasionally useful (e.g. overshoot easing).
 *
 * @param a - Value at `t = 0`.
 * @param b - Value at `t = 1`.
 * @param t - Interpolation factor, normally in `[0, 1]`.
 * @returns `a + (b - a) * t`.
 */
export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/**
 * Hermite smoothstep between two edges, remapping `[edge0, edge1]` to `[0, 1]`
 * with zero first derivative at both ends.
 *
 * When `edge0 === edge1` the step is treated as instantaneous: values below the
 * edge map to `0`, values at or above it to `1` (avoids a divide-by-zero that
 * would otherwise leak `NaN` into the scene).
 *
 * @param edge0 - Lower edge (output 0).
 * @param edge1 - Upper edge (output 1).
 * @param x - Input value.
 * @returns Eased value in `[0, 1]`.
 */
export function smoothstep(edge0: number, edge1: number, x: number): number {
  if (edge0 === edge1) {
    return x < edge0 ? 0 : 1;
  }
  const t = clamp01((x - edge0) / (edge1 - edge0));
  return t * t * (3 - 2 * t);
}

/**
 * Smoothly oscillates `0 -> 1 -> 0` over one normalized phase.
 *
 * `phase` is normalized time: `0` maps to `0`, `0.5` to `1`, and `1` back to
 * `0`, with the pattern repeating for every unit (integer or negative phases
 * wrap, so a looping timeline can feed raw `time / duration` values). Applying
 * `smoothstep` to the triangle wave gives zero velocity at both endpoints,
 * which reads as a natural back-and-forth motion.
 *
 * @param phase - Normalized phase; one unit is a full there-and-back cycle.
 * @returns Value in `[0, 1]`.
 */
export function pingPong(phase: number): number {
  const cycle = phase - Math.floor(phase);
  const triangle = 1 - Math.abs(2 * cycle - 1);
  return smoothstep(0, 1, triangle);
}
