/**
 * Pure easing and interpolation helpers for the timeline.
 *
 * Responsibility: small, deterministic scalar functions used by `sceneAnimator`
 * to drive motion, fades, and looping. Every function here must be **pure** and
 * independent of `frameIndex`/wall-clock so that frame N always looks the same.
 *
 * These are stubs (`not implemented`); signatures are stable.
 */

/**
 * Clamps a value into the inclusive range `[0, 1]`.
 *
 * @param value - Arbitrary input value.
 * @returns `value` clamped to `[0, 1]` (`NaN` handling decided by implementation).
 */
export function clamp01(_value: number): number {
  throw new Error('easing.clamp01: not implemented');
}

/**
 * Linearly interpolates between `a` and `b`.
 *
 * @param a - Value at `t = 0`.
 * @param b - Value at `t = 1`.
 * @param t - Interpolation factor; expected pre-normalized to `[0, 1]`.
 * @returns `a + (b - a) * t`.
 */
export function lerp(_a: number, _b: number, _t: number): number {
  throw new Error('easing.lerp: not implemented');
}

/**
 * Hermite smoothstep between two edges, remapping `[edge0, edge1]` to `[0, 1]`
 * with zero first derivative at both ends.
 *
 * @param edge0 - Lower edge (output 0).
 * @param edge1 - Upper edge (output 1).
 * @param x - Input value.
 * @returns Eased value in `[0, 1]`.
 */
export function smoothstep(_edge0: number, _edge1: number, _x: number): number {
  throw new Error('easing.smoothstep: not implemented');
}

/**
 * Maps a monotonically increasing input into a `0 -> 1 -> 0` triangle wave over a
 * fixed period, useful for symmetric back-and-forth motion.
 *
 * @param t - Elapsed time in seconds (or any monotonic scalar).
 * @param period - Length of one full there-and-back cycle, in the same units as `t`.
 * @returns Value in `[0, 1]`.
 */
export function pingPong(_t: number, _period: number): number {
  throw new Error('easing.pingPong: not implemented');
}
