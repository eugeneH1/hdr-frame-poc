/**
 * 2D affine matrix helpers.
 *
 * Responsibility: provide pure, allocation-light functions that build and combine
 * 3x3 affine transforms for shape geometry. The matrix layout is **column-major**
 * (`AffineMatrix` in `types/frame.ts`) so the same data can be uploaded to WGSL
 * `mat3x3<f32>` uniforms without transposition.
 *
 * Coordinate conventions:
 * - Design space origin is top-left, +X right, +Y down (matching canvas/NDC mapping
 *   performed later in the renderer).
 * - Positive rotation is counter-clockwise in math terms; the renderer may flip the
 *   sign when converting to clip space.
 *
 * All functions are stubs at this stage and throw `not implemented`. They are safe
 * to call only once real bodies land; signatures are considered stable.
 */

import type { AffineMatrix, Transform2D } from '../types/frame.ts';

/**
 * Returns the identity affine matrix.
 *
 * @returns Column-major identity: translation `(0, 0)`, unit scale, no rotation.
 */
export function identity(): AffineMatrix {
  throw new Error('mat3.identity: not implemented');
}

/**
 * Multiplies two affine matrices, producing `a * b` (apply `b` first, then `a`).
 *
 * @param a - Left-hand matrix (outer transform).
 * @param b - Right-hand matrix (inner transform).
 * @returns The composed column-major matrix.
 */
export function multiply(_a: AffineMatrix, _b: AffineMatrix): AffineMatrix {
  throw new Error('mat3.multiply: not implemented');
}

/**
 * Builds a pure translation matrix.
 *
 * @param tx - Translation along X in design pixels.
 * @param ty - Translation along Y in design pixels.
 * @returns Column-major translation matrix.
 */
export function translate(_tx: number, _ty: number): AffineMatrix {
  throw new Error('mat3.translate: not implemented');
}

/**
 * Builds a rotation matrix.
 *
 * @param radians - Rotation angle in radians (counter-clockwise positive).
 * @returns Column-major rotation matrix about the origin.
 */
export function rotate(_radians: number): AffineMatrix {
  throw new Error('mat3.rotate: not implemented');
}

/**
 * Builds a non-uniform scale matrix.
 *
 * @param sx - Scale factor along X.
 * @param sy - Scale factor along Y.
 * @returns Column-major scale matrix.
 */
export function scale(_sx: number, _sy: number): AffineMatrix {
  throw new Error('mat3.scale: not implemented');
}

/**
 * Builds a skew (shear) matrix.
 *
 * @param skewXRadians - Horizontal shear angle in radians (`tan` applied to X).
 * @param skewYRadians - Optional vertical shear angle in radians; defaults to 0.
 * @returns Column-major skew matrix.
 */
export function skew(_skewXRadians: number, _skewYRadians: number = 0): AffineMatrix {
  throw new Error('mat3.skew: not implemented');
}

/**
 * Folds a declarative `Transform2D` into a single affine matrix.
 *
 * Intended composition order (applied right-to-left to a local vertex):
 * `translate(pivot) * rotate * skew * scale * translate(-pivot) * translate(translation)`.
 * Keeping this order in one place guarantees every shape animates identically.
 *
 * @param transform - Declarative transform components from the timeline.
 * @returns Composed affine matrix ready for the shape vertex shader.
 */
export function composeShapeMatrix(_transform: Transform2D): AffineMatrix {
  throw new Error('mat3.composeShapeMatrix: not implemented');
}

/**
 * Converts a 2D affine matrix into a WGSL-friendly 4x4 matrix (column-major,
 * 16 floats) so it can share a uniform buffer layout with other 3D-ready data.
 *
 * The affine part occupies the upper-left 2x2 with translation in the last column.
 *
 * @param m - Source affine matrix.
 * @returns A 16-element column-major `mat4x4<f32>` representation.
 */
export function toWgslMat4(_m: AffineMatrix): Float32Array {
  throw new Error('mat3.toWgslMat4: not implemented');
}

/**
 * Converts a 2D affine matrix into a packed 3x3 column-major float array suitable
 * for a WGSL `mat3x3<f32>` uniform (9 floats, std140-friendly when padded by caller).
 *
 * @param m - Source affine matrix.
 * @returns A 9-element column-major float array.
 */
export function toWgslMat3(_m: AffineMatrix): Float32Array {
  throw new Error('mat3.toWgslMat3: not implemented');
}
