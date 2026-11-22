/**
 * 2D affine matrix helpers.
 *
 * Responsibility: provide pure, allocation-light functions that build and combine
 * 3x3 affine transforms for shape geometry. There is no GPU or DOM dependency
 * here; the output is plain numeric data that the renderer uploads later.
 *
 * ---------------------------------------------------------------------------
 * Matrix layout (single source of truth)
 * ---------------------------------------------------------------------------
 * `AffineMatrix` (see `types/frame.ts`) stores 9 numbers in **column-major**
 * order, matching WGSL `mat3x3<f32>` so uniform uploads need no transposition:
 *
 *   index:  0    1    2    3    4    5    6    7    8
 *   field: m00  m01  m02  m10  m11  m12  m20  m21  m22
 *          └── col 0 ──┘  └── col 1 ──┘  └── col 2 ──┘
 *
 * `mCR` means **column C, row R**. Written as a matrix:
 *
 *   | m00  m10  m20 |   | a  c  tx |
 *   | m01  m11  m21 | = | b  d  ty |
 *   | m02  m12  m22 |   | 0  0   1 |
 *
 * A homogeneous point `(x, y, 1)` therefore transforms as:
 *
 *   x' = m00*x + m10*y + m20
 *   y' = m01*x + m11*y + m21
 *
 * ---------------------------------------------------------------------------
 * Coordinate spaces and transform order
 * ---------------------------------------------------------------------------
 * - **Design space** is 1280x720 with its origin at the top-left, +X right and
 *   +Y down (the scene/timeline space). `composeShapeMatrix` outputs matrices
 *   in this space; `designToClip` (below) is the single, explicit bridge from
 *   design pixels to WebGPU NDC (`-1..1`, +Y up).
 * - **Shape-local space** has the shape's top-left at `(0, 0)` and extends to
 *   `(width, height)`, so the shape center sits at `(width/2, height/2)`.
 * - Because design space is **+Y down**, a positive rotation turns +X toward +Y,
 *   i.e. it reads as **clockwise on screen** (matching the Canvas 2D convention).
 *   Rotation and shear preserve handedness (`det = 1` here), so shapes are never
 *   mirrored. The `designToClip` Y flip is undone by the WebGPU viewport's
 *   NDC->framebuffer Y flip, keeping local +Y pointing down on screen.
 *
 * Scale and skew sanity:
 * - Scale is authored strictly positive (no zero crossing, no mirror flips), so
 *   `det = sx * sy > 0`; shear and rotation preserve area, so a 100x100 box
 *   never degenerates to zero area.
 * - Skew is a `tan` shear, so keep angles modest (roughly `|skew| <= 0.35 rad`,
 *   i.e. a shear factor under ~0.37) to avoid extreme slants.
 */

import type { AffineMatrix, Transform2D, TransformVec2 } from '../types/frame.ts';

/**
 * Returns the identity affine matrix.
 *
 * @returns Column-major identity: translation `(0, 0)`, unit scale, no rotation.
 */
export function identity(): AffineMatrix {
  return [
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
  ];
}

/**
 * Multiplies two affine matrices, producing `a * b`.
 *
 * Because points are column vectors, `a * b` means "apply `b` first, then `a`".
 * Each result column is the left matrix applied to the corresponding column of
 * the right matrix, which keeps the column-major layout consistent.
 *
 * @param a - Left-hand matrix (outer transform).
 * @param b - Right-hand matrix (inner transform).
 * @returns The composed column-major matrix.
 */
export function multiply(a: AffineMatrix, b: AffineMatrix): AffineMatrix {
  const [a00, a01, a02, a10, a11, a12, a20, a21, a22] = a;
  const [b00, b01, b02, b10, b11, b12, b20, b21, b22] = b;

  // Result column 0 = a * (b00, b01, b02).
  const c00 = a00 * b00 + a10 * b01 + a20 * b02;
  const c01 = a01 * b00 + a11 * b01 + a21 * b02;
  const c02 = a02 * b00 + a12 * b01 + a22 * b02;

  // Result column 1 = a * (b10, b11, b12).
  const c10 = a00 * b10 + a10 * b11 + a20 * b12;
  const c11 = a01 * b10 + a11 * b11 + a21 * b12;
  const c12 = a02 * b10 + a12 * b11 + a22 * b12;

  // Result column 2 = a * (b20, b21, b22).
  const c20 = a00 * b20 + a10 * b21 + a20 * b22;
  const c21 = a01 * b20 + a11 * b21 + a21 * b22;
  const c22 = a02 * b20 + a12 * b21 + a22 * b22;

  return [c00, c01, c02, c10, c11, c12, c20, c21, c22];
}

/**
 * Builds a pure translation matrix.
 *
 * @param tx - Translation along X in design pixels.
 * @param ty - Translation along Y in design pixels.
 * @returns Column-major translation matrix.
 */
export function translate(tx: number, ty: number): AffineMatrix {
  return [
    1, 0, 0,
    0, 1, 0,
    tx, ty, 1,
  ];
}

/**
 * Builds a rotation matrix.
 *
 * @param radians - Rotation angle in radians (clockwise-positive in the y-down
 *   design space, matching the Canvas 2D convention).
 * @returns Column-major rotation matrix about the origin.
 */
export function rotate(radians: number): AffineMatrix {
  const cos = Math.cos(radians);
  const sin = Math.sin(radians);
  return [
    cos, sin, 0,
    -sin, cos, 0,
    0, 0, 1,
  ];
}

/**
 * Builds a non-uniform scale matrix.
 *
 * @param sx - Scale factor along X.
 * @param sy - Scale factor along Y.
 * @returns Column-major scale matrix.
 */
export function scale(sx: number, sy: number): AffineMatrix {
  return [
    sx, 0, 0,
    0, sy, 0,
    0, 0, 1,
  ];
}

/**
 * Builds a skew (shear) matrix.
 *
 * Shear factors are tangents of the supplied angles:
 * `x' = x + tan(skewX) * y`, `y' = y + tan(skewY) * x`.
 *
 * @param skewXRadians - Horizontal shear angle in radians (`tan` applied to X).
 * @param skewYRadians - Optional vertical shear angle in radians; defaults to 0.
 * @returns Column-major skew matrix.
 */
export function skew(skewXRadians: number, skewYRadians: number = 0): AffineMatrix {
  const shearX = Math.tan(skewXRadians);
  const shearY = Math.tan(skewYRadians);
  return [
    1, shearY, 0,
    shearX, 1, 0,
    0, 0, 1,
  ];
}

/**
 * Folds a declarative `Transform2D` into a single affine matrix that maps
 * **shape-local pixels** to **design-space pixels**.
 *
 * Transform order, written as a product and applied right-to-left to a local
 * vertex `v`:
 *
 *   M = translate(translation) * rotate(rotation) * skew(skewX) *
 *       scale(scale) * translate(pivot - shapeCenter)
 *
 * Equivalently, `v` goes through:
 *   1. translate so the pivot lands on the local origin
 *      (`pivot - (width/2, height/2)`, i.e. the shape center by default),
 *   2. scale,
 *   3. skew,
 *   4. rotate about that pivot,
 *   5. translate to `transform.translation`, which is the pivot's position in
 *      design space.
 *
 * `transform.pivot` is expressed relative to the shape center, so the default
 * `{x: 0, y: 0}` pivots exactly around the center and reproduces the
 * `(-width/2, -height/2)` offset described above.
 *
 * @param transform - Declarative transform components from the timeline.
 * @param shapeSize - Untransformed local size of the shape in design pixels.
 * @returns Composed affine matrix mapping shape-local space to design space.
 */
export function composeShapeMatrix(
  transform: Transform2D,
  shapeSize: { readonly width: number; readonly height: number },
): AffineMatrix {
  const pivotOffset: TransformVec2 = {
    x: transform.pivot.x - shapeSize.width / 2,
    y: transform.pivot.y - shapeSize.height / 2,
  };

  // Local -> pivot-at-origin -> scale -> skew -> rotate -> design-space position.
  const toOrigin = translate(pivotOffset.x, pivotOffset.y);
  const scaled = multiply(scale(transform.scale.x, transform.scale.y), toOrigin);
  const skewed = multiply(skew(transform.skewX), scaled);
  const rotated = multiply(rotate(transform.rotation), skewed);
  return multiply(translate(transform.translation.x, transform.translation.y), rotated);
}

/**
 * Builds the design-space -> WebGPU clip-space matrix (NDC).
 *
 * Design space is 1280x720 pixels with origin top-left and +Y down; clip space
 * is `[-1, 1]` on both axes with +Y up (WebGPU convention). The mapping is:
 *
 *   clipX =  2 * x / width  - 1
 *   clipY = -2 * y / height + 1
 *
 * so design `(0, 0)` -> `(-1, +1)` (top-left) and `(width, height)` ->
 * `(+1, -1)` (bottom-right). Because the scale factors are independent, a
 * 100x100 shape at identity scale covers exactly 100x100 output pixels whenever
 * the output target matches `viewport`.
 *
 * Compose with `composeShapeMatrix` (`designToClip * shapeMatrix`) or use
 * `composeShapeClipMatrix` to go straight from shape-local space to NDC.
 *
 * @param viewport - Design/output size in pixels (1280x720 for the POC).
 * @returns Column-major design-pixel -> NDC matrix.
 */
export function designToClip(viewport: { readonly width: number; readonly height: number }): AffineMatrix {
  return [
    2 / viewport.width, 0, 0,
    0, -2 / viewport.height, 0,
    -1, 1, 1,
  ];
}

/**
 * Folds a `Transform2D` all the way from **shape-local pixels** to **NDC**.
 *
 * Equivalent to `multiply(designToClip(viewport), composeShapeMatrix(...))` and
 * is the matrix the shape vertex shader should consume directly. A 100x100
 * local box maps to 100x100 pixels on a matching `viewport`, independent of
 * the shapes' animated translation/rotation/scale/skew.
 *
 * @param transform - Declarative transform components from the timeline.
 * @param shapeSize - Untransformed local size of the shape in design pixels.
 * @param viewport - Design/output size in pixels.
 * @returns Composed affine matrix mapping shape-local space to WebGPU NDC.
 */
export function composeShapeClipMatrix(
  transform: Transform2D,
  shapeSize: { readonly width: number; readonly height: number },
  viewport: { readonly width: number; readonly height: number },
): AffineMatrix {
  return multiply(designToClip(viewport), composeShapeMatrix(transform, shapeSize));
}

/**
 * Converts a 2D affine matrix into a WGSL-friendly 4x4 matrix.
 *
 * The result is a 16-element **column-major** `Float32Array`, matching WGSL's
 * `mat4x4<f32>` uniform layout. `mat4x4<f32>` has no `vec3` alignment padding
 * traps, so this is the safest shape to put in a uniform buffer. The 2D affine
 * occupies the upper-left 2x2 with translation in the last column; Z passes
 * through unchanged (`z' = z`) and W is the affine homogeneous scale.
 *
 * @param m - Source affine matrix.
 * @returns A 16-element column-major `mat4x4<f32>` representation.
 */
export function toWgslMat4(m: AffineMatrix): Float32Array {
  const [m00, m01, m02, m10, m11, m12, m20, m21, m22] = m;
  const out = new Float32Array(16);

  // Column 0 and column 1 keep their x/y rows; the z row stays 0.
  out[0] = m00;
  out[1] = m01;
  out[2] = m02;
  out[4] = m10;
  out[5] = m11;
  out[6] = m12;

  // Column 2 is the Z basis: pass Z through unchanged.
  out[10] = 1;

  // Column 3 carries the 2D translation in x/y; z translation is 0 and w is
  // the affine homogeneous scale (1 for rigid/affine transforms).
  out[12] = m20;
  out[13] = m21;
  out[15] = m22;

  return out;
}

/**
 * Converts a 2D affine matrix into a packed 3x3 column-major float array.
 *
 * Useful for a WGSL `mat3x3<f32>` uniform; note that WGSL pads each `mat3x3`
 * *column* to 16 bytes, so callers placing this in a uniform buffer must add
 * the per-column padding themselves (hence "packed" in the name).
 *
 * @param m - Source affine matrix.
 * @returns A 9-element column-major float array.
 */
export function toWgslMat3(m: AffineMatrix): Float32Array {
  return new Float32Array(m);
}
