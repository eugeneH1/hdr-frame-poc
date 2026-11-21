/**
 * Shared domain types for the HDR frame rendering POC.
 *
 * Responsibilities of this module: declare the *data contracts* that flow between
 * the timeline, GPU renderers, readback, encoder and UI layers. It contains no
 * runtime logic so that every other module can depend on it without pulling in
 * WebGPU or DOM specifics.
 *
 * Key design assumptions (also see the individual JSDoc blocks below):
 * - Working color space is **linear Rec. 2020** (Rec. 2020 primaries, D65 white).
 * - Frames are encoded for delivery using a **Rec. 2100** (BT.2100) transfer
 *   function, defaulting to **PQ (SMPTE ST 2084)** with 10-bit output.
 * - Frame data is stored as `rgba16float` (half float, RGBA order, no row padding
 *   once read back) so values above 1.0 can represent HDR highlights.
 * - All geometry is authored in a fixed 1280x720 design space; scaling to other
 *   resolutions happens at the renderer level, not in the scene definitions.
 */

/** Half-float RGBA: 4 channels of 16-bit floats per pixel. */
export const BYTES_PER_PIXEL_F16 = 8;

/** Full-float RGBA: 4 channels of 32-bit floats per pixel (CPU-side analysis only). */
export const BYTES_PER_PIXEL_F32 = 16;

/**
 * EOTF / transfer function used to encode linear light for delivery.
 * - `pq`: Perceptual Quantizer (SMPTE ST 2084), the Rec. 2100 HDR default.
 * - `hlg`: Hybrid Log-Gamma (ARIB STD-B67), alternative Rec. 2100 transfer.
 * - `linear`: No encoding; useful for intermediate/debug output.
 */
export type TransferFunction = 'pq' | 'hlg' | 'linear';

/**
 * Describes a color space precisely enough to convert to/from linear light and
 * to tag an encoded output stream.
 */
export interface ColorSpaceSpec {
  /** Human-readable identifier, e.g. `"Rec.2020-linear"` or `"Rec.2100-PQ"`. */
  readonly name: string;
  /** Chromaticity coordinates of the red primary (CIE 1931 xy). */
  readonly redPrimary: readonly [x: number, y: number];
  /** Chromaticity coordinates of the green primary (CIE 1931 xy). */
  readonly greenPrimary: readonly [x: number, y: number];
  /** Chromaticity coordinates of the blue primary (CIE 1931 xy). */
  readonly bluePrimary: readonly [x: number, y: number];
  /** Chromaticity coordinates of the white point (D65 for Rec. 2020/2100). */
  readonly whitePoint: readonly [x: number, y: number];
  /** Transfer function applied when encoding linear values for this space. */
  readonly transfer: TransferFunction;
  /** Reference peak luminance in nits; `10000` for typical PQ, `1000` for grading. */
  readonly peakLuminanceNits: number;
}

/**
 * Static configuration for one render session. Frozen at pipeline start so that
 * every frame in the run is reproducible from `(frameIndex, config)` alone.
 */
export interface RenderConfig {
  /** Output width in pixels (POC default: 1280). */
  readonly width: number;
  /** Output height in pixels (POC default: 720). */
  readonly height: number;
  /** Frames per second (POC default: 30). */
  readonly fps: number;
  /** Total frame count for the run (POC default: 300 = 10s * 30fps). */
  readonly frameCount: number;
  /** Linear working color space used by shaders and the timeline. */
  readonly workingColorSpace: ColorSpaceSpec;
  /** Target delivery color space; conversion happens at preview/encode time. */
  readonly outputColorSpace: ColorSpaceSpec;
  /** GPU texture format for the offscreen render target (default `rgba16float`). */
  readonly renderFormat: GPUTextureFormat;
  /** Optional fixed seed so procedural/animated content is reproducible. */
  readonly seed: number;
}

/**
 * A 2D affine transform stored as a 3x3 matrix in **column-major** order, matching
 * WGSL's `mat3x3<f32>` memory layout: `[m00, m01, m02, m10, m11, m12, m20, m21, m22]`.
 *
 * Naming is `mCR` = column `C`, row `R`, so each consecutive triple is one column:
 *
 * ```
 * | m00  m10  m20 |   | a  c  tx |
 * | m01  m11  m21 | = | b  d  ty |
 * | m02  m12  m22 |   | 0  0   1 |
 * ```
 *
 * `m20`/`m21` therefore hold the translation and `m22` is the homogeneous scale.
 * Points transform as `x' = m00*x + m10*y + m20`, `y' = m01*x + m11*y + m21`.
 * `math/mat3.ts` is the authority for building and combining these matrices.
 */
export type AffineMatrix = readonly [
  m00: number, m01: number, m02: number,
  m10: number, m11: number, m12: number,
  m20: number, m21: number, m22: number,
];

/**
 * Declarative transform components for a shape. The timeline animates these, and
 * `composeShapeMatrix` in `math/mat3.ts` folds them into an `AffineMatrix`.
 *
 * Units: `translation` and `size` are in design pixels (1280x720 space),
 * `rotation` is in radians, `scale` is unitless, `skew` is in radians.
 */
export interface Transform2D {
  /** Translation in design pixels relative to the design-space origin. */
  readonly translation: TransformVec2;
  /** Rotation in radians (counter-clockwise positive). */
  readonly rotation: number;
  /** Non-uniform scale; `[1, 1]` is identity scale. */
  readonly scale: TransformVec2;
  /** Skew in radians applied on the X axis (shear factor = tan(skewX)). */
  readonly skewX: number;
  /** Pivot in local shape space (relative to the shape center) used for rotation/scale. */
  readonly pivot: TransformVec2;
}

/** Minimal 2-component vector; kept local to avoid coupling types to a math library. */
export interface TransformVec2 {
  readonly x: number;
  readonly y: number;
}

/**
 * Linear-light color in the working color space. Values may exceed 1.0 to express
 * HDR highlights (e.g. `4.0` ≈ 4x diffuse white).
 */
export interface LinearColor {
  /** Linear Rec. 2020 red channel (can be > 1 for HDR). */
  readonly r: number;
  /** Linear Rec. 2020 green channel (can be > 1 for HDR). */
  readonly g: number;
  /** Linear Rec. 2020 blue channel (can be > 1 for HDR). */
  readonly b: number;
  /** Alpha in `[0, 1]`; blended independently of color-space conversion. */
  readonly a: number;
}

/** Stable identifier for the three animated shapes in the scene. */
export type ShapeId = 'shape-a' | 'shape-b' | 'shape-c';

/**
 * Per-frame state of a single animated shape. Base dimensions are 100x100 design
 * pixels; `size` allows per-frame scaling driven entirely by the timeline.
 */
export interface ShapeState {
  /** Stable identity so GPU resources / diagnostics can be correlated per shape. */
  readonly id: ShapeId;
  /** Local size in design pixels before transform; base is `{100, 100}`. */
  readonly size: TransformVec2;
  /** Animated transform applied to the shape. */
  readonly transform: Transform2D;
  /** Linear HDR color of the shape fill. */
  readonly color: LinearColor;
  /** Global opacity multiplier in `[0, 1]`, applied after color conversion. */
  readonly opacity: number;
}

/**
 * Everything needed to render a single frame without consulting prior state.
 * `animationTime` is derived from `frameIndex / fps` and drives the timeline.
 */
export interface SceneState {
  /** Zero-based frame index in `[0, frameCount)`. */
  readonly frameIndex: number;
  /** Absolute animation time in seconds, i.e. `frameIndex / fps`. */
  readonly animationTime: number;
  /** Full-frame HDR background gradient (linear light); may exceed 1.0. */
  readonly background: LinearColor;
  /** Secondary background color for the gradient; ignored if equal to `background`. */
  readonly backgroundTo: LinearColor;
  /** The three animated shapes, in deterministic render order (back to front). */
  readonly shapes: readonly [ShapeState, ShapeState, ShapeState];
}

/** Bookkeeping metadata attached to every produced frame. */
export interface FrameMeta {
  /** Zero-based frame index. */
  readonly frameIndex: number;
  /** Presentation timestamp in microseconds (`frameIndex / fps * 1e6`). */
  readonly timestampUs: number;
  /** Frame duration in microseconds (`1e6 / fps`). */
  readonly durationUs: number;
  /** Frame width in pixels. */
  readonly width: number;
  /** Frame height in pixels. */
  readonly height: number;
  /** Color space the encoded pixels are tagged as (typically Rec. 2100 PQ). */
  readonly colorSpace: ColorSpaceSpec;
  /** Pixel format of `FloatFrame.pixels` (`rgba16float` in normal operation). */
  readonly format: GPUTextureFormat;
  /** Wall-clock milliseconds spent producing this frame (filled by the pipeline). */
  readonly renderTimeMs: number;
}

/**
 * A single decoded HDR frame delivered to the CPU or to a downstream consumer.
 *
 * **Memory note:** each 1280x720 `rgba16float` frame is ~7.03 MB. Retaining all
 * 300 frames would cost ~2.1 GB, so `FloatFrame.pixels` is intended to be
 * transferred/consumed and released per-frame rather than accumulated.
 */
export interface FloatFrame {
  /** Frame identity, timing, and format metadata. */
  readonly meta: FrameMeta;
  /**
   * Pixel payload as a typed array view. `Uint16Array` holds half-float bit
   * patterns (rgba16float); consumers convert with `float16` helpers as needed.
   */
  readonly pixels: Uint16Array;
  /**
   * Tightly packed row stride in bytes, equal to `width * BYTES_PER_PIXEL_F16`
   * (i.e. padding from GPU readback has already been stripped).
   */
  readonly bytesPerRow: number;
  /**
   * Releases the backing allocation. After calling this, `pixels` must not be
   * read; the pipeline calls it once the frame has been consumed.
   */
  release(): void;
}

/** Counters and timing values sampled while a render run is in progress. */
export interface TelemetrySnapshot {
  /** Frames fully produced so far in the current run. */
  readonly framesCompleted: number;
  /** Total frames expected for the current run. */
  readonly framesTotal: number;
  /** Frames per second actually achieved (averaged over a sliding window). */
  readonly measuredFps: number;
  /** Most recent frame's production time in milliseconds. */
  readonly lastFrameMs: number;
  /** Rolling average frame production time in milliseconds. */
  readonly averageFrameMs: number;
  /** Bytes transferred from GPU to CPU since the run started. */
  readonly readbackBytes: number;
  /** Estimated bytes currently retained in memory for buffered frames. */
  readonly retainedBytes: number;
  /** Non-fatal problems encountered (e.g. dropped preview frame). */
  readonly warnings: readonly string[];
}
