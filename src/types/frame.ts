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

/** Full-float RGBA: 4 channels of 32-bit floats per pixel (CPU frame payload). */
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
  /**
   * Rotation in radians, clockwise-positive in the y-down design space
   * (matching the Canvas 2D convention).
   */
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
  /** Color space the float pixels are expressed in (linear working space). */
  readonly colorSpace: ColorSpaceSpec;
  /**
   * GPU render-target format the pixels were read back from (`rgba16float` by
   * default, or `rgba32float` when the device supports it). `FloatFrame.pixels`
   * is always decoded to 32-bit floats on the CPU.
   */
  readonly format: GPUTextureFormat;
  /** Wall-clock milliseconds spent producing this frame (filled by the pipeline). */
  readonly renderTimeMs: number;
}

/**
 * Per-readback timing and staging-ring diagnostics.
 *
 * `waitMs` measures the GPU copy plus `mapAsync` wait, while `unpackMs` is the
 * CPU-side work of stripping row padding and decoding half floats. `totalMs`
 * additionally includes any time spent waiting for a staging buffer.
 */
export interface ReadbackTiming {
  /**
   * Milliseconds from requesting `mapAsync` immediately after copy submission
   * until it resolved. This measures GPU copy/queue time, not the caller's
   * deliberate readback lookahead.
   */
  readonly waitMs: number;
  /** Milliseconds spent stripping padding / decoding to `Float32Array`. */
  readonly unpackMs: number;
  /** Milliseconds from readback start (including pool wait) to decoded pixels. */
  readonly totalMs: number;
  /** Index of the staging buffer that carried this frame's copy. */
  readonly stagingBufferIndex: number;
  /** Whether `pixels` points at the manager's shared reuse buffer. */
  readonly reusedOutputBuffer: boolean;
}

/**
 * Wall-clock breakdown for one produced frame, in milliseconds.
 *
 * Stages are measured independently so the cost of `rgba16float` versus
 * `rgba32float` (which primarily affects `readbackMs`/`unpackMs`) can be compared
 * with rendering and optional encoding work.
 */
export interface FrameTimingBreakdown {
  /** Scene evaluation, command recording, and queue submission. */
  readonly renderMs: number;
  /** Recording the preview pass; `0` when preview is disabled. */
  readonly previewMs: number;
  /** GPU copy plus `mapAsync` wait. */
  readonly readbackMs: number;
  /** CPU row-strip / half-float decode. */
  readonly unpackMs: number;
  /** Optional WebCodecs conversion plus `encoder.encode()` call. */
  readonly encodeMs: number;
  /** Time spent inside the consumer's `onFrame` callback. */
  readonly consumerMs: number;
  /** Wall-clock time from frame start through consumer completion. */
  readonly totalMs: number;
}

/** Lifecycle state of the optional WebCodecs tap. */
export type EncoderTapState = 'configured' | 'closed' | 'errored';

/**
 * Snapshot of the optional `VideoEncoder` tap. Safe to render in the UI: it
 * contains counters only, never encoded bytes (muxing is out of scope).
 */
export interface EncoderTelemetry {
  /** Configured / closed / errored state of the encoder. */
  readonly state: EncoderTapState;
  /** Codec string accepted by `VideoEncoder.isConfigSupported`. */
  readonly codec: string;
  /** Pixel format packed into each `VideoFrame` (`I420P10` or `RGBA`). */
  readonly frameFormat: string;
  /** Compact color description, e.g. `bt2020/pq/bt2020-ncl`. */
  readonly colorSpace: string;
  /** Frames for which `encoder.encode()` accepted a VideoFrame. */
  readonly framesSubmitted: number;
  /** VideoFrames for which `close()` was called after submission. */
  readonly framesClosed: number;
  /** `EncodedVideoChunk`s emitted by the encoder so far. */
  readonly chunksEmitted: number;
  /** Sum of `EncodedVideoChunk.byteLength` across emitted chunks. */
  readonly chunkBytes: number;
  /** Number of emitted chunks with `type === 'key'`. */
  readonly keyFrames: number;
  /** Current `VideoEncoder.encodeQueueSize` (encoder backpressure signal). */
  readonly encodeQueueSize: number;
  /** Most recent encoder error message, if any. */
  readonly lastError: string | null;
}

/**
 * A single decoded HDR frame delivered to the CPU or to a downstream consumer.
 *
 * **Memory note:** a 1280x720 RGBA `Float32Array` frame is ~14.75 MB
 * (`1280 * 720 * 4 channels * 4 bytes`). Retaining all 300 frames would cost
 * ~4.4 GB, so `FloatFrame.pixels` is intended to be transferred/consumed and
 * released per-frame rather than accumulated. Reading back from an
 * `rgba16float` GPU target halves GPU-side bandwidth, but the CPU-side payload
 * is still expanded to `Float32Array` for uniform downstream handling.
 */
export interface FloatFrame {
  /** Frame identity, timing, and format metadata. */
  readonly meta: FrameMeta;
  /**
   * Tightly packed, linear Rec. 2020 RGBA pixels as 32-bit floats. Values may
   * exceed `1.0` to represent HDR highlights and are never clamped here.
   *
   * When readback uses `reuseOutputBuffer: true`, this array is shared across
   * frames and is overwritten by the next readback. Consumers must finish with
   * it (and call `release()`) before awaiting the next frame.
   */
  readonly pixels: Float32Array;
  /**
   * Tightly packed row stride in bytes, equal to
   * `width * BYTES_PER_PIXEL_F32` once GPU row padding has been stripped.
   */
  readonly bytesPerRow: number;
  /**
   * Readback-stage timing and staging-ring diagnostics for this frame. Present
   * for frames produced by `src/gpu/readback.ts`.
   */
  readonly readback?: ReadbackTiming;
  /**
   * Marks the frame as consumed. Idempotent. GPU staging memory is already
   * unmapped and recycled before the frame is returned, so this only signals
   * that the caller is done with the CPU `pixels` payload and may drop it.
   * When the output buffer is reused, this also releases the shared buffer for
   * the next readback.
   */
  release(): void;
}

/**
 * Sampled pixel statistics for one frame.
 *
 * A full min/max scan of a 1280x720 RGBA frame touches ~3.7M floats, so the
 * pipeline samples RGB values on a fixed pixel stride for cheap, deterministic
 * telemetry. `sample` reports one complete RGBA pixel (the frame center) so the
 * UI can show concrete linear values alongside the range.
 */
export interface FrameValueStats {
  /** Smallest sampled linear RGB channel value. */
  readonly min: number;
  /** Largest sampled linear RGB channel value (peaks above 1.0 indicate HDR). */
  readonly max: number;
  /** Center pixel RGBA in linear working-space units. */
  readonly sample: readonly [r: number, g: number, b: number, a: number];
  /** Number of pixels actually inspected to produce `min`/`max`. */
  readonly sampledPixels: number;
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
  /** Bytes copied from GPU texture to CPU staging buffers (padded rows) since run start. */
  readonly readbackBytes: number;
  /** Estimated bytes currently retained in memory for buffered frames. */
  readonly retainedBytes: number;
  /** Non-fatal problems encountered (e.g. dropped preview frame). */
  readonly warnings: readonly string[];
  /** Zero-based index of the most recently completed frame, or -1 before any. */
  readonly lastFrameIndex?: number;
  /** Wall-clock milliseconds elapsed since the run began. */
  readonly elapsedMs?: number;
  /** Actual HDR target format in use (may fall back from the requested format). */
  readonly renderFormat?: GPUTextureFormat;
  /** Sampled value statistics for the most recently completed frame. */
  readonly lastFrameValues?: FrameValueStats;
  /** Stage timing breakdown for the most recently completed frame. */
  readonly lastFrameTimings?: FrameTimingBreakdown;
  /** Rolling per-stage averages across all completed frames. */
  readonly averageFrameTimings?: FrameTimingBreakdown;
  /** Live optional-encoder telemetry; absent when the tap is disabled. */
  readonly encoder?: EncoderTelemetry;
}
