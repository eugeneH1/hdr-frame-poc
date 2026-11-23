/**
 * Optional downstream WebCodecs tap.
 *
 * Responsibility: feed produced `FloatFrame`s into a `VideoEncoder` so the POC
 * can validate a future Rec. 2100 export path without adding a container muxer.
 * This module is intentionally **optional**: `runFramePipeline` works with the
 * tap absent, and `createVideoEncoderTap` returns `null` when the browser or
 * hardware exposes no supported 10-bit HDR profile.
 *
 * Design assumptions:
 * - **Color metadata:** the encoder config and each `VideoFrame` are tagged as
 *   Rec. 2020 primaries, `bt2020-ncl` matrix, and PQ (default) or HLG transfer.
 *   `VideoEncoderConfig.colorSpace` is a forward-looking extension (the current
 *   WebCodecs spec derives output color from the input `VideoFrame`), so the
 *   `VideoFrame` tag is the authoritative one. Unknown dictionary members are
 *   ignored by UAs that do not recognize them.
 * - **HDR packing:** the preferred input format is planar `I420P10` (limited
 *   range, 10-bit samples in the low bits of little-endian 16-bit words). The
 *   WebCodecs `VideoFrame(BufferSource, ...)` constructor only accepts 8-bit
 *   formats in current browsers (`I420`, `NV12`, `RGBA`, `BGRA`, ...), so the
 *   tap probes `I420P10` and falls back to opaque 8-bit `RGBA`. That is an
 *   expected capability fallback and is reported through `onInfo`, not
 *   `onWarning`: encoding still works and the output keeps Rec. 2100 metadata.
 *   No clamping happens before the transfer function, and the CPU pixels
 *   remain linear Rec. 2020.
 * - **Orientation:** pixels are top-down in both `FloatFrame` and the packed
 *   planes. The tap never flips Y and writes `rotation: 0`, preserving the
 *   pipeline's existing orientation conventions.
 * - **Lifetime:** each `VideoFrame` is closed immediately after `encode()`
 *   accepts it. `flush()` and `close()` are idempotent and always called at the
 *   end of a run or on cancellation.
 * - **No muxing:** emitted chunks are counted (count, bytes, keyframes) and
 *   forwarded to the caller's output callback. The tap never retains chunk data.
 */

import type {
  EncoderTapState,
  EncoderTelemetry,
  FloatFrame,
  FrameMeta,
} from '../types/frame.ts';
import { clamp01 } from '../timeline/easing.ts';

import { logDebug, logError, logInfo, logWarn } from '../diagnostics/logger.ts';

/** Input pixel formats the tap can pack from a float RGBA frame. */
export type EncoderFrameFormat = 'I420P10' | 'RGBA';

/** Status snapshot for the optional tap. */
export interface EncoderTapStatus extends EncoderTelemetry {
  /** Whether the underlying `VideoEncoder` has been closed. */
  readonly closed: boolean;
}

/** Handle to the optional encoder tap. */
export interface VideoEncoderTap {
  /** Codec string that passed `VideoEncoder.isConfigSupported`. */
  readonly codec: string;
  /** Pixel format packed into each `VideoFrame`. */
  readonly frameFormat: EncoderFrameFormat;
  /** Current tap status (counters, queue size, last error). */
  status(): EncoderTapStatus;
  /** Frames still held by the encoder's internal queue (backpressure signal). */
  encodeQueueSize(): number;
  /**
   * Waits for the encoder's internal queue to drop below `targetQueueSize`
   * using the WebCodecs `dequeue` event instead of an expensive full flush.
   * Keeps the hardware encoder pipeline warm while preventing runaway memory.
   */
  drain(targetQueueSize?: number): Promise<void>;
  /**
   * Converts and submits one frame. Returns `false` when the tap is closed,
   * errored, or the frame was rejected. The intermediate `VideoFrame` is always
   * closed before this method returns.
   */
  encode(frame: FloatFrame, keyFrame?: boolean): boolean;
  /** Flushes trailing frames and closes the encoder. Idempotent. */
  flush(): Promise<void>;
  /** Closes the encoder without flushing. Idempotent. */
  close(): void;
}

/** Receives encoded chunks and their codec metadata (muxing happens elsewhere). */
export type EncoderOutputCallback = (
  chunk: EncodedVideoChunk,
  metadata: EncodedVideoChunkMetadata | undefined,
) => void;

/** Options for creating the WebCodecs tap. */
export interface VideoEncoderTapOptions {
  /**
   * Codec strings to probe in priority order. The first supported profile wins.
   * Defaults to HEVC Main10, AV1 Main 10-bit, then VP9 Profile 2 10-bit.
   */
  readonly codecs?: readonly string[];
  /** Rec. 2100 transfer function; default `'pq'`. */
  readonly transfer?: 'pq' | 'hlg';
  /** Preferred input format; default `'I420P10'` with RGBA fallback. */
  readonly frameFormat?: EncoderFrameFormat;
  /** Target bitrate in bits/s. Defaults to a 720p-friendly HDR bitrate. */
  readonly bitrate?: number;
  /**
   * Forced-keyframe interval in frames. Defaults to one second at the frame's
   * rate. Set `0` to only mark the first frame as a keyframe.
   */
  readonly keyFrameIntervalFrames?: number;
  /** Non-fatal notices (unsupported profile, format fallback). */
  readonly onWarning?: (message: string) => void;
  /**
   * Expected capability notices that are not problems (e.g. the 8-bit RGBA
   * input fallback). Hosts should surface these as informational text rather
   * than warnings.
   */
  readonly onInfo?: (message: string) => void;
  /** Fatal encoder errors; the tap stops accepting frames afterwards. */
  readonly onError?: (error: Error) => void;
}

/**
 * Sample format / colour-space members are newer than the DOM typings bundled
 * with this project, so the HDR values are described locally and narrowed at the
 * WebCodecs boundary.
 */
interface HdrVideoColorSpaceInit {
  readonly primaries: 'bt2020';
  readonly transfer: 'pq' | 'hlg';
  readonly matrix: 'bt2020-ncl' | 'rgb';
  readonly fullRange: boolean;
}

interface HdrVideoEncoderConfig extends VideoEncoderConfig {
  /**
   * Not in every implementation's `VideoEncoderConfig` yet. UAs that recognize
   * it use it as an output tag; others ignore the unknown member.
   */
  readonly colorSpace?: HdrVideoColorSpaceInit;
}

interface HdrVideoFrameInit
  extends Omit<VideoFrameBufferInit, 'format' | 'colorSpace'> {
  readonly format: string;
  readonly colorSpace?: HdrVideoColorSpaceInit;
}

const DEFAULT_CODEC_CANDIDATES: readonly string[] = [
  // HEVC Main10, Level 3.1 (sufficient for 1280x720p30).
  'hvc1.2.4.L93.B0',
  // AV1 Main profile, Level 4.0, 10-bit.
  'av01.0.08M.10',
  // VP9 Profile 2, Level 3.1, 10-bit.
  'vp09.02.31.10',
];

/**
 * Reference white for encoding. A linear scene value of `1.0` represents 100
 * nits; PQ normalizes against its 10 000-nit peak, and HLG against a 1000-nit
 * reference display. Scene highlights above 1.0 therefore retain headroom in
 * both transfers instead of clipping at diffuse white.
 */
const REFERENCE_WHITE_NITS = 100;
const PQ_PEAK_NITS = 10000;
const HLG_PEAK_NITS = 1000;
const PQ_LUMA_SCALE = REFERENCE_WHITE_NITS / PQ_PEAK_NITS;
const HLG_LUMA_SCALE = REFERENCE_WHITE_NITS / HLG_PEAK_NITS;
/** Scene-linear values above this are clipped by the transfer lookup table. */
const MAX_ENCODED_LINEAR = 16;
/** Lookup table resolution; sampled in sqrt(linear) space for shadow accuracy. */
const TRANSFER_LUT_SIZE = 4096;
const LUT_INDEX_SCALE = (TRANSFER_LUT_SIZE - 1) / Math.sqrt(MAX_ENCODED_LINEAR);

/** BT.2020 non-constant-luminance luma coefficients. */
const Y_R = 0.2627;
const Y_G = 0.678;
const Y_B = 0.0593;
const CB_DIVISOR = 1.8814;
const CR_DIVISOR = 1.4746;

let pqLut: Float32Array | null = null;
let hlgLut: Float32Array | null = null;
let pqLutU8: Uint8Array | null = null;
let hlgLutU8: Uint8Array | null = null;

/** SMPTE ST 2084 (PQ) OETF; input is normalized absolute luminance / 10000. */
function encodePq(normalizedLuminance: number): number {
  const m1 = 2610 / 16384;
  const m2 = (2523 / 4096) * 128;
  const c1 = 3424 / 4096;
  const c2 = (2413 / 4096) * 32;
  const c3 = (2392 / 4096) * 32;
  const n = Math.pow(normalizedLuminance, m1);
  const l = (c1 + c2 * n) / (1 + c3 * n);
  return Math.pow(l, m2);
}

/** ARIB STD-B67 (HLG) OETF; input is scene-linear luminance / 1000. */
function encodeHlg(sceneLinear: number): number {
  const a = 0.17883277;
  const b = 0.28466892; // 1 - 4a
  const c = 0.55991073;
  if (sceneLinear <= 1 / 12) {
    return Math.sqrt(3 * sceneLinear);
  }
  return a * Math.log(12 * sceneLinear - b) + c;
}

function buildTransferLut(transfer: 'pq' | 'hlg'): Float32Array {
  const lut = new Float32Array(TRANSFER_LUT_SIZE);
  const scale = transfer === 'pq' ? PQ_LUMA_SCALE : HLG_LUMA_SCALE;
  for (let index = 0; index < TRANSFER_LUT_SIZE; index++) {
    const u = index / (TRANSFER_LUT_SIZE - 1);
    const linear = u * u * MAX_ENCODED_LINEAR;
    const normalized = clamp01(linear * scale);
    lut[index] = transfer === 'pq' ? encodePq(normalized) : encodeHlg(normalized);
  }
  return lut;
}

function getTransferLut(transfer: 'pq' | 'hlg'): Float32Array {
  if (transfer === 'pq') {
    pqLut ??= buildTransferLut('pq');
    return pqLut;
  }
  hlgLut ??= buildTransferLut('hlg');
  return hlgLut;
}

function getTransferLutU8(transfer: 'pq' | 'hlg'): Uint8Array {
  if (transfer === 'pq') {
    if (!pqLutU8) {
      const f32 = getTransferLut('pq');
      pqLutU8 = new Uint8Array(TRANSFER_LUT_SIZE);
      for (let i = 0; i < TRANSFER_LUT_SIZE; i++) {
        pqLutU8[i] = Math.round(clamp01(f32[i]) * 255);
      }
    }
    return pqLutU8;
  }
  if (!hlgLutU8) {
    const f32 = getTransferLut('hlg');
    hlgLutU8 = new Uint8Array(TRANSFER_LUT_SIZE);
    for (let i = 0; i < TRANSFER_LUT_SIZE; i++) {
      hlgLutU8[i] = Math.round(clamp01(f32[i]) * 255);
    }
  }
  return hlgLutU8;
}

/**
 * Samples the nonlinear transfer LUT. The index is `sqrt(linear)`, which spends
 * most of the table on the perceptually important shadow range.
 */
function encodeTransfer(linear: number, lut: Float32Array): number {
  if (!(linear > 0)) return 0;
  const scaled = Math.sqrt(linear) * LUT_INDEX_SCALE;
  if (scaled >= TRANSFER_LUT_SIZE - 1) return lut[TRANSFER_LUT_SIZE - 1];
  const index = scaled | 0;
  const fraction = scaled - index;
  const low = lut[index];
  const high = lut[index + 1];
  return low + (high - low) * fraction;
}

function quantizeLuma(value: number): number {
  const code = Math.round(64 + 876 * clamp01(value));
  return code < 64 ? 64 : code > 940 ? 940 : code;
}

function quantizeChroma(value: number): number {
  const centered = value < -0.5 ? -0.5 : value > 0.5 ? 0.5 : value;
  const code = Math.round(512 + 896 * centered);
  return code < 64 ? 64 : code > 960 ? 960 : code;
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function positiveInteger(value: number): number {
  return Number.isFinite(value) ? Math.max(1, Math.floor(value)) : 1;
}

/**
 * Packs linear float RGBA into limited-range I420P10 (Y, U, V planes; 10 bits in
 * the low bits of little-endian 16-bit samples). Alpha is discarded.
 */
function packI420P10(
  pixels: Float32Array,
  width: number,
  height: number,
  transfer: 'pq' | 'hlg',
  output: Uint16Array,
): void {
  const lut = getTransferLut(transfer);
  const chromaWidth = Math.ceil(width / 2);
  const yPlane = 0;
  const uPlane = width * height;
  const vPlane = uPlane + chromaWidth * Math.ceil(height / 2);

  for (let y = 0; y < height; y += 2) {
    for (let x = 0; x < width; x += 2) {
      let cbSum = 0;
      let crSum = 0;
      let samples = 0;

      for (let dy = 0; dy < 2 && y + dy < height; dy++) {
        const row = (y + dy) * width;
        for (let dx = 0; dx < 2 && x + dx < width; dx++) {
          const pixel = (row + x + dx) * 4;
          const r = encodeTransfer(pixels[pixel], lut);
          const g = encodeTransfer(pixels[pixel + 1], lut);
          const b = encodeTransfer(pixels[pixel + 2], lut);

          const luma = Y_R * r + Y_G * g + Y_B * b;
          output[yPlane + row + x + dx] = quantizeLuma(luma);
          cbSum += (b - luma) / CB_DIVISOR;
          crSum += (r - luma) / CR_DIVISOR;
          samples++;
        }
      }

      if (samples === 0) continue;
      const chromaIndex = (y >> 1) * chromaWidth + (x >> 1);
      output[uPlane + chromaIndex] = quantizeChroma(cbSum / samples);
      output[vPlane + chromaIndex] = quantizeChroma(crSum / samples);
    }
  }
}

/** Packs nonlinear Rec. 2020 RGB into opaque 8-bit RGBA (fallback path). */
function packRgba8(
  pixels: Float32Array,
  width: number,
  height: number,
  transfer: 'pq' | 'hlg',
  output: Uint8Array,
): void {
  const lutU8 = getTransferLutU8(transfer);
  const total = width * height * 4;
  const maxIndex = TRANSFER_LUT_SIZE - 1;

  for (let i = 0; i < total; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];

    const idxR = r > 0 ? (Math.sqrt(r) * LUT_INDEX_SCALE + 0.5) | 0 : 0;
    const idxG = g > 0 ? (Math.sqrt(g) * LUT_INDEX_SCALE + 0.5) | 0 : 0;
    const idxB = b > 0 ? (Math.sqrt(b) * LUT_INDEX_SCALE + 0.5) | 0 : 0;

    output[i] = lutU8[idxR < maxIndex ? idxR : maxIndex];
    output[i + 1] = lutU8[idxG < maxIndex ? idxG : maxIndex];
    output[i + 2] = lutU8[idxB < maxIndex ? idxB : maxIndex];
    output[i + 3] = 255;
  }
}

interface FramePacker {
  readonly format: EncoderFrameFormat;
  pack(frame: FloatFrame): Uint16Array | Uint8Array;
  init(frame: FloatFrame): HdrVideoFrameInit;
}

interface I420P10LayoutInfo {
  /** Tightly packed plane offsets/strides, in bytes. */
  readonly layout: PlaneLayout[];
  /** Total 16-bit samples across all three planes. */
  readonly totalSamples: number;
}

/**
 * Computes the tightly packed I420P10 layout. WebCodecs' default layout for this
 * format is the same tight packing, so the explicit layout is only needed for
 * UAs that reject the default (or align it differently).
 */
function getI420P10Layout(width: number, height: number): I420P10LayoutInfo {
  const chromaWidth = Math.ceil(width / 2);
  const chromaHeight = Math.ceil(height / 2);
  const lumaSamples = width * height;
  const chromaSamples = chromaWidth * chromaHeight;
  const totalSamples = lumaSamples + chromaSamples * 2;
  const lumaStride = width * 2;
  const chromaStride = chromaWidth * 2;

  return {
    layout: [
      { offset: 0, stride: lumaStride },
      { offset: lumaSamples * 2, stride: chromaStride },
      { offset: (lumaSamples + chromaSamples) * 2, stride: chromaStride },
    ],
    totalSamples,
  };
}

function makeI420P10Packer(
  width: number,
  height: number,
  transfer: 'pq' | 'hlg',
  explicitLayout: boolean,
): FramePacker {
  const lumaSamples = width * height;
  const { layout, totalSamples } = getI420P10Layout(width, height);
  let scratch: Uint16Array | null = null;

  return {
    format: 'I420P10',
    pack(frame) {
      if (frame.pixels.length < lumaSamples * 4) {
        throw new Error(
          `encoder.webCodecsTap: frame has ${frame.pixels.length} floats, expected at least ${lumaSamples * 4}`,
        );
      }
      const output = (scratch ??= new Uint16Array(totalSamples));
      packI420P10(frame.pixels, width, height, transfer, output);
      return output;
    },
    init(frame) {
      return {
        format: 'I420P10',
        codedWidth: width,
        codedHeight: height,
        timestamp: frame.meta.timestampUs,
        duration: frame.meta.durationUs,
        ...(explicitLayout ? { layout } : {}),
        colorSpace: {
          primaries: 'bt2020',
          transfer,
          matrix: 'bt2020-ncl',
          fullRange: false,
        },
      };
    },
  };
}

function makeRgbaPacker(
  width: number,
  height: number,
  transfer: 'pq' | 'hlg',
): FramePacker {
  let scratch: Uint8Array | null = null;

  return {
    format: 'RGBA',
    pack(frame) {
      if (frame.pixels.length < width * height * 4) {
        throw new Error(
          `encoder.webCodecsTap: frame has ${frame.pixels.length} floats, expected at least ${width * height * 4}`,
        );
      }
      const output = (scratch ??= new Uint8Array(width * height * 4));
      packRgba8(frame.pixels, width, height, transfer, output);
      return output;
    },
    init(frame) {
      return {
        format: 'RGBA',
        codedWidth: width,
        codedHeight: height,
        timestamp: frame.meta.timestampUs,
        duration: frame.meta.durationUs,
        colorSpace: {
          primaries: 'bt2020',
          transfer,
          matrix: 'rgb',
          fullRange: true,
        },
      };
    },
  };
}

function buildColorSpace(
  transfer: 'pq' | 'hlg',
  matrix: 'bt2020-ncl' | 'rgb',
  fullRange: boolean,
): HdrVideoColorSpaceInit {
  return { primaries: 'bt2020', transfer, matrix, fullRange };
}

function buildEncoderConfig(
  codec: string,
  meta: FrameMeta,
  transfer: 'pq' | 'hlg',
  bitrate: number,
  framerate: number,
): HdrVideoEncoderConfig {
  return {
    codec,
    width: meta.width,
    height: meta.height,
    bitrate,
    framerate,
    bitrateMode: 'variable',
    latencyMode: 'quality',
    hardwareAcceleration: 'no-preference',
    alpha: 'discard',
    colorSpace: buildColorSpace(transfer, 'bt2020-ncl', false),
  };
}

/** Result of probing whether a VideoFrame format/layout can be constructed. */
interface FrameProbeResult {
  readonly ok: boolean;
  /** True when only an explicit plane layout was accepted. */
  readonly explicitLayout: boolean;
  /** Underlying UA error(s), already deduplicated for display. */
  readonly error: string | null;
}

function tryI420P10Frame(
  width: number,
  height: number,
  transfer: 'pq' | 'hlg',
  explicitLayout: boolean,
): string | null {
  let probe: VideoFrame | null = null;
  try {
    const { layout, totalSamples } = getI420P10Layout(width, height);
    const init: HdrVideoFrameInit = {
      format: 'I420P10',
      codedWidth: width,
      codedHeight: height,
      timestamp: 0,
      duration: 1,
      ...(explicitLayout ? { layout } : {}),
      colorSpace: buildColorSpace(transfer, 'bt2020-ncl', false),
    };
    probe = new VideoFrame(
      new Uint16Array(totalSamples),
      init as unknown as VideoFrameBufferInit,
    );
    return null;
  } catch (error) {
    return describeError(error);
  } finally {
    try {
      probe?.close();
    } catch {
      // Ignore probe teardown races.
    }
  }
}

/**
 * Probes I420P10 construction at the real frame size, first with an explicit
 * tight layout and then with the UA's default layout. Probing the actual size
 * (rather than a 2x2 toy frame) ensures a successful probe means real frames
 * will construct too.
 */
function probeI420P10(
  width: number,
  height: number,
  transfer: 'pq' | 'hlg',
): FrameProbeResult {
  const errors = new Set<string>();

  for (const explicitLayout of [true, false]) {
    const error = tryI420P10Frame(width, height, transfer, explicitLayout);
    if (!error) {
      return { ok: true, explicitLayout, error: null };
    }
    errors.add(error);
  }

  return { ok: false, explicitLayout: false, error: [...errors].join('; ') };
}

function tryRgbaFrame(
  width: number,
  height: number,
  transfer: 'pq' | 'hlg',
): string | null {
  let probe: VideoFrame | null = null;
  try {
    const init: HdrVideoFrameInit = {
      format: 'RGBA',
      codedWidth: width,
      codedHeight: height,
      timestamp: 0,
      duration: 1,
      colorSpace: buildColorSpace(transfer, 'rgb', true),
    };
    probe = new VideoFrame(
      new Uint8Array(width * height * 4),
      init as unknown as VideoFrameBufferInit,
    );
    return null;
  } catch (error) {
    return describeError(error);
  } finally {
    try {
      probe?.close();
    } catch {
      // Ignore probe teardown races.
    }
  }
}

function probeRgba(
  width: number,
  height: number,
  transfer: 'pq' | 'hlg',
): FrameProbeResult {
  const error = tryRgbaFrame(width, height, transfer);
  return error
    ? { ok: false, explicitLayout: false, error }
    : { ok: true, explicitLayout: false, error: null };
}

/**
 * Creates a `VideoEncoder`-backed tap if WebCodecs and a supported 10-bit HDR
 * profile are available; otherwise returns `null` so the pipeline can continue
 * without it.
 *
 * Profiles are probed with `VideoEncoder.isConfigSupported` in priority order,
 * and each candidate is tagged Rec. 2020 / PQ-or-HLG / `bt2020-ncl`. The tap then
 * checks whether `I420P10` VideoFrames are constructible, falling back to 8-bit
 * `RGBA` when necessary.
 *
 * @param meta - Metadata for the first frame, used to derive dimensions/rate.
 * @param output - Receives encoded chunks and metadata for later muxing.
 * @param options - Codec/profile, transfer, bitrate, and diagnostic callbacks.
 * @returns A ready `VideoEncoderTap`, or `null` when unsupported.
 */
export async function createVideoEncoderTap(
  meta: FrameMeta,
  output: EncoderOutputCallback,
  options: VideoEncoderTapOptions = {},
): Promise<VideoEncoderTap | null> {
  const warn = (message: string): void => {
    logWarn('encoder', message);
    try {
      options.onWarning?.(message);
    } catch {
      // Diagnostics must never break the encode path.
    }
  };
  const info = (message: string): void => {
    logInfo('encoder', message);
    try {
      options.onInfo?.(message);
    } catch {
      // Diagnostics must never break the encode path.
    }
  };
  const failNotice = (error: unknown): void => {
    try {
      options.onError?.(error instanceof Error ? error : new Error(describeError(error)));
    } catch {
      // Ignore a throwing diagnostics handler.
    }
  };

  if (typeof VideoEncoder === 'undefined' || typeof VideoFrame === 'undefined') {
    warn('WebCodecs VideoEncoder/VideoFrame is not available in this browser.');
    return null;
  }
  if (typeof VideoEncoder.isConfigSupported !== 'function') {
    warn('VideoEncoder.isConfigSupported is unavailable; cannot probe HDR profiles.');
    return null;
  }

  const width = positiveInteger(meta.width);
  const height = positiveInteger(meta.height);
  const fps = meta.durationUs > 0 ? 1e6 / meta.durationUs : 30;
  const transfer = options.transfer ?? 'pq';
  const bitrate =
    options.bitrate ?? Math.max(6_000_000, Math.round(width * height * fps * 0.3));
  const candidates =
    options.codecs && options.codecs.length > 0
      ? options.codecs
      : DEFAULT_CODEC_CANDIDATES;

  let config: HdrVideoEncoderConfig | null = null;
  let codec: string | null = null;
  const rejected: string[] = [];

  for (const candidate of candidates) {
    const candidateConfig = buildEncoderConfig(candidate, meta, transfer, bitrate, fps);
    try {
      const support = await VideoEncoder.isConfigSupported(candidateConfig);
      if (support.supported) {
        config = candidateConfig;
        codec = candidate;
        break;
      }
      rejected.push(candidate);
    } catch {
      // Invalid or unrecognized codec strings are treated as unsupported.
      rejected.push(candidate);
    }
  }

  if (!config || !codec) {
    warn(
      `No supported 10-bit HDR encoder profile (tried ${rejected.join(', ')}).`,
    );
    return null;
  }
  const selectedCodec: string = codec;

  let frameFormat: EncoderFrameFormat = options.frameFormat ?? 'I420P10';
  let explicitLayout = false;

  if (frameFormat === 'I420P10') {
    const probe = probeI420P10(width, height, transfer);
    if (probe.ok) {
      explicitLayout = probe.explicitLayout;
    } else {
      const rgbaProbe = probeRgba(width, height, transfer);
      if (!rgbaProbe.ok) {
        warn(
          `No usable VideoFrame input format in this UA ` +
            `(I420P10: ${probe.error ?? 'unavailable'}; RGBA: ${rgbaProbe.error ?? 'unavailable'}).`,
        );
        return null;
      }
      info(
        `This browser cannot construct 10-bit I420P10 VideoFrames from CPU buffers ` +
          `(${probe.error ?? 'format unsupported'}); using 8-bit RGBA input instead. ` +
          'The encoder still targets Rec.2020 PQ/HLG, but source precision is 8-bit.',
      );
      frameFormat = 'RGBA';
    }
  } else {
    const rgbaProbe = probeRgba(width, height, transfer);
    if (!rgbaProbe.ok) {
      warn(
        `RGBA VideoFrames are unavailable in this UA (${rgbaProbe.error ?? 'unsupported'}).`,
      );
      return null;
    }
  }

  const packer =
    frameFormat === 'I420P10'
      ? makeI420P10Packer(width, height, transfer, explicitLayout)
      : makeRgbaPacker(width, height, transfer);
  const keyFrameInterval = Math.max(
    0,
    Math.floor(
      options.keyFrameIntervalFrames ?? Math.max(1, Math.round(fps)),
    ),
  );

  logInfo('encoder', 'VideoEncoder tap configured', {
    codec: selectedCodec,
    width,
    height,
    fps: Number(fps.toFixed(3)),
    bitrate,
    transfer,
    frameFormat,
    explicitLayout,
    keyFrameInterval,
    rejectedCandidates: rejected,
  });

  let encoder: VideoEncoder;
  let framesSubmitted = 0;
  let framesClosed = 0;
  let chunksEmitted = 0;
  let chunkBytes = 0;
  let keyFrames = 0;
  let lastError: string | null = null;
  let failed = false;
  let closed = false;
  let flushPromise: Promise<void> | null = null;

  const reportError = (error: unknown, fatal: boolean): void => {
    const message = describeError(error);
    lastError = message;
    if (fatal) failed = true;
    logError('encoder', fatal ? 'VideoEncoder failed' : 'VideoEncoder output callback failed', error, {
      framesSubmitted,
      framesClosed,
      chunksEmitted,
      keyFrames,
      lastError: message,
    });
    failNotice(error);
  };

  let createdEncoder: VideoEncoder | null = null;
  try {
    createdEncoder = new VideoEncoder({
      output(chunk, metadata) {
        chunksEmitted++;
        chunkBytes += chunk.byteLength;
        if (chunk.type === 'key') keyFrames++;
        try {
          output(chunk, metadata);
        } catch (error) {
          // A broken consumer must not tear down the encoder.
          reportError(error, false);
        }
      },
      error(error) {
        reportError(error, true);
      },
    });
    createdEncoder.configure(config);
    encoder = createdEncoder;
  } catch (error) {
    try {
      createdEncoder?.close();
    } catch {
      // Ignore teardown races after a failed configure().
    }
    warn(
      `VideoEncoder could not be configured for ${selectedCodec}: ${describeError(error)}`,
    );
    return null;
  }

  const currentQueueSize = (): number => {
    if (closed) return 0;
    try {
      return encoder.encodeQueueSize;
    } catch {
      return 0;
    }
  };

  function closeEncoder(): void {
    if (closed) return;
    closed = true;
    logDebug('encoder', 'Closing VideoEncoder', {
      framesSubmitted,
      framesClosed,
      chunksEmitted,
      keyFrames,
      lastError,
    });
    try {
      encoder.close();
    } catch {
      // Already closed or torn down by the UA.
    }
  }

  function status(): EncoderTapStatus {
    const state: EncoderTapState = failed
      ? 'errored'
      : closed
        ? 'closed'
        : 'configured';
    return {
      state,
      codec: selectedCodec,
      frameFormat,
      colorSpace: `bt2020/${transfer}/bt2020-ncl`,
      framesSubmitted,
      framesClosed,
      chunksEmitted,
      chunkBytes,
      keyFrames,
      encodeQueueSize: currentQueueSize(),
      lastError,
      closed,
    };
  }

  async function drain(targetQueueSize: number = 4): Promise<void> {
    if (closed || failed) return;
    const queued = currentQueueSize();
    if (queued <= targetQueueSize) return;
    logDebug('encoder', 'Draining encoder queue to target', {
      encodeQueueSize: queued,
      targetQueueSize,
      framesSubmitted,
      chunksEmitted,
    });
    await new Promise<void>((resolve) => {
      let settled = false;
      const cleanup = () => {
        if (settled) return;
        settled = true;
        encoder.removeEventListener('dequeue', onDequeue);
        clearTimeout(timer);
        resolve();
      };

      const onDequeue = () => {
        if (currentQueueSize() <= targetQueueSize || closed || failed) {
          cleanup();
        }
      };

      encoder.addEventListener('dequeue', onDequeue);

      // Fallback timer in case dequeue is delayed or dropped by the platform
      const timer = setTimeout(cleanup, 250);
    });
  }

  function flush(): Promise<void> {
    if (closed) {
      logDebug('encoder', 'Flush requested after close; nothing to do');
      return Promise.resolve();
    }
    if (flushPromise) return flushPromise;

    const flushStartedAt = performance.now();
    logInfo('encoder', 'Flush started', {
      framesSubmitted,
      chunksEmitted,
      encodeQueueSize: currentQueueSize(),
    });

    flushPromise = (async () => {
      try {
        await encoder.flush();
        logInfo('encoder', 'Flush complete', {
          elapsedMs: Math.round(performance.now() - flushStartedAt),
          framesSubmitted,
          framesClosed,
          chunksEmitted,
          chunkBytes,
          keyFrames,
        });
      } catch (error) {
        // Flush can reject when the encoder was already torn down; keep the
        // original error visible in telemetry but still close cleanly.
        reportError(error, true);
      } finally {
        closeEncoder();
      }
    })();
    return flushPromise;
  }

  return {
    codec: selectedCodec,
    frameFormat,
    status,
    encodeQueueSize: currentQueueSize,
    drain,
    flush,
    close: closeEncoder,

    encode(frame, keyFrame = false): boolean {
      if (closed || failed) return false;

      if (frame.meta.width !== width || frame.meta.height !== height) {
        reportError(
          new Error(
            `frame ${frame.meta.frameIndex} is ${frame.meta.width}x${frame.meta.height}; ` +
              `encoder was configured for ${width}x${height}`,
          ),
          true,
        );
        return false;
      }

      let videoFrame: VideoFrame | null = null;
      try {
        const data = packer.pack(frame);
        const init = packer.init(frame);
        videoFrame = new VideoFrame(
          data as unknown as AllowSharedBufferSource,
          init as unknown as VideoFrameBufferInit,
        );

        const shouldKey =
          keyFrame ||
          (keyFrameInterval > 0 && framesSubmitted % keyFrameInterval === 0);
        encoder.encode(videoFrame, { keyFrame: shouldKey });
        framesSubmitted++;
        return true;
      } catch (error) {
        reportError(error, true);
        return false;
      } finally {
        // The encoder clones the frame during encode(), so the caller-side
        // VideoFrame must be released immediately to keep the pool available.
        if (videoFrame) {
          try {
            videoFrame.close();
            framesClosed++;
          } catch {
            // Closing an already-detached frame is harmless.
          }
        }
      }
    },
  };
}

/**
 * Submits a single frame to the encoder tap.
 *
 * A no-op that returns `false` when `tap` is `null`, closed, or errored. The
 * intermediate `VideoFrame` is closed inside the tap before this returns.
 *
 * @param tap - Tap from `createVideoEncoderTap`, or `null` to skip.
 * @param frame - Frame to encode; `pixels` must remain valid for the call.
 * @param keyFrame - Whether to force an IDR/keyframe for this frame.
 * @returns `true` if the frame was accepted, `false` if skipped/unsupported.
 */
export function encodeFrame(
  tap: VideoEncoderTap | null,
  frame: FloatFrame,
  keyFrame: boolean = false,
): boolean {
  if (!tap) return false;
  return tap.encode(frame, keyFrame);
}

/**
 * Flushes and closes the encoder tap.
 *
 * `await tap.flush()` emits trailing chunks and then closes the encoder. This
 * wrapper also calls `tap.close()` in a `finally` block, so a custom tap whose
 * `flush()` resolves early or rejects still releases its encoder. Idempotent and
 * safe when `tap` is `null`.
 *
 * @param tap - Tap to finalize, or `null` (no-op).
 */
export async function flushAndClose(tap: VideoEncoderTap | null): Promise<void> {
  if (!tap) return;
  try {
    await tap.flush();
  } catch (error) {
    // A rejected flush must not prevent resource release.
    logError('encoder', 'flushAndClose: tap.flush() failed', error);
  } finally {
    try {
      tap.close();
    } catch (error) {
      // Ignore teardown races.
      logDebug('encoder', 'flushAndClose: tap.close() threw during teardown', {
        error: describeError(error),
      });
    }
  }
}
