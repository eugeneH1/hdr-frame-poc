/**
 * Optional downstream WebCodecs tap.
 *
 * Responsibility: feed produced `FloatFrame`s into a `VideoEncoder` for future
 * export (e.g. Rec. 2100 PQ HEVC/AV1) once core frame generation is proven. This
 * module is intentionally **optional**: `runFramePipeline` should work with the tap
 * absent. A `VideoFrame` is constructed per `FloatFrame` using the declared
 * `VideoFrameInit` (format, `colorSpace`, timestamp) and closed as soon as the
 * encoder accepts it to avoid exhausting the decoder/encoder pool.
 *
 * Design assumptions:
 * - `VideoFrame` here wraps a CPU-side buffer (`BufferSource`); a GPU-backed path
 *   (importing the `GPUTexture` directly) is a later optimization.
 * - Encoder output is left to a caller-supplied `output` callback; the tap never
 *   buffers encoded chunks itself.
 */

import type { FloatFrame, FrameMeta } from '../types/frame.ts';

/** Summarizes how far the encoder has progressed. */
export interface EncoderTapStatus {
  /** Number of frames successfully submitted to the encoder. */
  readonly framesSubmitted: number;
  /** Number of frames for which `close()` was called after submission. */
  readonly framesClosed: number;
  /** Whether the encoder has been flushed and closed. */
  readonly closed: boolean;
}

/** Handle to the optional encoder tap. */
export interface VideoEncoderTap {
  /** Current tap status (submitted/closed counts). */
  readonly status: () => EncoderTapStatus;
  /** Frames still held by the encoder's internal queue (backpressure signal). */
  readonly encodeQueueSize: () => number;
}

/**
 * Creates a `VideoEncoder`-backed tap if WebCodecs and a supported codec are
 * available; otherwise returns `null` so the pipeline can continue without it.
 *
 * Intended steps: probe `VideoEncoder.isConfigSupported({ codec, ... })`, construct
 * the encoder, wire `output`/`error` callbacks, and expose status.
 *
 * @param meta - Metadata for the first frame, used to derive codec/color settings.
 * @param output - Receives encoded chunks (and their metadata) for muxing elsewhere.
 * @returns A `VideoEncoderTap`, or `null` when unsupported.
 */
export function createVideoEncoderTap(
  _meta: FrameMeta,
  _output: (chunk: EncodedVideoChunk, metadata: EncodedVideoChunkMetadata | undefined) => void,
): VideoEncoderTap | null {
  throw new Error('encoder.webCodecsTap.createVideoEncoderTap: not implemented');
}

/**
 * Submits a single frame to the encoder tap.
 *
 * Intended steps: build a `VideoFrame` from `frame.pixels` using `frame.meta`
 * (timestamp/duration/colorSpace), call `encoder.encode(...)`, then `frame.close()`
 * immediately. Should be a no-op when `tap` is `null`.
 *
 * @param tap - Tap from `createVideoEncoderTap`, or `null` to skip.
 * @param frame - Frame to encode; `pixels` must remain valid for the call.
 * @param keyFrame - Whether to force an IDR/keyframe for this frame.
 * @returns `true` if the frame was accepted, `false` if skipped/unsupported.
 */
export function encodeFrame(
  _tap: VideoEncoderTap | null,
  _frame: FloatFrame,
  _keyFrame: boolean = false,
): boolean {
  throw new Error('encoder.webCodecsTap.encodeFrame: not implemented');
}

/**
 * Flushes and closes the encoder tap.
 *
 * Intended steps: `await encoder.flush()` so trailing frames are emitted, then
 * `encoder.close()`. Should be idempotent and safe when `tap` is `null`.
 *
 * @param tap - Tap to finalize, or `null` (no-op).
 */
export async function flushAndClose(_tap: VideoEncoderTap | null): Promise<void> {
  throw new Error('encoder.webCodecsTap.flushAndClose: not implemented');
}
