/**
 * Container selection for encoded HDR output.
 *
 * Responsibility: choose the right container for a codec string and normalise
 * the result, so callers (the workbench) do not need to know which muxer ran.
 *
 * Rationale: codec availability differs sharply between browsers. Chrome on
 * some platforms exposes **no AV1 or HEVC encoder**, only VP9 Profile 2
 * (10-bit) and H.264. VP9 belongs in WebM; AV1/HEVC belong in MP4. Rather than
 * forcing one container, the export path asks this module what to do.
 *
 * This module contains no encoding or container-writing logic of its own.
 */

import { canMuxCodec, muxVideoToMp4 } from './mp4Muxer.ts';
import type { MuxVideoChunk } from './mp4Muxer.ts';

export type { MuxVideoChunk };
import { muxVideoToWebM } from './webmMuxer.ts';

/** A finalized video file, regardless of container. */
export interface MuxedVideoFile {
  readonly buffer: ArrayBuffer;
  readonly mimeType: string;
  /** Container extension without the dot: `mp4` or `webm`. */
  readonly extension: string;
  readonly codecFamily: string;
  readonly frameCount: number;
  readonly durationSeconds: number;
}

/** Options shared by both muxers. */
export interface MuxOptions {
  readonly chunks: readonly MuxVideoChunk[];
  readonly codec: string;
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  readonly transfer: 'pq' | 'hlg';
}

/** Whether any supported container can carry this codec. */
export function canMuxVideo(codec: string): boolean {
  return canMuxCodec(codec) || codec.startsWith('vp09') || codec.startsWith('vp8');
}

/**
 * Muxes encoded chunks into the container that matches the codec.
 *
 * @param options - Chunks, codec string, dimensions, rate, and transfer.
 * @returns The finalized file, or `null` when there are no chunks.
 * @throws If no container supports the codec, or the muxer rejects the stream.
 */
export function muxVideo(options: MuxOptions): MuxedVideoFile | null {
  const { codec } = options;

  if (codec.startsWith('vp09') || codec.startsWith('vp8')) {
    const webm = muxVideoToWebM(options);
    if (!webm) return null;
    return {
      buffer: webm.buffer,
      mimeType: webm.mimeType,
      extension: webm.extension,
      codecFamily: webm.codecFamily,
      frameCount: webm.frameCount,
      durationSeconds: webm.durationSeconds,
    };
  }

  if (canMuxCodec(codec)) {
    const mp4 = muxVideoToMp4(options);
    if (!mp4) return null;
    return {
      buffer: mp4.buffer,
      mimeType: mp4.mimeType,
      extension: mp4.extension,
      codecFamily: mp4.codecFamily,
      frameCount: mp4.frameCount,
      durationSeconds: mp4.durationSeconds,
    };
  }

  throw new Error(
    `encoder.videoContainer: no container supports codec "${codec}" ` +
      '(expected AV1/HEVC/H.264 for MP4, or VP8/VP9 for WebM)',
  );
}
