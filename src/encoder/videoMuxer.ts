/**
 * WebCodecs chunk -> container muxer.
 *
 * Responsibility: take the `EncodedVideoChunk`s emitted by the optional
 * Rec.2100 HDR `VideoEncoder` tap and write them into a playable, downloadable
 * file using Mediabunny. This module contains no encoding or rendering logic; it
 * only packages already-encoded packets.
 *
 * Design assumptions:
 * - The first chunk is a key frame and carries a `decoderConfig` (including the
 *   codec-specific `description` needed by HEVC/AV1). The tap forces frame 0 to
 *   be a key frame.
 * - Chunks arrive in decode order, as required by `EncodedVideoPacketSource`.
 * - Color metadata is authoritative from this pipeline: each output frame was
 *   authored as Rec.2020 / PQ or HLG / `bt2020-ncl`. We overlay that on the
 *   decoder config so the container gets a `colr`/`Colour` element even if the
 *   browser's encoder metadata omits it.
 * - MP4 is used for HEVC/AV1 (broad HDR playback support); WebM is used for VP9
 *   because VP9-in-MP4 is much less portable.
 * - The entire 10-second file is held in memory (~a few MB at the POC bitrate),
 *   which keeps the download path simple.
 */

import {
  ALL_FORMATS,
  BlobSource,
  BufferTarget,
  EncodedPacket,
  EncodedVideoPacketSource,
  Input,
  Mp4OutputFormat,
  Output,
  WebMOutputFormat,
} from 'mediabunny';
import { logDebug, logInfo } from '../diagnostics/logger.ts';

/** Container codec names that this pipeline knows how to mux. */
export type MuxableVideoCodec = 'hevc' | 'av1' | 'vp9';

/** One encoded chunk plus the metadata WebCodecs emitted alongside it. */
export interface CollectedVideoChunk {
  readonly chunk: EncodedVideoChunk;
  readonly metadata?: EncodedVideoChunkMetadata;
}

/** Everything needed to mux one run's encoded output. */
export interface MuxVideoOptions {
  readonly chunks: readonly CollectedVideoChunk[];
  /** Full codec string from the encoder tap (e.g. `hvc1.2.4.L93.B0`). */
  readonly codec: string;
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  /** Rec.2100 transfer used by the encoder tap. */
  readonly transfer: 'pq' | 'hlg';
}

/** A finalized in-memory video file ready for download. */
export interface MuxedVideo {
  readonly buffer: ArrayBuffer;
  readonly mimeType: string;
  /** Container extension, including the dot (`.mp4` or `.webm`). */
  readonly extension: string;
  readonly codec: MuxableVideoCodec;
  readonly transfer: 'pq' | 'hlg';
  readonly width: number;
  readonly height: number;
  readonly frameCount: number;
}

/** Metadata read back out of the finalized file, proving what reached the container. */
export interface MuxedVideoInspection {
  readonly codec: string | null;
  readonly width: number;
  readonly height: number;
  readonly durationSeconds: number | null;
  readonly colorSpace: VideoColorSpaceInit;
  readonly highDynamicRange: boolean;
}

/** Maps a WebCodecs codec string onto a Mediabunny container codec name. */
export function muxableCodecFromString(codec: string): MuxableVideoCodec | null {
  if (codec.startsWith('hvc1') || codec.startsWith('hev1')) return 'hevc';
  if (codec.startsWith('av01')) return 'av1';
  if (codec.startsWith('vp09')) return 'vp9';
  return null;
}

/**
 * Muxes encoded HDR chunks into MP4 (HEVC/AV1) or WebM (VP9).
 *
 * @param options - Encoded chunks, codec string, dimensions/rate and transfer.
 * @returns The finalized file, or `null` when there were no chunks to mux.
 * @throws If the codec is unsupported, the first chunk is not a key frame, or
 *   the muxer rejects the stream (for example a missing codec description).
 */
export async function muxVideoChunks(options: MuxVideoOptions): Promise<MuxedVideo | null> {
  const { chunks, codec, width, height, fps, transfer } = options;
  if (chunks.length === 0) return null;

  const muxCodec = muxableCodecFromString(codec);
  if (!muxCodec) {
    throw new Error(`encoder.videoMuxer: unsupported codec string "${codec}"`);
  }
  if (chunks[0].chunk.type !== 'key') {
    throw new Error('encoder.videoMuxer: first encoded chunk is not a key frame');
  }

  const format =
    muxCodec === 'vp9'
      ? new WebMOutputFormat()
      : new Mp4OutputFormat({ fastStart: 'in-memory' });

  if (!format.getSupportedVideoCodecs().includes(muxCodec)) {
    throw new Error(
      `encoder.videoMuxer: ${format.fileExtension} cannot contain ${muxCodec}`,
    );
  }

  logInfo('muxer', 'Muxing encoded HDR video', {
    codec,
    muxCodec,
    container: format.fileExtension,
    chunks: chunks.length,
    width,
    height,
    fps,
    transfer,
  });

  const target = new BufferTarget();
  const output = new Output({ format, target });
  const source = new EncodedVideoPacketSource(muxCodec);
  output.addVideoTrack(source, { frameRate: fps });

  // UAs normally attach the decoder config to the first output, but locate the
  // first chunk that actually has one rather than assuming it is index 0.
  const firstConfigured = chunks.find((entry) => entry.metadata?.decoderConfig);

  try {
    await output.start();

    for (let index = 0; index < chunks.length; index++) {
      const { chunk } = chunks[index];
      const packet = EncodedPacket.fromEncodedChunk(chunk);

      if (index === 0) {
        const metadata = firstConfigured?.metadata;
        const baseConfig = metadata?.decoderConfig;
        // Authoritative HDR description for the container. Keep the codec's
        // `description` (hvcC/av1C/VP9 config) but replace color metadata.
        // The bundled DOM typings omit the PQ/HLG transfer values, so cast the
        // local Rec.2100 description at the WebCodecs boundary.
        const intendedColorSpace = {
          primaries: 'bt2020',
          transfer,
          matrix: 'bt2020-ncl',
          fullRange: false,
        } as unknown as VideoColorSpaceInit;
        const decoderConfig: VideoDecoderConfig = {
          ...(baseConfig ?? {}),
          codec: baseConfig?.codec ?? codec,
          codedWidth: baseConfig?.codedWidth ?? width,
          codedHeight: baseConfig?.codedHeight ?? height,
          colorSpace: intendedColorSpace,
        };
        const firstMetadata: EncodedVideoChunkMetadata = {
          ...metadata,
          decoderConfig,
        };

        logDebug('muxer', 'First packet decoder config', {
          codec: decoderConfig.codec,
          codedWidth: decoderConfig.codedWidth,
          codedHeight: decoderConfig.codedHeight,
          colorSpace: decoderConfig.colorSpace,
          hasDescription: Boolean(decoderConfig.description),
          encoderColorSpace: baseConfig?.colorSpace ?? null,
        });

        await source.add(packet, firstMetadata);
      } else {
        await source.add(packet);
      }
    }
  } catch (error) {
    await output.cancel().catch(() => undefined);
    throw error;
  } finally {
    source.close();
  }

  await output.finalize();

  const buffer = target.buffer;
  if (!buffer) {
    throw new Error('encoder.videoMuxer: muxer finalized without producing a buffer');
  }

  logInfo('muxer', 'HDR video muxed', {
    container: format.fileExtension,
    mimeType: format.mimeType,
    codec: muxCodec,
    chunks: chunks.length,
    bytes: buffer.byteLength,
  });

  return {
    buffer,
    mimeType: format.mimeType,
    extension: format.fileExtension,
    codec: muxCodec,
    transfer,
    width,
    height,
    frameCount: chunks.length,
  };
}

/**
 * Reads the finalized file back through Mediabunny's demuxer and reports the
 * codec, dimensions, duration, colour tags and HDR flag actually present in the
 * container. This is the in-browser equivalent of an `ffprobe` sanity check.
 */
export async function inspectMuxedVideo(video: MuxedVideo): Promise<MuxedVideoInspection> {
  const input = new Input({
    formats: ALL_FORMATS,
    source: new BlobSource(new Blob([video.buffer], { type: video.mimeType })),
  });

  try {
    const track = await input.getPrimaryVideoTrack();
    if (!track) {
      throw new Error('encoder.videoMuxer: muxed file contains no video track');
    }

    const [codec, width, height, colorSpace, highDynamicRange, durationSeconds] =
      await Promise.all([
        track.getCodec(),
        track.getCodedWidth(),
        track.getCodedHeight(),
        track.getColorSpace(),
        track.hasHighDynamicRange(),
        input.computeDuration(),
      ]);

    return { codec, width, height, durationSeconds, colorSpace, highDynamicRange };
  } finally {
    input.dispose();
  }
}
