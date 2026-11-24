/**
 * Minimal in-memory MP4 muxer for WebCodecs `EncodedVideoChunk`s.
 *
 * Responsibility: wrap an already-encoded AV1/H.264/HEVC packet stream in an
 * ISO base media file so it can be downloaded and played. This module contains
 * no encoding or rendering logic and has no external dependencies.
 *
 * Scope and deliberate limitations:
 * - **One video track, no audio.** Every sample belongs to a single track, so
 *   the multi-track interleaving problem does not arise.
 * - **Progressive (non-fragmented) MP4.** The whole file is built in memory and
 *   `moov` is written after `mdat`. A 10-second 720p clip is a few MB, so a
 *   streaming target would be premature complexity.
 * - **Codec config comes from WebCodecs or is reconstructed for AV1.** The
 *   first chunk carrying `decoderConfig.description` supplies the record
 *   verbatim (`hvcC`/`avcC`). Chrome's AV1 encoder provides no description, so
 *   an `av1C` record is rebuilt from the codec string plus the sequence-header
 *   OBU found in the first key frame. Codec *bitstream* bytes are never
 *   synthesized, only the container-level configuration record.
 * - **Colour metadata is authored explicitly.** Rec. 2020 / PQ or HLG /
 *   `bt2020-ncl` is written to `colr`, because browsers often omit it from the
 *   bitstream even when the encoder was configured for it.
 */

import { logDebug } from '../diagnostics/logger.ts';

/** One encoded chunk plus the metadata that accompanied it. */
export interface MuxVideoChunk {
  readonly chunk: EncodedVideoChunk;
  readonly metadata?: EncodedVideoChunkMetadata;
}

/** Options for {@link muxVideoToMp4}. */
export interface MuxVideoOptions {
  readonly chunks: readonly MuxVideoChunk[];
  /** Full codec string from the encoder, e.g. `av01.0.08M.10`. */
  readonly codec: string;
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  /** Rec. 2100 transfer function used to author the frames. */
  readonly transfer: 'pq' | 'hlg';
}

/** A finalized MP4 file ready for download. */
export interface MuxedVideo {
  readonly buffer: ArrayBuffer;
  readonly mimeType: string;
  readonly extension: string;
  readonly codecFamily: 'av1' | 'hevc' | 'avc';
  readonly frameCount: number;
  readonly durationSeconds: number;
}

/** ISO/IEC 14496-12 four-character box type. */
type FourCC = string;

const MAX_UINT32 = 0xffffffff;

/**
 * Growable big-endian byte writer, used to lay out boxes without allocating a
 * temporary array per field.
 */
class ByteWriter {
  private buffer = new Uint8Array(256);
  private length = 0;

  private ensure(extra: number): void {
    if (this.length + extra <= this.buffer.length) return;
    let next = this.buffer.length * 2;
    while (next < this.length + extra) next *= 2;
    const grown = new Uint8Array(next);
    grown.set(this.buffer.subarray(0, this.length));
    this.buffer = grown;
  }

  u8(value: number): this {
    this.ensure(1);
    this.buffer[this.length++] = value & 0xff;
    return this;
  }

  u16(value: number): this {
    this.ensure(2);
    this.buffer[this.length++] = (value >>> 8) & 0xff;
    this.buffer[this.length++] = value & 0xff;
    return this;
  }

  u32(value: number): this {
    this.ensure(4);
    this.buffer[this.length++] = (value >>> 24) & 0xff;
    this.buffer[this.length++] = (value >>> 16) & 0xff;
    this.buffer[this.length++] = (value >>> 8) & 0xff;
    this.buffer[this.length++] = value & 0xff;
    return this;
  }

  ascii(text: string): this {
    for (let i = 0; i < text.length; i++) this.u8(text.charCodeAt(i));
    return this;
  }

  bytes(data: Uint8Array): this {
    this.ensure(data.length);
    this.buffer.set(data, this.length);
    this.length += data.length;
    return this;
  }

  zeros(count: number): this {
    this.ensure(count);
    this.length += count;
    return this;
  }

  /** A copy of exactly the bytes written so far. */
  finish(): Uint8Array {
    return this.buffer.slice(0, this.length);
  }
}

/** Builds `size + type + payload`. */
function box(type: FourCC, payload: Uint8Array): Uint8Array {
  if (payload.length + 8 > MAX_UINT32) {
    throw new Error(`encoder.mp4Muxer: box "${type}" exceeds the 32-bit size limit`);
  }
  const out = new Uint8Array(8 + payload.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, out.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(payload, 8);
  return out;
}

/** Concatenates byte arrays. */
function concat(parts: readonly Uint8Array[]): Uint8Array {
  let total = 0;
  for (const part of parts) total += part.length;
  const out = new Uint8Array(total);
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

/** Builds a container box from child boxes. */
function container(type: FourCC, children: readonly Uint8Array[]): Uint8Array {
  return box(type, concat(children));
}

/** Converts microseconds to a media timescale (integer, rounded). */
function microsecondsToTimescale(us: number, timescale: number): number {
  return Math.round((us * timescale) / 1_000_000);
}

/** Maps a codec string to its MP4 sample-entry family. */
function codecFamily(codec: string): 'av1' | 'hevc' | 'avc' {
  if (codec.startsWith('av01')) return 'av1';
  if (codec.startsWith('hvc1') || codec.startsWith('hev1')) return 'hevc';
  if (codec.startsWith('avc1') || codec.startsWith('avc3')) return 'avc';
  throw new Error(`encoder.mp4Muxer: unsupported codec string "${codec}"`);
}

/**
 * Whether this muxer can carry the given codec string.
 *
 * Hosts should check this *before* encoding: a full encode pass followed by a
 * mux rejection wastes time and surfaces the failure long after the cause.
 * VP9 is deliberately absent; VP9-in-MP4 has poor player support, so a WebM
 * muxer would be required instead.
 */
export function canMuxCodec(codec: string): boolean {
  return (
    codec.startsWith('av01') ||
    codec.startsWith('hvc1') ||
    codec.startsWith('hev1') ||
    codec.startsWith('avc1') ||
    codec.startsWith('avc3')
  );
}

/** The sample entry and configuration box types for a codec family. */
function sampleEntryType(family: 'av1' | 'hevc' | 'avc'): {
  entry: FourCC;
  config: FourCC;
} {
  switch (family) {
    case 'av1':
      return { entry: 'av01', config: 'av1C' };
    case 'hevc':
      return { entry: 'hvc1', config: 'hvcC' };
    case 'avc':
      return { entry: 'avc1', config: 'avcC' };
  }
}

/** ITU-T H.273 code points for Rec. 2100 colour. */
const PRIMARIES_BT2020 = 9;
const TRANSFER_PQ = 16;
const TRANSFER_HLG = 18;
const MATRIX_BT2020_NCL = 9;
/** Bit 7 of the final byte is `full_range_flag`; the encoder packs limited range. */
const LIMITED_RANGE = 0x00;

/**
 * Builds `colr` declaring Rec. 2020 / PQ-or-HLG / bt2020-ncl.
 *
 * `nclx` (not `nclc`) is required: the PQ and HLG transfer codes are above 10
 * and only exist in the `nclx` flavour.
 */
function colrBox(transfer: 'pq' | 'hlg'): Uint8Array {
  return box(
    'colr',
    new ByteWriter()
      .ascii('nclx')
      .u16(PRIMARIES_BT2020)
      .u16(transfer === 'pq' ? TRANSFER_PQ : TRANSFER_HLG)
      .u16(MATRIX_BT2020_NCL)
      .u8(LIMITED_RANGE)
      .finish(),
  );
}

/** `pasp` with square pixels. */
function paspBox(): Uint8Array {
  return box('pasp', new ByteWriter().u32(1).u32(1).finish());
}

/** Codec configuration box, embedding WebCodecs' `description` verbatim. */
function decoderConfigBox(family: 'av1' | 'hevc' | 'avc', description: ArrayBuffer): Uint8Array {
  return box(sampleEntryType(family).config, new Uint8Array(description));
}

/** `av01`/`hvc1`/`avc1` visual sample entry. */
function visualSampleEntry(
  family: 'av1' | 'hevc' | 'avc',
  width: number,
  height: number,
  description: ArrayBuffer,
  transfer: 'pq' | 'hlg',
): Uint8Array {
  const header = new ByteWriter()
    .zeros(6) // reserved
    .u16(1) // data_reference_index
    .u16(0) // pre_defined
    .u16(0) // reserved
    .zeros(12) // pre_defined[3]
    .u16(width)
    .u16(height)
    .u32(0x00480000) // 72 dpi horizontal (16.16)
    .u32(0x00480000) // 72 dpi vertical (16.16)
    .u32(0) // reserved
    .u16(1) // frame_count
    .zeros(32) // compressorname (length-prefixed, unused)
    .u16(0x0018) // depth: colour, no alpha
    .u16(0xffff) // pre_defined = -1
    .finish();

  return box(
    sampleEntryType(family).entry,
    concat([
      header,
      decoderConfigBox(family, description),
      colrBox(transfer),
      paspBox(),
    ]),
  );
}

/** One sample's timing, size, and offset within `mdat`. */
interface SampleRecord {
  readonly durationTimescale: number;
  readonly size: number;
  readonly isKey: boolean;
}

/** `stts`: run-length encoded sample durations. */
function sttsBox(samples: readonly SampleRecord[]): Uint8Array {
  const runs: Array<{ count: number; delta: number }> = [];
  for (const sample of samples) {
    const last = runs[runs.length - 1];
    if (last && last.delta === sample.durationTimescale) last.count++;
    else runs.push({ count: 1, delta: sample.durationTimescale });
  }
  const writer = new ByteWriter().u32(0).u32(runs.length);
  for (const run of runs) writer.u32(run.count).u32(run.delta);
  return box('stts', writer.finish());
}

/** `stss`: one-based sample numbers of sync frames. */
function stssBox(samples: readonly SampleRecord[]): Uint8Array {
  const keys: number[] = [];
  samples.forEach((sample, index) => {
    if (sample.isKey) keys.push(index + 1);
  });
  const writer = new ByteWriter().u32(0).u32(keys.length);
  for (const key of keys) writer.u32(key);
  return box('stss', writer.finish());
}

/** `stsc`: all samples live in a single chunk (1-based, first_chunk=1). */
function stscBox(): Uint8Array {
  return box('stsc', new ByteWriter().u32(0).u32(1).u32(1).u32(1).u32(1).finish());
}

/** `stsz`: per-sample byte sizes. */
function stszBox(samples: readonly SampleRecord[]): Uint8Array {
  const writer = new ByteWriter().u32(0).u32(0).u32(samples.length);
  for (const sample of samples) writer.u32(sample.size);
  return box('stsz', writer.finish());
}

/** `stco`: absolute byte offset of the single chunk. */
function stcoBox(dataOffset: number): Uint8Array {
  return box('stco', new ByteWriter().u32(0).u32(1).u32(dataOffset).finish());
}

/** `dinf` with a self-contained `dref`, required by most demuxers. */
function dinfBox(): Uint8Array {
  const url = box('url ', new ByteWriter().u32(1).finish());
  return container('dinf', [box('dref', new ByteWriter().u32(0).u32(1).bytes(url).finish())]);
}

/** `vmhd`: video media header. */
function vmhdBox(): Uint8Array {
  return box('vmhd', new ByteWriter().u32(1).u16(0).u16(0).u16(0).u16(0).finish());
}

/** `hdlr`: declares the track as video and gives it a name. */
function hdlrBox(): Uint8Array {
  return box(
    'hdlr',
    concat([
      new ByteWriter().u32(0).u32(0).ascii('vide').zeros(12).finish(),
      new ByteWriter().ascii('HDR').u8(0).finish(),
    ]),
  );
}

/** `mdhd`: media timescale and duration. */
function mdhdBox(timescale: number, durationTimescale: number): Uint8Array {
  return box(
    'mdhd',
    new ByteWriter()
      .u32(0) // version/flags
      .u32(0) // creation_time
      .u32(0) // modification_time
      .u32(timescale)
      .u32(durationTimescale)
      .u16(0x55c4) // language: "und"
      .u16(0) // pre_defined
      .finish(),
  );
}

/** Identity display matrix shared by `mvhd` and `tkhd`. */
function identityMatrix(): Uint8Array {
  return new ByteWriter()
    .u32(0x00010000).u32(0).u32(0)
    .u32(0).u32(0x00010000).u32(0)
    .u32(0).u32(0).u32(0x40000000)
    .finish();
}

/** `tkhd`: track header, enabled and in the movie. */
function tkhdBox(trackId: number, durationMovie: number, width: number, height: number): Uint8Array {
  return box(
    'tkhd',
    concat([
      new ByteWriter()
        .u32(0x00000003) // version 0, flags: enabled | in movie
        .u32(0) // creation_time
        .u32(0) // modification_time
        .u32(trackId)
        .u32(0) // reserved
        .u32(durationMovie)
        .zeros(8) // reserved
        .u16(0) // layer
        .u16(0) // alternate_group
        .u16(0) // volume (0 for video)
        .u16(0) // reserved
        .finish(),
      identityMatrix(),
      new ByteWriter().u32(width << 16).u32(height << 16).finish(),
    ]),
  );
}

/** `mvhd`: movie header. */
function mvhdBox(timescale: number, durationMovie: number): Uint8Array {
  return box(
    'mvhd',
    concat([
      new ByteWriter()
        .u32(0) // version/flags
        .u32(0) // creation_time
        .u32(0) // modification_time
        .u32(timescale)
        .u32(durationMovie)
        .u32(0x00010000) // rate 1.0
        .u16(0x0100) // volume 1.0
        .zeros(10) // reserved
        .finish(),
      identityMatrix(),
      new ByteWriter().zeros(24).u32(2).finish(), // pre_defined[6] + next_track_ID
    ]),
  );
}

/**
 * Normalises a WebCodecs `description` to a standalone `ArrayBuffer`.
 *
 * `decoderConfig.description` is a `BufferSource`, which may be a view with a
 * non-zero offset; slicing keeps the resulting box bytes exact.
 */
function toArrayBuffer(source: AllowSharedBufferSource): ArrayBuffer {
  if (source instanceof ArrayBuffer) return source;
  if (!ArrayBuffer.isView(source)) {
    // A SharedArrayBuffer: copy it into a plain ArrayBuffer for the file bytes.
    return new Uint8Array(source as ArrayBufferLike).slice().buffer as ArrayBuffer;
  }
  return source.buffer.slice(source.byteOffset, source.byteOffset + source.byteLength) as ArrayBuffer;
}

/** AV1 OBU type for a sequence header (AV1 spec, section 6.2.2). */
const AV1_OBU_SEQUENCE_HEADER = 1;

/** Reads an unsigned LEB128 value; returns the value and the next byte index. */
function readLeb128(bytes: Uint8Array, offset: number): { value: number; next: number } {
  let value = 0;
  let shift = 0;
  let index = offset;
  for (;;) {
    if (index >= bytes.length) {
      throw new Error('encoder.mp4Muxer: truncated LEB128 value in OBU stream');
    }
    const byte = bytes[index++];
    value += (byte & 0x7f) * 2 ** shift;
    if ((byte & 0x80) === 0) break;
    shift += 7;
    if (shift > 56) {
      throw new Error('encoder.mp4Muxer: oversized LEB128 value in OBU stream');
    }
  }
  return { value, next: index };
}

/**
 * Finds the first AV1 sequence header OBU in an encoded keyframe.
 *
 * AV1 samples are a sequence of length-delimited OBUs. Keyframes normally begin
 * with a temporal delimiter followed by the sequence header, but its position is
 * not assumed: the whole sample is scanned.
 *
 * @returns The complete OBU (header, size field, and payload), or `null`.
 */
function findAv1SequenceHeaderObu(sample: Uint8Array): Uint8Array | null {
  let offset = 0;
  while (offset < sample.length) {
    const headerOffset = offset;
    const header = sample[offset++];
    const obuType = (header >> 3) & 0x0f;
    const hasExtension = (header & 0x04) !== 0;
    const hasSize = (header & 0x02) !== 0;
    if (hasExtension) offset += 1;
    if (!hasSize) {
      // Low-overhead bitstreams always carry a size field; without one the
      // remaining sample is a single OBU.
      if (obuType === AV1_OBU_SEQUENCE_HEADER) return sample.subarray(headerOffset);
      return null;
    }
    const { value: payloadSize, next } = readLeb128(sample, offset);
    offset = next;
    const end = offset + payloadSize;
    if (end > sample.length) {
      throw new Error('encoder.mp4Muxer: truncated OBU payload in AV1 keyframe');
    }
    if (obuType === AV1_OBU_SEQUENCE_HEADER) return sample.subarray(headerOffset, end);
    offset = end;
  }
  return null;
}

/** Fields from an `av01.<profile>.<level><tier>.<bitDepth>` codec string. */
interface Av1CodecStringInfo {
  readonly profile: number;
  readonly level: number;
  readonly tier: number;
  readonly bitDepth: number;
}

/** Parses the AV1 codec string, e.g. `av01.0.08M.10`. */
function parseAv1CodecString(codec: string): Av1CodecStringInfo {
  const match = /^av01\.(\d+)\.(\d{2})([MH])\.(\d+)$/.exec(codec);
  if (!match) {
    throw new Error(`encoder.mp4Muxer: cannot parse AV1 codec string "${codec}"`);
  }
  return {
    profile: Number(match[1]),
    level: Number(match[2]),
    tier: match[3] === 'H' ? 1 : 0,
    bitDepth: Number(match[4]),
  };
}

/**
 * Builds an AV1CodecConfigurationRecord (`av1C`) for a keyframe sample.
 *
 * Chrome's AV1 `VideoEncoder` emits no `decoderConfig.description`, so the
 * record has to be reconstructed. The header fields come from the codec string
 * (which WebCodecs guarantees describes the output), and `configOBUs` carries
 * the sequence header OBU copied out of the first keyframe so players have the
 * same configuration the bitstream does.
 *
 * Record layout (AV1 ISOBMFF binding):
 *   marker(1) version(7) | seq_profile(3) seq_level_idx_0(5) |
 *   seq_tier_0(1) high_bitdepth(1) twelve_bit(1) monochrome(1)
 *   chroma_subsampling_x(1) chroma_subsampling_y(1) chroma_sample_position(2) |
 *   reserved(3) initial_presentation_delay_present(1) delay_or_reserved(4) |
 *   configOBUs[]
 *
 * @param codec - AV1 codec string from the encoder tap.
 * @param keySample - Byte payload of the first key frame.
 * @returns The `av1C` payload bytes (without the box header).
 */
function synthesizeAv1Config(codec: string, keySample: Uint8Array): ArrayBuffer {
  const { profile, level, tier, bitDepth } = parseAv1CodecString(codec);
  const highBitDepth = bitDepth > 8 ? 1 : 0;
  const twelveBit = bitDepth === 12 ? 1 : 0;

  // Chroma layout is implied by the profile for the depths WebCodecs exposes:
  // profile 0 is 4:2:0, profile 1 is 4:4:4, and profile 2 is 4:2:2 at 10 bits
  // (4:4:4 is only reachable at 12 bits).
  let subsamplingX = 1;
  let subsamplingY = 1;
  if (profile === 1) {
    subsamplingX = 0;
    subsamplingY = 0;
  } else if (profile === 2) {
    subsamplingX = twelveBit ? 0 : 1;
    subsamplingY = 0;
  }
  const monochrome = 0;
  const chromaSamplePosition = 0; // CSP_UNKNOWN

  const sequenceHeader = findAv1SequenceHeaderObu(keySample);
  const configObus = sequenceHeader ?? new Uint8Array(0);

  const encoded = new ByteWriter()
    .u8(0x80 | 0x01) // marker = 1, version = 1
    .u8(((profile & 0x7) << 5) | (level & 0x1f))
    .u8(
      ((tier & 0x1) << 7) |
        ((highBitDepth & 0x1) << 6) |
        ((twelveBit & 0x1) << 5) |
        ((monochrome & 0x1) << 4) |
        ((subsamplingX & 0x1) << 3) |
        ((subsamplingY & 0x1) << 2) |
        (chromaSamplePosition & 0x3),
    )
    .u8(0) // reserved + no initial presentation delay
    .bytes(configObus)
    .finish();

  return encoded.buffer.slice(
    encoded.byteOffset,
    encoded.byteOffset + encoded.byteLength,
  ) as ArrayBuffer;
}

/**
 * Muxes encoded video chunks into a progressive MP4.
 *
 * @param options - Chunks, codec string, dimensions, frame rate, and transfer.
 * @returns The finalized file, or `null` when there are no chunks.
 * @throws If the codec is unsupported, the first chunk is not a key frame, or
 *   the encoder never supplied a decoder configuration.
 */
export function muxVideoToMp4(options: MuxVideoOptions): MuxedVideo | null {
  const { chunks, codec, width, height, fps, transfer } = options;
  if (chunks.length === 0) return null;
  if (chunks[0].chunk.type !== 'key') {
    throw new Error('encoder.mp4Muxer: the first encoded chunk must be a key frame');
  }

  const family = codecFamily(codec);

  const MOVIE_TIMESCALE = 1000;
  const MEDIA_TIMESCALE = 90000;
  const fallbackDurationUs = 1_000_000 / fps;

  const sampleBytes: Uint8Array[] = [];
  const samples: SampleRecord[] = [];
  let mdatPayloadSize = 0;
  let elapsedUs = 0;

  for (const { chunk } of chunks) {
    const data = new Uint8Array(chunk.byteLength);
    chunk.copyTo(data);
    const durationUs = chunk.duration ?? fallbackDurationUs;
    samples.push({
      durationTimescale: microsecondsToTimescale(durationUs, MEDIA_TIMESCALE),
      size: data.length,
      isKey: chunk.type === 'key',
    });
    sampleBytes.push(data);
    mdatPayloadSize += data.length;
    elapsedUs += durationUs;
  }

  // WebCodecs only attaches `decoderConfig` when the configuration changes, so
  // the first chunk carrying a description is the authoritative one. Chrome's
  // AV1 encoder supplies no description at all; for AV1 the `av1C` record is
  // reconstructed from the codec string plus the in-band sequence header OBU.
  const configured = chunks.find(
    (entry) => entry.metadata?.decoderConfig?.description !== undefined,
  );
  const description = configured?.metadata?.decoderConfig?.description;
  let descriptionBuffer: ArrayBuffer;
  if (description) {
    descriptionBuffer = toArrayBuffer(description);
  } else if (family === 'av1') {
    descriptionBuffer = synthesizeAv1Config(codec, sampleBytes[0]);
    logDebug('encoder', 'Synthesized av1C (encoder supplied no description)', {
      codec,
      bytes: descriptionBuffer.byteLength,
    });
  } else {
    throw new Error(
      'encoder.mp4Muxer: the encoder supplied no decoder configuration; ' +
        'cannot build the hvcC/avcC sample entry',
    );
  }

  const mediaDuration = samples.reduce((sum, sample) => sum + sample.durationTimescale, 0);
  const movieDuration = microsecondsToTimescale(elapsedUs, MOVIE_TIMESCALE);

  const ftyp = box(
    'ftyp',
    new ByteWriter()
      .ascii('isom')
      .u32(0x200) // minor_version
      .ascii('isom')
      .ascii('iso2')
      .ascii('av01')
      .finish(),
  );
  const mdat = box('mdat', concat(sampleBytes));

  // Sample 0 starts immediately after ftyp and mdat's 8-byte header, because
  // all samples are contiguous inside one chunk.
  const dataOffset = ftyp.length + 8;

  const stbl = container('stbl', [
    box(
      'stsd',
      concat([
        new ByteWriter().u32(0).u32(1).finish(), // version/flags + entry_count
        visualSampleEntry(family, width, height, descriptionBuffer, transfer),
      ]),
    ),
    sttsBox(samples),
    stssBox(samples),
    stscBox(),
    stszBox(samples),
    stcoBox(dataOffset),
  ]);

  const trak = container('trak', [
    tkhdBox(1, movieDuration, width, height),
    container('mdia', [
      mdhdBox(MEDIA_TIMESCALE, mediaDuration),
      hdlrBox(),
      container('minf', [vmhdBox(), dinfBox(), stbl]),
    ]),
  ]);

  const moov = container('moov', [mvhdBox(MOVIE_TIMESCALE, movieDuration), trak]);

  const file = concat([ftyp, mdat, moov]);
  return {
    buffer: file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) as ArrayBuffer,
    mimeType: 'video/mp4',
    extension: 'mp4',
    codecFamily: family,
    frameCount: samples.length,
    durationSeconds: elapsedUs / 1_000_000,
  };
}
