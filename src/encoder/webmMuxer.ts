/**
 * Minimal in-memory WebM (Matroska/EBML) muxer for WebCodecs `EncodedVideoChunk`s.
 *
 * Responsibility: wrap an encoded **VP8/VP9** packet stream in a WebM file so it
 * can be downloaded and played. No dependencies, no encoding logic.
 *
 * Why this exists: Chrome currently exposes no AV1 or HEVC encoder on some
 * platforms, but does expose VP9 Profile 2 (10-bit). VP9 belongs in WebM, not
 * MP4, so this module complements `mp4Muxer.ts` rather than replacing it.
 *
 * Scope and deliberate limitations:
 * - **One video track, no audio.**
 * - **Live/streaming-style layout:** a `Segment` of unknown size whose `Cluster`s
 *   are still placed at known offsets once built, so `SeekHead`/`Cues` remain
 *   optional. A `Cues` index is written so seeking works in players that use it.
 * - **EBML is little-endian for IDs, big-endian for values.** IDs are written as
 *   their canonical byte sequences; all numeric values are big-endian.
 * - **Colour metadata is authored explicitly.** Matroska has first-class
 *   `Colour`/`MasteringMetadata` elements, so Rec. 2100 / PQ or HLG is written
 *   there rather than inferred from the bitstream.
 * - **Block timestamps are signed 16-bit milliseconds relative to a cluster.**
 *   A new cluster is started whenever that would overflow, which also bounds
 *   cluster duration for seeking.
 */

import { computeChunkTiming } from './chunkTiming.ts';

/** One encoded chunk plus the metadata that accompanied it. */
export interface WebMChunk {
  readonly chunk: EncodedVideoChunk;
  readonly metadata?: EncodedVideoChunkMetadata;
}

/** Options for {@link muxVideoToWebM}. */
export interface WebMOptions {
  readonly chunks: readonly WebMChunk[];
  /** WebCodecs codec string, e.g. `vp09.02.31.10`. */
  readonly codec: string;
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  /** Rec. 2100 transfer function used to author the frames. */
  readonly transfer: 'pq' | 'hlg';
}

/** A finalized WebM file ready for download. */
export interface MuxedWebM {
  readonly buffer: ArrayBuffer;
  readonly mimeType: string;
  readonly extension: string;
  readonly codecFamily: 'vp9';
  readonly frameCount: number;
  readonly durationSeconds: number;
}

/** Maximum signed 16-bit block timestamp (milliseconds) within one cluster. */
const MAX_BLOCK_TIMESTAMP_MS = 32767;
/** Target cluster duration; keeps seeking granular and timestamps in range. */
const TARGET_CLUSTER_MS = 5000;

/**
 * Encodes an EBML element ID as its canonical byte sequence.
 *
 * IDs are defined by their spec width (1-4 bytes); the numeric constant already
 * includes the marker bit, so the byte length is derived from its magnitude.
 */
function elementId(id: number): Uint8Array {
  if (id <= 0xff) return Uint8Array.of(id);
  if (id <= 0xffff) return Uint8Array.of(id >>> 8, id & 0xff);
  if (id <= 0xffffff) return Uint8Array.of(id >>> 16, (id >>> 8) & 0xff, id & 0xff);
  return Uint8Array.of(
    (id >>> 24) & 0xff,
    (id >>> 16) & 0xff,
    (id >>> 8) & 0xff,
    id & 0xff,
  );
}

/**
 * Encodes an EBML variable-length size field.
 *
 * Widths are chosen from the value so small sizes stay compact; the 8-byte form
 * is used for values that do not fit in 7 bytes of a shorter width.
 */
function elementSize(size: number): Uint8Array {
  if (size < 0x7f) {
    return Uint8Array.of(0x80 | size);
  }
  if (size < 0x3fff) {
    return Uint8Array.of(0x40 | (size >>> 8), size & 0xff);
  }
  if (size < 0x1fffff) {
    return Uint8Array.of(0x20 | (size >>> 16), (size >>> 8) & 0xff, size & 0xff);
  }
  if (size < 0x0fffffff) {
    return Uint8Array.of(
      0x10 | (size >>> 24),
      (size >>> 16) & 0xff,
      (size >>> 8) & 0xff,
      size & 0xff,
    );
  }
  throw new Error('encoder.webmMuxer: element too large for a 32-bit size field');
}

/** The EBML "unknown size" marker (all value bits set in a 1-byte field). */
const UNKNOWN_SIZE = Uint8Array.of(0xff);

/** Builds `id + size + payload` for a binary/container element. */
function element(id: number, payload: Uint8Array, unknownSize = false): Uint8Array {
  const size = unknownSize ? UNKNOWN_SIZE : elementSize(payload.length);
  const out = new Uint8Array(elementId(id).length + size.length + payload.length);
  let offset = 0;
  out.set(elementId(id), offset);
  offset += elementId(id).length;
  out.set(size, offset);
  offset += size.length;
  out.set(payload, offset);
  return out;
}

/** Builds an unsigned integer element with the minimum number of bytes. */
function uintElement(id: number, value: number): Uint8Array {
  if (value === 0) return element(id, Uint8Array.of(0));
  const bytes: number[] = [];
  let remaining = value;
  while (remaining > 0) {
    bytes.unshift(remaining % 256);
    remaining = Math.floor(remaining / 256);
  }
  return element(id, Uint8Array.from(bytes));
}

/** Builds a float64 element (used for `Duration`). */
function floatElement(id: number, value: number): Uint8Array {
  const payload = new Uint8Array(8);
  new DataView(payload.buffer).setFloat64(0, value, false);
  return element(id, payload);
}

/** Builds an ASCII string element (used for `MuxingApp`/`WritingApp`). */
function stringElement(id: number, value: string): Uint8Array {
  const bytes = new Uint8Array(value.length);
  for (let i = 0; i < value.length; i++) bytes[i] = value.charCodeAt(i);
  return element(id, bytes);
}

// --- EBML element IDs (canonical encodings, marker bits included) -------------
const ID_EBML = 0x1a45dfa3;
const ID_EBML_VERSION = 0x4286;
const ID_EBML_READ_VERSION = 0x42f7;
const ID_EBML_MAX_ID_LENGTH = 0x42f2;
const ID_EBML_MAX_SIZE_LENGTH = 0x42f3;
const ID_DOCTYPE = 0x4282;
const ID_DOCTYPE_VERSION = 0x4287;
const ID_DOCTYPE_READ_VERSION = 0x4285;
const ID_SEGMENT = 0x18538067;
const ID_INFO = 0x1549a966;
const ID_TIMESTAMP_SCALE = 0x2ad7b1;
const ID_DURATION = 0x4489;
const ID_MUXING_APP = 0x4d80;
const ID_WRITING_APP = 0x5741;
const ID_TRACKS = 0x1654ae6b;
const ID_TRACK_ENTRY = 0xae;
const ID_TRACK_NUMBER = 0xd7;
const ID_TRACK_UID = 0x73c5;
const ID_TRACK_TYPE = 0x83;
const ID_FLAG_LACING = 0x9c;
const ID_DEFAULT_DURATION = 0x23e383;
const ID_CODEC_ID = 0x86;
const ID_VIDEO = 0xe0;
const ID_PIXEL_WIDTH = 0xb0;
const ID_PIXEL_HEIGHT = 0xba;
const ID_COLOUR = 0x55b0;
const ID_MATRIX_COEFFICIENTS = 0x55b1;
const ID_BITS_PER_CHANNEL = 0x55b2;
const ID_CHROMA_SUBSAMPLING_HORZ = 0x55b3;
const ID_CHROMA_SUBSAMPLING_VERT = 0x55b4;
const ID_RANGE = 0x55b9;
const ID_TRANSFER_CHARACTERISTICS = 0x55ba;
const ID_PRIMARIES = 0x55bb;
const ID_CLUSTER = 0x1f43b675;
const ID_TIMESTAMP = 0xe7;
const ID_SIMPLE_BLOCK = 0xa3;
const ID_CUES = 0x1c53bb6b;
const ID_CUE_POINT = 0xbb;
const ID_CUE_TIME = 0xb3;
const ID_CUE_TRACK_POSITIONS = 0xb7;
const ID_CUE_TRACK = 0xf7;
const ID_CUE_CLUSTER_POSITION = 0xf1;
const ID_SEEK_HEAD = 0x114d9b74;
const ID_SEEK = 0x4dbb;
const ID_SEEK_ID = 0x53ab;
const ID_SEEK_POSITION = 0x53ac;

/** Matroska `TrackType` value for video. */
const TRACK_TYPE_VIDEO = 1;
/** Timestamp scale: 1 ms per tick, in nanoseconds. */
const TIMESTAMP_SCALE_NS = 1_000_000;

/**
 * Maps a VP9 codec string to its 10-bit-ness.
 *
 * `vp09.<profile>.<level>.<bitDepth>`; profile 2 is the 10-bit profile.
 */
function parseVp9CodecString(codec: string): { profile: number; bitDepth: number } {
  const match = /^vp09\.(\d{2})\.(\d{2})\.(\d{2})(?:\.(\d{2}))*$/.exec(codec);
  if (!match) {
    throw new Error(`encoder.webmMuxer: cannot parse VP9 codec string "${codec}"`);
  }
  return { profile: Number(match[1]), bitDepth: Number(match[3]) };
}

/** Matroska `Colour` element describing Rec. 2100 / PQ-or-HLG / limited range. */
function colourElement(transfer: 'pq' | 'hlg', bitDepth: number): Uint8Array {
  const children = [
    // 9 = BT.2020 non-constant luminance.
    uintElement(ID_MATRIX_COEFFICIENTS, 9),
    uintElement(ID_BITS_PER_CHANNEL, bitDepth),
    // 4:2:0
    uintElement(ID_CHROMA_SUBSAMPLING_HORZ, 1),
    uintElement(ID_CHROMA_SUBSAMPLING_VERT, 1),
    // 0 = limited (studio) range.
    uintElement(ID_RANGE, 0),
    // 16 = PQ, 18 = HLG.
    uintElement(ID_TRANSFER_CHARACTERISTICS, transfer === 'pq' ? 16 : 18),
    // 9 = BT.2020 primaries.
    uintElement(ID_PRIMARIES, 9),
  ];
  // Content light level is unknown for this synthetic scene; stating the peak in
  // MaxCLL/MaxFALL terms would require per-frame measurement.
  const payload = concatBytes(children);
  return element(ID_COLOUR, payload);
}

/** Concatenates byte arrays. */
function concatBytes(parts: readonly Uint8Array[]): Uint8Array {
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

/** Wraps an encoded frame in a `SimpleBlock` (track 1, keyframe flag honoured). */
function simpleBlock(
  relativeTimestampMs: number,
  isKey: boolean,
  data: Uint8Array,
): Uint8Array {
  const payload = new Uint8Array(4 + data.length);
  // Track number as a 1-byte EBML vint (track 1 => 0x81).
  payload[0] = 0x81;
  const view = new DataView(payload.buffer);
  view.setInt16(1, relativeTimestampMs, false);
  // Bit 7 = keyframe for SimpleBlock.
  payload[3] = isKey ? 0x80 : 0x00;
  payload.set(data, 4);
  return element(ID_SIMPLE_BLOCK, payload);
}

/**
 * Muxes encoded VP9 chunks into a WebM file.
 *
 * @param options - Chunks, codec string, dimensions, rate, and transfer.
 * @returns The finalized file, or `null` when there are no chunks.
 * @throws If the codec string is not VP9 or the first chunk is not a key frame.
 */
export function muxVideoToWebM(options: WebMOptions): MuxedWebM | null {
  const { chunks, codec, width, height, fps, transfer } = options;
  if (chunks.length === 0) return null;
  if (chunks[0].chunk.type !== 'key') {
    throw new Error('encoder.webmMuxer: the first encoded chunk must be a key frame');
  }

  const { bitDepth } = parseVp9CodecString(codec);

  // Derive timing from presentation timestamps rather than trusting (or summing)
  // the optional `duration` field; see `chunkTiming.ts` for why.
  const timing = computeChunkTiming(chunks, fps);

  // --- header ---------------------------------------------------------------
  const ebml = element(
    ID_EBML,
    concatBytes([
      uintElement(ID_EBML_VERSION, 1),
      uintElement(ID_EBML_READ_VERSION, 1),
      uintElement(ID_EBML_MAX_ID_LENGTH, 4),
      uintElement(ID_EBML_MAX_SIZE_LENGTH, 8),
      stringElement(ID_DOCTYPE, 'webm'),
      uintElement(ID_DOCTYPE_VERSION, 2),
      uintElement(ID_DOCTYPE_READ_VERSION, 2),
    ]),
  );

  // --- Info -----------------------------------------------------------------
  const samples: Array<{
    data: Uint8Array;
    timestampMs: number;
    isKey: boolean;
  }> = [];

  // Prefer presentation timestamps when they are ordered; otherwise accumulate
  // durations so block times stay monotonic for the player.
  let cumulativeUs = 0;
  for (let index = 0; index < chunks.length; index++) {
    const { chunk } = chunks[index];
    const data = new Uint8Array(chunk.byteLength);
    chunk.copyTo(data);
    const startUs = timing.monotonic ? timing.startsUs[index] : cumulativeUs;
    samples.push({
      data,
      timestampMs: Math.round(startUs / 1000),
      isKey: chunk.type === 'key',
    });
    cumulativeUs += timing.durationsUs[index];
  }
  const durationMs = timing.totalUs / 1000;

  const info = element(
    ID_INFO,
    concatBytes([
      uintElement(ID_TIMESTAMP_SCALE, TIMESTAMP_SCALE_NS),
      floatElement(ID_DURATION, durationMs),
      stringElement(ID_MUXING_APP, 'hdr-frame-poc'),
      stringElement(ID_WRITING_APP, 'hdr-frame-poc'),
    ]),
  );

  // --- Tracks ---------------------------------------------------------------
  const video = element(
    ID_VIDEO,
    concatBytes([
      uintElement(ID_PIXEL_WIDTH, width),
      uintElement(ID_PIXEL_HEIGHT, height),
      colourElement(transfer, bitDepth),
    ]),
  );

  const trackEntry = element(
    ID_TRACK_ENTRY,
    concatBytes([
      uintElement(ID_TRACK_NUMBER, 1),
      uintElement(ID_TRACK_UID, 1),
      uintElement(ID_TRACK_TYPE, TRACK_TYPE_VIDEO),
      uintElement(ID_FLAG_LACING, 0),
      uintElement(ID_DEFAULT_DURATION, Math.round(1e9 / fps)),
      stringElement(ID_CODEC_ID, 'V_VP9'),
      video,
    ]),
  );
  const tracks = element(ID_TRACKS, trackEntry);

  // --- Clusters -------------------------------------------------------------
  // Clusters are built first so their byte offsets are known for the cue index.
  const clusters: Uint8Array[] = [];
  /** Pending cue points, resolved to byte offsets once all clusters are built. */
  const pendingCues: Array<{ timeMs: number; clusterIndex: number }> = [];
  /** Cue points with their byte offset relative to the first cluster filled in. */
  const cuePoints: Array<{ timeMs: number; clusterOffset: number }> = [];
  /** Byte offset of each cluster relative to the first cluster. */
  const clusterOffsets: number[] = [];

  let currentBlocks: Uint8Array[] = [];
  let clusterStartMs = 0;

  const flushCluster = (): void => {
    if (currentBlocks.length === 0) return;
    const body = concatBytes([
      uintElement(ID_TIMESTAMP, clusterStartMs),
      ...currentBlocks,
    ]);
    const built = element(ID_CLUSTER, body);
    const previousEnd =
      clusters.length === 0 ? 0 : clusterOffsets[clusters.length - 1] + clusters[clusters.length - 1].length;
    clusterOffsets.push(previousEnd);
    clusters.push(built);
    currentBlocks = [];
  };

  for (const sample of samples) {
    const needsNewCluster =
      currentBlocks.length === 0 ||
      sample.timestampMs - clusterStartMs > MAX_BLOCK_TIMESTAMP_MS ||
      sample.timestampMs - clusterStartMs >= TARGET_CLUSTER_MS;
    if (needsNewCluster) {
      flushCluster();
      clusterStartMs = sample.timestampMs;
      if (sample.isKey) {
        pendingCues.push({ timeMs: sample.timestampMs, clusterIndex: clusters.length });
      }
    }
    currentBlocks.push(
      simpleBlock(sample.timestampMs - clusterStartMs, sample.isKey, sample.data),
    );
  }
  flushCluster();

  for (const pending of pendingCues) {
    cuePoints.push({
      timeMs: pending.timeMs,
      clusterOffset: clusterOffsets[pending.clusterIndex] ?? 0,
    });
  }

  // --- Cues -----------------------------------------------------------------
  // Offsets inside a Segment are relative to the first byte of its payload, so
  // the layout must be settled before any offset can be written. Order is:
  // SeekHead, Info, Tracks, Cues, Clusters. Cues' own length depends on the
  // magnitude of the offsets it stores, which depend on its own length, so it is
  // rebuilt until the length is stable (positions grow monotonically, so this
  // converges in a couple of passes).
  const infoLength = info.length;
  const tracksLength = tracks.length;

  // SeekHead is rebuilt until its length is stable for the same reason.
  let seekHead = buildSeekHead(0, 0);
  for (let pass = 0; pass < 8; pass++) {
    const infoPos = seekHead.length;
    const tracksPos = infoPos + infoLength;
    const next = buildSeekHead(infoPos, tracksPos);
    const stable = next.length === seekHead.length;
    seekHead = next;
    if (stable) break;
  }

  let cues = buildCues(cuePoints, 0);
  // Cues size depends on the magnitude of the offsets it stores, and the
  // offsets depend on the size of Cues, so iterate until the length settles.
  for (let pass = 0; pass < 8; pass++) {
    const clustersBaseOffset = seekHead.length + infoLength + tracksLength + cues.length;
    const next = buildCues(cuePoints, clustersBaseOffset);
    const stable = next.length === cues.length;
    cues = next;
    if (stable) break;
  }

  const segmentPayload = concatBytes([seekHead, info, tracks, cues, ...clusters]);
  const segment = element(ID_SEGMENT, segmentPayload, true);

  const file = concatBytes([ebml, segment]);
  return {
    buffer: file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength) as ArrayBuffer,
    mimeType: 'video/webm',
    extension: 'webm',
    codecFamily: 'vp9',
    frameCount: samples.length,
    durationSeconds: timing.totalUs / 1_000_000,
  };
}

/**
 * Builds a `SeekHead` pointing at `Info` and `Tracks`.
 *
 * Offsets are relative to the start of the Segment payload, so the caller must
 * pass the positions that follow the final (stable-length) SeekHead.
 */
function buildSeekHead(infoPos: number, tracksPos: number): Uint8Array {
  const seek = (idBytes: Uint8Array, position: number): Uint8Array =>
    element(
      ID_SEEK,
      concatBytes([
        element(ID_SEEK_ID, idBytes),
        uintElement(ID_SEEK_POSITION, position),
      ]),
    );
  return element(
    ID_SEEK_HEAD,
    concatBytes([
      seek(elementId(ID_INFO), infoPos),
      seek(elementId(ID_TRACKS), tracksPos),
    ]),
  );
}

/** Builds a `Cues` index from keyframe positions.
 *
 * Cluster positions are measured from the start of the Segment payload, so the
 * caller supplies the offset at which the first Cluster begins.
 */
function buildCues(
  cuePoints: ReadonlyArray<{ timeMs: number; clusterOffset: number }>,
  clustersBaseOffset: number,
): Uint8Array {
  // Cue positions are resolved against the cluster boundaries recorded during
  // muxing; those are recomputed from the shared base offset each pass.
  const points = cuePoints.map((point) =>
    element(
      ID_CUE_POINT,
      concatBytes([
        uintElement(ID_CUE_TIME, point.timeMs),
        element(
          ID_CUE_TRACK_POSITIONS,
          concatBytes([
            uintElement(ID_CUE_TRACK, 1),
            uintElement(
              ID_CUE_CLUSTER_POSITION,
              clustersBaseOffset + point.clusterOffset,
            ),
          ]),
        ),
      ]),
    ),
  );
  return element(ID_CUES, concatBytes(points));
}
