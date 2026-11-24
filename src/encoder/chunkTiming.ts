/**
 * Shared chunk timing for the container muxers.
 *
 * Responsibility: turn a list of `EncodedVideoChunk`s into presentation starts,
 * per-sample durations, and a total duration, without trusting WebCodecs'
 * optional `duration` field blindly.
 *
 * Why this exists: both muxers previously did
 *
 *     const durationUs = chunk.duration ?? (1_000_000 / fps);
 *
 * which has two independent flaws:
 *
 * 1. **`0` is not nullish.** Encoders commonly report `duration: null`, but some
 *    report `0` for the final chunk. `??` accepts `0`, so the last frame
 *    contributed no time and the file ended one frame early (~9.967 s).
 * 2. **Summing integer microsecond durations loses the frame grid.** 300 frames
 *    at 33,333 µs is 9,999,900 µs = 9.9999 s, which players floor to
 *    "9 seconds". Deriving durations from presentation timestamps makes the
 *    total land on the real grid (9,966,667 + 33,333 = 10,000,000 µs).
 *
 * Timestamps are preferred when present and monotonic. Encode order is not
 * guaranteed to equal presentation order, so a non-monotonic stream falls back
 * to reported durations and then to the nominal frame rate.
 */

/** Timing derived from an encoded chunk sequence. */
export interface ChunkTiming {
  /** Presentation start of each sample, relative to the first sample, in µs. */
  readonly startsUs: readonly number[];
  /** Duration of each sample in µs; always positive. */
  readonly durationsUs: readonly number[];
  /** Presentation end of the whole sequence in µs (first start = 0). */
  readonly totalUs: number;
  /** Whether `startsUs` is non-decreasing, i.e. presentation order is reliable. */
  readonly monotonic: boolean;
}

/** The subset of `EncodedVideoChunk` that timing depends on. */
export interface TimingChunk {
  readonly chunk: Pick<EncodedVideoChunk, 'timestamp' | 'duration'>;
}

/**
 * Computes presentation timing for a chunk sequence.
 *
 * @param chunks - Encoded chunks in emission order.
 * @param fps - Nominal frame rate, used only when no better duration exists.
 * @returns Starts, per-sample durations, total duration, and monotonicity.
 */
export function computeChunkTiming(
  chunks: readonly TimingChunk[],
  fps: number,
): ChunkTiming {
  const fallbackUs =
    Number.isFinite(fps) && fps > 0 ? 1_000_000 / fps : 1_000_000 / 30;
  if (chunks.length === 0) {
    return { startsUs: [], durationsUs: [], totalUs: 0, monotonic: true };
  }

  const baseUs = chunks[0].chunk.timestamp;
  const startsUs = chunks.map((entry) => entry.chunk.timestamp - baseUs);

  let monotonic = true;
  for (let i = 1; i < startsUs.length; i++) {
    if (startsUs[i] < startsUs[i - 1]) {
      monotonic = false;
      break;
    }
  }

  const durationsUs = chunks.map((entry, index) => {
    const reported = entry.chunk.duration;
    const hasReported =
      reported !== null &&
      reported !== undefined &&
      Number.isFinite(reported) &&
      reported > 0;

    // When presentation order is trustworthy, the timestamp grid is the
    // authority: it is the only source that yields an exact total (see the
    // module comment). The final frame has no following timestamp, so it
    // inherits the previous interval, which matches the pipeline's constant
    // cadence. A reported duration that disagrees with the grid (including a
    // bogus 30 ms) is therefore ignored rather than truncating the file.
    if (monotonic && startsUs.length > 1) {
      if (index + 1 < startsUs.length) {
        const delta = startsUs[index + 1] - startsUs[index];
        if (delta > 0) return delta;
      } else {
        const lastDelta = startsUs[index] - startsUs[index - 1];
        if (lastDelta > 0) return lastDelta;
      }
    }
    if (hasReported) return reported as number;
    if (index + 1 < startsUs.length) {
      const delta = startsUs[index + 1] - startsUs[index];
      if (delta > 0) return delta;
    }
    return fallbackUs;
  });

  // Total is the furthest presentation end, not the sum: with monotonic
  // timestamps the last start already encodes all prior gaps.
  let totalUs = 0;
  for (let i = 0; i < startsUs.length; i++) {
    totalUs = Math.max(totalUs, startsUs[i] + durationsUs[i]);
  }
  if (!(totalUs > 0)) {
    totalUs = durationsUs.reduce((sum, value) => sum + value, 0);
  }

  return { startsUs, durationsUs, totalUs, monotonic };
}
