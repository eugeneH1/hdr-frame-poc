/**
 * Fair-comparison rendering for exported HDR video.
 *
 * Responsibility: decode a frame from the muxed file and present it the *same
 * way* the live preview presents the source, so the two can actually be
 * compared. This module contains no muxing, encoding, or scene logic.
 *
 * Why this exists: the preview is an **SDR estimate** (ACES tone curve + sRGB
 * encoding, see `shaders/sdrPreview.wgsl`), while the export is **PQ HDR**. Put
 * side by side unmodified, they look completely different even when the content
 * is correct — a decoded PQ frame shown without a tone curve is far too dark and
 * desaturated on an SDR display. That asymmetry makes it impossible to tell a
 * real content bug from an expected transfer-function difference.
 *
 * The comparison path therefore:
 *   1. decodes the exported file with `VideoDecoder` into a `VideoFrame`,
 *   2. draws it to a 2D canvas to obtain RGB pixels,
 *   3. inverts the PQ transfer to recover normalised display light,
 *   4. applies the **same** Narkowicz ACES curve and sRGB encoding as the preview.
 *
 * What this does not do: Rec. 2020 -> Rec. 709 gamut conversion. The preview
 * does not perform it either, so leaving it out keeps the comparison fair rather
 * than making the export match a colour-managed reference.
 */

/** One decoded, preview-matched RGBA image. */
export interface ComparisonImage {
  readonly width: number;
  readonly height: number;
  /** RGBA8 pixels, already tone-mapped and sRGB-encoded. Backed by an ArrayBuffer. */
  readonly pixels: Uint8ClampedArray<ArrayBuffer>;
}

/**
 * Splits a WebCodecs codec string into the pieces a `VideoDecoder` cares about.
 *
 * The demuxing-free path below relies on the browser's own media stack, so only
 * the container MIME type and codec are needed here.
 */
function containerMimeType(extension: string, codec: string): string {
  if (extension === 'webm') return `video/webm; codecs="${codec}"`;
  return `video/mp4; codecs="${codec}"`;
}

/** SMPTE ST 2084 (PQ) inverse EOTF: normalised signal -> linear display light. */
function pqToLinear(signal: number): number {
  const m1 = 2610 / 16384;
  const m2 = (2523 / 4096) * 128;
  const c1 = 3424 / 4096;
  const c2 = (2413 / 4096) * 32;
  const c3 = (2392 / 4096) * 32;
  const v = Math.pow(signal, 1 / m2);
  const numerator = Math.max(v - c1, 0);
  const denominator = c2 - c3 * v;
  if (denominator <= 0) return 0;
  return Math.pow(numerator / denominator, 1 / m1);
}

/** Narkowicz 2015 ACES approximation, identical to `sdrPreview.wgsl`. */
function acesFilmic(linear: number): number {
  const a = 2.51;
  const b = 0.03;
  const c = 2.43;
  const d = 0.59;
  const e = 0.14;
  const mapped = (linear * (a * linear + b)) / (linear * (c * linear + d) + e);
  return mapped < 0 ? 0 : mapped > 1 ? 1 : mapped;
}

/** IEC 61966-2-1 sRGB encoding, identical to `sdrPreview.wgsl`. */
function linearToSrgb(linear: number): number {
  const safe = linear < 0 ? 0 : linear;
  return safe <= 0.0031308 ? safe * 12.92 : 1.055 * Math.pow(safe, 1 / 2.4) - 0.055;
}

/**
 * Reference white used by the encoder packer: scene-linear 1.0 is 100 nits and
 * PQ normalises against a 10 000-nit peak (`webCodecsTap.ts`).
 */
const PQ_LUMA_SCALE = 100 / 10000;

/**
 * Converts decoded PQ-encoded RGB back to display light and applies the preview
 * curve.
 *
 * The decoder hands back the encoded signal as if it were 8-bit display values,
 * so the byte is first normalised, then inverse-PQ'd to absolute luminance,
 * re-scaled into the packer's scene-linear units, and finally tone-mapped.
 */
function previewMatchedPixel(encoded: number): number {
  const signal = encoded / 255;
  const displayLinear = pqToLinear(signal);
  // Undo the 100-nit reference so ACES sees the same scene-linear value the
  // preview shader saw before its tone curve.
  const sceneLinear = displayLinear / PQ_LUMA_SCALE;
  return Math.round(linearToSrgb(acesFilmic(sceneLinear)) * 255);
}

/**
 * Decodes one frame of an exported file and renders it with the preview's curve.
 *
 * Uses the browser's media stack (`<video>` + `drawImage`) rather than a
 * hand-written demuxer: it exercises the same path a user's player would, so a
 * container or codec problem shows up here too.
 *
 * @param file - The exported blob (MP4 or WebM).
 * @param frameIndex - Zero-based frame to sample.
 * @param frameCount - Total frames, used to seek to the right timestamp.
 * @param fps - Frame rate, used to convert the index to a time.
 * @returns The tone-mapped RGBA image.
 * @throws If the browser cannot decode the file or the frame is unavailable.
 */
export async function renderComparisonFrame(
  file: Blob,
  frameIndex: number,
  frameCount: number,
  fps: number,
): Promise<ComparisonImage> {
  const url = URL.createObjectURL(file);
  const video = document.createElement('video');
  video.muted = true;
  video.preload = 'auto';

  try {
    video.src = url;
    await new Promise<void>((resolve, reject) => {
      const onError = (): void => {
        reject(
          new Error(
            `videoContainer: the browser could not decode the export (${video.error?.message ?? 'unknown media error'})`,
          ),
        );
      };
      video.addEventListener('loadeddata', () => resolve(), { once: true });
      video.addEventListener('error', onError, { once: true });
    });

    // Seek slightly inside the frame to avoid landing on a boundary.
    const targetSeconds = Math.min(
      (frameIndex + 0.5) / fps,
      Math.max((frameCount - 1) / fps, 0),
    );
    if (Math.abs(video.currentTime - targetSeconds) > 1e-3) {
      await new Promise<void>((resolve, reject) => {
        video.addEventListener('seeked', () => resolve(), { once: true });
        video.addEventListener(
          'error',
          () => reject(new Error('videoContainer: seek failed')),
          { once: true },
        );
        video.currentTime = targetSeconds;
      });
    }

    const width = video.videoWidth;
    const height = video.videoHeight;
    if (width === 0 || height === 0) {
      throw new Error('videoContainer: decoded frame has zero size');
    }

    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('videoContainer: 2D context unavailable');
    context.drawImage(video, 0, 0, width, height);

    const image = context.getImageData(0, 0, width, height);
    const pixels = image.data;
    for (let i = 0; i < pixels.length; i += 4) {
      pixels[i] = previewMatchedPixel(pixels[i]);
      pixels[i + 1] = previewMatchedPixel(pixels[i + 1]);
      pixels[i + 2] = previewMatchedPixel(pixels[i + 2]);
      pixels[i + 3] = 255;
    }

    return { width, height, pixels };
  } finally {
    video.removeAttribute('src');
    video.load();
    URL.revokeObjectURL(url);
  }
}

/** Describes a container for logging. */
export function describeContainer(extension: string, codec: string): string {
  return containerMimeType(extension, codec);
}
