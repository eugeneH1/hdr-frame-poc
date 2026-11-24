/**
 * Workbench DOM setup (plain TypeScript + HTML, no framework).
 *
 * Responsibility: build the POC's minimal UI — a preview `<canvas>`, Start /
 * Cancel controls, an HDR-target format selector, and a status/telemetry
 * readout — and wire it to `runFramePipeline`. It deliberately contains no
 * rendering logic: all GPU state stays in `src/gpu/` and `src/pipeline/`, while
 * this module owns DOM state and event listeners.
 *
 * Design assumptions:
 * - The canvas backing store is 1280x720 (matching the design space); CSS keeps
 *   it aspect-aware and scales it down on narrow screens.
 * - Cancel maps to an `AbortController` passed to the pipeline as an
 *   `AbortSignal`; the workbench never touches GPU objects.
 * - Telemetry arrives already throttled by the pipeline; `updateTelemetry`
 *   merely formats it into stable DOM nodes.
 * - The workbench registers `onFrame` to count streamed Float32 frames and keep
 *   one user-selected frame (copied before the pipeline releases it) for `.npy`
 *   download; it never retains all 300 frames.
 */

import type {
  ColorSpaceSpec,
  EncoderTelemetry,
  FloatFrame,
  FrameMeta,
  RenderConfig,
  SequencePeak,
  TelemetrySnapshot,
} from '../types/frame.ts';
import { getWorkingColorSpace } from '../timeline/sceneAnimator.ts';
import { runFramePipeline, scanFrameExact } from '../pipeline/framePipeline.ts';
import type { ExactFrameScan } from '../pipeline/framePipeline.ts';
import { createVideoEncoderTap, flushAndClose } from '../encoder/webCodecsTap.ts';
import type { VideoEncoderTap } from '../encoder/webCodecsTap.ts';
import { canMuxVideo, muxVideo } from '../encoder/videoContainer.ts';
import type { MuxedVideoFile, MuxVideoChunk } from '../encoder/videoContainer.ts';
import { renderComparisonFrame, comparisonTimestampSeconds } from '../encoder/videoComparison.ts';
import { computeChunkTiming } from '../encoder/chunkTiming.ts';
import { describeLogging, logDebug, logError, logInfo, logWarn } from '../diagnostics/logger.ts';

const DEFAULT_WIDTH = 1280;
const DEFAULT_HEIGHT = 720;
const DEFAULT_FPS = 30;
const DEFAULT_FRAME_COUNT = 300;

/**
 * Scene-linear luminance of reference white, matching `REFERENCE_WHITE_NITS` in
 * the pipeline and encoder. Used only to annotate linear values with nits.
 */
const SCENE_WHITE_NITS = 100;

/**
 * Codecs the export pass is allowed to select, in preference order.
 *
 * Constrained deliberately: encoding with a codec no muxer can carry wastes a
 * full encode pass and then fails at the container step. HEVC and AV1 go into
 * MP4; VP9 goes into WebM. H.264 is excluded because the export is HDR and
 * 8-bit AVC cannot carry PQ.
 */
const EXPORT_CODEC_CANDIDATES = [
  'hvc1.2.4.L93.B0',
  'av01.0.08M.10',
  'vp09.02.31.10',
] as const;

/** Fixed run parameters plus the user-selectable HDR target format. */
export interface WorkbenchRunOptions {
  readonly width: number;
  readonly height: number;
  readonly fps: number;
  readonly frameCount: number;
  /** Requested offscreen HDR format; the GPU layer may fall back if unsupported. */
  readonly renderFormat: GPUTextureFormat;
}

/** Options accepted when mounting the workbench. */
export interface WorkbenchOptions {
  /** Element to render the workbench into (typically `#app`). */
  readonly root: HTMLElement;
  /**
   * Builds the pristine `RenderConfig` for a run from the current UI selection.
   * Called once per Start press. Defaults to the POC's 1280x720/30fps/300 config.
   */
  readonly createConfig?: (selection: WorkbenchRunOptions) => RenderConfig;
}

/** Handles onto the workbench's DOM and run controls. */
export interface Workbench {
  /** The preview canvas the pipeline renders into. */
  readonly canvas: HTMLCanvasElement;
  /** Element used to display status text and telemetry. */
  readonly statusEl: HTMLElement;
  /** Triggers a new render run; no-op if one is already active. */
  run(): Promise<void>;
  /** Cancels the active run, if any. */
  cancel(): void;
  /** Renders a telemetry snapshot into the status display. */
  updateTelemetry(snapshot: TelemetrySnapshot): void;
  /** Removes event listeners and clears the workbench DOM. */
  unmount(): void;
}

type WorkbenchState = 'idle' | 'running' | 'cancelling' | 'cancelled' | 'done' | 'error';

/** Which stat group a cell belongs to; drives emphasis and ordering. */
type StatTier = 'hdr' | 'diag';

interface StatCell {
  readonly cell: HTMLDivElement;
  readonly value: HTMLElement;
}

/** A conformance badge in the header strip (role="status" so it is announced). */
interface MetricBadge {
  readonly element: HTMLSpanElement;
  state: 'unknown' | 'pass' | 'warn' | 'fail';
}

/** One encoded chunk plus the metadata WebCodecs emitted alongside it. */
interface CollectedVideoChunk extends MuxVideoChunk {}

/** One completed CPU frame copied out of the streaming pipeline for download. */
interface RetainedFrame {
  readonly frameIndex: number;
  readonly width: number;
  readonly height: number;
  /** Copied out of the streaming pipeline, so this owns a regular ArrayBuffer. */
  readonly pixels: Float32Array<ArrayBuffer>;
  /** Exact rescan of `pixels`, computed lazily and cached per retained frame. */
  exactScan?: ExactFrameScan;
}

function createElement<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  className?: string,
  text?: string,
): HTMLElementTagNameMap[K] {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function createOption(value: string, label: string, disabled = false): HTMLOptionElement {
  const option = document.createElement('option');
  option.value = value;
  option.textContent = label;
  option.disabled = disabled;
  return option;
}

function createStat(label: string, tier: StatTier = 'diag', title?: string): StatCell {
  const cell = createElement('div', `wb-stat wb-stat--${tier}`);
  const term = createElement('dt', 'wb-stat-label', label);
  const value = createElement('dd', 'wb-stat-value', '—');
  if (title) term.title = title;
  cell.append(term, value);
  return { cell, value };
}

/** Creates a header badge whose dot colour encodes pass/warn/fail/unknown. */
function createBadge(label: string, title: string): MetricBadge {
  const element = createElement('span', 'wb-metric');
  element.dataset.state = 'unknown';
  element.title = title;
  element.append(
    createElement('span', 'wb-metric-label', label),
    createElement('span', 'wb-metric-value', '—'),
  );
  return { element, state: 'unknown' };
}

function setBadge(badge: MetricBadge, state: MetricBadge['state'], text: string): void {
  badge.state = state;
  badge.element.dataset.state = state;
  const value = badge.element.lastElementChild;
  if (value) value.textContent = text;
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** Formats a sampled value as linear-light plus the display nits it decodes to. */
function formatValueWithNits(value: number): string {
  return `${formatFixed(value, 2)} (${formatNits(value * SCENE_WHITE_NITS)})`;
}

function formatFixed(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

/** Binary megabytes, labelled MiB so it cannot be confused with the decimal MB used for video files. */
function formatMiB(bytes: number): string {
  return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
}

/** `748` -> `748 nits`, `12000` -> `12.0k nits`. */
function formatNits(nits: number): string {
  if (!Number.isFinite(nits)) return '—';
  return nits >= 10_000 ? `${(nits / 1000).toFixed(1)}k nits` : `${Math.round(nits)} nits`;
}

/** Formats cumulative streamed frame bytes using GiB once past 1 GiB. */
function formatStreamedBytes(bytes: number): string {
  const mebibytes = bytes / (1024 * 1024);
  return mebibytes >= 1024
    ? `${(mebibytes / 1024).toFixed(2)} GiB`
    : `${mebibytes.toFixed(1)} MiB`;
}

/**
 * Deterministic FNV-1a checksum over the raw float bits. Used to prove that the
 * frame delivered through `onFrame` is real Float32 data and to compare runs.
 */
function checksumFloats(values: Float32Array): string {
  const words = new Uint32Array(values.buffer, values.byteOffset, values.length);
  let hash = 0x811c9dc5;
  for (let index = 0; index < words.length; index++) {
    hash ^= words[index];
    hash = Math.imul(hash, 0x01000193);
  }
  return (hash >>> 0).toString(16).padStart(8, '0');
}

/**
 * Builds a NumPy v1.0 `.npy` header for a C-order `(height, width, 4)` float32
 * array. This lets the downloaded frame load directly with
 * `numpy.load(path)` without any custom parser.
 */
function buildNpyHeader(width: number, height: number): Uint8Array<ArrayBuffer> {
  const preambleLength = 10; // 6 magic + 2 version + 2 header length
  const dict =
    `{'descr': '<f4', 'fortran_order': False, 'shape': (${height}, ${width}, 4), }`;
  // The format requires magic + version + length + header to be 64-byte aligned.
  const padding = (64 - ((preambleLength + dict.length + 1) % 64)) % 64;
  const headerText = `${dict}${' '.repeat(padding)}\n`;
  const header = new Uint8Array(preambleLength + headerText.length);

  header.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59], 0); // \x93NUMPY
  header[6] = 1; // major version
  header[7] = 0; // minor version
  header[8] = headerText.length & 0xff;
  header[9] = (headerText.length >>> 8) & 0xff;
  for (let index = 0; index < headerText.length; index++) {
    header[preambleLength + index] = headerText.charCodeAt(index);
  }
  return header;
}

/** Downloads one retained frame as `(height, width, 4)` little-endian float32. */
function downloadNpyFrame(frame: RetainedFrame): void {
  const blob = new Blob([buildNpyHeader(frame.width, frame.height), frame.pixels], {
    type: 'application/octet-stream',
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  const paddedIndex = String(frame.frameIndex).padStart(3, '0');
  link.href = url;
  link.download =
    `hdr-frame-${paddedIndex}-${frame.width}x${frame.height}-rec2020-linear-rgba32f.npy`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

/**
 * Downloads a muxed HDR video.
 *
 * The filename records the container, codec family, and transfer function,
 * because those three facts are what determine whether a player shows it as HDR.
 */
function downloadMuxedVideo(video: MuxedVideoFile, transfer: 'pq' | 'hlg', url?: string): void {
  const objectUrl = url ?? URL.createObjectURL(new Blob([video.buffer], { type: video.mimeType }));
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download =
    `hdr-${video.frameCount}f-rec2020-${transfer}-${video.codecFamily}.${video.extension}`;
  document.body.append(link);
  link.click();
  link.remove();
  if (!url) setTimeout(() => URL.revokeObjectURL(objectUrl), 60_000);
}

/** Rec. 2100 PQ delivery space used as the default output description. */
function getRec2100PqColorSpace(): ColorSpaceSpec {
  return {
    name: 'Rec.2100-PQ',
    redPrimary: [0.708, 0.292],
    greenPrimary: [0.17, 0.797],
    bluePrimary: [0.131, 0.046],
    whitePoint: [0.3127, 0.329],
    transfer: 'pq',
    peakLuminanceNits: 1000,
  };
}

function createDefaultConfig(selection: WorkbenchRunOptions): RenderConfig {
  return {
    width: selection.width,
    height: selection.height,
    fps: selection.fps,
    frameCount: selection.frameCount,
    workingColorSpace: getWorkingColorSpace(),
    outputColorSpace: getRec2100PqColorSpace(),
    renderFormat: selection.renderFormat,
    seed: 1,
  };
}

/** Minimal frame metadata for probing encoder dimensions/rate before frame 0. */
function createEncoderProbeMeta(config: RenderConfig): FrameMeta {
  return {
    frameIndex: 0,
    timestampUs: 0,
    durationUs: Math.round(1e6 / config.fps),
    width: config.width,
    height: config.height,
    colorSpace: config.workingColorSpace,
    format: config.renderFormat,
    renderTimeMs: 0,
  };
}

/**
 * Probes whether the adapter can render *and* blend into `rgba32float`, which is
 * the pair of features `initGpuContext` requires before selecting that format.
 */
async function detectFloat32RenderTargetSupport(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !navigator.gpu) return false;
  try {
    const adapter =
      (await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })) ??
      (await navigator.gpu.requestAdapter());
    if (!adapter) return false;
    return (
      adapter.features.has('float32-renderable' as GPUFeatureName) &&
      adapter.features.has('float32-blendable' as GPUFeatureName)
    );
  } catch (error) {
    logDebug('workbench', 'rgba32float capability probe failed', {
      error: describeError(error),
    });
    return false;
  }
}

/**
 * Mounts the workbench UI into `options.root`.
 *
 * Creates the canvas, controls, and telemetry grid, wires Start/Cancel to
 * `runFramePipeline`, and manages the active `AbortController`. Format
 * capability probing runs asynchronously and only enables `rgba32float` when the
 * adapter exposes both required features.
 *
 * @param options - Mount target and optional config factory.
 * @returns A `Workbench` handle for programmatic control.
 */
export function mountWorkbench(options: WorkbenchOptions): Workbench {
  const { root } = options;
  const createConfig = options.createConfig ?? createDefaultConfig;
  const webgpuAvailable =
    typeof navigator !== 'undefined' && 'gpu' in navigator && Boolean(navigator.gpu);
  const webCodecsAvailable =
    typeof VideoEncoder !== 'undefined' && typeof VideoFrame !== 'undefined';

  let running = false;
  let unmounted = false;
  let activeController: AbortController | null = null;
  let retainedFrame: RetainedFrame | null = null;
  /** Sampled run peak, kept so an exact scan can be compared against it. */
  let sampledRunPeak: SequencePeak | null = null;
  /** True once the user edits the keep-frame field, disabling auto-follow. */
  let userChoseKeepFrame = false;
  let muxedVideo: MuxedVideoFile | null = null;
  let muxedTransfer: 'pq' | 'hlg' = 'pq';
  let videoObjectUrl: string | null = null;
  let muxedBlob: Blob | null = null;
  let showingComparison = false;
  /** Frame currently on screen from the comparison canvas; null when hidden. */
  let comparisonFrameIndex: number | null = null;

  // --- Header ----------------------------------------------------------------
  const container = createElement('div', 'workbench');

  const header = createElement('div', 'wb-header');
  const headingGroup = createElement('div', 'wb-heading');
  headingGroup.append(
    createElement('h1', 'wb-title', 'HDR Frame Workbench'),
    createElement(
      'p',
      'wb-subtitle',
      'Deterministic 300-frame linear Rec. 2020 render, previewed through an SDR tone-mapping pass.',
    ),
  );

  const stateBadge = createElement('span', 'wb-state', 'Idle');
  stateBadge.dataset.state = 'idle';

  // Conformance strip: the three facts that make the output HDR — the values
  // actually left the GPU above 1.0, the target stayed float, and the frame
  // contract (resolution / rate / duration) held. Placed above the fold because
  // these are the properties being demonstrated, not diagnostics.
  const badges = createElement('div', 'wb-badges');
  const hdrRangeBadge = createBadge(
    'HDR range',
    'Brightest linear sample across the whole run. A value above 1.0 is what makes the frame HDR rather than merely float.',
  );
  const targetBadge = createBadge(
    'HDR target',
    'Float render target actually negotiated, plus the linear working space and the delivery transfer function.',
  );
  const formatBadge = createBadge(
    'Frame contract',
    'Requested resolution, frame rate and total duration, and whether every frame was returned as Float32.',
  );
  badges.append(hdrRangeBadge.element, targetBadge.element, formatBadge.element);

  const controls = createElement('div', 'wb-controls');
  const formatField = createElement('label', 'wb-field');
  const formatSelect = createElement('select', 'wb-select');
  const option16 = createOption('rgba16float', 'rgba16float');
  const option32 = createOption('rgba32float', 'rgba32float (checking…)', true);
  formatSelect.append(option16, option32);
  formatField.append(createElement('span', 'wb-field-label', 'HDR target'), formatSelect);

  const startButton = createElement('button', 'wb-button wb-button--primary', 'Start');
  startButton.type = 'button';
  const cancelButton = createElement('button', 'wb-button', 'Cancel');
  cancelButton.type = 'button';
  cancelButton.disabled = true;

  const exportButton = createElement('button', 'wb-button', 'Export HDR video');
  exportButton.type = 'button';
  exportButton.disabled = !webCodecsAvailable;
  exportButton.title =
    'Re-render all frames deterministically and encode them offline (no live preview). ' +
    'Rendering is a pure function of the frame index, so the exported frames match the preview run.';
  if (!webCodecsAvailable) {
    exportButton.title = 'WebCodecs VideoEncoder is not available in this browser.';
  }

  const frameField = createElement('label', 'wb-field');
  const frameIndexInput = document.createElement('input');
  frameIndexInput.type = 'number';
  frameIndexInput.className = 'wb-number';
  frameIndexInput.min = '0';
  frameIndexInput.max = String(DEFAULT_FRAME_COUNT - 1);
  frameIndexInput.step = '1';
  frameIndexInput.value = '0';
  frameIndexInput.title =
    'Frame copied out through onFrame for the .npy download. Follows the run ' +
    'peak after the first run unless you edit it.';
  frameField.append(
    createElement('span', 'wb-field-label', 'Frame to keep'),
    frameIndexInput,
  );

  const downloadButton = createElement('button', 'wb-button', 'Download frame');
  downloadButton.type = 'button';
  downloadButton.disabled = true;
  downloadButton.title =
    'Download the kept frame as .npy (H×W×4 float32, linear Rec.2020)';

  // One-off exact check of the retained frame. The per-frame telemetry samples
  // pixels to stay cheap, so this button answers "is the sampled peak the real
  // peak?" without paying for a full scan on every frame.
  const verifyButton = createElement('button', 'wb-button', 'Verify peak (exact)');
  verifyButton.type = 'button';
  verifyButton.disabled = true;
  verifyButton.title =
    'Scan every pixel of the kept frame and compare the result with the sampled run peak. ';

  const downloadVideoButton = createElement('button', 'wb-button', 'Download video');
  downloadVideoButton.type = 'button';
  downloadVideoButton.disabled = true;
  downloadVideoButton.title = 'Download the muxed file produced by “Export HDR video”';

  const compareButton = createElement('button', 'wb-button', 'Compare like preview');
  compareButton.type = 'button';
  compareButton.disabled = true;
  compareButton.title =
    'Show the frame selected in “Frame to keep”, decoded from the export and tone-mapped ' +
    'with the preview\'s own ACES + sRGB curve (raw PQ looks far too dark on an SDR display). ' +
    'Toggling back seeks the player to the same frame.';

  controls.append(
    formatField,
    frameField,
    startButton,
    cancelButton,
    downloadButton,
    verifyButton,
    exportButton,
    downloadVideoButton,
    compareButton,
  );
  header.append(headingGroup, stateBadge);

  // --- Preview stage ---------------------------------------------------------
  const stage = createElement('div', 'wb-stage');
  const canvas = createElement('canvas', 'wb-canvas');
  canvas.width = DEFAULT_WIDTH;
  canvas.height = DEFAULT_HEIGHT;

  // Plays the muxed export in place once it exists, so "did the file actually
  // decode?" is answered without leaving the page.
  const videoPreview = createElement('video', 'wb-video');
  videoPreview.controls = true;
  videoPreview.playsInline = true;
  videoPreview.muted = true;
  videoPreview.hidden = true;

  // Shows the exported frame passed through the *preview's own* ACES+sRGB curve,
  // so it is directly comparable with the live preview instead of looking far
  // too dark (PQ shown on an SDR display without a tone curve).
  const comparisonCanvas = createElement('canvas', 'wb-canvas');
  comparisonCanvas.width = DEFAULT_WIDTH;
  comparisonCanvas.height = DEFAULT_HEIGHT;
  comparisonCanvas.hidden = true;

  stage.append(canvas, videoPreview, comparisonCanvas);

  // --- Telemetry -------------------------------------------------------------
  // Two tiers. `hdr` answers "is this a working HDR pipeline and can it sustain
  // the brief?"; `diag` is per-stage detail that explains a slow or odd run.
  const hdrStats = createElement('dl', 'wb-stats wb-stats--hdr');
  const diagStats = createElement('dl', 'wb-stats wb-stats--diag');

  const peakStat = createStat(
    'HDR peak (run)',
    'hdr',
    'Brightest sampled channel over every delivered frame, with the frame that hit it, in nits (1.0 = 100 nits).',
  );
  const sampleStat = createStat(
    'Center pixel',
    'hdr',
    'Linear Rec.2020 RGBA at pixel (width/2, height/2) of the latest frame.',
  );
  const frameStat = createStat('Frames rendered', 'hdr');
  const returnedStat = createStat(
    'Frames returned',
    'hdr',
    'Float32 frames delivered through onFrame, with the cumulative CPU payload size.',
  );
  const fpsStat = createStat(
    'Measured FPS',
    'hdr',
    'Frames completed per second over the last 30 deliveries. The timeline is a fixed 30 fps; this is throughput, not the output rate.',
  );
  const elapsedStat = createStat('Elapsed', 'hdr');

  const targetStat = createStat('Active HDR target', 'diag');
  const timingStat = createStat('Frame time', 'diag');
  const renderTimingStat = createStat(
    'Render / preview',
    'diag',
    'CPU time to evaluate the scene, record commands and submit. Not GPU execution time; GPU time appears in the readback wait.',
  );
  const readbackTimingStat = createStat(
    'Readback / unpack',
    'diag',
    'mapAsync wait (includes GPU copy time) and CPU half-float decode.',
  );
  const encodeTimingStat = createStat(
    'Encode (tap)',
    'diag',
    'Time spent submitting a frame to the encoder plus any back-pressure drain. Does not measure asynchronous encode completion.',
  );
  const readbackStat = createStat(
    'GPU staging readback',
    'diag',
    'Bytes copied GPU->CPU at source precision. rgba16float is 8 B/px, so half of the Float32 payload consumers receive.',
  );
  const retainedStat = createStat(
    'Staging memory',
    'diag',
    'MAP_READ staging ring allocated for the run, plus any shared reuse buffer the pipeline holds.',
  );
  const keptFrameStat = createStat(
    'Exact scan',
    'diag',
    'Full every-pixel scan of the kept frame, used to verify the sampled run peak instead of trusting it.',
  );
  const encoderStat = createStat('VideoEncoder', 'diag');
  const chunksStat = createStat('Encoded chunks', 'diag');

  // HDR tier first so the evidence sits next to the preview, not below the fold.
  hdrStats.append(
    peakStat.cell,
    sampleStat.cell,
    frameStat.cell,
    returnedStat.cell,
    fpsStat.cell,
    elapsedStat.cell,
  );
  diagStats.append(
    targetStat.cell,
    timingStat.cell,
    renderTimingStat.cell,
    readbackTimingStat.cell,
    encodeTimingStat.cell,
    readbackStat.cell,
    retainedStat.cell,
    keptFrameStat.cell,
    encoderStat.cell,
    chunksStat.cell,
  );

  const note = createElement(
    'p',
    'wb-note',
    'Frames are streamed: each Float32 frame (14.06 MiB at 1280×720) is delivered through ' +
      'onFrame and released afterwards; only the selected frame is copied for download, so a ' +
      '300-frame run never retains all frames (~4.12 GiB if it did). “Download frame” saves the ' +
      'kept frame as .npy (H×W×4 float32, linear Rec.2020). Rendering is render-only: ' +
      '“Export HDR video” re-renders the same deterministic timeline headlessly and encodes it ' +
      'offline as I420P10 (I420/RGBA fallback), so encoding never slows the live preview.',
  );

  // Expected capability fallbacks (currently the 8-bit RGBA input path) belong
  // here rather than in the warning banner: they reduce precision but are not
  // errors and cannot be avoided in current browser WebCodecs implementations.
  const encoderNote = createElement('p', 'wb-note');
  encoderNote.hidden = true;

  const warningEl = createElement('p', 'wb-warning');
  warningEl.hidden = true;

  container.append(
    header,
    badges,
    stage,
    controls,
    hdrStats,
    diagStats,
    note,
    encoderNote,
    warningEl,
  );
  root.replaceChildren(container);

  function setState(state: WorkbenchState, label: string): void {
    stateBadge.dataset.state = state;
    stateBadge.textContent = label;
  }

  function showWarning(message: string): void {
    warningEl.hidden = message.length === 0;
    warningEl.textContent = message;
  }

  function clearRetainedFrame(): void {
    retainedFrame = null;
    // `sampledRunPeak` deliberately survives: the exact scan is compared against
    // the peak of the run that produced the frame, not the peak of the pass that
    // happens to be starting.
    keptFrameStat.value.textContent = '—';
    downloadButton.disabled = true;
    verifyButton.disabled = true;
    verifyButton.textContent = 'Verify peak (exact)';
    // The peak stat's tooltip describes the previous run's coverage; leaving it
    // alongside a cleared value would pair stale prose with an em dash.
    peakStat.value.title = '';
  }

  /**
   * Runs the every-pixel scan on the retained frame and reconciles it with the
   * sampled run peak.
   *
   * The scan is cached on the retained frame, so pressing twice is free. When the
   * kept frame is the one that set the sampled peak, the two numbers must agree
   * to within float noise; a disagreement means the sampler missed a highlight,
   * which is exactly the failure mode this check exists to catch.
   */
  function onVerifyClick(): void {
    if (!retainedFrame) return;

    verifyButton.disabled = true;
    verifyButton.textContent = 'Scanning…';
    // Yield once so the button's busy state paints before the synchronous scan.
    setTimeout(() => {
      if (!retainedFrame) {
        verifyButton.textContent = 'Verify peak (exact)';
        return;
      }
      try {
        const scan =
          retainedFrame.exactScan ??
          (retainedFrame.exactScan = scanFrameExact(retainedFrame.pixels, retainedFrame.width));
        const keptIndex = retainedFrame.frameIndex;
        const peakAtSameFrame =
          sampledRunPeak !== null && sampledRunPeak.frameIndex === keptIndex;

        keptFrameStat.value.textContent =
          `${formatValueWithNits(scan.max)} @ (${scan.peakX}, ${scan.peakY}) · ` +
          `${scan.elapsedMs.toFixed(0)} ms`;

        if (peakAtSameFrame && sampledRunPeak) {
          const delta = scan.max - sampledRunPeak.value;
          const relative = sampledRunPeak.value > 0 ? Math.abs(delta) / sampledRunPeak.value : 0;
          // Float32 compositing makes exact equality unrealistic; 0.5% is well
          // inside one sampling-lattice miss and far below a real discrepancy.
          const agrees = relative <= 0.005;
          setBadge(
            hdrRangeBadge,
            agrees ? 'pass' : 'warn',
            `${scan.max.toFixed(2)} exact · ${formatNits(scan.max * SCENE_WHITE_NITS)}` +
              (agrees ? '' : ` (sampled ${sampledRunPeak.value.toFixed(2)})`),
          );
          logInfo('workbench', 'Exact scan of kept frame', {
            frameIndex: keptIndex,
            exactMax: scan.max,
            sampledMaxAtSameFrame: sampledRunPeak.value,
            delta,
            agrees,
            peak: { x: scan.peakX, y: scan.peakY },
            scanMs: Math.round(scan.elapsedMs),
          });
        } else {
          // The kept frame is not the run's peak frame, so the scan verifies the
          // scan itself (value and location), not the run peak.
          verifyButton.title =
            `Kept frame #${keptIndex} is not the run-peak frame ` +
            `(#${sampledRunPeak?.frameIndex ?? '?'}), so this checks frame ` +
            `#${keptIndex} only.`;
          logInfo('workbench', 'Exact scan of kept frame (not the peak frame)', {
            frameIndex: keptIndex,
            exactMax: scan.max,
            runPeakFrame: sampledRunPeak?.frameIndex ?? null,
            peak: { x: scan.peakX, y: scan.peakY },
            scanMs: Math.round(scan.elapsedMs),
          });
        }
      } finally {
        verifyButton.textContent = 'Verify peak (exact)';
        verifyButton.disabled = retainedFrame === null;
      }
    }, 0);
  }

  /**
   * Clears every run-scoped readout before a new pass starts. Without this a
   * new run or export displays the previous run's peak, timings and frame
   * counts until telemetry arrives, which reads as if the new run inherited
   * them.
   */
  function resetRunReadouts(): void {
    for (const stat of [
      frameStat,
      returnedStat,
      elapsedStat,
      timingStat,
      renderTimingStat,
      readbackTimingStat,
      readbackStat,
      retainedStat,
      fpsStat,
    ]) {
      stat.value.textContent = '—';
    }
    peakStat.value.textContent = '—';
    sampleStat.value.textContent = '—';
    encodeTimingStat.value.textContent = '—';
    sampledRunPeak = null;

    // Badges describe the data, not the request, so they reset to unknown and
    // only turn green once a frame has actually proved the property.
    setBadge(hdrRangeBadge, 'unknown', 'awaiting first frame');
    setBadge(targetBadge, 'unknown', 'not negotiated yet');
  }

  function onDownloadClick(): void {
    if (!retainedFrame) return;
    logInfo('workbench', 'Downloading retained frame', {
      frameIndex: retainedFrame.frameIndex,
      width: retainedFrame.width,
      height: retainedFrame.height,
      bytes: retainedFrame.pixels.byteLength,
    });
    downloadNpyFrame(retainedFrame);
  }

  /**
   * Waits for a media event, but never longer than `timeoutMs`.
   *
   * A `<video>` with a bad source emits `error` rather than the event we want, and
   * some browsers never fire `seeked` for an unseekable stream. Resolving on
   * timeout keeps the UI responsive in both cases; the caller can still inspect
   * `currentTime`/`readyState` afterwards.
   */
  function waitForMediaEvent(
    video: HTMLVideoElement,
    event: 'loadedmetadata' | 'seeked',
    timeoutMs = 2000,
  ): Promise<void> {
    return new Promise((resolve) => {
      let timer: ReturnType<typeof setTimeout> | null = null;
      const done = (): void => {
        if (timer !== null) clearTimeout(timer);
        video.removeEventListener(event, done);
        video.removeEventListener('error', done);
        resolve();
      };
      timer = setTimeout(done, timeoutMs);
      video.addEventListener(event, done, { once: true });
      video.addEventListener('error', done, { once: true });
    });
  }

  /**
   * Puts the on-page player on exactly the frame the comparison canvas decoded.
   *
   * This is the whole point of the toggle: both views must show the same frame,
   * so "raw PQ" is a transfer-function comparison rather than a comparison
   * between frame N and whatever position the player happened to be at (0 s on a
   * fresh export). The seek target comes from `comparisonTimestampSeconds`, the
   * same helper the offscreen decode uses.
   */
  async function seekPreviewToFrame(frameIndex: number): Promise<void> {
    const video = videoPreview;
    const frameCount = muxedVideo?.frameCount ?? DEFAULT_FRAME_COUNT;
    const targetSeconds = comparisonTimestampSeconds(frameIndex, frameCount, DEFAULT_FPS);

    // A preview run detaches the player's src to release its decoder. If the
    // user then compares the previous export, reattach the same blob so the
    // toggle has something to reveal.
    if (!video.hasAttribute('src') && videoObjectUrl) {
      video.src = videoObjectUrl;
    }

    // Pause before seeking so playback cannot advance out from under the seek.
    video.pause();

    // HAVE_METADATA === 1; `duration` and a usable timeline need at least this.
    if (video.readyState < 1) {
      await waitForMediaEvent(video, 'loadedmetadata');
    }

    const duration = Number.isFinite(video.duration) ? video.duration : targetSeconds;
    const target = Math.min(targetSeconds, Math.max(duration - 1e-3, 0));
    if (Math.abs(video.currentTime - target) > 1e-3) {
      const seeked = waitForMediaEvent(video, 'seeked');
      video.currentTime = target;
      await seeked;
    }

    logInfo('workbench', 'Seeked raw player to compared frame', {
      frameIndex,
      targetSeconds: Number(target.toFixed(4)),
      durationSeconds: Number.isFinite(video.duration)
        ? Number(video.duration.toFixed(4))
        : null,
      currentTime: Number.isFinite(video.currentTime)
        ? Number(video.currentTime.toFixed(4))
        : null,
    });
  }

  /**
   * Drops any comparison view and restores the default button state.
   *
   * Called when the stage is about to be reused (new run or new export). Without
   * this, a comparison canvas can remain visible on top of the live preview, and
   * the button keeps a `showingComparison` state pointing at a file that no longer
   * exists.
   */
  function resetComparisonView(): void {
    showingComparison = false;
    comparisonFrameIndex = null;
    comparisonCanvas.hidden = true;
    compareButton.textContent = 'Compare like preview';
  }

  /**
   * Renders the exported frame through the preview's own tone curve so the two
   * images are comparable. Toggles back to the raw player on a second press.
   */
  async function onCompareClick(): Promise<void> {
    if (!muxedVideo || !muxedBlob) return;

    if (showingComparison) {
      // Seek the visible player to the compared frame *before* revealing it, so
      // the toggle never flashes frame 0 on its way to the right time.
      const compared = comparisonFrameIndex;
      compareButton.disabled = true;
      compareButton.textContent = 'Seeking…';
      try {
        if (compared !== null) await seekPreviewToFrame(compared);
        showingComparison = false;
        comparisonFrameIndex = null;
        comparisonCanvas.hidden = true;
        videoPreview.hidden = false;
        compareButton.textContent = 'Compare like preview';
      } catch (error) {
        logError('workbench', 'Seeking raw player to compared frame failed', error);
        showWarning(`Could not seek the player: ${describeError(error)}`);
        // Still reveal the player; leaving the comparison canvas up with a
        // "Show raw PQ" label would be the more confusing state.
        showingComparison = false;
        comparisonFrameIndex = null;
        comparisonCanvas.hidden = true;
        videoPreview.hidden = false;
        compareButton.textContent = 'Compare like preview';
      } finally {
        compareButton.disabled = false;
      }
      return;
    }

    compareButton.disabled = true;
    compareButton.textContent = 'Comparing…';
    // Freeze the player while it is hidden, so its position cannot drift away
    // from the frame being decoded.
    videoPreview.pause();
    try {
      const keepIndex = Math.min(
        muxedVideo.frameCount - 1,
        Math.max(0, Math.floor(Number.parseInt(frameIndexInput.value, 10) || 0)),
      );
      const image = await renderComparisonFrame(
        muxedBlob,
        keepIndex,
        muxedVideo.frameCount,
        DEFAULT_FPS,
      );

      comparisonCanvas.width = image.width;
      comparisonCanvas.height = image.height;
      const context = comparisonCanvas.getContext('2d');
      if (!context) throw new Error('2D context unavailable');
      context.putImageData(new ImageData(image.pixels, image.width, image.height), 0, 0);

      comparisonCanvas.hidden = false;
      videoPreview.hidden = true;
      showingComparison = true;
      comparisonFrameIndex = keepIndex;
      compareButton.textContent = `Show raw PQ (frame ${keepIndex})`;
      logInfo('workbench', 'Rendered preview-matched comparison frame', {
        frameIndex: keepIndex,
        width: image.width,
        height: image.height,
        note: 'exported frames tone-mapped with the preview ACES+sRGB curve',
      });
    } catch (error) {
      logError('workbench', 'Comparison render failed', error);
      showWarning(`Comparison failed: ${describeError(error)}`);
      // Return the stage to the player: a half-entered comparison state would
      // leave a hidden canvas under a "Show raw PQ" label.
      showingComparison = false;
      comparisonFrameIndex = null;
      comparisonCanvas.hidden = true;
      videoPreview.hidden = false;
      compareButton.textContent = 'Compare like preview';
    } finally {
      compareButton.disabled = false;
    }
  }

  function onDownloadVideoClick(): void {
    if (!muxedVideo) return;
    logInfo('workbench', 'Downloading muxed HDR video', {
      bytes: muxedVideo.buffer.byteLength,
      frameCount: muxedVideo.frameCount,
      codecFamily: muxedVideo.codecFamily,
      container: muxedVideo.extension,
      transfer: muxedTransfer,
      durationSeconds: Number(muxedVideo.durationSeconds.toFixed(3)),
    });
    downloadMuxedVideo(muxedVideo, muxedTransfer, videoObjectUrl ?? undefined);
  }

  function onFrameIndexChange(): void {
    if (running) return;
    // Once the user picks a frame by hand, a later run must not silently move
    // their selection out from under them.
    userChoseKeepFrame = true;
    // Drop any "auto-set to the run peak" wording left by a previous run.
    frameIndexInput.title =
      'Frame copied out through onFrame for the .npy download (chosen manually).';
    clearRetainedFrame();
  }

  /**
   * Encodes the whole timeline offline in a headless pass.
   *
   * Rendering is a pure function of the frame index, so re-rendering produces the
   * same pixels the preview pass showed. Running it separately keeps encoding cost
   * and back-pressure out of the interactive preview entirely.
   */
  async function exportVideo(): Promise<void> {
    if (running || unmounted || !webCodecsAvailable) return;
    if (!webgpuAvailable) {
      showWarning('WebGPU is not available; cannot export.');
      return;
    }

    running = true;
    startButton.disabled = true;
    exportButton.disabled = true;
    cancelButton.disabled = false;
    formatSelect.disabled = true;
    frameIndexInput.disabled = true;
    showWarning('');
    setState('running', 'Exporting');
    resetRunReadouts();

    const controller = new AbortController();
    activeController = controller;
    const runStart = performance.now();

    const selection: WorkbenchRunOptions = {
      width: DEFAULT_WIDTH,
      height: DEFAULT_HEIGHT,
      fps: DEFAULT_FPS,
      frameCount: DEFAULT_FRAME_COUNT,
      renderFormat: formatSelect.value === 'rgba32float' ? 'rgba32float' : 'rgba16float',
    };
    const config = createConfig(selection);

    const videoChunks: CollectedVideoChunk[] = [];
    let encoderTap: VideoEncoderTap | null = null;

    encoderStat.value.textContent = 'Starting…';
    chunksStat.value.textContent = '—';
    encoderNote.hidden = true;
    encoderNote.textContent = '';

    // Export returns frames to the encoder, not to `onFrame`, so the returned
    // tile must describe this pass rather than keep the preview run's numbers.
    returnedStat.value.textContent = `none (frames go to encoder)`;
    clearRetainedFrame();
    setContractBadge(config, 'fail', 0);

    try {
      logInfo('workbench', 'Creating VideoEncoder tap for offline export', {
        width: config.width,
        height: config.height,
        fps: config.fps,
        transfer: config.outputColorSpace.transfer,
      });

      encoderTap = await createVideoEncoderTap(
        createEncoderProbeMeta(config),
        (chunk, metadata) => {
          videoChunks.push({ chunk, metadata });
        },
        {
          // Constrain selection to codecs the MP4 muxer can carry.
          codecs: EXPORT_CODEC_CANDIDATES,
          transfer: config.outputColorSpace.transfer === 'hlg' ? 'hlg' : 'pq',
          onWarning: (message) => {
            showWarning(`Video encoder: ${message}`);
          },
          onInfo: (message) => {
            encoderNote.hidden = false;
            encoderNote.textContent = message;
          },
          onError: (error) => {
            showWarning(`Video encoder: ${error.message}`);
          },
        },
      );

      if (!encoderTap) {
        logInfo('workbench', 'No VideoEncoder tap available; export aborted');
        if (warningEl.hidden) {
          showWarning(
            'No supported 10-bit HDR VideoEncoder profile; nothing to export.',
          );
        }
        setState('error', 'Unsupported');
        updateEncoderReadout(null, true);
        return;
      }

      const tap = encoderTap;

      // Backstop: even with a constrained candidate list, never start a full
      // encode pass whose output the muxer cannot carry. Failing here keeps the
      // error attributable to codec selection rather than the container writer.
      if (!canMuxVideo(tap.codec)) {
        logWarn('workbench', 'Export codec cannot be muxed; aborting before encode', {
          codec: tap.codec,
        });
        showWarning(
          `The selected encoder (${tap.codec}) cannot be written to a container. ` +
            'Export requires AV1/HEVC (MP4) or VP9 (WebM); see the console ' +
            '“Codec probe results” log.',
        );
        setState('error', 'Unsupported');
        updateEncoderReadout(tap.status(), true);
        return;
      }

      logInfo('workbench', 'VideoEncoder tap ready for export', {
        codec: tap.codec,
        frameFormat: tap.frameFormat,
        state: tap.status().state,
      });

      const summary = await runFramePipeline({
        config,
        // Headless: no preview canvas, so export does not touch the swap chain.
        previewCanvas: null,
        signal: controller.signal,
        encoderTap: tap,
        callbacks: {
          onTelemetry: (snapshot) => {
            updateTelemetry(snapshot);
            chunksStat.value.textContent =
              `${videoChunks.length} chunks · ${formatMiB(tap.status().chunkBytes)} encoded`;
          },
          onWarning: showWarning,
        },
      });

      updateEncoderReadout(tap.status(), true);
      setState(summary.cancelled ? 'cancelled' : 'done', summary.cancelled ? 'Cancelled' : 'Complete');
      // The export contract is "every frame reached the encoder"; `onFrame` is
      // not used on this path, so the encoder's own submission count is the
      // evidence that no frame was dropped under back-pressure.
      const submitted = tap.status().framesSubmitted;
      setContractBadge(
        config,
        !summary.cancelled && submitted === config.frameCount ? 'pass' : 'fail',
        submitted,
      );

      // Timing is derived from chunk timestamps, not summed durations; logging
      // it makes an off-by-one-frame shortfall immediately visible.
      const timing = computeChunkTiming(videoChunks, config.fps);
      logInfo('workbench', 'Export complete', {
        framesEncoded: summary.framesCompleted,
        chunks: videoChunks.length,
        chunkBytes: tap.status().chunkBytes,
        keyFrames: tap.status().keyFrames,
        codec: tap.codec,
        frameFormat: tap.frameFormat,
        elapsedMs: Math.round(summary.elapsedMs),
        encodedDurationSeconds: Number((timing.totalUs / 1_000_000).toFixed(3)),
        expectedDurationSeconds: config.frameCount / config.fps,
        chunkTimestampsMonotonic: timing.monotonic,
        lastChunkDurationUs: videoChunks[videoChunks.length - 1]?.chunk.duration ?? null,
      });

      if (!summary.cancelled && videoChunks.length > 0) {
        // Muxing is synchronous and cheap (~a few MB of memcpy), and it needs the
        // whole chunk list, so it runs after the encode pass finishes.
        try {
          const produced = muxVideo({
            chunks: videoChunks,
            codec: tap.codec,
            width: config.width,
            height: config.height,
            fps: config.fps,
            transfer: config.outputColorSpace.transfer === 'hlg' ? 'hlg' : 'pq',
          });

          if (produced) {
            muxedVideo = produced;
            muxedTransfer = config.outputColorSpace.transfer === 'hlg' ? 'hlg' : 'pq';
            downloadVideoButton.disabled = false;
            compareButton.disabled = false;

            // One blob URL shared by the in-page player and the save button, so
            // the (potentially multi-MB) file is not copied twice.
            if (videoObjectUrl) URL.revokeObjectURL(videoObjectUrl);
            muxedBlob = new Blob([produced.buffer], { type: produced.mimeType });
            videoObjectUrl = URL.createObjectURL(muxedBlob);
            // A new file invalidates any comparison of the previous one.
            resetComparisonView();
            videoPreview.src = videoObjectUrl;
            videoPreview.hidden = false;
            canvas.hidden = true;

            const megabytes = (produced.buffer.byteLength / 1_000_000).toFixed(2);
            logInfo('workbench', 'Muxed HDR video', {
              container: produced.extension,
              codecFamily: produced.codecFamily,
              transfer: muxedTransfer,
              frames: produced.frameCount,
              bytes: produced.buffer.byteLength,
              megabytes: Number(megabytes),
              durationSeconds: Number(produced.durationSeconds.toFixed(3)),
            });
            showMuxedVideoSummary(produced, muxedTransfer);
            showWarning('');
          } else {
            showWarning('Export finished but produced no muxable chunks.');
          }
        } catch (error) {
          // A mux failure must not hide the fact that encoding succeeded.
          logError('workbench', 'Muxing encoded chunks failed', error, {
            chunks: videoChunks.length,
            codec: tap.codec,
          });
          showWarning(`Video mux failed: ${describeError(error)}`);
        }
      } else if (!summary.cancelled) {
        showWarning('Export finished but produced no encoded chunks.');
      }
    } catch (error) {
      logError('workbench', 'Export failed', error, {
        elapsedMs: Math.round(performance.now() - runStart),
      });
      setState('error', 'Error');
      showWarning(`Export failed: ${describeError(error)}`);
    } finally {
      await flushAndClose(encoderTap);
      // Only refresh the tile from the tap when no file summary was written:
      // `showMuxedVideoSummary` is the more useful final state and the live
      // counters are preserved in its tooltip.
      if (encoderTap && !muxedVideo) updateEncoderReadout(encoderTap.status(), true);

      if (activeController === controller) activeController = null;
      running = false;
      if (!unmounted) {
        startButton.disabled = !webgpuAvailable;
        cancelButton.disabled = true;
        formatSelect.disabled = false;
        frameIndexInput.disabled = false;
        exportButton.disabled = !webCodecsAvailable;
      }
    }
  }

  /**
   * Renders a tap status. `requested` distinguishes "encoder switched off" from
   * "WebCodecs/profile unavailable", which are very different outcomes for a run.
   */
  function updateEncoderReadout(status: EncoderTelemetry | null, requested: boolean): void {
    if (!status) {
      encoderStat.value.textContent = requested ? 'Unavailable' : 'Off';
      chunksStat.value.textContent = '—';
      if (!requested) encodeTimingStat.value.textContent = '—';
      return;
    }

    const formatLabel =
      status.frameFormat === 'I420P10'
        ? 'I420P10 (10-bit)'
        : `${status.frameFormat} (8-bit fallback)`;
    encoderStat.value.textContent =
      `${status.state} · ${status.codec} · ${formatLabel} · ${status.colorSpace}`;
    encoderStat.value.title =
      `${status.framesSubmitted} sent · ${status.chunksEmitted} chunks · ` +
      `${formatMiB(status.chunkBytes)} · ${status.keyFrames} key · q${status.encodeQueueSize}`;
    if (status.lastError) {
      showWarning(`VideoEncoder: ${status.lastError}`);
    }
  }

  /**
   * Reports the finalized container instead of live encoder counters.
   *
   * The live counters only describe the encode pass; once the file exists, the
   * facts that matter are its format, size, and duration — in particular that
   * the duration really is the 10 s the brief asks for.
   */
  function showMuxedVideoSummary(video: MuxedVideoFile, transfer: 'pq' | 'hlg'): void {
    const sizeMb = video.buffer.byteLength / 1_000_000;
    // Video file sizes are decimal (MB), unlike the binary MiB used for memory.
    encoderStat.value.textContent =
      `${video.codecFamily} · ${video.extension.toUpperCase()} · Rec.2020/${transfer}`;
    chunksStat.value.textContent =
      `${video.frameCount} frames · ${video.durationSeconds.toFixed(2)} s · ${sizeMb.toFixed(2)} MB`;
    chunksStat.value.title =
      `Muxed file: ${video.frameCount} frames, ${video.durationSeconds.toFixed(3)} s, ` +
      `${video.buffer.byteLength} bytes, ${video.mimeType}`;
  }

  /**
   * States the brief's own numbers up front so a run can be checked against the
   * requirements without counting tiles: resolution, rate, duration, and how many
   * frames must come back through `onFrame`. Outcome (`n / total`) is supplied by
   * the caller: the preview path counts `onFrame` deliveries, the export path
   * counts frames handed to the encoder.
   */
  function setContractBadge(config: RenderConfig, outcome: 'pass' | 'fail', delivered: number): void {
    const seconds = config.frameCount / config.fps;
    setBadge(
      formatBadge,
      outcome,
      `${config.width}×${config.height} · ${config.fps} fps · ${seconds.toFixed(2)} s · ` +
        `${delivered}/${config.frameCount} frames`,
    );
  }

  function updateTelemetry(snapshot: TelemetrySnapshot): void {
    // `framesCompleted` is a count (0..total), so the final sample reads
    // `300 / 300` rather than the last 0-based frame index (`299`).
    frameStat.value.textContent = `${snapshot.framesCompleted} / ${snapshot.framesTotal}`;
    elapsedStat.value.textContent = `${Math.round(snapshot.elapsedMs ?? 0)} ms`;

    // `average` here is cumulative (elapsed / frames), so the label says so
    // rather than implying a moving window.
    timingStat.value.textContent =
      snapshot.framesCompleted > 0
        ? `${snapshot.lastFrameMs.toFixed(1)} ms · cum avg ${snapshot.averageFrameMs.toFixed(1)} ms`
        : '—';

    // Frames completed per second over the last 30 deliveries. Labelled `fps`
    // and annotated with the fixed timeline rate so render throughput is never
    // mistaken for the output frame rate. Two decimals keep jitter visible.
    const budgetMs = 1000 / DEFAULT_FPS;
    fpsStat.value.textContent =
      snapshot.measuredFps > 0 ? `${snapshot.measuredFps.toFixed(2)} fps` : '—';
    fpsStat.value.title =
      `Render throughput (last 30 frames) vs the fixed ${DEFAULT_FPS} fps timeline ` +
      `(${budgetMs.toFixed(1)} ms/frame budget).`;

    const timing = snapshot.lastFrameTimings;
    const average = snapshot.averageFrameTimings;
    renderTimingStat.value.textContent = timing
      ? `${timing.renderMs.toFixed(1)} / ${timing.previewMs.toFixed(1)} ms` +
        (average
          ? ` · avg ${average.renderMs.toFixed(1)} / ${average.previewMs.toFixed(1)}`
          : '')
      : '—';
    readbackTimingStat.value.textContent = timing
      ? `${timing.readbackMs.toFixed(1)} / ${timing.unpackMs.toFixed(1)} ms` +
        (average
          ? ` · avg ${average.readbackMs.toFixed(1)} / ${average.unpackMs.toFixed(1)}`
          : '')
      : '—';
    encodeTimingStat.value.textContent =
      snapshot.encoder && timing
        ? `${timing.encodeMs.toFixed(1)} ms` +
          (average ? ` · avg ${average.encodeMs.toFixed(1)} ms` : '')
        : '—';

    // Run-wide peak: the single most useful HDR signal, so it leads with nits
    // and notes the frame that produced it. `min–max` of the last frame is a
    // tooltip detail, not the headline.
    const peak = snapshot.sequencePeak;
    const values = snapshot.lastFrameValues;
    if (peak) {
      // Remembered so `onVerifyClick` can reconcile an exact scan against it.
      sampledRunPeak = peak;
      peakStat.value.textContent =
        `${peak.value.toFixed(2)} (${formatNits(peak.nits)}) @ #${peak.frameIndex}`;
      // Coverage comes from the peak frame, not the current one, so the two
      // numbers in this tooltip always describe the same frame.
      const coveragePct = (peak.coverage * 100).toFixed(2);
      peakStat.value.title =
        peak.value > 1
          ? `Pixel R ${formatValueWithNits(peak.rgb[0])} · G ${formatValueWithNits(peak.rgb[1])} · ` +
            `B ${formatValueWithNits(peak.rgb[2])}. Sampled ${coveragePct}% of the peak ` +
            `frame's pixels (frame #${peak.frameIndex}); use “Verify peak (exact)” for a ` +
            `full scan.`
          : `No sample exceeded reference white (1.0); sampled ${coveragePct}% of pixels.`;
      setBadge(
        hdrRangeBadge,
        peak.value > 1 ? 'pass' : 'warn',
        peak.value > 1
          ? `${peak.value.toFixed(2)} peak · ${formatNits(peak.nits)}`
          : `<= 1.0 (${formatNits(peak.nits)})`,
      );
    } else if (values) {
      // First frame, before the sequence peak exists.
      peakStat.value.textContent = `${formatFixed(values.min)} – ${formatFixed(values.max)}`;
    }

    sampleStat.value.textContent = values
      ? `R ${formatValueWithNits(values.sample[0])} · G ${formatValueWithNits(values.sample[1])} · ` +
        `B ${formatValueWithNits(values.sample[2])} · A ${formatFixed(values.sample[3], 2)}`
      : '—';

    readbackStat.value.textContent =
      `${formatMiB(snapshot.readbackBytes)} total · ${formatMiB(snapshot.readbackPerFrameBytes)}/frame`;
    retainedStat.value.textContent =
      `${formatMiB(snapshot.stagingBytes)} staging` +
      (snapshot.retainedBytes > 0 ? ` · ${formatMiB(snapshot.retainedBytes)} reuse` : '');

    if (snapshot.renderFormat) {
      targetStat.value.textContent = snapshot.renderFormat;
      setBadge(
        targetBadge,
        snapshot.renderFormat.endsWith('float') ? 'pass' : 'fail',
        `${snapshot.renderFormat} · Rec.2020-linear`,
      );
    }

    if (snapshot.encoder) {
      updateEncoderReadout(snapshot.encoder, true);
    }

    if (snapshot.warnings.length > 0) {
      showWarning(snapshot.warnings.join('\n'));
    }
  }

  /**
   * Moves the keep-frame selection onto the frame that set the run peak.
   *
   * Skipped when the run was cancelled (the peak may be partial), when no peak
   * was recorded, or once the user has typed a frame index themselves. The peak
   * frame has already been delivered by this point, so the updated selection
   * means "keep this frame next time" — it never re-delivers the current run.
   */
  function followRunPeak(config: RenderConfig, cancelled: boolean): void {
    if (cancelled || userChoseKeepFrame || sampledRunPeak === null) return;

    const peakIndex = Math.min(
      config.frameCount - 1,
      Math.max(0, sampledRunPeak.frameIndex),
    );
    const previous = Number.parseInt(frameIndexInput.value, 10);
    if (previous === peakIndex) return;

    frameIndexInput.value = String(peakIndex);
    frameIndexInput.title =
      `Frame copied out through onFrame for the .npy download (auto-set to the run peak, ` +
      `frame #${peakIndex})`;
    logInfo('workbench', 'Keep-frame selection followed the run peak', {
      previousKeepIndex: Number.isFinite(previous) ? previous : null,
      keepIndex: peakIndex,
      peakFrameIndex: sampledRunPeak.frameIndex,
      peakValue: Number(sampledRunPeak.value.toFixed(4)),
    });
  }

  async function run(): Promise<void> {
    if (running || unmounted) return;

    if (!webgpuAvailable) {
      const message = 'WebGPU is not available in this browser; cannot start a run.';
      logWarn('workbench', message);
      showWarning(message);
      setState('error', 'Unsupported');
      return;
    }

    running = true;
    startButton.disabled = true;
    cancelButton.disabled = false;
    formatSelect.disabled = true;
    frameIndexInput.disabled = true;
    showWarning('');
    setState('running', 'Running');

    const controller = new AbortController();
    activeController = controller;
    const runStart = performance.now();

    const selection: WorkbenchRunOptions = {
      width: DEFAULT_WIDTH,
      height: DEFAULT_HEIGHT,
      fps: DEFAULT_FPS,
      frameCount: DEFAULT_FRAME_COUNT,
      renderFormat: formatSelect.value === 'rgba32float' ? 'rgba32float' : 'rgba16float',
    };
    const config = createConfig(selection);

    const requestedKeepIndex = Number.parseInt(frameIndexInput.value, 10);
    const keepFrameIndex = Number.isFinite(requestedKeepIndex)
      ? Math.min(config.frameCount - 1, Math.max(0, Math.floor(requestedKeepIndex)))
      : 0;
    frameIndexInput.max = String(config.frameCount - 1);
    // Keep the input in sync with the clamped value actually used.
    frameIndexInput.value = String(keepFrameIndex);
    if (!userChoseKeepFrame) {
      frameIndexInput.title =
        'Frame copied out through onFrame for the .npy download. Follows the run ' +
        'peak unless you edit it.';
    }

    let returnedFrames = 0;
    let returnedBytes = 0;
    clearRetainedFrame();
    resetRunReadouts();
    setContractBadge(config, 'fail', 0);
    returnedStat.value.textContent = `0 / ${config.frameCount} · 0 MiB`;

    logInfo('workbench', 'Starting run', {
      width: config.width,
      height: config.height,
      fps: config.fps,
      frameCount: config.frameCount,
      renderFormat: config.renderFormat,
      webCodecsAvailable,
      logging: describeLogging(),
    });

    // The preview swap chain is sized from the canvas drawing buffer, so it must
    // be set before the pipeline creates the preview renderer.
    canvas.width = config.width;
    canvas.height = config.height;
    // A fresh render replaces any previously exported video in the stage.
    videoPreview.hidden = true;
    videoPreview.removeAttribute('src');
    // If a comparison was showing, clear it: the stage is now the live canvas,
    // and a stale comparison canvas would sit on top of it.
    resetComparisonView();
    canvas.hidden = false;
    targetStat.value.textContent = selection.renderFormat;
    encoderStat.value.textContent = 'Not used (offline export only)';
    chunksStat.value.textContent = '—';
    encodeTimingStat.value.textContent = '—';
    encoderNote.hidden = true;
    encoderNote.textContent = '';

    try {
      // The preview pass never encodes. Encoding is a separate, explicitly
      // requested offline pass (see exportVideo) so a normal run pays no encoder
      // cost and the preview cannot be throttled by encode back-pressure.
      const summary = await runFramePipeline({
        config,
        previewCanvas: canvas,
        signal: controller.signal,
        // No encoder tap: the preview pass is render-only.
        encoderTap: null,
        callbacks: {
          onFrame: (frame: FloatFrame) => {
            returnedFrames++;
            returnedBytes += frame.pixels.byteLength;
            returnedStat.value.textContent =
              `${returnedFrames} / ${config.frameCount} · ${formatStreamedBytes(returnedBytes)}`;

            if (!retainedFrame && frame.meta.frameIndex === keepFrameIndex) {
              // Copy now: the pipeline releases (or reuses) its payload as soon
              // as this callback returns.
              const pixels = frame.pixels.slice();
              retainedFrame = {
                frameIndex: frame.meta.frameIndex,
                width: frame.meta.width,
                height: frame.meta.height,
                pixels,
              };
              const checksum = checksumFloats(pixels);
              // The tile reports the scan result, so the checksum lives in the
              // tooltip and the console until a scan runs.
              keptFrameStat.value.textContent = `frame #${frame.meta.frameIndex} · not scanned`;
              keptFrameStat.value.title = `Retained frame #${frame.meta.frameIndex}, FNV-1a ${checksum}`;
              downloadButton.disabled = false;
              verifyButton.disabled = false;
              // Point the exact check at the frame most likely to matter: the one
              // the sampled telemetry flagged as the sequence peak.
              verifyButton.title =
                sampledRunPeak && sampledRunPeak.frameIndex !== frame.meta.frameIndex
                  ? `Exact-rescan frame #${frame.meta.frameIndex} and compare with the sampled ` +
                    `run peak (frame #${sampledRunPeak.frameIndex}).`
                  : `Exact-rescan frame #${frame.meta.frameIndex} and compare with the sampled run peak.`;
              logInfo('workbench', 'Retained frame for download', {
                frameIndex: frame.meta.frameIndex,
                width: frame.meta.width,
                height: frame.meta.height,
                bytes: pixels.byteLength,
                checksum,
              });
            }
          },
          onTelemetry: updateTelemetry,
          onWarning: showWarning,
        },
      });

      updateTelemetry(summary.telemetry);
      setState(summary.cancelled ? 'cancelled' : 'done', summary.cancelled ? 'Cancelled' : 'Complete');
      const returnedAll = !summary.cancelled && returnedFrames === config.frameCount;
      setContractBadge(config, returnedAll ? 'pass' : 'fail', returnedFrames);

      if (summary.framesCompleted !== returnedFrames) {
        logWarn('workbench', 'onFrame count does not match completed frames', {
          onFrame: returnedFrames,
          framesCompleted: summary.framesCompleted,
        });
      }
      if (!retainedFrame) {
        keptFrameStat.value.textContent = `#${keepFrameIndex} not delivered`;
      } else if (!retainedFrame.exactScan) {
        // A completed run is the moment to make the offer concrete.
        keptFrameStat.value.textContent =
          `frame #${retainedFrame.frameIndex} · press Verify peak`;
      }

      logInfo('workbench', summary.cancelled ? 'Run cancelled' : 'Run complete', {
        framesCompleted: summary.framesCompleted,
        framesReturned: returnedFrames,
        framesTotal: config.frameCount,
        streamedMiB: Number((returnedBytes / (1024 * 1024)).toFixed(1)),
        keptFrameIndex: retainedFrame?.frameIndex ?? null,
        elapsedMs: Math.round(summary.elapsedMs),
        measuredFps: Number(summary.telemetry.measuredFps.toFixed(2)),
        warnings: summary.telemetry.warnings,
        encoder: summary.telemetry.encoder,
      });

      if (!summary.cancelled && summary.framesCompleted < config.frameCount) {
        const message =
          `Run ended after ${summary.framesCompleted}/${config.frameCount} frames ` +
          'without reporting an error. Check the console for the last frame and any GPU ' +
          'device-lost/uncaptured errors.';
        logWarn('workbench', message, {
          lastFrameIndex: summary.telemetry.lastFrameIndex,
          elapsedMs: Math.round(summary.elapsedMs),
          warnings: summary.telemetry.warnings,
        });
        showWarning(message);
      }

      // Point the keep-frame field at the frame that actually set the run peak,
      // so “Download frame” and “Verify peak (exact)” act on the most
      // interesting frame without the user having to read the peak's index and
      // type it back in. The frame itself has already been delivered, so this
      // changes the selection for the *next* run.
      followRunPeak(config, summary.cancelled);
    } catch (error) {
      logError('workbench', 'Run failed', error, {
        framesRequested: config.frameCount,
        elapsedMs: Math.round(performance.now() - runStart),
      });
      setState('error', 'Error');
      showWarning(`Run failed: ${describeError(error)}`);
    } finally {
      if (activeController === controller) activeController = null;
      running = false;
      if (!unmounted) {
        startButton.disabled = !webgpuAvailable;
        cancelButton.disabled = true;
        formatSelect.disabled = false;
        frameIndexInput.disabled = false;
        exportButton.disabled = !webCodecsAvailable;
      }
    }
  }

  function cancel(): void {
    if (!activeController || activeController.signal.aborted) return;
    logInfo('workbench', 'Cancel requested');
    activeController.abort();
    setState('cancelling', 'Cancelling');
    cancelButton.disabled = true;
  }

  function unmount(): void {
    if (unmounted) return;
    unmounted = true;
    if (activeController) {
      logInfo('workbench', 'Unmounting during an active run; aborting it');
    }
    activeController?.abort();
    activeController = null;
    if (videoObjectUrl) {
      URL.revokeObjectURL(videoObjectUrl);
      videoObjectUrl = null;
    }
    startButton.removeEventListener('click', onStartClick);
    cancelButton.removeEventListener('click', onCancelClick);
    downloadVideoButton.removeEventListener('click', onDownloadVideoClick);
    compareButton.removeEventListener('click', onCompareClick);
    downloadButton.removeEventListener('click', onDownloadClick);
    verifyButton.removeEventListener('click', onVerifyClick);
    frameIndexInput.removeEventListener('change', onFrameIndexChange);
    container.remove();
  }

  function onStartClick(): void {
    void run();
  }

  function onCancelClick(): void {
    cancel();
  }

  function onExportClick(): void {
    void exportVideo();
  }

  startButton.addEventListener('click', onStartClick);
  cancelButton.addEventListener('click', onCancelClick);
  exportButton.addEventListener('click', onExportClick);
  downloadVideoButton.addEventListener('click', onDownloadVideoClick);
  compareButton.addEventListener('click', () => void onCompareClick());
  downloadButton.addEventListener('click', onDownloadClick);
  verifyButton.addEventListener('click', onVerifyClick);
  frameIndexInput.addEventListener('change', onFrameIndexChange);
  updateEncoderReadout(null, false);

  if (!webgpuAvailable) {
    startButton.disabled = true;
    showWarning('WebGPU is not available in this browser. This POC requires a WebGPU-capable browser.');
    setState('error', 'Unsupported');
  } else {
    void detectFloat32RenderTargetSupport().then((supported) => {
      if (unmounted) return;
      option32.disabled = !supported;
      option32.textContent = supported ? 'rgba32float' : 'rgba32float (unsupported)';
    });
  }

  return {
    canvas,
    statusEl: hdrStats,
    run,
    cancel,
    updateTelemetry,
    unmount,
  };
}
