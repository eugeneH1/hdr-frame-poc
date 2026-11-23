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
  TelemetrySnapshot,
} from '../types/frame.ts';
import { getWorkingColorSpace } from '../timeline/sceneAnimator.ts';
import { runFramePipeline } from '../pipeline/framePipeline.ts';
import { createVideoEncoderTap, flushAndClose } from '../encoder/webCodecsTap.ts';
import type { VideoEncoderTap } from '../encoder/webCodecsTap.ts';
import type { CollectedVideoChunk, MuxedVideo } from '../encoder/videoMuxer.ts';
import { describeLogging, logDebug, logError, logInfo, logWarn } from '../diagnostics/logger.ts';

const DEFAULT_WIDTH = 1280;
const DEFAULT_HEIGHT = 720;
const DEFAULT_FPS = 30;
const DEFAULT_FRAME_COUNT = 300;

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

interface StatCell {
  readonly cell: HTMLDivElement;
  readonly value: HTMLElement;
}

/** One completed CPU frame copied out of the streaming pipeline for download. */
interface RetainedFrame {
  readonly frameIndex: number;
  readonly width: number;
  readonly height: number;
  /** Copied out of the streaming pipeline, so this owns a regular ArrayBuffer. */
  readonly pixels: Float32Array<ArrayBuffer>;
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

function createStat(label: string): StatCell {
  const cell = createElement('div', 'wb-stat');
  const term = createElement('dt', 'wb-stat-label', label);
  const value = createElement('dd', 'wb-stat-value', '—');
  cell.append(term, value);
  return { cell, value };
}

function describeError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatFixed(value: number, digits = 3): string {
  return Number.isFinite(value) ? value.toFixed(digits) : '—';
}

function formatMegabytes(bytes: number): string {
  return (bytes / (1024 * 1024)).toFixed(1);
}

/** Formats cumulative streamed frame bytes using GB once past 1 GiB. */
function formatStreamedBytes(bytes: number): string {
  const megabytes = bytes / (1024 * 1024);
  return megabytes >= 1024
    ? `${(megabytes / 1024).toFixed(2)} GB`
    : `${megabytes.toFixed(1)} MB`;
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

/** Downloads the finalized muxed HDR video. */
function downloadMuxedVideo(video: MuxedVideo): void {
  const blob = new Blob([video.buffer], { type: video.mimeType });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download =
    `hdr-${video.frameCount}f-${video.width}x${video.height}-` +
    `${video.codec}-${video.transfer}${video.extension}`;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
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
  let muxedVideo: MuxedVideo | null = null;

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

  const encoderField = createElement('label', 'wb-field wb-field--inline');
  const encoderToggle = document.createElement('input');
  encoderToggle.type = 'checkbox';
  encoderToggle.className = 'wb-checkbox';
  encoderToggle.checked = false;
  encoderToggle.disabled = !webCodecsAvailable;
  if (!webCodecsAvailable) {
    encoderToggle.title = 'WebCodecs VideoEncoder is not available in this browser.';
  }
  encoderField.append(
    encoderToggle,
    createElement('span', 'wb-field-label', 'WebCodecs encode'),
  );

  const frameField = createElement('label', 'wb-field');
  const frameIndexInput = document.createElement('input');
  frameIndexInput.type = 'number';
  frameIndexInput.className = 'wb-number';
  frameIndexInput.min = '0';
  frameIndexInput.max = String(DEFAULT_FRAME_COUNT - 1);
  frameIndexInput.step = '1';
  frameIndexInput.value = '0';
  frameIndexInput.title = 'Frame copied out through onFrame for the .npy download';
  frameField.append(
    createElement('span', 'wb-field-label', 'Frame to keep'),
    frameIndexInput,
  );

  const downloadButton = createElement('button', 'wb-button', 'Download frame');
  downloadButton.type = 'button';
  downloadButton.disabled = true;
  downloadButton.title =
    'Download the kept frame as .npy (H×W×4 float32, linear Rec.2020)';

  const downloadVideoButton = createElement('button', 'wb-button', 'Download video');
  downloadVideoButton.type = 'button';
  downloadVideoButton.disabled = true;
  downloadVideoButton.title =
    'Download the muxed 10-bit HDR video (MP4 for HEVC/AV1, WebM for VP9)';

  controls.append(
    formatField,
    encoderField,
    frameField,
    startButton,
    cancelButton,
    downloadButton,
    downloadVideoButton,
  );
  header.append(headingGroup, stateBadge);

  // --- Preview stage ---------------------------------------------------------
  const stage = createElement('div', 'wb-stage');
  const canvas = createElement('canvas', 'wb-canvas');
  canvas.width = DEFAULT_WIDTH;
  canvas.height = DEFAULT_HEIGHT;
  stage.append(canvas);

  // --- Telemetry -------------------------------------------------------------
  const stats = createElement('dl', 'wb-stats');
  const frameStat = createStat('Frames rendered');
  const returnedStat = createStat('Frames returned');
  const elapsedStat = createStat('Elapsed');
  const timingStat = createStat('Frame time');
  const renderTimingStat = createStat('Render / preview');
  const readbackTimingStat = createStat('Readback / unpack');
  const encodeTimingStat = createStat('Encode (tap)');
  const peakStat = createStat('Peak (sampled)');
  const sampleStat = createStat('Center pixel RGBA');
  const readbackStat = createStat('Readback');
  const targetStat = createStat('Active HDR target');
  const retainedStat = createStat('Pipeline retained');
  const keptFrameStat = createStat('Kept frame');
  const videoStat = createStat('HDR video');
  const fpsStat = createStat('FPS');
  const encoderStat = createStat('VideoEncoder');
  const chunksStat = createStat('Encoded chunks');
  stats.append(
    frameStat.cell,
    returnedStat.cell,
    elapsedStat.cell,
    timingStat.cell,
    renderTimingStat.cell,
    readbackTimingStat.cell,
    encodeTimingStat.cell,
    peakStat.cell,
    sampleStat.cell,
    readbackStat.cell,
    targetStat.cell,
    retainedStat.cell,
    keptFrameStat.cell,
    videoStat.cell,
    fpsStat.cell,
    encoderStat.cell,
    chunksStat.cell,
  );

  const note = createElement(
    'p',
    'wb-note',
    'Frames are streamed: each Float32 frame (~14.75 MB at 1280×720) is delivered through ' +
      'onFrame and released afterwards; only the selected frame is copied for download, so a ' +
      '300-frame run never retains all frames (~4.4 GB if it did). “Download frame” saves the ' +
      'kept frame as .npy (H×W×4 float32, linear Rec.2020). Enable WebCodecs encode to pack ' +
      'each frame as I420P10 (RGBA fallback) and feed a Rec. 2100 PQ/HLG encoder; emitted ' +
      'chunks are muxed into a downloadable MP4/WebM with Rec.2020/PQ/HLG colour metadata. ' +
      '10-bit encoder input requires the browser to expose I420P10 VideoFrames; otherwise the ' +
      'tap falls back to 8-bit RGBA.',
  );

  // Expected capability fallbacks (currently the 8-bit RGBA input path) belong
  // here rather than in the warning banner: they reduce precision but are not
  // errors and cannot be avoided in current browser WebCodecs implementations.
  const encoderNote = createElement('p', 'wb-note');
  encoderNote.hidden = true;

  const warningEl = createElement('p', 'wb-warning');
  warningEl.hidden = true;

  container.append(header, stage, controls, stats, note, encoderNote, warningEl);
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
    keptFrameStat.value.textContent = '—';
    downloadButton.disabled = true;
  }

  function clearMuxedVideo(): void {
    muxedVideo = null;
    videoStat.value.textContent = '—';
    downloadVideoButton.disabled = true;
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

  function onDownloadVideoClick(): void {
    if (!muxedVideo) return;
    logInfo('workbench', 'Downloading muxed HDR video', {
      codec: muxedVideo.codec,
      container: muxedVideo.extension,
      frameCount: muxedVideo.frameCount,
      bytes: muxedVideo.buffer.byteLength,
    });
    downloadMuxedVideo(muxedVideo);
  }

  function onFrameIndexChange(): void {
    if (!running) clearRetainedFrame();
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
      status.frameFormat === 'RGBA' ? 'RGBA (8-bit fallback)' : status.frameFormat;
    encoderStat.value.textContent =
      `${status.state} · ${status.codec} · ${formatLabel} · ${status.colorSpace}`;
    chunksStat.value.textContent =
      `${status.framesSubmitted} sent · ${status.chunksEmitted} chunks · ` +
      `${formatMegabytes(status.chunkBytes)} MB · ${status.keyFrames} key · q${status.encodeQueueSize}`;
    if (status.lastError) {
      showWarning(`VideoEncoder: ${status.lastError}`);
    }
  }

  function updateTelemetry(snapshot: TelemetrySnapshot): void {
    // `framesCompleted` is a count (0..total), so the final sample reads
    // `300 / 300` rather than the last 0-based frame index (`299`).
    frameStat.value.textContent = `${snapshot.framesCompleted} / ${snapshot.framesTotal}`;
    elapsedStat.value.textContent = `${Math.round(snapshot.elapsedMs ?? 0)} ms`;

    timingStat.value.textContent =
      snapshot.framesCompleted > 0
        ? `${snapshot.lastFrameMs.toFixed(1)} ms · avg ${snapshot.averageFrameMs.toFixed(1)} ms`
        : '—';

    // Frames completed per second over the pipeline's sliding timing window.
    // Two decimals so small frame-to-frame jitter stays visible.
    fpsStat.value.textContent =
      snapshot.measuredFps > 0 ? snapshot.measuredFps.toFixed(2) : '—';

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

    const values = snapshot.lastFrameValues;
    peakStat.value.textContent = values
      ? `${formatFixed(values.min)} – ${formatFixed(values.max)}`
      : '—';
    sampleStat.value.textContent = values
      ? values.sample.map((channel) => formatFixed(channel, 2)).join(', ')
      : '—';

    readbackStat.value.textContent = `${formatMegabytes(snapshot.readbackBytes)} MB`;
    retainedStat.value.textContent = `${formatMegabytes(snapshot.retainedBytes)} MB`;
    if (snapshot.renderFormat) targetStat.value.textContent = snapshot.renderFormat;

    updateEncoderReadout(snapshot.encoder ?? null, encoderToggle.checked);

    if (snapshot.warnings.length > 0) {
      showWarning(snapshot.warnings.join('\n'));
    }
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
    encoderToggle.disabled = true;
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
    const wantEncoder = encoderToggle.checked && webCodecsAvailable;

    const requestedKeepIndex = Number.parseInt(frameIndexInput.value, 10);
    const keepFrameIndex = Number.isFinite(requestedKeepIndex)
      ? Math.min(config.frameCount - 1, Math.max(0, Math.floor(requestedKeepIndex)))
      : 0;
    frameIndexInput.max = String(config.frameCount - 1);
    frameIndexInput.value = String(keepFrameIndex);

    let returnedFrames = 0;
    let returnedBytes = 0;
    clearRetainedFrame();
    clearMuxedVideo();
    returnedStat.value.textContent = `0 / ${config.frameCount} · 0 MB`;
    videoStat.value.textContent = wantEncoder ? 'Pending…' : 'Off';

    logInfo('workbench', 'Starting run', {
      width: config.width,
      height: config.height,
      fps: config.fps,
      frameCount: config.frameCount,
      renderFormat: config.renderFormat,
      encoderRequested: wantEncoder,
      webCodecsAvailable,
      logging: describeLogging(),
    });

    // The preview swap chain is sized from the canvas drawing buffer, so it must
    // be set before the pipeline creates the preview renderer.
    canvas.width = config.width;
    canvas.height = config.height;
    targetStat.value.textContent = selection.renderFormat;
    encoderStat.value.textContent = wantEncoder ? 'Starting…' : 'Off';
    chunksStat.value.textContent = '—';
    encodeTimingStat.value.textContent = '—';
    encoderNote.hidden = true;
    encoderNote.textContent = '';

    let encoderTap: VideoEncoderTap | null = null;
    const videoChunks: CollectedVideoChunk[] = [];

    try {
      if (wantEncoder) {
        logInfo('workbench', 'Creating VideoEncoder tap', {
          width: config.width,
          height: config.height,
          fps: config.fps,
          transfer: config.outputColorSpace.transfer,
        });
        const encoderWatchdog = setInterval(() => {
          logWarn('workbench', 'VideoEncoder creation has not completed yet', {
            waitingMs: Math.round(performance.now() - runStart),
            pageVisibility:
              typeof document !== 'undefined' ? document.visibilityState : 'unknown',
          });
        }, 5000);
        try {
          encoderTap = await createVideoEncoderTap(
            createEncoderProbeMeta(config),
            (chunk, metadata) => {
              // Keep the encoded packets for muxing after the run. The chunk
              // objects own their data and remain valid after the callback.
              videoChunks.push({ chunk, metadata });
            },
            {
              // WebCodecs profiles are tagged Rec.2100 PQ or HLG; linear/unknown
              // output descriptions fall back to PQ for the encoder metadata.
              transfer: config.outputColorSpace.transfer === 'hlg' ? 'hlg' : 'pq',
              onWarning: (message) => {
                // The encoder tap already logs to the console; keep the fallback
                // reason available in the UI after later warnings replace it.
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
        } finally {
          clearInterval(encoderWatchdog);
        }

        logInfo(
          'workbench',
          encoderTap ? 'VideoEncoder tap ready' : 'VideoEncoder tap unavailable',
          encoderTap
            ? {
                codec: encoderTap.codec,
                frameFormat: encoderTap.frameFormat,
                state: encoderTap.status().state,
              }
            : undefined,
        );

        if (!encoderTap) {
          // `createVideoEncoderTap` reports the specific reason through
          // `onWarning`; only add a fallback when it was silent.
          logInfo('workbench', 'Continuing without a VideoEncoder tap');
          if (warningEl.hidden) {
            showWarning(
              'No supported 10-bit HDR VideoEncoder profile; continuing without encoding.',
            );
          }
          updateEncoderReadout(null, true);
        }
      }

      const summary = await runFramePipeline({
        config,
        previewCanvas: canvas,
        signal: controller.signal,
        encoderTap,
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
              keptFrameStat.value.textContent = `#${frame.meta.frameIndex} · ${checksum}`;
              downloadButton.disabled = false;
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

      if (summary.framesCompleted !== returnedFrames) {
        logWarn('workbench', 'onFrame count does not match completed frames', {
          onFrame: returnedFrames,
          framesCompleted: summary.framesCompleted,
        });
      }
      if (!retainedFrame) {
        keptFrameStat.value.textContent = `#${keepFrameIndex} not delivered`;
      }

      if (videoChunks.length > 0 && encoderTap) {
        videoStat.value.textContent = 'Muxing…';
        try {
          // Mediabunny is loaded on demand so a frame-only run never pays its
          // bundle cost; the muxer is only needed once encoded chunks exist.
          const { inspectMuxedVideo, muxVideoChunks } = await import(
            '../encoder/videoMuxer.ts'
          );
          muxedVideo = await muxVideoChunks({
            chunks: videoChunks,
            codec: encoderTap.codec,
            width: config.width,
            height: config.height,
            fps: config.fps,
            transfer: config.outputColorSpace.transfer === 'hlg' ? 'hlg' : 'pq',
          });
          if (muxedVideo) {
            const megabytes = (muxedVideo.buffer.byteLength / (1024 * 1024)).toFixed(2);
            let verification = '';
            try {
              const inspection = await inspectMuxedVideo(muxedVideo);
              logInfo('workbench', 'Muxed video self-check', {
                codec: inspection.codec,
                width: inspection.width,
                height: inspection.height,
                durationSeconds: inspection.durationSeconds,
                colorSpace: inspection.colorSpace,
                highDynamicRange: inspection.highDynamicRange,
              });
              const primaries = inspection.colorSpace.primaries ?? '?';
              const transfer = inspection.colorSpace.transfer ?? '?';
              verification =
                ` · ${primaries}/${transfer}` +
                (inspection.highDynamicRange ? ' · HDR ✓' : ' · HDR ?');
              if (!inspection.highDynamicRange) {
                logWarn('workbench', 'Muxed video did not report HDR metadata', {
                  colorSpace: inspection.colorSpace,
                  codec: inspection.codec,
                });
              }
            } catch (error) {
              logWarn('workbench', 'Muxed video self-check failed', {
                error: describeError(error),
              });
            }
            videoStat.value.textContent =
              `${muxedVideo.extension.slice(1)} · ${megabytes} MB · ` +
              `${muxedVideo.frameCount} chunks · ${muxedVideo.codec}/${muxedVideo.transfer}` +
              verification;
            downloadVideoButton.disabled = false;
          } else {
            videoStat.value.textContent = 'No encoded chunks';
          }
        } catch (error) {
          videoStat.value.textContent = 'Mux failed';
          logError('workbench', 'Muxing HDR video failed', error, {
            chunks: videoChunks.length,
            codec: encoderTap.codec,
          });
          showWarning(`Video mux failed: ${describeError(error)}`);
        }
      } else if (wantEncoder) {
        videoStat.value.textContent = 'No encoded chunks';
        logWarn('workbench', 'Encoder was requested but produced no chunks to mux');
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
    } catch (error) {
      logError('workbench', 'Run failed', error, {
        framesRequested: config.frameCount,
        elapsedMs: Math.round(performance.now() - runStart),
      });
      setState('error', 'Error');
      showWarning(`Run failed: ${describeError(error)}`);
    } finally {
      // Guarantees trailing chunks are emitted even if GPU setup failed before
      // the pipeline's own flush block was reached. Idempotent.
      await flushAndClose(encoderTap);
      updateEncoderReadout(encoderTap?.status() ?? null, wantEncoder);

      if (activeController === controller) activeController = null;
      running = false;
      if (!unmounted) {
        startButton.disabled = !webgpuAvailable;
        cancelButton.disabled = true;
        formatSelect.disabled = false;
        encoderToggle.disabled = !webCodecsAvailable;
        frameIndexInput.disabled = false;
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
    startButton.removeEventListener('click', onStartClick);
    cancelButton.removeEventListener('click', onCancelClick);
    encoderToggle.removeEventListener('change', onEncoderToggleChange);
    downloadButton.removeEventListener('click', onDownloadClick);
    downloadVideoButton.removeEventListener('click', onDownloadVideoClick);
    frameIndexInput.removeEventListener('change', onFrameIndexChange);
    container.remove();
  }

  function onStartClick(): void {
    void run();
  }

  function onCancelClick(): void {
    cancel();
  }

  function onEncoderToggleChange(): void {
    if (!running) {
      updateEncoderReadout(null, encoderToggle.checked);
      if (!encoderToggle.checked) {
        encoderNote.hidden = true;
        encoderNote.textContent = '';
        clearMuxedVideo();
        videoStat.value.textContent = 'Off';
      }
    }
  }

  startButton.addEventListener('click', onStartClick);
  cancelButton.addEventListener('click', onCancelClick);
  encoderToggle.addEventListener('change', onEncoderToggleChange);
  downloadButton.addEventListener('click', onDownloadClick);
  downloadVideoButton.addEventListener('click', onDownloadVideoClick);
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
    statusEl: stats,
    run,
    cancel,
    updateTelemetry,
    unmount,
  };
}
