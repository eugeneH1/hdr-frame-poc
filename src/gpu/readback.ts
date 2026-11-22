/**
 * GPU-to-CPU float buffer readback.
 *
 * Responsibility: copy the HDR target texture into a mappable staging buffer,
 * wait for `mapAsync`, strip WebGPU's row padding, decode to tightly packed
 * `Float32Array` RGBA pixels, and return a `FloatFrame`.
 *
 * Key assumptions:
 * - **Row alignment:** `copyTextureToBuffer` requires `bytesPerRow` to be a
 *   multiple of **256**. The tight stride is `width * bytesPerPixel` (10240 B
 *   for 1280 px `rgba16float`, 20480 B for `rgba32float`); both happen to be
 *   256-aligned at this width, but the code rounds up for general widths.
 * - **Formats:** `rgba16float` is decoded from half floats to 32-bit floats;
 *   `rgba32float` is copied directly. `FloatFrame.pixels` is always
 *   `Float32Array`, so a 1280x720 frame is ~14.75 MB of CPU memory.
 * - **Linear HDR:** values above `1.0` are preserved; readback never clamps.
 * - **Staging ring:** a configurable pool of `MAP_READ | COPY_DST` buffers is
 *   created up front (default 3). Each buffer is owned by exactly one copy/map
 *   job at a time, so GPU copies and `mapAsync` reads can overlap without
 *   buffer-state races. `beginFrameReadback` takes a buffer synchronously and
 *   submits the copy; `finish()` later awaits the map and decodes. The pipeline
 *   uses this to keep one copy in flight while unpacking the previous frame.
 * - **Cancellation:** waiting acquisitions accept an `AbortSignal`. A queued
 *   but unstarted job is cancelled by *draining* its copy (mapping once), which
 *   guarantees the GPU is done with the buffer before it returns to the ring.
 *   `dispose()` destroys all buffers, which rejects any pending maps safely.
 * - **Output reuse:** `reuseOutputBuffer` (default `false`) makes every frame
 *   share one CPU `Float32Array` instead of allocating ~14.75 MB per frame.
 *   The returned frame must be consumed and `release()`d before the next
 *   readback. A second readback while the previous shared frame is still live
 *   throws instead of silently clobbering it.
 * - The staging buffer is unmapped and recycled before the frame resolves, so
 *   `FloatFrame.release()` only releases the CPU payload / shared-output claim.
 * - **Ordering:** the caller must submit the render command buffer that drew
 *   into `source.target` before beginning a readback; the copy is submitted on
 *   its own encoder and cannot see unsubmitted render work.
 */

import type { GpuContext } from './context.ts';
import type { HdrRenderer } from './hdrRenderer.ts';
import type { FloatFrame, FrameMeta, ReadbackTiming, RenderConfig } from '../types/frame.ts';
import { BYTES_PER_PIXEL_F32 } from '../types/frame.ts';
import { GpuBufferUsage, GpuMapMode } from './webgpuFlags.ts';

/** Default number of staging buffers kept in the ring. */
const DEFAULT_STAGING_BUFFER_COUNT = 3;
/** Upper bound to keep accidental option values from allocating unbounded memory. */
const MAX_STAGING_BUFFER_COUNT = 8;
/** WebGPU `copyTextureToBuffer` row alignment requirement. */
const ROW_ALIGNMENT = 256;

/** Options for creating the staging ring and CPU-output policy. */
export interface ReadbackOptions {
  /**
   * Number of `MAP_READ | COPY_DST` staging buffers to allocate (default `3`,
   * clamped to `1..8`). More buffers allow more concurrent readbacks; the
   * pipeline overlaps one copy with the previous frame's unpack by default.
   */
  readonly stagingBufferCount?: number;
  /**
   * Reuse a single CPU `Float32Array` for every frame instead of allocating a
   * fresh ~14.75 MB array per frame. Default `false` (non-clobbering).
   *
   * When `true`, the previous `FloatFrame` must be fully consumed and released
   * before the next readback; otherwise `readFrameFloats` throws rather than
   * overwriting pixels that may still be in use.
   */
  readonly reuseOutputBuffer?: boolean;
}

/** Per-call options for `readFrameFloats`. */
export interface ReadFrameOptions {
  /** Cancels only the wait for a free staging buffer. */
  readonly signal?: AbortSignal;
}

/** Handle to the readback staging resources. */
export interface ReadbackManager {
  /** Actual GPU format being read back. */
  readonly format: GPUTextureFormat;
  /** Width in pixels this manager was sized for. */
  readonly width: number;
  /** Height in pixels this manager was sized for. */
  readonly height: number;
  /** Bytes per decoded pixel on the GPU (`8` for rgba16float, `16` for rgba32float). */
  readonly bytesPerPixel: number;
  /** Tight (unpadded) bytes per GPU pixel-row before CPU conversion. */
  readonly tightBytesPerRow: number;
  /** Padded bytes per GPU pixel-row used for the copy (multiple of 256). */
  readonly paddedBytesPerRow: number;
  /** Total padded staging-buffer size in bytes. */
  readonly bufferSize: number;
  /** Number of buffers in the staging ring. */
  readonly stagingBufferCount: number;
  /** Whether frames share one CPU `Float32Array`. */
  readonly reuseOutputBuffer: boolean;
  /** Staging buffers currently owned by a copy/map job (for diagnostics). */
  readonly activeReadbacks: number;
  /** Destroys staging buffers and rejects any queued acquisitions. Idempotent. */
  dispose(): void;
}

/**
 * A readback whose GPU copy has already been submitted.
 *
 * `beginFrameReadback` is synchronous so callers can submit the copy before
 * recording another frame into the same target texture. Awaiting `finish()`
 * later maps and decodes the pixels; `cancel()` drains the copy without
 * retaining a frame.
 */
export interface ReadbackJob {
  /** Frame index captured when the job was created. */
  readonly frameIndex: number;
  /** Staging buffer owned by this job until it finishes or is cancelled. */
  readonly stagingBufferIndex: number;
  /** Maps and decodes the frame. Idempotent; returns the same promise. */
  finish(): Promise<FloatFrame>;
  /**
   * Abandons the frame. The queued GPU copy is still drained (mapped once) so
   * the staging buffer is only recycled once the GPU is done with it. The
   * resulting rejection is handled internally.
   */
  cancel(): void;
}

interface StagingSlot {
  readonly index: number;
  readonly buffer: GPUBuffer;
  inUse: boolean;
}

interface Waiter {
  resolve(slot: StagingSlot): void;
  reject(reason: unknown): void;
}

/**
 * A submitted copy whose map has already been requested. Mapping immediately
 * after submission lets the copy resolve in the background while the pipeline
 * records/renders the next frame, and makes `waitMs` measure GPU copy cost rather
 * than the caller's deliberate lookahead delay.
 */
interface SlotCopy {
  readonly slot: StagingSlot;
  readonly mapStartedAt: number;
  readonly mapped: Promise<void>;
}

/**
 * Module-private pool/job operations. They live outside the public interface so
 * callers cannot misuse buffer ownership.
 */
interface ReadbackInternals {
  readonly device: GPUDevice;
  acquire(signal?: AbortSignal): Promise<StagingSlot>;
  tryAcquire(): StagingSlot;
  release(slot: StagingSlot): void;
  reserveOutput(): void;
  releaseOutput(): void;
  submitCopy(slot: StagingSlot, source: HdrRenderer): SlotCopy;
  complete(
    copy: SlotCopy,
    frameIndex: number,
    startedAt: number,
    isDiscarded: () => boolean,
  ): Promise<FloatFrame>;
}

const readbackInternals = new WeakMap<ReadbackManager, ReadbackInternals>();

/** Lazily built 65536-entry half-float lookup table (256 KB). */
let halfToFloatTable: Float32Array | null = null;

function decodeHalfBits(bits: number): number {
  const sign = (bits & 0x8000) !== 0 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const mantissa = bits & 0x3ff;

  if (exponent === 0) {
    // Subnormals (and signed zero).
    return sign * (mantissa / 1024) * 2 ** -14;
  }
  if (exponent === 0x1f) {
    return mantissa === 0 ? sign * Infinity : Number.NaN;
  }
  return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

function getHalfToFloatTable(): Float32Array {
  if (halfToFloatTable) return halfToFloatTable;
  const table = new Float32Array(0x10000);
  for (let bits = 0; bits < 0x10000; bits++) {
    table[bits] = decodeHalfBits(bits);
  }
  halfToFloatTable = table;
  return table;
}

function abortReason(signal: AbortSignal): unknown {
  return signal.reason ?? new DOMException('Readback aborted', 'AbortError');
}

function clampStagingCount(requested: number | undefined): number {
  if (requested === undefined) return DEFAULT_STAGING_BUFFER_COUNT;
  if (!Number.isFinite(requested)) return DEFAULT_STAGING_BUFFER_COUNT;
  return Math.min(MAX_STAGING_BUFFER_COUNT, Math.max(1, Math.floor(requested)));
}

/**
 * Creates the readback manager and its staging ring.
 *
 * @param context - Active GPU context (its negotiated format is read back).
 * @param config - Run configuration (dimensions determine buffer sizes).
 * @param options - Ring size and CPU output reuse policy.
 * @returns A ready `ReadbackManager`.
 */
export function createReadbackManager(
  context: GpuContext,
  config: RenderConfig,
  options: ReadbackOptions = {},
): ReadbackManager {
  const { device } = context;
  const format = context.renderFormat;
  const sourceFormat = format;

  const bytesPerPixel = format === 'rgba16float' ? 8 : format === 'rgba32float' ? 16 : 0;
  if (bytesPerPixel === 0) {
    throw new Error(
      `gpu.readback.createReadbackManager: unsupported format "${format}" (expected rgba16float or rgba32float)`,
    );
  }

  const width = config.width;
  const height = config.height;
  const tightBytesPerRow = width * bytesPerPixel;
  const paddedBytesPerRow = Math.ceil(tightBytesPerRow / ROW_ALIGNMENT) * ROW_ALIGNMENT;
  const bufferSize = paddedBytesPerRow * height;
  const stagingBufferCount = clampStagingCount(options.stagingBufferCount);
  const reuseOutputBuffer = options.reuseOutputBuffer === true;
  const outputPixelCount = width * height * 4;

  const slots: StagingSlot[] = [];
  for (let index = 0; index < stagingBufferCount; index++) {
    slots.push({
      index,
      buffer: device.createBuffer({
        label: `hdr-readback-staging-${index}`,
        size: bufferSize,
        usage: GpuBufferUsage.COPY_DST | GpuBufferUsage.MAP_READ,
      }),
      inUse: false,
    });
  }

  const waiters: Waiter[] = [];
  let disposed = false;
  let reuseReserved = false;
  // Allocate the shared payload up front so reuse mode performs no large
  // allocations during the run and telemetry's retained-bytes value is exact.
  const sharedOutput: Float32Array | null = reuseOutputBuffer
    ? new Float32Array(outputPixelCount)
    : null;

  function takeFreeSlot(): StagingSlot | null {
    for (const slot of slots) {
      if (!slot.inUse) {
        slot.inUse = true;
        return slot;
      }
    }
    return null;
  }

  function acquire(signal?: AbortSignal): Promise<StagingSlot> {
    if (disposed) {
      return Promise.reject(new Error('gpu.readback: manager has been disposed'));
    }
    if (signal?.aborted) {
      return Promise.reject(abortReason(signal));
    }

    const free = takeFreeSlot();
    if (free) return Promise.resolve(free);

    return new Promise<StagingSlot>((resolve, reject) => {
      let cleanedUp = false;
      const cleanup = (): void => {
        if (cleanedUp) return;
        cleanedUp = true;
        signal?.removeEventListener('abort', onAbort);
      };
      const waiter: Waiter = {
        resolve(slot) {
          cleanup();
          resolve(slot);
        },
        reject(reason) {
          cleanup();
          reject(reason);
        },
      };
      const onAbort = (): void => {
        const index = waiters.indexOf(waiter);
        if (index >= 0) waiters.splice(index, 1);
        waiter.reject(abortReason(signal!));
      };

      waiters.push(waiter);
      signal?.addEventListener('abort', onAbort, { once: true });
    });
  }

  function tryAcquire(): StagingSlot {
    if (disposed) {
      throw new Error('gpu.readback: manager has been disposed');
    }
    const slot = takeFreeSlot();
    if (!slot) {
      throw new Error(
        `gpu.readback: all ${stagingBufferCount} staging buffers are in use; ` +
          'finish or cancel an earlier readback before beginning another',
      );
    }
    return slot;
  }

  function release(slot: StagingSlot): void {
    if (disposed) {
      slot.inUse = false;
      slot.buffer.destroy();
      return;
    }

    // Defensive: a buffer must be unmapped before another copy can target it.
    if (slot.buffer.mapState === 'mapped') {
      try {
        slot.buffer.unmap();
      } catch {
        // Ignore: teardown or a concurrent map may already have unmapped it.
      }
    }

    const waiter = waiters.shift();
    if (waiter) {
      // Ownership transfers directly; the slot remains `inUse`.
      waiter.resolve(slot);
    } else {
      slot.inUse = false;
    }
  }

  function reserveOutput(): void {
    if (!reuseOutputBuffer) return;
    if (reuseReserved) {
      throw new Error(
        'gpu.readback: reuseOutputBuffer is enabled but the previous FloatFrame ' +
          'has not been released; consume and release it before the next readback',
      );
    }
    reuseReserved = true;
  }

  function releaseOutput(): void {
    if (reuseOutputBuffer) reuseReserved = false;
  }

  function extract(mapped: ArrayBuffer): Float32Array {
    const pixels = sharedOutput ?? new Float32Array(outputPixelCount);
    const tightStride = width * 4;

    if (format === 'rgba32float') {
      const source = new Float32Array(mapped);
      const sourceStride = paddedBytesPerRow / 4;
      for (let y = 0; y < height; y++) {
        const from = y * sourceStride;
        pixels.set(source.subarray(from, from + tightStride), y * tightStride);
      }
      return pixels;
    }

    // rgba16float: WebGPU returns IEEE 754 binary16 bit patterns. This MUST be
    // decoded through the lookup table; constructing a Float32Array over these
    // Uint16 bytes directly would reinterpret half-float bits as float32 and
    // produce garbage. `pixels` is the float32 payload consumers see.
    const table = getHalfToFloatTable();
    const source = new Uint16Array(mapped);
    const sourceStride = paddedBytesPerRow / 2;
    for (let y = 0; y < height; y++) {
      const from = y * sourceStride;
      const to = y * tightStride;
      for (let i = 0; i < tightStride; i++) {
        pixels[to + i] = table[source[from + i]];
      }
    }
    return pixels;
  }

  function submitCopy(slot: StagingSlot, source: HdrRenderer): SlotCopy {
    const encoder = device.createCommandEncoder({ label: `hdr-readback-${slot.index}-encoder` });
    encoder.copyTextureToBuffer(
      { texture: source.target, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
      {
        buffer: slot.buffer,
        offset: 0,
        bytesPerRow: paddedBytesPerRow,
        rowsPerImage: height,
      },
      { width, height, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);

    const mapStartedAt = performance.now();
    const mapped = slot.buffer.mapAsync(GpuMapMode.READ);
    // A job may be abandoned by its caller; mark the rejection handled while
    // still letting `complete()` observe and surface it.
    void mapped.catch(() => {});
    return { slot, mapStartedAt, mapped };
  }

  async function complete(
    copy: SlotCopy,
    frameIndex: number,
    startedAt: number,
    isDiscarded: () => boolean,
  ): Promise<FloatFrame> {
    const slot = copy.slot;
    let pixels: Float32Array | null = null;
    let waitMs = 0;
    let unpackMs = 0;

    try {
      await copy.mapped;
      const mappedAt = performance.now();
      waitMs = mappedAt - copy.mapStartedAt;

      if (isDiscarded()) {
        throw new Error('gpu.readback: readback cancelled');
      }

      try {
        pixels = extract(slot.buffer.getMappedRange());
      } finally {
        if (slot.buffer.mapState === 'mapped') {
          slot.buffer.unmap();
        }
      }
      unpackMs = performance.now() - mappedAt;
    } catch (error) {
      releaseOutput();
      throw error;
    } finally {
      release(slot);
    }

    if (!pixels) {
      releaseOutput();
      throw new Error('gpu.readback: map resolved without producing pixels');
    }

    const timestampUs = Math.round((frameIndex / config.fps) * 1e6);
    const durationUs = Math.round(1e6 / config.fps);

    const meta: FrameMeta = {
      frameIndex,
      timestampUs,
      durationUs,
      width,
      height,
      colorSpace: config.workingColorSpace,
      format: sourceFormat,
      renderTimeMs: 0,
    };

    const readback: ReadbackTiming = {
      waitMs,
      unpackMs,
      totalMs: performance.now() - startedAt,
      stagingBufferIndex: slot.index,
      reusedOutputBuffer: reuseOutputBuffer,
    };

    let released = false;

    return {
      meta,
      pixels,
      // CPU payload stride: RGBA float32 (16 bytes per pixel), independent of
      // the GPU source format. `tightBytesPerRow` is the GPU-side tight stride.
      bytesPerRow: width * BYTES_PER_PIXEL_F32,
      readback,
      release(): void {
        if (released) return;
        released = true;
        releaseOutput();
      },
    };
  }

  const manager: ReadbackManager = {
    format,
    width,
    height,
    bytesPerPixel,
    tightBytesPerRow,
    paddedBytesPerRow,
    bufferSize,
    stagingBufferCount,
    reuseOutputBuffer,
    get activeReadbacks(): number {
      let active = 0;
      for (const slot of slots) {
        if (slot.inUse) active++;
      }
      return active;
    },

    dispose(): void {
      if (disposed) return;
      disposed = true;
      reuseReserved = false;

      for (const waiter of waiters.splice(0)) {
        waiter.reject(
          new Error('gpu.readback: manager disposed while waiting for a staging buffer'),
        );
      }

      for (const slot of slots) {
        slot.inUse = false;
        try {
          if (slot.buffer.mapState === 'mapped') slot.buffer.unmap();
        } catch {
          // Ignore: a concurrent map may already have unmapped it.
        }
        slot.buffer.destroy();
      }
    },
  };

  readbackInternals.set(manager, {
    device,
    acquire,
    tryAcquire,
    release,
    reserveOutput,
    releaseOutput,
    submitCopy,
    complete,
  });
  return manager;
}

function getInternals(
  context: GpuContext,
  manager: ReadbackManager,
  source: HdrRenderer,
  frameIndex: number,
  config: RenderConfig,
): ReadbackInternals {
  if (frameIndex < 0 || frameIndex >= config.frameCount) {
    throw new RangeError(
      `gpu.readback: frameIndex ${frameIndex} is outside [0, ${config.frameCount})`,
    );
  }
  if (source.format !== manager.format) {
    throw new Error(
      `gpu.readback: source format "${source.format}" does not match manager "${manager.format}"`,
    );
  }
  if (config.width !== manager.width || config.height !== manager.height) {
    throw new Error(
      `gpu.readback: config ${config.width}x${config.height} does not match manager ` +
        `${manager.width}x${manager.height}`,
    );
  }

  const internals = readbackInternals.get(manager);
  if (!internals) {
    throw new Error('gpu.readback: manager was not created by createReadbackManager');
  }
  if (internals.device !== context.device) {
    throw new Error('gpu.readback: manager belongs to a different GPU device');
  }
  return internals;
}

/**
 * Copies the rendered HDR target to the CPU as a float frame.
 *
 * Acquires a staging buffer (waiting if the ring is saturated), submits a
 * `copyTextureToBuffer`, awaits `mapAsync`, strips row padding, converts half
 * floats when needed, and recycles the staging buffer before resolving.
 *
 * @param context - Active GPU context.
 * @param manager - Readback resources from `createReadbackManager`.
 * @param source - HDR renderer whose `target` texture is copied.
 * @param frameIndex - Frame being read back; used for timestamps/metadata.
 * @param config - Run configuration.
 * @param options - Optional abort signal for the pool wait.
 * @returns A `FloatFrame` whose pixels are valid until the caller drops it.
 */
export async function readFrameFloats(
  context: GpuContext,
  manager: ReadbackManager,
  source: HdrRenderer,
  frameIndex: number,
  config: RenderConfig,
  options: ReadFrameOptions = {},
): Promise<FloatFrame> {
  const internals = getInternals(context, manager, source, frameIndex, config);
  const startedAt = performance.now();

  internals.reserveOutput();

  let slot: StagingSlot;
  try {
    slot = await internals.acquire(options.signal);
  } catch (error) {
    internals.releaseOutput();
    throw error;
  }

  let copy: SlotCopy;
  try {
    copy = internals.submitCopy(slot, source);
  } catch (error) {
    internals.release(slot);
    internals.releaseOutput();
    throw error;
  }

  return internals.complete(copy, frameIndex, startedAt, () => false);
}

/**
 * Begins a pipelined readback: acquires a staging buffer synchronously, submits
 * the copy, and returns a job whose `finish()` resolves after mapping/unpacking.
 *
 * This is synchronous on purpose. The caller must submit the render pass that
 * produced `source.target` first, then call this function *before* recording
 * another pass into the same texture. Awaiting `finish()` later lets the GPU
 * copy overlap with subsequent CPU work.
 *
 * @param context - Active GPU context.
 * @param manager - Readback resources from `createReadbackManager`.
 * @param source - HDR renderer whose `target` texture is copied.
 * @param frameIndex - Frame being read back.
 * @param config - Run configuration.
 * @returns A submitted `ReadbackJob`.
 * @throws If the staging ring is exhausted.
 */
export function beginFrameReadback(
  context: GpuContext,
  manager: ReadbackManager,
  source: HdrRenderer,
  frameIndex: number,
  config: RenderConfig,
): ReadbackJob {
  const internals = getInternals(context, manager, source, frameIndex, config);
  const startedAt = performance.now();

  internals.reserveOutput();

  let slot: StagingSlot;
  try {
    slot = internals.tryAcquire();
  } catch (error) {
    internals.releaseOutput();
    throw error;
  }

  let copy: SlotCopy;
  try {
    copy = internals.submitCopy(slot, source);
  } catch (error) {
    internals.release(slot);
    internals.releaseOutput();
    throw error;
  }

  let finishPromise: Promise<FloatFrame> | null = null;
  let discarded = false;
  let settled = false;

  function finish(): Promise<FloatFrame> {
    if (finishPromise) return finishPromise;
    finishPromise = (async () => {
      try {
        const frame = await internals.complete(
          copy,
          frameIndex,
          startedAt,
          () => discarded,
        );
        settled = true;
        return frame;
      } catch (error) {
        settled = true;
        throw error;
      }
    })();
    // Prevent an unhandled rejection if the job is abandoned; callers that
    // await `finish()` still observe the error.
    void finishPromise.catch(() => {});
    return finishPromise;
  }

  function cancel(): void {
    if (settled) return;
    discarded = true;
    // The GPU copy is already queued, so the buffer can only be recycled after
    // a map completes. Drain it and discard the decoded pixels.
    void finish().catch(() => {
      // Cancellation is deliberately silent; finish()/cancel() callers own the
      // policy for reporting it.
    });
  }

  return {
    frameIndex,
    stagingBufferIndex: slot.index,
    finish,
    cancel,
  };
}

/**
 * Releases readback resources (compatibility wrapper for `manager.dispose()`).
 *
 * @param manager - Manager previously returned by `createReadbackManager`.
 */
export function disposeReadback(manager: ReadbackManager): void {
  manager.dispose();
}
