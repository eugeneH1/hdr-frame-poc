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
 * - **Staging pool:** two `MAP_READ | COPY_DST` buffers are reused across
 *   frames (double buffering), avoiding a fresh allocation per frame. A third
 *   concurrent call waits until one is unmapped and returned.
 * - The staging buffer is unmapped and recycled before `readFrameFloats`
 *   resolves, so `FloatFrame.release()` only marks the CPU payload as consumed;
 *   the pixels remain readable until the caller drops the frame.
 * - **Ordering:** the caller must submit the render command buffer that drew
 *   into `source.target` before calling `readFrameFloats`; the copy is submitted
 *   on its own encoder and cannot see unsubmitted render work.
 */

import type { GpuContext } from './context.ts';
import type { HdrRenderer } from './hdrRenderer.ts';
import type { FloatFrame, FrameMeta, RenderConfig } from '../types/frame.ts';
import { BYTES_PER_PIXEL_F32 } from '../types/frame.ts';
import { GpuBufferUsage, GpuMapMode } from './webgpuFlags.ts';

/** Number of staging buffers kept for overlapping readbacks. */
const STAGING_POOL_SIZE = 2;
/** WebGPU `copyTextureToBuffer` row alignment requirement. */
const ROW_ALIGNMENT = 256;

/** Handle to the readback staging resources. */
export interface ReadbackManager {
  /** Actual GPU format being read back. */
  readonly format: GPUTextureFormat;
  /** Bytes per decoded pixel on the GPU (`8` for rgba16float, `16` for rgba32float). */
  readonly bytesPerPixel: number;
  /** Tight (unpadded) bytes per GPU pixel-row before CPU conversion. */
  readonly tightBytesPerRow: number;
  /** Padded bytes per GPU pixel-row used for the copy (multiple of 256). */
  readonly paddedBytesPerRow: number;
  /** Total padded staging-buffer size in bytes. */
  readonly bufferSize: number;
  /** Destroys staging buffers and rejects any queued acquisitions. Idempotent. */
  dispose(): void;
}

/**
 * Module-private pool operations. They live outside the public interface so
 * callers cannot misuse buffer ownership, while `readFrameFloats` can still
 * acquire/recycle staging memory.
 */
interface ReadbackInternals {
  acquire(): Promise<GPUBuffer>;
  release(buffer: GPUBuffer): void;
  extract(mapped: ArrayBuffer): Float32Array;
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

/**
 * Creates the readback manager and its staging-buffer pool.
 *
 * @param context - Active GPU context (its negotiated format is read back).
 * @param config - Run configuration (dimensions determine buffer sizes).
 * @returns A ready `ReadbackManager`.
 */
export function createReadbackManager(context: GpuContext, config: RenderConfig): ReadbackManager {
  const { device } = context;
  const format = context.renderFormat;

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

  const pool: GPUBuffer[] = [];
  const free: GPUBuffer[] = [];
  const waiters: Array<{
    resolve: (buffer: GPUBuffer) => void;
    reject: (reason: unknown) => void;
  }> = [];
  let disposed = false;

  function createStagingBuffer(): GPUBuffer {
    return device.createBuffer({
      label: 'hdr-readback-staging',
      size: bufferSize,
      usage: GpuBufferUsage.COPY_DST | GpuBufferUsage.MAP_READ,
    });
  }

  function acquire(): Promise<GPUBuffer> {
    if (disposed) {
      return Promise.reject(new Error('gpu.readback: manager has been disposed'));
    }
    const reusable = free.pop();
    if (reusable) return Promise.resolve(reusable);
    if (pool.length < STAGING_POOL_SIZE) {
      const created = createStagingBuffer();
      pool.push(created);
      return Promise.resolve(created);
    }
    return new Promise<GPUBuffer>((resolve, reject) => {
      waiters.push({ resolve, reject });
    });
  }

  function release(buffer: GPUBuffer): void {
    if (disposed) {
      buffer.destroy();
      return;
    }
    const waiter = waiters.shift();
    if (waiter) {
      waiter.resolve(buffer);
    } else {
      free.push(buffer);
    }
  }

  function extract(mapped: ArrayBuffer): Float32Array {
    const pixels = new Float32Array(width * height * 4);
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

  const manager: ReadbackManager = {
    format,
    bytesPerPixel,
    tightBytesPerRow,
    paddedBytesPerRow,
    bufferSize,

    dispose(): void {
      if (disposed) return;
      disposed = true;

      for (const waiter of waiters) {
        waiter.reject(
          new Error('gpu.readback: manager disposed while waiting for a staging buffer'),
        );
      }
      waiters.length = 0;

      for (const buffer of pool) {
        if (buffer.mapState === 'mapped') {
          try {
            buffer.unmap();
          } catch {
            // Ignore: a concurrent call may already have unmapped it.
          }
        }
        buffer.destroy();
      }
      pool.length = 0;
      free.length = 0;
    },
  };

  readbackInternals.set(manager, { acquire, release, extract });
  return manager;
}

/**
 * Copies the rendered HDR target to the CPU as a float frame.
 *
 * Records and submits a `copyTextureToBuffer`, awaits `mapAsync`, strips row
 * padding, converts half floats when needed, and unmaps/recycles the staging
 * buffer before resolving.
 *
 * @param context - Active GPU context.
 * @param manager - Readback resources from `createReadbackManager`.
 * @param source - HDR renderer whose `target` texture is copied.
 * @param frameIndex - Frame being read back; used for timestamps/metadata.
 * @param config - Run configuration.
 * @returns A `FloatFrame` whose pixels are valid until the caller drops it.
 */
export async function readFrameFloats(
  context: GpuContext,
  manager: ReadbackManager,
  source: HdrRenderer,
  frameIndex: number,
  config: RenderConfig,
): Promise<FloatFrame> {
  if (frameIndex < 0 || frameIndex >= config.frameCount) {
    throw new RangeError(
      `gpu.readback.readFrameFloats: frameIndex ${frameIndex} is outside [0, ${config.frameCount})`,
    );
  }
  if (source.format !== manager.format) {
    throw new Error(
      `gpu.readback.readFrameFloats: source format "${source.format}" does not match manager "${manager.format}"`,
    );
  }

  const internals = readbackInternals.get(manager);
  if (!internals) {
    throw new Error('gpu.readback.readFrameFloats: manager was not created by createReadbackManager');
  }

  const { device } = context;
  const buffer = await internals.acquire();
  let pixels: Float32Array;

  try {
    const encoder = device.createCommandEncoder({ label: 'hdr-readback-encoder' });
    encoder.copyTextureToBuffer(
      { texture: source.target, mipLevel: 0, origin: { x: 0, y: 0, z: 0 } },
      {
        buffer,
        offset: 0,
        bytesPerRow: manager.paddedBytesPerRow,
        rowsPerImage: config.height,
      },
      { width: config.width, height: config.height, depthOrArrayLayers: 1 },
    );
    device.queue.submit([encoder.finish()]);

    await buffer.mapAsync(GpuMapMode.READ);
    try {
      pixels = internals.extract(buffer.getMappedRange());
    } finally {
      buffer.unmap();
    }
  } finally {
    internals.release(buffer);
  }

  const timestampUs = Math.round((frameIndex / config.fps) * 1e6);
  const durationUs = Math.round(1e6 / config.fps);

  const meta: FrameMeta = {
    frameIndex,
    timestampUs,
    durationUs,
    width: config.width,
    height: config.height,
    colorSpace: config.workingColorSpace,
    format: source.format,
    renderTimeMs: 0,
  };

  let released = false;

  return {
    meta,
    pixels,
    // CPU payload stride: RGBA float32 (16 bytes per pixel), independent of the
    // GPU source format. `manager.tightBytesPerRow` is the GPU-side tight stride.
    bytesPerRow: config.width * BYTES_PER_PIXEL_F32,
    release(): void {
      if (released) return;
      released = true;
      // Staging memory was already unmapped and returned to the pool; the CPU
      // Float32Array is owned by the caller and GC'd when dropped.
    },
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
