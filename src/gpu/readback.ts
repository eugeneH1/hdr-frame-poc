/**
 * GPU-to-CPU float buffer readback manager.
 *
 * Responsibility: copy the HDR target texture into a mappable buffer and hand back
 * `FloatFrame`s as `Uint16Array` half-float payloads. Manages double-buffered staging
 * so `mapAsync` from frame N overlaps rendering of frame N+1.
 *
 * Design assumptions (important for the next step):
 * - **Row alignment:** WebGPU requires `bytesPerRow` to be a multiple of 256 when
 *   copying `GPUTexture -> GPUBuffer`. For 1280px `rgba16float`, the tight row is
 *   10240 bytes, so the padded stride is 10240 (already a multiple of 256). General
 *   widths must round up to the next 256; readback strips padding back to a tight
 *   stride before producing `FloatFrame.pixels`.
 * - Buffers are created with `MAP_READ | COPY_DST` usage; the source texture needs
 *   `COPY_SRC` usage (allocated by `hdrRenderer`).
 * - Only one frame's bytes are materialized per call; callers must `release()` the
 *   frame to keep peak memory bounded rather than retaining all 300 frames.
 */

import type { GpuContext } from './context.ts';
import type { HdrRenderer } from './hdrRenderer.ts';
import type { FloatFrame, RenderConfig } from '../types/frame.ts';

/** Handle to the readback staging resources. */
export interface ReadbackManager {
  /** Tight (unpadded) bytes per pixel-row in the produced frames. */
  readonly tightBytesPerRow: number;
  /** Padded bytes per pixel-row used for the GPU copy (multiple of 256). */
  readonly paddedBytesPerRow: number;
  /** Total padded buffer size in bytes. */
  readonly bufferSize: number;
}

/**
 * Creates the readback manager and its staging buffers.
 *
 * Intended steps: compute tight and 256-aligned padded row strides, allocate one or
 * two `MAP_READ | COPY_DST` buffers of `paddedBytesPerRow * height`, and prepare a
 * second buffer for double-buffering.
 *
 * @param context - Active GPU context.
 * @param config - Run configuration (dimensions determine buffer sizes).
 * @returns A ready `ReadbackManager`.
 */
export function createReadbackManager(_context: GpuContext, _config: RenderConfig): ReadbackManager {
  throw new Error('gpu.readback.createReadbackManager: not implemented');
}

/**
 * Copies the rendered HDR target to the CPU as a float frame.
 *
 * Intended steps: record a `copyTextureToBuffer` using `paddedBytesPerRow`, submit,
 * `await buffer.mapAsync(GPUMapMode.READ)`, strip row padding into a tight
 * `Uint16Array`, build `FrameMeta` (timestamps from `frameIndex / fps`), and return
 * a `FloatFrame` whose `release()` unmaps the buffer.
 *
 * @param context - Active GPU context.
 * @param manager - Readback resources from `createReadbackManager`.
 * @param source - HDR renderer whose `target` texture is copied.
 * @param frameIndex - Frame being read back; used for timestamps/metadata.
 * @param config - Run configuration.
 * @returns A `FloatFrame` valid until `release()` is called.
 */
export async function readFrameFloats(
  _context: GpuContext,
  _manager: ReadbackManager,
  _source: HdrRenderer,
  _frameIndex: number,
  _config: RenderConfig,
): Promise<FloatFrame> {
  throw new Error('gpu.readback.readFrameFloats: not implemented');
}

/**
 * Releases readback resources.
 *
 * Intended behaviour: destroy staging buffers and drop pending mappings. Should be
 * idempotent and safe to call after a cancelled run.
 *
 * @param manager - Manager previously returned by `createReadbackManager`.
 */
export function disposeReadback(_manager: ReadbackManager): void {
  throw new Error('gpu.readback.disposeReadback: not implemented');
}
