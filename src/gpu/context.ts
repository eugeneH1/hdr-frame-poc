/**
 * WebGPU device/context lifecycle wrapper.
 *
 * Responsibility: own the single `GPUAdapter` / `GPUDevice` pair for the POC,
 * expose the preferred canvas format, and centralize device-lost handling and
 * teardown. Every other GPU module receives the `GpuContext` returned here.
 *
 * Design assumptions:
 * - Requires a browser with WebGPU enabled (`navigator.gpu`); callers must handle
 *   `initGpuContext` rejecting when unsupported.
 * - The context is created once per workbench mount and destroyed on unmount.
 */

import type { RenderConfig } from '../types/frame.ts';

/** Result of a successful WebGPU initialization. */
export interface GpuContext {
  /** Negotiated adapter (used for limits/features inspection). */
  readonly adapter: GPUAdapter;
  /** Logical device used for all resource creation and submissions. */
  readonly device: GPUDevice;
  /** Preferred swap-chain format for the preview canvas (e.g. `bgra8unorm`). */
  readonly preferredFormat: GPUTextureFormat;
  /** Format used for offscreen HDR render targets (usually `rgba16float`). */
  readonly renderFormat: GPUTextureFormat;
  /**
   * Subscribes to device-lost events. Returns an unsubscribe function so the
   * pipeline can detach before teardown.
   */
  onDeviceLost(handler: (reason: string) => void): () => void;
}

/**
 * Initializes WebGPU for the POC.
 *
 * Intended steps: request adapter, request device (requesting float-renderable
 * features/limits as needed for `rgba16float` targets), and read the preferred
 * canvas format. Should also verify the device supports `renderFormat`.
 *
 * @param config - Run configuration providing `renderFormat` and target sizes.
 * @returns A ready `GpuContext`.
 * @throws If WebGPU is unavailable or the required format/features are missing.
 */
export async function initGpuContext(_config: RenderConfig): Promise<GpuContext> {
  throw new Error('gpu.context.initGpuContext: not implemented');
}

/**
 * Tears down the GPU context.
 *
 * Intended behaviour: destroy device-owned resources (buffers, textures, pipelines)
 * created by child modules, remove event listeners, and let the device be GC'd.
 * Should be idempotent.
 *
 * @param context - Context previously returned by `initGpuContext`.
 */
export function destroyGpuContext(_context: GpuContext): void {
  throw new Error('gpu.context.destroyGpuContext: not implemented');
}
