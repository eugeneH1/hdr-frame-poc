/**
 * Offscreen HDR render target manager.
 *
 * Responsibility: own the render pipelines and the `rgba16float` texture that the
 * scene is drawn into each frame, plus the GPU-side buffer used for readback. The
 * target keeps full linear HDR values (no preview clamping to `[0, 1]`).
 *
 * Design assumptions:
 * - Output texture format is `config.renderFormat` (`rgba16float`), so highlights
 *   above 1.0 survive until transfer-function encoding at preview/encode time.
 * - Ping-ponging between two textures is preferred over waiting on the previous
 *   frame, keeping the CPU pipeline from stalling on `mapAsync` (see `readback.ts`).
 */

import type { GpuContext } from './context.ts';
import type { RenderConfig, SceneState } from '../types/frame.ts';

/** Handle to the HDR rendering resources. */
export interface HdrRenderer {
  /** Offscreen texture format in use (mirrors `config.renderFormat`). */
  readonly format: GPUTextureFormat;
  /** Texture width in pixels. */
  readonly width: number;
  /** Texture height in pixels. */
  readonly height: number;
  /** Underlying render target texture (exposed for `previewRenderer`/`readback`). */
  readonly target: GPUTexture;
}

/**
 * Creates the HDR renderer and its offscreen target.
 *
 * Intended steps: compile the shape/background WGSL shaders, build the render
 * pipeline against `config.renderFormat`, and allocate the target texture plus
 * any uniform buffers (per-shape transforms/colors).
 *
 * @param context - Active GPU context.
 * @param config - Run configuration (dimensions, format, color spaces).
 * @returns A ready `HdrRenderer`.
 */
export function createHdrRenderer(_context: GpuContext, _config: RenderConfig): HdrRenderer {
  throw new Error('gpu.hdrRenderer.createHdrRenderer: not implemented');
}

/**
 * Renders one scene into the HDR target.
 *
 * Intended steps: upload `scene` uniforms (background colors, three shape
 * transforms via `composeShapeMatrix`, linear colors, opacities), begin a render
 * pass clearing to the background gradient, draw the three shapes, and leave the
 * result in the target texture (linear, unbounded).
 *
 * @param context - Active GPU context.
 * @param renderer - Renderer/target produced by `createHdrRenderer`.
 * @param scene - Fully evaluated scene for the current frame.
 * @returns The target texture holding the rendered linear HDR frame.
 */
export function renderHdrFrame(
  _context: GpuContext,
  _renderer: HdrRenderer,
  _scene: SceneState,
): GPUTexture {
  throw new Error('gpu.hdrRenderer.renderHdrFrame: not implemented');
}
