/**
 * Preview canvas renderer.
 *
 * Responsibility: present the HDR frame on a `<canvas>` for live feedback. A full
 * screen/quad pass samples the offscreen `rgba16float` texture, applies the output
 * transfer function (Rec. 2100 PQ by default) and tone-maps/clamps for SDR canvas
 * presentation, then draws to the swap-chain texture.
 *
 * Design assumptions:
 * - Canvas context is configured with `format = context.preferredFormat` and
 *   `alphaMode: 'opaque'`.
 * - Preview is allowed to lag or skip frames under load; the pipeline treats a
 *   dropped preview as non-fatal (surfaced via `TelemetrySnapshot.warnings`).
 */

import type { GpuContext } from './context.ts';
import type { HdrRenderer } from './hdrRenderer.ts';
import type { ColorSpaceSpec } from '../types/frame.ts';

/** Handle to the preview presentation resources. */
export interface PreviewRenderer {
  /** Canvas used for preview output. */
  readonly canvas: HTMLCanvasElement;
  /** Canvas drawing-buffer width in device pixels. */
  readonly width: number;
  /** Canvas drawing-buffer height in device pixels. */
  readonly height: number;
}

/**
 * Creates the preview renderer bound to an existing canvas.
 *
 * Intended steps: obtain `getContext('webgpu')`, `configure({ device, format,
 * alphaMode: 'opaque' })`, compile the presentation shader (texture sample +
 * transfer-function encode), and create the swap pipeline.
 *
 * @param context - Active GPU context.
 * @param canvas - Target canvas element (sized by the workbench).
 * @param outputColorSpace - Delivery color space used for encoding the preview.
 * @returns A ready `PreviewRenderer`.
 */
export function createPreviewRenderer(
  _context: GpuContext,
  _canvas: HTMLCanvasElement,
  _outputColorSpace: ColorSpaceSpec,
): PreviewRenderer {
  throw new Error('gpu.previewRenderer.createPreviewRenderer: not implemented');
}

/**
 * Presents one rendered HDR frame to the preview canvas.
 *
 * Intended steps: acquire the current swap-chain texture, run a full-screen pass
 * sampling `source.target`, encode linear light with the output transfer function,
 * and submit. Swap-chain texture-view creation should happen per-present.
 *
 * @param context - Active GPU context.
 * @param preview - Preview resources from `createPreviewRenderer`.
 * @param source - The HDR renderer whose `target` texture should be shown.
 */
export function renderPreview(
  _context: GpuContext,
  _preview: PreviewRenderer,
  _source: HdrRenderer,
): void {
  throw new Error('gpu.previewRenderer.renderPreview: not implemented');
}
