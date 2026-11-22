/**
 * Preview canvas renderer.
 *
 * Responsibility: present the offscreen HDR frame on a `<canvas>` for live
 * feedback. The canvas is configured as a WebGPU swap chain, and a fullscreen
 * pass samples the linear Rec. 2020 HDR texture, tone-maps highlights for an SDR
 * display, and writes sRGB-encoded pixels.
 *
 * Design assumptions:
 * - The canvas context is configured with `context.preferredFormat` (typically
 *   `bgra8unorm`) and `alphaMode: 'opaque'`. Because preferred formats are
 *   non-sRGB, `sdrPreview.wgsl` performs the sRGB transfer encoding itself.
 * - `renderPreview` records into a caller-owned command encoder and does **not**
 *   submit. The pipeline batches the scene pass and preview pass into one
 *   submission.
 * - The HDR source texture view is stable for a run, so its bind group is cached
 *   and rebuilt only when a different view is supplied.
 */

import type { GpuContext } from './context.ts';
import { GpuShaderStage } from './webgpuFlags.ts';
import sdrPreviewShader from './shaders/sdrPreview.wgsl?raw';

/** Handle to the preview presentation resources. */
export interface PreviewRenderer {
  /** Canvas used for preview output. */
  readonly canvas: HTMLCanvasElement;
  /** Canvas drawing-buffer width in device pixels. */
  readonly width: number;
  /** Canvas drawing-buffer height in device pixels. */
  readonly height: number;
  /** Swap-chain format the canvas was configured with. */
  readonly format: GPUTextureFormat;
  /**
   * Records one fullscreen preview pass into `commandEncoder`, sampling the
   * supplied HDR texture view. Does not submit.
   */
  renderPreview(commandEncoder: GPUCommandEncoder, hdrTextureView: GPUTextureView): void;
  /** Releases preview resources and unconfigures the canvas context. Idempotent. */
  dispose(): void;
}

/**
 * Creates the preview renderer bound to an existing canvas.
 *
 * Configures `canvas.getContext('webgpu')`, compiles the tone-mapping shader, and
 * builds the presentation pipeline. Rendering happens later, one call per frame,
 * using the view supplied to `renderPreview`.
 *
 * @param context - Active GPU context.
 * @param canvas - Target canvas element, sized by the workbench before the run.
 * @returns A ready `PreviewRenderer`.
 * @throws If WebGPU cannot be obtained from the canvas or the canvas is zero-sized.
 */
export function createPreviewRenderer(
  context: GpuContext,
  canvas: HTMLCanvasElement,
): PreviewRenderer {
  const { device } = context;
  const format = context.preferredFormat;

  if (canvas.width <= 0 || canvas.height <= 0) {
    throw new Error(
      'gpu.previewRenderer.createPreviewRenderer: canvas must have a non-zero drawing buffer',
    );
  }

  // The bundled DOM typings expose `GPUCanvasContext` but do not add a
  // `getContext("webgpu")` overload, so the context id falls back to the generic
  // `RenderingContext` union. Narrow it explicitly.
  const canvasContext = canvas.getContext('webgpu') as unknown as GPUCanvasContext | null;
  if (!canvasContext) {
    throw new Error(
      'gpu.previewRenderer.createPreviewRenderer: could not obtain a WebGPU canvas context',
    );
  }

  canvasContext.configure({
    device,
    format,
    alphaMode: 'opaque',
  });

  const shaderModule = device.createShaderModule({
    label: 'sdr-preview-shader',
    code: sdrPreviewShader,
  });

  // `unfilterable-float` with `textureLoad` works for both rgba16float and
  // rgba32float; core WebGPU does not guarantee filterability for the latter.
  const bindGroupLayout = device.createBindGroupLayout({
    label: 'sdr-preview-bgl',
    entries: [
      {
        binding: 0,
        visibility: GpuShaderStage.FRAGMENT,
        texture: { sampleType: 'unfilterable-float' },
      },
    ],
  });

  const pipeline = device.createRenderPipeline({
    label: 'sdr-preview-pipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
    vertex: { module: shaderModule, entryPoint: 'vsPreview' },
    fragment: {
      module: shaderModule,
      entryPoint: 'fsPreview',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
  });

  let disposed = false;
  let cachedView: GPUTextureView | null = null;
  let cachedBindGroup: GPUBindGroup | null = null;

  function bindGroupFor(hdrTextureView: GPUTextureView): GPUBindGroup {
    if (cachedView === hdrTextureView && cachedBindGroup) {
      return cachedBindGroup;
    }
    cachedBindGroup = device.createBindGroup({
      label: 'sdr-preview-bind-group',
      layout: bindGroupLayout,
      entries: [{ binding: 0, resource: hdrTextureView }],
    });
    cachedView = hdrTextureView;
    return cachedBindGroup;
  }

  return {
    canvas,
    format,
    get width(): number {
      return canvas.width;
    },
    get height(): number {
      return canvas.height;
    },

    renderPreview(commandEncoder: GPUCommandEncoder, hdrTextureView: GPUTextureView): void {
      if (disposed) {
        throw new Error('gpu.previewRenderer.renderPreview: preview renderer has been disposed');
      }

      // Acquire the swap-chain texture before beginning the pass so any canvas
      // validation error surfaces without leaving an open render pass behind.
      const swapChainView = canvasContext.getCurrentTexture().createView();
      const bindGroup = bindGroupFor(hdrTextureView);

      const pass = commandEncoder.beginRenderPass({
        label: 'sdr-preview-pass',
        colorAttachments: [
          {
            view: swapChainView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });

      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    },

    dispose(): void {
      if (disposed) return;
      disposed = true;
      cachedBindGroup = null;
      cachedView = null;

      try {
        canvasContext.unconfigure();
      } catch {
        // A canvas may already be unconfigured or lost; teardown is best-effort.
      }
    },
  };
}
