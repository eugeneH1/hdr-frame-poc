/**
 * Offscreen HDR render target and scene renderer.
 *
 * Responsibility: own the `rgba16float`/`rgba32float` render target and the
 * pipelines/uniform buffers that draw an HDR background plus three transformed
 * shape quads. Rendering is linear and unbounded; no tone mapping or transfer
 * function is applied here.
 *
 * Key assumptions:
 * - **No clamping:** shader outputs and blending stay in linear Rec. 2020, so
 *   highlights above `1.0` survive in the float target until preview/encode.
 * - **Premultiplied alpha:** shape fragments emit `rgb * opacity` with alpha
 *   `opacity`, and the pipeline blends `one` / `one-minus-src-alpha`. This avoids
 *   dark fringes when translucent shapes composite over HDR values.
 * - **Coordinate spaces:** the CPU composes shape-local -> design pixels ->
 *   WebGPU NDC via `composeShapeClipMatrix`, so pixel orientation is resolved in
 *   one place (design space is top-left/+Y down; NDC is +Y up).
 * - **Format:** `context.renderFormat` is the negotiated format. `rgba16float`
 *   is always renderable/blendable; `rgba32float` requires the adapter's
 *   `float32-renderable` + `float32-blendable` features (see `context.ts`).
 * - **Uniform uploads:** `renderSceneToTexture` writes uniforms to the queue
 *   before recording the pass. Callers should submit that frame's command
 *   buffer before recording another frame, otherwise the newer uniform values
 *   would be observed by the earlier (still unsubmitted) draws.
 */

import type { GpuContext } from './context.ts';
import type { RenderConfig, SceneState } from '../types/frame.ts';
import { composeShapeClipMatrix, multiply, scale, toWgslMat4 } from '../math/mat3.ts';
import { GpuBufferUsage, GpuShaderStage, GpuTextureUsage } from './webgpuFlags.ts';
import hdrSceneShader from './shaders/hdrScene.wgsl?raw';

const SHAPE_COUNT = 3;
/** `minUniformBufferOffsetAlignment` default is 256 bytes; use it as stride. */
const SHAPE_UNIFORM_STRIDE = 256;
/** mat4x4<f32> (64 B) + vec4<f32> (16 B). */
const SHAPE_UNIFORM_BYTES = 80;
/** Two vec4<f32>: background `from` and `to`. */
const BACKGROUND_UNIFORM_BYTES = 32;

/** Handle to the HDR rendering resources. */
export interface HdrRenderer {
  /** Offscreen texture format in use (mirrors `context.renderFormat`). */
  readonly format: GPUTextureFormat;
  /** Texture width in pixels. */
  readonly width: number;
  /** Texture height in pixels. */
  readonly height: number;
  /** Underlying render target texture (exposed for `previewRenderer`/`readback`). */
  readonly target: GPUTexture;
  /**
   * Uploads one scene's uniforms and records the background + three shapes into
   * `commandEncoder`. Does **not** submit; the caller owns encoder lifecycle.
   */
  renderSceneToTexture(commandEncoder: GPUCommandEncoder, sceneState: SceneState): void;
  /** Destroys the target texture and uniform buffers. Idempotent. */
  dispose(): void;
}

/**
 * Creates the HDR renderer and its offscreen target.
 *
 * @param context - Active GPU context (its `renderFormat` is authoritative).
 * @param config - Run configuration (dimensions).
 * @returns A ready `HdrRenderer`.
 */
export function createHdrRenderer(context: GpuContext, config: RenderConfig): HdrRenderer {
  const { device } = context;
  const format = context.renderFormat;

  if (format !== 'rgba16float' && format !== 'rgba32float') {
    throw new Error(
      `gpu.hdrRenderer.createHdrRenderer: unsupported HDR target format "${format}"`,
    );
  }

  const width = config.width;
  const height = config.height;
  const viewport = { width, height };

  const target = device.createTexture({
    label: 'hdr-scene-target',
    size: { width, height },
    format,
    usage:
      GpuTextureUsage.RENDER_ATTACHMENT |
      GpuTextureUsage.COPY_SRC |
      GpuTextureUsage.TEXTURE_BINDING,
  });
  const targetView = target.createView();

  const shaderModule = device.createShaderModule({
    label: 'hdr-scene-shader',
    code: hdrSceneShader,
  });

  // --- Background pipeline ---------------------------------------------------
  const backgroundLayout = device.createBindGroupLayout({
    label: 'hdr-background-bgl',
    entries: [
      {
        binding: 0,
        visibility: GpuShaderStage.FRAGMENT,
        buffer: { type: 'uniform' },
      },
    ],
  });

  const backgroundPipeline = device.createRenderPipeline({
    label: 'hdr-background-pipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [backgroundLayout] }),
    vertex: { module: shaderModule, entryPoint: 'vsBackground' },
    fragment: {
      module: shaderModule,
      entryPoint: 'fsBackground',
      targets: [{ format }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // --- Shape pipeline (premultiplied alpha blending) -------------------------
  const shapeLayout = device.createBindGroupLayout({
    label: 'hdr-shape-bgl',
    entries: [
      {
        binding: 1,
        visibility: GpuShaderStage.VERTEX | GpuShaderStage.FRAGMENT,
        buffer: {
          type: 'uniform',
          hasDynamicOffset: true,
          minBindingSize: SHAPE_UNIFORM_BYTES,
        },
      },
    ],
  });

  const premultipliedBlend: GPUBlendState = {
    color: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
  };

  const shapePipeline = device.createRenderPipeline({
    label: 'hdr-shape-pipeline',
    layout: device.createPipelineLayout({ bindGroupLayouts: [shapeLayout] }),
    vertex: { module: shaderModule, entryPoint: 'vsShape' },
    fragment: {
      module: shaderModule,
      entryPoint: 'fsShape',
      targets: [{ format, blend: premultipliedBlend }],
    },
    primitive: { topology: 'triangle-list' },
  });

  // --- Uniform buffers and bind groups --------------------------------------
  const backgroundBuffer = device.createBuffer({
    label: 'hdr-background-uniforms',
    size: BACKGROUND_UNIFORM_BYTES,
    usage: GpuBufferUsage.UNIFORM | GpuBufferUsage.COPY_DST,
  });

  const backgroundBindGroup = device.createBindGroup({
    label: 'hdr-background-bind-group',
    layout: backgroundLayout,
    entries: [{ binding: 0, resource: { buffer: backgroundBuffer } }],
  });

  const shapeBuffer = device.createBuffer({
    label: 'hdr-shape-uniforms',
    size: SHAPE_UNIFORM_STRIDE * SHAPE_COUNT,
    usage: GpuBufferUsage.UNIFORM | GpuBufferUsage.COPY_DST,
  });

  const shapeBindGroup = device.createBindGroup({
    label: 'hdr-shape-bind-group',
    layout: shapeLayout,
    entries: [
      {
        binding: 1,
        resource: { buffer: shapeBuffer, offset: 0, size: SHAPE_UNIFORM_STRIDE },
      },
    ],
  });

  // Scratch arrays reused every frame to avoid per-frame allocation churn.
  const backgroundData = new Float32Array(BACKGROUND_UNIFORM_BYTES / 4);
  const shapeData = new Float32Array((SHAPE_UNIFORM_STRIDE / 4) * SHAPE_COUNT);

  let disposed = false;

  return {
    format,
    width,
    height,
    target,

    renderSceneToTexture(commandEncoder: GPUCommandEncoder, sceneState: SceneState): void {
      if (disposed) {
        throw new Error('gpu.hdrRenderer.renderSceneToTexture: renderer has been disposed');
      }

      // Background: linear Rec. 2020 `from`/`to` colors, including HDR values.
      const { background, backgroundTo } = sceneState;
      backgroundData[0] = background.r;
      backgroundData[1] = background.g;
      backgroundData[2] = background.b;
      backgroundData[3] = background.a;
      backgroundData[4] = backgroundTo.r;
      backgroundData[5] = backgroundTo.g;
      backgroundData[6] = backgroundTo.b;
      backgroundData[7] = backgroundTo.a;
      device.queue.writeBuffer(backgroundBuffer, 0, backgroundData);

      // Shapes: precompose local -> design pixels -> NDC, then scale the unit
      // quad (0..1) into the shape's local pixel extents.
      const shapes = sceneState.shapes;
      for (let i = 0; i < SHAPE_COUNT; i++) {
        const shape = shapes[i];
        const localToClip = composeShapeClipMatrix(
          shape.transform,
          { width: shape.size.x, height: shape.size.y },
          viewport,
        );
        const unitToClip = multiply(
          localToClip,
          scale(shape.size.x, shape.size.y),
        );
        const matrix = toWgslMat4(unitToClip);

        const base = (SHAPE_UNIFORM_STRIDE / 4) * i;
        shapeData.set(matrix, base);
        shapeData[base + 16] = shape.color.r;
        shapeData[base + 17] = shape.color.g;
        shapeData[base + 18] = shape.color.b;
        // Opacity is carried in alpha; the shader premultiplies by it.
        shapeData[base + 19] = shape.opacity;
      }
      device.queue.writeBuffer(shapeBuffer, 0, shapeData);

      const pass = commandEncoder.beginRenderPass({
        label: 'hdr-scene-pass',
        colorAttachments: [
          {
            view: targetView,
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      });

      // Opaque background covers the whole frame; blending is unnecessary.
      pass.setPipeline(backgroundPipeline);
      pass.setBindGroup(0, backgroundBindGroup);
      pass.draw(3);

      // Three translucent quads, back to front, with premultiplied blending.
      pass.setPipeline(shapePipeline);
      for (let i = 0; i < SHAPE_COUNT; i++) {
        pass.setBindGroup(0, shapeBindGroup, [SHAPE_UNIFORM_STRIDE * i]);
        pass.draw(6);
      }

      pass.end();
    },

    dispose(): void {
      if (disposed) return;
      disposed = true;
      backgroundBuffer.destroy();
      shapeBuffer.destroy();
      target.destroy();
    },
  };
}

/**
 * Renders one scene into the HDR target and submits the GPU work.
 *
 * Convenience wrapper around `renderer.renderSceneToTexture` for callers that do
 * not need to batch multiple passes into one command encoder.
 *
 * @param context - Active GPU context.
 * @param renderer - Renderer/target produced by `createHdrRenderer`.
 * @param scene - Fully evaluated scene for the current frame.
 * @returns The target texture holding the rendered linear HDR frame.
 */
export function renderHdrFrame(
  context: GpuContext,
  renderer: HdrRenderer,
  scene: SceneState,
): GPUTexture {
  const encoder = context.device.createCommandEncoder({ label: 'hdr-frame-encoder' });
  renderer.renderSceneToTexture(encoder, scene);
  context.device.queue.submit([encoder.finish()]);
  return renderer.target;
}
