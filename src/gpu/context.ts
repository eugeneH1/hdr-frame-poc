/**
 * WebGPU device/context lifecycle wrapper.
 *
 * Responsibility: own the single `GPUAdapter` / `GPUDevice` pair for the POC,
 * expose the preferred canvas format, centralize device-lost handling, and
 * provide an idempotent teardown. Every other GPU module receives the
 * `GpuContext` returned here.
 *
 * Format negotiation:
 * - `rgba16float` is renderable **and blendable** in core WebGPU, so it is the
 *   safe default for an HDR target (including premultiplied-alpha blending).
 * - `rgba32float` is only usable as a render attachment when the adapter
 *   exposes the `float32-renderable` feature, and only blendable with
 *   `float32-blendable`. Because the scene blends translucent shapes, both are
 *   required before we select `rgba32float`; otherwise we fall back cleanly to
 *   `rgba16float`.
 * - The chosen format is exposed as `GpuContext.renderFormat`; callers should
 *   use it (not `config.renderFormat`) so the fallback is honoured.
 */

import type { RenderConfig } from '../types/frame.ts';
import { logDebug, logError, logInfo, logWarn } from '../diagnostics/logger.ts';

/** Result of a successful WebGPU initialization. */
export interface GpuContext {
  /** Negotiated adapter (used for limits/features inspection). */
  readonly adapter: GPUAdapter;
  /** Logical device used for all resource creation and submissions. */
  readonly device: GPUDevice;
  /** Preferred swap-chain format for the preview canvas (e.g. `bgra8unorm`). */
  readonly preferredFormat: GPUTextureFormat;
  /**
   * Format actually used for offscreen HDR render targets. Equals
   * `config.renderFormat` when supported, or `rgba16float` on fallback.
   */
  readonly renderFormat: GPUTextureFormat;
  /**
   * Subscribes to device-lost events. Returns an unsubscribe function so the
   * pipeline can detach before teardown. Handlers also fire if the device is
   * lost unexpectedly.
   */
  onDeviceLost(handler: (reason: string) => void): () => void;
}

interface ContextState {
  readonly device: GPUDevice;
  readonly lossHandlers: Set<(reason: string) => void>;
  destroyed: boolean;
}

/**
 * Per-context mutable state. Kept out of the public (readonly) interface while
 * still letting `destroyGpuContext` clear listeners and destroy the device.
 */
const contextState = new WeakMap<GpuContext, ContextState>();

const FEATURE_FLOAT32_RENDERABLE = 'float32-renderable' as GPUFeatureName;
const FEATURE_FLOAT32_BLENDABLE = 'float32-blendable' as GPUFeatureName;

/** Structural view of `GPUAdapterInfo`; older DOM typings omit the `info` field. */
interface AdapterInfoLike {
  readonly vendor?: string;
  readonly architecture?: string;
  readonly device?: string;
  readonly description?: string;
}

/** Structural view of a `GPUError` raised as an `uncapturederror` event. */
interface GpuErrorLike {
  readonly type?: string;
  readonly message?: string;
}

function describeAdapter(adapter: GPUAdapter): Record<string, unknown> {
  const info = (adapter as unknown as { info?: AdapterInfoLike }).info;
  return {
    vendor: info?.vendor ?? 'unknown',
    architecture: info?.architecture ?? 'unknown',
    device: info?.device ?? 'unknown',
    description: info?.description ?? 'unknown',
    features: Array.from(adapter.features),
    maxBufferSize: adapter.limits.maxBufferSize,
    maxTextureDimension2D: adapter.limits.maxTextureDimension2D,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
}

/**
 * Initializes WebGPU for the POC.
 *
 * Requests a high-performance adapter, falling back to the platform default if
 * none is available, then requests a device with only the optional features the
 * adapter actually exposes. If `rgba32float` cannot be rendered *and* blended,
 * the context reports `rgba16float` instead.
 *
 * @param config - Run configuration providing `renderFormat` and target sizes.
 * @returns A ready `GpuContext`.
 * @throws If WebGPU is unavailable or the required format/features are missing.
 */
export async function initGpuContext(config: RenderConfig): Promise<GpuContext> {
  if (typeof navigator === 'undefined' || !navigator.gpu) {
    throw new Error('gpu.context.initGpuContext: WebGPU is not available in this environment');
  }

  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch {
    // Some implementations reject unknown power preferences; retry with defaults.
  }
  if (!adapter) {
    adapter = await navigator.gpu.requestAdapter();
  }
  if (!adapter) {
    throw new Error('gpu.context.initGpuContext: no suitable GPUAdapter was found');
  }

  logInfo('gpu', 'Adapter acquired', describeAdapter(adapter));

  const supportedFeatures = new Set<string>(Array.from(adapter.features));
  const requiredFeatures: GPUFeatureName[] = [];
  let renderFormat: GPUTextureFormat = 'rgba16float';

  if (config.renderFormat === 'rgba32float') {
    const canRender32 = supportedFeatures.has(FEATURE_FLOAT32_RENDERABLE);
    const canBlend32 = supportedFeatures.has(FEATURE_FLOAT32_BLENDABLE);
    if (canRender32 && canBlend32) {
      renderFormat = 'rgba32float';
      requiredFeatures.push(FEATURE_FLOAT32_RENDERABLE, FEATURE_FLOAT32_BLENDABLE);
    } else {
      // Otherwise fall through to rgba16float (core-renderable and blendable).
      logWarn('gpu', 'Requested rgba32float is not fully supported; using rgba16float', {
        float32Renderable: canRender32,
        float32Blendable: canBlend32,
      });
    }
  } else if (config.renderFormat === 'rgba16float') {
    renderFormat = 'rgba16float';
  } else {
    // The POC keeps HDR values above 1.0, which only float targets preserve.
    // Any non-float request falls back to the universally supported HDR format.
    renderFormat = 'rgba16float';
  }

  let device: GPUDevice;
  try {
    device = await adapter.requestDevice({ requiredFeatures });
  } catch (error) {
    if (renderFormat === 'rgba16float') {
      logError('gpu', 'requestDevice failed', error, { requiredFeatures });
      throw error;
    }
    logWarn('gpu', 'requestDevice with rgba32float features failed; falling back to rgba16float', {
      error: error instanceof Error ? error.message : String(error),
      requiredFeatures,
    });
    // Feature request or format validation failed; retry with the universally
    // supported HDR format rather than failing the whole run.
    renderFormat = 'rgba16float';
    device = await adapter.requestDevice();
  }

  const lossHandlers = new Set<(reason: string) => void>();

  // WebGPU validation/OOM failures are reported as events rather than thrown
  // exceptions. Without this listener they only appear in the browser's own
  // console formatting, which makes a stalling run look error-free.
  device.addEventListener('uncapturederror', (event) => {
    const gpuError = (event as unknown as { error?: GpuErrorLike }).error;
    logError('gpu', 'Uncaptured WebGPU error', gpuError, {
      type: gpuError?.type ?? 'unknown',
      message: gpuError?.message ?? '(no message)',
    });
  });

  // Surface unexpected device loss to subscribers. `destroyGpuContext` clears
  // the set first, so a deliberate destroy does not notify handlers.
  void device.lost.then((info) => {
    const state = contextState.get(context);
    const expected = state === undefined || state.destroyed;
    const lossDetails = {
      reason: info.reason,
      message: info.message || '(none)',
    };

    if (expected) {
      logDebug('gpu', 'GPU device destroyed', lossDetails);
      return;
    }

    // A lost device stops making progress but does not reject every in-flight
    // operation on all implementations, so make the cause explicit.
    logError('gpu', `GPU device lost (${info.reason})`, undefined, lossDetails);

    const reason = info.message || info.reason;
    for (const handler of lossHandlers) {
      try {
        handler(reason);
      } catch {
        // A broken telemetry handler must not mask the device-lost event.
      }
    }
    lossHandlers.clear();
  });

  const context: GpuContext = {
    adapter,
    device,
    preferredFormat: navigator.gpu.getPreferredCanvasFormat(),
    renderFormat,
    onDeviceLost(handler) {
      if (contextState.get(context)?.destroyed) return () => {};
      lossHandlers.add(handler);
      return () => {
        lossHandlers.delete(handler);
      };
    },
  };

  contextState.set(context, { device, lossHandlers, destroyed: false });
  logInfo('gpu', 'WebGPU device ready', {
    requestedFormat: config.renderFormat,
    renderFormat,
    requiredFeatures,
    preferredCanvasFormat: context.preferredFormat,
  });
  return context;
}

/**
 * Tears down the GPU context.
 *
 * Destroys the device (which invalidates all device-owned resources) and clears
 * loss listeners. Idempotent: repeated calls are no-ops, which makes cleanup
 * safe after cancellation or an error path.
 *
 * @param context - Context previously returned by `initGpuContext`.
 */
export function destroyGpuContext(context: GpuContext): void {
  const state = contextState.get(context);
  if (!state || state.destroyed) return;

  state.destroyed = true;
  state.lossHandlers.clear();
  logDebug('gpu', 'Destroying GPU device');
  state.device.destroy();
  contextState.delete(context);
}
