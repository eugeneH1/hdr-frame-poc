# hdr-frame-poc

> **Note:** this README is working notes, not a specification. It records what
> was built and attempted during development and is expected to drift out of
> date. Do not treat it as a guide to how this project should work — the
> requirements live in `Assignment.md`. This file will be rewritten before it is
> submitted as the assignment's README.

## Diagnostics

The pipeline emits verbose console diagnostics by default. Lines are prefixed
with `[hdr-frame-poc:<scope> +<seconds>s]`, where the scope is one of `pipeline`,
`gpu`, `readback`, `encoder`, `workbench`, or `app`.

While a run is active you get:

- One line per frame at each stage (`begin`, `waiting for readback`,
  `submitted`, `delivered`) with timings, staging-buffer index, readback state,
  encoder queue depth, and progress (`n/300`).
- Explicit logs for GPU device loss, uncaptured WebGPU errors, codec selection,
  encoder errors, cancellation, page-visibility changes, and the final
  `framesCompleted / framesTotal` summary.
- A stall watchdog that repeats the pipeline's state every 5 seconds whenever a
  readback map, encoder flush, GPU initialization, or consumer callback has not
  settled, so a hung run shows the last frame and the reason it is waiting.

Verbose output (the `info`/`debug` lines) can be turned off with `?verbose=0`
in the URL; the preference is persisted in `localStorage` under
`hdr-frame-poc.verbose`. Use `?verbose=1` to force it back on. Warnings and
errors are always logged.

## Encoder input precision

WebCodecs' `VideoFrame(BufferSource, …)` constructor only accepts the 8-bit
image formats in current browsers (`I420`, `NV12`, `RGBA`, `BGRA`, …). The tap
probes `I420P10` first, then falls back to 8-bit planar `I420`, and only uses
`RGBA` if `I420` is also unavailable. `I420` is preferred over `RGBA` because the
tap performs the Rec. 2020 YUV conversion and limited-range quantization itself
(Y 16..235, C 16..240) instead of leaving an RGB-to-YUV conversion to the codec,
and because the input is 1.5 bytes/pixel instead of 4. This is expected, not an
error: the encoder is still configured for a 10-bit Rec. 2100 profile and the
output carries PQ/HLG metadata, but the source pixels are quantized to 8 bits,
so smooth HDR gradients can band.

True 10-bit input needs a different path, such as a native/WASM encoder that
accepts 16-bit samples, or 10-bit frames decoded from an external source;
WebCodecs does not currently expose 10-bit CPU buffer frames to JavaScript.
Both 8-bit paths apply an 8x8 ordered (Bayer) dither before quantization, which
turns band edges into fine texture rather than hard steps.

## Implementation notes (draft)

These notes capture decisions and known limitations while they are fresh; this
section will be expanded into the final README.

### Rendering approach

- The timeline is a pure function of `(frameIndex, RenderConfig)`. Every frame
  is derived from `frameIndex / fps`, so a slow preview or encoder cannot alter
  frame content. Integer animation cycles make the 10 s loop seamless.
- WebGPU renders a full-frame HDR gradient plus three 100x100 quads into an
  offscreen `rgba16float` target (or `rgba32float` when the adapter exposes
  `float32-renderable` + `float32-blendable`). Colors are authored directly in
  linear Rec. 2020; values above 1.0 are intentional HDR highlights.
- The target is copied to a `MAP_READ` staging buffer, half floats are decoded,
  and tightly packed `Float32Array` RGBA frames are delivered through
  `onFrame`. A bounded staging ring (default 3 buffers) and lookahead of 2 keep
  at most two CPU frames live while overlapping copy/unpack with the next render.
- An SDR preview pass samples the HDR target, applies an ACES-style curve and
  sRGB encoding, and presents it on the canvas while frames are produced.
- An optional WebCodecs tap converts each float frame to `I420P10` (or 8-bit
  `I420`/`RGBA` fallback) and feeds a 10-bit Rec. 2100 PQ/HLG `VideoEncoder`.

### Decisions to explain in the final README

- **WebGPU over WebGL/Canvas 2D:** first-class float render targets, explicit
  texture formats, deterministic command ordering, and no 8-bit canvas
  intermediate for the HDR render.
- **CPU readback rather than GPU-direct encode:** WebCodecs does not accept a
  `GPUTexture` as encoder input, so frames must be copied to CPU memory.
- **Streaming frames:** retaining all 300 frames at 1280x720 would cost about
  4.4 GB. `onFrame` receives each frame and the pipeline releases it afterwards.
  The workbench counts deliveries and retains only the user-selected frame.
- **Float frame export:** the workbench keeps one frame and offers a `.npy`
  download (shape `(720, 1280, 4)`, little-endian float32, C order) so the
  returned data can be inspected with NumPy.
- **Colour/nits:** a scene-linear value of 1.0 maps to 100 nits; PQ normalizes
  to 10 000 nits and HLG to a 1 000-nit reference. Authored colors peak around
  560 nits. These conventions need an explicit README explanation.
- **SDR preview is an estimate:** it applies tone mapping and sRGB encoding but
  does not yet perform a Rec. 2020 -> Rec. 709 gamut conversion, so highly
  saturated Rec. 2020 colors are not a colour-managed SDR rendering.
- **Encoder metadata:** the tap tags each `VideoFrame` with Rec. 2020 / PQ or
  HLG / `bt2020-ncl`; the encoder is configured for a 10-bit profile. Chunks are
  currently counted but not muxed.

### Browser limitations and risks

- The WebCodecs `VideoFrame(BufferSource, ...)` constructor accepts only 8-bit
  formats in current browsers, so true 10-bit input is unavailable without a
  WASM/native encoder (or externally decoded 10-bit frames).
- WebGPU requires a secure context and a supported browser/OS. Device loss,
  adapter loss, and uncaptured validation/OOM errors are handled as diagnostics;
  a lost device can still leave an in-flight readback waiting.
- `rgba32float` is only usable with the optional renderable/blendable features;
  the context falls back to `rgba16float` otherwise.
- WebCodecs codec availability varies by OS/browser/hardware (HEVC, AV1, VP9
  Profile 2; software encoders can be very slow). Measured floor: software AV1
  Main10 encodes 720p at roughly 26 fps, below the pipeline's submit rate, so
  without back-pressure the queue grows all run and the final flush has to drain
  hundreds of frames. The pipeline now drains when `encodeQueueSize` reaches 8,
  and the final flush is bounded at 30s with a force-close.
- HDR canvas presentation is not portable across browsers/displays; the live
  preview is deliberately SDR tone-mapped.
- Background tabs can throttle timers and presentation; frame content is still
  deterministic because it never depends on wall-clock time.
- Large readbacks and float expansions create GC/memory pressure; the staging
  ring and `reuseOutputBuffer` option bound the pipeline's own allocations.
- No muxer yet, so no container-level `colr`/`mdcv`/`clli` boxes and no
  downloadable HDR video.

### Optimisation roadmap

- Bound `VideoEncoder.encodeQueueSize` and pause/drop frames under back-pressure.
  (Pausing is implemented; dropping frames would need a timestamp policy.)
- Offload readback unpacking and encoding to a worker; `VideoFrame` is
  transferable.
- Expose `reuseOutputBuffer` to remove per-frame 14.75 MB CPU allocations.
- Render the SDR preview at a lower resolution or throttle it; skip it while the
  document is hidden.
- Deliver half-float data or avoid the float32 expansion when consumers can use
  it, and batch/stage copies to reduce readback stalls.
- Use GPU timestamp queries for precise per-stage timing; the current stats are
  CPU wall-clock measurements.
- Consider a WASM 10-bit encoder if source precision matters more than speed.
- Investigate HDR canvas output (`rec2100pq`) behind current Chrome flags, with
  SDR preview as the portable fallback.

### Verification and next steps

- `pnpm build` runs TypeScript and the Vite production build.
- No automated tests yet. Good candidates: matrix composition, timeline
  determinism/loop seam, half-float decode, I420/RGBA packers, and an end-to-end
  browser run asserting 300 `onFrame` deliveries and HDR values above 1.0.
- Muxing the encoder output (for example with `mp4-muxer`) and verifying the
  color metadata in the resulting file is the next functional milestone beyond
  the core frame-generation brief.

### Open items

- [x] Deliver Float32 frames through `onFrame`, count them in the UI, retain one
  selected frame and download it as `.npy`.
- [ ] Add automated unit/end-to-end tests and CI.
- [ ] Mux 10-bit HDR chunks into a playable MP4/WebM and inspect the colour
  metadata that reaches the container.
- [ ] Add a Rec. 2020 -> Rec. 709 gamut matrix to the SDR preview (and document
  the tone curve).
- [x] Apply encoder back-pressure instead of only warning about queue depth.
- [ ] Fail the run promptly on GPU device loss and reject in-flight readbacks.
- [ ] Render text (Canvas 2D/SDF source, linearized and scaled into HDR range).
- [ ] Investigate/present HDR output accurately where the browser supports it.
- [ ] Write the final README (this draft is the source material).
