# hdr-frame-poc

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
