// HDR scene shader: fullscreen linear background + three transformed shape quads.
//
// Coordinate conventions (important):
// - CPU-side design space is 1280x720 pixels, origin top-left, +Y down.
// - WebGPU clip space (NDC) is x,y in [-1, 1] with +Y **up**; visible fragments
//   have 0 <= z <= w. The CPU pre-multiplies each shape's local->design matrix
//   by the design->clip matrix (`composeShapeClipMatrix` in math/mat3.ts), so
//   this shader consumes clip-space positions directly and never guesses about
//   pixel orientation.
// - The fullscreen background triangle is authored directly in clip space and
//   covers [-1, 1] on both axes. Its UVs put (0,0) at the NDC bottom-left and
//   (1,1) at the NDC top-right, so the gradient's `from` color is at the screen
//   bottom and `to` is at the screen top.
// - Y handedness: design space is +Y down, `designToClip` flips to NDC +Y up,
//   and the WebGPU viewport flips again into framebuffer rows. The two flips
//   cancel, so shape-local +Y (down) renders downward and quads are never
//   vertically mirrored. All three shapes share this one path.
//
// Color conventions:
// - All colors are **linear Rec. 2020 RGBA**, never sRGB-encoded, and are not
//   clamped: values above 1.0 are intentional HDR highlights.
// - Shape output is **premultiplied alpha** (`rgb * a`, `a`). The renderer uses
//   a `one` / `one-minus-src-alpha` blend state, which avoids dark fringes when
//   translucent shapes composite over HDR values.
// - The background is opaque and also written premultiplied for consistency.
// - Shape edges use analytic coverage computed from `fwidth(localUv)` instead of
//   MSAA. Coverage fades over roughly one pixel regardless of rotation, scale,
//   or skew, and multiplies into the premultiplied color so edge pixels blend
//   correctly with whatever is behind them.

struct BackgroundUniforms {
  // `from` is a reserved word in WGSL (see WGSL spec §16.2), so the gradient
  // start color uses a non-reserved member name.
  fromColor: vec4<f32>,
  to: vec4<f32>,
};

struct ShapeUniforms {
  // Shape-local (unit quad) -> WebGPU clip space, column-major mat4x4.
  transform: mat4x4<f32>,
  // Linear Rec. 2020 RGB with opacity in `.a`.
  color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> background: BackgroundUniforms;
@group(0) @binding(1) var<uniform> shape: ShapeUniforms;

struct BackgroundOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

// Fullscreen triangle: a single oversized triangle covers NDC [-1, 1] without
// an index/vertex buffer. UV.y = 0 is the bottom edge, UV.y = 1 the top edge.
@vertex
fn vsBackground(@builtin(vertex_index) vertexIndex: u32) -> BackgroundOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );

  let clip = positions[vertexIndex];
  var out: BackgroundOut;
  out.position = vec4<f32>(clip, 0.0, 1.0);
  out.uv = clip * 0.5 + vec2<f32>(0.5, 0.5);
  return out;
}

@fragment
fn fsBackground(in: BackgroundOut) -> @location(0) vec4<f32> {
  // Smooth vertical gradient between the scene's two linear HDR colors.
  let t = smoothstep(0.0, 1.0, clamp(in.uv.y, 0.0, 1.0));
  let rgb = mix(background.fromColor.rgb, background.to.rgb, t);
  let alpha = mix(background.fromColor.a, background.to.a, t);
  return vec4<f32>(rgb * alpha, alpha);
}

struct ShapeOut {
  @builtin(position) position: vec4<f32>,
  @location(0) localUv: vec2<f32>,
};

// Unit quad in shape-local space (0..1). The uniform matrix already includes
// the shape's size, scale, skew, rotation, pivot offset, animation translation,
// and the design-pixel -> clip-space conversion.
@vertex
fn vsShape(@builtin(vertex_index) vertexIndex: u32) -> ShapeOut {
  var corners = array<vec2<f32>, 6>(
    vec2<f32>(0.0, 0.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(0.0, 1.0),
    vec2<f32>(1.0, 0.0),
    vec2<f32>(1.0, 1.0),
  );

  let unit = corners[vertexIndex];
  var out: ShapeOut;
  out.position = shape.transform * vec4<f32>(unit, 0.0, 1.0);
  out.localUv = unit;
  return out;
}

@fragment
fn fsShape(in: ShapeOut) -> @location(0) vec4<f32> {
  // Analytic antialiasing for the quad's four edges. `localUv` is 0..1 across
  // the quad; its screen-space derivative is the UV footprint of one pixel, so
  // `edgeDistance / footprint` is the fraction of the pixel covered by the
  // shape. Doing this per-axis and taking the minimum also handles the corner
  // regions, where two edges are close. The result is deterministic (derived
  // from rasterization, not a random/sample pattern) and avoids needing an MSAA
  // target, which would force a resolve before readback and encode.
  let edgeDistance = min(in.localUv, vec2<f32>(1.0) - in.localUv);
  let pixelFootprint = max(fwidth(in.localUv), vec2<f32>(1.0e-6));
  let coverage = clamp(
    min(edgeDistance.x / pixelFootprint.x, edgeDistance.y / pixelFootprint.y),
    0.0,
    1.0,
  );

  // Flat linear HDR fill; opacity comes from `.a`. Premultiply so the
  // one / one-minus-src-alpha blend state composites correctly over HDR.
  let alpha = clamp(shape.color.a, 0.0, 1.0) * coverage;
  return vec4<f32>(shape.color.rgb * alpha, alpha);
}
