// SDR preview shader: fullscreen tone-mapped presentation of the HDR target.
//
// Input is the offscreen render target produced by `hdrScene.wgsl`, which holds
// **linear Rec. 2020** RGBA values that may exceed 1.0 (HDR highlights). This
// pass estimates what a standard SDR display would show by:
//   1. sampling the HDR texture (no clamping on read),
//   2. applying an ACES-style filmic tone curve in linear light to compress
//      highlights smoothly instead of hard-clipping them,
//   3. encoding the result with the sRGB transfer function for an `rgba8unorm`
//      swap-chain (the canvas preferred format is non-sRGB, so the encoding
//      happens here rather than in the compositor).
//
// Orientation:
// - The offscreen target's row 0 is the top of the image (WebGPU framebuffer
//   convention), while clip-space +Y points up. The vertex stage therefore maps
//   NDC bottom-left to UV.y = 1 and NDC top-left to UV.y = 0, so the preview is
//   not vertically flipped relative to the scene.
// - `textureLoad` is used with pixel coordinates, so the shader works for both
//   `rgba16float` (filterable) and `rgba32float` (not filterable in core) without
//   requiring a filtering sampler.

@group(0) @binding(0) var hdrTexture: texture_2d<f32>;

struct PreviewOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
};

// Fullscreen triangle; one oversized triangle covers NDC [-1, 1] with no vertex
// buffer. UV.y is flipped so texture row 0 lands at the top of the canvas.
@vertex
fn vsPreview(@builtin(vertex_index) vertexIndex: u32) -> PreviewOut {
  var positions = array<vec2<f32>, 3>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(3.0, -1.0),
    vec2<f32>(-1.0, 3.0),
  );

  let clip = positions[vertexIndex];
  var out: PreviewOut;
  out.position = vec4<f32>(clip, 0.0, 1.0);
  out.uv = vec2<f32>(clip.x * 0.5 + 0.5, 0.5 - clip.y * 0.5);
  return out;
}

// Narkowicz 2015 ACES filmic approximation, evaluated on linear light. Maps
// 0 -> 0, 1 -> ~0.80, and rolls highlights off smoothly toward 1 instead of
// producing the harsh flat top of a clamp.
fn acesFilmic(linearRgb: vec3<f32>) -> vec3<f32> {
  let a = 2.51;
  let b = 0.03;
  let c = 2.43;
  let d = 0.59;
  let e = 0.14;
  let mapped = (linearRgb * (a * linearRgb + b)) / (linearRgb * (c * linearRgb + d) + e);
  return clamp(mapped, vec3<f32>(0.0), vec3<f32>(1.0));
}

// IEC 61966-2-1 sRGB optical transfer function (linear -> display encoding).
fn linearToSrgb(linearRgb: vec3<f32>) -> vec3<f32> {
  let safe = max(linearRgb, vec3<f32>(0.0));
  let lower = safe * 12.92;
  let higher = 1.055 * pow(safe, vec3<f32>(1.0 / 2.4)) - 0.055;
  return select(higher, lower, safe <= vec3<f32>(0.0031308));
}

@fragment
fn fsPreview(in: PreviewOut) -> @location(0) vec4<f32> {
  let dimensions = vec2<f32>(textureDimensions(hdrTexture));
  // Convert the interpolated UV to an integer texel and clamp to the valid
  // range, so UV == 1 samples the last texel rather than reading out of bounds.
  let pixel = clamp(floor(in.uv * dimensions), vec2<f32>(0.0), dimensions - vec2<f32>(1.0));
  let hdr = textureLoad(hdrTexture, vec2<i32>(pixel), 0).rgb;

  let toneMapped = acesFilmic(max(hdr, vec3<f32>(0.0)));
  let encoded = linearToSrgb(toneMapped);

  // The preview canvas is configured with `alphaMode: "opaque"`, but writing an
  // explicit 1.0 keeps the output well-defined for any swap-chain alpha mode.
  return vec4<f32>(encoded, 1.0);
}
