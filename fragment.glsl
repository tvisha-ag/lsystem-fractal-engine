/*
 * shaders/fragment.glsl — Cyberpunk Neon Wireframe Fragment Shader
 * ================================================================
 * Implements the full neon cyberpunk visual effect:
 *
 *   1. Depth-based colour palette:
 *        root (t=0) → deep magenta / violet
 *        mid  (t≈0.5) → electric cyan
 *        tips (t=1) → hot white-yellow
 *      Achieved via piecewise smoothstep colour interpolation.
 *
 *   2. Chromatic aberration pulse:
 *      A sinusoidal time-dependent shift on R and B channels
 *      mimics CRT scan-line distortion — signature cyberpunk look.
 *
 *   3. Depth-based brightness modulation:
 *      Deeper branches are slightly dimmer, making the root structure
 *      visually heavier — matches organic growth intuition.
 *
 *   4. Edge glow (simulated without actual framebuffer post-processing):
 *      The alpha channel carries a Gaussian falloff around the line centre.
 *      Combined with additive blending in the renderer, this creates
 *      a soft bloom / glow halo.
 *
 *   5. Time-animated flicker:
 *      A very-low-amplitude, high-frequency flicker on brightness
 *      simulates electrical/neon discharge instability.
 *
 * Palette design (in HSV terms):
 *   Magenta  #FF00FF  (H=300°, S=1, V=1)
 *   Cyan     #00FFFF  (H=180°, S=1, V=1)
 *   Yellow   #FFFF00  (H=60°,  S=1, V=1)
 *   The interpolation stays in the high-saturation / high-value region
 *   to maintain the pure neon aesthetic throughout the depth range.
 */

#version 120

varying float v_color_t;
varying float v_depth;
varying vec2  v_world_pos;

uniform float u_time;
uniform float u_max_depth;
uniform float u_glow_intensity;    /* 0.0 – 2.0, user-controlled */
uniform float u_chromatic_strength; /* 0.0 – 0.05 */
uniform int   u_color_scheme;      /* 0=cyberpunk, 1=fire, 2=ice, 3=matrix */

/* ── Colour palette definitions ─────────────────────────────────────────── */

vec3 palette_cyberpunk(float t) {
    /* Magenta → Cyan → White-Yellow */
    vec3 magenta = vec3(1.0,  0.0,  1.0);
    vec3 cyan    = vec3(0.0,  1.0,  1.0);
    vec3 yellow  = vec3(1.0,  1.0,  0.2);

    if (t < 0.5) {
        return mix(magenta, cyan, smoothstep(0.0, 0.5, t));
    } else {
        return mix(cyan, yellow, smoothstep(0.5, 1.0, t));
    }
}

vec3 palette_fire(float t) {
    vec3 deep_red  = vec3(0.8,  0.0,  0.0);
    vec3 orange    = vec3(1.0,  0.5,  0.0);
    vec3 yellow    = vec3(1.0,  1.0,  0.3);

    if (t < 0.5) {
        return mix(deep_red, orange, smoothstep(0.0, 0.5, t));
    } else {
        return mix(orange, yellow, smoothstep(0.5, 1.0, t));
    }
}

vec3 palette_ice(float t) {
    vec3 deep_blue  = vec3(0.0,  0.1,  0.6);
    vec3 cyan       = vec3(0.0,  0.9,  1.0);
    vec3 white      = vec3(0.9,  1.0,  1.0);

    if (t < 0.5) {
        return mix(deep_blue, cyan, smoothstep(0.0, 0.5, t));
    } else {
        return mix(cyan, white, smoothstep(0.5, 1.0, t));
    }
}

vec3 palette_matrix(float t) {
    vec3 black      = vec3(0.0,  0.05, 0.0);
    vec3 dark_green = vec3(0.0,  0.5,  0.1);
    vec3 neon_green = vec3(0.2,  1.0,  0.3);

    if (t < 0.5) {
        return mix(black, dark_green, smoothstep(0.0, 0.5, t));
    } else {
        return mix(dark_green, neon_green, smoothstep(0.5, 1.0, t));
    }
}

vec3 get_palette_color(float t, int scheme) {
    if      (scheme == 1) return palette_fire(t);
    else if (scheme == 2) return palette_ice(t);
    else if (scheme == 3) return palette_matrix(t);
    else                  return palette_cyberpunk(t);
}

/* ── Noise / flicker utilities ───────────────────────────────────────────── */

float hash(float n) {
    return fract(sin(n) * 43758.5453123);
}

float flicker(float t, float seed) {
    /* Low-amplitude high-frequency oscillation */
    float f = sin(t * 47.3 + seed) * 0.5 + 0.5;
    return 1.0 - 0.06 * f;
}

/* ── Main ────────────────────────────────────────────────────────────────── */

void main() {
    float t = clamp(v_color_t, 0.0, 1.0);

    /* Base colour from selected palette */
    vec3 base_color = get_palette_color(t, u_color_scheme);

    /* Depth modulation: tips slightly dimmer */
    float depth_brightness = 1.0 - 0.25 * t;

    /* Chromatic aberration: R and B shift by time-varying amount */
    float ca = u_chromatic_strength * sin(u_time * 1.3 + t * 3.14);
    vec3 neon_color;
    neon_color.r = get_palette_color(clamp(t + ca, 0.0, 1.0), u_color_scheme).r;
    neon_color.g = base_color.g;
    neon_color.b = get_palette_color(clamp(t - ca, 0.0, 1.0), u_color_scheme).b;

    /* Electrical flicker — very subtle, runs at different rates per branch */
    float f = flicker(u_time, v_depth * 0.1 + floor(v_world_pos.x * 0.01));
    neon_color *= f;

    /* Glow intensity modulation */
    neon_color *= (1.0 + u_glow_intensity * 0.5);

    /* Depth brightness */
    neon_color *= depth_brightness;

    /* Clamp to prevent over-saturation while allowing slight bloom headroom */
    neon_color = clamp(neon_color, 0.0, 2.0);

    /* Alpha: slightly pulsing for a "breathing neon" feel */
    float pulse = 0.85 + 0.15 * sin(u_time * 2.0 + t * 6.28);
    float alpha = pulse;

    gl_FragColor = vec4(neon_color, alpha);
}
