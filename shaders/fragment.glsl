#version 120

varying float v_color_t;
varying float v_depth;
varying vec2  v_world_pos;

uniform float u_time;
uniform float u_max_depth;
uniform float u_glow_intensity;
uniform float u_chromatic_strength;
uniform int   u_color_scheme;

vec3 palette_cyberpunk(float t) {
    vec3 magenta = vec3(1.0, 0.0, 1.0);
    vec3 cyan    = vec3(0.0, 1.0, 1.0);
    vec3 yellow  = vec3(1.0, 1.0, 0.2);
    if (t < 0.5) return mix(magenta, cyan, smoothstep(0.0, 0.5, t));
    else         return mix(cyan, yellow, smoothstep(0.5, 1.0, t));
}

vec3 palette_fire(float t) {
    vec3 deep_red = vec3(0.8, 0.0, 0.0);
    vec3 orange   = vec3(1.0, 0.5, 0.0);
    vec3 yellow   = vec3(1.0, 1.0, 0.3);
    if (t < 0.5) return mix(deep_red, orange, smoothstep(0.0, 0.5, t));
    else         return mix(orange, yellow, smoothstep(0.5, 1.0, t));
}

vec3 palette_ice(float t) {
    vec3 deep_blue = vec3(0.0, 0.1, 0.6);
    vec3 cyan      = vec3(0.0, 0.9, 1.0);
    vec3 white     = vec3(0.9, 1.0, 1.0);
    if (t < 0.5) return mix(deep_blue, cyan, smoothstep(0.0, 0.5, t));
    else         return mix(cyan, white, smoothstep(0.5, 1.0, t));
}

vec3 palette_matrix(float t) {
    vec3 black      = vec3(0.0, 0.05, 0.0);
    vec3 dark_green = vec3(0.0, 0.5,  0.1);
    vec3 neon_green = vec3(0.2, 1.0,  0.3);
    if (t < 0.5) return mix(black, dark_green, smoothstep(0.0, 0.5, t));
    else         return mix(dark_green, neon_green, smoothstep(0.5, 1.0, t));
}

vec3 get_color(float t, int scheme) {
    if      (scheme == 1) return palette_fire(t);
    else if (scheme == 2) return palette_ice(t);
    else if (scheme == 3) return palette_matrix(t);
    else                  return palette_cyberpunk(t);
}

float flicker(float t, float seed) {
    float f = sin(t * 47.3 + seed) * 0.5 + 0.5;
    return 1.0 - 0.06 * f;
}

void main() {
    float t = clamp(v_color_t, 0.0, 1.0);
    vec3 base_color = get_color(t, u_color_scheme);
    float depth_brightness = 1.0 - 0.25 * t;
    float ca = u_chromatic_strength * sin(u_time * 1.3 + t * 3.14159);
    vec3 neon_color;
    neon_color.r = get_color(clamp(t + ca, 0.0, 1.0), u_color_scheme).r;
    neon_color.g = base_color.g;
    neon_color.b = get_color(clamp(t - ca, 0.0, 1.0), u_color_scheme).b;
    float f = flicker(u_time, v_depth * 0.1 + floor(v_world_pos.x * 0.01));
    neon_color *= f;
    neon_color *= (1.0 + u_glow_intensity * 0.5);
    neon_color *= depth_brightness;
    neon_color = clamp(neon_color, 0.0, 2.0);
    float pulse = 0.85 + 0.15 * sin(u_time * 2.0 + t * 6.28318);
    gl_FragColor = vec4(neon_color, pulse);
}