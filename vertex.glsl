/*
 * shaders/vertex.glsl — Cyberpunk Neon Wireframe Vertex Shader
 * =============================================================
 * Transforms geometry from world space to clip space.
 * Passes per-vertex data to the fragment shader for neon coloring.
 *
 * Inputs (per vertex, set by Python renderer):
 *   a_position  : vec2  — (x, y) world-space position
 *   a_color_t   : float — normalized depth [0, 1] for color interpolation
 *   a_depth     : float — raw tree depth for bloom calculation
 *
 * Uniforms (set once per frame by Python renderer):
 *   u_resolution : vec2  — window width, height in pixels
 *   u_offset     : vec2  — pan offset in world space
 *   u_scale      : float — zoom scale factor
 *   u_time       : float — elapsed seconds (for animation)
 */

#version 120

attribute vec2  a_position;
attribute float a_color_t;
attribute float a_depth;

uniform vec2  u_resolution;
uniform vec2  u_offset;
uniform float u_scale;
uniform float u_time;

varying float v_color_t;
varying float v_depth;
varying vec2  v_world_pos;

void main() {
    /* Apply pan + zoom transform */
    vec2 world = (a_position + u_offset) * u_scale;

    /* Normalize to [-1, 1] clip space */
    vec2 clip = (world / u_resolution) * 2.0 - 1.0;
    clip.y = -clip.y;   /* Flip Y: screen-down → OpenGL-up */

    gl_Position = vec4(clip, 0.0, 1.0);

    /* Pass-through to fragment shader */
    v_color_t   = a_color_t;
    v_depth     = a_depth;
    v_world_pos = a_position;
}
