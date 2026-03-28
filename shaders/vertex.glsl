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
    vec2 world = (a_position + u_offset) * u_scale;
    vec2 clip = (world / u_resolution) * 2.0 - 1.0;
    clip.y = -clip.y;
    gl_Position = vec4(clip, 0.0, 1.0);
    v_color_t   = a_color_t;
    v_depth     = a_depth;
    v_world_pos = a_position;
}