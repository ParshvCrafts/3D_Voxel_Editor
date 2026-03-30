#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;  // Dummy UV to match vertex format

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;

void main() {
    vec4 world_pos = u_model * vec4(in_position, 1.0);
    gl_Position = u_projection * u_view * world_pos;
    // in_uv is unused but required for vertex format compatibility
}
