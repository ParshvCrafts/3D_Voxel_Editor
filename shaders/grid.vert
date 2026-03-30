#version 330 core

layout(location = 0) in vec3 in_position;

uniform mat4 u_projection;
uniform mat4 u_view;

out vec3 v_world_pos;

void main() {
    v_world_pos = in_position;
    gl_Position = u_projection * u_view * vec4(in_position, 1.0);
}
