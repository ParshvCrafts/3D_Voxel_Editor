#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec2 in_uv;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform vec3 u_center;
uniform float u_radius;

out vec2 v_uv;

void main() {
    // Billboard quad facing camera
    vec3 camera_right = vec3(u_view[0][0], u_view[1][0], u_view[2][0]);
    vec3 camera_up = vec3(u_view[0][1], u_view[1][1], u_view[2][1]);

    vec3 vertex_pos = u_center +
                      camera_right * in_position.x * u_radius +
                      camera_up * in_position.y * u_radius;

    gl_Position = u_projection * u_view * vec4(vertex_pos, 1.0);
    v_uv = in_uv;
}
