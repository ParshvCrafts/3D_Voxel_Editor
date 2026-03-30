#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 instance_position;
layout(location = 2) in float instance_size;
layout(location = 3) in float instance_alpha;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform float u_time;

out float v_alpha;
out vec2 v_uv;

void main() {
    // Billboard - always face camera
    vec3 camera_right = vec3(u_view[0][0], u_view[1][0], u_view[2][0]);
    vec3 camera_up = vec3(u_view[0][1], u_view[1][1], u_view[2][1]);

    vec3 vertex_pos = instance_position +
                      camera_right * in_position.x * instance_size +
                      camera_up * in_position.y * instance_size;

    gl_Position = u_projection * u_view * vec4(vertex_pos, 1.0);

    v_alpha = instance_alpha;
    v_uv = in_position.xy + 0.5;  // Convert -0.5 to 0.5 -> 0 to 1
}
