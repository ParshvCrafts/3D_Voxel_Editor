#version 330 core

layout(location = 0) in vec2 in_position;
layout(location = 1) in vec2 in_uv;

uniform vec2 u_position;   // Screen position (pixels)
uniform vec2 u_size;       // Size (pixels)
uniform vec2 u_screen_size;

out vec2 v_uv;

void main() {
    // Convert pixel coordinates to NDC (-1 to 1)
    vec2 pos = in_position * u_size + u_position;
    vec2 ndc = (pos / u_screen_size) * 2.0 - 1.0;
    ndc.y = -ndc.y;  // Flip Y for screen coordinates

    gl_Position = vec4(ndc, 0.0, 1.0);
    v_uv = in_uv;
}
