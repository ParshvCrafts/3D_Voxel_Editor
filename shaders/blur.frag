#version 330 core

in vec2 v_uv;

uniform sampler2D u_texture;
uniform vec2 u_direction;  // (1,0) for horizontal, (0,1) for vertical
uniform vec2 u_resolution;

out vec4 frag_color;

// 9-tap Gaussian blur weights
const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec2 tex_offset = u_direction / u_resolution;

    // Center sample
    vec4 result = texture(u_texture, v_uv) * weights[0];

    // Surrounding samples
    for (int i = 1; i < 5; i++) {
        result += texture(u_texture, v_uv + tex_offset * float(i)) * weights[i];
        result += texture(u_texture, v_uv - tex_offset * float(i)) * weights[i];
    }

    frag_color = result;
}
