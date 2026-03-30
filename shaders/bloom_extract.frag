#version 330 core

in vec2 v_uv;

uniform sampler2D u_texture;
uniform float u_threshold;

out vec4 frag_color;

void main() {
    vec4 color = texture(u_texture, v_uv);

    // Calculate luminance
    float luminance = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));

    // Extract bright pixels
    if (luminance > u_threshold) {
        frag_color = color;
    } else {
        frag_color = vec4(0.0);
    }
}
