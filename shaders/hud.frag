#version 330 core

in vec2 v_uv;

uniform sampler2D u_texture;
uniform float u_alpha;
uniform vec4 u_border_color;
uniform float u_border_width;
uniform vec2 u_size;
uniform float u_mirror_x;  // Phase 5: 1.0 to mirror horizontally, 0.0 for normal

out vec4 frag_color;

void main() {
    // Apply horizontal mirror if enabled (Phase 5)
    vec2 uv = v_uv;
    if (u_mirror_x > 0.5) {
        uv.x = 1.0 - uv.x;
    }

    vec4 tex_color = texture(u_texture, uv);

    // Border effect
    float border = 0.0;
    vec2 border_uv = v_uv * u_size;

    if (border_uv.x < u_border_width || border_uv.x > u_size.x - u_border_width ||
        border_uv.y < u_border_width || border_uv.y > u_size.y - u_border_width) {
        border = 1.0;
    }

    // Corner markers (JARVIS style)
    float corner_size = 20.0;
    float corner_thickness = 3.0;

    // Top-left
    if ((border_uv.x < corner_size && border_uv.y < corner_thickness) ||
        (border_uv.x < corner_thickness && border_uv.y < corner_size)) {
        border = 1.0;
    }
    // Top-right
    if ((border_uv.x > u_size.x - corner_size && border_uv.y < corner_thickness) ||
        (border_uv.x > u_size.x - corner_thickness && border_uv.y < corner_size)) {
        border = 1.0;
    }
    // Bottom-left
    if ((border_uv.x < corner_size && border_uv.y > u_size.y - corner_thickness) ||
        (border_uv.x < corner_thickness && border_uv.y > u_size.y - corner_size)) {
        border = 1.0;
    }
    // Bottom-right
    if ((border_uv.x > u_size.x - corner_size && border_uv.y > u_size.y - corner_thickness) ||
        (border_uv.x > u_size.x - corner_thickness && border_uv.y > u_size.y - corner_size)) {
        border = 1.0;
    }

    vec4 final_color = mix(tex_color, u_border_color, border);
    final_color.a *= u_alpha;

    frag_color = final_color;
}
