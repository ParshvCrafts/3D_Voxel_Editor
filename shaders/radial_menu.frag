#version 330 core

in vec2 v_uv;

uniform float u_time;
uniform int u_num_colors;
uniform int u_selected_index;
uniform vec3 u_colors[8];  // Max 8 colors

out vec4 frag_color;

const float PI = 3.14159265359;

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;  // Center UV to -1 to 1

    float dist = length(uv);
    float angle = atan(uv.y, uv.x);

    // Ring boundaries
    float inner_radius = 0.4;
    float outer_radius = 0.9;

    // Check if in ring area
    if (dist < inner_radius || dist > outer_radius) {
        // Draw ring outlines
        float ring_alpha = 0.0;

        // Inner ring
        ring_alpha += (1.0 - smoothstep(inner_radius - 0.02, inner_radius, dist)) * 0.5;
        ring_alpha += smoothstep(inner_radius, inner_radius + 0.02, dist) *
                      (1.0 - smoothstep(inner_radius + 0.02, inner_radius + 0.04, dist)) * 0.8;

        // Outer ring
        ring_alpha += smoothstep(outer_radius - 0.04, outer_radius - 0.02, dist) *
                      (1.0 - smoothstep(outer_radius - 0.02, outer_radius, dist)) * 0.8;
        ring_alpha += smoothstep(outer_radius, outer_radius + 0.02, dist) *
                      (1.0 - smoothstep(outer_radius + 0.02, outer_radius + 0.04, dist)) * 0.3;

        if (ring_alpha < 0.01) {
            discard;
        }

        // Cyan outline color
        frag_color = vec4(0.0, 0.831, 1.0, ring_alpha);
        return;
    }

    // Calculate which segment we're in
    float segment_angle = 2.0 * PI / float(u_num_colors);
    float adjusted_angle = mod(angle + PI + segment_angle / 2.0, 2.0 * PI);
    int segment = int(adjusted_angle / segment_angle);

    // Get segment color
    vec3 color = u_colors[segment % 8];

    // Segment divider lines
    float line_angle = mod(adjusted_angle, segment_angle);
    float line_dist = min(line_angle, segment_angle - line_angle);
    float line_alpha = 1.0 - smoothstep(0.0, 0.05, line_dist);

    // Selected segment highlight
    float selected_alpha = 0.0;
    if (segment == u_selected_index) {
        selected_alpha = 0.3 + 0.1 * sin(u_time * 4.0);

        // Make selected segment larger (push outward)
        float expand = 0.05 * (0.5 + 0.5 * sin(u_time * 3.0));
        if (dist > outer_radius - 0.1 - expand) {
            selected_alpha += 0.2;
        }
    }

    // Final color
    float alpha = 0.7 + selected_alpha;
    vec3 final_color = color;

    // Add white line on dividers
    final_color = mix(final_color, vec3(1.0), line_alpha * 0.5);

    frag_color = vec4(final_color, alpha);
}
