#version 330 core

in vec3 v_world_pos;

uniform vec3 u_camera_pos;
uniform vec3 u_grid_color;
uniform float u_grid_spacing;
uniform float u_fade_distance;
uniform float u_time;
uniform float u_opacity;  // AR mode opacity (Phase 4)

out vec4 frag_color;

float grid_line(float coord, float line_width) {
    float grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    return 1.0 - min(grid / line_width, 1.0);
}

void main() {
    // Grid lines on XZ plane
    float line_x = grid_line(v_world_pos.x / u_grid_spacing, 1.5);
    float line_z = grid_line(v_world_pos.z / u_grid_spacing, 1.5);

    // Major grid lines (every 5 units)
    float major_x = grid_line(v_world_pos.x / (u_grid_spacing * 5.0), 2.0);
    float major_z = grid_line(v_world_pos.z / (u_grid_spacing * 5.0), 2.0);

    // Combine lines
    float grid = max(line_x, line_z);
    float major_grid = max(major_x, major_z);

    // Distance fade
    float dist = length(v_world_pos.xz - u_camera_pos.xz);
    float fade = 1.0 - smoothstep(0.0, u_fade_distance, dist);

    // Color
    vec3 color = u_grid_color;

    // Major lines are brighter
    color = mix(color, color * 1.5, major_grid);

    // Subtle animation
    float pulse = 0.9 + 0.1 * sin(u_time * 0.5);

    // Final color with fade
    float alpha = grid * fade * pulse * 0.6;

    // Add origin highlight
    float origin_dist = length(v_world_pos.xz);
    if (origin_dist < 0.5) {
        alpha = max(alpha, 0.8 * (1.0 - origin_dist * 2.0));
        color = vec3(1.0);  // White at origin
    }

    // Apply AR mode opacity (Phase 4)
    float final_opacity = u_opacity > 0.0 ? u_opacity : 1.0;
    alpha *= final_opacity;

    frag_color = vec4(color, alpha);
}
