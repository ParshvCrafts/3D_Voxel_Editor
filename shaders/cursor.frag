#version 330 core

in vec2 v_uv;
in vec3 v_world_pos;

uniform vec3 u_color;
uniform float u_time;
uniform float u_pulse;
uniform float u_progress;  // 0-1 for loading circle

out vec4 frag_color;

const float PI = 3.14159265359;

float ring(vec2 uv, float radius, float thickness) {
    float dist = length(uv);
    return smoothstep(radius - thickness, radius, dist) -
           smoothstep(radius, radius + thickness, dist);
}

float arc(vec2 uv, float radius, float thickness, float start_angle, float end_angle) {
    float dist = length(uv);
    float angle = atan(uv.y, uv.x);

    // Normalize angle to 0-2PI
    angle = mod(angle + PI, 2.0 * PI);
    start_angle = mod(start_angle + PI, 2.0 * PI);
    end_angle = mod(end_angle + PI, 2.0 * PI);

    float ring_mask = smoothstep(radius - thickness, radius, dist) -
                      smoothstep(radius, radius + thickness, dist);

    float angle_mask;
    if (start_angle < end_angle) {
        angle_mask = step(start_angle, angle) * step(angle, end_angle);
    } else {
        angle_mask = step(start_angle, angle) + step(angle, end_angle);
    }

    return ring_mask * angle_mask;
}

void main() {
    vec2 uv = v_uv * 2.0 - 1.0;  // Center UV

    // Check if this is a line (wireframe) - UVs will be (0,0) for lines
    // For lines, just output solid color
    if (v_uv.x == 0.0 && v_uv.y == 0.0) {
        float pulse = 0.8 + 0.2 * sin(u_time * 4.0);
        frag_color = vec4(u_color * pulse, 0.95);
        return;
    }

    // Pulsing glow
    float pulse = 0.8 + 0.2 * sin(u_time * u_pulse);

    // Inner dot
    float inner_dot = 1.0 - smoothstep(0.0, 0.15, length(uv));

    // Middle ring (rotating)
    vec2 rotated_uv = vec2(
        uv.x * cos(u_time) - uv.y * sin(u_time),
        uv.x * sin(u_time) + uv.y * cos(u_time)
    );
    float middle_ring = ring(uv, 0.35, 0.03);

    // Outer ring with markers (counter-rotating)
    vec2 counter_rotated = vec2(
        uv.x * cos(-u_time * 0.5) - uv.y * sin(-u_time * 0.5),
        uv.x * sin(-u_time * 0.5) + uv.y * cos(-u_time * 0.5)
    );
    float outer_ring = ring(uv, 0.6, 0.02);

    // Tick marks on outer ring
    float tick_angle = atan(counter_rotated.y, counter_rotated.x);
    float ticks = step(0.9, cos(tick_angle * 8.0)) * ring(uv, 0.6, 0.08);

    // Loading progress arc
    float progress_arc = 0.0;
    if (u_progress > 0.01) {
        float end_angle = -PI/2.0 + u_progress * 2.0 * PI;
        progress_arc = arc(uv, 0.45, 0.05, -PI/2.0, end_angle);
    }

    // Combine all elements
    float alpha = inner_dot + middle_ring * 0.8 + outer_ring * 0.5 + ticks * 0.6 + progress_arc;
    alpha *= pulse;

    // Color
    vec3 color = u_color;

    // Progress arc is brighter
    if (progress_arc > 0.1) {
        color = mix(color, vec3(1.0), 0.3);
    }

    frag_color = vec4(color * (1.0 + alpha * 0.5), alpha);
}
