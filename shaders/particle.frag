#version 330 core

in float v_alpha;
in vec2 v_uv;

uniform vec3 u_color;
uniform float u_time;

out vec4 frag_color;

void main() {
    // Circular particle with soft edges
    vec2 center = v_uv - 0.5;
    float dist = length(center) * 2.0;

    // Soft circle
    float alpha = 1.0 - smoothstep(0.0, 1.0, dist);
    alpha *= v_alpha;

    // Subtle twinkle
    float twinkle = 0.7 + 0.3 * sin(u_time * 3.0 + gl_FragCoord.x * 0.1);
    alpha *= twinkle;

    frag_color = vec4(u_color, alpha * 0.6);
}
