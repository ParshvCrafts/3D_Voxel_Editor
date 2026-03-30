#version 330 core

uniform vec3 u_color;
uniform float u_time;

out vec4 frag_color;

void main() {
    // Simple pulsing glow for wireframe
    float pulse = 0.8 + 0.2 * sin(u_time * 4.0);
    vec3 color = u_color * pulse;
    
    // High opacity for visibility
    frag_color = vec4(color, 0.95);
}
