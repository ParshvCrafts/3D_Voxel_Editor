#version 330 core

in vec3 v_color;

uniform float u_time;
uniform float u_glow_intensity;
uniform float u_opacity;  // AR mode opacity (Phase 4)

out vec4 frag_color;

void main() {
    // YouTube reference: Edge lines are bright and fully opaque
    // LineBasicMaterial({ color: 0x00f0ff }) - no transparency
    
    // Subtle pulsing glow effect
    float pulse = 0.9 + 0.1 * sin(u_time * 2.0);

    // Bright wireframe color - edges should be very visible
    vec3 glow_color = v_color * u_glow_intensity * pulse;
    
    // Boost brightness for edge visibility
    glow_color = min(glow_color * 1.5, vec3(1.0));

    // Apply AR mode opacity (Phase 4)
    // Wireframe should be nearly fully opaque for visibility
    float final_opacity = u_opacity > 0.0 ? u_opacity : 1.0;
    
    // Keep wireframe at high opacity (0.95) for clear edges
    float alpha = 0.95 * final_opacity;

    frag_color = vec4(glow_color, alpha);
}
