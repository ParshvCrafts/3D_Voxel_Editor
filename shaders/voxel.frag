#version 330 core

// Inputs from vertex shader
in vec3 v_world_pos;
in vec3 v_normal;
in vec3 v_color;
in vec3 v_view_dir;
in float v_selected;

// Uniforms
uniform float u_time;
uniform float u_fresnel_power;
uniform float u_fresnel_bias;
uniform float u_glow_intensity;
uniform float u_opacity;  // AR mode opacity (Phase 4)

// Output
out vec4 frag_color;

void main() {
    // Normalize interpolated normal
    vec3 normal = normalize(v_normal);
    vec3 view_dir = normalize(v_view_dir);

    // Fresnel effect - edges glow brighter
    float fresnel = u_fresnel_bias + (1.0 - u_fresnel_bias) * pow(1.0 - max(dot(view_dir, normal), 0.0), u_fresnel_power);

    // YouTube reference (createFinalCube function):
    // - Base color: 0x001122 (very dark blue)
    // - Emissive: bright cyan (0x00f0ff)
    // - emissiveIntensity: 0.4
    // - opacity: 0.8
    
    // Dark base color (like YouTube's 0x001122)
    vec3 dark_base = vec3(0.0, 0.067, 0.133);
    
    // Emissive glow color (the voxel's assigned color acts as emissive)
    vec3 emissive_color = v_color;
    float emissive_intensity = 0.4;
    
    // Combine dark base with emissive glow
    vec3 final_color = dark_base + emissive_color * emissive_intensity;
    
    // Add fresnel edge glow (brighter edges)
    final_color += emissive_color * fresnel * 0.6;

    // Selection highlight
    if (v_selected > 0.5) {
        float pulse = 0.7 + 0.3 * sin(u_time * 5.0);
        final_color = mix(final_color, vec3(1.0, 1.0, 0.5), 0.3 * pulse);
    }

    // Apply glow intensity
    final_color *= u_glow_intensity;

    // YouTube uses opacity: 0.8 - much more visible than before
    // Base alpha is 0.85 (slightly higher than YouTube's 0.8 for better visibility)
    float alpha = 0.85;

    // Apply AR mode opacity (Phase 4)
    float final_opacity = u_opacity > 0.0 ? u_opacity : 1.0;
    alpha *= final_opacity;

    frag_color = vec4(final_color, alpha);
}
