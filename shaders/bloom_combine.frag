#version 330 core

in vec2 v_uv;

uniform sampler2D u_scene;
uniform sampler2D u_bloom;
uniform float u_bloom_intensity;
uniform float u_time;
uniform bool u_scanlines_enabled;
uniform float u_scanline_density;

out vec4 frag_color;

void main() {
    vec4 scene_color = texture(u_scene, v_uv);
    vec4 bloom_color = texture(u_bloom, v_uv);

    // PHASE 9.3 FIX: Early out for fully transparent pixels (AR mode)
    // This prevents any color bleeding or effects on transparent areas
    if (scene_color.a < 0.01 && bloom_color.a < 0.01) {
        frag_color = vec4(0.0, 0.0, 0.0, 0.0);
        return;
    }

    // Additive blend for bloom
    vec3 final_color = scene_color.rgb + bloom_color.rgb * u_bloom_intensity;

    // Optional scanline effect - only apply where there's content
    if (u_scanlines_enabled && scene_color.a > 0.01) {
        float scanline = sin(v_uv.y * u_scanline_density + u_time * 2.0) * 0.5 + 0.5;
        scanline = mix(0.85, 1.0, scanline);
        final_color *= scanline;
    }

    // Subtle vignette - only apply where there's content
    if (scene_color.a > 0.01) {
        vec2 center_offset = v_uv - 0.5;
        float vignette = 1.0 - dot(center_offset, center_offset) * 0.5;
        final_color *= vignette;

        // Slight color grading for holographic feel
        final_color.b *= 1.1;  // Slight blue boost
    }

    // CRITICAL: Preserve alpha from scene for AR mode transparency
    // scene_color.a will be 0 where nothing was rendered (transparent)
    // This allows proper compositing over webcam background
    frag_color = vec4(final_color, scene_color.a);
}
