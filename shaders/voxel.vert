#version 330 core

// Vertex attributes
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;

// Instanced attributes
layout(location = 2) in vec3 instance_position;
layout(location = 3) in vec3 instance_color;
layout(location = 4) in vec3 instance_rotation;  // Euler angles for scattered voxels
layout(location = 5) in float instance_selected;

// Uniforms
uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;  // PHASE 9.3: Group transform (grab/rotate)
uniform vec3 u_camera_pos;
uniform float u_time;

// Outputs to fragment shader
out vec3 v_world_pos;
out vec3 v_normal;
out vec3 v_color;
out vec3 v_view_dir;
out float v_selected;

// Rotation matrix from euler angles
mat3 rotation_matrix(vec3 angles) {
    float cx = cos(angles.x);
    float sx = sin(angles.x);
    float cy = cos(angles.y);
    float sy = sin(angles.y);
    float cz = cos(angles.z);
    float sz = sin(angles.z);

    return mat3(
        cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz,
        cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz,
        -sy,   sx*cy,            cx*cy
    );
}

void main() {
    // Apply rotation for scattered voxels
    mat3 rot = rotation_matrix(instance_rotation);
    vec3 rotated_pos = rot * in_position;
    vec3 rotated_normal = rot * in_normal;

    // Local world position (before group transform)
    vec3 local_world_pos = rotated_pos + instance_position;
    
    // PHASE 9.3: Apply group transform (grab/rotate)
    // This transforms the entire voxel group without rebuilding buffers
    vec4 transformed = u_model * vec4(local_world_pos, 1.0);
    v_world_pos = transformed.xyz;

    // Transform to clip space
    gl_Position = u_projection * u_view * vec4(v_world_pos, 1.0);

    // Pass data to fragment shader
    // Transform normal by model matrix (rotation only, no translation)
    v_normal = mat3(u_model) * rotated_normal;
    v_color = instance_color;
    v_view_dir = normalize(u_camera_pos - v_world_pos);
    v_selected = instance_selected;
}
