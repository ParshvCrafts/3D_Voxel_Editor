#version 330 core

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 instance_position;
layout(location = 2) in vec3 instance_color;
layout(location = 3) in vec3 instance_rotation;

uniform mat4 u_projection;
uniform mat4 u_view;
uniform mat4 u_model;  // PHASE 9.3: Group transform (grab/rotate)

out vec3 v_color;

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
    mat3 rot = rotation_matrix(instance_rotation);
    vec3 local_world_pos = rot * in_position + instance_position;
    
    // PHASE 9.3: Apply group transform (grab/rotate)
    vec4 transformed = u_model * vec4(local_world_pos, 1.0);
    vec3 world_pos = transformed.xyz;

    gl_Position = u_projection * u_view * vec4(world_pos, 1.0);
    v_color = instance_color;
}
