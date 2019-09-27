__kernel void gravity(__global float4* positions,
                      __global float4* velocities,
                      __global float4* galaxies,
                      __const uint body_count,
                      __const uint galaxy_count,
                      __const float dt,
                      __const float G) {

    uint index = get_global_id(0);

    float4 force = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 delta_pos;

    // update particle position
    positions[index] += dt * velocities[index];

    for (int i=0; i<galaxy_count; i++) {
        delta_pos = galaxies[i] - positions[index];
        float dist = length(delta_pos.xyz);
        if (dist < 0.001) {
            dist = 0.001;
        }
        force += normalize(delta_pos) * G / (dist * dist);

    }
    velocities[index] += force * dt;
}