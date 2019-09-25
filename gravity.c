__kernel void gravity(__global float3* positions,
                      __global float3* velocities,
                      __global float4* galaxies,
                      __const uint body_count,
                      __const uint galaxy_count,
                      __const float dt,
                      __const float G) {

    int index = get_global_id(0);

    float3 force = (float3)(0.0f, 0.0f, 0.0f);
    float3 delta_pos;

    // update particle position
    positions[index] += dt * velocities[index];

    for (int i=0; i<galaxy_count; i++) {
        delta_pos = galaxies[i].xyz - positions[index];
        float dist = length(delta_pos);
        if (dist < 0.001) {
            dist = 0.001;
        }
        force += normalize(delta_pos) * G / (dist);

    }

    velocities[index] += force * dt;

}