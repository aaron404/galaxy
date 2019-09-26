__kernel void gravity(__global float3* positions,
                      __global float3* velocities,
                      __global float4* galaxies,
                      __const uint body_count,
                      __const uint galaxy_count,
                      __const float dt,
                      __const float G) {

    uint index = get_global_id(0);

    float3 force = (float3)(0.0f, 0.0f, 0.0f);
    float3 delta_pos;

    // update particle position
    //positions[index] += dt * velocities[index];
    positions[index] += (float3)(0, 1, 0) * dt;

    for (int i=0; i<galaxy_count; i++) {
        delta_pos = galaxies[i].xyz - positions[index];
        float dist = length(delta_pos.xyz);
        if (dist < 0.001) {
            dist = 0.001;
        }
        force += normalize(delta_pos) * G / (dist * dist);

    }

    velocities[index] += force * dt;

}