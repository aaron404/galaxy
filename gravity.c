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

/*
 *
state = 1
def xorshift64s(seed=None):
global state
if seed:
    state = seed

    x = state
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    x &= 0xffffffffffffffff
    state = x
    x = x * 0x2545f4914f6cdd1d
    val = (x & 0xffffffff) / 0xffffffff
    return val
*/

__kernel void init(__global float4* positions,
                   __const  float   mu,
                   __const  float   sigma) {

    ulong index = get_global_id(0);
    ulong state = index = 1;
    ulong x = state;
    float u, v;

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state = x;

    x = (x * 0x2545f4914f6cdd1d) & 0xffffffff;
    u = sqrt(2.0f * log((float)x / (float)0xffffffff));

    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;

    x = (x * 0x2545f4914f6cdd1d) & 0xffffffff;
    v = 2.0f * M_PI * (float)x / (float)0xffffffff;

    float z0 = u * cos(v);
    float z1 = u * sin(v);

    float rand_normal = z0 * sigma * mu;
}
