__kernel void gravity(__global float4 *positions,
                      __global float4 *velocities,
                      __global float4 *galaxies,
                      __const  uint    body_count,
                      __const  uint    galaxy_count,
                      __const  float   dt,
                      __const  float   G) {

    uint index = get_global_id(0);

    float4 force = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 delta_pos;

    // update particle position
    positions[index] += dt * velocities[index];

    for (int i=0; i<galaxy_count; i++) {
        delta_pos = galaxies[i] - positions[index];
        float dist = length(delta_pos.xyz);
        if (dist < 0.01) {
            dist = 0.01;
        }
        force += normalize(delta_pos) * G * galaxies[i].w / (dist * dist);

    }
    velocities[index] += force * dt;
}

float random(ulong* state) {

    ulong x = *state ^ (*state >> 12);
    x ^= x << 25;
    x ^= x >> 27;
    
    *state = x;
    x *= 0x2545F4914F6CDD1D;

    return (float)(x & 0xffffffff) / (float)0xffffffff;
}

__kernel void initialize(__global float4 *positions,
                         __global float4 *velocities,
                         __global float4 *colors,
                         const    uint    body_count,
                         const    float4  galaxy_pos,
                         const    float4  galaxy_vel,
                         const    float4  galaxy_rot,
                         const    float   G,
                         const    float   dt,
                         const    float   mu,
                         const    float   sigma,
                         const    float   eccentricity) {
    
    float3 UP = (float3)(0, 1, 0);

    float r;
    float theta;
    float height;

    float3 position;
    float3 velocity;

    uint index = get_global_id(0);
    ulong state = index + 1;

    float seed = (float)index / (float)body_count;

    float rand1, rand2;
    float gaussian1, gaussian2;

    theta = 2.0f * M_PI * random(&state);
    
    rand1 = random(&state);
    rand2 = random(&state);
    gaussian1 = sqrt(-2.0f * log(rand1)) * cos(-2.0f * M_PI * rand2) * sigma + mu;
    gaussian2 = sqrt(-2.0f * log(rand1)) * sin(-2.0f * M_PI * rand2) * sigma;

    r = gaussian1;
    height = 0.1 * gaussian2;

    float e2 = eccentricity * eccentricity;
  

    // scale x values by this to get eccentric galaxy
    float scale = sqrt(1.0f / (1.0f - eccentricity * eccentricity));
    float shift = sqrt(1.0f / (1.0f - eccentricity * eccentricity) - 1.0f);

    
    position = (float3)(r * cos(theta), height, r * sin(theta));
    //position.x *= scale;
    //position.x += shift;
    velocity = normalize(cross(UP, position));
    //velocity.x *= scale;
    //velocity = normalize(velocity);
    velocity *= sqrt(G * galaxy_pos.w / length(position));
    //velocity *= sqrt(G * (2.0f / length(position) - (1.0f / scale)));

    // rotate position and velocity
    // https://gamedev.stackexchange.com/questions/28395/rotating-vector3-by-a-quaternion
    float3 u = galaxy_rot.xyz; // vector part of rotation quaternion
    float s  = galaxy_rot.w; // scalar part of rotation quaternion
    float k  = (s * s - dot(u, u));
    position = galaxy_pos.xyz + 2.0f * dot(u, position) * u + k * position + 2.0f * s * cross(u, position);
    velocity = galaxy_vel.xyz + 2.0f * dot(u, velocity) * u + k * velocity + 2.0f * s * cross(u, velocity);

    positions[index].xyz = position;
    positions[index].w = 1.0f;
    velocities[index].xyz = velocity;
    velocities[index].w = 0.0f;
    colors[index] = (float4)(0.0, 1.0, 1.0, 1.0);
}