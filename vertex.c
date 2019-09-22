#version 450

in vec4 position;

uniform float x;
uniform float y;
uniform float z;

void main()
{
    vec3 eye    = vec3(x, y, z);
    vec3 center = vec3(0, 0, 0);
    vec3 up     = vec3(0, 1, 0);
    vec3 zaxis  = normalize(center - eye);
    vec3 xaxis  = normalize(cross(up, zaxis));
    vec3 yaxis  = cross(zaxis, xaxis);

    mat4 model_matrix = mat4(
        xaxis.x, yaxis.x, zaxis.x, 0,
        xaxis.y, yaxis.y, zaxis.y, 0,
        xaxis.z, yaxis.z, zaxis.z, 0,
        dot(xaxis, -eye), dot(yaxis, -eye), dot(zaxis, -eye), 1);

    float s = 1.0 / tan(15 * 3.14159 / 180);
    mat4 perspective_matrix = mat4(
        s, 0, 0, 0,
        0, s, 0, 0,
        0, 0, 1, -1,
        0, 0, -1, 0);
    

    gl_Position =  model_matrix * perspective_matrix * position;
}