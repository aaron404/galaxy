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

    
    float aspect = 1920.0f / 1080.0f;
    float fov = 1.0 / tan((15) * 3.14159f / 180);
    float zFar = 100.0f;
    float zNear = 1.0f;
    mat4 perspective_matrix = mat4(
        fov/aspect, 0, 0, 0,
        0, fov, 0, 0,
        0, 0, (zFar + zNear) / (zNear - zFar), 2.0f * zFar * zNear / (zNear - zFar),
        0, 0, -1, 0);

    vec3 f = normalize(center - eye);
    vec3 s = cross(f, up);
    vec3 u = cross(normalize(s), f);
    mat4 lookat_matrix = mat4(
        s.x,  u.x, -f.x,   0,
        s.y,  u.y, -f.y,   0,
        s.z,  u.z, -f.z,   0,
          0,    0,    0,   1);

    mat4 translation_matrix = mat4(
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        -eye.x, -eye.y, -eye.z, 1);
    

    gl_Position = perspective_matrix * (lookat_matrix * translation_matrix) * position;
    // close ! gl_Position = perspective_matrix * (lookat_matrix * translation_matrix) * position;
}