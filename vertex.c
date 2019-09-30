#version 450

in vec4 position;
in vec4 color;

uniform float x;
uniform float y;
uniform float z;

uniform mat4 modelview_matrix;
uniform mat4 projection_matrix;

out flat vec4 ex_color;

void main() {
    gl_Position = transpose(projection_matrix) * modelview_matrix * position;
    float size = 2.50f * atan(1.0f, length(vec3(x, y, z) - position.xyz));
    if (size < 1.0f) {
        size = 1.0f;
    }

    ex_color = vec4(color.xyz + 0.0f * color.xyz, 1);
    gl_PointSize = size;
}