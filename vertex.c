#version 450

in vec4 position;

uniform float x;
uniform float y;
uniform float z;

uniform mat4 modelview_matrix;
uniform mat4 perspective_matrix;

void main()
{
    gl_Position = transpose(perspective_matrix) * modelview_matrix * position;
}