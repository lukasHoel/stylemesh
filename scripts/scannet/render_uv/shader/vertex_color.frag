#version 400

uniform sampler2D texture_rgb;

in vec3 vcolor;
out vec4 color;

void main( )
{
    color = vec4(vcolor, 1.0f);
}