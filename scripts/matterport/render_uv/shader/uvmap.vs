#version 400

// VERTEX SHADER 
// Uses the RGB colors instead of texture coordinates

layout ( location = 0 ) in vec3 position;
layout ( location = 1 ) in vec3 normal;
layout ( location = 2 ) in vec2 texCoords;
layout ( location = 3 ) in vec3 tangent;
layout ( location = 4 ) in vec3 bitangent;
layout ( location = 5 ) in vec3 color;

out vec2 uv;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;


void main( )
{
    gl_Position = projection * view * model * vec4( position, 1.0f );
    uv = texCoords;
}