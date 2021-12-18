#version 400

uniform sampler2D texture_rgb;

in vec2 uv;
out vec3 color;

void main( )
{
    vec2 mipmapLevel = textureQueryLod(texture_rgb, uv);
    color = vec3(uv, mipmapLevel.x);

    // color = vec3(uv, 0.0f);
}