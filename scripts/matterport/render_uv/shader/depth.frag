#version 330 core

// FRAGMENT SHADER
// Takes in the RGB color and uses it fully opaque

out vec3 color;

float near = 0.1;
float far  = 10.0;

float LinearizeDepth(float depth)
{
    float z = depth * 2.0 - 1.0; // back to NDC
    return (2.0 * near * far) / (far + near - z * (far - near));
}

void main( )
{
    color = vec3(LinearizeDepth(gl_FragCoord.z));
}