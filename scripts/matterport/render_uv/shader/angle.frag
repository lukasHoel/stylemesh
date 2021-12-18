#version 330 core

// FRAGMENT SHADER
// Takes in the RGB color and uses it fully opaque

in vec3 normalV;
in vec3 fragPos;

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
    vec3 norm = normalize(normalV);
    vec3 viewDir = normalize(-fragPos);
    float diff = max(dot(norm, viewDir), 0.0);

    // float depth = LinearizeDepth(gl_FragCoord.z);
    // float depth = 1.0f;

    // float w = (diff * diff) / (depth * depth);

    float w = diff;

    color = vec3(w);
}