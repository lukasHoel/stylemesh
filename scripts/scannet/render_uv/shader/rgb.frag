#version 400

uniform sampler2D texture_rgb;

in vec2 uv;

in vec3 FragPos;
in vec3 Normal;

out vec4 color;

void main( )
{
    color = texture(texture_rgb, uv);

    vec3 lightColor = vec3(1.0, 1.0, 1.0);

    float ambientStrength = 0.4;
    vec3 ambient = ambientStrength * lightColor;

    vec3 lightPos = FragPos + vec3(0.1, 0.1, 0.1);
    vec3 norm = normalize(Normal);
    vec3 lightDir = normalize(lightPos - FragPos);  
    float diff = max(dot(norm, lightDir), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 result = (ambient + diffuse) * color.xyz;
    color = vec4(result, 1.0);
}