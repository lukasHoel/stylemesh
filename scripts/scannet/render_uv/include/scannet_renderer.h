#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/opencv.hpp>

#include "scannet_parser.h"
#include "renderer.h"

class Scannet_Renderer : public Renderer {
public:
    Scannet_Renderer(string const &pathToMesh, Scannet_Parser& parser, int w, int h, int shader_mode, const std::string rgb_texture = "");
    ~Scannet_Renderer();
    void renderTrajectory(const std::string save_path = "", const std::string suffix = "npy", bool flip = false);
    glm::mat4 getProjection();

private:
    glm::mat4 projection;
    Scannet_Parser parser;
};