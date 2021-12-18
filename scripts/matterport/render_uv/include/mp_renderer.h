#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <opencv2/opencv.hpp>

#include "mp_parser.h"
#include "renderer.h"

class MP_Renderer : public Renderer {

public:
    MP_Renderer(string const &pathToMesh, MP_Parser const &mp_parser, int region_index, int w, int h, int shader_mode, const std::string rgb_texture = "");
    ~MP_Renderer();
    void renderImages(const std::string save_path = "", const std::string suffix = "npy", bool flip = false);
    void copyImages(const std::string src_dir, const std::string save_path = "");

    glm::mat4 getProjection();
    glm::mat4 getStartExtrinsics();

private:
    MP_Parser mp_parser;
    int region_index;
    glm::mat4 m_projection;
    glm::mat4 m_start_extrinsics;
};