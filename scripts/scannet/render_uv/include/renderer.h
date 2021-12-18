#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <opencv2/opencv.hpp>

#include "model.h"
#include "camera.h"

const unsigned int DEF_WIDTH = 1280;
const unsigned int DEF_HEIGHT = 1024;

constexpr float kNearPlane{0.1f};
constexpr float kFarPlane{10.0f};

class Renderer {
public:
    Renderer(string const &pathToMesh, int width, int height, int shader_mode, const std::string rgb_texture = "");
    ~Renderer();
    void renderInteractive(glm::mat4 start_extrinsics, glm::mat4 projection,
                           const std::string rgb_dir, const std::string pose_dir,
                           const std::string rgb_suffix = "png", const std::string pose_suffix = "txt");
    int init();

    void readRGB(cv::Mat& image, bool flip = false);
    void readDepth(cv::Mat& image, bool flip = false);

    void saveUV(std::string filename, bool flip = false);

    Model* m_model = nullptr;

    static const int uvmap = 0;
    static const int angle = 1;
    static const int depth = 2;
    static const int rgb = 3;
    static const int vertex_color = 4;
protected:

    int shader_mode;
    int m_buffer_width = DEF_WIDTH;
    int m_buffer_height = DEF_HEIGHT;
    bool m_initialized = false;
    string rgb_texture;

    GLFWwindow* m_window = nullptr;
    Shader* m_shader = nullptr;

    // unsigned int fbo;
    // unsigned int texture;

    GLuint fbo, origin_color, rbo, image_uv;
    GLenum DrawBuffers[2] = {GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1};

    void render(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection);
};