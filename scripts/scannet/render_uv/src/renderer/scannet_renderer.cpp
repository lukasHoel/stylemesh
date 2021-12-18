#include "scannet_renderer.h"
#include "util.h"
#include <set>
#include <tuple>

Scannet_Renderer::Scannet_Renderer(string const &pathToMesh, Scannet_Parser& parser, int w, int h, int shader_mode, const std::string rgb_texture) : Renderer(pathToMesh, w, h, shader_mode, rgb_texture), parser(parser) {
    glm::mat3 intr = parser.getIntrinsics();
    projection = camera_utils::perspective(intr, parser.getWidth(), parser.getHeight(), kNearPlane, kFarPlane);
    // projection[2][0] = 0; // remove cx
    // projection[2][1] = 0; // remove cy
}

Scannet_Renderer::~Scannet_Renderer() {}

glm::mat4 Scannet_Renderer::getProjection(){
    return projection;
}

void Scannet_Renderer::renderTrajectory(const std::string save_path, const std::string suffix, bool flip){
    
    for(int i=0; i<parser.getPoseCount(); i++){
        glm::mat4 extr = parser.getExtrinsics(i); // is 4x4 matrix with format [R1 R2 R3 | T]

        // adapted the view matrix calculation from official ScanNet repository: https://github.com/ScanNet/ScanNet/blob/master/AnnotationTools/ProjectAnnotations/Visualizer.cpp

        // create a camera from the extrinsic matrix
        glm::vec3 eye = glm::vec3(extr[3][0], extr[3][1], extr[3][2]);      // T
        glm::vec3 right = glm::vec3(extr[0][0], extr[0][1], extr[0][2]);    // R1
        glm::vec3 up = glm::vec3(extr[1][0], extr[1][1], extr[1][2]);       // R2
        glm::vec3 look = glm::vec3(extr[2][0], extr[2][1], extr[2][2]);     // R3

        right = glm::normalize(right);
        up = glm::normalize(up);
        look = glm::normalize(look);

        // define the camera as "lookAt" matrix
        glm::mat4 view = glm::mat4(1.0f);
        
        // first row
        view[0][0] = right.x;
        view[1][0] = right.y;
        view[2][0] = right.z;
        view[3][0] = - glm::dot(right, eye);

        // second row
        view[0][1] = up.x;
        view[1][1] = up.y;
        view[2][1] = up.z;
        view[3][1] = - glm::dot(up, eye);

        // third row --> need to multiply with -1. Assumably, because we use OpenGL and handiness of coordinate systems is different (original code is for D3D)
        view[0][2] = -look.x;
        view[1][2] = -look.y;
        view[2][2] = -look.z;
        view[3][2] =  glm::dot(look, eye);

        render(glm::mat4(1.0f), view, projection);

        std::stringstream filename;
        filename << save_path << "/" << parser.getPoseName(i) << "." << suffix;

        if(shader_mode != rgb && shader_mode != vertex_color){
            saveUV(filename.str(), flip);
        } else {
            // read image into openCV matrix
            cv::Mat colorImage;
            readRGB(colorImage, flip);
            cv::imwrite(filename.str(), colorImage);
        }
        

        // read image into openCV matrix
        // cv::Mat colorImage;
        // readRGB(colorImage);
        // cv::imshow("color image", colorImage);
        // cv::waitKey(0);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // show image in window
        glfwSwapBuffers(m_window);
    
    }
    
}