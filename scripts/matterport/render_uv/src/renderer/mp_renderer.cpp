#include "mp_renderer.h"
#include "util.h"

#include <filesystem>

int get_width(int w, int h, int orig_w, int orig_h) {
    if(w != -1){
        // if concrete value is specified, use it
        return w;
    } else if(h == -1){
        // if there is also no concrete value for h, return original width
        return orig_w;
    } else {
        // if there is concrete value for h, return scaled width that retains original aspect ratio
        double aspect = 1.0 * orig_w / orig_h;
        return static_cast<int>(h * aspect);
    }
}

int get_height(int w, int h, int orig_w, int orig_h) {
    if(h != -1){
        // if concrete value is specified, use it
        return h;
    } else if(w == -1){
        // if there is also no concrete value for w, return original height
        return orig_h;
    } else {
        // if there is concrete value for w, return scaled height that retains original aspect ratio
        double aspect = 1.0 * orig_w / orig_h;
        return static_cast<int>(w * aspect);
    }
}

MP_Renderer::MP_Renderer(string const &pathToMesh, MP_Parser const &mp_parser, int region_index, int w, int h, int shader_mode, const std::string rgb_texture): 
                    mp_parser(mp_parser),
                    region_index(region_index),
                    Renderer(pathToMesh,
                             get_width(w, h, mp_parser.regions[region_index == -1 ? 0 : region_index]->panoramas[0]->images[0]->width, mp_parser.regions[region_index == -1 ? 0 : region_index]->panoramas[0]->images[0]->height),
                             get_height(w, h, mp_parser.regions[region_index == -1 ? 0 : region_index]->panoramas[0]->images[0]->width, mp_parser.regions[region_index == -1 ? 0 : region_index]->panoramas[0]->images[0]->height),
                             shader_mode,
                             rgb_texture) {

    auto image = mp_parser.regions[region_index == -1 ? 0 : region_index]->panoramas[0]->images[0];

    // set projection to value of first frame
    glm::mat3 intr = glm::make_mat3(image->intrinsics);
    if(m_buffer_width != w || m_buffer_height != h){
        // if we choose another width/height than what is saved under image->width and image->height, we have to modify intrinsics accordingly!
        intr[0][0] *= 1.0 * m_buffer_width / image->width;
        intr[1][1] *= 1.0 * m_buffer_height / image->height;
        intr[0][2] *= 1.0 * m_buffer_width / image->width;
        intr[1][2] *= 1.0 * m_buffer_height / image->height;
    }
    m_projection = camera_utils::perspective(intr, m_buffer_width, m_buffer_height, kNearPlane, kFarPlane);

    // set start extrinsics
    m_start_extrinsics = glm::transpose(glm::make_mat4(image->extrinsics));
}

glm::mat4 MP_Renderer::getProjection(){
    return m_projection;
}

glm::mat4 MP_Renderer::getStartExtrinsics(){
    return m_start_extrinsics;
}

MP_Renderer::~MP_Renderer() = default;

/*
    - The region_X.ply are still in world-coordinates, e.g. region0 is left and region6 is centered.
    - Thus I can use the camera extrinsics/intrinsics also for the regions only
    - This means, that I can use regions + vseg file (Alternative: use whole house mesh and parse fseg file instead of vseg)
    - For each image (matterport_color_images.zip) we have a corresponding extrinsic/intrinsic file with same name
        --> Use this for calculating the view and projection matrices
        --> But these parameters are distorted, e.g. the intrinsic files contain arbitrary 3x3 matrix
        --> This is solved in undistorted_camera_parameters.zip
        --> The same values as in undistorted_camera_parameters.zip are also present in the .house file
        --> Just use the extrinsic/intrinsic parameters from the .house file!
        --> Note that the extrinsic parameters differ in the .house file and in the undistorted file. What is correct?
    - Find out which image corresponds to which region. It only makes sense to use the images for the corresponding region
        --> Otherwise we would look at nothing because in that case the region is not present
        --> Can I do it like this? Parse .house file and go like this: Image Name --> Panorama Index --> Region Index ? --> Yes!
*/
void MP_Renderer::renderImages(const std::string save_path, const std::string suffix, bool flip){

    for(int r=0; r<mp_parser.regions.size(); r++){
        
        if(region_index != -1 && region_index != r){
            continue;
        }

        for(int i=0; i<mp_parser.regions[r]->panoramas.size(); i++){
            for(MPImage* image : mp_parser.regions[r]->panoramas[i]->images){

                glm::mat4 extr = glm::transpose(glm::make_mat4(image->extrinsics));
                glm::mat3 intr = glm::make_mat3(image->intrinsics);

                int w = image->width;
                int h = image->height;
                if(m_buffer_width != w || m_buffer_height != h){
                    // if we choose another width/height than what is saved under image->width and image->height, we have to modify intrinsics accordingly!
                    intr[0][0] *= 1.0 * m_buffer_width / w;
                    intr[1][1] *= 1.0 * m_buffer_height / h;
                    intr[0][2] *= 1.0 * m_buffer_width / w;
                    intr[1][2] *= 1.0 * m_buffer_height / h;
                }

                glm::mat4 projection = camera_utils::perspective(intr, m_buffer_width, m_buffer_height, kNearPlane, kFarPlane);

                std::stringstream filename;
                filename << save_path << "/" << image->color_filename << "." << suffix;

                if(shader_mode == pose) {
                    savePose(filename.str(), extr);

                    // always save original intrinsics, not the modified one
                    filename << ".intrinsics.txt";
                    saveIntrinsics(filename.str(), glm::make_mat3(image->intrinsics), w, h);

                } else {
                    // render image
                    render(glm::mat4(1.0f), extr, projection);

                    if(shader_mode != rgb && shader_mode != vertex_color){
                        saveUV(filename.str(), flip);
                    } else {
                        // read image into openCV matrix
                        cv::Mat colorImage;
                        readRGB(colorImage, flip);
                        cv::imwrite(filename.str(), colorImage);
                    }
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
    }
}

void MP_Renderer::copyImages(const std::string src_dir, const std::string save_path){

    for(int r=0; r<mp_parser.regions.size(); r++){
    
        if(region_index != -1 && region_index != r){
            continue;
        }

        for(int i=0; i<mp_parser.regions[r]->panoramas.size(); i++){
            for(MPImage* image : mp_parser.regions[r]->panoramas[i]->images){

                std::stringstream dst;
                dst << save_path << "/";

                std::stringstream src;
                src << src_dir << "/";

                if(shader_mode == copy_color) {
                    dst << image->color_filename;
                    src << image->color_filename;

                } else if(shader_mode == copy_depth) {
                    dst << image->depth_filename;
                    src << image->depth_filename;
                }

                std::filesystem::copy_file(src.str(), dst.str(), std::filesystem::copy_options::skip_existing);
            }
        }
    }
}