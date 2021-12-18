#include <iostream>

#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <filesystem>

#include <glm/glm.hpp>

#include "scannet_parser.h"
#include "scannet_renderer.h"

/*
        Input: paths to...
            - mesh.obj with the uv coordinates in it
            - dir with all the pose .txt files that are produced by the prepare_data script --> loop through each file and render from it / only load the mesh once during this process
            - intrinsics file (is called <scene>.txt): contains fx/fy/cx/cy and width/height 
            - output folder where the uv map renderings should be placed (i.e. images/uv)

        Output:
            - for each pose renders the image in same width/height as the color/depth/label images, but with the uv maps as content
            - saves each pose as rendered .png file to the output folder with naming consistent to color/depth/label names, i.e. 0.png, 20.png etc.


        Example call:
        ./scannet_uv_renderer ~/datasets/ScanNet/scans/scene0568_01/scene0568_01_vh_clean_2_reduce0.5_uvatlas.obj ~/datasets/ScanNet/images/scene0568_01/pose ~/datasets/ScanNet/scans/scene0568_01/scene0568_01.txt ~/datasets/ScanNet/images/scene0568_01/uv_new
*/

int main(int argc, char** argv){

    if(argc != 5 && argc != 8 && argc != 9 && argc != 11){
        std::cout << "Usage: " << argv[0] << " <mesh_with_uv.obj> <pose_dir> <scene.txt> <output_dir> [<flip=0> <w=640> <h=480> <rgb_texture.jpg> <interactive=0> <mesh_with_colors.obj>]" << std::endl;
        return EXIT_FAILURE;
    }

    // get size for renderer
    // twice the size of camera images: 640, 480 | w/h in color intrinsics file: 1296, 968 | 'big': 2560, 1920
    int flip = 0;
    int w = 640;
    int h = 480;
    if(argc >= 8){
        flip = std::stoi(argv[5]);
        w = std::stoi(argv[6]);
        h = std::stoi(argv[7]);
    }

    string mesh_file(argv[1]);
    string pose_dir(argv[2]);
    string intrinsics_file(argv[3]);
    string output_dir(argv[4]);

    std::filesystem::create_directory(output_dir);

    try{
        Scannet_Parser parser(pose_dir, intrinsics_file, true);

        if(argc < 9) {
            // if no rgb_texture.jpg is provided, we render uv/angle and put those in output_dir
            Scannet_Renderer uv_renderer(mesh_file, parser, w, h, Renderer::uvmap);
            uv_renderer.renderTrajectory(output_dir, "npy", flip);

            Scannet_Renderer angle_renderer(mesh_file, parser, w, h, Renderer::angle);
            angle_renderer.renderTrajectory(output_dir, "angle.npy", flip);

            Scannet_Renderer depth_renderer(mesh_file, parser, w, h, Renderer::depth);
            depth_renderer.renderTrajectory(output_dir, "rendered_depth.npy", flip);
        } else {
            string rgb_texture(argv[8]);
            int interactive = 0;
            if(argc >= 10){
                interactive = std::stoi(argv[9]);
            }

            if(! interactive){
                // if rgb_texture.jpg is provided, we render rgb from that texture and put those in output_dir
                Scannet_Renderer rgb_renderer(mesh_file, parser, w, h, Renderer::rgb, rgb_texture);
                rgb_renderer.renderTrajectory(output_dir, "textured.jpg", flip);
            } else {
                // use rgb_renderer to create the new poses and texture.jpg images and save them in subfolders
                string rgb_dir = output_dir + "/color";
                string novel_pose_dir = output_dir + "/pose";
                std::filesystem::create_directory(rgb_dir);
                std::filesystem::create_directory(novel_pose_dir);                

                string mesh_colors_file(argv[10]);
                Scannet_Renderer rgb_renderer(mesh_colors_file, parser, w, h, Renderer::vertex_color);

                int startIndex = 0;
                glm::mat4 start = parser.getExtrinsics(startIndex);
                std::cout << "Using start position of extrinsic file: " << parser.getPoseName(startIndex) << ".txt" << std::endl;
                rgb_renderer.renderInteractive(start, rgb_renderer.getProjection(), rgb_dir, novel_pose_dir);

                // afterwards, construct new parser, uv, angle and depth renderer to render the missing image types.
                string uv_dir = output_dir + "/uv";
                string depth_dir = output_dir + "/depth";
                std::filesystem::create_directory(uv_dir);
                std::filesystem::create_directory(depth_dir);      
                
                Scannet_Parser parser_novel_poses(novel_pose_dir, intrinsics_file, true);

                Scannet_Renderer uv_renderer(mesh_file, parser_novel_poses, w, h, Renderer::uvmap);
                uv_renderer.renderTrajectory(uv_dir, "npy", true);

                Scannet_Renderer angle_renderer(mesh_file, parser_novel_poses, w, h, Renderer::angle);
                angle_renderer.renderTrajectory(uv_dir, "angle.npy", true);

                Scannet_Renderer depth_renderer(mesh_file, parser_novel_poses, w, h, Renderer::depth);
                depth_renderer.renderTrajectory(uv_dir, "rendered_depth.npy", true);

                // also render the multi size steps
                // TODO: allow to add min, max, steps, aspect as arguments in cmd line
                int min = 256;
                int max = 960;
                int steps = 5;
                int delta = (int) ((max - min) / (steps-1));
                float aspect = 1.0 * 1280 / 960;
                
                for(int i=0; i<steps; i++){
                    int hi = min + i*delta;
                    int wi = (int)(hi * aspect);

                    string uv_i_dir = output_dir + "/uv_" + std::to_string(hi);
                    std::filesystem::create_directory(uv_i_dir);

                    Scannet_Renderer uv_multi_size_renderer(mesh_file, parser_novel_poses, wi, hi, Renderer::uvmap);
                    uv_multi_size_renderer.renderTrajectory(uv_i_dir, "npy", true);

                    Scannet_Renderer angle_multi_size_renderer(mesh_file, parser_novel_poses, wi, hi, Renderer::angle);
                    angle_multi_size_renderer.renderTrajectory(uv_i_dir, "angle.npy", true);

                    Scannet_Renderer depth_multi_size_renderer(mesh_file, parser_novel_poses, wi, hi, Renderer::depth);
                    depth_multi_size_renderer.renderTrajectory(uv_i_dir, "rendered_depth.npy", true);
                }

                // heights = np.linspace(opt.multi_size_min, opt.multi_size_max, num=opt.multi_size_steps)
                // widths = [int(round(h * opt.multi_size_aspect)) for h in heights]


            }
            
        }

    } catch(const exception& e){
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}