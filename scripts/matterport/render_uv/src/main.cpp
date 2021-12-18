#include <iostream>

#include "mp_renderer.h"
#include "mp_parser.h"
#include "segmentation_provider.h"
#include "mesh_transformer.h"

#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include <filesystem>

#include <glm/glm.hpp>

/*
        Input:
            - <path/to/Matterport3D/data/v1/scans>
            - <scanID>
            - optional 1: flip, w, h
            - optional 2: rgb_texture
            - optional 3: interactive

        Output:
            - for each pose in region 0: [uv, angle, depth, vertex_color] or [textured_images] depending on the input


        Example call:
        ./matterport_renderer ...
*/

int main(int argc, char** argv){
    
    if(argc != 4 && argc != 7 && argc != 10){
        std::cout << "Usage: " << argv[0] << " <path/to/Matterport3D/data/v1/scans> <scanID> <region_index> [<flip=0> <w=-1> <h=-1> <rgb_texture.jpg> <out_rgb_texture> <interactive>]" << std::endl;
        return EXIT_FAILURE;
    }

    // get size for renderer
    int flip = 0;
    int w = -1;
    int h = -1;
    if(argc >= 6){
        flip = std::stoi(argv[4]);
        w = std::stoi(argv[5]);
        h = std::stoi(argv[6]);
    }

    // read matterport path arguments
    string path(argv[1]);
    string scanID(argv[2]);
    string regionIndex(argv[3]);
    int region_index_int = std::stoi(regionIndex);

    // create MP-Parser from the .house file of this scan
    string pathToHouseFile = path + "/" + scanID + "/house_segmentations/" + scanID + "/house_segmentations/" + scanID + ".house";
    MP_Parser mp(pathToHouseFile.c_str());
    std::cout << "parsed .house file" << std::endl;

    try{
        string regionPath = "";

        if(region_index_int != -1){
            // select region
            regionPath = path + "/" + scanID + "/region_segmentations/" + scanID + "/region_segmentations/region" + regionIndex;

        } else {
            // select complete house
            regionPath = path + "/" + scanID + "/house_segmentations/" + scanID + "/house_segmentations/" + scanID;
        }

        // find original mesh with rgb colors and uv mesh
        string rgb_mesh_path = regionPath + ".ply";
        string uv_mesh_path = regionPath + "_uvs_blender.ply";

        if(argc >= 8){
            // only render from the specified texture file
            // TODO add mode interactive!!
            string rgb_texture(argv[7]);
            string out_rgb_texture(argv[8]);
            int interactive = std::stoi(argv[9]);

            MP_Renderer texture_renderer(uv_mesh_path, mp, region_index_int, w, h, Renderer::rgb, rgb_texture);
            std::filesystem::create_directory(out_rgb_texture);

            if(! interactive){
                texture_renderer.renderImages(out_rgb_texture, "textured.png", flip);
            } else {
                texture_renderer.renderInteractive(texture_renderer.getStartExtrinsics(), texture_renderer.getProjection(),
                                                    out_rgb_texture, out_rgb_texture,
                                                    "textured.png", "pose.txt");
                
                //MP_Renderer vertex_color_renderer(rgb_mesh_path, mp, region_index_int, w, h, Renderer::vertex_color);
                //vertex_color_renderer.renderInteractive(texture_renderer.getStartExtrinsics(), texture_renderer.getProjection(),
                //                                        out_rgb_texture, out_rgb_texture,
                //                                        "textured.png", "pose.txt");
            }
            
        } else {
            //render uvs, angles, depth and vertex_colors

            // root outdir for everything
            string outdir = path + "/" + scanID + "/rendered/region_" + regionIndex;
            std::filesystem::create_directories(outdir);

            string suffix = (w != -1 || h != -1) ? ("_" + std::to_string(w) + "_" + std::to_string(h)) : "";

            // render poses
            MP_Renderer pose_renderer(rgb_mesh_path, mp, region_index_int, w, h, Renderer::pose);
            string outdir_pose = outdir + "/pose";
            std::filesystem::create_directory(outdir_pose);
            pose_renderer.renderImages(outdir_pose, "pose.txt", flip);
            std::cout << "Render poses completed" << std::endl;

            // copy color files of selected region
            MP_Renderer copy_color_files(rgb_mesh_path, mp, region_index_int, w, h, Renderer::copy_color);
            string outdir_copy_color = outdir + "/color";
            string color_src = path + "/" + scanID + "/matterport_color_images/" + scanID + "/matterport_color_images";
            std::filesystem::create_directory(outdir_copy_color);
            copy_color_files.copyImages(color_src, outdir_copy_color);
            std::cout << "Copy color completed" << std::endl;

            // copy depth files of selected region
            MP_Renderer copy_depth_files(rgb_mesh_path, mp, region_index_int, w, h, Renderer::copy_depth);
            string outdir_copy_depth = outdir + "/depth";
            string depth_src = path + "/" + scanID + "/matterport_depth_images/" + scanID + "/matterport_depth_images";
            std::filesystem::create_directory(outdir_copy_depth);
            copy_depth_files.copyImages(depth_src, outdir_copy_depth);
            std::cout << "Copy depth completed" << std::endl;

            // render uvs
            MP_Renderer uv_renderer(uv_mesh_path, mp, region_index_int, w, h, Renderer::uvmap);
            string outdir_uv = outdir + "/uv" + suffix;
            std::filesystem::create_directory(outdir_uv);
            uv_renderer.renderImages(outdir_uv, "uvs.npy", flip);
            std::cout << "Render uvs completed" << std::endl;

            // render angles
            MP_Renderer angle_renderer(uv_mesh_path, mp, region_index_int, w, h, Renderer::angle);
            string outdir_angle = outdir + "/angle" + suffix;
            std::filesystem::create_directory(outdir_angle);
            angle_renderer.renderImages(outdir_angle, "angle.npy", flip);
            std::cout << "Render angles completed" << std::endl;

            // render depth
            MP_Renderer depth_renderer(uv_mesh_path, mp, region_index_int, w, h, Renderer::depth);
            string outdir_depth = outdir + "/rendered_depth" + suffix;
            std::filesystem::create_directory(outdir_depth);
            depth_renderer.renderImages(outdir_depth, "rendered_depth.npy", flip);
            std::cout << "Render depth completed" << std::endl;

            // render vertex_colors
            MP_Renderer vertex_color_renderer(rgb_mesh_path, mp, region_index_int, w, h, Renderer::vertex_color);
            string outdir_vertex_color = outdir + "/vertex_color" + suffix;
            std::filesystem::create_directory(outdir_vertex_color);
            vertex_color_renderer.renderImages(outdir_vertex_color, "vertex_color.png", flip);
            std::cout << "Render vertex_color completed" << std::endl;
        }

        std::cout << "Render all images completed" << std::endl;

    } catch(const exception& e){
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    
    

    return EXIT_SUCCESS;
}