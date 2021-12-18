#pragma once

#include <iostream>
#include <string>

#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <cstring>
#include <dirent.h>

#include <glm/glm.hpp>

#include <map>

#include <cmath>
#define PI 3.14159265

class Scannet_Parser {
public:
    Scannet_Parser(std::string pose_dir, std::string intrinsics_file, bool intrinsics_color=true): pose_dir(pose_dir), intrinsics_file(intrinsics_file) {
        // Parse pose directory --> get overall number of pose files
        int len;
        struct dirent *pDirent;
        DIR *pDir = NULL;
        pose_count = 0;

        pDir = opendir(pose_dir.c_str());
        if (pDir != NULL)
        {
            while ((pDirent = readdir(pDir)) != NULL)
            {
                len = strlen (pDirent->d_name);
                if (len >= 4)
                {
                    if (strcmp (".txt", &(pDirent->d_name[len - 4])) == 0)
                    {
                        std::string name(pDirent->d_name);
                        int index = std::stoi(name.substr(0, len - 4));
                        indexToName[pose_count] = index;
                        pose_count++;
                    }
                }
            }
            closedir (pDir);
        }

        // Parse intrinsics file --> extract the intrinsics information
        std::ifstream intrinsics_file_stream(intrinsics_file);
        char line_data[300];
        while(1)
        {
            intrinsics_file_stream.getline(line_data, 300);

            if ( intrinsics_file_stream.eof()){
                break;
            }
            
            std::string line_data_str(line_data);
            line_data_str = line_data_str.substr(line_data_str.find(" = ")+3);

            std::istringstream iss;
            iss.str(line_data_str);

            // parse intrinsics matrix and width+height depending on color or depth flag
            if (intrinsics_color) {
                if (strstr(line_data,"fx_color") != NULL) {
                    iss >> fx;
                }
                if (strstr(line_data,"fy_color") != NULL) {
                    iss >> fy;
                }
                if (strstr(line_data,"mx_color") != NULL) {
                    iss >> cx;
                }
                if (strstr(line_data,"my_color") != NULL) {
                    iss >> cy;
                }
                if (strstr(line_data,"colorWidth") != NULL) {
                    iss >> width;
                }
                if (strstr(line_data,"colorHeight") != NULL) {
                    iss >> height;
                }
            } else {
                if (strstr(line_data,"fx_depth") != NULL) {
                    iss >> fx;
                }
                if (strstr(line_data,"fy_depth") != NULL) {
                    iss >> fy;
                }
                if (strstr(line_data,"mx_depth") != NULL) {
                    iss >> cx;
                }
                if (strstr(line_data,"my_depth") != NULL) {
                    iss >> cy;
                }
                if (strstr(line_data,"depthWidth") != NULL) {
                    iss >> width;
                }
                if (strstr(line_data,"depthHeight") != NULL) {
                    iss >> height;
                }
            }

            // parse axisAlignment row
            if (strstr(line_data,"axisAlignment") != NULL) {
                for (int row=0; row<4; row++) {
                    for (int col=0; col<4; col++) {
                        iss >> axis_alignment[row][col];
                        if(row != 3 && col != 3) {
                            iss.ignore(1,' ');        
                        }
                    }
                }
            }

        }
    }

    int getPoseCount(){
        return pose_count;
    }

    int getPoseName(int index){
        return indexToName[index];
    }

    glm::mat4 getExtrinsics(int index){
        char text_file_name[360];
        sprintf(text_file_name, "%s/%d.txt", pose_dir.c_str(), indexToName[index]);
        std::ifstream extrinsics_file_stream(text_file_name);

        char line_data[300];
        glm::mat4 extrinsics;
        int row = 0;

        while(1)
        {
            extrinsics_file_stream.getline(line_data, 300);

            if ( extrinsics_file_stream.eof()){
                break;
            }
            std::string line_data_str(line_data);
            std::istringstream iss;
            iss.str(line_data_str);

            iss >> extrinsics[row][0];
            iss.ignore(1,' ');
            iss >> extrinsics[row][1];
            iss.ignore(1,' ');
            iss >> extrinsics[row][2];
            iss.ignore(1,' ');
            iss >> extrinsics[row][3];

            row++;
        }

        return glm::transpose(extrinsics); // convert to column-mayor
    }

    glm::mat3 getIntrinsics(){
        glm::mat3 K(0.0);
        K[0][0] = fx;
        K[1][1] = fy;
        K[0][2] = cx;
        K[1][2] = cy;
        K[2][2] = 1;

        return K;
    }

    int getWidth(){
        return width;
    }

    int getHeight(){
        return height;
    }

    glm::mat4 getAxisAlignment(){
        return axis_alignment;
    }

private:
    // Directory variables
    std::string pose_dir;
    std::string intrinsics_file;

    // pose dir attributes
    int pose_count; // how many pose files are in the pose_dir
    std::map<int, int> indexToName; // what is the file_name number of the i-th pose file (i.e. the 2nd pose file could be 40.txt if the pose_dir contains every 40-th pose file)

    // intrinsics
    int width, height;
    float fx, fy, cx, cy;
    glm::mat4 axis_alignment;
};