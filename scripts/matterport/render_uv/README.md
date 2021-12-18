# Matterport Renderer
The matterport renderer allows to render uv maps, angle maps and depth maps out of the original Matterport3D camera images.
You can also interactively record new trajectories and render RGB images of it from the textured mesh.
See src/main.cpp for more details about usage.

## Prerequisites
- at least c++11 capable compiler (gcc, ...)
   - Check if installed: gcc --version
- cmake
   - sudo apt-get install cmake
- At least the following 3 directories unzipped for each Matterport3D scan that shall be used:
   - matterport_color_images
   - region_segmentations
   - house_segmentations

## Setup (Dependencies for this program)

    sudo apt-get install assimp-utils libassimp-dev
    sudo apt-get install libopencv-dev
    sudo apt-get install libglm-dev
    sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
    sudo apt-get install libglfw3-dev libglfw3
    sudo apt-get install libglew-dev

## Build the Program
    mkdir build
    cd build
    cmake ..
    make