# StyleMesh

This is the official repository that contains source code for [StyleMesh](https://stylemesh.github.io).

[Arxiv](https://arxiv.org/abs/2112.01530)[Project Page](https://lukashoel.github.io/stylemesh/)[Video](https://www.youtube.com/watch?v=ZqgiTLcNcks)

If you find StyleMesh useful for your work please cite:
```
@misc{höllein2021stylemesh,
      title={StyleMesh: Style Transfer for Indoor 3D Scene Reconstructions}, 
      author={Lukas Höllein and Justin Johnson and Matthias Nießner},
      year={2021},
      eprint={2112.01530},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Preprocessing
The following steps are necessary to prepare all data.

- Download VGG Model: https://bethgelab.org/media/uploads/pytorch_models/vgg_conv.pth
- Download Matterport and Scannet datasets (we use train scannet scenes only, see [scripts/scannet/scannet_train.txt](scripts/scannet/scannet_train.txt) for a complete list)
- Prepare Scannet Scenes: see [scripts/scannet/prepare_data](scripts/scannet/prepare_data) and [scripts/scannet/filter/filter_blurry.py](scripts/scannet/filter/filter_blurry.py). This extracts RGB images and removes blurry frames.
- Compile Matterport and Scannet renderers: see [scripts/matterport/render_uv](scripts/matterport/render_uv) and [scripts/scannet/render_uv](scripts/scannet/render_uv) for more details.
- Create texture parametrization for each mesh (requires blender): see [scripts/matterport/create_uvs.py](scripts/matterport/create_uvs.py) and [scripts/scannet/create_uvs.py](scripts/scannet/create_uvs.py). This saves a new version of each mesh file, that contains a uv texture parametrization.
- Render uv maps for Mattertport and Scannet: see [scripts/matterport/render_uvs.py](scripts/matterport/render_uvs.py) and [scripts/scannet/render_uvs.py](scripts/scannet/render_uvs.py). This preprocessing step speeds up the optimization by precomputing the texture lookups for each viewpoint.
- Change dataset and vgg-model paths in [scripts/train/*](scripts/train) to your local machine locations
- Create a conda environment and use it for all further executions: see [requirements.txt](requirements.txt)

## Texture Optimization
The following steps allow to optimize a texture for a specific scene/style.
You can easily select scene/style by modifying the corresponding values in the scripts (--scene for ScanNet and additionally --region for Matterport).
It also allows to fine-tune loss weights, if you want to experiment with your own settings.

All style images that are used in the main paper are listed in [styles](styles).

By default, run files (texture, hparams, logging) are saved in [style-mesh/lightning_logs](lightning_logs)

- Optimize Scannet: see [scripts/train/optimize_texture_scannet*.sh](scripts/train)
- Optimize Matterport: see [scripts/train/optimize_texture_matterport*.sh](scripts/train)

The suffix "with_angle_and_depth" is used for comparisons in Fig. 4,5,6,7,8,9,11.
The suffixes "only2D" and "with_angle" are used for ablation study in Fig. 7.
The suffix "dip" is used for the DIP baseline in Fig. 4,5,6

## Render optimized Texture
You can render images with Mipmapping and Shading with our OpenGL renderers for each dataset.
Alternatively, you can use the generated texture files after each optimization together with the generated meshes
and view the textured mesh in any mesh viewer (e.g. Meshlab or Blender).

- Renderer Scannet: see [scripts/scannet/render_mipmap_scannet.py](scripts/scannet/render_mipmap_scannet.py)
- Renderer Matterport: see [scripts/matterport/render_mipmap_matterport.py](scripts/matterport/render_mipmap_matterport.py)

## Evaluate Reprojection Error
We use the file [scripts/eval/eval_image_folders.py](scripts/eval/eval_image_folders.py) for calculation of the reprojection error (Tab. 1)

## Evaluate Circle Metric
We use the file [scripts/eval/measure_circles.py](scripts/eval/measure_circles.py) for calculation of the circle's metric (Tab. 2, Fig. 8)
