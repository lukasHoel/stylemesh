from argparse import ArgumentParser

import numpy as np
import torch
from tqdm.auto import tqdm

from PIL import Image

import os

from os.path import join

from data.scannet_single_scene_dataset import ScanNet_Single_House_Dataset
from model.texture.utils import get_rgb_transform, get_uv_transform, get_label_transform
from torchvision.transforms import Compose


def main(opt):
    # get images
    extensions = ["jpg", "png"]
    files = os.listdir(opt.image_dir)
    files = [f for f in files if any(f.endswith(x) for x in extensions) and 'masked' not in f]
    # files = [f for f in files if not "tex" in f and "V2" in f]
    # files = [f for f in files if "styled" in f and not "reprojected" in f and not "other" in f]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))
    # files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[1]))
    # files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[1]) * 100 + int(x.split(".")[0].split("_")[2]))
    files = [join(opt.image_dir, f) for f in files]

    transform_rgb = Compose([
        get_rgb_transform()
    ])
    transform_label = get_label_transform()
    transform_uv = get_uv_transform()
    d = ScanNet_Single_House_Dataset(root_path=opt.data_dir,
                                     scene=opt.scene,
                                     verbose=True,
                                     transform_rgb=transform_rgb,
                                     transform_label=transform_label,
                                     transform_uv=transform_uv,
                                     load_uvs=True,
                                     load_uv_pyramid=False,
                                     pyramid_levels=1,
                                     min_pyramid_depth=0.25,
                                     min_pyramid_height=256,
                                     create_instance_map=False,
                                     crop=False,
                                     crop_size=256,
                                     resize=True,
                                     resize_size=480,
                                     max_images=1000,
                                     min_images=1)

    for f, item in tqdm(zip(files, d), total=len(files)):
        # read styled image and convert it to RGBA
        styled = np.asarray(Image.open(f))
        h, w = styled.shape[:2]
        styled_RGBA = np.dstack((styled, np.zeros((h, w), dtype=np.uint8) + 255))  # Add an alpha channel, fully opaque (255)

        # get mask for this image and make it to 4 channels (RGBA)
        mask = item[12].cpu().detach()
        mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), (h, w), mode='bilinear').squeeze() > 0
        mask4 = torch.stack([mask]*4, dim=2).numpy()

        # mask styled image to be transparent/black at masked regions
        styled_RGBA = np.where(mask4, styled_RGBA, np.zeros_like(styled_RGBA))

        # save image as image with transparency (.png file)
        styled_RGBA = Image.fromarray(styled_RGBA)
        parts = f.split('.')
        name = '.'.join(parts[:-1])
        out_file = f"{name}_masked.png"
        styled_RGBA.save(out_file)
        print(out_file)


if __name__ == '__main__':
    parser = ArgumentParser()

    # add all custom flags
    parser.add_argument('--image_dir', required=True, help="path to images")
    parser.add_argument('--data_dir', required=True, help="path to scannet dataset")
    parser.add_argument('--scene', required=True, help="scannet scene", default='scene0481_00_extremeAndGoodAngles')

    # parse arguments given from command line (implicitly takes the args from main...)
    args = parser.parse_args()

    # run program with args
    main(args)
