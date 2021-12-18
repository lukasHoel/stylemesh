# taken from: https://github.com/krrish94/ENet-ScanNet/blob/master/data/scannet.py

import os
import cv2
import numpy as np
import torchvision
from PIL import Image
from os.path import join
from data.abstract_dataset import Abstract_Dataset, Abstract_DataModule
from model.texture.texture import NeuralTexture
from tqdm.auto import tqdm
import torch


class ScanNet_DataModule(Abstract_DataModule):

    def __init__(self,
                 root_path: str,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 resize_size=(256, 256),
                 shuffle: bool = False,
                 sampler_mode: str = "random",
                 index_repeat: int = 1,
                 verbose: bool = False):

        root_paths = {
            "train": join(root_path, "train/images"),
            "val": join(root_path, "val/images"),
        }

        Abstract_DataModule.__init__(self,
                                     dataset=ScanNetDataset,
                                     root_path=root_paths,
                                     transform_rgb=transform_rgb,
                                     transform_label=transform_label,
                                     transform_uv=transform_uv,
                                     resize_size=resize_size,
                                     verbose=verbose,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=shuffle,
                                     sampler_mode=sampler_mode,
                                     index_repeat=index_repeat,
                                     split_mode="folder")

    def after_create_dataset(self, d, root_path):
        if isinstance(d, ScanNetDataset):
            d.create_data()


class ScanNetDataset(Abstract_Dataset):

    orig_sizes = {
        # in format (h, w)
        "rgb": (240, 320),
        "label": (240, 320),
        "uv": (480, 640)
    }

    stylized_images_sort_key = {
        "default": lambda x: int(x.split(".")[0]),
    }

    def __init__(self,
                 root_path,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 resize_size=(256, 256),
                 pyramid_levels=5,
                 min_pyramid_depth=0.25,
                 min_pyramid_height=256,
                 verbose=False):

        self.set_uv_pyramid_mode(min_pyramid_depth, min_pyramid_height)
        self.set_pyramid_levels(pyramid_levels)

        Abstract_Dataset.__init__(self,
                                  root_path=root_path,
                                  transform_rgb=transform_rgb,
                                  transform_label=transform_label,
                                  transform_uv=transform_uv,
                                  resize_size=resize_size,
                                  verbose=verbose)

    def set_uv_pyramid_mode(self,
                            min_pyramid_depth: float = 0.25,
                            min_pyramid_height: int = 256):
        self.min_pyramid_depth = min_pyramid_depth
        self.min_pyramid_height = min_pyramid_height

    def set_pyramid_levels(self, pyramid_levels):
        self.pyramid_levels = pyramid_levels

    def get_scenes(self):
        return os.listdir(self.root_path)

    def get_colors(self, scene_path, extensions=["jpg", "png"]):
        """
        Return absolute paths to all colors images for the scene (sorted!)
        """
        color_path = join(scene_path, "color")
        sort_key = ScanNetDataset.stylized_images_sort_key["default"]
        if not os.path.exists(color_path) or not os.path.isdir(color_path):
            return []

        colors = os.listdir(color_path)
        colors = [c for c in colors if any(c.endswith(x) for x in extensions)]
        colors = sorted(colors, key=sort_key)
        colors = [join(color_path, f) for f in colors]

        return colors

    def get_depth(self, scene_path):
        """
        Return absolute paths to all depth images for the scene (sorted!)
        """

        # load rendered opengl depth
        def load_rendered_depth(scene_path):
            uv_path = join(scene_path, "uv")

            if not os.path.exists(uv_path) or not os.path.isdir(uv_path):
                return []

            files = sorted(os.listdir(uv_path), key=lambda x: int(x.split(".")[0]))
            return [join(uv_path, f) for f in files if "npy" in f and 'depth' in f]
        rendered_depth_npy = load_rendered_depth(scene_path)

        # load original sensor depth
        depth_path = join(scene_path, "depth")
        if not os.path.exists(depth_path) or not os.path.isdir(depth_path):
            return []

        depth = sorted(os.listdir(depth_path), key=lambda x: int(x.split(".")[0]))
        depth = [join(depth_path, f) for f in depth]

        # choose opengl depth, if sensor depth not available
        if len(depth) == 0:
            self.rendered_depth = True
            return rendered_depth_npy
        else:
            self.rendered_depth = False
            return depth

    def get_extrinsics(self, scene_path):
        """
        Return absolute paths to all extrinsic images for the scene (sorted!)
        """
        extrinsics_path = join(scene_path, "pose")

        if not os.path.exists(extrinsics_path) or not os.path.isdir(extrinsics_path):
            return []

        extrinsics = sorted(os.listdir(extrinsics_path), key=lambda x: int(x.split(".")[0]))
        extrinsics = [join(extrinsics_path, f) for f in extrinsics]

        return extrinsics

    def get_intrinsics(self, scene_path):
        """
        Return 3x3 numpy array as intrinsic K matrix for the scene and (W,H) image dimensions if available
        """
        intrinsics = np.identity(4, dtype=np.float32)
        w = 0
        h = 0
        file = [join(scene_path, f) for f in os.listdir(scene_path) if ".txt" in f]
        if len(file) == 1:
            file = file[0]
            self.intrinsics_file = file
            with open(file) as f:
                lines = f.readlines()
                for l in lines:
                    l = l.strip()
                    if "fx_color" in l:
                        fx = float(l.split(" = ")[1])
                        intrinsics[0,0] = fx
                    if "fy_color" in l:
                        fy = float(l.split(" = ")[1])
                        intrinsics[1,1] = fy
                    if "mx_color" in l:
                        mx = float(l.split(" = ")[1])
                        intrinsics[0,2] = mx
                    if "my_color" in l:
                        my = float(l.split(" = ")[1])
                        intrinsics[1,2] = my
                    if "colorWidth" in l:
                        w = int(l.split(" = ")[1])
                    if "colorHeight" in l:
                        h = int(l.split(" = ")[1])

        return intrinsics, (w,h)

    def get_uvs(self, scene_path):
        """
        Return absolute paths to all uvmap images for the scene (sorted!)
        """
        def load_folder(folder):
            if not os.path.exists(folder) or not os.path.isdir(folder):
                return [], []

            files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
            uvs_npy = [join(folder, f) for f in files if "npy" in f and not 'angle' in f and not 'depth' in f]
            uvs_png = [join(folder, f) for f in files if "png" in f and not 'angle' in f and not 'depth' in f]
            return uvs_npy, uvs_png

        pyramid_folders = [f for f in os.listdir(scene_path) if 'uv_' in f and is_float(f.split('_')[1])]
        pyramid_folders = sorted(pyramid_folders, key=lambda x: float(x.split('_')[1]))

        # filter same folders, that only differ in suffix '.0', e.g. 256 vs 256.0
        # --> they should be the same, so just use one of them
        duplicates = []
        for i, f in enumerate(pyramid_folders):
            size = float(f.split('_')[1])
            if i < len(pyramid_folders) - 1:
                next_size = float(pyramid_folders[i+1].split('_')[1])
                if size == next_size:
                    duplicates.append(i + 1)
        pyramid_folders = [f for i, f in enumerate(pyramid_folders) if i not in duplicates]

        # save unfiltered level steps
        self.all_levels = np.array([float(x.split('_')[1]) for x in pyramid_folders])

        # filter minimum uv size
        pyramid_folders = [f for f in pyramid_folders if float(f.split('_')[1]) >= self.min_pyramid_height]

        # filter how many levels we want to have
        pyramid_folders = pyramid_folders[:self.pyramid_levels]

        # save the filtered level steps
        self.levels = np.array([float(x.split('_')[1]) for x in pyramid_folders])

        # return the paths to the levels
        pyramid_folders = [join(scene_path, f) for f in pyramid_folders]
        return [load_folder(f)[0] for f in pyramid_folders]

    def get_angles(self, scene_path):
        """
        Return absolute paths to all angle images for the scene (sorted!)
        """
        def load_folder(folder):
            if not os.path.exists(folder) or not os.path.isdir(folder):
                return [], []

            files = sorted(os.listdir(folder), key=lambda x: int(x.split(".")[0]))
            angles_npy = [join(folder, f) for f in files if "npy" in f and 'angle' in f]
            return angles_npy

        # always return angle maps without pyramid_levels, because we only need one angle map anyways
        uv_path = join(scene_path, "uv")
        angles_npy = load_folder(uv_path)
        return angles_npy

    def load_extrinsics(self, idx):
        """
        load the extrinsics item from self.extrinsics

        :param idx: the item to load

        :return: the extrinsics as numpy array
        """

        extrinsics = open(self.extrinsics[idx], "r").readlines()
        extrinsics = [[float(item) for item in line.split(" ")] for line in extrinsics]
        extrinsics = np.array(extrinsics, dtype=np.float32)

        return extrinsics

    def load_uvmap(self, idx, pyramid_idx=0):
        """
        load the uvmap item from self.uv_maps

        :param idx: the item to load

        :return: the uvmap as PIL image or numpy array
        """

        file = self.uv_maps[pyramid_idx][idx]
        return np.load(file)

    def load_anglemap(self, idx):
        """
        load the angle_map item from self.angle_maps

        :param idx: the item to load

        :return: the angle_map as PIL image or numpy array
        """
        file = self.angle_maps[idx]
        angle = np.load(file)
        angle = angle[:, :, :1]  # only keep first channel, but slice it instead of '[:,:,0]' to keep the dim
        return angle

    def load_depth(self, idx):
        file = self.depth_images[idx]
        if not self.rendered_depth:
            d = np.asarray(Image.open(file)) / 1000.0
        else:
            d = np.load(file)
            d = d[:, :, :1]  # only keep first channel, but slice it instead of '[:,:,0]' to keep the dim

        return d

    def calculate_mask(self, uvmap, depth=None):
        """
        calculate the uvmap mask item from uvmap (valid values == 1)

        :param idx: the uvmap from which to calculate the mask

        :return: the mask as PIL image
        """

        mask = np.asarray(uvmap)
        mask_bool = mask[:, :, 0] != 0
        mask_bool += mask[:, :, 1] != 0
        mask = mask_bool

        if depth is not None:
            depth = cv2.resize(depth, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
            mask = mask * (depth > 0)

        mask = Image.fromarray(mask)

        return mask

    def calculate_depth_level(self, uv, depth, transform_uv, transform_depth):
        """
        calculate the depth level per pixel from uv map and depth images.

        :param uv: the uvmap as numpy array
        :param depth: the depth as PIL image

        :return: the depth level as numpy array
        """
        min_uv_height = 32
        min_depth = self.min_pyramid_depth  # e.g. 0.25
        n_levels = len(self.levels)  # e.g. 5
        depth_factor = depth.squeeze() / min_depth  # how much larger depth is in comparison to min_depth
        uv_height = min_uv_height * depth_factor  # this would be the ideal uv height
        x = np.subtract.outer(uv_height, self.levels)  # get distance to all available levels (H, W, #levels)
        rounded_levels = np.argmin(abs(x), axis=2)  # get index of smallest distance == closest level (because self.levels is sorted)
        residues = self.levels[rounded_levels] - uv_height  # how far are we away from ideal uv height with selected level, e.g. 128 - 113.25
        discrete_residues = np.where(residues > 0, -1, 1)  # is next or previous level the 2nd closest one
        discrete_residues[residues == 0] = 0  # if we have perfect uv match, set to 0 instead of 1
        other_levels = rounded_levels + discrete_residues  # 2nd closest level
        other_levels[other_levels < 0] = 0  # mask if smaller than first level
        other_levels[other_levels >= n_levels] = n_levels - 1  # mask if larger than last level
        height_difference = abs(self.levels[rounded_levels] - self.levels[other_levels])  # difference between neighboring uv heights
        level_residues = abs(residues / (height_difference + 1e-6))  # interpolation weight between uv heights
        level_residues[height_difference == 0] = 0  # if we have perfect uv match
        level_residues = 1 - level_residues  # inverse because level_residues interpolation weight refers to rounded_levels, which is always the closer one
        continuous_depth_level = np.where(residues > 0, other_levels + level_residues, other_levels - level_residues)
        continuous_depth_level[level_residues == 1] = rounded_levels[level_residues == 1]

        # continuous_depth_level: depth level and interpolation weight combined
        #                         e.g. 1.8 -> rounded_levels: 2, other_levels: 1, level_residues: 0.8
        #                         e.g. 1.4 -> rounded_levels: 1, other_levels: 2, level_residues: 0.6
        # rounded_levels: nearest level
        # other_levels: 2nd nearest level
        # level_residues: interpolation weight in (0, 1), telling how much we should weight rounded_levels.
        return continuous_depth_level.astype(np.float32), rounded_levels.astype(np.int),\
               other_levels.astype(np.int), level_residues.astype(np.float32)


def is_float(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
