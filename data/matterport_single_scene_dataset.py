import numpy as np

from PIL import Image
from os.path import join
from tqdm.auto import tqdm

import torch
import torchvision
import random

from data.matterport_dataset import MatterportDataset
from data.abstract_dataset import Abstract_DataModule


class Matterport_Single_Scene_DataModule(Abstract_DataModule):
    def __init__(self,
                 root_path: str,
                 batch_size: int = 32,
                 num_workers: int = 1,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 resize_size=(256, 256),
                 pyramid_levels=5,
                 min_pyramid_depth=0.25,
                 min_pyramid_height=32,
                 scene=None,
                 min_images=1000,
                 max_images=-1,
                 shuffle: bool = False,
                 sampler_mode: str = "random",
                 index_repeat: int = 1,
                 split: list = [0.8, 0.2],
                 split_mode: str = "skip",
                 region_index=0,
                 verbose: bool = False):

        Abstract_DataModule.__init__(self,
                                     dataset=Matterport_Single_House_Dataset,
                                     root_path=join(root_path, "v1/scans"),
                                     transform_rgb=transform_rgb,
                                     transform_label=transform_label,
                                     transform_uv=transform_uv,
                                     resize_size=resize_size,
                                     use_scene_filter=True,
                                     scene=scene,
                                     min_images=min_images,
                                     max_images=max_images,
                                     verbose=verbose,
                                     batch_size=batch_size,
                                     num_workers=num_workers,
                                     shuffle=shuffle,
                                     sampler_mode=sampler_mode,
                                     index_repeat=index_repeat,
                                     split=split,
                                     split_mode=split_mode)

        self.min_pyramid_depth = min_pyramid_depth
        self.min_pyramid_height = min_pyramid_height
        self.pyramid_levels = pyramid_levels
        self.region_index = region_index

    def after_create_dataset(self, d, root_path):
        if isinstance(d, MatterportDataset):
            d.set_uv_pyramid_mode(self.min_pyramid_depth, self.min_pyramid_height)
            d.set_pyramid_levels(self.pyramid_levels)
            d.region_index = self.region_index
            d.create_data()


class Matterport_Single_House_Dataset(MatterportDataset):

    def __init__(self,
                 root_path,
                 scene=None,
                 min_images=1000,
                 max_images=-1,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 resize_size=(256, 256),
                 pyramid_levels=5,
                 min_pyramid_depth=0.25,
                 min_pyramid_height=32,
                 region_index=0,
                 verbose=False):

        self.input_scene = scene
        self.min_images = min_images
        self.max_images = max_images

        MatterportDataset.__init__(self,
                                root_path=root_path,
                                transform_rgb=transform_rgb,
                                transform_label=transform_label,
                                transform_uv=transform_uv,
                                resize_size=resize_size,
                                min_pyramid_depth=min_pyramid_depth,
                                min_pyramid_height=min_pyramid_height,
                                pyramid_levels=pyramid_levels,
                                region_index=region_index,
                                verbose=verbose)

    def create_data(self):
        self.scene_dict = self.parse_scenes()[-1]
        self.rgb_images, self.extrinsics, self.intrinsics, self.intrinsic_image_sizes, self.depth_images, self.uv_maps, self.angle_maps, self.size, self.scene = self.get_scene(self.input_scene, self.min_images, self.max_images)

        print(f"Using scene: {self.scene}. Input was: {self.input_scene}")

        # use this to finally set the self.uvs_npy, self.angle_npy and self.rendered_depth attributes correctly to the state of the chosen scene
        self.get_depth(join(self.root_path, self.scene))
        self.get_uvs(join(self.root_path, self.scene))
        self.get_angles(join(self.root_path, self.scene))

    def get_scene(self, scene, min_images, max_images):
        items = self.get_scene_items(scene)
        if self.in_range(min_images, max_images, items):
            return self.parse_scene(scene)
        else:
            return self.find_house(min_images, max_images)

    def get_scene_items(self, scene):
        if scene is None:
            return None
        elif scene not in self.scene_dict:
            return 0
        else:
            return self.scene_dict[scene]["items"]

    def in_range(self, min, max, value):
        return (value is not None) and (min == -1 or value >= min) and (max == -1 or value <= max)

    def parse_scene(self, scene):
        h = self.scene_dict[scene]
        return h["color"], h["extrinsics"], h["intrinsics"], h["image_size"], h["depth"], h["uv_map"], h["angle_map"], len(h["color"]), scene

    def find_house(self, min_images, max_images):
        max = -1
        min = -1
        scenes = [s for s in self.scene_dict.keys()]
        random.shuffle(scenes)
        if self.verbose:
            scenes = tqdm(scenes)
            print(f"Searching for a house with more than {min_images} images")
        for h in scenes:
            size = self.get_scene_items(h)
            if max == -1 or size > max:
                max = size
            if min == -1 or size < min:
                min = size
            if self.in_range(min_images, max_images, size):
                if self.verbose:
                    print(f"Using scene '{h}' which has {size} images")
                return self.parse_scene(h)
        raise ValueError(f"No scene found with {min_images} <= i <= {max_images} images. Min/Max available: {min}/{max}")
