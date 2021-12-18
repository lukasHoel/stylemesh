import torchvision
import torch
import numpy as np
import os

from torch.utils.data import Dataset, SubsetRandomSampler, SequentialSampler, Sampler
from torch.utils.data import DataLoader

from PIL import Image

from data.utils import get_image_transform

from tqdm.auto import tqdm

import os.path
from os.path import join

import pytorch_lightning as pl

from typing import Optional

from abc import ABC, abstractmethod

from typing import Union
import cv2



class Abstract_Dataset(Dataset, ABC):

    def __init__(self,
                 root_path,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 resize_size=(256, 256),
                 verbose=False):
        # save all constructor arguments
        self.transform_rgb = transform_rgb
        self.transform_label = get_image_transform(transform_label)
        self.transform_uv = transform_uv
        self.resize_size = resize_size
        self.verbose = verbose
        self.root_path = root_path

        # create data for this dataset
        self.create_data()

    def create_data(self):
        self.rgb_images, self.uv_maps, self.angle_maps, self.extrinsics, self.intrinsics, self.intrinsic_image_sizes, self.depth_images, self.size, self.scene_dict = self.parse_scenes()

    @abstractmethod
    def get_scenes(self):
        """
        Return names to all scenes for the dataset.
        """
        pass

    @abstractmethod
    def get_colors(self, scene_path):
        """
        Return absolute paths to all colors images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_depth(self, scene_path):
        """
        Return absolute paths to all depth images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_extrinsics(self, scene_path):
        """
        Return absolute paths to all extrinsic images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_uvs(self, scene_path):
        """
        Return absolute paths to all uvmap images for the scene (sorted!)
        """
        pass

    @abstractmethod
    def get_angles(self, scene_path):
        """
        Return absolute paths to all angle images for the scene (sorted!)
        """
        pass

    def get_intrinsics(self, scene_path):
        """
        Return 3x3 numpy array as intrinsic K matrix for the scene and (W,H) image dimensions if available
        """
        return np.identity(4, dtype=np.float32), (0, 0)

    def parse_scenes(self):
        rgb_images = []
        depth_images = []
        uv_maps = []
        angle_maps = []
        extrinsics_matrices = []
        intrinsic_matrices = []
        intrinsic_image_sizes = []
        scene_dict = {}

        scenes = self.get_scenes()
        if self.verbose:
            print("Collecting images...")
            scenes = tqdm(scenes)

        for scene in scenes:
            scene_path = join(self.root_path, scene)
            if os.path.isdir(scene_path):
                scene_dict[scene] = {
                    "path": scene_path,
                    "items": 0,
                }

                colors = self.get_colors(scene_path)
                depth = self.get_depth(scene_path)
                uvs = self.get_uvs(scene_path)
                angles = self.get_angles(scene_path)

                extrinsics = self.get_extrinsics(scene_path)
                intrinsics, image_size = self.get_intrinsics(scene_path)
                intrinsics = [intrinsics for i in range(len(colors))]
                image_size = [image_size for i in range(len(colors))]

                if len(colors) > 0 and len(colors) == len(depth) and \
                        len(uvs) > 0 and all(len(colors) == len(uv) for uv in uvs) and (
                        len(uv_maps) == 0 or len(uv_maps) == len(uvs)) and \
                        len(colors) == len(angles) and len(colors) == len(extrinsics):
                    rgb_images.extend(colors)
                    depth_images.extend(depth)

                    if len(uv_maps) == 0:
                        uv_maps.extend(uvs)
                    else:
                        for x, y in zip(uv_maps, uvs):
                            x.extend(y)

                    angle_maps.extend(angles)
                    extrinsics_matrices.extend(extrinsics)
                    intrinsic_matrices.extend(intrinsics)
                    intrinsic_image_sizes.extend(image_size)
                    scene_dict[scene]["items"] = len(colors)
                    scene_dict[scene]["color"] = colors
                    scene_dict[scene]["depth"] = depth
                    scene_dict[scene]["extrinsics"] = extrinsics
                    scene_dict[scene]["intrinsics"] = intrinsics
                    scene_dict[scene]["image_size"] = image_size
                    scene_dict[scene]["uv_map"] = uvs
                    scene_dict[scene]["angle_map"] = angles

                elif self.verbose:
                    print(
                        f"Scene {scene_path} rendered incomplete --> is skipped. colors: {len(colors)}, uvs: {len(uvs)}, angles: {len(angles)}, extr: {len(extrinsics)}")

        assert (len(rgb_images) == len(angle_maps))
        assert (len(rgb_images) == len(extrinsics_matrices))
        assert all(len(rgb_images) == len(uv) for uv in uv_maps)

        return rgb_images, uv_maps, angle_maps, extrinsics_matrices, intrinsic_matrices, intrinsic_image_sizes, depth_images, len(rgb_images), scene_dict

    @abstractmethod
    def load_extrinsics(self, idx):
        """
        load the extrinsics item from self.extrinsics

        :param idx: the item to load

        :return: the extrinsics as numpy array
        """
        pass

    def load_intrinsics(self, idx):
        """
        load the intrinsics item from self.intrinsics

        :param idx: the item to load

        :return: the intrinsics as numpy array
        """
        return self.intrinsics[idx]

    @abstractmethod
    def load_uvmap(self, idx, pyramid_idx=0):
        """
        load the uvmap item from self.uv_maps

        :param idx: the item to load

        :return: the uvmap as PIL image or numpy array
        """
        pass

    @abstractmethod
    def load_anglemap(self, idx):
        """
        load the angle_map item from self.angle_maps

        :param idx: the item to load

        :return: the angle_map as PIL image or numpy array
        """
        pass

    @abstractmethod
    def calculate_mask(self, uvmap, depth=None):
        """
        calculate the uvmap mask item from uvmap (valid values == 1)

        :param idx: the uvmap from which to calculate the mask

        :return: the mask as PIL image
        """
        pass

    @abstractmethod
    def calculate_depth_level(self, uv, depth, transform_uv, transform_depth):
        """
        calculate the depth level per pixel from uv map and depth images.

        :param uv: the uvmap as numpy array
        :param depth: the depth as PIL image

        :return: the depth level as numpy array
        """
        pass

    def prepare_getitem(self, idx):
        """
        Implementations can prepare anything necessary for loading this idx, i.e. load a .hdf5 file
        :param idx:
        :return:
        """
        pass

    def finalize_getitem(self, idx):
        """
        Implementations can finalize anything necessary after loading this idx, i.e. close a .hdf5 file
        :param idx:
        :return:
        """
        pass

    def load_rgb(self, idx):
        return Image.open(self.rgb_images[idx])

    def load_depth(self, idx):
        return Image.open(self.depth_images[idx])

    def modify_intrinsics_matrix(self, intrinsics, intrinsics_image_size, rgb_image_size):
        if intrinsics_image_size != rgb_image_size:
            intrinsics = np.array(intrinsics)
            intrinsics[0, 0] = (intrinsics[0, 0] / intrinsics_image_size[0]) * rgb_image_size[0]
            intrinsics[1, 1] = (intrinsics[1, 1] / intrinsics_image_size[1]) * rgb_image_size[1]
            intrinsics[0, 2] = (intrinsics[0, 2] / intrinsics_image_size[0]) * rgb_image_size[0]
            intrinsics[1, 2] = (intrinsics[1, 2] / intrinsics_image_size[1]) * rgb_image_size[1]

        return intrinsics

    def __len__(self):
        return self.size

    def __getitem__(self, item, only_cam=False):
        self.prepare_getitem(item)

        extrinsics = self.load_extrinsics(item)
        extrinsics = torch.from_numpy(extrinsics)
        intrinsics = self.load_intrinsics(item)
        intrinsics = torch.from_numpy(intrinsics)

        if only_cam:
            self.finalize_getitem(item)
            return extrinsics, intrinsics

        rgb = self.load_rgb(item)
        depth = self.load_depth(item)

        pyramid_size = len(self.uv_maps)
        uv = [self.load_uvmap(item, i) for i in range(pyramid_size)]

        mask = self.calculate_mask(uv[-1], depth)
        angle = self.load_anglemap(item)

        if isinstance(self.resize_size, int):
            w, h = rgb.size
            h_new = self.resize_size
            w_new = round(w * h_new / h)
            resize_size = (w_new, h_new)
        else:
            resize_size = self.resize_size

        rgb = rgb.resize(resize_size)
        if isinstance(depth, np.ndarray):
            depth = cv2.resize(depth, resize_size, interpolation=cv2.INTER_LINEAR)
        else:
            depth = depth.resize(resize_size)

        if isinstance(angle, np.ndarray):
            # PIL library is not able to convert a (W,H,2) image of type np.float32
            # for this case we use cv2 which can do it
            angle = cv2.resize(angle, resize_size, interpolation=cv2.INTER_NEAREST)
        else:
            angle = angle.resize(resize_size, Image.NEAREST)
        mask = mask.resize(resize_size, Image.NEAREST)

        # fix intrinsics to resized item
        intrinsics = self.modify_intrinsics_matrix(intrinsics, self.intrinsic_image_sizes[item], rgb.size)

        # load + transform depth level
        depth_level, rounded_depth_level, other_depth_level, depth_level_interpolation_weight = self.calculate_depth_level(uv[-1], depth, self.transform_uv, self.transform_label)
        depth_level = torchvision.transforms.ToTensor()(depth_level)
        rounded_depth_level = torchvision.transforms.ToTensor()(rounded_depth_level)
        other_depth_level = torchvision.transforms.ToTensor()(other_depth_level)
        depth_level_interpolation_weight = torchvision.transforms.ToTensor()(depth_level_interpolation_weight)

        if self.transform_rgb:
            rgb = self.transform_rgb(rgb)

        if self.transform_label:
            depth = self.transform_label(depth)

        result = (rgb, extrinsics, intrinsics, depth, depth_level, rounded_depth_level, other_depth_level, depth_level_interpolation_weight, item)

        if self.transform_uv:
            # transform uv, angle, mask
            uv = [self.transform_uv(v) for v in uv]
            angle = torchvision.transforms.ToTensor()(angle)
            mask = torchvision.transforms.ToTensor()(mask)
            mask = mask > 0
            mask = mask.squeeze()  # Final shape: H x W
            angle_degrees = torch.rad2deg(torch.acos(angle))  # saved in the file is cos(angle)
            angle_guidance = angle

            # add all to result
            result += (uv, mask, angle_guidance, angle_degrees)

        self.finalize_getitem(item)
        return result


class Abstract_DataModule(pl.LightningDataModule):
    split_modes = [
        "folder",  # we are given three different root_paths to train/val/test subdirectories
        "sequential",  # the last split[1] percent images are the test images
    ]

    sampler_modes = [
        "random",
        # uses SubsetRandomSampler (might be in addition to the self.shuffle argument -> initially shuffled + other order every epoch)
        "sequential",
        # uses SequentialSampler (might be in addition to the self.shuffle argument -> initially shuffled + same order every epoch)
        "repeat",
        # uses custom RepeatedSampler and the self.index_repeat argument to sequentially iterate and repeat each train-item self.index_repeat times (might be in addition to self.shuffle argument --> initially shuffled + same repeated order every epoch)
    ]

    def __init__(self,
                 dataset,
                 root_path: Union[str, dict],
                 batch_size: int = 32,
                 num_workers: int = 1,
                 transform_rgb=None,
                 transform_label=None,
                 transform_uv=None,
                 resize_size=(256, 256),
                 use_scene_filter=False,
                 scene=None,
                 min_images=1000,
                 max_images=-1,
                 shuffle: bool = False,
                 sampler_mode: str = "random",
                 index_repeat: int = 1,
                 split: list = [0.8, 0.2],
                 split_mode: str = "noise",
                 verbose: bool = False):
        super().__init__()

        self.dataset_class = dataset
        self.root_path = root_path
        self.transform_rgb = transform_rgb
        self.transform_label = transform_label
        self.transform_uv = transform_uv
        self.resize_size = resize_size
        self.verbose = verbose
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.sampler_mode = sampler_mode
        self.index_repeat = index_repeat
        self.split = split
        self.split_mode = split_mode

        self.use_scene_filter = use_scene_filter
        self.scene = scene  # the input scene
        self.selected_scene = None  # the scene after creating the first dataset (should be the same scene for all other creations)
        self.min_images = min_images
        self.max_images = max_images

    def create_dataset(self, root_path) -> Abstract_Dataset:
        if not self.use_scene_filter:
            d = self.dataset_class(root_path=root_path,
                                   transform_rgb=self.transform_rgb,
                                   transform_label=self.transform_label,
                                   transform_uv=self.transform_uv,
                                   resize_size=self.resize_size,
                                   verbose=self.verbose)
        else:
            d = self.dataset_class(root_path=root_path,
                                   scene=self.selected_scene if self.selected_scene else self.scene,
                                   min_images=self.min_images,
                                   max_images=self.max_images,
                                   transform_rgb=self.transform_rgb,
                                   transform_label=self.transform_label,
                                   transform_uv=self.transform_uv,
                                   resize_size=self.resize_size,
                                   verbose=self.verbose)
            self.selected_scene = d.scene
            d.input_scene = d.scene

        self.after_create_dataset(d, root_path)

        return d

    @abstractmethod
    def after_create_dataset(self, d, root_path):
        pass

    def setup(self, stage: Optional[str] = None):
        # create datasets based on the specified path and further arguments
        if isinstance(self.root_path, dict):
            train_path = self.root_path["train"]
            val_path = self.root_path["val"]

            self.train_dataset = self.create_dataset(train_path)
            self.val_dataset = self.create_dataset(val_path)
        else:
            self.train_dataset = self.create_dataset(self.root_path)
            self.val_dataset = self.create_dataset(self.root_path)

        # create train/val/test split from the loaded datasets based on the split_mode
        if self.split_mode == "folder":
            self.train_indices = [i for i in range(self.train_dataset.__len__())]
            self.val_indices = [i for i in range(self.val_dataset.__len__())]

            if self.shuffle:
                np.random.shuffle(self.train_indices)
                np.random.shuffle(self.val_indices)

        else:
            if isinstance(self.root_path, dict):
                raise ValueError(
                    f"Cannot use multiple root_path arguments (train/val) when split_mode is not 'folder'!")

            # create train/val/test split
            len = self.train_dataset.__len__()
            indices = [i for i in range(len)]
            train_split = int(self.split[0] * len)

            if self.shuffle:
                np.random.shuffle(indices)

            self.train_indices = indices[:train_split]
            self.val_indices = indices[train_split:]

    def train_dataloader(self):
        if self.sampler_mode == "sequential":
            sampler = SequentialSampler(self.train_dataset)
        elif self.sampler_mode == "random":
            sampler = SubsetRandomSampler(self.train_indices)
        elif self.sampler_mode == "repeat":
            sampler = RepeatingSampler(self.train_indices, self.index_repeat)
        else:
            raise ValueError(f"Unsupported sampler mode: {self.sampler_mode}")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def val_dataloader(self):
        if self.sampler_mode == "sequential":
            sampler = SequentialSampler(self.val_dataset)
        elif self.sampler_mode == "random":
            sampler = SubsetRandomSampler(self.val_indices)
        elif self.sampler_mode == "repeat":
            # validation does not need to be repeated
            sampler = RepeatingSampler(self.val_indices, 1)
        else:
            raise ValueError(f"Unsupported sampler mode: {self.sampler_mode}")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, sampler=sampler)

    def test_dataloader(self):
        raise NotImplementedError()


class RepeatingSampler(Sampler):
    def __init__(self, indices, index_repeat):
        super().__init__(indices)
        if isinstance(index_repeat, int):
            self.indices = [item for item in indices for i in range(index_repeat)]
        elif isinstance(index_repeat, list):
            self.indices = [item for item in indices for i in range(index_repeat[item])]
        else:
            raise ValueError('unsupported index_repeat type', index_repeat)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)