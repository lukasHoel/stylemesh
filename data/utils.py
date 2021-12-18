from torchvision.transforms.functional import InterpolationMode
import os
from PIL import Image
import numpy as np
from collections import OrderedDict

from tqdm.auto import tqdm

import torchvision
import torch


def to_numpy(x):
    x_ = np.array(x)
    x_ = x_.astype(np.float32)
    return x_


def get_image_transform(transform):
    # fix for this issue: https://github.com/pytorch/vision/issues/2194
    if transform is not None and isinstance(transform, torchvision.transforms.Compose) and (transform.transforms[-1], torchvision.transforms.ToTensor):
        transform = torchvision.transforms.Compose([
            *transform.transforms[:-1],
            torchvision.transforms.Lambda(to_numpy),
            torchvision.transforms.ToTensor()
        ])
    elif isinstance(transform, torchvision.transforms.ToTensor):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Lambda(to_numpy),
            torchvision.transforms.ToTensor()
        ])

    return transform


def unproject(cam2world, intrinsic, depth):
    # get dimensions
    bs, _, H, W = depth.shape

    # create meshgrid with image dimensions (== pixel coordinates of source image)
    y = torch.linspace(0, H - 1, H).type_as(depth).int()
    x = torch.linspace(0, W - 1, W).type_as(depth).int()
    xx, yy = torch.meshgrid(x, y)
    xx = torch.transpose(xx, 0, 1).repeat(bs, 1, 1)
    yy = torch.transpose(yy, 0, 1).repeat(bs, 1, 1)

    # get intrinsics and depth in correct format to match image dimensions
    fx = intrinsic[:, 0, 0].unsqueeze(1).unsqueeze(1).expand_as(xx)
    cx = intrinsic[:, 0, 2].unsqueeze(1).unsqueeze(1).expand_as(xx)
    fy = intrinsic[:, 1, 1].unsqueeze(1).unsqueeze(1).expand_as(yy)
    cy = intrinsic[:, 1, 2].unsqueeze(1).unsqueeze(1).expand_as(yy)
    depth = depth.squeeze()

    # inverse projection (K_inv) on pixel coordinates --> 3D point-cloud
    x = (xx - cx) / fx * depth
    y = (yy - cy) / fy * depth

    # combine each point into an (x,y,z,1) vector
    coords = torch.zeros(bs, H, W, 4).type_as(depth).float()
    coords[:, :, :, 0] = x
    coords[:, :, :, 1] = y
    coords[:, :, :, 2] = depth
    coords[:, :, :, 3] = 1

    # extrinsic view projection to target view
    coords = coords.view(bs, -1, 4)
    coords = torch.bmm(coords, cam2world)
    coords = coords.view(bs, H, W, 4)

    return coords


def reproject(cam2world_src, cam2world_tar, W, H, intrinsic, depth_src, depth_tar, color_tar, mask_tar):
    # get batch_size
    bs = mask_tar.shape[0]

    # calculate src2tar extrinsic matrix
    world2cam_tar = torch.inverse(cam2world_tar)
    src2tar = torch.transpose(torch.bmm(world2cam_tar, cam2world_src), 1, 2)

    # create meshgrid with image dimensions (== pixel coordinates of source image)
    y = torch.linspace(0, H - 1, H).type_as(color_tar).int()
    x = torch.linspace(0, W - 1, W).type_as(color_tar).int()
    xx, yy = torch.meshgrid(x, y)
    xx = torch.transpose(xx, 0, 1).repeat(bs, 1, 1)
    yy = torch.transpose(yy, 0, 1).repeat(bs, 1, 1)

    # get intrinsics and depth in correct format to match image dimensions
    fx = intrinsic[:,0,0].unsqueeze(1).unsqueeze(1).expand_as(xx)
    cx = intrinsic[:,0,2].unsqueeze(1).unsqueeze(1).expand_as(xx)
    fy = intrinsic[:,1,1].unsqueeze(1).unsqueeze(1).expand_as(yy)
    cy = intrinsic[:,1,2].unsqueeze(1).unsqueeze(1).expand_as(yy)
    depth_src = depth_src.squeeze()

    # inverse projection (K_inv) on pixel coordinates --> 3D point-cloud
    x = (xx - cx) / fx * depth_src
    y = (yy - cy) / fy * depth_src

    # combine each point into an (x,y,z,1) vector
    coords = torch.zeros(bs, H, W, 4).type_as(color_tar).float()
    coords[:, :, :, 0] = x
    coords[:, :, :, 1] = y
    coords[:, :, :, 2] = depth_src
    coords[:, :, :, 3] = 1

    # extrinsic view projection to target view
    coords = coords.view(bs, -1, 4)
    coords = torch.bmm(coords, src2tar)
    coords = coords.view(bs, H, W, 4)

    # projection (K) on 3D point-cloud --> pixel coordinates
    z_tar = coords[:, :, :, 2]
    x = coords[:, :, :, 0] / (1e-8 + z_tar) * fx + cx
    y = coords[:, :, :, 1] / (1e-8 + z_tar) * fy + cy

    # mask invalid pixel coordinates because of invalid source depth
    mask0 = (depth_src == 0)

    # mask invalid pixel coordinates after projection:
    # these coordinates are not visible in target view (out of screen bounds)
    mask1 = (x < 0) + (y < 0) + (x >= W - 1) + (y >= H - 1)

    # create 4 target pixel coordinates which map to the nearest integer coordinate
    # (left, top, right, bottom)
    lx = torch.floor(x).float()
    ly = torch.floor(y).float()
    rx = (lx + 1).float()
    ry = (ly + 1).float()

    def make_grid(x, y):
        """
        converts pixel coordinates from [0..W] or [0..H] to [-1..1] and stacks them together.
        :param x: x pixel coordinates with shape NxHxW
        :param y: y pixel coordinates with shape NxHxW
        :return: (x,y) pixel coordinate grid with shape NxHxWx2
        """
        x = (2.0 * x / W) - 1.0
        y = (2.0 * y / H) - 1.0
        grid = torch.stack((x, y), dim=3)
        return grid

    # combine to (x,y) pixel coordinates: (top-left, ..., bottom-right)
    ll = make_grid(lx, ly)
    lr = make_grid(lx, ry)
    rl = make_grid(rx, ly)
    rr = make_grid(rx, ry)

    # calculate difference between depth in target view after reprojection and gt depth in target view
    z_tar = z_tar.unsqueeze(1)
    sample_z1 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, ll,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))
    sample_z2 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, lr,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))
    sample_z3 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, rl,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))
    sample_z4 = torch.abs(z_tar - torch.nn.functional.grid_sample(depth_tar, rr,
                                                                  mode="nearest",
                                                                  padding_mode='border',
                                                                  align_corners=True))

    # mask invalid pixel coordinates because of too high difference in depth
    mask2 = torch.minimum(torch.minimum(sample_z1, sample_z2), torch.minimum(sample_z3, sample_z4)) > 0.1
    mask2 = mask2.int().squeeze()

    # combine all masks
    mask_remap = (1 - (mask0 + mask1 + mask2 > 0).int()).float().unsqueeze(1)

    # create (x,y) pixel coordinate grid with reprojected float coordinates
    map_x = x.float()
    map_y = y.float()
    map = make_grid(map_x, map_y)

    # warp target rgb/mask to the new pixel coordinates based on the reprojection
    # also mask the results
    color_tar_to_src = torch.nn.functional.grid_sample(color_tar, map,
                                                                  mode="bilinear",
                                                                  padding_mode='border',
                                                                  align_corners=True)
    mask_tar = mask_tar.float().unsqueeze(1)
    mask = torch.nn.functional.grid_sample(mask_tar, map,
                                            mode="bilinear",
                                            padding_mode='border',
                                            align_corners=True)
    mask = (mask > 0.99) * mask_remap
    mask = mask.bool()
    color_tar_to_src *= mask

    return color_tar_to_src, mask.squeeze(1)
