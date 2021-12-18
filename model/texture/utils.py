from torchvision.transforms import ToTensor, Resize, Compose, Lambda, ToPILImage, RandomCrop
from PIL import Image
import torch


def to_grid_range(x):
    """from [0,1] to [-1,1]"""
    return (x * 2.0) - 1


def from_grid_range(x):
    """from [-1,1] to [0,1]"""
    return (x + 1) / 2.0


def cut_b_channel(x):
    """from RGB to RG image (i.e. remove the B channel from the UV image)"""
    return x[:2]


def add_b_channel(x):
    """from RG image to RGB image (b is filled with -1, i.e. the zero value in the grid range)"""
    return torch.cat((x, torch.full_like(x[0], -1).unsqueeze(0)), dim=0)


def chw_to_hwc(x):
    """image format CHW to HWC"""
    if len(x.shape) == 3:
        return x.permute(1, 2, 0)
    else:
        return x.permute(0, 2, 3, 1)


def hwc_to_chw(x):
    """image format HWC to CHW"""
    if len(x.shape) == 3:
        return x.permute(2, 0, 1)
    else:
        return x.permute(0, 3, 1, 2)


def to_grid_format(x):
    """both of these need to be applied to be valid for grid_sample"""
    x = cut_b_channel(x)
    x = chw_to_hwc(x)
    return x


def from_grid_format(x):
    """both of these undo the grid_format conversion"""
    x = hwc_to_chw(x)
    x = add_b_channel(x)
    return x


def to_grid(x):
    """this changes a UV map as tensor image (CHW) to a grid valid for usage with grid_sample"""
    x = to_grid_range(x)
    x = to_grid_format(x)
    return x


def from_grid(x):
    """this changes a grid valid for usage with grid_sample to a UV map as tensor image (CHW)"""
    x = from_grid_format(x)
    x = from_grid_range(x)
    return x


def numpy_to_pil(x):
    """numpy to PIL image conversion"""
    return Image.fromarray(x)


def get_rgb_transform():
    return Compose([
        ToTensor()
    ])


def get_label_transform():
    return Compose([
        ToTensor()
    ])


def get_uv_transform():
    return Compose([
        ToTensor(),
        Lambda(to_grid)
    ])
