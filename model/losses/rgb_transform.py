import torch
import torchvision.transforms as transforms


def pre():
    return transforms.Compose([
        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to BGR
        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # subtract imagenet mean
                             std=[1, 1, 1]),
        transforms.Lambda(lambda x: x.mul_(255)), # from 0..1 to 0..255
    ])


def post():
    return transforms.Compose([transforms.Lambda(lambda x: x.cpu() if torch.cuda.is_available() else x),
        transforms.Lambda(lambda x: x.mul_(1. / 255)), # from 0..255 to 0..1
        transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # add imagenet mean
                          std=[1, 1, 1]),
        transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # turn to RGB
        transforms.Lambda(lambda x: x.clamp(0, 1)),  # clamp 0..1
    ])
