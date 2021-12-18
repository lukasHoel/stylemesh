import torch
import torch.nn as nn
import torch.nn.functional as F
from os.path import join

from torchvision.transforms import ToPILImage
from model.texture.utils import from_grid_range


def to_image(texture, startIndex=0, padChannels=True, normalize_transform=from_grid_range):
    texture = texture.detach().cpu()
    texture = texture[startIndex:(startIndex + 3), :, :]
    if padChannels and texture.shape[0] != 3:
        c = 3-texture.shape[0]
        h = texture.shape[1]
        w = texture.shape[2]
        texture = torch.cat((texture, torch.zeros(c, h, w).type_as(texture)), dim=0)
    texture = normalize_transform(texture)
    return ToPILImage()(texture)


class NeuralTexture(nn.Module):
    def __init__(self, W, H, C, random_init=False):
        super(NeuralTexture, self).__init__()
        self.W = W
        self.H = H
        self.C = C

        if random_init:
            self.data = torch.nn.Parameter(torch.rand(C, H, W, requires_grad=True), requires_grad=True)
        else:
            self.data = torch.nn.Parameter(torch.zeros(C, H, W, requires_grad=True), requires_grad=True)

    @staticmethod
    def from_tensor(data: torch.Tensor):
        C, H, W = data.shape
        texture = NeuralTexture(W, H, C)
        texture.data = torch.nn.Parameter(data, requires_grad=True)
        return texture

    def normalize(self):
        with torch.no_grad():
            normalized_data = torch.clamp(self.data, -123.6800, 151.0610)
            self.data.copy_(normalized_data)

    def forward(self, x):
        self.normalize()
        batch_size = x.shape[0]
        y = F.grid_sample(self.data.repeat(batch_size, 1, 1, 1),
                          x,
                          mode="bilinear",
                          padding_mode='border',
                          align_corners=True)  # this treats (0,0) as origin and not as the center of the lower left texel
        return y

    def get_image(self):
        return self.data

    def save_image(self, dir, prefix="", normalize_transform=from_grid_range):
        image = to_image(self.get_image().detach().cpu(), normalize_transform=normalize_transform)
        file_path = join(dir, f"{prefix}texture.jpg")
        image.save(file_path)

    def save_layers(self, dir, prefix="", normalize_transform=from_grid_range):
        self.save_image(dir, prefix, normalize_transform)

    def save_texture(self, dir, prefix=""):
        image = self.get_image().detach().cpu()
        file_path = join(dir, f"{prefix}texture.pt")
        torch.save(image, file_path)


class HierarchicalNeuralTexture(nn.Module):
    def __init__(self, W, H, C, num_layers=4, random_init=False):
        super(HierarchicalNeuralTexture, self).__init__()
        self.W = W
        self.H = H
        self.C = C

        # laplacian pyramid, i.e. first layer is (W, H), second is (W//2, H//2), third is (W//4, H//4), ...
        self.layers = nn.ModuleList([NeuralTexture(W // pow(2, i), H // pow(2, i), C, random_init) for i in range(num_layers)])

    @staticmethod
    def from_tensor(data: list):
        textures = []
        C, H, W = data[0].shape
        for i, d in enumerate(data):
            ci, hi, wi = d.shape
            assert(W//pow(2,i) == wi and H//pow(2,i) == hi and C == ci)
            texture = NeuralTexture.from_tensor(d)
            textures.append(texture)
        texture = HierarchicalNeuralTexture(W, H, C, num_layers=len(textures))
        texture.layers = nn.ModuleList(textures)
        return texture

    def forward(self, x):
        y = [layer(x) for layer in self.layers]
        y = torch.stack(y)
        y = torch.sum(y, dim=0)
        return y

    def regularizer(self, weights):
        reg = 0.0

        for i, layer in enumerate(self.layers):
            reg += torch.mean(torch.pow(layer.data, 2.0)) * weights[i]

        return reg

    def get_image(self):
        w_range = torch.arange(0, self.W, dtype=torch.float) / (self.W - 1.0) * 2.0 - 1.0
        h_range = torch.arange(0, self.H, dtype=torch.float) / (self.H - 1.0) * 2.0 - 1.0

        v, u = torch.meshgrid(h_range, w_range)
        uv_id = torch.stack([u, v], 2)
        uv_id = uv_id.unsqueeze(0)
        uv_id = uv_id.type_as(self.layers[0].data)

        texture = self.forward(uv_id)[0, 0:3, :, :]

        return texture

    def save_image(self, dir, prefix="", normalize_transform=from_grid_range):
        texture = self.get_image()
        texture = to_image(texture.detach().cpu(), normalize_transform=normalize_transform)
        file_path = join(dir, f"{prefix}texture.jpg")
        texture.save(file_path)

    def save_layers(self, dir, prefix="", normalize_transform=from_grid_range):
        for i,l in enumerate(self.layers):
            l.save_image(dir, prefix+f"_layer{str(i)}_", normalize_transform)

    def save_texture(self, dir, prefix=""):
        for i, l in enumerate(self.layers):
            l.save_texture(dir, f"{prefix}layer-{i}-")
