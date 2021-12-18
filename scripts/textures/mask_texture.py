from argparse import ArgumentParser

from os.path import join
import os

import numpy as np
import torch
from PIL import Image

import cv2

from tqdm.auto import tqdm

from model.texture.texture import NeuralTexture


def main(opt):
    uvs = [join(opt.uvs, f) for f in os.listdir(opt.uvs) if not 'angle' in f]

    texture_image = Image.open(opt.tex)
    texture_image = np.array(texture_image)
    print(np.min(texture_image), np.max(texture_image))
    texture_image = np.concatenate([texture_image, np.ones_like(texture_image[:, :, :1])*255], axis=2)

    #texture = NeuralTexture(texture_image.shape[0], texture_image.shape[1], 3)
    #texel_count = torch.zeros(texture_image.shape[0], texture_image.shape[1])

    mask = np.zeros_like(texture_image)
    texel_count = np.zeros_like(texture_image)

    for uv in tqdm(uvs):
        uv = np.load(uv)

        """
        uv_torch = torch.from_numpy(uv[:,:,:2]).unsqueeze(0) * 2 - 1
        with torch.no_grad():
            _, texel_mask, _, _ = texture.inverse(uv_torch, torch.ones(1, 3, uv_torch.shape[1], uv_torch.shape[2]))
            texel_mask[0, 0] = 0
            texel_count += texel_mask

            import matplotlib.pyplot as plt
            plt.imshow(texel_mask.numpy())
            plt.show()

            plt.imshow(texel_count.numpy())
            plt.show()
        """

        uv[:, :, 0] *= texture_image.shape[0]
        uv[:, :, 1] *= texture_image.shape[1]

        ll = np.floor(uv).astype(np.int)
        lr = np.copy(ll)
        lr[:, :, 0] += 1
        rl = np.copy(ll)
        rl[:, :, 1] += 1
        rr = np.copy(lr)
        rr[:, :, 1] += 1

        for coords in [ll, lr, rl, rr]:
            coords = coords.reshape((-1, 2))

            coords[:, 0] = np.clip(coords[:, 0], 0, texture_image.shape[0] - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, texture_image.shape[1] - 1)

            mask[coords[:, 1], coords[:, 0], 0] = 1
            mask[coords[:, 1], coords[:, 0], 1] = 1
            mask[coords[:, 1], coords[:, 0], 2] = 1
            mask[coords[:, 1], coords[:, 0], 3] = 1

            texel_count[coords[:, 1], coords[:, 0], 0] += 1
            texel_count[coords[:, 1], coords[:, 0], 1] += 1
            texel_count[coords[:, 1], coords[:, 0], 2] += 1
            texel_count[coords[:, 1], coords[:, 0], 3] += 1

    texture_image *= mask

    """
    edges = cv2.Laplacian(mask, cv2.CV_64F)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))

    import matplotlib.pyplot as plt
    plt.imshow(mask * 255)
    plt.show()

    plt.imshow(edges)
    plt.show()

    tex[edges > 0] = 0
    """

    mask_texel_count = texel_count > len(uvs) * 0.02
    texture_image *= mask_texel_count

    out_name = opt.tex[:-4]
    out_name = out_name + "_masked_texel.png"

    import matplotlib.pyplot as plt
    plt.imsave(out_name, texture_image)

    print(f'Saved as: {out_name}')


if __name__ == '__main__':
    parser = ArgumentParser()

    # add all custom flags
    parser.add_argument('--tex', required=True, help="path to texture file")
    parser.add_argument('--uvs', required=True, help="path to dir with all uv maps")

    # parse arguments given from command line (implicitly takes the args from main...)
    args = parser.parse_args()

    # run program with args
    main(args)
