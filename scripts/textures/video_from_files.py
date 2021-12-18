import cv2

import os
from os.path import join
import argparse
from tqdm.auto import tqdm

from PIL import Image

import numpy as np


def main(opt):

    if not os.path.isdir(opt.imgs_dir):
        return

    out_dir = opt.out_dir if opt.out_dir else opt.imgs_dir

    # get images
    extensions = ["jpg", "png"]
    files = os.listdir(opt.imgs_dir)
    files = [f for f in files if any(f.endswith(x) for x in extensions) and 'masked' not in f]
    try:
        # scannet sorting
        files = sorted(files, key=lambda x: int(x.split(".")[0]))
    except:
        # matterport sorting
        files = sorted(files, key=lambda x: [x.split(".")[0].split('_')[0], int(x.split(".")[0].split('_')[1][1]) * 100 + int(x.split(".")[0].split('_')[2])])

    files = [join(opt.imgs_dir, f) for f in files]

    # get size of all images
    img1 = files[0]
    img1 = Image.open(img1)
    img1 = np.asarray(img1)

    # create video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(join(out_dir, "video.mp4"), fourcc, 20.0, (img1.shape[1], img1.shape[0]))

    for f in tqdm(files):
        img = Image.open(f)
        img = np.asarray(img)

        if hasattr(opt, 'flip') and opt.flip:
            img = cv2.flip(img, 0)

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgs_dir', required=True)
    parser.add_argument('--flip', required=False, default=False, action="store_true")
    parser.add_argument('--out_dir', required=False, default=None)

    opt = parser.parse_args()
    main(opt)


