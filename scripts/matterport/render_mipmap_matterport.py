import os
import argparse
from os.path import join



import subprocess


FNULL = open(os.devnull, 'w')


def main(args):
    if not args.out:
        args.out = join(os.path.dirname(args.tex), "render_mipmap")

    flip = 1

    os.chdir(os.path.dirname(args.renderer))
    subprocess.run([args.renderer, args.data_root, args.scene, str(args.region_index), str(flip), str(args.w), str(args.h), args.tex, args.out, '0'], stdout=FNULL, stderr=FNULL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tex', required=True)
    parser.add_argument('--out', required=False, default=None)
    parser.add_argument('--renderer', required=True, help='path/to/renderer/executable')
    parser.add_argument('--data_root', required=True, help='path/to/matterport/v1/scans')
    parser.add_argument('--scene', required=True, help='matterport scene')
    parser.add_argument('--region_index', required=True,  type=int, help='region index of matterport scene')
    parser.add_argument('--h', required=False, default=480, type=int)
    parser.add_argument('--w', required=False, default=640, type=int)

    opt = parser.parse_args()
    main(opt)
