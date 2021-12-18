import os
import argparse
from os.path import join



import subprocess


FNULL = open(os.devnull, 'w')


custom_poses_names = [
    'orthogonal',
    'center',
    'closeup',
    'extremeAndGoodAngles',
    'grazing'
]


def main(args):
    if not args.out:
        args.out = join(os.path.dirname(args.tex), "render_mipmap")

    flip = 0
    if any([p in args.poses for p in custom_poses_names]):
        flip = 1
        print('using flip\n')

    os.chdir(os.path.dirname(args.renderer))
    subprocess.run([args.renderer, args.mesh, args.poses, args.intrinsics, args.out, str(flip), str(args.w), str(args.h), args.tex], stdout=FNULL, stderr=FNULL)

    print(args.renderer, args.mesh, args.poses, args.intrinsics, args.out, str(flip), str(args.w), str(args.h), args.tex)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tex', required=True)
    parser.add_argument('--out', required=False, default=None)
    parser.add_argument('--renderer', required=True, help='path/to/renderer/executable')
    parser.add_argument('--mesh', required=True, help='path/to/mesh')
    parser.add_argument('--poses', required=True, help='path/to/poses')
    parser.add_argument('--intrinsics', required=True, help='path/to/intrinsics')
    parser.add_argument('--h', required=False, default=480, type=int)
    parser.add_argument('--w', required=False, default=640, type=int)

    opt = parser.parse_args()
    main(opt)
