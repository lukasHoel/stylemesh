import os
from os.path import join
import argparse
from tqdm.auto import tqdm

import numpy as np

from pathlib import Path

import subprocess

FNULL = open(os.devnull, 'w')


def main(opt):
    # need to be in renderer dir for shader lookup to work
    os.chdir(os.path.dirname(opt.renderer))

    stages = ["v1/scans"]
    counter = {k: 0 for k in stages}
    flip = '1'

    # we have train, val and test stage-subfolders
    for stage in stages:
        path = join(opt.dir, stage)
        if os.path.exists(path):

            # for each scan in the stage-subfolder
            if opt.verbose:
                print(f"Rendering uvs in {path}")
            for scan in tqdm(os.listdir(path)):
                if opt.scene and scan != opt.scene:
                    print(f"skip {scan} because it does not equal to specified scene {opt.scene}")
                    continue

                if opt.verbose:
                    print(f"Rendering uvs in {join(path, scan)}")

                region_segmentations_path = join(path, scan, 'region_segmentations', scan, 'region_segmentations')
                meshes = [f for f in os.listdir(region_segmentations_path) if 'uvs_blender.ply' in f]
                regions = [m.split("_")[0].replace('region', '') for m in meshes]
                meshes = [join(region_segmentations_path, m) for m in meshes]

                house_segmentations_path = join(path, scan, 'house_segmentations', scan, 'house_segmentations')
                #meshes.extend([join(house_segmentations_path, f) for f in os.listdir(house_segmentations_path) if 'uvs_blender.ply' in f])
                #regions.append('-1')

                for region, mesh in zip(regions, meshes):
                    if not opt.multi_size:
                        runs = [{
                            'root': path,
                            'scan': scan,
                            'region': region,
                            'flip': flip,
                            'w': '-1',
                            'h': '-1',
                        }]
                    else:
                        runs = []
                        heights = np.linspace(opt.multi_size_min, opt.multi_size_max, num=opt.multi_size_steps)
                        widths = [int(round(h * opt.multi_size_aspect)) for h in heights]

                        for h, w in zip(heights, widths):
                            runs.append({
                                'root': path,
                                'scan': scan,
                                'region': region,
                                'flip': flip,
                                'w': str(w),
                                'h': str(int(h)),
                            })
                    for r in runs:
                        if opt.verbose:
                            print(f"...run {r}")

                        uv_name = 'uv' if r['h'] == '-1' and r['w'] == -1 else f'uv_{r["w"]}_{r["h"]}'
                        uv_dir = join(path, scan, 'rendered', f'region_{region}', uv_name)
                        existing = os.path.isdir(uv_dir)

                        if opt.override or not existing:
                            subprocess.run([opt.renderer, *r.values()], stdout=FNULL, stderr=FNULL)
                            counter[stage] += 1
                        else:
                            if opt.verbose:
                                print(f"skip because it is non-empty")

    print(f"Render count: {counter}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='path/to/Matterport3D')
    parser.add_argument('--renderer', required=True, help='path/to/renderer/executable')
    parser.add_argument('--verbose', required=False, default=False, action="store_true", help='prints messages if set to true')
    parser.add_argument('--override', required=False, default=False, action="store_true", help='if existing renderings should be replaced by new renderings')
    parser.add_argument('--scene', required=False, default=None, help='only render for this scene')

    parser.add_argument('--multi_size', required=False, default=False, action='store_true', help='if true, renders uv maps at different resolutions specified by below arguments')
    parser.add_argument('--multi_size_steps', required=False, default=5, type=int, help='this many different scales will be rendered')
    parser.add_argument('--multi_size_min', required=False, default=256, type=int, help='start height of uv maps')
    parser.add_argument('--multi_size_max', required=False, default=960, type=int, help='end height of uv maps')
    parser.add_argument('--multi_size_aspect', required=False, default=1.0*1280/1024, type=float, help='aspect ratio of uv maps')

    opt = parser.parse_args()
    main(opt)


