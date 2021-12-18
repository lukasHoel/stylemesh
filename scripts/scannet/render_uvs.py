import os
from os.path import join
import argparse
from tqdm.auto import tqdm

import numpy as np

from pathlib import Path

import subprocess

FNULL = open(os.devnull, 'w')

custom_poses_names = [
    'orthogonal',
    'center',
    'closeup',
    'extremeAndGoodAngles'
]


def main(opt):
    stages = ["train", "val", "test"]
    counter = {k: 0 for k in stages}

    # we have train, val and test stage-subfolders
    for stage in stages:
        path = join(opt.dir, stage, "images")
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

                flip = '0'
                if any([p in scan for p in custom_poses_names]):
                    flip = '1'
                    print('using flip\n')

                truncated_scene = scan.split("_")
                truncated_scene = truncated_scene[:2]  # scene0673_00_orthogonal -> scene0673_00. they share same mesh!
                truncated_scene.insert(1, "_")
                truncated_scene = "".join(truncated_scene)
                if opt.no_decimate:
                    mesh_name = f"{truncated_scene}_vh_clean_uvs_blender.ply"
                else:
                    mesh_name = f"{truncated_scene}_vh_clean_decimate_{opt.decimate_number}_uvs_blender.ply"
                mesh_path = join(opt.dir, stage, "scans", truncated_scene, mesh_name)
                intr_path = join(opt.dir, stage, "scans", truncated_scene, f"{truncated_scene}.txt")

                if not os.path.exists(mesh_path):
                    if opt.verbose:
                        print(f"skip {scan} because mesh {mesh_path} does not exist")
                    continue

                if not os.path.exists(intr_path):
                    if opt.verbose:
                        print(f"skip {scan} because intrinsics file {intr_path} does not exist")
                    continue

                if not opt.multi_size:
                    runs = [{
                        'uv': join(path, scan, "uv"),
                        "uv_noise": join(path, scan, f"uv{opt.noise_suffix}"),
                        'pose': join(path, scan, "pose"),
                        "pose_noise": join(path, scan, f"pose{opt.noise_suffix}"),
                        'h': '480',
                        'w': '640'
                    }]
                else:
                    runs = []
                    heights = np.linspace(opt.multi_size_min, opt.multi_size_max, num=opt.multi_size_steps)
                    widths = [int(round(h * opt.multi_size_aspect)) for h in heights]

                    for h, w in zip(heights, widths):
                        runs.append({
                            'uv': join(path, scan, f"uv_{h}"),
                            "uv_noise": join(path, scan, f"uv_{h}{opt.noise_suffix}"),
                            'pose': join(path, scan, "pose"),
                            "pose_noise": join(path, scan, f"pose{opt.noise_suffix}"),
                            'h': str(h),
                            'w': str(w)
                        })
                for r in runs:
                    if opt.verbose:
                        print(f"...run {r}")
                    Path(r['uv']).mkdir(parents=True, exist_ok=True)
                    Path(r['uv_noise']).mkdir(parents=True, exist_ok=True)

                    if opt.override or len(os.listdir(r['uv'])) == 0:
                        subprocess.run([opt.renderer, mesh_path, r['pose'], intr_path, r['uv'], flip, r['w'], r['h']], stdout=FNULL, stderr=FNULL)
                        counter[stage] += 1
                    else:
                        if opt.verbose:
                            print(f"skip {r['uv']} because it is non-empty")

                    if opt.override or len(os.listdir(r['uv_noise'])) == 0:
                        subprocess.run([opt.renderer, mesh_path, r['pose_noise'], intr_path, r['uv_noise'], flip, r['w'], r['h']], stdout=FNULL, stderr=FNULL)
                        counter[stage] += 1
                    else:
                        if opt.verbose:
                            print(f"skip {r['uv_noise']} because it is non-empty")

    print(f"Render count: {counter}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='path/to/scannet')
    parser.add_argument('--renderer', required=True, help='path/to/renderer/executable')
    parser.add_argument('--decimate_number', required=False, default=500000, type=int, help='decimate face number to use for looking up the mesh file')
    parser.add_argument('--no_decimate', required=False, default=False, action="store_true", help='if the mesh with or without decimation should be chosen (i.e. <scene>_decimate_500000_uvs_blender.ply or <scene>_uvs_blender.ply)')
    parser.add_argument('--verbose', required=False, default=False, action="store_true", help='prints messages if set to true')
    parser.add_argument('--override', required=False, default=False, action="store_true", help='if existing renderings should be replaced by new renderings')
    parser.add_argument('--noise_suffix', required=False, default="_noise", help='suffix of pose noise folder, i.e. pose_noise')
    parser.add_argument('--scene', required=False, default=None, help='only render for this scene')

    parser.add_argument('--multi_size', required=False, default=False, action='store_true', help='if true, renders uv maps at different resolutions specified by below arguments')
    parser.add_argument('--multi_size_steps', required=False, default=5, type=int, help='this many different scales will be rendered')
    parser.add_argument('--multi_size_min', required=False, default=256, type=int, help='start height of uv maps')
    parser.add_argument('--multi_size_max', required=False, default=960, type=int, help='end height of uv maps')
    parser.add_argument('--multi_size_aspect', required=False, default=1.0*1280/960, type=float, help='aspect ratio of uv maps')

    opt = parser.parse_args()
    main(opt)


