import argparse

from data.utils import reproject
from model.losses.content_and_style_losses import VGG
from model.losses.rgb_transform import pre, post
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose, Lambda, ToPILImage
import torchmetrics

from tqdm.auto import tqdm
import os
import numpy as np
from os.path import join
import random
import torch
import json
import matplotlib.pyplot as plt

from datetime import datetime
from pathlib import Path

import lpips


def load_intrinsics(intrinsics_file):
    intrinsics = np.identity(4, dtype=np.float32)
    w = 0
    h = 0
    with open(intrinsics_file) as f:
        lines = f.readlines()
        for l in lines:
            l = l.strip()
            if "fx_color" in l:
                fx = float(l.split(" = ")[1])
                intrinsics[0, 0] = fx
            if "fy_color" in l:
                fy = float(l.split(" = ")[1])
                intrinsics[1, 1] = fy
            if "mx_color" in l:
                mx = float(l.split(" = ")[1])
                intrinsics[0, 2] = mx
            if "my_color" in l:
                my = float(l.split(" = ")[1])
                intrinsics[1, 2] = my
            if "colorWidth" in l:
                w = int(l.split(" = ")[1])
            if "colorHeight" in l:
                h = int(l.split(" = ")[1])

    return intrinsics, (w, h)


def modify_intrinsics_matrix(intrinsics, intrinsics_image_size, rgb_image_size):
    if intrinsics_image_size != rgb_image_size:
        intrinsics = np.array(intrinsics)
        intrinsics[0,0] = (intrinsics[0,0] / intrinsics_image_size[0]) * rgb_image_size[0]
        intrinsics[1,1] = (intrinsics[1,1] / intrinsics_image_size[1]) * rgb_image_size[1]
        intrinsics[0,2] = (intrinsics[0,2] / intrinsics_image_size[0]) * rgb_image_size[0]
        intrinsics[1,2] = (intrinsics[1,2] / intrinsics_image_size[1]) * rgb_image_size[1]

    return intrinsics


def load_extrinsics(pose_file):
    extrinsics = open(pose_file, "r").readlines()
    extrinsics = [[float(item) for item in line.split(" ")] for line in extrinsics]
    extrinsics = np.array(extrinsics, dtype=np.float32)

    return extrinsics


def load_depth(depth_file):
    if 'npy' in depth_file:
        d = np.load(depth_file)
        d = d[:, :, :1]  # only keep first channel, but slice it instead of '[:,:,0]' to keep the dim
        return d
    else:
        return Image.fromarray(np.asarray(Image.open(depth_file)) / 1000.0)


def load_image(file):
    return Image.open(file)


def get_image_transform(image_size: int, device):
    return Compose([
        ToTensor(),
        pre(),
        Resize(image_size),
        Lambda(lambda x: x.unsqueeze(0)),
        Lambda(lambda x: x.to(device))
    ])


def get_depth_transform(image_size: int, device):
    return Compose([
        ToTensor(),
        Resize(image_size),
        Lambda(lambda x: x.unsqueeze(0)),
        Lambda(lambda x: x.to(device))
    ])


def get_matrix_transform(device):
    return Compose([
        ToTensor(),
        Lambda(lambda x: x.to(device))
    ])


def get_pil_transform():
    return Compose([
        Lambda(lambda x: x.squeeze(0).cpu()),
        post(),
        ToPILImage()
    ])


def get_01_transform(device):
    return Compose([
        Lambda(lambda x: x.squeeze(0)),
        post(),
        Lambda(lambda x: x.unsqueeze(0)),
        Lambda(lambda x: x.to(device))
    ])


def get_lpips_transform():
    return Compose([
        Lambda(lambda x: x.squeeze(0)),
        post(),
        Lambda(lambda x: x.unsqueeze(0)),
        Lambda(lambda x: x * 2 - 1),
    ])


def get_files(folder, extensions=["jpg", "png", "txt"]):
    # only keep the files with valid extensions, also skip any directories etc.
    files_orig = os.listdir(folder)
    files_orig = [f for f in files_orig if os.path.isfile(join(folder, f))]
    files_orig = [f for f in files_orig if any(f.endswith(x) for x in extensions) and 'masked' not in f and 'intrinsic' not in f]
    try:
        # assume the files are named <int>.<extension>
        files = sorted(files_orig, key=lambda x: int(x.split(".")[0]))
    except:
        try:
            # assume the files are named styled-<int>.<extension> + first filter those that do not match this pattern
            files = [f for f in files_orig if "-" in f.split(".")[0]]
            files = sorted(files, key=lambda x: int(x.split(".")[0].split("-")[1]))
            if len(files) == 0:
                raise ValueError("no files found matching the format styled-<int>.<extension> in", folder)
        except:
            try:
                # assume the files are named styled_<batch>_<idx_in_batch>.<extension> and that styled_0_0 < styled_0_1 < styled_1_0 etc.
                files = sorted(files_orig, key=lambda x: int(x.split(".")[0].split("_")[1])*100+int(x.split(".")[0].split("_")[2]))

                # check if we have styledV2 and styled images - in this case only keep the styledV2 images
                v2 = False
                for f in files:
                    if "styledV2" in f:
                        v2 = True
                        break
                if v2:
                    files = [f for f in files if "styledV2" in f]
            except:
                try:
                    # assume the files are named styled_test_<idx>.<extension>
                    files = [f for f in files_orig if "test" in f]
                    files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[2]))
                    if len(files) == 0:
                        raise ValueError("no files found matching the format styled-<int>.<extension> in", folder)
                except:
                    try:
                        # assume the files are named e.g., 180_stylized_mosaic_2_91.jpg
                        files = sorted(files_orig, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                    except:
                        # assume the files are named e.g., 00_s<idx>_es.<extension>
                        files = sorted(files_orig, key=lambda x: int(x.split(".")[0].split("_")[1][1:]))

    files = [os.path.join(folder, f) for f in files]

    return files


def sample_pairs(n, threshold=10):
    pairs = []
    for i in range(n):
        start = max(0, i-threshold)
        end = min(n, i+threshold)
        pair_index = random.choice([j for j in range(start, end) if j != i])
        pairs.append(pair_index)

    return pairs


def sample_pairs_det(n, threshold=10):
    pairs = []
    for i in range(n):
        left = i - threshold
        right = i + threshold
        pair_index = left if left >= 0 else right if right < n else i
        pairs.append(pair_index)

    return pairs


def main(opt):
    # gpu check
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # create out dirs + files
    now = datetime.now()  # current date and time
    date_time = now.strftime("%d.%m.%Y-%H:%M:%S")
    if not opt.out_dir:
        opt.out_dir = opt.styled
    image_dir = join(opt.out_dir, f"eval_image_data_{date_time}")
    Path(opt.out_dir).mkdir(parents=True, exist_ok=True)
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    out_file = os.path.join(opt.out_dir, f"{date_time}_output.txt")

    # for sample_pairs
    random.seed(opt.random_seed)

    # to torch, also adds batch_dimension of 1
    transform_image = get_image_transform(opt.image_size, device)
    transform_depth = get_depth_transform(opt.image_size, device)
    transform_matrix = get_matrix_transform(device)
    transform_pil = get_pil_transform()
    transform_01 = get_01_transform(device)

    vgg = VGG(model_path=opt.vgg_model_path).to(device)

    reprojection_accuracy = torchmetrics.MeanSquaredError(compute_on_step=False)
    short_reprojection_accuracy = torchmetrics.MeanSquaredError(compute_on_step=False)
    long_reprojection_accuracy = torchmetrics.MeanSquaredError(compute_on_step=False)

    lpips_accuracy = lpips.LPIPS(net='vgg')  # closer to "traditional" perceptual loss, when used for optimization
    lpips_accuracy.to(device)
    lpips_accuracy.eval()
    reprojection_accuracy_lpips = []
    short_reprojection_accuracy_lpips = []
    long_reprojection_accuracy_lpips = []

    # find all files, ensure consistency
    rgb_images = get_files(opt.rgb, extensions=["jpg", "png"])
    styled_images = get_files(opt.styled, extensions=["jpg", "png"])
    depth_images = get_files(opt.depth, extensions=["jpg", "png"])
    if len(depth_images) == 0:
        depth_images = get_files(opt.depth, extensions=["rendered_depth.npy"])
    pose_files = get_files(opt.pose, extensions=["txt"])
    assert len(rgb_images) == len(styled_images), f'rgb-path: {opt.rgb}, styled-path: {opt.styled}'
    assert len(styled_images) == len(depth_images), f'depth-path: {opt.depth}, styled-path: {opt.styled}'
    assert len(depth_images) == len(pose_files), f'depth-path: {opt.depth}, pose-path: {opt.pose}'

    # intrinsics is same for all files
    intrinsics, (w,h) = load_intrinsics(opt.intrinsics)
    h_t, w_t = transform_image(load_image(rgb_images[0])).shape[2:4]
    intrinsics = modify_intrinsics_matrix(intrinsics, (w,h), (w_t, h_t))
    intrinsics = transform_matrix(intrinsics)


    # style image is same for all files
    style_image = load_image(opt.style_image)
    style_image = transform_image(style_image)
    # resize spatial dimension
    style_image = torch.nn.functional.interpolate(style_image, (h_t, w_t), mode="bilinear")
    style_image_pil = transform_pil(style_image.clone())

    # pair-wise matching for reprojection
    pairs = sample_pairs(len(rgb_images), opt.pair_threshold)
    short_pairs = sample_pairs_det(len(rgb_images), opt.pair_threshold_short)
    long_pairs = sample_pairs_det(len(rgb_images), opt.pair_threshold_long)

    def masked(img, mask3):
        return torch.where(mask3.bool(), img, torch.zeros_like(img))

    def masked_flatten(img, mask3):
        return torch.masked_select(img, mask3)

    def check_size(styled, rgb):
        # sanity check: if styled image not same size as everything else, resize it accordingly
        if styled.shape != rgb.shape:
            styled = torch.nn.functional.interpolate(styled, (h_t, w_t), mode="bilinear")
        return styled

    def eval_reprojection(accuracy, lpips, lpips_list, pairs, rgb, styled, depth, pose):
        with torch.no_grad():
            # reprojection accuracy also needs files of paired frame
            pose_other = load_extrinsics(pose_files[pairs[i]])
            pose_other = transform_matrix(pose_other)
            depth_other = load_depth(depth_images[pairs[i]])
            depth_other = transform_depth(depth_other)
            styled_other = load_image(styled_images[pairs[i]])
            styled_other = transform_image(styled_other)
            styled_other = check_size(styled_other, rgb)
            mask_other = (depth_other > 0).squeeze(1)

            # calculate reprojection + mask
            styled_reprojected, mask = reproject(pose, pose_other, styled_other.shape[3], styled_other.shape[2],
                                                intrinsics, depth, depth_other, styled_other, mask_other)

            # calculate reprojection accuracy as MSE of masked regions
            mask3 = torch.stack([mask, mask, mask], dim=1)  # from (B x H x W) to (B x 3 x H x W)
            accuracy(masked_flatten(styled.clone(), mask3).cpu().detach(), masked_flatten(styled_reprojected.clone(), mask3).cpu().detach())

            # calculate lpips accuracy of masked regions
            lpips_list.append(lpips(masked(styled.clone(), mask3), masked(styled_reprojected.clone(), mask3)).cpu().detach())

        return styled_other, styled_reprojected, mask3

    # calculate accuracies per file
    print("START: Calculate accuracies per frame")
    for i, (rgb, styled, depth, pose) in enumerate(tqdm(zip(rgb_images, styled_images, depth_images, pose_files), total=len(rgb_images))):
        # load all files
        rgb = load_image(rgb)
        rgb = transform_image(rgb)
        styled = load_image(styled)
        styled = transform_image(styled)
        styled = check_size(styled, rgb)
        depth = load_depth(depth)
        depth = transform_depth(depth)
        pose = load_extrinsics(pose)
        pose = transform_matrix(pose)

        # reprojection randomly
        styled_other, styled_reprojected, mask3 = eval_reprojection(reprojection_accuracy, lpips_accuracy, reprojection_accuracy_lpips, pairs, rgb, styled, depth, pose)

        # reprojection short and long
        styled_other_short, styled_reprojected_short, mask3_short = eval_reprojection(short_reprojection_accuracy, lpips_accuracy, short_reprojection_accuracy_lpips, short_pairs, rgb, styled, depth, pose)
        styled_other_long, styled_reprojected_long, _ = eval_reprojection(long_reprojection_accuracy, lpips_accuracy, long_reprojection_accuracy_lpips, long_pairs, rgb, styled, depth, pose)

        # save images for visualization
        residual_image = torch.abs(masked(styled, mask3) - masked(styled_reprojected, mask3))
        residual_image = transform_pil(residual_image)
        residual_image.save(join(image_dir, f"residual_image_{i}.jpg"))
        rgb = transform_pil(rgb)
        rgb.save(join(image_dir, f"rgb_{i}.jpg"))
        styled = transform_pil(styled)
        styled.save(join(image_dir, f"styled_{i}.jpg"))
        styled_other = transform_pil(styled_other)
        styled_other.save(join(image_dir, f"styled_other_{i}_{pairs[i]}.jpg"))
        styled_reprojected = transform_pil(styled_reprojected)
        styled_reprojected.save(join(image_dir, f"styled_reprojected_{i}.jpg"))
        styled_other_short = transform_pil(styled_other_short)
        styled_other_short.save(join(image_dir, f"styled_other_short_{i}_{short_pairs[i]}.jpg"))
        styled_reprojected_short = transform_pil(styled_reprojected_short)
        styled_reprojected_short.save(join(image_dir, f"styled_reprojected_short_{i}.jpg"))
        styled_other_long = transform_pil(styled_other_long)
        styled_other_long.save(join(image_dir, f"styled_other_long_{i}_{long_pairs[i]}.jpg"))
        styled_reprojected_long = transform_pil(styled_reprojected_long)
        styled_reprojected_long.save(join(image_dir, f"styled_reprojected_long_{i}.jpg"))

        if opt.debug:
            print("Index", i)
            print("Intrinsics", intrinsics)
            print("Other index", pairs[i])
            print("Other index short", short_pairs[i])
            print("Other index long", long_pairs[i])

            fig, ax = plt.subplots(1, 13) if not opt.only_reprojection else plt.subplots(1, 10)
            ax[0].imshow(rgb)
            ax[1].imshow(styled)
            ax[2].imshow(styled_other)
            ax[3].imshow(styled_reprojected)
            ax[4].imshow(style_image_pil)
            ax[5].imshow(residual_image)
            ax[6].imshow(styled_other_short)
            ax[7].imshow(styled_reprojected_short)
            ax[8].imshow(styled_other_long)
            ax[9].imshow(styled_reprojected_long)
            plt.show()

    print("END: Calculate accuracies per frame")

    # Save results to file
    with open(out_file, "w") as f:
        params = opt.__dict__
        params["number_files"] = len(rgb_images)
        params["date_time"] = date_time
        params["pairs"] = pairs
        params["long_pairs"] = long_pairs
        params["short_pairs"] = short_pairs

        reprojection_acc = reprojection_accuracy.compute()
        print("Reprojection Accuracy", reprojection_acc)

        reprojection_acc_short = short_reprojection_accuracy.compute()
        print("Reprojection Accuracy Short", reprojection_acc_short)

        reprojection_acc_long = long_reprojection_accuracy.compute()
        print("Reprojection Accuracy Long", reprojection_acc_long)

        reprojection_accuracy_lpips = sum([f.sum().cpu().numpy().item() for f in reprojection_accuracy_lpips])
        print("Reprojection Accuracy LPIPS", reprojection_accuracy_lpips)

        short_reprojection_accuracy_lpips = sum([f.sum().cpu().numpy().item() for f in short_reprojection_accuracy_lpips])
        print("Reprojection Accuracy Short LPIPS", short_reprojection_accuracy_lpips)

        long_reprojection_accuracy_lpips = sum([f.sum().cpu().numpy().item() for f in long_reprojection_accuracy_lpips])
        print("Reprojection Accuracy Long LPIPS", long_reprojection_accuracy_lpips)

        params["accuracies"] = {
            "reprojection": reprojection_acc.cpu().numpy().item(),
            "reprojection_short": reprojection_acc_short.cpu().numpy().item(),
            "reprojection_long": reprojection_acc_long.cpu().numpy().item(),
            "reprojection_lpips": reprojection_accuracy_lpips,
            "reprojection_short_lpips": short_reprojection_accuracy_lpips,
            "reprojection_long_lpips": long_reprojection_accuracy_lpips
        }

        json.dump(params, f, indent=2)

    print("Saved results as:", out_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rgb', required=True, help='path to rgb image folder')
    parser.add_argument('--styled', required=True, help='path to styled image folder')
    parser.add_argument('--pose', required=True, help='path to pose folder (extrinsic camera matrices per frame)')
    parser.add_argument('--intrinsics', required=True, help='path to intrinsics file')
    parser.add_argument('--depth', required=True, help='path to depth image folder')
    parser.add_argument('--vgg_model_path', required=True, help='path to pretrained vgg model')
    parser.add_argument('--style_image', required=True, help='path to style image')
    parser.add_argument('--random_seed', required=False, help='random seed for image pairing', default=42, type=int)
    parser.add_argument('--out_dir', required=False, help='output directory', default=None)
    parser.add_argument('--debug', required=False, default=False, action="store_true", help='shows each frame results via matplotlib before continuing with next frame')
    parser.add_argument('--image_size', required=False, default=256, type=int, help='image height for everything, width will scale accordingly')
    parser.add_argument('--pair_threshold', required=False, default=20, type=int, help='pair threshold, pair will not be more than n frames before/after current frame')
    parser.add_argument('--pair_threshold_short', required=False, default=1, type=int, help='pair threshold for short evaluation')
    parser.add_argument('--pair_threshold_long', required=False, default=10, type=int, help='pair threshold for long evaluation')
    opt = parser.parse_args()

    main(opt)