from argparse import ArgumentParser

import numpy as np
import numpy, json
import torch
from tqdm.auto import tqdm

import cv2 as cv
import cv2
import os

import random as rng
from os.path import join

from scipy.ndimage import label

from data.scannet_single_scene_dataset import ScanNet_Single_House_Dataset
from model.texture.utils import get_rgb_transform, get_uv_transform, get_label_transform
from data.utils import unproject
from torchvision.transforms import Resize, ToTensor, Compose
from model.gatys_style.rgb_transform import pre, post
from scipy.spatial import distance as dist


def filter_hsv(src):
    # It converts the BGR color space of image to HSV color space
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    # Threshold of red in HSV space (beginning of H)
    lower_red = np.array([0, int(0.6 * 255), int(0.6 * 255)])
    upper_red = np.array([15, int(1.0 * 255), int(1.0 * 255)])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Threshold of red in HSV space (end of H)
    lower_red = np.array([160, int(0.4 * 255), int(0.4 * 255)])
    upper_red = np.array([179, int(1.0 * 255), int(1.0 * 255)])
    mask += cv2.inRange(hsv, lower_red, upper_red)

    # The black region in the mask has the value of 0,
    # so when multiplied with original image removes all non-blue regions
    hsv_filtered = cv2.bitwise_and(src, src, mask=mask)

    return hsv_filtered


def cov_from_lookup(center, ys, lut, filter_zero=True):
    assert len(center) == len(ys)

    xs = []
    for c in center:
        pos_w, pos_h = c
        x = lut[pos_h, pos_w, 0]
        xs.append(x)

    xy = sorted(zip(xs, ys), key=lambda pair: pair[0])

    if filter_zero:
        xy = [i for i in xy if i[0] != 0]

    xs = [i[0] for i in xy]
    ys = [i[1] for i in xy]

    A = np.array([xs, ys])
    corr = np.corrcoef(A)[0, 1]

    return corr, A, xs, ys


def order_points(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return tl, tr, br, bl


def mid_point(a, b):
    return a + (b - a) / 2.0


def is_in_range(p, w, h):
    x, y = p
    x = round(x)
    y = round(y)
    return 0 <= x < w and 0 <= y < h


def clamp_to_range(p, w, h):
    x, y = p
    x = round(x)
    y = round(y)
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))

    return x, y


def ellipse_stats(a, b):
    radius = (a / 2.0 + b / 2.0) / 2.0  # average of both diameters
    stretch = abs(1.0 * a / b) if a > b else abs(1.0 * b / a)
    size = a * b

    if a == 0 or b == 0:
        raise ValueError('foo')

    return radius, stretch, size


def is_valid_point(p, depth):
    w, h = p
    return depth.squeeze()[h, w] > 0


def median_radius_level(radii, opt, suffix=''):
    statistics = {
        f"smallest{suffix}": 0,
        f"small{suffix}": 0,
        f"large{suffix}": 0,
        f"largest{suffix}": 0
    }

    colors = []

    median_radius = np.median(np.array(radii))
    t = opt.t
    n = len(radii)

    for radius in radii:
        if radius < median_radius / t:
            colors.append((255, 0, 0))  # blue
            k = 'smallest'
        elif radius < median_radius:
            colors.append((0, 255, 0))  # green
            k = 'small'
        elif median_radius < radius < median_radius * t:
            colors.append((0, 255, 255))  # yellow
            k = 'large'
        else:
            colors.append((255, 0, 255))  # purple
            k = 'largest'
        k = f"{k}{suffix}"
        statistics[k] = statistics[k] + 1

    statistics = {k: v * 1.0 / n for k, v in statistics.items()}

    return statistics, n, colors


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in degrees between vectors 'v1' and 'v2'::
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)

    dot = np.dot(v1_u, v2_u)
    dot_clip = np.clip(dot, -1.0, 1.0)
    arccos_dot_clip = np.arccos(dot_clip)

    deg_arccos_dot_clip = np.rad2deg(arccos_dot_clip)

    return deg_arccos_dot_clip


def measure(img, depth, angle, coords, opt):
    # read src rgb image
    src = cv.imread(img)
    img_h, img_w, _ = src.shape

    # keep the red circles only (this helps in denoising the image)
    hsv_filtered = filter_hsv(src)

    # create bw image for contour detection
    gray = cv.cvtColor(hsv_filtered, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(gray, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    bw = cv.fastNlMeansDenoising(bw, h=100)
    _, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    # make it an RGB image again for verbose visualization later
    bw_color = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(image=bw, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)

    # detect ellipses from contours
    ellipses = []
    radii = []
    centers = []
    horizontal_edges = []
    vertical_edges = []
    stretches = []
    for cnt in contours:
        try:
            # get convex hull and check how many defects we have. we are interested in the biggest deviation
            # https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gada4437098113fd8683c932e0567f47ba
            # https://docs.opencv.org/master/d3/dc0/group__imgproc__shape.html#gada4437098113fd8683c932e0567f47ba
            hull = cv.convexHull(cnt, returnPoints=False)
            defects = cv.convexityDefects(cnt, hull)
            max_deviation = 0
            if defects is not None:
                for d in defects:
                    max_deviation = max(max_deviation, d[0][3] / 256.0)

            # filter all those contours that differ a lot from its convex hull
            # we do this, because many red areas are not really circles, but noisy non-perfect stylizations
            # circles and ellipses deviate from their convex hull by a small degree, whereas noisy areas deviate a lot more
            if max_deviation > 2: # 2
                continue

            # fit an ellipse to the contour and check ellipse attributes
            ellipse = cv.fitEllipse(cnt)

            center = ellipse[0]
            w, h = ellipse[1]

            if w == 0 or h == 0:
                continue

            radius, stretch, size = ellipse_stats(w, h)

            # filter values for ellipse shape
            theta_stretch = 10  # this amount of stretch is still tolerated (everything else is considered noise)
            theta_min_size = 10  # ellipses smaller than this are considered noise
            theta_max_size = 10000  # ellipses larger than this are considered noise

            if stretch < theta_stretch and theta_min_size < size < theta_max_size:
                ellipses.append(ellipse)
                radii.append(radius)
                stretches.append(stretch)

                box = cv.boxPoints(ellipse)
                tl, tr, br, bl = order_points(box)

                half_tr_br = mid_point(tr, br)
                half_tl_bl = mid_point(tl, bl)
                half_tl_tr = mid_point(tl, tr)
                half_bl_br = mid_point(bl, br)

                if is_in_range(half_tr_br, img_w, img_h):
                    horizontal_edges.append(clamp_to_range(half_tr_br, img_w, img_h))
                else:
                    horizontal_edges.append(clamp_to_range(half_tl_bl, img_w, img_h))

                if is_in_range(half_tl_tr, img_w, img_h):
                    vertical_edges.append(clamp_to_range(half_tl_tr, img_w, img_h))
                else:
                    vertical_edges.append(clamp_to_range(half_bl_br, img_w, img_h))

                centers.append(clamp_to_range(center, img_w, img_h))

                if not (is_valid_point(centers[-1], depth) and is_valid_point(horizontal_edges[-1], depth) and is_valid_point(vertical_edges[-1], depth)):
                    if opt.verbose:
                        print('Skip point because depth is zero')
                    centers = centers[:-1]
                    horizontal_edges = horizontal_edges[:-1]
                    vertical_edges = vertical_edges[:-1]
                    ellipses = ellipses[:-1]
                    radii = radii[:-1]
                    stretches = stretches[:-1]

        except Exception as e:
            filtered_errors = [
                "-201:Incorrect size of input array",  # opencv cannot detect an ellipse for every contour, this is ok
                "(-5:Bad argument) The convex hull indices are not monotonous", # opencv cannot detect defects for non-monotonous convex hulls, this is ok
            ]
            # opencv cannot detect defects for non-monotonous convex hulls, this is ok
            # every other exception should still be raised
            msg = str(e)
            if all([i not in msg for i in filtered_errors]):
                raise e

    # 3D calculations
    radii_3D = []
    stretches_3D = []
    centers_3D = []
    for c, he, ve in zip(centers, horizontal_edges, vertical_edges):
        if not (is_valid_point(c, depth) and is_valid_point(he, depth) and is_valid_point(ve, depth)):
            if opt.verbose:
                print('Skip point because depth is zero for at least one of the pixel coordinates: ', c, he, ve)
            continue

        # get the 3D point from the coordinates
        C = coords[c[1], c[0], :3]
        HE = coords[he[1], he[0], :3]
        VE = coords[ve[1], ve[0], :3]

        # width/height of ellipse is now the 3D norm of the 3D points
        A = HE - C
        B = VE - C

        a = np.linalg.norm(A)
        b = np.linalg.norm(B)

        if a == 0 or b == 0:
            continue

        # calculate same statistics as above, but now from the 3D ellipse
        radius, stretch, _ = ellipse_stats(a, b)

        radii_3D.append(radius)
        stretches_3D.append(stretch)
        centers_3D.append(c)

    # calculate base statistics of all ellipse sizes
    statistics, n, colors = median_radius_level(radii, opt, '_2D')
    statistics_3D, _, _ = median_radius_level(radii_3D, opt, '_3D')
    statistics.update(statistics_3D)

    # calculate depth VS radius
    corr_depth, A_depth, xs_depth, ys_depth = cov_from_lookup(centers, radii, depth)
    statistics['corr_depth_2D'] = corr_depth

    corr_depth_3D, A_depth_3D, xs_depth_3D, ys_depth_3D = cov_from_lookup(centers_3D, radii_3D, depth)
    statistics['corr_depth_3D'] = corr_depth_3D

    # calculate angle VS stretch
    corr_angle, _, xs_angle, ys_angle = cov_from_lookup(centers, stretches, angle)
    statistics['corr_angle_2D'] = corr_angle
    statistics['mean_stretch_2D'] = np.mean(ys_angle)
    statistics['median_stretch_2D'] = np.median(ys_angle)
    statistics['std_stretch_2D'] = np.std(ys_angle)

    corr_angle_3D, _, xs_angle_3D, ys_angle_3D = cov_from_lookup(centers_3D, stretches_3D, angle)
    statistics['corr_angle_3D'] = corr_angle_3D
    statistics['mean_stretch_3D'] = np.mean(ys_angle_3D)
    statistics['median_stretch_3D'] = np.median(ys_angle_3D)
    statistics['std_stretch_3D'] = np.std(ys_angle_3D)

    if opt.verbose:
        #cv.imshow("color", src)
        #cv.imshow("bw", bw)
        #cv.imshow("hsv_filtered", hsv_filtered)
        cv.imshow("depth", depth / 10.0)
        #cv.imshow("angle", angle / 90.0)
        print("number contours", len(contours))

        # Draw all contours
        # -1 signifies drawing all contours
        contour_img = np.zeros_like(src)
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 1)
        cv.imshow("contours", contour_img)

        for i, color, he, ve in zip(ellipses, colors, horizontal_edges, vertical_edges):
            cv.ellipse(src, i, color, thickness=2)
            cv.circle(src, (int(i[0][0]), int(i[0][1])), 1, color, thickness=1)
            cv.circle(src, (he[0], he[1]), 1, (0, 0, 255), thickness=2)
            cv.circle(src, (ve[0], ve[1]), 1, (0, 0, 255), thickness=2)

            cv.ellipse(bw_color, i, color, thickness=2)
            cv.circle(bw_color, (int(i[0][0]), int(i[0][1])), 1, color, thickness=1)
            cv.circle(bw_color, (he[0], he[1]), 1, (0, 0, 255), thickness=2)
            cv.circle(bw_color, (ve[0], ve[1]), 1, (0, 0, 255), thickness=2)

        statistics_str = {k: str(v) for k, v in statistics.items()}
        print(f"N: {n}, Statistics: {json.dumps(statistics_str, indent=4)}")
        cv.imshow("detected ellipses", src)
        #cv.imshow("detected ellipses bw", bw_color)
        cv.waitKey(0)

        import matplotlib.pyplot as plt
        plt.scatter(xs_depth, ys_depth)
        plt.title('2D: Depth (X) vs Radius (Y)')
        #plt.show()

        import matplotlib.pyplot as plt
        plt.scatter(xs_angle, ys_angle)
        plt.title('2D: Angle (X) vs Stretch (Y)')
        #plt.show()

        import matplotlib.pyplot as plt
        plt.scatter(xs_depth_3D, ys_depth_3D)
        plt.title('3D: Depth (X) vs Radius (Y)')
        #plt.show()

        import matplotlib.pyplot as plt
        plt.scatter(xs_angle_3D, ys_angle_3D)
        plt.title('3D: Angle (X) vs Stretch (Y)')
        #plt.show()

    return statistics, n


def main(opt):
    if not os.path.isdir(opt.dir):
        return

    # get images
    extensions = ["jpg", "png"]
    files = os.listdir(opt.dir)
    files = [f for f in files if any(f.endswith(x) for x in extensions)]
    # files = [f for f in files if not "tex" in f and "V2" in f]
    # files = [f for f in files if "styled" in f and not "reprojected" in f and not "other" in f]
    files = sorted(files, key=lambda x: int(x.split(".")[0]))
    # files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[1]))
    # files = sorted(files, key=lambda x: int(x.split(".")[0].split("_")[1]) * 100 + int(x.split(".")[0].split("_")[2]))
    files = [join(opt.dir, f) for f in files]

    n_total = 0
    statistics_total = {}

    transform_rgb = Compose([
        get_rgb_transform(),
        pre()
    ])
    transform_label = get_label_transform()
    transform_uv = get_uv_transform()
    d = ScanNet_Single_House_Dataset(root_path="/path/to/datasets/scannet/train/images",
                                     scene=opt.scene,
                                     verbose=True,
                                     transform_rgb=transform_rgb,
                                     transform_label=transform_label,
                                     transform_uv=transform_uv,
                                     pyramid_levels=1,
                                     min_pyramid_depth=0.25,
                                     min_pyramid_height=256,
                                     resize_size=480,
                                     max_images=1000,
                                     min_images=1)

    max_difference = 0
    min_difference = 10
    max_idx = 0
    min_idx = 0

    for i, (f, item) in enumerate(tqdm(zip(files, d), total=len(files))):
        depth_torch = item[5]
        extrinsics = item[3]
        intrinsics = torch.from_numpy(item[4])

        coords = unproject(extrinsics.unsqueeze(0), intrinsics.unsqueeze(0), depth_torch.unsqueeze(0))
        coords = coords.squeeze().numpy()

        depth = depth_torch.permute(1,2,0).numpy()
        angle = item[-1].permute(1,2,0).numpy()

        opt.verbose = i % 100000 == 0

        statistics, n = measure(f, depth, angle, coords, opt)
        n_total += n
        for k in statistics.keys():
            if k not in statistics_total:
                statistics_total[k] = 0
            statistics_total[k] += statistics[k] * n

        difference = abs(statistics["corr_depth_2D"] - statistics["corr_depth_3D"])
        if difference > max_difference:
            max_idx = item[10]
            max_difference = difference
            print('new max idx', max_idx, statistics["corr_depth_2D"], statistics["corr_depth_3D"])
        if difference < min_difference:
            min_idx = item[10]
            min_difference = difference
            print('new min idx', min_idx, statistics["corr_depth_2D"], statistics["corr_depth_3D"])

    for k in statistics_total.keys():
        statistics_total[k] /= n_total

    print(json.dumps(statistics_total, indent=4))
    print('min', min_idx, 'max', max_idx)


if __name__ == '__main__':
    parser = ArgumentParser()

    # add all custom flags
    parser.add_argument('--dir', required=True, help="path to images")
    parser.add_argument('--scene', required=True, help="scannet scene", default='scene0481_00_extremeAndGoodAngles')
    parser.add_argument('--t', required=False, type=int, default=1.5, help="scale factor to create the 4 categories")
    parser.add_argument('--verbose', required=False, action="store_true", help="show each circle detection image")

    # parse arguments given from command line (implicitly takes the args from main...)
    args = parser.parse_args()

    # run program with args
    main(args)
