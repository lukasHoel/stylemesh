import cv2

from shutil import move

import os
from os.path import join
import argparse
from tqdm.auto import tqdm

from pathlib import Path

import matplotlib.pyplot as plt

def main(opt):
    stages = ["train", "val", "test"]
    counter = {k: 0 for k in stages}

    # we have train, val and test stage-subfolders
    for stage in stages:
        path = join(opt.dir, stage, "images")
        if os.path.exists(path):

            # for each scan in the stage-subfolder
            print(f"Filtering images in {path}")
            for scan in tqdm(os.listdir(path)):

                # create filtered directories
                filtered_path = join(path, scan, "filtered")
                filtered_image_path = join(filtered_path, "color")
                filtered_label_path = join(filtered_path, "label")
                filtered_instance_path = join(filtered_path, "instance")
                filtered_depth_path = join(filtered_path, "depth")
                filtered_pose_path = join(filtered_path, "pose")
                Path(filtered_path).mkdir(parents=True, exist_ok=True)
                Path(filtered_image_path).mkdir(parents=True, exist_ok=True)
                Path(filtered_label_path).mkdir(parents=True, exist_ok=True)
                Path(filtered_instance_path).mkdir(parents=True, exist_ok=True)
                Path(filtered_depth_path).mkdir(parents=True, exist_ok=True)
                Path(filtered_pose_path).mkdir(parents=True, exist_ok=True)

                if not opt.undo:
                    # loop through images and check which ones need to be filtered
                    image_path = join(path, scan, "color")
                    if os.path.exists(image_path):

                        for image_name in os.listdir(image_path):
                            image = cv2.imread(join(image_path, image_name))
                            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                            variance_of_laplacian = cv2.Laplacian(gray, cv2.CV_64F).var()

                            if variance_of_laplacian < float(opt.treshold):
                                if not opt.debug:
                                    # move images to filtered folder
                                    prefix = image_name.split(".")[0]  # get part of the image before the file suffix, i.e. "123.jpg" --> "123"
                                    move(join(image_path, image_name), filtered_image_path)
                                    move(join(path, scan, "label", f"{prefix}.png"), filtered_label_path)
                                    move(join(path, scan, "instance", f"{prefix}.png"), filtered_instance_path)
                                    move(join(path, scan, "pose", f"{prefix}.txt"), filtered_pose_path)
                                    move(join(path, scan, "depth", f"{prefix}.png"), filtered_depth_path)
                                else:
                                    # visualize image to be filtered
                                    cv2.putText(image, "{}: {:.2f}".format("Blurry: ", variance_of_laplacian), (10, 30),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                                    cv2.imshow(f"{join(image_path, image_name)}", image)
                                    cv2.waitKey(0)
                                    cv2.destroyAllWindows()

                                counter[stage] += 1
                else:
                    # move back color images
                    for img in os.listdir(filtered_image_path):
                        move(join(filtered_image_path, img), join(path, scan, "color"))

                    # move back depth images
                    for depth in os.listdir(filtered_depth_path):
                        move(join(filtered_depth_path, depth), join(path, scan, "depth"))

                    # move back label images
                    for label in os.listdir(filtered_label_path):
                        move(join(filtered_label_path, label), join(path, scan, "label"))

                    # move back instance images
                    for instance in os.listdir(filtered_instance_path):
                        move(join(filtered_instance_path, instance), join(path, scan, "instance"))

                    # move back pose images
                    for pose in os.listdir(filtered_pose_path):
                        move(join(filtered_pose_path, pose), join(path, scan, "pose"))

                    print(f"Moved back all filtered items in {filtered_path}")

    print(f"Filter count: {counter}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='path/to/scannet')
    parser.add_argument('--treshold', required=False, default="150.0", help='minimum value for an image to not be considered blurry')
    parser.add_argument('--debug', default=False, action="store_true", help='if true will only visualize blurry images instead of moving them')
    parser.add_argument('--undo', default=False, action="store_true", help='if true will move back all images in the filtered folder to the original location')

    opt = parser.parse_args()
    main(opt)


