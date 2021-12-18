# pre-process ScanNet 2D data
# note: depends on the sens file reader from ScanNet:
#       https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
# if export_label_images flag is on:
#   - depends on https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/util.py
#   - also assumes that label images are unzipped as scene*/label*/*.png
# expected file structure:
#  - prepare_2d_data.py
#  - https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/util.py
#  - https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/SensorData.py
#
# example usage:
#    python prepare_2d_data.py --scannet_path data/scannetv2 --output_path data/scannetv2_images --export_label_images

import argparse
import os, sys
import numpy as np
import skimage.transform as sktf
import imageio

try:
    from SensorData import SensorData
except:
    print('Failed to import SensorData (from ScanNet code toolbox)')
    sys.exit(-1)
try:
    import util
except:
    print('Failed to import ScanNet code toolbox util')
    sys.exit(-1)


# params
parser = argparse.ArgumentParser()
parser.add_argument('--scannet_path', required=True, help='path to scannet data')
parser.add_argument('--output_path', required=True, help='where to output 2d data')
parser.add_argument('--export_label_images', dest='export_label_images', action='store_true')
parser.add_argument('--label_type', default='label-filt', help='which labels (label or label-filt)')
parser.add_argument('--frame_skip', type=int, default=20, help='export every nth frame')
parser.add_argument('--label_map_file', default='', help='path to scannetv2-labels.combined.tsv (required for label export only)')
parser.add_argument('--output_image_width', type=int, default=320, help='export image width')
parser.add_argument('--output_image_height', type=int, default=240, help='export image height')

parser.set_defaults(export_label_images=False)
opt = parser.parse_args()
if opt.export_label_images:
    assert opt.label_map_file != ''
print(opt)


def print_error(message):
    sys.stderr.write('ERROR: ' + str(message) + '\n')
    sys.exit(-1)

# from https://github.com/ScanNet/ScanNet/tree/master/BenchmarkScripts/2d_helpers/convert_scannet_label_image.py
def map_label_image(image, label_mapping):
    mapped = np.copy(image)

    # boolean mask image. True: this pixel was not mapped. False: this pixel was mapped
    unmapped_pixels = np.ones_like(image, dtype=np.bool)

    for k,v in label_mapping.items():
        mapped[image==k] = v
        unmapped_pixels[image==k] = 0 # mark all pixels as mapped that are equal to k

    # the zero value is not contained in the scannet mapping file but it is also correct that this value stays zero.
    # so we do not count these cases, because they are "correct" although not mapped.
    unmapped_pixels[image==0] = 0

    if np.sum(unmapped_pixels) > 0:
        #print(f"Unmapped pixels: count={np.sum(unmapped_pixels)}, values={np.unique(image[unmapped_pixels])}")
        mapped[unmapped_pixels] = 0

    return mapped.astype(np.uint8)

def main():
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    label_mapping = None
    if opt.export_label_images:
        label_map = util.read_label_mapping(opt.label_map_file, label_from='id', label_to='nyu40id')

    scenes = [d for d in os.listdir(opt.scannet_path) if os.path.isdir(os.path.join(opt.scannet_path, d))]
    print('Found %d scenes' % len(scenes))
    for i in range(len(scenes)):
        sens_file = os.path.join(opt.scannet_path, scenes[i], scenes[i] + '.sens')
        label_path = os.path.join(opt.scannet_path, scenes[i], opt.label_type)
        if opt.export_label_images and not os.path.isdir(label_path):
            print_error('Error: using export_label_images option but label path %s does not exist' % label_path)

        export_sens = True

        output_color_path = os.path.join(opt.output_path, scenes[i], 'color')
        if not os.path.isdir(output_color_path):
            os.makedirs(output_color_path)
            export_sens = True
        output_depth_path = os.path.join(opt.output_path, scenes[i], 'depth')
        if not os.path.isdir(output_depth_path):
            os.makedirs(output_depth_path)
            export_sens = True
        output_pose_path = os.path.join(opt.output_path, scenes[i], 'pose')
        if not os.path.isdir(output_pose_path):
            os.makedirs(output_pose_path)
            export_sens = True
        output_label_path = os.path.join(opt.output_path, scenes[i], 'label')
        if opt.export_label_images and not os.path.isdir(output_label_path):
            os.makedirs(output_label_path)

        # read and export
        if export_sens:
            sys.stdout.write('\r[ %d | %d ] %s\tloading...' % ((i + 1), len(scenes), scenes[i]))
            sys.stdout.flush()
            sd = SensorData(sens_file)
            sys.stdout.write('\r[ %d | %d ] %s\texporting...' % ((i + 1), len(scenes), scenes[i]))
            sys.stdout.flush()
            sd.export_color_images(output_color_path, image_size=[opt.output_image_height, opt.output_image_width], frame_skip=opt.frame_skip)
            sd.export_depth_images(output_depth_path, image_size=[opt.output_image_height, opt.output_image_width], frame_skip=opt.frame_skip)
            sd.export_poses(output_pose_path, frame_skip=opt.frame_skip)

        if opt.export_label_images:
            sys.stdout.write('\r[ %d | %d ] %s\tmapping labels...' % ((i + 1), len(scenes), scenes[i]))
            label_file_length = len(next(os.walk(label_path))[2])  # dir is your directory path as string
            for f in range(0, label_file_length, opt.frame_skip):
                print("FILE", f)
                label_file = os.path.join(label_path, str(f) + '.png')
                image = np.array(imageio.imread(label_file))

                #print("BEFORE RESIZE MAPPING")
                mapped_image = map_label_image(image.astype(np.uint8), label_map)

                image = sktf.resize(image, [opt.output_image_height, opt.output_image_width], order=0, preserve_range=True, anti_aliasing=False)

                #print("AFTER RESIZE MAPPING")
                mapped_image = map_label_image(image.astype(np.uint8), label_map)

                imageio.imwrite(os.path.join(output_label_path, str(f) + '.png'), mapped_image)
    print('')


if __name__ == '__main__':
    main()

