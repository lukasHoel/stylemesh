from argparse import ArgumentParser

from model.model import TextureOptimizationStyleTransferPipeline
from data.scannet_single_scene_dataset import ScanNet_Single_Scene_DataModule
from data.matterport_single_scene_dataset import Matterport_Single_Scene_DataModule
from data.abstract_dataset import Abstract_DataModule
from model.losses.rgb_transform import pre
from model.losses.content_and_style_losses import ContentAndStyleLoss

from pytorch_lightning.trainer import Trainer

from scripts.textures.video_from_files import main as render_video
from scripts.eval.eval_image_folders import main as eval_image_folders
from scripts.scannet.render_mipmap_scannet import main as render_mipmap_scannet
from scripts.matterport.render_mipmap_matterport import main as render_mipmap_matterport

import os
from argparse import Namespace

from model.texture.utils import get_rgb_transform, get_uv_transform, get_label_transform

from os.path import join

from PIL import Image
from torchvision.transforms import Resize, ToTensor, Compose


def main(args):
    # create normal trainer (joint training)
    trainer = Trainer.from_argparse_args(args)
    log_dir = join(trainer.logger.save_dir, f"lightning_logs/version_{trainer.logger.version}")

    transform_rgb = Compose([
        get_rgb_transform(),
        pre()
    ])
    transform_label = get_label_transform()
    transform_uv = get_uv_transform()

    # construct splits manually from the three command args (easier than doing some command line magic)
    splits = [args.train_split, args.val_split]

    # create lightning DataModule from requested dataset
    if args.dataset == "scannet":
        dm = ScanNet_Single_Scene_DataModule(root_path=args.root_path,
                                             transform_rgb=transform_rgb,
                                             transform_label=transform_label,
                                             transform_uv=transform_uv,
                                             resize_size=args.resize_size,
                                             pyramid_levels=args.pyramid_levels,
                                             min_pyramid_depth=args.min_pyramid_depth,
                                             min_pyramid_height=args.min_pyramid_height,
                                             scene=args.scene,
                                             min_images=args.min_images,
                                             max_images=args.max_images,
                                             verbose=True,
                                             shuffle=args.shuffle,
                                             sampler_mode=args.sampler_mode,
                                             index_repeat=args.index_repeat,
                                             split=splits,
                                             split_mode=args.split_mode,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers)

    elif args.dataset == "matterport":
        dm = Matterport_Single_Scene_DataModule(root_path=args.root_path,
                                             transform_rgb=transform_rgb,
                                             transform_label=transform_label,
                                             transform_uv=transform_uv,
                                             resize_size=args.resize_size,
                                             pyramid_levels=args.pyramid_levels,
                                             min_pyramid_depth=args.min_pyramid_depth,
                                             min_pyramid_height=args.min_pyramid_height,
                                             scene=args.scene,
                                             min_images=args.min_images,
                                             max_images=args.max_images,
                                             verbose=True,
                                             shuffle=args.shuffle,
                                             sampler_mode=args.sampler_mode,
                                             index_repeat=args.index_repeat,
                                             split=splits,
                                             split_mode=args.split_mode,
                                             region_index=args.matterport_region_index,
                                             batch_size=args.batch_size,
                                             num_workers=args.num_workers)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # this sets up the lightning DataModule, i.e. creates train/val/test splits
    dm.prepare_data()
    dm.setup()

    # save all lightning parameters in this dict and add the train/val/test indices for future reproducibility
    selected_scene = dm.train_dataset.scene if hasattr(dm.train_dataset, "scene") else ""
    extra_args = {
        **vars(args),
        "indices": {
            "train": dm.train_indices,
            "val": dm.val_indices,
        },
        "selected_scene": selected_scene
    }

    # parse args.loss_weights into a dictionary.
    # format from commandline is for example: [['image', '1.0'], ['texture', '1.0']]
    # required format by the model is: {'image': 1.0, 'texture': 1.0}
    if args.loss_weights:
        args.loss_weights = {l[0]: float(l[1]) for l in args.loss_weights}

    # parse args.tex_reg_weights into a dictionary.
    # format from commandline is for example: [['0', '1.0'], ['1', '0.0']]
    # required format by the model is: [1.0, 2.0] where the i-th index is weight for layer i in the textures
    if args.tex_reg_weights:
        args.tex_reg_weights = {int(w[0]): float(w[1]) for w in args.tex_reg_weights}
        args.tex_reg_weights = [args.tex_reg_weights[i] for i in range(len(args.tex_reg_weights))]

    # get style image
    import PIL
    PIL.Image.MAX_IMAGE_PIXELS = 933120000
    style_image = Image.open(args.style_image_path)

    if style_image.size[0] > 2048 or style_image.size[1] > 2048:
        style_image = Resize(2048)(style_image)

    style_image = ToTensor()(style_image)
    style_image = pre()(style_image)

    # create model from all provided arguments
    model = TextureOptimizationStyleTransferPipeline(
        # texture dims
        W=args.texture_size[0], H=args.texture_size[1],

        # texture construction configuration
        hierarchical_texture=args.hierarchical,
        hierarchical_layers=args.hierarchical_layers,
        random_texture_init=args.random_texture_init,

        # style transfer configuration
        style_image=style_image,
        style_layers=args.style_layers,
        content_layers=args.content_layers,
        style_weights=args.style_weights,
        content_weights=args.content_weights,
        vgg_gatys_model_path=args.vgg_gatys_model_path,
        use_angle_weight=not args.no_angle_weight,
        use_depth_scaling=not args.no_depth_scaling,
        angle_threshold=args.angle_threshold,
        style_pyramid_mode=args.style_pyramid_mode,
        gram_mode=args.gram_mode,

        # optimization hyperparameters
        learning_rate=args.learning_rate,
        tex_reg_weights=args.tex_reg_weights,
        decay_gamma=args.decay_gamma,
        decay_step_size=args.decay_step_size,
        loss_weights=args.loss_weights,
        extra_args=extra_args,

        # logging parameters
        log_images_nth=args.log_images_nth,
        save_texture=args.save_texture,
        texture_dir=log_dir)

    # start the training loop (creates train/val logs; save texture every epoch if specified;
    trainer.fit(model, dm)

    if selected_scene != '':
        textures = [t for t in os.listdir(log_dir) if "texture.jpg" in t]
        textures = [t for t in textures if "layer" not in t]
        textures = sorted(textures, key=lambda x: int(x.split("_")[1]))
        textures = [join(log_dir, t) for t in textures]
        if len(textures) > 0:
            if args.dataset == 'scannet':
                truncated_scene = selected_scene.split("_")
                truncated_scene = truncated_scene[:2] # scene0673_00_orthogonal -> scene0673_00. they share same mesh!
                truncated_scene.insert(1, "_")
                truncated_scene = "".join(truncated_scene)
                mesh_path = join(os.path.dirname(dm.train_dataset.root_path), "scans", truncated_scene)
                mesh = [o for o in os.listdir(mesh_path) if 'uvs' in o and 'ply' in o]
                mesh = join(mesh_path, mesh[0])
                render_args = {
                    'tex': textures[-1],
                    'out': None,
                    'h': 480,
                    'w': 640,
                    'intrinsics': dm.train_dataset.intrinsics_file,
                    'poses': os.path.dirname(dm.train_dataset.extrinsics[0]),
                    'mesh': mesh,
                    'renderer': args.renderer_mipmap
                }

                render_args = Namespace(**render_args)
                render_mipmap_scannet(render_args)

            elif args.dataset == 'matterport':
                render_args = {
                    'tex': textures[-1],
                    'out': None,
                    'h': 480,
                    'w': -1,
                    'data_root': dm.root_path,
                    'scene': dm.selected_scene,
                    'region_index': args.matterport_region_index,
                    'renderer': args.renderer_mipmap
                }

                render_args = Namespace(**render_args)
                render_mipmap_matterport(render_args)
            else:
                raise ValueError('cannot render mipmap for dataset: ', args.dataset)

            render_args.imgs_dir = render_args.out  # render_mipmap_* fills this internally
            render_args.out_dir = None
            render_video(render_args)

            eval_args = {
                "rgb": os.path.dirname(dm.train_dataset.rgb_images[0]),
                "pose": os.path.dirname(dm.train_dataset.extrinsics[0]),
                "depth": os.path.dirname(dm.train_dataset.depth_images[0]),
                "intrinsics": dm.train_dataset.intrinsics_file,
                "styled": render_args.out,
                "vgg_model_path": args.vgg_gatys_model_path,
                "style_image": args.style_image_path,
                "image_size": 256,
                "out_dir": render_args.out,
                "random_seed": 42,
                "debug": False,
                "pair_threshold": 20,
                "pair_threshold_short": 1,
                "pair_threshold_long": 10,
                "only_reprojection": False,
            }
            eval_args = Namespace(**eval_args)
            eval_image_folders(eval_args)


if __name__ == '__main__':
    parser = ArgumentParser()

    # add all flags from lightnings Trainer (i.e. --gpus)
    parser = Trainer.add_argparse_args(parser)

    # add all custom flags
    parser.add_argument('--root_path', default="/path/to/datasets/scannet")
    parser.add_argument('--dataset', default="scannet", choices=["icl", "scannet", "vase", "3dfuture", "matterport"])
    parser.add_argument('--matterport_region_index', default=0, type=int)
    parser.add_argument('--train_split', default=0.8, type=float)
    parser.add_argument('--val_split', default=0.2, type=float)
    parser.add_argument('--split_mode', default="sequential", type=str, choices=Abstract_DataModule.split_modes)
    parser.add_argument('--scene', default="")
    parser.add_argument('--max_images', default=-1, type=int)
    parser.add_argument('--min_images', default=1000, type=int)
    parser.add_argument('--resize_size', default=256, type=int)
    parser.add_argument('--texture_size', default="512,512", type=lambda s: [int(f) for f in s.split(",")], dest='texture_size')
    parser.add_argument('--hierarchical', default=False, action="store_true")
    parser.add_argument('--hierarchical_layers', default=4, type=int)
    parser.add_argument('--random_texture_init', default=False, action="store_true")
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--learning_rate', default=1, type=float)
    parser.add_argument("--loss_weight", action='append', type=lambda kv: kv.split("="), dest='loss_weights')
    parser.add_argument("--tex_reg_weight", action='append', type=lambda kv: kv.split("="), dest='tex_reg_weights')
    parser.add_argument('--decay_gamma', default=0.1, type=float)
    parser.add_argument('--decay_step_size', default=30, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--log_images_nth', default=-1, type=int)
    parser.add_argument('--save_texture', default=False, action="store_true")
    parser.add_argument('--shuffle', default=False, action="store_true")
    parser.add_argument('--sampler_mode', default="repeat", type=str, choices=Abstract_DataModule.sampler_modes)
    parser.add_argument('--index_repeat', default=1, type=int)

    # add all style transfer flags
    parser.add_argument('--vgg_gatys_model_path', default="/path/to/models/vgg_conv.pth", type=str)
    parser.add_argument('--style_image_path', required=True, default="/path/to/datasets/styles/3style/14-2.jpg", type=str)
    parser.add_argument('--style_layers', type=lambda s: [f for f in s.split(",")], dest='style_layers', default=ContentAndStyleLoss.style_layers)
    parser.add_argument('--content_layers', type=lambda s: [f for f in s.split(",")], dest='content_layers', default=ContentAndStyleLoss.content_layers)
    parser.add_argument('--style_weights', type=lambda s: [float(f) for f in s.split(",")], dest='style_weights', default=ContentAndStyleLoss.style_weights)
    parser.add_argument('--content_weights', type=lambda s: [float(f) for f in s.split(",")], dest='content_weights', default=ContentAndStyleLoss.content_weights)
    parser.add_argument('--no_angle_weight', default=False, action="store_true")
    parser.add_argument('--no_depth_scaling', default=False, action="store_true")
    parser.add_argument('--angle_threshold', default=60.0, required=False, type=float)
    parser.add_argument('--pyramid_levels', default=8, required=False, type=int)
    parser.add_argument('--min_pyramid_depth', default=0.25, required=False, type=float)
    parser.add_argument('--min_pyramid_height', default=32, required=False, type=int)
    parser.add_argument('--style_pyramid_mode', default='single', required=False, choices=ContentAndStyleLoss.style_pyramid_modes)
    parser.add_argument('--gram_mode', default='current', required=False, choices=ContentAndStyleLoss.gram_modes)

    parser.add_argument('--renderer_mipmap', default=None, required=False, type=str)

    # parse arguments given from command line (implicitly takes the args from main...)
    args = parser.parse_args()

    # run program with args
    main(args)
