import pytorch_lightning as pl

from model.texture.texture import NeuralTexture, HierarchicalNeuralTexture, to_image
from model.losses.rgb_transform import post

import torch

from model.texture.utils import from_grid
from model.losses.content_and_style_losses import ContentAndStyleLoss

from torchvision.transforms import ToTensor
import torch.nn.functional as F
from torchvision.utils import make_grid


class TextureOptimizationStyleTransferPipeline(pl.LightningModule):
    states = ["train", "val"]
    loss_types = [
        "tex_reg",  # regularization on texture weights
        "content", "style",  # content/style loss between prediction and image/style_image
        'total'
    ]
    default_loss_weights = {l: 0.0 for l in loss_types}

    def __init__(self,
                 # texture dims
                 W, H,

                 # texture construction configuration
                 hierarchical_texture=True,
                 hierarchical_layers=4,
                 random_texture_init=False,

                 # style transfer configuration
                 style_image=None,
                 style_layers=ContentAndStyleLoss.style_layers,
                 content_layers=ContentAndStyleLoss.content_layers,
                 style_weights=ContentAndStyleLoss.style_weights,
                 content_weights=ContentAndStyleLoss.content_weights,
                 vgg_gatys_model_path=None,
                 use_angle_weight=True,
                 use_depth_scaling=True,
                 style_pyramid_mode='single',
                 gram_mode='current',
                 angle_threshold=60,

                 # logging parameters
                 log_images_nth=-1,
                 save_texture=True,
                 texture_dir="",
                 texture_prefix="",

                 # optimization hyperparameters
                 learning_rate=1e-3,
                 decay_gamma=0.1,
                 decay_step_size=30,
                 loss_weights=default_loss_weights,
                 tex_reg_weights=None,
                 extra_args={}  # any additional hparams that should be logged but are not needed further by this model
                 ):
        super().__init__()

        # --------------------
        # LOGGING HPARAMS PREPARATION
        # --------------------
        # will include all constructor params in the hparam dict in tensorboard logger
        # we have to make this workaround to solve this issue because the dicts contain non-string values:
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/4496
        orig_style_image = style_image
        style_image = None
        self.save_hyperparameters()
        style_image = orig_style_image

        # --------------------
        # TEXTURE PREPARATION
        # --------------------
        self.hierarchical_texture = hierarchical_texture
        self.hierarchical_layers = hierarchical_layers
        self.C = 3
        if hierarchical_texture:
            self.texture = HierarchicalNeuralTexture(W, H, self.C, hierarchical_layers, random_texture_init)
        else:
            self.texture = NeuralTexture(W, H, self.C, random_texture_init)

        self.tex_reg_weights = tex_reg_weights
        if hierarchical_texture and not tex_reg_weights:
            self.tex_reg_weights = [pow(2, hierarchical_layers - i - 1) for i in range(hierarchical_layers)]
            self.tex_reg_weights[-1] = 0
            print(f"No tex_reg_weights specified. Setting them to {self.tex_reg_weights}")
        if hierarchical_texture and hierarchical_layers != len(self.tex_reg_weights):
            raise ValueError(
                f"Have {hierarchical_layers} texture layers, but only {len(self.tex_reg_weights)} weights specified")

        # --------------------
        # LOSS FUNCTIONS PREPARATION
        # --------------------
        self.loss_history = {loss: {k: [] for k in TextureOptimizationStyleTransferPipeline.states} for loss in
                             TextureOptimizationStyleTransferPipeline.loss_types}
        self.loss_weights = loss_weights if loss_weights else {}
        for loss in self.loss_history.keys():
            if loss not in self.loss_weights:
                self.loss_weights[loss] = TextureOptimizationStyleTransferPipeline.default_loss_weights[loss]
                print(f"No weight specified for the '{loss}' loss. Setting it to {self.loss_weights[loss]}")

        self.vgg_gatys_model_path = vgg_gatys_model_path
        self.vgg_loss = ContentAndStyleLoss(vgg_gatys_model_path,
                                             style_layers, content_layers,
                                             style_weights, content_weights,
                                             angle_threshold=angle_threshold,
                                             style_pyramid_mode=style_pyramid_mode,
                                             gram_mode=gram_mode)

        # --------------------
        # OTHER PREPARATION
        # --------------------
        self.style_image = style_image
        self.orig_style_image = style_image.clone()
        self.angle_threshold = angle_threshold
        self.style_pyramid_mode = style_pyramid_mode
        self.gram_mode = gram_mode

        self.use_angle_weight = use_angle_weight
        self.use_depth_scaling = use_depth_scaling

        self.learning_rate = learning_rate
        self.decay_gamma = decay_gamma
        self.decay_step_size = decay_step_size

        self.log_images_nth = log_images_nth
        self.save_texture = save_texture
        self.texture_prefix = texture_prefix
        self.texture_dir = texture_dir

        # lightning offers no way to get the total batch sizes for train/val so we track them manually here
        self.batches_per_epoch = {k: 0 for k in TextureOptimizationStyleTransferPipeline.states}

        # currently (lightning 1.2.6), on_epoch_end gets called AFTER train end and BEFORE val end.
        # joint processing that should happen after both TRAIN and VAL have ended is only possible by keeping track
        # of these flags myself ... :(
        self.train_epoch_end = False
        self.val_epoch_end = False

    def forward(self, x):
        # input is: (rgb, extrinsics, intrinsics, depth,
        #               depth_level, rounded_depth_level, other_depth_level, depth_level_interpolation_weight,
        #               idx_item, uvs, mask, angle_guidance, angle_degrees)
        image, extrinsics, _, _, _, _, _, _, _, uv_map, _, _, _ = x

        if self.style_image.shape != image.shape:
            if len(self.style_image.shape) != 4:
                # add batch dimension
                self.style_image = self.style_image.repeat(image.shape[0], 1, 1, 1).type_as(image)
                self.vgg_loss.set_style_image(self.style_image)

        # TEXTURE SAMPLING
        pred_pyramid = []
        for v in uv_map:
            screen_space_texture = self.texture(v)
            pred_pyramid.append(screen_space_texture)

        return pred_pyramid

    def tex_reg_loss(self):
        if self.hierarchical_texture:
            loss = self.texture.regularizer(self.tex_reg_weights)
        else:
            # move to cuda device regardless, not just return "0.0"
            loss = torch.zeros(1)
            loss = loss.type_as(self.texture.data)

        return loss

    def update_batch_count(self, batch_idx, state):
        # on-the-fly construct the batch_length by checking if this batch_idx is the largest one
        # since this is fully determined after the first epoch (i.e. epoch 0) the log_idx above will always be correct
        self.batches_per_epoch[state] = max(self.batches_per_epoch[state], batch_idx + 1)

    def forward_with_loss(self, batch, batch_idx, state):
        # the tensorboard index for batch-level logging (i.e. global step w.r.t. all batches in all epochs)
        log_idx = batch_idx + self.current_epoch * self.batches_per_epoch[state]
        self.update_batch_count(batch_idx, state)

        input_rgb_image, _, _, depth, depth_level, rounded_depth_level, other_depth_level, depth_level_interpolation_weight, _, uv_map, mask, angle_guidance, angle_degrees = batch
        pred_pyramid = self.forward(batch)
        out_pyramid_index, pred_image = find_pyramid_size(pred_pyramid, input_rgb_image)

        # only apply loss calculations on areas where a valid UV mapping exists
        mask3 = torch.stack([mask, mask, mask], dim=1)  # from (B x H x W) to (B x 3 x H x W)
        mask = mask.unsqueeze(1).float()  # from (B x H x W) to (B x 1 x H x W) float tensor
        def masked(img):
            return torch.where(mask3.bool(), img, torch.zeros_like(img))

        losses = {}

        if self.use_angle_weight:
            for i, p in enumerate(pred_pyramid):
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    def apply(grad):
                        angle_guidance_i = F.interpolate(angle_guidance, grad.shape[2:], mode='bilinear')
                        grad = grad * angle_guidance_i
                        return grad
                    p.register_hook(apply)

        def erode(x, kernel_size=3):
            k = torch.ones(1,1,kernel_size,kernel_size).type_as(x)
            erosion_mask = torch.nn.functional.conv2d(x, k, padding=(1, 1)) / kernel_size**2
            erosion_mask = torch.clamp(erosion_mask, 0, 1)
            return x * (erosion_mask == 1)

        def mask_depth(pyramid_index, pyramid_image):
            m1 = rounded_depth_level == pyramid_index
            m2 = other_depth_level == pyramid_index
            m = (m1 + m2).float()
            m = m * mask

            # filter with 3x3 kernel to remove artifacts because of noisy depth
            m = erode(m)

            m = F.interpolate(m, pyramid_image.shape[2:], mode='nearest')

            return (m > 0).float()

        def mask_interpolation_weight(pyramid_index, pyramid_image):
            # mask each depth_level map with uv mask and selected pyramid_index
            m1 = (rounded_depth_level == pyramid_index) * mask
            m2 = (other_depth_level == pyramid_index) * mask

            # filter with 3x3 kernel to remove artifacts because of noisy depth
            m1 = erode(m1)
            m2 = erode(m2)

            # add interpolation weight
            m1 = m1 * depth_level_interpolation_weight
            m2 = m2 * (1 - depth_level_interpolation_weight)

            # combine
            m = m1 + m2
            m = F.interpolate(m, pyramid_image.shape[2:], mode='nearest')
            return m

        if self.use_depth_scaling:
            pyramid_masks = [mask_depth(i, p) for i, p in enumerate(pred_pyramid)]
            pyramid_interpolation_weights = [mask_interpolation_weight(i, p) for i, p in enumerate(pred_pyramid)]

            for i, p in enumerate(pred_pyramid):
                if isinstance(p, torch.Tensor) and p.requires_grad:
                    def apply(grad):
                        _, interpolation_weight = find_pyramid_size(pyramid_interpolation_weights, grad)
                        grad = grad * interpolation_weight
                        return grad
                    p.register_hook(apply)
        else:
            pyramid_masks = [(F.interpolate(torch.zeros_like(mask), p.shape[2:], mode='nearest') > 0).float() for p in pred_pyramid]
            pyramid_masks[-1] = (F.interpolate(mask, pred_pyramid[-1].shape[2:], mode='nearest') > 0).float()

        pred_pyramid = [p for p, m in zip(pred_pyramid, pyramid_masks) if torch.sum(m) > 0]
        pyramid_masks = [m for m in pyramid_masks if torch.sum(m) > 0]

        style_loss, content_loss, pyramid = self.vgg_loss(pred_pyramid, input_rgb_image, pyramid_masks, angle_unnormalized=angle_degrees)

        losses["content"] = self.loss_weights["content"] * content_loss
        losses["style"] = self.loss_weights["style"] * style_loss

        if self.loss_weights["tex_reg"] > 0:
            losses["tex_reg"] = self.loss_weights["tex_reg"] * self.tex_reg_loss()
        else:
            losses["tex_reg"] = torch.zeros_like(losses["content"])

            # calculate total loss for backprop
        losses["total"] = sum(losses.values())

        res = {
            "loss": losses["total"],  # required for backprop
        }

        # add loss logging
        for loss_type, loss in losses.items():
            if loss_type in self.loss_history.keys():
                self.loss_history[loss_type][state].append(
                    loss.detach().cpu())  # for calculating the per-epoch mean loss later
                self.logger.experiment.add_scalar(f"Batch/Loss/{state}/{loss_type}", loss.detach().cpu(),
                                                  log_idx)  # log batch-loss to tensorboard

        # add images + textures logging
        if self.log_images_nth != -1 and batch_idx % self.log_images_nth == 0:
            # contains one image per batch
            images = []

            # add all images and their batch-wise transform function that should be part of the visualization
            image_parts = [
                {"img": masked(pred_image), "transform": to_tensor_image},
                {"img": input_rgb_image, "transform": to_tensor_image},
                {"img": self.style_image, "transform": lambda x: F.interpolate(to_tensor_image(x).unsqueeze(0), (
                pred_image.shape[2], pred_image.shape[3]), mode='bilinear').squeeze(0)},
                {"img": uv_map[out_pyramid_index], "transform": lambda x: from_grid(x).detach().cpu()},
                {"img": depth, "transform": lambda x: x.repeat(3, 1, 1).detach().cpu() / 10.0},
                {"img": mask3, "transform": lambda x: x.detach().cpu()},
                {"img": angle_guidance, "transform": lambda x: x.repeat(3, 1, 1).detach().cpu()},
                {"img": angle_degrees, "transform": lambda x: x.repeat(3, 1, 1).detach().cpu() / 90.0},
                {"img": depth_level, "transform": lambda x: x.repeat(3, 1, 1).detach().cpu() / (len(uv_map) - 1)},
            ]

            # construct one image per batch
            batch_size = pred_image.shape[0]
            for b in range(batch_size):
                batch_images = []
                for part in image_parts:
                    batch_images.append(part["transform"](part["img"][b]))
                batch_images = make_grid(batch_images, nrow=len(batch_images))
                images.append(batch_images)

            # add final image to logger
            self.logger.experiment.add_image(f"Images/{state}", make_grid(images, padding=5, nrow=1), log_idx)

            # add texture images per layer and per 3 channels
            layers = 1 if isinstance(self.texture, NeuralTexture) else len(self.texture.layers)
            for l in range(layers):
                channel_images = []
                for c in range(0, self.C, 3):
                    if isinstance(self.texture, NeuralTexture):
                        channel_images.append(to_tensor_image(self.texture.get_image(), c))
                    else:
                        channel_images.append(to_tensor_image(self.texture.layers[l].get_image(), c))
                channel_images = make_grid(channel_images, nrow=len(channel_images))  # all channels in one row
                self.logger.experiment.add_image(f"Texture/{state}/Layer-{l}", channel_images, log_idx)

        return res

    def reset_loss_count(self, state):
        # reset losses
        for loss_type in self.loss_history.keys():
            self.loss_history[loss_type][state].clear()

    def compute_mean_loss(self, state):
        # calculate per-epoch mean losses
        for loss_type, loss in self.loss_history.items():
            if isinstance(state, list):
                # show all states in one plot (i.e. train/val in one plot)
                mean_loss = {s: torch.stack(loss[s]).mean() for s in state}
                self.logger.experiment.add_scalars(f"Loss/{'-'.join(state)}/{loss_type}", mean_loss, self.current_epoch)
            else:
                # show single state in one plot (i.e. test in one plot)
                mean_loss = torch.stack(loss[state]).mean()
                self.logger.experiment.add_scalar(f"Loss/{state}/{loss_type}", mean_loss, self.current_epoch)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self.forward_with_loss(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.forward_with_loss(batch, batch_idx, "val")

    def on_train_epoch_start(self) -> None:
        self.train_epoch_end = False
        self.val_epoch_end = False
        self.reset_loss_count("train")

    def on_validation_epoch_start(self) -> None:
        self.val_epoch_end = False
        self.reset_loss_count("val")

    def on_train_epoch_end(self) -> None:
        self.train_epoch_end = True

    def on_validation_epoch_end(self) -> None:
        self.val_epoch_end = True

    def on_epoch_end(self) -> None:
        if not self.train_epoch_end or not self.val_epoch_end:
            # currently (lightning 1.2.6), on_epoch_end gets called AFTER train end and BEFORE val end.
            # joint processing that should happen after both TRAIN and VAL have ended is only possible by keeping track
            # of these flags myself ... :(
            return

        self.compute_mean_loss("train")  # train in one plot
        self.compute_mean_loss("val")  # val in one plot
        self.compute_mean_loss(["train", "val"])  # train and val in one plot

        if self.save_texture:
            with torch.no_grad():
                self.texture.save_layers(self.texture_dir,
                                    f"{self.texture_prefix}epoch_{self.current_epoch}",
                                    normalize_transform=post())
                self.texture.save_image(self.texture_dir,
                                   f"{self.texture_prefix}epoch_{self.current_epoch}_",
                                   normalize_transform=post())

    def configure_optimizers(self):
        param_list = []

        # add texture data
        param_list.append({'params': self.texture.parameters(), 'weight_decay': 0.0,
                           'lr': self.learning_rate})

        # Adam for automatic optimization with PL
        optimizer = torch.optim.Adam(params=param_list, lr=self.learning_rate, weight_decay=0.0)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    gamma=self.decay_gamma,
                                                    step_size=self.decay_step_size)

        return [optimizer], [scheduler]


def to_tensor_image(t, idx=0):
    if len(t.shape) == 4:
        t_out = []
        for b in range(t.shape[0]):
            t_out.append(to_tensor_image(t[b], idx))
        return torch.stack(t_out, dim=0)
    else:
        t = to_image(t, idx, normalize_transform=post())
        t = ToTensor()(t)
        return t


def find_pyramid_size(pyramid, sample):
    for i, p in enumerate(pyramid):
        if p.shape[2] == sample.shape[2]:
            return i, p
    return 0, p[0]
