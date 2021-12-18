import torch
import torch.nn as nn
import torch.nn.functional as F


# vgg definition that conveniently let's you grab the outputs from any layer
class VGG(nn.Module):
    def __init__(self, pool='max', model_path=None, freeze=True):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        if model_path:
            self.load_state_dict(torch.load(model_path))

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return {k: out[k] for k in out_keys}


# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        Fl = input.view(b, c, h*w)
        G = torch.bmm(Fl, Fl.transpose(1,2))
        G.div_(h*w)
        return G


def image_pyramid(img, levels, reverse=False, minimum_size=256):
    def downsample_dimension(x, l):
        return int(x / 2 ** l)

    def sample_min_entry(h, w, img, minimum_size):
        if w > h:
            h_down = minimum_size
            w_down = int(w * h_down / h)
        else:
            w_down = minimum_size
            h_down = int(h * w_down / w)
        return F.interpolate(img, (h_down, w_down), mode='bilinear')

    h, w = img.shape[2:]
    pyramid = []
    min_entry = None
    min_index = len(levels)

    # create pyramid of image which is never smaller than min_entry
    # result: (orig_resolution_img, ..., min_entry, ..., min_entry)
    for i, level in enumerate(levels):
        if level == 0:
            # level 0 does not need to be interpolated, save directly
            pyramid.append(img)
        else:
            h_down = downsample_dimension(h, level)
            w_down = downsample_dimension(w, level)
            if h_down < minimum_size or w_down < minimum_size:
                if min_entry is None:
                    # only need to interpolate this once, save it for later usage in the loop
                    min_entry = sample_min_entry(h, w, img, minimum_size)
                    # set position of minimum entry in pyramid
                    min_index = i
                # add minimum entry as downsampled entry for this level
                pyramid.append(min_entry)
            else:
                # add downsampled entry for this level
                pyramid.append(F.interpolate(img, (h_down, w_down), mode='bilinear'))

    if reverse:
        # reverse pyramid up to first occurence of min_entry
        # result: (min_entry, ..., orig_resolution_img)
        reverse_pyramid = pyramid[:min_index+1]
        reverse_pyramid = reverse_pyramid[::-1]
        # fill remaining slots with orig_resolution_img
        # result: (min_entry, ..., orig_resolution_img, ... , orig_resolution_img)
        while len(reverse_pyramid) < len(pyramid):
            reverse_pyramid.append(img)
        pyramid = reverse_pyramid

    return pyramid


def masked_features(features, mask):
    features_cropped = features[:, :, mask.squeeze() > 0]
    features_cropped = features_cropped.unsqueeze(3)

    if features_cropped.shape[2] == 0:
        return torch.zeros_like(features).reshape(features.shape[0], features.shape[1], -1).unsqueeze(3)
    else:
        return features_cropped


def calculate_pyramid(pred_pyramid, content_encodings, pyramid_masks, angle_degrees, angle_threshold, vgg: VGG):
    last_level = len(pred_pyramid) - 1
    factors = []
    masks = []
    masks_passed_angle_filter = []
    masks_failed_angle_filter = []

    pred_pyramid_passed_angle_filter = []
    pred_pyramid_failed_angle_filter = []

    layers = set()
    content_pyramid = []

    for pyramid_index, pyramid_encodings in enumerate(pred_pyramid):
        mask = pyramid_masks[pyramid_index]
        passed_angle_filter = F.interpolate(angle_degrees, mask.shape[2:], mode='bilinear') < angle_threshold

        factors_i = {}
        masks_i = {}
        masks_i_passed_angle_filter = {}
        masks_i_failed_angle_filter = {}
        pred_pyramid_i_passed_angle_filter = {}
        pred_pyramid_i_failed_angle_filter = {}
        content_targets_i = {}
        for k, o in pyramid_encodings.items():
            with torch.no_grad():
                mask_i = F.interpolate(mask, o.shape[2:], mode='nearest')
                mask_i_passed_angle_filter = F.interpolate(mask * passed_angle_filter, o.shape[2:], mode='nearest')
                mask_i_failed_angle_filter = F.interpolate(mask * (~passed_angle_filter), o.shape[2:], mode='nearest')

                content_targets_i[k] = F.interpolate(content_encodings[k], o.shape[2:], mode='bilinear')
                content_targets_i[k] = masked_features(content_targets_i[k], mask_i)

                if pyramid_index == last_level:
                    layers.add(k)
                factors_i[k] = torch.mean(mask_i)
                masks_i[k] = mask_i
                masks_i_passed_angle_filter[k] = mask_i_passed_angle_filter
                masks_i_failed_angle_filter[k] = mask_i_failed_angle_filter

            # mask pred_pyramid and content_encodings
            pred_pyramid[pyramid_index][k] = masked_features(o, mask_i)
            pred_pyramid_i_passed_angle_filter[k] = masked_features(o, mask_i_passed_angle_filter)
            pred_pyramid_i_failed_angle_filter[k] = masked_features(o, mask_i_failed_angle_filter)

        factors.append(factors_i)
        masks.append(mask_i)
        masks_passed_angle_filter.append(masks_i_passed_angle_filter)
        masks_failed_angle_filter.append(masks_i_failed_angle_filter)
        pred_pyramid_passed_angle_filter.append(pred_pyramid_i_passed_angle_filter)
        pred_pyramid_failed_angle_filter.append(pred_pyramid_i_failed_angle_filter)
        content_pyramid.append(content_targets_i)

    # normalize all factors
    for k in layers:
        factors_k = [factors[i][k] for i in range(len(masks_passed_angle_filter))]
        factor_sum = sum(factors_k)
        for i in range(len(masks_passed_angle_filter)):
            factors[i][k] = factors[i][k] / factor_sum

    return {
        'p': pred_pyramid,
        'p_passed_angle_filter': pred_pyramid_passed_angle_filter,
        'p_failed_angle_filter': pred_pyramid_failed_angle_filter,
        'c': content_pyramid,
        'c_orig': content_encodings,
        'm': masks,
        'm_passed_angle_filter': masks_passed_angle_filter,
        'm_failed_angle_filter': masks_failed_angle_filter,
        'f': factors,
        'size': len(masks)
    }


class ContentAndStyleLoss(torch.nn.Module):
    # define layers, loss functions, weights and compute optimization targets
    style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    content_layers = ['r42']

    # these are good weights settings:
    style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
    content_weights = [1 for i in range(len(content_layers))]

    # how to use the style image pyramid in the loss
    style_pyramid_modes = [
        'single',  # uses smallest style image pyramid entry for the loss
        'multi'  # uses smallest style image pyramid entry for the loss and adds large style image pyramid entry depending on the vgg layer
    ]

    gram_modes = [
        'current',  # only uses gram matrix of current image to calculate MSE to style image's gram matrix
        'average'  # uses average of last N gram matrices to calculate MSE to style image's gram matrix.
    ]

    def __init__(self, model_path,
                 style_layers=style_layers, content_layers=content_layers,
                 style_weights=style_weights, content_weights=content_weights,
                 angle_threshold=60,
                 style_pyramid_mode='single',
                 gram_mode='current'):
        super(ContentAndStyleLoss, self).__init__()

        # create vgg model
        if not model_path:
            raise ValueError("No model_path provided")
        self.vgg = VGG(model_path=model_path)

        # define layers
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.layers = style_layers + content_layers

        # define weights
        self.style_weights = style_weights
        self.content_weights = content_weights

        self.style_pyramid_mode = style_pyramid_mode

        # define loss function
        self.mse = nn.MSELoss()

        self.gram_cache = {k: [] for k in self.style_layers}
        self.gram_mode = gram_mode

        self.style_targets = None
        self.angle_threshold = angle_threshold

    def set_style_image(self, style_image, num_levels=5):
        # all levels supported with this style image
        levels = [i for i in range(num_levels)]

        # compute style targets.
        # downscale style image according to levels -> encode this with vgg once -> normal gram matrix computation
        style_pyramid = image_pyramid(style_image, levels, reverse=True)

        print('Use style image pyramid of shapes:')
        for p in style_pyramid:
            print(p.shape)

        style_pyramid = [self.vgg(p, self.style_layers) for p in style_pyramid]
        self.style_targets = [{l: GramMatrix()(style_pyramid[k][i]).detach() for k, l in enumerate(levels)} for i in self.style_layers]

    def forward(self, pred, target_content, pyramid_masks, angle_unnormalized=None):

        # encode prediction pyramid once and apply guidance to every layer in every pyramid
        pred = [self.vgg(p, self.layers) for p in pred]

        # encode content target at original resolution only
        content_targets_orig = self.vgg(target_content, self.layers)

        pyramid = calculate_pyramid(pred, content_targets_orig, pyramid_masks, angle_unnormalized, self.angle_threshold, self.vgg)

        style_loss = torch.zeros(1, requires_grad=True).type_as(target_content)
        content_loss = torch.zeros(1, requires_grad=True).type_as(target_content)

        for p_index in range(pyramid['size']):

            # accumulate style loss over all layers
            for layer_index, layer in enumerate(self.style_layers):
                if self.style_pyramid_mode == 'single':
                    y = self.style_targets[layer_index][0]
                elif self.style_pyramid_mode == 'multi':
                    y = self.style_targets[layer_index][2]
                else:
                    raise ValueError(f"Unsupported style_pyramid_mode: {self.style_pyramid_mode}")

                if self.style_pyramid_mode == 'multi':
                    # when we have multiple resolutions of style image available, we will split stylization like this:
                    # 1) areas passed angle filter (good angles) are stylized normally (with low-res style image and high res style image)
                    y_hat = GramMatrix()(pyramid['p_passed_angle_filter'][p_index][layer])
                else:
                    y_hat = GramMatrix()(pyramid['p'][p_index][layer])

                if self.gram_mode == 'average':
                    self.gram_cache[layer] = self.gram_cache[layer][:9]
                    self.gram_cache[layer] = [g.detach() for g in self.gram_cache[layer]]
                    self.gram_cache[layer].insert(0, y_hat)
                    y_hat = torch.mean(torch.stack(self.gram_cache[layer]), dim=0)

                f = pyramid['f'][p_index][layer]
                l = self.style_weights[layer_index] * f * self.mse(y, y_hat)

                if self.style_pyramid_mode == 'multi':
                    # when we have multiple resolutions of style image available, we will split stylization like this:
                    # 2) areas failed angle filter (bad angles) are stylized only with larger style image (stroke patterns are not as stretched out)
                    y_hat_failed_angle_filter = GramMatrix()(pyramid['p_failed_angle_filter'][p_index][layer])
                    if torch.sum(pyramid['m_failed_angle_filter'][p_index][layer]) > 0:
                        l += self.style_weights[layer_index] * f * self.mse(y, y_hat_failed_angle_filter)

                    # 1) areas passed angle filter (good angles) are stylized normally (with low-res style image and high res style image)
                    y_smaller = self.style_targets[layer_index][0]
                    if layer_index > 2:
                        l += self.style_weights[layer_index] * f * self.mse(y_smaller, y_hat)

                style_loss += l

            # accumulate content loss over all layers
            for layer_index, layer in enumerate(self.content_layers):
                y = pyramid['c'][p_index][layer]
                y_hat = pyramid['p'][p_index][layer]
                f = pyramid['f'][p_index][layer]
                l = self.content_weights[layer_index] * f * self.mse(y, y_hat)
                content_loss += l

        return style_loss, content_loss, pyramid
