"""MVDet heads on frozen, official VGGT multi-view patch features."""

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F

from multiview_detector.models.persp_trans_detector import PerspTransDetector


class VGGTDetector(PerspTransDetector):
    def __init__(self, dataset, weights=None, input_width=518):
        if input_width <= 0 or input_width % 14:
            raise ValueError('vggt_input_width must be a positive multiple of 14')
        super().__init__(dataset, arch='resnet18')
        del self.base_pt1, self.base_pt2
        try:
            from huggingface_hub import hf_hub_download
            from vggt.models.vggt import VGGT
        except ImportError as exc:
            raise ImportError('VGGT encoder requires the official vggt package and huggingface_hub') from exc

        self.aggregator = VGGT(enable_camera=False, enable_point=False,
                               enable_depth=False, enable_track=False).aggregator
        checkpoint = weights or hf_hub_download(repo_id='facebook/VGGT-1B', filename='model.pt')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
        if 'state_dict' in state:
            state = state['state_dict']
        aggregator_state = {key[len('aggregator.'):]: value for key, value in state.items()
                            if key.startswith('aggregator.')}
        if not aggregator_state:
            raise ValueError('VGGT checkpoint contains no aggregator weights')
        self.aggregator.load_state_dict(aggregator_state, strict=True)
        del state, aggregator_state
        self.aggregator.requires_grad_(False)
        self.aggregator.to('cuda:0').eval()
        self.feature_adapter = nn.Conv2d(2048, 512, kernel_size=1).to('cuda:0')
        self.input_width = input_width
        self.input_height = max(14, round(dataset.img_shape[0] / dataset.img_shape[1]
                                          * input_width / 14) * 14)
        self.register_buffer('imagenet_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
                             persistent=False)
        self.register_buffer('imagenet_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
                             persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.aggregator.eval()
        return self

    def _encode_images(self, imgs):
        batch, views = imgs.shape[:2]
        imgs = imgs.to('cuda:0')
        # The dataset supplies ImageNet-normalized full frames. VGGT accepts [0, 1]
        # and applies ImageNet normalization inside its official aggregator.
        rgb = (imgs * self.imagenet_std.to(imgs.device) + self.imagenet_mean.to(imgs.device)).clamp(0, 1)
        rgb = F.interpolate(rgb.flatten(0, 1), (self.input_height, self.input_width),
                            mode='bilinear', align_corners=False).view(
                                batch, views, 3, self.input_height, self.input_width)
        use_amp = rgb.is_cuda
        dtype = (torch.bfloat16 if torch.cuda.get_device_capability(rgb.device)[0] >= 8
                 else torch.float16) if use_amp else torch.float32
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=dtype) if use_amp else nullcontext():
                tokens_by_layer, patch_start = self.aggregator(rgb)
                tokens = tokens_by_layer[-1][:, :, patch_start:, :]
        height, width = self.input_height // 14, self.input_width // 14
        feature = tokens.reshape(batch, views, height, width, 2048)
        feature = feature.permute(0, 1, 4, 2, 3).contiguous().float()
        feature = self.feature_adapter(feature.flatten(0, 1))
        return list(feature.view(batch, views, 512, height, width).unbind(dim=1))
