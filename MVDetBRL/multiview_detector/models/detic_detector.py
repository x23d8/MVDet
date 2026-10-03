"""MVDet heads on features from an official, frozen Detic detector."""

import inspect
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from multiview_detector.models.persp_trans_detector import PerspTransDetector


DETIC_CONFIG = 'configs/Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.yaml'
DETIC_WEIGHTS = ('https://dl.fbaipublicfiles.com/detic/'
                 'Detic_LCOCOI21k_CLIP_R5021k_640b32_4x_ft4x_max-size.pth')


class DeticDetector(PerspTransDetector):
    def __init__(self, dataset, root, weights=None, feature='p2', input_width=640):
        if input_width <= 0:
            raise ValueError('detic_input_width must be positive')
        if not root:
            raise ValueError('--arch detic requires --detic_root pointing to the official checkout')
        root = Path(root).expanduser().resolve()
        config = root / DETIC_CONFIG
        if not config.is_file():
            raise FileNotFoundError(f'Detic config missing: {config}; clone the official Detic repository with submodules')
        super().__init__(dataset, arch='resnet18')
        del self.base_pt1, self.base_pt2
        sys.path.insert(0, str(root))
        sys.path.insert(0, str(root / 'third_party' / 'CenterNet2'))
        try:
            from detectron2.config import get_cfg
            from detectron2.modeling import build_model
            from detectron2.checkpoint import DetectionCheckpointer
            from centernet.config import add_centernet_config
            from detic.config import add_detic_config
            import centernet.modeling  # noqa: F401; register CenterNet2 components
            import detic.modeling  # noqa: F401; register Detic components
        except ImportError as exc:
            raise ImportError('Detic needs detectron2 and the official Detic repository with CenterNet2 submodule') from exc
        from timm.models.helpers import build_model_with_cfg
        from timm.models.resnet import default_cfgs as resnet_cfgs
        if (not isinstance(resnet_cfgs.get('resnet50'), dict) or
                'default_cfg' not in inspect.signature(build_model_with_cfg).parameters):
            raise RuntimeError('This Detic checkout requires timm==0.5.4; install it with '
                               '`python -m pip install --no-deps timm==0.5.4` and restart the Python process')
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)
        cfg.merge_from_file(str(config))
        cfg.MODEL.DEVICE = 'cuda:0'
        # The classifier is unused; avoid loading a dataset-relative CLIP file at construction.
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = 'rand'
        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = str(root / cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH)
        cfg.freeze()
        detector = build_model(cfg).eval()
        DetectionCheckpointer(detector).load(str(weights or DETIC_WEIGHTS))
        if feature not in detector.backbone.output_shape():
            raise ValueError(f'Unknown Detic FPN feature {feature!r}; choose from {list(detector.backbone.output_shape())}')
        channels = detector.backbone.output_shape()[feature].channels
        self.backbone = detector.backbone.requires_grad_(False).eval()
        self.adapter = nn.Conv2d(channels, 512, 1).to('cuda:0')
        self.feature = feature
        self.input_width = input_width
        self.input_height = max(32, round(dataset.img_shape[0] / dataset.img_shape[1] * input_width / 32) * 32)
        self.input_format = cfg.INPUT.FORMAT
        self.register_buffer('pixel_mean', torch.tensor(cfg.MODEL.PIXEL_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('pixel_std', torch.tensor(cfg.MODEL.PIXEL_STD).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1), persistent=False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _encode_images(self, imgs):
        batch, views = imgs.shape[:2]
        rgb = imgs.flatten(0, 1).to('cuda:0')
        rgb = (rgb * self.std.to(rgb.device) + self.mean.to(rgb.device)).clamp(0, 1)
        rgb = F.interpolate(rgb, (self.input_height, self.input_width), mode='bilinear', align_corners=False)
        # Detectron2 Detic configs expect BGR in 0..255, then subtract PIXEL_MEAN.
        detector_input = (rgb if self.input_format == 'RGB' else rgb[:, [2, 1, 0]]) * 255.0
        with torch.no_grad():
            features = self.backbone((detector_input - self.pixel_mean.to(detector_input.device)) /
                                     self.pixel_std.to(detector_input.device))
        encoded = self.adapter(features[self.feature].float())
        return list(encoded.view(batch, views, 512, *encoded.shape[-2:]).unbind(1))
