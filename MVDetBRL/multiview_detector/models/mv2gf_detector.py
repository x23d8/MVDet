"""DA3 feature fusion and pointmap aggregation adapted to MVDet/BRL labels.

DA3 outputs are precomputed by generate_da3_cache.py. This shares the TGF/FPA
principles of MV2GF but retains MVDet's dense heatmap heads and BRL loss.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from multiview_detector.models.persp_trans_detector import PerspTransDetector


LAYERS = (19, 27, 33, 39)


class MV2GFDetector(PerspTransDetector):
    requires_frames = True

    def __init__(self, dataset, cache_dir, voxel_height=4):
        super().__init__(dataset, arch='resnet18')
        if voxel_height <= 0:
            raise ValueError('voxel_height must be positive')
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        if not self.cache_dir.is_dir():
            raise FileNotFoundError(f'DA3 cache directory missing: {self.cache_dir}')
        self.voxel_height = voxel_height
        self.da3_adapters = nn.ModuleList([nn.LazyConv2d(64, 1) for _ in LAYERS]).to('cuda:0')
        self.fusion = nn.Sequential(nn.Conv2d(512 + 64 * len(LAYERS), 128, 3, padding=1),
                                    nn.ReLU()).to('cuda:0')
        self.img_classifier = nn.Sequential(nn.Conv2d(128, 64, 1), nn.ReLU(),
                                            nn.Conv2d(64, 2, 1, bias=False)).to('cuda:0')
        self.map_classifier = nn.Sequential(nn.Conv2d(128 * voxel_height + 2, 512, 3, padding=1),
                                             nn.ReLU(), nn.Conv2d(512, 512, 3, padding=2, dilation=2),
                                             nn.ReLU(), nn.Conv2d(512, 1, 3, padding=4, dilation=4,
                                                                  bias=False)).to('cuda:0')
        # Both datasets use calibrated world coordinates, in centimetres for Wildtrack.
        unit = 0.01 if dataset.base.__name__ == 'Wildtrack' else 1.0
        worldgrid_to_m = dataset.base.worldgrid2worldcoord_mat.astype(np.float64).copy()
        worldgrid_to_m[:2] *= unit
        world_to_grid = np.linalg.inv(worldgrid_to_m)
        self.register_buffer('world_to_grid', torch.from_numpy(world_to_grid).float(), persistent=False)
        self.grid_reduce = dataset.grid_reduce

    def _load_cache(self, frames, device):
        arrays = []
        for frame in frames.tolist():
            path = self.cache_dir / f'{int(frame):04d}.npz'
            if not path.is_file():
                raise FileNotFoundError(f'Missing DA3 output {path}; run generate_da3_cache.py first')
            with np.load(path, allow_pickle=False) as data:
                names = ('point',) + tuple(f'feat_layer_{layer}' for layer in LAYERS)
                missing = [name for name in names if name not in data]
                if missing:
                    raise ValueError(f'{path} lacks {missing}')
                arrays.append(tuple(torch.from_numpy(data[name].copy()).to(device=device, dtype=torch.float32)
                                    for name in names))
        points = torch.stack([row[0] for row in arrays])
        features = [torch.stack([row[i + 1] for row in arrays]) for i in range(len(LAYERS))]
        return points, features

    def _aggregate(self, features, points):
        batch, views, channels, height, width = features.shape
        map_h, map_w = self.reducedgrid_shape
        # Keep point sampling tied to the feature pixels; nearest avoids mixing 3D surfaces.
        points = F.interpolate(points.permute(0, 1, 4, 2, 3).flatten(0, 1),
                               (height, width), mode='nearest').view(batch, views, 3, height, width)
        fused = []
        for b in range(batch):
            xyz = points[b].permute(0, 2, 3, 1).reshape(-1, 3)
            transform = self.world_to_grid.to(xyz.device)
            grid = xyz[:, :2] @ transform[:2, :2].T + transform[:2, 2]
            row = torch.floor(grid[:, 0] / self.grid_reduce).long()
            col = torch.floor(grid[:, 1] / self.grid_reduce).long()
            z = torch.floor(xyz[:, 2] / 0.5).long()
            valid = (torch.isfinite(xyz).all(1) & (row >= 0) & (row < map_h) &
                     (col >= 0) & (col < map_w) & (z >= 0) & (z < self.voxel_height))
            index = ((z[valid] * map_h + row[valid]) * map_w + col[valid])
            values = features[b].permute(0, 2, 3, 1).reshape(-1, channels)[valid].T
            voxels = features.new_full((channels, self.voxel_height * map_h * map_w), -torch.inf)
            if index.numel():
                voxels = voxels.scatter_reduce(1, index.unsqueeze(0).expand(channels, -1),
                                               values, reduce='amax', include_self=True)
            fused.append(torch.where(torch.isfinite(voxels), voxels, 0).view(
                channels * self.voxel_height, map_h, map_w))
        return torch.stack(fused)

    def forward(self, imgs, frames):
        batch, views = imgs.shape[:2]
        if views != self.num_cam:
            raise ValueError(f'Expected {self.num_cam} cameras, got {views}')
        device = next(self.fusion.parameters()).device
        points, da3_features = self._load_cache(frames, device)
        if points.shape[1] != views:
            raise ValueError(f'DA3 cache has {points.shape[1]} cameras; expected {views}')
        task_features = torch.stack(self._encode_images(imgs), dim=1)
        _, _, _, height, width = task_features.shape
        geom = []
        for adapter, feature in zip(self.da3_adapters, da3_features):
            if feature.shape[:2] != (batch, views):
                raise ValueError('DA3 features have wrong batch or camera dimension')
            x = adapter(feature.permute(0, 1, 4, 2, 3).flatten(0, 1))
            geom.append(F.interpolate(x, (height, width), mode='bilinear', align_corners=False))
        image_features = self.fusion(torch.cat([task_features.flatten(0, 1)] + geom, dim=1))
        imgs_result = [self.img_classifier(x) for x in image_features.view(
            batch, views, 128, height, width).unbind(1)]
        voxel_bev = self._aggregate(image_features.view(batch, views, 128, height, width), points)
        coord = self.coord_map.to(device).expand(batch, -1, -1, -1)
        map_result = self.map_classifier(torch.cat([voxel_bev, coord], dim=1))
        return map_result, imgs_result
