from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.trainer import PerspectiveTrainer


class _TwoFrames(Dataset):
    base = SimpleNamespace(indexing="xy", __name__="Wildtrack")
    grid_reduce = 4
    map_kernel = torch.ones(1, 1, 1, 1)
    img_kernel = torch.eye(2).reshape(2, 2, 1, 1)

    def __len__(self):
        return 2

    def __getitem__(self, index):
        image = torch.full((1, 1, 2, 2), float(index))
        map_target = torch.zeros(1, 2, 2)
        map_target[0, index, index] = 1.0
        image_target = [torch.zeros(2, 2, 2)]
        return image, map_target, image_target, index


class _MarkerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, images):
        batch = images.shape[0]
        output = torch.zeros(batch, 1, 2, 2, device=images.device) + self.anchor
        marker = images[:, 0, 0, 0, 0].long()
        output[torch.arange(batch), 0, marker, marker] = 1.0 + self.anchor
        image_output = [torch.zeros(batch, 2, 2, 2, device=images.device)]
        return output, image_output


def test_batch_size_two_writes_predictions_for_both_frames(tmp_path):
    dataset = _TwoFrames()
    result_path = tmp_path / "result.txt"
    gt_path = tmp_path / "gt.txt"
    np.savetxt(gt_path, np.array([[0, 0, 0], [1, 4, 4]]), fmt="%d")
    trainer = PerspectiveTrainer(
        _MarkerModel(), GaussianMSE(), str(tmp_path), lambda value: value,
        cls_thres=0.4, nms_radius_grid=1.0,
    )
    trainer.test(
        DataLoader(dataset, batch_size=2),
        str(result_path), str(gt_path), visualize=False,
    )
    result = np.loadtxt(result_path).reshape(-1, 3)
    assert set(result[:, 0].astype(int)) == {0, 1}
