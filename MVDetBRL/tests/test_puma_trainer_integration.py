import torch
from torch.utils.data import DataLoader, Dataset

from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.camera_drop_consistency import CameraDropConsistencyLoss
from multiview_detector.loss.pu_query_loss import PUQuerySetLoss
from multiview_detector.models.puma_hybrid_detector import PUMAHybridDetector
from multiview_detector.trainer import PerspectiveTrainer
from test_puma_dense_detector import _TinyDataset


class _OneFrame(Dataset):
    map_kernel = torch.ones(1, 1, 1, 1)
    img_kernel = torch.eye(2).reshape(2, 2, 1, 1)

    def __len__(self):
        return 1

    def __getitem__(self, index):
        images = torch.randn(2, 3, 64, 96)
        map_target = torch.zeros(1, 4, 6)
        map_target[0, 2, 3] = 1.0
        image_targets = [torch.zeros(2, 64, 96) for _ in range(2)]
        return images, map_target, image_targets, index


def test_hybrid_query_loss_runs_through_legacy_trainer(tmp_path):
    torch.manual_seed(11)
    model = PUMAHybridDetector(
        _TinyDataset(),
        feature_channels=8,
        fused_channels=8,
        heights_m=(0.0,),
        world_unit_to_m=1.0,
        num_queries=2,
        query_hidden_channels=8,
        query_layers=1,
    )
    trainer = PerspectiveTrainer(
        model,
        GaussianMSE(),
        str(tmp_path),
        denormalize=lambda value: value,
        query_criterion=PUQuerySetLoss(annotation_propensity=0.5),
        query_loss_weight=0.2,
        query_warmup_epochs=0,
        query_ramp_epochs=1,
        consistency_criterion=CameraDropConsistencyLoss(confidence_threshold=0.0),
        consistency_loss_weight=0.1,
        camera_drop_prob=0.5,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss, _ = trainer.train(
        1, DataLoader(_OneFrame(), batch_size=1), optimizer, log_interval=1
    )
    assert torch.isfinite(torch.tensor(loss))
    assert model.query_refiner.objectness.weight.grad is not None


def test_camera_dropout_always_keeps_and_drops_a_view_when_possible():
    for probability in (0.01, 0.5, 0.99):
        mask = PerspectiveTrainer._sample_camera_mask(
            batch=32, num_views=7, drop_probability=probability, device=torch.device("cpu")
        )
        assert mask.any(dim=1).all()
        assert (~mask).any(dim=1).all()
