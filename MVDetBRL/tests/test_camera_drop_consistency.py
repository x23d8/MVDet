import torch

from multiview_detector.loss.camera_drop_consistency import CameraDropConsistencyLoss


def test_consistency_supervises_only_confident_positive_regions():
    full = torch.tensor([[[[0.0, 0.2, 0.8]]]])
    dropped = torch.zeros_like(full, requires_grad=True)
    loss = CameraDropConsistencyLoss(0.3)(dropped, full)
    loss.backward()
    assert dropped.grad[0, 0, 0, 0] == 0
    assert dropped.grad[0, 0, 0, 1] == 0
    assert dropped.grad[0, 0, 0, 2] < 0


def test_empty_confident_region_has_zero_gradient():
    full = torch.full((1, 1, 2, 2), 0.1)
    dropped = torch.randn_like(full, requires_grad=True)
    loss = CameraDropConsistencyLoss(0.3)(dropped, full)
    loss.backward()
    torch.testing.assert_close(dropped.grad, torch.zeros_like(dropped))
