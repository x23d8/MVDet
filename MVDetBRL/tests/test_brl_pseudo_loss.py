import torch

from multiview_detector.loss.brl_gaussian_mse import BRLGaussianMSE


def unit_kernel():
    return torch.ones((1, 1, 1, 1), dtype=torch.float32)


def test_pseudo_only_adds_separately_normalized_weak_positive():
    prediction = torch.zeros((1, 1, 2, 2), requires_grad=True)
    gt = torch.zeros_like(prediction)
    pseudo = torch.zeros_like(prediction)
    pseudo[:, :, 0, 0] = 1.0
    confidence = pseudo * 0.8
    criterion = BRLGaussianMSE(
        use_confuse=False,
        lambda_pseudo=0.1,
        pos_thr=0.1,
        pseudo_thr=0.1,
    )

    loss = criterion(prediction, gt, unit_kernel(), pseudo, confidence)

    assert torch.isclose(loss, torch.tensor(0.1))
    assert criterion.last_components["pseudo_pixels"].item() == 1
    loss.backward()
    assert prediction.grad[0, 0, 0, 0] < 0


def test_gt_has_priority_over_overlapping_pseudo():
    prediction = torch.zeros((1, 1, 2, 2))
    gt = torch.zeros_like(prediction)
    gt[:, :, 0, 0] = 1.0
    pseudo = gt.clone()
    criterion = BRLGaussianMSE(use_confuse=False, lambda_pseudo=0.5)

    criterion(prediction, gt, unit_kernel(), pseudo, pseudo)

    assert criterion.last_components["gt_pixels"].item() == 1
    assert criterion.last_components["pseudo_pixels"].item() == 0


def test_pseudo_is_removed_from_confuse_region():
    prediction = torch.full((1, 1, 2, 2), 0.8)
    gt = torch.zeros_like(prediction)
    pseudo = torch.zeros_like(prediction)
    pseudo[:, :, 0, 0] = 1.0
    criterion = BRLGaussianMSE(
        use_confuse=True,
        confuse_pred_thr=0.3,
        lambda_pseudo=0.1,
    )

    criterion(prediction, gt, unit_kernel(), pseudo, pseudo)

    assert criterion.last_components["pseudo_pixels"].item() == 1
    assert criterion.last_components["confuse_pixels"].item() == 3


def test_original_brl_path_still_covers_every_pixel():
    prediction = torch.tensor([[[[0.0, 0.8], [0.2, 0.6]]]])
    gt = torch.zeros_like(prediction)
    gt[:, :, 0, 0] = 1.0
    criterion = BRLGaussianMSE(use_confuse=True, confuse_pred_thr=0.3)

    criterion(prediction, gt, unit_kernel())

    counts = sum(
        criterion.last_components[name].item()
        for name in ("gt_pixels", "pseudo_pixels", "confuse_pixels", "negative_pixels")
    )
    assert counts == prediction.numel()
