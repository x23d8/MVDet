import torch
import torch.nn.functional as F

from multiview_detector.loss.pu_gaussian_mse import PUGaussianMSE


def _example():
    prediction = torch.zeros(1, 1, 9, 9, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[0, 0, 4, 4] = 1.0
    kernel = torch.tensor(
        [[[[0.25, 0.5, 0.25], [0.5, 1.0, 0.5], [0.25, 0.5, 0.25]]]]
    )
    return prediction, target, kernel


def test_full_annotation_matches_original_gaussian_mse():
    prediction, target, kernel = _example()
    criterion = PUGaussianMSE(annotation_propensity=1.0)
    transformed = criterion._target_transform(prediction, target, kernel)
    expected = F.mse_loss(prediction, transformed)
    actual = criterion(prediction, target, kernel)
    torch.testing.assert_close(actual, expected)


def test_partial_annotation_risk_is_finite_and_differentiable():
    prediction, target, kernel = _example()
    criterion = PUGaussianMSE(annotation_propensity=0.4, pos_thr=0.1)
    loss = criterion(prediction, target, kernel)
    assert torch.isfinite(loss)
    assert 0.0 < criterion.last_components["class_prior"] < 1.0
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_empty_observed_positive_does_not_create_false_background_supervision():
    prediction = torch.randn(1, 1, 5, 5, requires_grad=True)
    target = torch.zeros_like(prediction)
    kernel = torch.ones(1, 1, 1, 1)
    criterion = PUGaussianMSE(annotation_propensity=0.4)
    loss = criterion(prediction, target, kernel)
    torch.testing.assert_close(loss, prediction.new_zeros(()))
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))


def test_scar_risk_uses_complete_marginal_not_only_unlabeled_cells():
    prediction = torch.tensor([[[[0.1, 0.2], [0.3, 0.4]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0], [0.0, 0.0]]]])
    kernel = torch.ones(1, 1, 1, 1)
    criterion = PUGaussianMSE(annotation_propensity=0.5, pos_thr=0.5)
    actual = criterion(prediction, target, kernel)

    positive_mass = prediction.new_tensor(0.25 / 0.5)
    positive_risk = positive_mass * (prediction[0, 0, 0, 0] - 1.0).pow(2)
    marginal_negative = prediction.pow(2).mean()
    negative_on_positive = positive_mass * prediction[0, 0, 0, 0].pow(2)
    expected = positive_risk + marginal_negative - negative_on_positive
    torch.testing.assert_close(actual, expected)
