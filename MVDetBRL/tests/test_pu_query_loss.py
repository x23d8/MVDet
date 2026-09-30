import torch
import pytest

from multiview_detector.loss.pu_query_loss import PUQuerySetLoss


def test_query_prior_cap_must_be_a_probability():
    with pytest.raises(ValueError, match="max_dynamic_prior"):
        PUQuerySetLoss(max_dynamic_prior=1.1)


def _grid():
    yy, xx = torch.meshgrid(torch.arange(4), torch.arange(5), indexing="ij")
    return torch.stack([xx, yy], dim=-1).float()


def test_full_annotation_penalizes_unmatched_query_as_background():
    queries = {
        "xy_m": torch.tensor([[[2.0, 1.0], [4.0, 3.0]]], requires_grad=True),
        "objectness": torch.tensor([[2.0, 2.0]], requires_grad=True),
    }
    target = torch.zeros(1, 1, 4, 5)
    target[0, 0, 1, 2] = 1.0
    loss = PUQuerySetLoss(annotation_propensity=1.0)(queries, target, _grid())
    loss.backward()
    assert queries["objectness"].grad[0, 0] < 0
    assert queries["objectness"].grad[0, 1] > 0


def test_empty_partial_frame_does_not_create_false_query_negatives():
    logits = torch.randn(1, 3, requires_grad=True)
    queries = {"xy_m": torch.randn(1, 3, 2, requires_grad=True), "objectness": logits}
    target = torch.zeros(1, 1, 4, 5)
    loss = PUQuerySetLoss(annotation_propensity=0.4)(queries, target, _grid())
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_matched_coordinate_receives_regression_gradient():
    xy = torch.tensor([[[1.5, 1.0], [4.0, 3.0]]], requires_grad=True)
    queries = {"xy_m": xy, "objectness": torch.zeros(1, 2, requires_grad=True)}
    target = torch.zeros(1, 1, 4, 5)
    target[0, 0, 1, 2] = 1.0
    loss = PUQuerySetLoss(annotation_propensity=0.5)(queries, target, _grid())
    loss.backward()
    assert xy.grad[0, 0, 0] < 0


def test_close_confident_queries_receive_separation_gradient():
    xy = torch.tensor([[[1.0, 1.0], [1.1, 1.0]]], requires_grad=True)
    queries = {"xy_m": xy, "objectness": torch.full((1, 2), 4.0, requires_grad=True)}
    target = torch.zeros(1, 1, 4, 5)
    criterion = PUQuerySetLoss(
        annotation_propensity=1.0,
        coordinate_weight=0.0,
        separation_weight=1.0,
        minimum_separation_m=0.25,
    )
    criterion(queries, target, _grid()).backward()
    assert xy.grad[0, 0, 0] > 0
    assert xy.grad[0, 1, 0] < 0


def test_sar_selection_likelihood_trains_objectness_and_propensity():
    objectness = torch.zeros(1, 3, requires_grad=True)
    propensity = torch.zeros(1, 3, requires_grad=True)
    queries = {
        "xy_m": torch.tensor([[[2.0, 1.0], [4.0, 3.0], [0.0, 0.0]]], requires_grad=True),
        "objectness": objectness,
        "propensity_logit": propensity,
    }
    target = torch.zeros(1, 1, 4, 5)
    target[0, 0, 1, 2] = 1.0
    criterion = PUQuerySetLoss(
        annotation_propensity=0.4,
        propensity_mode="sar",
        separation_weight=0.0,
    )
    loss = criterion(queries, target, _grid())
    loss.backward()
    assert torch.isfinite(loss)
    assert objectness.grad is not None and torch.isfinite(objectness.grad).all()
    assert propensity.grad is not None and torch.isfinite(propensity.grad).all()


def test_coordinate_nll_calibrates_predicted_sigma():
    xy = torch.tensor([[[0.0, 0.0], [4.0, 3.0]]], requires_grad=True)
    log_sigma = torch.zeros(1, 2, 2, requires_grad=True)
    queries = {
        "xy_m": xy,
        "objectness": torch.zeros(1, 2, requires_grad=True),
        "log_sigma_m": log_sigma,
    }
    target = torch.zeros(1, 1, 4, 5)
    target[0, 0, 1, 2] = 1.0
    criterion = PUQuerySetLoss(
        annotation_propensity=1.0,
        separation_weight=0.0,
    )
    criterion(queries, target, _grid()).backward()
    # The matched query has a large localization error, so robust NLL still
    # asks gradient descent to increase sigma.
    matched = torch.cdist(xy.detach()[0], torch.tensor([[2.0, 1.0]])).argmin()
    assert (log_sigma.grad[0, matched] < 0).any()


def test_scar_query_risk_uses_all_queries_as_the_marginal():
    logits = torch.tensor([[-2.0, 0.0, 1.0, -1.0]], requires_grad=True)
    queries = {
        "xy_m": torch.tensor(
            [[[2.0, 1.0], [4.0, 3.0], [0.0, 0.0], [4.0, 0.0]]],
            requires_grad=True,
        ),
        "objectness": logits,
    }
    target = torch.zeros(1, 1, 4, 5)
    target[0, 0, 1, 2] = 1.0
    criterion = PUQuerySetLoss(
        annotation_propensity=0.5,
        coordinate_weight=0.0,
        separation_weight=0.0,
    )
    actual = criterion(queries, target, _grid())

    positive_mass = logits.new_tensor(0.25 / 0.5)
    positive = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :1], torch.ones_like(logits[:, :1])
    )
    marginal_negative = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, torch.zeros_like(logits)
    )
    negative_on_positive = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :1], torch.zeros_like(logits[:, :1])
    )
    expected = positive_mass * positive + marginal_negative - positive_mass * negative_on_positive
    torch.testing.assert_close(actual, expected)
