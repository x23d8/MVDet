import torch

from multiview_detector.models.puma_queries import (
    CylindricalQueryRefiner,
    DensePeakQueryInitializer,
)


def test_peak_initializer_returns_separated_metric_queries():
    score = torch.zeros(1, 1, 5, 6)
    score[0, 0, 1, 1] = 4.0
    score[0, 0, 3, 4] = 3.0
    feature = torch.arange(1 * 3 * 5 * 6, dtype=torch.float32).reshape(1, 3, 5, 6)
    yy, xx = torch.meshgrid(torch.arange(5), torch.arange(6), indexing="ij")
    bev_xy = torch.stack([xx, yy], dim=-1).float()
    initializer = DensePeakQueryInitializer(num_queries=2, suppression_kernel=3)
    xy, query_feature, query_score, indices = initializer(score, feature, bev_xy)
    torch.testing.assert_close(xy[0, 0], torch.tensor([1.0, 1.0]))
    torch.testing.assert_close(xy[0, 1], torch.tensor([4.0, 3.0]))
    torch.testing.assert_close(query_score[0], torch.tensor([4.0, 3.0]))
    assert query_feature.shape == (1, 2, 3)
    assert indices.shape == (1, 2)


def _projection(num_views=2):
    matrix = torch.tensor(
        [[20.0, 0.0, 0.0, 48.0], [0.0, 20.0, 0.0, 32.0], [0.0, 0.0, 0.0, 1.0]]
    )
    return matrix.unsqueeze(0).repeat(num_views, 1, 1)


def test_cylindrical_refiner_shapes_gradients_and_camera_permutation():
    torch.manual_seed(7)
    refiner = CylindricalQueryRefiner(
        image_channels=4,
        query_channels=6,
        hidden_channels=8,
        heights_m=(0.0, 1.0),
        radial_samples=2,
        num_layers=2,
        image_shape=(64, 96),
    )
    image_features = torch.randn(1, 2, 4, 8, 12, requires_grad=True)
    projection = _projection()
    xy = torch.tensor([[[0.0, 0.0], [0.5, 0.25], [-0.25, 0.5]]])
    query_feature = torch.randn(1, 3, 6, requires_grad=True)
    score = torch.tensor([[2.0, 1.0, 0.5]])

    output = refiner(image_features, projection, xy, query_feature, score)
    assert output["xy_m"].shape == (1, 3, 2)
    assert output["objectness"].shape == (1, 3)
    assert output["log_sigma_m"].shape == (1, 3, 2)
    assert torch.isfinite(output["attention"]).all()

    permutation = torch.tensor([1, 0])
    permuted = refiner(
        image_features[:, permutation], projection[permutation], xy, query_feature, score
    )
    torch.testing.assert_close(output["xy_m"], permuted["xy_m"], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        output["objectness"], permuted["objectness"], atol=1e-5, rtol=1e-5
    )

    (output["objectness"].sum() + output["xy_m"].sum()).backward()
    assert image_features.grad is not None
    assert query_feature.grad is not None


def test_cylindrical_refiner_consumes_multiple_native_scales():
    refiner = CylindricalQueryRefiner(
        image_channels=4,
        query_channels=6,
        hidden_channels=8,
        heights_m=(0.0,),
        radial_samples=2,
        num_layers=1,
        num_scales=3,
        image_shape=(64, 96),
    )
    features = [
        torch.randn(1, 2, 4, 8, 12),
        torch.randn(1, 2, 4, 4, 6),
        torch.randn(1, 2, 4, 2, 3),
    ]
    output = refiner(
        features,
        _projection(),
        torch.tensor([[[0.0, 0.0], [0.5, 0.25]]]),
        torch.randn(1, 2, 6),
        torch.ones(1, 2),
    )
    # 2 cameras x (centre + 2 ring points) x 3 FPN scales, plus null.
    assert output["attention"].shape == (1, 2, 19)
