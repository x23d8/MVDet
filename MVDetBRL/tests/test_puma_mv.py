import torch

from multiview_detector.models.puma_mv import (
    MultiHeightFeatureSampler,
    UncertaintyGatedSetFusion,
)


def test_multi_height_sampler_identity_projection():
    feature = torch.arange(9, dtype=torch.float32).reshape(1, 1, 1, 3, 3)
    projection = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]]
    )
    y, x = torch.meshgrid(torch.arange(3), torch.arange(3), indexing="ij")
    bev_xy = torch.stack([x, y], dim=-1).float()
    sampler = MultiHeightFeatureSampler(heights_m=(0.0, 1.0), image_shape=(3, 3))
    samples, valid = sampler(feature, projection, bev_xy)
    assert samples.shape == (1, 1, 2, 1, 3, 3)
    assert valid.all()
    torch.testing.assert_close(samples[0, 0, 0], feature[0, 0])
    torch.testing.assert_close(samples[0, 0, 1], feature[0, 0])


def test_set_fusion_is_permutation_invariant_over_cameras():
    torch.manual_seed(3)
    fusion = UncertaintyGatedSetFusion(4, 6, num_heights=2)
    samples = torch.randn(2, 3, 2, 4, 5, 7)
    valid = torch.ones(2, 3, 2, 1, 5, 7, dtype=torch.bool)
    uncertainty = torch.rand(2, 3, 2, 1, 5, 7)
    output, _ = fusion(samples, valid, uncertainty)
    permutation = torch.tensor([2, 0, 1])
    permuted_output, _ = fusion(
        samples[:, permutation], valid[:, permutation], uncertainty[:, permutation]
    )
    torch.testing.assert_close(output, permuted_output, rtol=1e-5, atol=1e-6)


def test_invalid_camera_cannot_change_fused_output():
    torch.manual_seed(5)
    fusion = UncertaintyGatedSetFusion(3, 4, num_heights=1)
    samples = torch.randn(1, 2, 1, 3, 4, 4)
    valid = torch.ones(1, 2, 1, 1, 4, 4, dtype=torch.bool)
    valid[:, 1] = False
    output, weights = fusion(samples, valid)
    changed = samples.clone()
    changed[:, 1] = 1e6
    changed_output, changed_weights = fusion(changed, valid)
    torch.testing.assert_close(output, changed_output)
    torch.testing.assert_close(weights, changed_weights)
    assert torch.count_nonzero(weights[:, 1]).item() == 0


def test_multiscale_fusion_is_camera_permutation_invariant():
    torch.manual_seed(7)
    fusion = UncertaintyGatedSetFusion(
        in_channels=3, out_channels=4, num_heights=2, num_scales=3
    )
    samples = torch.randn(1, 2, 2, 3, 3, 3, 4)
    valid = torch.ones(1, 2, 2, 3, 1, 3, 4, dtype=torch.bool)
    output, _ = fusion(samples, valid)
    permutation = torch.tensor([1, 0])
    permuted, _ = fusion(samples[:, permutation], valid[:, permutation])
    torch.testing.assert_close(output, permuted, rtol=1e-5, atol=1e-6)


def test_nonfinite_projection_is_masked_without_nan_output():
    feature = torch.randn(1, 1, 2, 3, 3)
    projection = torch.full((1, 3, 4), float("inf"))
    bev_xy = torch.zeros(2, 2, 2)
    sampler = MultiHeightFeatureSampler(heights_m=(0.0,), image_shape=(3, 3))
    samples, valid = sampler(feature, projection, bev_xy)
    assert not valid.any()
    assert torch.isfinite(samples).all()
    torch.testing.assert_close(samples, torch.zeros_like(samples))


def test_large_calibration_coefficients_do_not_overflow_half_features():
    if not torch.cuda.is_available():
        return
    feature = torch.randn(1, 1, 2, 3, 3, device="cuda", dtype=torch.float16)
    projection = torch.tensor(
        [[[1e5, 0.0, 0.0, 0.0], [0.0, 1e5, 0.0, 0.0], [0.0, 0.0, 0.0, 1e3]]],
        device="cuda",
    )
    bev_xy = torch.full((2, 2, 2), 0.01, device="cuda")
    sampler = MultiHeightFeatureSampler(heights_m=(0.0,), image_shape=(3, 3)).cuda()
    samples, valid = sampler(feature, projection, bev_xy)
    assert valid.all()
    assert torch.isfinite(samples).all()
