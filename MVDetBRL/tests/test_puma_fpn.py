import torch

from multiview_detector.models.puma_fpn import ResNet18FPNEncoder


def test_fpn_returns_three_true_spatial_scales():
    encoder = ResNet18FPNEncoder(out_channels=10).eval()
    with torch.no_grad():
        p3, p4, p5 = encoder(torch.randn(1, 3, 64, 96))
    assert p3.shape == (1, 10, 8, 12)
    assert p4.shape == (1, 10, 4, 6)
    assert p5.shape == (1, 10, 2, 3)
