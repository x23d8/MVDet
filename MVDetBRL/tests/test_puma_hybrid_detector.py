import torch

from multiview_detector.models.puma_hybrid_detector import PUMAHybridDetector
from test_puma_dense_detector import _TinyDataset


def test_hybrid_detector_is_end_to_end_and_preserves_trainer_interface():
    model = PUMAHybridDetector(
        _TinyDataset(),
        feature_channels=8,
        fused_channels=8,
        heights_m=(0.0, 1.0),
        world_unit_to_m=1.0,
        num_queries=3,
        query_hidden_channels=8,
        query_layers=1,
    ).eval()
    images = torch.randn(1, 2, 3, 64, 96, requires_grad=True)
    map_result, image_results, auxiliary = model(images, return_aux=True)
    assert map_result.shape == (1, 1, 4, 6)
    assert len(image_results) == 2
    assert auxiliary["queries"]["xy_m"].shape == (1, 3, 2)
    assert auxiliary["query_map"].shape == map_result.shape
    assert 0.0 < auxiliary["query_residual_gate"] < 0.1
    map_result.square().mean().backward()
    assert images.grad is not None
    assert model.query_refiner.objectness.weight.grad is not None
    assert model.query_refiner.offset_heads[0].weight.grad is not None


def test_masked_camera_cannot_change_hybrid_bev_output():
    torch.manual_seed(13)
    model = PUMAHybridDetector(
        _TinyDataset(),
        feature_channels=8,
        fused_channels=8,
        heights_m=(0.0,),
        world_unit_to_m=1.0,
        num_queries=2,
        query_hidden_channels=8,
        query_layers=1,
    ).eval()
    images = torch.randn(1, 2, 3, 64, 96)
    camera_mask = torch.tensor([[True, False]])
    changed = images.clone()
    changed[:, 1] = torch.randn_like(changed[:, 1]) * 1000
    with torch.no_grad():
        output, _ = model(images, camera_mask=camera_mask)
        changed_output, _ = model(changed, camera_mask=camera_mask)
    torch.testing.assert_close(output, changed_output, rtol=1e-5, atol=1e-6)
