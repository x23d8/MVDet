import numpy as np
import torch

from multiview_detector.models.puma_dense_detector import PUMADenseDetector


class _TinyBase:
    indexing = "xy"

    def __init__(self, num_cam):
        intrinsic = np.array(
            [[20.0, 0.0, 48.0], [0.0, 20.0, 32.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        extrinsic = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        self.intrinsic_matrices = [intrinsic.copy() for _ in range(num_cam)]
        self.extrinsic_matrices = [extrinsic.copy() for _ in range(num_cam)]

    @staticmethod
    def get_worldcoord_from_worldgrid(worldgrid):
        return worldgrid.astype(np.float32)


class _TinyDataset:
    num_cam = 2
    img_shape = [64, 96]
    reducedgrid_shape = [4, 6]
    grid_reduce = 4
    base = _TinyBase(num_cam)


class _NamedBase(_TinyBase):
    def __init__(self, num_cam, name):
        super().__init__(num_cam)
        self.__name__ = name


class _NamedDataset(_TinyDataset):
    def __init__(self, name):
        self.base = _NamedBase(self.num_cam, name)


def test_puma_dense_detector_preserves_mvdet_interface():
    model = PUMADenseDetector(
        _TinyDataset(),
        feature_channels=8,
        fused_channels=8,
        heights_m=(0.0, 1.0),
        world_unit_to_m=1.0,
    ).eval()
    images = torch.randn(1, 2, 3, 64, 96)
    with torch.no_grad():
        map_result, image_results, auxiliary = model(images, return_aux=True)
    assert map_result.shape == (1, 1, 4, 6)
    assert len(image_results) == 2
    assert all(result.shape[:2] == (1, 2) for result in image_results)
    # 2 cameras x 2 heights x 3 FPN scales, plus one learned null token.
    assert auxiliary["fusion_weights"].shape == (1, 13, 1, 4, 6)
    assert torch.isfinite(map_result).all()


def test_non_power_of_two_channel_counts_are_supported():
    model = PUMADenseDetector(
        _TinyDataset(),
        feature_channels=10,
        fused_channels=14,
        heights_m=(0.0,),
        world_unit_to_m=1.0,
    ).eval()
    with torch.no_grad():
        map_result, _ = model(torch.randn(1, 2, 3, 64, 96))
    assert map_result.shape == (1, 1, 4, 6)


def test_world_unit_is_inferred_per_dataset():
    wildtrack = PUMADenseDetector(_NamedDataset("Wildtrack"), 8, 8)
    multiviewx = PUMADenseDetector(_NamedDataset("MultiviewX"), 8, 8)
    assert wildtrack.world_unit_to_m == 0.01
    assert multiviewx.world_unit_to_m == 1.0
    # The same native grid coordinate represents centimetres vs metres.
    torch.testing.assert_close(multiviewx.bev_xy_m, wildtrack.bev_xy_m * 100)


def test_parallel_and_sequential_view_encoding_have_equal_eval_output():
    torch.manual_seed(17)
    sequential = PUMADenseDetector(
        _TinyDataset(), 8, 8, heights_m=(0.0,), world_unit_to_m=1.0
    ).eval()
    parallel = PUMADenseDetector(
        _TinyDataset(), 8, 8, heights_m=(0.0,), world_unit_to_m=1.0,
        parallel_view_encoding=True,
    ).eval()
    parallel.load_state_dict(sequential.state_dict())
    images = torch.randn(1, 2, 3, 64, 96)
    with torch.no_grad():
        sequential_map, sequential_images = sequential(images)
        parallel_map, parallel_images = parallel(images)
    torch.testing.assert_close(sequential_map, parallel_map, rtol=1e-5, atol=1e-6)
    for sequential_image, parallel_image in zip(sequential_images, parallel_images):
        torch.testing.assert_close(sequential_image, parallel_image, rtol=1e-5, atol=1e-6)
