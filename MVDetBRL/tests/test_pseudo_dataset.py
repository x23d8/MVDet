import json
from types import SimpleNamespace

import numpy as np

from multiview_detector.datasets.frameDataset import frameDataset


def test_load_pseudo_labels_maps_world_grid_to_reduced_bev(tmp_path):
    payload = {
        "frame": 5,
        "points": [
            {"grid_x": 12, "grid_y": 8, "confidence": 0.7},
            {"grid_x": 13, "grid_y": 9, "confidence": 0.9},
        ],
    }
    (tmp_path / "00000005.json").write_text(json.dumps(payload), encoding="utf-8")

    dataset = frameDataset.__new__(frameDataset)
    dataset.reducedgrid_shape = [4, 5]
    dataset.grid_reduce = 4
    dataset.pseudo_dir = str(tmp_path)
    dataset.base = SimpleNamespace(indexing="xy")
    dataset.map_gt = {5: object(), 10: object()}
    dataset.map_pseudo = {}
    dataset.map_pseudo_conf = {}

    dataset.load_pseudo_labels()

    assert dataset.map_pseudo[5][2, 3] == 1.0
    assert np.isclose(dataset.map_pseudo_conf[5][2, 3], 0.9)
    assert dataset.map_pseudo[10].sum() == 0
