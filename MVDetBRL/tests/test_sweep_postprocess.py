from pathlib import Path
import runpy

import numpy as np


MODULE = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "sweep_postprocess.py"))


def test_detection_extraction_preserves_close_peaks_with_small_nms():
    maps = np.zeros((1, 5, 8), dtype=np.float32)
    maps[0, 2, 1] = 0.9
    maps[0, 2, 5] = 0.8
    detections = MODULE["detections_from_maps"](
        maps, np.array([7]), threshold=0.4, radius_grid=3,
        grid_reduce=1, indexing="xy",
    )
    assert detections.shape == (2, 3)
    assert set(detections[:, 0].astype(int)) == {7}
