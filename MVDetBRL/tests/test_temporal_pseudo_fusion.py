import sys
from pathlib import Path

import numpy as np


TOOLS_DIR = Path(__file__).resolve().parents[2] / "SHOTBRL" / "tools"
sys.path.insert(0, str(TOOLS_DIR))

from generate_yolo_pose_bev_pseudo import merge_candidates, select_temporal_clusters


def _candidate(camera, x, confidence=0.9):
    return {
        "camera": camera,
        "world_m": np.asarray([x, 0.0], dtype=np.float64),
        "confidence": confidence,
    }


def test_merge_never_places_two_detections_from_same_camera_in_one_cluster():
    clusters = merge_candidates(
        [_candidate(0, 0.0), _candidate(0, 0.1), _candidate(1, 0.05)],
        radius_m=0.6,
        min_views=1,
    )

    assert len(clusters) == 2
    assert sorted(cluster["num_views"] for cluster in clusters) == [1, 2]


def test_temporal_singleton_only_continues_an_unmatched_previous_track():
    multi_view = {
        "center_m": np.asarray([2.0, 0.0]),
        "confidence": 0.9,
        "num_views": 2,
    }
    singleton = {
        "center_m": np.asarray([0.2, 0.0]),
        "confidence": 0.8,
        "num_views": 1,
    }

    accepted, _, rescued = select_temporal_clusters(
        [multi_view, singleton],
        min_views=2,
        previous_tracks=[np.asarray([0.0, 0.0])],
        singleton_conf=0.65,
        temporal_radius_m=0.6,
    )

    assert len(accepted) == 2
    assert rescued == 1
    assert singleton["temporal_rescued"] is True


def test_current_multi_view_detection_prevents_duplicate_singleton_rescue():
    multi_view = {
        "center_m": np.asarray([0.1, 0.0]),
        "confidence": 0.9,
        "num_views": 2,
    }
    singleton = {
        "center_m": np.asarray([0.15, 0.0]),
        "confidence": 0.8,
        "num_views": 1,
    }

    accepted, _, rescued = select_temporal_clusters(
        [multi_view, singleton],
        min_views=2,
        previous_tracks=[np.asarray([0.0, 0.0])],
        singleton_conf=0.65,
        temporal_radius_m=0.6,
    )

    assert accepted == [multi_view]
    assert rescued == 0

