import runpy
import json
from pathlib import Path


MODULE = runpy.run_path(str(
    Path(__file__).parents[2] / "simulate_dropped_annotations.py"
))


def _person(visible_views, area, num_cam=4):
    side = int(area ** 0.5)
    views = []
    for camera in range(num_cam):
        if camera < visible_views:
            views.append({"xmin": 10, "xmax": 10 + side, "ymin": 20, "ymax": 20 + side})
        else:
            views.append({"xmin": -1, "xmax": -1, "ymin": -1, "ymax": -1})
    return {"personID": visible_views, "views": views}


def test_visibility_sar_preserves_count_and_is_deterministic():
    persons = [_person(1, 100), _person(2, 400), _person(3, 900), _person(4, 1600)]
    split = MODULE["split_instances_visibility_sar"]
    observed_a, hidden_a = split(persons, 0.5, 7, 4, seed=3)
    observed_b, hidden_b = split(persons, 0.5, 7, 4, seed=3)
    assert len(observed_a) == 2
    assert len(hidden_a) == 2
    assert hidden_a == hidden_b
    assert observed_a == observed_b


def test_visibility_statistics_ignore_missing_boxes():
    visible, area = MODULE["_view_statistics"](_person(1, 100), 4)
    assert visible == 1
    assert area == 100


def test_visibility_sar_drops_difficult_person_more_frequently():
    difficult = _person(1, 100)
    easy = _person(4, 1600)
    split = MODULE["split_instances_visibility_sar"]
    difficult_drops = 0
    easy_drops = 0
    for frame_id in range(200):
        _, hidden = split([difficult, easy], 0.5, frame_id, 4, seed=9)
        difficult_drops += difficult in hidden
        easy_drops += easy in hidden
    assert difficult_drops > easy_drops * 2


def test_simulator_accepts_dataset_root_without_named_wrapper(tmp_path):
    dataset_root = tmp_path / "arbitrary_dataset_name"
    annotations = dataset_root / "annotations_positions"
    annotations.mkdir(parents=True)
    person = _person(2, 400, num_cam=7)
    (annotations / "00000000.json").write_text(json.dumps([person]))

    output_root = tmp_path / "writable_output"
    MODULE["simulate"](
        "Wildtrack", str(dataset_root), 0.9, ["drop20"], seed=1,
        output_root=str(output_root),
    )

    output = (
        output_root / "drop_annotations" / "drop_20"
        / "annotations_positions" / "00000000.json"
    )
    assert output.is_file()
