import numpy as np

from multiview_detector.evaluation.pyeval.evaluateDetection import evaluateDetection_py


def test_zero_detection_frame_is_counted_as_false_negative(tmp_path):
    gt_path = tmp_path / "gt.txt"
    result_path = tmp_path / "result.txt"
    np.savetxt(gt_path, np.array([[0, 10, 10], [1, 20, 20]]), fmt="%d")
    np.savetxt(result_path, np.array([[0, 10, 10]]), fmt="%d")
    recall, precision, moda, _ = evaluateDetection_py(
        result_path, gt_path, "Wildtrack", frames=[0, 1]
    )
    assert recall == 50.0
    assert precision == 100.0
    assert moda == 50.0


def test_empty_result_evaluates_all_requested_frames(tmp_path):
    gt_path = tmp_path / "gt.txt"
    result_path = tmp_path / "result.txt"
    np.savetxt(gt_path, np.array([[0, 10, 10], [1, 20, 20]]), fmt="%d")
    result_path.write_text("", encoding="utf-8")
    recall, precision, moda, modp = evaluateDetection_py(
        result_path, gt_path, "Wildtrack", frames=[0, 1]
    )
    assert (recall, precision, moda, modp) == (0, 0, 0, 0)
