from pathlib import Path

from main import snapshot_training_sources


def test_snapshot_excludes_bundled_mot_devkit_and_keeps_training_code(tmp_path):
    project = tmp_path / "project"
    package = project / "multiview_detector"
    package.mkdir(parents=True)
    (package / "model.py").write_text("VALUE = 1\n")
    external = package / "evaluation" / "motchallenge-devkit" / "very" / "deep"
    external.mkdir(parents=True)
    (external / "ignored.py").write_text("raise RuntimeError\n")
    (external / "weights.mat").write_bytes(b"large")
    (project / "main.py").write_text("print('train')\n")
    destination = tmp_path / "snapshot"

    snapshot_training_sources(project, destination)

    assert (destination / "multiview_detector" / "model.py").is_file()
    assert (destination / "main.py").is_file()
    assert not (destination / "multiview_detector" / "evaluation" / "motchallenge-devkit").exists()
