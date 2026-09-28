"""Run a low-weight temporal YOLO-pseudo grid on the original MVDet model."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
PROJECT_ROOT = WORK_ROOT / "mvdet_temporal_project"
RESULT_ROOT = WORK_ROOT / "mvdet_temporal_low_lambda_results"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log("RUN: " + " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def find_code_archive() -> Path:
    matches = sorted(INPUT_ROOT.rglob("mvdetbrl_temporal_code.zip"))
    if not matches:
        raise FileNotFoundError("mvdetbrl_temporal_code.zip is missing from Kaggle inputs")
    return matches[0]


def find_wildtrack_root() -> Path:
    candidates = []
    for annotations in INPUT_ROOT.rglob("annotations_positions"):
        root = annotations.parent
        required = [
            root / "Image_subsets" / "C1",
            root / "Image_subsets" / "C7",
            root / "calibrations" / "intrinsic_zero",
            root / "calibrations" / "extrinsic",
            root / "drop_annotations" / "drop_45" / "annotations_positions",
        ]
        if all(path.is_dir() for path in required):
            candidates.append(root)
    if not candidates:
        raise FileNotFoundError("Could not find a complete Wildtrack root in /kaggle/input")
    candidates.sort(key=lambda path: ("wildtrack" not in str(path).lower(), len(path.parts)))
    log("Wildtrack candidates: " + json.dumps([str(path) for path in candidates]))
    return candidates[0]


def prepare_project() -> Path:
    archive = find_code_archive()
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    PROJECT_ROOT.mkdir(parents=True)
    log(f"Extracting code archive: {archive}")
    shutil.unpack_archive(str(archive), str(PROJECT_ROOT))
    mvdet_root = PROJECT_ROOT / "MVDetBRL"
    if not (mvdet_root / "main.py").is_file():
        raise FileNotFoundError(f"Invalid code archive: {mvdet_root / 'main.py'} not found")
    return mvdet_root


def install_runtime() -> None:
    # Keep Kaggle's CUDA-enabled torch/torchvision builds; only add the two
    # dependencies that are not guaranteed to be present in the base image.
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "kornia==0.7.3",
            "ultralytics==8.4.69",
        ]
    )


def save_run_manifest(data_root: Path, environment: dict[str, str]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    selected = [
        "DATASET",
        "DATA_PATH",
        "PSEUDO_DIR",
        "DROP_RATIOS",
        "LAMBDAS",
        "SEEDS",
        "EPOCHS",
        "BATCH_SIZE",
        "NUM_WORKERS",
        "LR",
        "YOLO_MODEL",
        "YOLO_IMGSZ",
        "YOLO_CONF",
        "YOLO_KPT_CONF",
        "YOLO_CANDIDATE_CONF",
        "YOLO_FOOT_ANCHOR",
        "YOLO_MIN_VIEWS",
        "YOLO_MERGE_RADIUS_M",
        "YOLO_TEMPORAL_SINGLETONS",
        "YOLO_TEMPORAL_CONF",
        "YOLO_TEMPORAL_RADIUS_M",
    ]
    manifest = {
        "data_root": str(data_root),
        "python": sys.version,
        "settings": {key: environment[key] for key in selected},
    }
    (RESULT_ROOT / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def collect_outputs(mvdet_root: Path, pseudo_dir: Path) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    logs = mvdet_root / "logs"
    if logs.exists():
        destination = RESULT_ROOT / "logs"
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(logs, destination)
    summary = pseudo_dir / "generation_summary.json"
    if summary.is_file():
        shutil.copy2(summary, RESULT_ROOT / summary.name)
    archive = shutil.make_archive(
        str(WORK_ROOT / "mvdet_temporal_low_lambda_results"),
        "zip",
        root_dir=RESULT_ROOT,
    )
    log(f"Results archived at {archive}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
    log("Starting separate MVDet temporal-pseudo Kaggle session")
    install_runtime()
    mvdet_root = prepare_project()
    data_root = find_wildtrack_root()
    pseudo_dir = WORK_ROOT / "pseudo_labels" / "wildtrack_yolo26x_temporal"

    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON_BIN": sys.executable,
            "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
            "TORCH_HOME": str(WORK_ROOT / ".cache" / "torch"),
            "YOLO_CONFIG_DIR": str(WORK_ROOT / ".config" / "Ultralytics"),
            "DATASET": "wildtrack",
            "DATA_PATH": str(data_root),
            "PSEUDO_DIR": str(pseudo_dir),
            "DROP_RATIOS": "45",
            "LAMBDAS": "0.005 0.01 0.025",
            "SEEDS": "1",
            "EPOCHS": "7",
            "BATCH_SIZE": "1",
            "NUM_WORKERS": "4",
            "LR": "0.1",
            "MOMENTUM": "0.5",
            "WEIGHT_DECAY": "0.0005",
            "ALPHA": "1.0",
            "BRL_POS_THR": "0.1",
            "PSEUDO_THR": "0.1",
            "GPU": "0",
            "MIN_PSEUDO_FILES": "360",
            "AUTO_PREPARE_PSEUDO": "1",
            "FORCE_REGENERATE_PSEUDO": "1",
            "YOLO_MODEL": str(WORK_ROOT / "yolo26x-pose.pt"),
            "YOLO_IMGSZ": "1280",
            "YOLO_DEVICE": "0",
            "YOLO_CONF": "0.25",
            "YOLO_KPT_CONF": "0.35",
            "YOLO_CANDIDATE_CONF": "0.15",
            "YOLO_FOOT_ANCHOR": "pose_x_bbox_y",
            "YOLO_MIN_VIEWS": "2",
            "YOLO_MERGE_RADIUS_M": "0.60",
            "YOLO_TEMPORAL_SINGLETONS": "1",
            "YOLO_TEMPORAL_CONF": "0.65",
            "YOLO_TEMPORAL_RADIUS_M": "0.60",
            "YOLO_INFERENCE_BATCH": "4",
        }
    )
    save_run_manifest(data_root, environment)
    log("Grid: pseudo_only, lambda=[0.005, 0.01, 0.025], drop=45, epochs=7")
    try:
        run(["bash", "train_grid_pseudo_only.sh", "--log_interval", "10"], cwd=mvdet_root, env=environment)
    finally:
        collect_outputs(mvdet_root, pseudo_dir)
    log("MVDet temporal-pseudo grid completed successfully")


if __name__ == "__main__":
    main()
