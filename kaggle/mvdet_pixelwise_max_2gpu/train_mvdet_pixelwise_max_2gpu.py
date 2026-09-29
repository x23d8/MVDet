"""Kaggle runner for a two-GPU MVDet pixel-wise-max pseudo-label ablation."""

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
PROJECT_ROOT = WORK_ROOT / "mvdet_pixelwise_max_project"
RESULT_ROOT = WORK_ROOT / "mvdet_pixelwise_max_2gpu_results"
REPO_URL = os.environ.get("REPO_URL", "https://github.com/x23d8/MVDet.git")
REPO_BRANCH = os.environ.get("REPO_BRANCH", "pseudolabelloss")
REPO_SUBDIR = os.environ.get("REPO_SUBDIR", "MVDetBRL")
WILDTRACK_ROOT = Path(
    os.environ.get(
        "WILDTRACK_ROOT",
        "/kaggle/input/thesis-dataset-new/Wildtrack/Wildtrack",
    )
)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def run(command: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    log("RUN: " + " ".join(command))
    subprocess.run(command, cwd=cwd, env=env, check=True)


def require_two_gpus() -> None:
    import torch

    count = torch.cuda.device_count()
    devices = [torch.cuda.get_device_name(index) for index in range(count)]
    log(f"CUDA devices ({count}): {devices}")
    if count < 2:
        raise RuntimeError(
            "This notebook is configured for two GPUs but Kaggle exposed fewer than two. "
            "Select a 2xT4 accelerator before starting the session."
        )


def find_wildtrack_root(drop_ratios: str) -> Path:
    drops = [value for value in drop_ratios.split() if value != "0"]

    def is_complete(root: Path) -> bool:
        required = [
            root / "annotations_positions",
            root / "Image_subsets" / "C1",
            root / "Image_subsets" / "C7",
            root / "calibrations" / "intrinsic_zero",
            root / "calibrations" / "extrinsic",
        ]
        required.extend(
            root / "drop_annotations" / f"drop_{drop}" / "annotations_positions"
            for drop in drops
        )
        return all(path.is_dir() for path in required)

    if is_complete(WILDTRACK_ROOT):
        log(f"Wildtrack root: {WILDTRACK_ROOT}")
        return WILDTRACK_ROOT
    candidates = []
    for annotations in INPUT_ROOT.rglob("annotations_positions"):
        root = annotations.parent
        if is_complete(root):
            candidates.append(root)
    if not candidates:
        raise FileNotFoundError(
            f"Wildtrack is incomplete. Checked {WILDTRACK_ROOT} and scanned /kaggle/input"
        )
    candidates.sort(key=lambda path: ("wildtrack" not in str(path).lower(), len(path.parts)))
    log("Wildtrack candidates: " + json.dumps([str(path) for path in candidates]))
    return candidates[0]


def validate_project(mvdet_root: Path) -> None:
    checks = {
        PROJECT_ROOT / "SHOTBRL" / "tools" / "generate_yolo_pose_bev_pseudo.py": "pixelwise_max",
        mvdet_root / "multiview_detector" / "loss" / "brl_gaussian_mse.py": "pseudo_aggregation",
        mvdet_root / "main.py": "--device_ids",
        mvdet_root / "train_grid_pseudo_only.sh": "PSEUDO_AGGREGATION",
    }
    for path, marker in checks.items():
        if not path.is_file() or marker not in path.read_text(encoding="utf-8"):
            raise RuntimeError(
                f"The cloned branch is missing required feature {marker!r} in {path}. "
                "Push the pixelwise-max/2-GPU changes to REPO_BRANCH before running Kaggle."
            )
    model_source = (
        mvdet_root / "multiview_detector" / "models" / "persp_trans_detector.py"
    ).read_text(encoding="utf-8")
    if "cuda:0" in model_source:
        raise RuntimeError("PerspTransDetector still hard-codes cuda:0 and cannot use DataParallel safely")


def prepare_project() -> Path:
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    log(f"Cloning {REPO_URL} branch {REPO_BRANCH}")
    run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, str(PROJECT_ROOT)])
    mvdet_root = PROJECT_ROOT / REPO_SUBDIR
    validate_project(mvdet_root)
    return mvdet_root


def install_runtime() -> None:
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "kornia==0.7.3",
        "ultralytics==8.4.69",
    ])


def ensure_yolo_model(model_name: str) -> str:
    model_path = Path(model_name).expanduser()
    if model_path.is_file():
        return str(model_path)
    from ultralytics import YOLO

    previous_cwd = Path.cwd()
    try:
        os.chdir(WORK_ROOT)
        YOLO(model_name)
    finally:
        os.chdir(previous_cwd)
    downloaded = WORK_ROOT / Path(model_name).name
    resolved = str(downloaded) if downloaded.is_file() else model_name
    log(f"YOLO model ready: {resolved}")
    return resolved


def save_manifest(data_root: Path, environment: dict[str, str]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    keys = [
        "DATA_PATH", "DROP_RATIOS", "LAMBDAS", "SEEDS", "EPOCHS", "BATCH_SIZE",
        "NUM_WORKERS", "LR", "GPU", "DEVICE_IDS", "PSEUDO_AGGREGATION",
        "YOLO_MODEL", "YOLO_IMGSZ", "YOLO_CONF", "YOLO_CANDIDATE_CONF",
        "YOLO_FUSION_MODE", "YOLO_FOOT_ANCHOR", "YOLO_INFERENCE_BATCH",
    ]
    manifest = {
        "data_root": str(data_root),
        "repository": {"url": REPO_URL, "branch": REPO_BRANCH, "subdir": REPO_SUBDIR},
        "python": sys.version,
        "settings": {key: environment[key] for key in keys},
    }
    (RESULT_ROOT / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


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
    archive = shutil.make_archive(str(RESULT_ROOT), "zip", root_dir=RESULT_ROOT)
    log(f"Results archived at {archive}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    os.environ["PYTHONUNBUFFERED"] = "1"
    require_two_gpus()
    install_runtime()
    mvdet_root = prepare_project()

    environment = os.environ.copy()
    defaults = {
        "PYTHON_BIN": sys.executable,
        "PYTHONUNBUFFERED": "1",
        "MPLBACKEND": "Agg",
        "TORCH_HOME": str(WORK_ROOT / ".cache" / "torch"),
        "YOLO_CONFIG_DIR": str(WORK_ROOT / ".config" / "Ultralytics"),
        "DATASET": "wildtrack",
        "DROP_RATIOS": "60",
        "LAMBDAS": "0.01",
        "SEEDS": "1",
        "EPOCHS": "10",
        "BATCH_SIZE": "2",
        "NUM_WORKERS": "4",
        "LR": "0.1",
        "MOMENTUM": "0.5",
        "WEIGHT_DECAY": "0.0005",
        "ALPHA": "1.0",
        "BRL_POS_THR": "0.1",
        "PSEUDO_THR": "0.1",
        "PSEUDO_AGGREGATION": "max",
        "GPU": "0,1",
        "DEVICE_IDS": "0,1",
        "MIN_PSEUDO_FILES": "360",
        "AUTO_PREPARE_PSEUDO": "1",
        "FORCE_REGENERATE_PSEUDO": "1",
        "YOLO_MODEL": "yolo26x.pt",
        "YOLO_IMGSZ": "1280",
        "YOLO_DEVICE": "0",
        "YOLO_CONF": "0.20",
        "YOLO_CANDIDATE_CONF": "0.20",
        "YOLO_FUSION_MODE": "pixelwise_max",
        "YOLO_FOOT_ANCHOR": "bbox_bottom",
        "YOLO_MIN_VIEWS": "1",
        "YOLO_MERGE_RADIUS_M": "0.60",
        "YOLO_TEMPORAL_SINGLETONS": "0",
        "YOLO_TEMPORAL_CONF": "0.65",
        "YOLO_TEMPORAL_RADIUS_M": "0.60",
        "YOLO_INFERENCE_BATCH": "1",
    }
    environment.update({key: os.environ.get(key, value) for key, value in defaults.items()})
    data_root = find_wildtrack_root(environment["DROP_RATIOS"])
    pseudo_dir = WORK_ROOT / "pseudo_labels" / "wildtrack_yolo26x_pixelwise_max"
    environment["DATA_PATH"] = str(data_root)
    environment["PSEUDO_DIR"] = str(pseudo_dir)
    environment["YOLO_MODEL"] = ensure_yolo_model(environment["YOLO_MODEL"])
    save_manifest(data_root, environment)
    log(
        "Training pixelwise-max pseudo_only with global batch "
        f"{environment['BATCH_SIZE']} on CUDA {environment['GPU']}"
    )
    try:
        run(["bash", "train_grid_pseudo_only.sh", "--log_interval", "10"], cwd=mvdet_root, env=environment)
    finally:
        collect_outputs(mvdet_root, pseudo_dir)
    log("Two-GPU pixelwise-max experiment completed successfully")


if __name__ == "__main__":
    main()
