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


def find_wildtrack_root() -> Path:
    def is_complete(root: Path) -> bool:
        required = [
            root / "annotations_positions",
            root / "Image_subsets" / "C1",
            root / "Image_subsets" / "C7",
            root / "calibrations" / "intrinsic_zero",
            root / "calibrations" / "extrinsic",
            root / "drop_annotations" / "drop_45" / "annotations_positions",
        ]
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
            f"Wildtrack is incomplete. Checked explicit root {WILDTRACK_ROOT} "
            "and scanned below /kaggle/input"
        )
    candidates.sort(key=lambda path: ("wildtrack" not in str(path).lower(), len(path.parts)))
    log("Wildtrack candidates: " + json.dumps([str(path) for path in candidates]))
    return candidates[0]


def configure_yolo_bbox_detector() -> None:
    """Force the cloned branch to use detection boxes, never pose keypoints."""
    generator = PROJECT_ROOT / "SHOTBRL" / "tools" / "generate_yolo_pose_bev_pseudo.py"
    if not generator.is_file():
        raise FileNotFoundError(f"YOLO pseudo-label generator not found: {generator}")
    source = generator.read_text(encoding="utf-8")
    legacy_keypoint_setting = '                "keypoint_conf": args.kpt_conf,\n'
    if "def extract_bbox_predictions(result):" in source:
        if legacy_keypoint_setting in source:
            source = source.replace(legacy_keypoint_setting, "", 1)
            generator.write_text(source, encoding="utf-8")
            log("Removed stale keypoint_conf reference from detection-only generator")
        log("YOLO generator already uses detection-only bbox footprints")
        return

    bbox_extractor = '''def extract_bbox_predictions(result):
    """Convert person detections to TrackTacular-style bottom-center feet."""
    if result.boxes is None:
        return []
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    scores = result.boxes.conf.detach().cpu().numpy()
    predictions = []
    for box, score in zip(boxes, scores):
        foot = np.asarray([(box[0] + box[2]) * 0.5, box[3]], dtype=np.float32)
        predictions.append({
            "box": box.astype(np.float32),
            "box_conf": float(score),
            "foot": foot,
            "foot_conf": 1.0,
            "confidence": float(score),
            "source": "bbox_bottom",
        })
    return predictions
'''
    replacements = [
        ("from yolo_pose_foot_demo import extract_predictions", bbox_extractor),
        ('parser.add_argument("--model", default="yolo26m-pose.pt")',
         'parser.add_argument("--model", default="yolo26x.pt")'),
        ('default="pose_x_bbox_y",', 'default="bbox_bottom",'),
        ("                conf=args.conf,\n                device=args.device,",
         "                conf=args.conf,\n                classes=[0],\n                device=args.device,"),
        ("predictions = extract_predictions(result, args.kpt_conf)",
         "predictions = extract_bbox_predictions(result)"),
    ]
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"Cannot configure detection-only YOLO; missing source fragment: {old!r}")
        source = source.replace(old, new, 1)
    source = source.replace(legacy_keypoint_setting, "", 1)
    generator.write_text(source, encoding="utf-8")
    log("Configured YOLO26x detection-only footprints: ((x1+x2)/2, y2), classes=[0]")


def prepare_project() -> Path:
    if PROJECT_ROOT.exists():
        shutil.rmtree(PROJECT_ROOT)
    PROJECT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    log(f"Cloning {REPO_URL} branch {REPO_BRANCH}")
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            REPO_BRANCH,
            REPO_URL,
            str(PROJECT_ROOT),
        ]
    )
    mvdet_root = PROJECT_ROOT / REPO_SUBDIR
    if not (mvdet_root / "main.py").is_file():
        raise FileNotFoundError(
            f"Invalid repository layout: {mvdet_root / 'main.py'} not found"
        )
    configure_yolo_bbox_detector()
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


def ensure_yolo_model(preferred: str, fallback: str) -> str:
    preferred_path = Path(preferred).expanduser()
    if preferred_path.is_file():
        log(f"Using YOLO checkpoint: {preferred_path}")
        return str(preferred_path)

    # A plain Ultralytics model name is itself a valid downloadable source.
    selected = preferred
    if preferred_path.is_absolute() or preferred_path.parent != Path("."):
        selected = fallback
        log(
            f"Preferred YOLO checkpoint is missing: {preferred_path}. "
            f"Falling back to automatic Ultralytics download: {selected}"
        )

    from ultralytics import YOLO

    previous_cwd = Path.cwd()
    try:
        os.chdir(WORK_ROOT)
        YOLO(selected)
    finally:
        os.chdir(previous_cwd)

    downloaded = WORK_ROOT / Path(selected).name
    if downloaded.is_file():
        log(f"YOLO fallback ready: {downloaded}")
        return str(downloaded)

    # Ultralytics may resolve the model through its cache even when no file is
    # materialized directly below /kaggle/working.
    log(f"YOLO fallback resolved by Ultralytics: {selected}")
    return selected


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
        "YOLO_FALLBACK_MODEL",
        "YOLO_IMGSZ",
        "YOLO_CONF",
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
        "repository": {
            "url": REPO_URL,
            "branch": REPO_BRANCH,
            "subdir": REPO_SUBDIR,
        },
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
    pseudo_dir = WORK_ROOT / "pseudo_labels" / "wildtrack_yolo26x_bbox_temporal"

    environment = os.environ.copy()
    defaults = {
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
        "YOLO_MODEL": "yolo26x.pt",
        "YOLO_FALLBACK_MODEL": "yolo26x.pt",
        "YOLO_IMGSZ": "1280",
        "YOLO_DEVICE": "0",
        "YOLO_CONF": "0.20",
        "YOLO_CANDIDATE_CONF": "0.20",
        "YOLO_FOOT_ANCHOR": "bbox_bottom",
        "YOLO_MIN_VIEWS": "2",
        "YOLO_MERGE_RADIUS_M": "0.60",
        "YOLO_TEMPORAL_SINGLETONS": "1",
        "YOLO_TEMPORAL_CONF": "0.65",
        "YOLO_TEMPORAL_RADIUS_M": "0.60",
        "YOLO_INFERENCE_BATCH": "1",
    }
    environment.update({key: os.environ.get(key, str(value)) for key, value in defaults.items()})
    environment["YOLO_MODEL"] = ensure_yolo_model(
        environment["YOLO_MODEL"], environment["YOLO_FALLBACK_MODEL"]
    )
    save_run_manifest(data_root, environment)
    log(
        "Grid: pseudo_only, "
        f"lambda={environment['LAMBDAS']}, drop={environment['DROP_RATIOS']}, "
        f"epochs={environment['EPOCHS']}"
    )
    try:
        run(["bash", "train_grid_pseudo_only.sh", "--log_interval", "10"], cwd=mvdet_root, env=environment)
    finally:
        collect_outputs(mvdet_root, pseudo_dir)
    log("MVDet temporal-pseudo grid completed successfully")


if __name__ == "__main__":
    main()
