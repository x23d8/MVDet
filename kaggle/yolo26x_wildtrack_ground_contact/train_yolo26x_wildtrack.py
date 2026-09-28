"""Fine-tune YOLO26x-pose for a single Wildtrack ground-contact keypoint.

Wildtrack's BEV positionID is projected into every calibrated camera.  The
resulting image point is supervised as one pose keypoint, while the native
per-camera annotation supplies the person bounding box.  All cameras for a
timestamp stay in the same temporal train/validation split.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import torch


INPUT_ROOT = Path(os.environ.get("INPUT_ROOT", "/kaggle/input/thesis-dataset-new"))
WILDTRACK_ROOT = Path(
    os.environ.get(
        "WILDTRACK_ROOT",
        "/kaggle/input/thesis-dataset-new/Wildtrack/Wildtrack",
    )
)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "/kaggle/temp/wildtrack_ground_contact"))
OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", "/kaggle/working"))
RUNS_ROOT = OUTPUT_ROOT / "runs"
MODEL_NAME = os.environ.get("MODEL_NAME", "yolo26x-pose.pt")
EPOCHS = int(os.environ.get("EPOCHS", "30"))
IMAGE_SIZE = int(os.environ.get("IMAGE_SIZE", "960"))
VAL_FRACTION = float(os.environ.get("VAL_FRACTION", "0.20"))
SEED = int(os.environ.get("SEED", "42"))
DEVICE = os.environ.get("DEVICE", "auto")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "0"))
WORKERS = int(os.environ.get("WORKERS", "4"))
OPTIMIZER = os.environ.get("OPTIMIZER", "AdamW")
LR0 = float(os.environ.get("LR0", "0.001"))
PATIENCE = int(os.environ.get("PATIENCE", "8"))
FREEZE = int(os.environ.get("FREEZE", "10"))
CLOSE_MOSAIC = int(os.environ.get("CLOSE_MOSAIC", "5"))
PREDICT_CONF = float(os.environ.get("PREDICT_CONF", "0.25"))


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


COS_LR = env_flag("COS_LR", True)
AMP = env_flag("AMP", True)
CACHE = env_flag("CACHE", False)
DETERMINISTIC = env_flag("DETERMINISTIC", True)

INTRINSIC_FILES = [
    "intr_CVLab1.xml",
    "intr_CVLab2.xml",
    "intr_CVLab3.xml",
    "intr_CVLab4.xml",
    "intr_IDIAP1.xml",
    "intr_IDIAP2.xml",
    "intr_IDIAP3.xml",
]
EXTRINSIC_FILES = [
    "extr_CVLab1.xml",
    "extr_CVLab2.xml",
    "extr_CVLab3.xml",
    "extr_CVLab4.xml",
    "extr_IDIAP1.xml",
    "extr_IDIAP2.xml",
    "extr_IDIAP3.xml",
]


def find_wildtrack_root() -> Path:
    def is_complete(root: Path) -> bool:
        required = [
            root / "annotations_positions",
            root / "Image_subsets" / "C1",
            root / "Image_subsets" / "C7",
            root / "calibrations" / "intrinsic_zero",
            root / "calibrations" / "extrinsic",
        ]
        return all(path.is_dir() for path in required)

    if is_complete(WILDTRACK_ROOT):
        print(f"Wildtrack root: {WILDTRACK_ROOT}", flush=True)
        return WILDTRACK_ROOT

    candidates = []
    for annotation_dir in INPUT_ROOT.rglob("annotations_positions"):
        root = annotation_dir.parent
        if is_complete(root):
            candidates.append(root)
    if not candidates:
        raise FileNotFoundError(
            f"Wildtrack is incomplete. Checked explicit root {WILDTRACK_ROOT} "
            f"and scanned below {INPUT_ROOT}"
        )
    candidates.sort(key=lambda path: ("wildtrack" not in str(path).lower(), len(path.parts)))
    print("Wildtrack root candidates:", [str(path) for path in candidates])
    return candidates[0]


def load_calibration(root: Path, camera: int) -> tuple[np.ndarray, np.ndarray]:
    intrinsic_path = root / "calibrations" / "intrinsic_zero" / INTRINSIC_FILES[camera]
    storage = cv2.FileStorage(str(intrinsic_path), cv2.FILE_STORAGE_READ)
    intrinsic = storage.getNode("camera_matrix").mat()
    storage.release()
    if intrinsic is None:
        raise ValueError(f"Could not read camera_matrix from {intrinsic_path}")

    extrinsic_path = root / "calibrations" / "extrinsic" / EXTRINSIC_FILES[camera]
    xml_root = ET.parse(extrinsic_path).getroot()
    rvec = np.asarray([float(value) for value in xml_root.find("rvec").text.split()], dtype=np.float64)
    tvec = np.asarray([float(value) for value in xml_root.find("tvec").text.split()], dtype=np.float64)
    rotation, _ = cv2.Rodrigues(rvec)
    extrinsic = np.hstack((rotation, tvec.reshape(3, 1)))
    return intrinsic.astype(np.float64), extrinsic.astype(np.float64)


def project_position(position_id: int, intrinsic: np.ndarray, extrinsic: np.ndarray) -> tuple[float, float] | None:
    grid_x = position_id % 480
    grid_y = position_id // 480
    world_x_cm = -300.0 + 2.5 * grid_x
    world_y_cm = -900.0 + 2.5 * grid_y
    point = np.asarray([world_x_cm, world_y_cm, 0.0, 1.0], dtype=np.float64)
    homogeneous = intrinsic @ extrinsic @ point
    if not np.isfinite(homogeneous).all() or homogeneous[2] <= 1e-8:
        return None
    return float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2])


def image_index(root: Path) -> tuple[list[dict[str, Path]], list[tuple[int, int]]]:
    indices = []
    shapes = []
    for camera in range(7):
        folder = root / "Image_subsets" / f"C{camera + 1}"
        paths = {
            path.stem: path
            for path in folder.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        }
        if not paths:
            raise FileNotFoundError(f"No images found in {folder}")
        sample = cv2.imread(str(next(iter(paths.values()))), cv2.IMREAD_COLOR)
        if sample is None:
            raise ValueError(f"Could not read an image from {folder}")
        indices.append(paths)
        shapes.append((sample.shape[0], sample.shape[1]))
    return indices, shapes


def normalized_pose_label(
    view: dict,
    projected_point: tuple[float, float] | None,
    image_height: int,
    image_width: int,
) -> str | None:
    x1 = float(view.get("xmin", -1))
    y1 = float(view.get("ymin", -1))
    x2 = float(view.get("xmax", -1))
    y2 = float(view.get("ymax", -1))
    if x1 < 0 or x2 <= x1 or y2 <= y1:
        return None

    x1 = float(np.clip(x1, 0, image_width - 1))
    x2 = float(np.clip(x2, 0, image_width - 1))
    y1 = float(np.clip(y1, 0, image_height - 1))
    y2 = float(np.clip(y2, 0, image_height - 1))
    if x2 <= x1 or y2 <= y1:
        return None

    center_x = ((x1 + x2) / 2.0) / image_width
    center_y = ((y1 + y2) / 2.0) / image_height
    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    keypoint_x = keypoint_y = 0.0
    visibility = 0
    if projected_point is not None:
        point_x, point_y = projected_point
        if 0 <= point_x < image_width and 0 <= point_y < image_height:
            keypoint_x = point_x / image_width
            keypoint_y = point_y / image_height
            visibility = 2

    values = [
        0,
        center_x,
        center_y,
        width,
        height,
        keypoint_x,
        keypoint_y,
        visibility,
    ]
    return " ".join(str(value) if isinstance(value, int) else f"{value:.8f}" for value in values)


def prepare_dataset(root: Path) -> tuple[Path, dict]:
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    for split in ("train", "val"):
        (DATA_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATA_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    annotations = sorted((root / "annotations_positions").glob("*.json"))
    image_paths, image_shapes = image_index(root)
    available_stems = set.intersection(*(set(index) for index in image_paths))
    annotations = [path for path in annotations if path.stem in available_stems]
    if len(annotations) < 2:
        raise RuntimeError("Need at least two synchronized annotated Wildtrack frames")

    val_count = max(1, round(len(annotations) * VAL_FRACTION))
    val_count = min(val_count, len(annotations) - 1)
    split_index = len(annotations) - val_count
    calibrations = [load_calibration(root, camera) for camera in range(7)]
    stats = {
        "wildtrack_root": str(root),
        "split_strategy": "last contiguous temporal block for validation",
        "train_frames": split_index,
        "val_frames": val_count,
        "train_images": 0,
        "val_images": 0,
        "boxes": 0,
        "visible_ground_keypoints": 0,
        "missing_ground_keypoints": 0,
    }

    for annotation_index, annotation_path in enumerate(annotations):
        split = "train" if annotation_index < split_index else "val"
        people = json.loads(annotation_path.read_text(encoding="utf-8"))
        for camera in range(7):
            source_image = image_paths[camera][annotation_path.stem]
            image_name = f"C{camera + 1}_{annotation_path.stem}{source_image.suffix.lower()}"
            target_image = DATA_ROOT / "images" / split / image_name
            target_image.symlink_to(source_image)
            height, width = image_shapes[camera]
            intrinsic, extrinsic = calibrations[camera]
            lines = []
            for person in people:
                view_by_camera = {
                    int(view["viewNum"]): view for view in person.get("views", [])
                }
                view = view_by_camera.get(camera)
                if view is None:
                    continue
                projected = project_position(int(person["positionID"]), intrinsic, extrinsic)
                line = normalized_pose_label(view, projected, height, width)
                if line is None:
                    continue
                lines.append(line)
                stats["boxes"] += 1
                if line.endswith(" 2"):
                    stats["visible_ground_keypoints"] += 1
                else:
                    stats["missing_ground_keypoints"] += 1

            label_path = DATA_ROOT / "labels" / split / f"{Path(image_name).stem}.txt"
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            stats[f"{split}_images"] += 1

    yaml_path = DATA_ROOT / "wildtrack_ground_contact.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {DATA_ROOT}",
                "train: images/train",
                "val: images/val",
                "names:",
                "  0: person",
                "kpt_shape: [1, 3]",
                "flip_idx: [0]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path, stats


def ensure_ultralytics() -> None:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", "--upgrade", "ultralytics==8.4.69"]
    )


def main() -> None:
    if not 0 < VAL_FRACTION < 1:
        raise ValueError("VAL_FRACTION must be between 0 and 1")
    if EPOCHS <= 0 or IMAGE_SIZE <= 0:
        raise ValueError("EPOCHS and IMAGE_SIZE must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("This kernel requires a Kaggle GPU accelerator")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    ensure_ultralytics()
    from ultralytics import YOLO
    import ultralytics

    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(index) for index in range(gpu_count)]
    device = ",".join(str(index) for index in range(gpu_count)) if DEVICE == "auto" else DEVICE
    batch = gpu_count if BATCH_SIZE <= 0 else BATCH_SIZE
    print("GPU devices:", gpu_names)
    print(f"Ultralytics={ultralytics.__version__}, device={device}, global_batch={batch}")

    wildtrack_root = find_wildtrack_root()
    data_yaml, data_stats = prepare_dataset(wildtrack_root)
    run_config = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": batch,
        "device": device,
        "seed": SEED,
        "workers": WORKERS,
        "optimizer": OPTIMIZER,
        "lr0": LR0,
        "gpus": gpu_names,
        "ultralytics": ultralytics.__version__,
        "dataset": data_stats,
    }
    (OUTPUT_ROOT / "training_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )
    print(json.dumps(run_config, indent=2))

    model = YOLO(MODEL_NAME)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=batch,
        device=device,
        workers=WORKERS,
        project=str(RUNS_ROOT),
        name="yolo26x_wildtrack_ground_contact",
        exist_ok=True,
        pretrained=True,
        optimizer=OPTIMIZER,
        lr0=LR0,
        cos_lr=COS_LR,
        patience=PATIENCE,
        amp=AMP,
        cache=CACHE,
        freeze=FREEZE,
        close_mosaic=CLOSE_MOSAIC,
        plots=True,
        seed=SEED,
        deterministic=DETERMINISTIC,
        verbose=True,
    )

    run_dir = RUNS_ROOT / "yolo26x_wildtrack_ground_contact"
    best_weights = run_dir / "weights" / "best.pt"
    if not best_weights.is_file():
        raise FileNotFoundError(f"Training finished without {best_weights}")
    stable_weights = OUTPUT_ROOT / "yolo26x_wildtrack_ground_contact_best.pt"
    shutil.copy2(best_weights, stable_weights)

    validation_images = sorted((DATA_ROOT / "images" / "val").iterdir())[:12]
    best_model = YOLO(str(best_weights))
    best_model.predict(
        source=[str(path) for path in validation_images],
        imgsz=IMAGE_SIZE,
        conf=PREDICT_CONF,
        device=0,
        save=True,
        project=str(OUTPUT_ROOT),
        name="validation_preview",
        exist_ok=True,
        verbose=False,
    )
    print(f"Best checkpoint: {stable_weights}")


if __name__ == "__main__":
    main()
