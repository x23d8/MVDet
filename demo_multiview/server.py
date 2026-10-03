"""Standalone, read-only MultiviewX/Wildtrack detection viewer."""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
import re
from collections import defaultdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from inference import MVDetPredictor, load_model_specs

PREDICTOR_BACKENDS = {"mvdet": MVDetPredictor}


HERE = Path(__file__).resolve().parent
SPECS = {
    "wildtrack": {"name": "Wildtrack", "width": 1440, "height": 480, "pos_width": 480, "cameras": 7},
    "multiviewx": {"name": "MultiviewX", "width": 1000, "height": 640, "pos_width": 1000, "cameras": 6},
}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp"}
POM_LINE = re.compile(r"\bRECTANGLE\s+(\d+)\s+(\d+)(?:\s+(-?\d+)\s+(-?\d+)\s+(\d+)\s+(\d+))?", re.I)


def resolve_root(path: str | None, key: str) -> Path | None:
    if not path:
        return None
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        return None
    candidates = [root]
    candidates += [child for child in root.iterdir() if child.is_dir()]
    candidates += [grand for child in candidates[1:] for grand in child.iterdir() if grand.is_dir()]
    for candidate in candidates:
        if (candidate / "Image_subsets").is_dir() and (candidate / "annotations_positions").is_dir():
            return candidate
    return None


def grid_position(key: str, x: float, y: float) -> int:
    spec = SPECS[key]
    return round(x) + round(y) * spec["pos_width"]


def map_point(key: str, x: float, y: float) -> list[float]:
    spec = SPECS[key]
    if key == "wildtrack":
        return [y / spec["width"], x / spec["height"]]
    return [x / spec["width"], y / spec["height"]]


def point_from_pos(key: str, position: int) -> tuple[int, int]:
    width = SPECS[key]["pos_width"]
    return position % width, position // width


def valid_point(key: str, x: float, y: float) -> bool:
    if not all(map(math.isfinite, (x, y))):
        return False
    px, py = map_point(key, x, y)
    return 0 <= px < 1 and 0 <= py < 1


def read_predictions(path: Path | None, key: str) -> dict[int, list[tuple[float, float]]]:
    frames = defaultdict(list)
    if path is None:
        return frames
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            parts = line.strip().split()
            if not parts or line.lstrip().startswith("#"):
                continue
            if len(parts) < 3:
                raise ValueError(f"{path}:{line_number}: expected frame x y")
            try:
                frame, x, y = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: expected numeric frame x y") from error
            if not frame.is_integer() or not valid_point(key, x, y):
                raise ValueError(f"{path}:{line_number}: invalid frame or ground-plane point")
            frames[int(frame)].append((x, y))
    return dict(frames)


def frame_files(root: Path) -> dict[int, Path]:
    result = {}
    for file in (root / "annotations_positions").glob("*.json"):
        if file.stem.isdigit():
            result[int(file.stem)] = file
    return result


def camera_files(root: Path, camera: int) -> dict[int, Path]:
    result = {}
    folder = root / "Image_subsets" / f"C{camera + 1}"
    if not folder.is_dir():
        return result
    for file in folder.iterdir():
        if file.suffix.lower() in IMAGE_TYPES and file.stem.isdigit():
            result[int(file.stem)] = file
    return result


def valid_box(view: dict) -> list[float] | None:
    try:
        box = [float(view[name]) for name in ("xmin", "ymin", "xmax", "ymax")]
    except (KeyError, TypeError, ValueError):
        return None
    if not all(map(math.isfinite, box)) or min(box) < 0 or box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def read_pom_boxes(path: Path, positions: set[int]) -> dict[int, dict[int, list[int]]]:
    """Keep only POM rows needed by model predictions, avoiding a huge index."""
    boxes = defaultdict(dict)
    if not path.is_file() or not positions:
        return boxes
    with path.open(encoding="utf-8", errors="replace") as source:
        for line in source:
            match = POM_LINE.search(line)
            if not match:
                continue
            camera, position = int(match[1]), int(match[2])
            if position not in positions or match[3] is None:
                continue
            left, top, right, bottom = map(int, match.group(3, 4, 5, 6))
            if right > left and bottom > top:
                boxes[position][camera] = [max(0, left), max(0, top), min(1919, right), min(1079, bottom)]
    return dict(boxes)


class Dataset:
    def __init__(self, key: str, root: Path, prediction_path: Path | None):
        self.key, self.root = key, root
        self.annotations = frame_files(root)
        self.cameras = [camera_files(root, camera) for camera in range(SPECS[key]["cameras"])]
        self.predictions = read_predictions(prediction_path, key)
        self.prediction_path = prediction_path
        self.frames = sorted(set(self.annotations) & set.intersection(*(set(files) for files in self.cameras)))
        positions = {grid_position(key, x, y) for points in self.predictions.values() for x, y in points}
        self.pom = read_pom_boxes(root / "rectangles.pom", positions)

    @lru_cache(maxsize=12)
    def frame(self, frame: int, source: str) -> dict:
        if frame not in self.frames:
            raise ValueError("Frame unavailable")
        if source not in ("model", "annotations"):
            raise ValueError("Unknown source")
        with self.annotations[frame].open(encoding="utf-8") as file:
            annotations = json.load(file)
        if not isinstance(annotations, list):
            raise ValueError("Invalid annotation JSON")
        nodes = []
        if source == "model":
            for index, (x, y) in enumerate(self.predictions.get(frame, []), 1):
                position = grid_position(self.key, x, y)
                boxes = [{"camera": camera, "bbox": box, "kind": "projected"}
                         for camera, box in sorted(self.pom.get(position, {}).items())]
                nodes.append({"id": f"D{index:03d}", "x": x, "y": y,
                              "point": map_point(self.key, x, y), "boxes": boxes,
                              "position": position})
        else:
            for person in annotations:
                try:
                    position = int(person["positionID"])
                    person_id = int(person["personID"])
                except (KeyError, TypeError, ValueError):
                    continue
                x, y = point_from_pos(self.key, position)
                if not valid_point(self.key, x, y):
                    continue
                boxes = []
                for index, view in enumerate(person.get("views", [])):
                    box = valid_box(view)
                    if box:
                        boxes.append({"camera": int(view.get("viewNum", index)), "bbox": box,
                                      "kind": "annotation"})
                nodes.append({"id": str(person_id), "x": x, "y": y,
                              "point": map_point(self.key, x, y), "boxes": boxes,
                              "position": position})
        return {"dataset": self.key, "frame": frame, "source": source,
                "nodes": nodes, "cameras": len(self.cameras),
                "bbox_kind": "projected" if source == "model" else "annotation"}


def make_handler(datasets: dict[str, Dataset], models: dict[str, MVDetPredictor] | None = None):
    models = models or {}
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store" if content_type.startswith("application/json") else "public, max-age=3600")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value, status=200):
            self.send_bytes(json.dumps(value).encode("utf-8"), "application/json; charset=utf-8", status)

        def do_GET(self):
            parsed = urlsplit(self.path)
            params = parse_qs(parsed.query)
            try:
                if parsed.path == "/api/datasets":
                    return self.send_json([{"key": key, "name": SPECS[key]["name"],
                                            "cameras": len(ds.cameras), "frames": ds.frames,
                                            "has_predictions": bool(ds.prediction_path),
                                            "prediction_frames": len(ds.predictions),
                                            "has_pom": (ds.root / "rectangles.pom").is_file()}
                                           for key, ds in datasets.items()])
                if parsed.path == "/api/models":
                    return self.send_json([{key: value for key, value in predictor.spec.items()
                                            if key not in ("weights", "zip_member")}
                                           for predictor in models.values()])
                inference = re.fullmatch(r"/api/infer/([a-z0-9_-]+)/([a-z]+)/([0-9]+)", parsed.path)
                if inference:
                    model_id, key, frame_text = inference.groups()
                    if model_id not in models or key not in datasets or models[model_id].dataset is not datasets[key]:
                        return self.send_json({"error": "This model does not support the selected dataset."}, 404)
                    return self.send_json(models[model_id].infer(int(frame_text)))
                match = re.fullmatch(r"/api/(frame|image)/([a-z]+)/([0-9]+)(?:/([0-9]+))?", parsed.path)
                if match:
                    kind, key, frame_text, camera_text = match.groups()
                    if key not in datasets:
                        return self.send_json({"error": "Dataset unavailable"}, 404)
                    dataset, frame = datasets[key], int(frame_text)
                    if kind == "frame":
                        source = params.get("source", ["model" if dataset.prediction_path else "annotations"])[0]
                        return self.send_json(dataset.frame(frame, source))
                    camera = int(camera_text) if camera_text is not None else -1
                    if frame not in dataset.frames or camera < 0 or camera >= len(dataset.cameras):
                        return self.send_json({"error": "Image unavailable"}, 404)
                    file = dataset.cameras[camera][frame]
                    return self.send_bytes(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "application/octet-stream")
                static = {"/": "index.html", "/workspace.css": "workspace.css", "/app.js": "app.js"}
                if parsed.path in static:
                    file = HERE / static[parsed.path]
                    return self.send_bytes(file.read_bytes(), mimetypes.guess_type(file.name)[0] or "text/plain")
                self.send_json({"error": "Not found"}, 404)
            except (ValueError, OSError, json.JSONDecodeError) as error:
                self.send_json({"error": str(error)}, 400)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wildtrack", help="Wildtrack dataset root or containing folder")
    parser.add_argument("--multiviewx", help="MultiviewX dataset root or containing folder")
    parser.add_argument("--wildtrack-results", help="MVDet test.txt for Wildtrack")
    parser.add_argument("--multiviewx-results", help="MVDet test.txt for MultiviewX")
    parser.add_argument("--gaussian-archive", default=r"C:\Users\ADMIN\Documents\mvdet_yolo26x_results.zip",
                        help="ZIP containing gaussian_log/MultiviewDetector.pth")
    parser.add_argument("--models-config", help="Optional JSON registry for additional model weights")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    datasets = {}
    for key in SPECS:
        root = resolve_root(getattr(args, key), key)
        if root:
            results = getattr(args, f"{key}_results")
            result_path = Path(results).expanduser().resolve() if results else None
            datasets[key] = Dataset(key, root, result_path)
            print(f"{SPECS[key]['name']}: {len(datasets[key].frames)} synchronized frames at {root}")
        elif getattr(args, key):
            print(f"Dataset unavailable: {getattr(args, key)}")
    specs = load_model_specs(Path(args.models_config) if args.models_config else None,
                             Path(args.gaussian_archive) if args.gaussian_archive else None)
    models = {spec["id"]: PREDICTOR_BACKENDS[spec["backend"]](spec, datasets[spec["dataset"]])
              for spec in specs if spec["dataset"] in datasets}
    for predictor in models.values():
        print(f"Model: {predictor.spec['method']} / {predictor.spec['name']} ({predictor.dataset.key})")
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(datasets, models))
    print(f"Demo: http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
