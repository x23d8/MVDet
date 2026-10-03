"""Lazy, checkpoint-backed MVDet inference for a selected synchronized frame."""

from __future__ import annotations

import bisect
import base64
import io
import json
import sys
import threading
import zipfile
from pathlib import Path
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parent.parent
MODEL_SOURCE = PROJECT / "MVDetBRL"


class PomIndex:
    """Sparse on-disk index for the large rectangles.pom file."""

    def __init__(self, path: Path, stride: int = 512):
        self.path = path
        self.checkpoints = {}
        self.stride = stride
        if not path.is_file():
            return
        with path.open("rb") as source:
            while True:
                offset = source.tell()
                line = source.readline()
                if not line:
                    break
                if not line.startswith(b"RECTANGLE "):
                    continue
                parts = line.split(maxsplit=3)
                if len(parts) < 3:
                    continue
                try:
                    camera, position = int(parts[1]), int(parts[2])
                except ValueError:
                    continue
                checkpoints = self.checkpoints.setdefault(camera, [])
                if not checkpoints or position - checkpoints[-1][0] >= stride:
                    checkpoints.append((position, offset))

    def boxes(self, position: int) -> dict[int, list[int]]:
        boxes = {}
        if not self.checkpoints:
            return boxes
        with self.path.open("rb") as source:
            for camera, checkpoints in self.checkpoints.items():
                index = bisect.bisect_right(checkpoints, (position, float("inf"))) - 1
                if index < 0:
                    continue
                source.seek(checkpoints[index][1])
                for line in source:
                    if not line.startswith(b"RECTANGLE "):
                        continue
                    parts = line.split()
                    if len(parts) < 3:
                        continue
                    try:
                        line_camera, line_position = int(parts[1]), int(parts[2])
                    except ValueError:
                        continue
                    if line_camera != camera or line_position > position:
                        break
                    if line_position == position and len(parts) >= 7:
                        try:
                            left, top, right, bottom = map(int, parts[3:7])
                        except ValueError:
                            break
                        if right > left and bottom > top:
                            boxes[camera] = [max(0, left), max(0, top), min(1919, right), min(1079, bottom)]
                        break
        return boxes


class MVDetPredictor:
    def __init__(self, spec: dict, dataset):
        self.spec, self.dataset = spec, dataset
        self.model = None
        self.pom = None
        self.lock = threading.Lock()
        self.cache = {}

    def _load(self):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("This MVDet checkpoint requires CUDA, but no CUDA GPU is available.")
        if str(MODEL_SOURCE) not in sys.path:
            sys.path.insert(0, str(MODEL_SOURCE))
        from multiview_detector.datasets.Wildtrack import Wildtrack
        from multiview_detector.datasets.MultiviewX import MultiviewX
        from multiview_detector.models.persp_trans_detector import PerspTransDetector

        base = (Wildtrack if self.dataset.key == "wildtrack" else MultiviewX)(str(self.dataset.root))
        adapter = SimpleNamespace(base=base, num_cam=base.num_cam, img_shape=base.img_shape,
                                  reducedgrid_shape=[dimension // 4 for dimension in base.worldgrid_shape],
                                  grid_reduce=4, img_reduce=4)
        model = PerspTransDetector(adapter, self.spec.get("arch", "resnet18"))
        path = Path(self.spec["weights"])
        if self.spec.get("zip_member"):
            with zipfile.ZipFile(path) as archive:
                with archive.open(self.spec["zip_member"]) as weight_file:
                    weights = torch.load(io.BytesIO(weight_file.read()), map_location="cpu", weights_only=True)
        else:
            weights = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=True)
        model.eval()
        self.model = model
        self.pom = PomIndex(self.dataset.root / "rectangles.pom")

    def infer(self, frame: int) -> dict:
        if frame not in self.dataset.frames:
            raise ValueError("Frame unavailable")
        with self.lock:
            if frame in self.cache:
                return self.cache[frame]
            if self.model is None:
                self._load()
            import torch
            import torchvision.transforms as T
            from PIL import Image
            from multiview_detector.utils.nms import nms
            from server import grid_position, map_point, valid_point

            transform = T.Compose([T.Resize([720, 1280]), T.ToTensor(),
                                   T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            images = []
            for camera in self.dataset.cameras:
                with Image.open(camera[frame]) as image:
                    images.append(transform(image.convert("RGB")))
            batch = torch.stack(images).unsqueeze(0)
            with torch.inference_mode():
                heatmap, _ = self.model(batch)
            heatmap = heatmap.detach().cpu().squeeze()
            threshold = float(self.spec.get("threshold", 0.4))
            import numpy as np
            heat_array = heatmap.numpy()
            peak = max(float(np.max(heat_array)), threshold)
            intensity = np.clip((heat_array - threshold * 0.25) / max(peak - threshold * 0.25, 1e-6), 0, 1)
            rgba = np.empty((*intensity.shape, 4), dtype=np.uint8)
            rgba[..., 0] = 20 + intensity * 225
            rgba[..., 1] = 185 + intensity * 55
            rgba[..., 2] = 165 - intensity * 105
            rgba[..., 3] = intensity * 225
            heat_image = io.BytesIO()
            Image.fromarray(rgba, "RGBA").save(heat_image, format="PNG")
            heatmap_url = "data:image/png;base64," + base64.b64encode(heat_image.getvalue()).decode("ascii")
            rows = (heatmap > threshold).nonzero()
            scores = heatmap[heatmap > threshold]
            if self.dataset.key == "multiviewx":
                coordinates = rows[:, [1, 0]].float() * 4
            else:
                coordinates = rows.float() * 4
            keep, count = nms(coordinates, scores, 20, float("inf")) if len(scores) else ([], 0)
            nodes = []
            for index, score_index in enumerate(keep[:count], 1):
                x, y = map(float, coordinates[score_index].tolist())
                if not valid_point(self.dataset.key, x, y):
                    continue
                position = grid_position(self.dataset.key, x, y)
                boxes = [{"camera": camera, "bbox": box, "kind": "projected"}
                         for camera, box in sorted(self.pom.boxes(position).items())]
                nodes.append({"id": f"D{index:03d}", "x": x, "y": y,
                              "point": map_point(self.dataset.key, x, y), "boxes": boxes,
                              "position": position, "score": round(float(scores[score_index]), 4)})
            result = {"dataset": self.dataset.key, "frame": frame, "source": "inference",
                      "model": self.spec["id"], "nodes": nodes,
                      "cameras": len(self.dataset.cameras), "bbox_kind": "projected",
                      "heatmap": heatmap_url}
            self.cache[frame] = result
            return result


def load_model_specs(config_path: Path | None, gaussian_archive: Path | None) -> list[dict]:
    specs = []
    if gaussian_archive and gaussian_archive.is_file():
        specs.append({"id": "gaussian", "method": "Gaussian + YOLO pseudo",
                      "name": "MultiviewDetector.pth", "dataset": "wildtrack",
                      "backend": "mvdet", "arch": "resnet18", "threshold": 0.4,
                      "default_frame": 1800,
                      "weights": str(gaussian_archive),
                      "zip_member": "gaussian_log/MultiviewDetector.pth"})
    if config_path and config_path.is_file():
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("Model config must be a JSON list")
        for item in raw:
            if item.get("backend") != "mvdet" or item.get("dataset") not in ("wildtrack", "multiviewx"):
                raise ValueError("Unsupported model backend or dataset")
            if not Path(item["weights"]).is_file():
                raise FileNotFoundError(item["weights"])
            specs.append(item)
    if len({item["id"] for item in specs}) != len(specs):
        raise ValueError("Duplicate model id")
    return specs
