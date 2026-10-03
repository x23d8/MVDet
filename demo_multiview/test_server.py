import json
import tempfile
import unittest
from pathlib import Path

from server import Dataset, grid_position, map_point, read_predictions
from inference import PomIndex, load_model_specs


class DatasetTest(unittest.TestCase):
    def test_coordinate_conventions(self):
        self.assertEqual(grid_position("wildtrack", 12, 40), 19212)
        self.assertEqual(map_point("wildtrack", 12, 40), [40 / 1440, 12 / 480])
        self.assertEqual(grid_position("multiviewx", 12, 40), 40012)
        self.assertEqual(map_point("multiviewx", 12, 40), [12 / 1000, 40 / 640])

    def test_prediction_and_annotation_have_distinct_boxes_and_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "annotations_positions").mkdir()
            for camera in range(1, 7):
                folder = root / "Image_subsets" / f"C{camera}"
                folder.mkdir(parents=True)
                (folder / "00000000.png").write_bytes(b"image")
            annotation = [{"personID": 42, "positionID": 40012,
                           "views": [{"viewNum": 0, "xmin": 100, "ymin": 200,
                                      "xmax": 300, "ymax": 500}]}]
            (root / "annotations_positions" / "00000000.json").write_text(json.dumps(annotation))
            (root / "rectangles.pom").write_text("RECTANGLE 0 40012 110 210 310 510\n")
            prediction = root / "test.txt"
            prediction.write_text("0 12 40\n")
            dataset = Dataset("multiviewx", root, prediction)
            self.assertEqual(dataset.frames, [0])
            model = dataset.frame(0, "model")["nodes"][0]
            gt = dataset.frame(0, "annotations")["nodes"][0]
            self.assertEqual(model["id"], "D001")
            self.assertEqual(model["boxes"][0]["bbox"], [110, 210, 310, 510])
            self.assertEqual(model["boxes"][0]["kind"], "projected")
            self.assertEqual(gt["id"], "42")
            self.assertEqual(gt["boxes"][0]["bbox"], [100, 200, 300, 500])
            self.assertEqual(gt["boxes"][0]["kind"], "annotation")

    def test_invalid_prediction_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "test.txt"
            path.write_text("0 9999 40\n")
            with self.assertRaises(ValueError):
                read_predictions(path, "multiviewx")

    def test_sparse_pom_lookup_and_model_registry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pom = root / "rectangles.pom"
            pom.write_text("WIDTH 12\nRECTANGLE 0 0 notvisible\nRECTANGLE 0 2 10 20 30 40\n"
                           "RECTANGLE 1 0 notvisible\nRECTANGLE 1 2 50 60 70 80\n")
            index = PomIndex(pom, stride=1)
            self.assertEqual(index.boxes(2), {0: [10, 20, 30, 40], 1: [50, 60, 70, 80]})
            self.assertEqual(index.boxes(1), {})
            weight = root / "second.pth"
            weight.write_bytes(b"placeholder")
            config = root / "models.json"
            config.write_text(json.dumps([{"id": "second", "method": "Second method",
                                           "name": "second.pth", "dataset": "wildtrack",
                                           "backend": "mvdet", "weights": str(weight)}]))
            specs = load_model_specs(config, None)
            self.assertEqual(specs[0]["id"], "second")


if __name__ == "__main__":
    unittest.main()
