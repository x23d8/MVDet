import tempfile
import unittest
from pathlib import Path

from multiview_detector.datasets.path_utils import detect_dataset_root, resolve_annotation_dirs


DATASET_FILES = {
    'wildtrack': (
        'calibrations/intrinsic_zero/intr_CVLab1.xml',
        'calibrations/extrinsic/extr_CVLab1.xml',
    ),
    'multiviewx': (
        'calibrations/intrinsic/intr_Camera1.xml',
        'calibrations/extrinsic/extr_Camera1.xml',
    ),
}


def make_dataset(root, dataset_name):
    root.mkdir(parents=True)
    (root / 'Image_subsets').mkdir()
    (root / 'annotations_positions').mkdir()
    for relative_path in DATASET_FILES[dataset_name]:
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


class DatasetDetectionTest(unittest.TestCase):
    def test_detects_partial_wildtrack_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'wildtrack-partial-40-percent'
            make_dataset(root, 'wildtrack')

            dataset_name, resolved_root = detect_dataset_root(root)

            self.assertEqual(dataset_name, 'wildtrack')
            self.assertEqual(Path(resolved_root), root.resolve())

    def test_detects_nested_multiviewx_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir) / 'kaggle-export'
            root = parent / 'version-2' / 'multiviewx-partial'
            make_dataset(root, 'multiviewx')

            dataset_name, resolved_root = detect_dataset_root(parent)

            self.assertEqual(dataset_name, 'multiviewx')
            self.assertEqual(Path(resolved_root), root.resolve())

    def test_rejects_unknown_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(FileNotFoundError, 'Could not detect'):
                detect_dataset_root(temp_dir)

    def test_rejects_parent_containing_both_datasets(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            make_dataset(parent / 'wildtrack', 'wildtrack')
            make_dataset(parent / 'multiviewx', 'multiviewx')

            with self.assertRaisesRegex(ValueError, 'multiple datasets'):
                detect_dataset_root(parent)

    def test_dataset_choice_disambiguates_shared_parent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            parent = Path(temp_dir)
            make_dataset(parent / 'Wildtrack', 'wildtrack')
            make_dataset(parent / 'MultiviewX', 'multiviewx')

            dataset_name, resolved_root = detect_dataset_root(
                parent,
                dataset_name='multiviewx',
            )

            self.assertEqual(dataset_name, 'multiviewx')
            self.assertEqual(Path(resolved_root), (parent / 'MultiviewX').resolve())


class PartialAnnotationPathTest(unittest.TestCase):
    def test_pa_zero_uses_full_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Wildtrack'
            make_dataset(root, 'wildtrack')

            annotation_dir, hidden_dir = resolve_annotation_dirs(root, 'wildtrack', 0)

            self.assertEqual(Path(annotation_dir), (root / 'annotations_positions').resolve())
            self.assertIsNone(hidden_dir)

    def test_resolves_sibling_dropped_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Wildtrack'
            make_dataset(root, 'wildtrack')
            setting_root = Path(temp_dir) / 'Wildtrack_dropped' / 'drop20'
            (setting_root / 'annotations_positions').mkdir(parents=True)
            (setting_root / 'hidden_annotations_positions').mkdir()

            annotation_dir, hidden_dir = resolve_annotation_dirs(root, 'wildtrack', 20)

            self.assertEqual(Path(annotation_dir), (setting_root / 'annotations_positions').resolve())
            self.assertEqual(Path(hidden_dir), (setting_root / 'hidden_annotations_positions').resolve())

    def test_resolves_dropped_annotations_from_supplied_search_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / 'dataset-bundle'
            root = bundle / 'complete' / 'Wildtrack'
            make_dataset(root, 'wildtrack')
            setting_root = bundle / 'generated' / 'Wildtrack_dropped' / 'drop60'
            (setting_root / 'annotations_positions').mkdir(parents=True)
            (setting_root / 'hidden_annotations_positions').mkdir()

            annotation_dir, hidden_dir = resolve_annotation_dirs(
                root,
                'wildtrack',
                60,
                search_root=bundle,
            )

            self.assertEqual(Path(annotation_dir), (setting_root / 'annotations_positions').resolve())
            self.assertEqual(Path(hidden_dir), (setting_root / 'hidden_annotations_positions').resolve())

    def test_resolves_separate_kaggle_inputs_with_duplicate_wrapper(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            kaggle_input = Path(temp_dir) / 'input'
            complete_input = kaggle_input / 'multiviewx-3d'
            dropped_input = kaggle_input / 'thesis-dataset'
            root = complete_input / 'MultiviewX'
            make_dataset(root, 'multiviewx')
            setting_root = (
                dropped_input
                / 'MultiviewX_dropped'
                / 'MultiviewX_dropped'
                / 'drop45'
            )
            (setting_root / 'annotations_positions').mkdir(parents=True)
            (setting_root / 'hidden_annotations_positions').mkdir()

            dataset_name, dataset_root = detect_dataset_root(
                complete_input,
                dataset_name='multiviewx',
            )
            annotation_dir, hidden_dir = resolve_annotation_dirs(
                dataset_root,
                dataset_name,
                45,
                search_root=complete_input,
                dropped_path=dropped_input,
            )

            self.assertEqual(Path(annotation_dir), (setting_root / 'annotations_positions').resolve())
            self.assertEqual(Path(hidden_dir), (setting_root / 'hidden_annotations_positions').resolve())

    def test_dropped_path_may_point_directly_to_setting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'complete' / 'MultiviewX'
            make_dataset(root, 'multiviewx')
            setting_root = Path(temp_dir) / 'partial' / 'drop20'
            (setting_root / 'annotations_positions').mkdir(parents=True)
            (setting_root / 'hidden_annotations_positions').mkdir()

            annotation_dir, hidden_dir = resolve_annotation_dirs(
                root,
                'multiviewx',
                20,
                dropped_path=setting_root,
            )

            self.assertEqual(Path(annotation_dir), (setting_root / 'annotations_positions').resolve())
            self.assertEqual(Path(hidden_dir), (setting_root / 'hidden_annotations_positions').resolve())

    def test_resolves_pa_45_annotations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / 'Wildtrack'
            make_dataset(root, 'wildtrack')
            setting_root = Path(temp_dir) / 'Wildtrack_dropped' / 'drop45'
            (setting_root / 'annotations_positions').mkdir(parents=True)
            (setting_root / 'hidden_annotations_positions').mkdir()

            annotation_dir, hidden_dir = resolve_annotation_dirs(root, 'wildtrack', 45)

            self.assertEqual(Path(annotation_dir), (setting_root / 'annotations_positions').resolve())
            self.assertEqual(Path(hidden_dir), (setting_root / 'hidden_annotations_positions').resolve())


if __name__ == '__main__':
    unittest.main()
