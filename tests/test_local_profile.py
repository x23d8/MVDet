import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from main import apply_run_profile, build_local_diagnostic, init_wandb
from multiview_detector.datasets.frameDataset import frameDataset
from multiview_detector.evaluation.pyeval.evaluateDetection import evaluateDetection_py
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.trainer import PerspectiveTrainer


class LocalProfileTest(unittest.TestCase):
    def test_disabled_wandb_does_not_import_or_initialize_wandb(self):
        args = SimpleNamespace(wandb_mode='disabled')

        with patch('builtins.__import__', side_effect=AssertionError('unexpected import')):
            run = init_wandb(args, None, None, None, 'unused')

        self.assertIsNone(run.log({'metric': 1.0}))
        self.assertIsNone(run.finish())

    def test_local_profile_activates_bounded_single_gpu_run(self):
        args = SimpleNamespace(
            run_profile='local_2ep',
            pa=45,
            variant='default',
            epochs=10,
            batch_size=4,
            num_workers=4,
            wandb_mode='online',
            device_mode='auto',
            input_height=720,
            input_width=1280,
            grid_reduce=4,
            img_reduce=4,
            max_train_frames=None,
            max_test_frames=None,
            skip_initial_eval=False,
            skip_final_eval=False,
            brl_warmup_epochs=1,
            brl_ramp_epochs=2,
            eval_thresholds=None,
            lr=0.1,
            grad_clip_norm=0.0,
            brl_hard_negative_weight=1.0,
            view_hard_negative_weight=0.5,
            resume=None,
        )

        apply_run_profile(args)

        self.assertEqual(args.epochs, 2)
        self.assertEqual(args.device_mode, 'single')
        self.assertEqual(args.num_workers, 0)
        self.assertEqual(args.input_height, 360)
        self.assertEqual(args.input_width, 640)
        self.assertEqual(args.grid_reduce, 8)
        self.assertEqual(args.img_reduce, 8)
        self.assertEqual(args.max_train_frames, 64)
        self.assertEqual(args.max_test_frames, 20)
        self.assertEqual(args.brl_warmup_epochs, 0)
        self.assertEqual(args.brl_ramp_epochs, 2)
        self.assertEqual(args.eval_thresholds, [
            0.30, 0.35, 0.40, 0.42, 0.44, 0.46, 0.48,
            0.50, 0.60, 0.70, 0.80, 0.90,
        ])
        self.assertEqual(args.lr, 0.005)
        self.assertEqual(args.grad_clip_norm, 1.0)
        self.assertEqual(args.brl_hard_negative_weight, 2.0)
        self.assertEqual(args.view_hard_negative_weight, 1.0)
        self.assertTrue(args.skip_initial_eval)
        self.assertTrue(args.skip_final_eval)

    def test_local_profile_resume_runs_one_final_evaluation(self):
        args = SimpleNamespace(
            run_profile='local_2ep', pa=45, variant='default', epochs=10,
            batch_size=4, num_workers=4, wandb_mode='online', device_mode='auto',
            input_height=720, input_width=1280, grid_reduce=4, img_reduce=4,
            max_train_frames=None, max_test_frames=None, skip_initial_eval=False,
            skip_final_eval=True, brl_warmup_epochs=1, brl_ramp_epochs=2,
            eval_thresholds=None, lr=0.1, grad_clip_norm=0.0,
            brl_hard_negative_weight=1.0, view_hard_negative_weight=0.5,
            resume='saved-run',
        )

        apply_run_profile(args)

        self.assertFalse(args.skip_final_eval)

    def test_auto_device_mode_uses_cuda_zero_on_one_gpu(self):
        with patch.object(torch.cuda, 'is_available', return_value=True), \
                patch.object(torch.cuda, 'device_count', return_value=1):
            backbone, fusion = PerspTransDetector.resolve_devices('auto')

        self.assertEqual(str(backbone), 'cuda:0')
        self.assertEqual(str(fusion), 'cuda:0')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is required for the local GPU smoke test')
    def test_default_detector_forward_and_backward_on_one_gpu(self):
        num_cameras = 7
        intrinsic = np.eye(3, dtype=np.float32)
        extrinsic = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        base = SimpleNamespace(
            intrinsic_matrices=[intrinsic.copy() for _ in range(num_cameras)],
            extrinsic_matrices=[extrinsic.copy() for _ in range(num_cameras)],
            worldgrid2worldcoord_mat=np.eye(3, dtype=np.float32),
        )
        dataset = SimpleNamespace(
            num_cam=num_cameras,
            img_shape=[64, 96],
            reducedgrid_shape=[12, 16],
            img_reduce=8,
            grid_reduce=4,
            base=base,
        )
        model = PerspTransDetector(dataset, arch='resnet18', device_mode='single')
        images = torch.randn(1, num_cameras, 3, 32, 48)

        map_logits, view_logits = model(images)
        loss = map_logits.square().mean()
        loss = loss + sum(result.square().mean() for result in view_logits)
        loss.backward()

        self.assertEqual(map_logits.device, torch.device('cuda:0'))
        self.assertEqual(len(view_logits), num_cameras)
        self.assertTrue(torch.isfinite(loss))


class LimitedEvaluationTest(unittest.TestCase):
    def test_threshold_sweep_can_separate_calibration_from_false_positives(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gt_path = directory / 'gt.txt'
            low_path = directory / 'low.txt'
            high_path = directory / 'high.txt'
            np.savetxt(gt_path, np.array([[0, 0, 0]], dtype=float), fmt='%.1f')
            candidates = torch.tensor([
                [0, 0, 0, 0.90],
                [0, 100, 100, 0.50],
            ], dtype=torch.float32)

            low = PerspectiveTrainer._evaluate_candidates(
                candidates, 0.4, str(low_path), str(gt_path), 'Wildtrack'
            )
            high = PerspectiveTrainer._evaluate_candidates(
                candidates, 0.8, str(high_path), str(gt_path), 'Wildtrack'
            )

        self.assertAlmostEqual(low['moda_percent'], 0.0)
        self.assertAlmostEqual(high['moda_percent'], 100.0)
        self.assertEqual(low['num_detections'], 2)
        self.assertEqual(high['num_detections'], 1)

    def test_local_diagnostic_preserves_fixed_and_selected_threshold_results(self):
        history = [{
            'epoch': 1,
            'moda_percent': 0.0,
            'selected_cls_threshold': 0.8,
            'selected_moda_percent': 25.0,
            'selected_detection_precision_percent': 60.0,
            'selected_detection_recall_percent': 50.0,
            'bev/evidence_negative_cells': 100.0,
            'view/foot_evidence_negative_cells': 200.0,
            'bev/hard_negative_cells': 10.0,
            'view/foot_hard_negative_cells': 20.0,
            'threshold_sweep': {},
        }]

        diagnostic = build_local_diagnostic(history, fixed_threshold=0.4)

        self.assertEqual(diagnostic['status'], 'pass')
        self.assertEqual(diagnostic['best_threshold'], 0.8)
        self.assertEqual(diagnostic['fixed_threshold_best_moda_percent'], 0.0)
        self.assertTrue(diagnostic['loss_path_active'])
        self.assertTrue(diagnostic['hard_negative_path_active'])
        self.assertTrue(diagnostic['threshold_calibration_recovered_positive_moda'])

    def test_local_diagnostic_fails_when_new_loss_path_is_inactive(self):
        history = [{
            'epoch': 1,
            'moda_percent': 10.0,
            'selected_cls_threshold': 0.4,
            'selected_moda_percent': 10.0,
            'selected_detection_precision_percent': 60.0,
            'selected_detection_recall_percent': 50.0,
            'bev/evidence_negative_cells': 0.0,
            'view/foot_evidence_negative_cells': 0.0,
        }]

        diagnostic = build_local_diagnostic(history)

        self.assertEqual(diagnostic['status'], 'fail')
        self.assertFalse(diagnostic['loss_path_active'])

    def test_frame_limit_is_evenly_spaced_and_gt_is_filtered(self):
        class FakeBase:
            __name__ = 'FakeWildtrack'
            num_cam = 1
            num_frame = 10
            img_shape = [32, 48]
            worldgrid_shape = [16, 24]
            indexing = 'ij'

            def __init__(self, root):
                self.root = str(root)

            @staticmethod
            def get_image_fpaths(frame_ids):
                return {0: {frame: f'{frame}.png' for frame in frame_ids}}

            @staticmethod
            def get_worldgrid_from_pos(position):
                return np.array([position % 16, position // 16], dtype=int)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            annotation_dir = root / 'annotations_positions'
            annotation_dir.mkdir()
            for frame in range(10):
                record = [{
                    'personID': frame,
                    'positionID': frame,
                    'views': [{
                        'xmin': 1,
                        'xmax': 3,
                        'ymin': 1,
                        'ymax': 4,
                    }],
                }]
                (annotation_dir / f'{frame}.json').write_text(
                    json.dumps(record),
                    encoding='utf-8',
                )

            cache_dir = root / 'cache'
            with patch.dict('os.environ', {'MVDET_CACHE_DIR': str(cache_dir)}):
                dataset = frameDataset(
                    FakeBase(root),
                    train=True,
                    train_ratio=0.8,
                    max_frames=3,
                )

            gt = np.loadtxt(dataset.gt_fpath, ndmin=2)

        self.assertEqual(dataset.frame_ids, (0, 3, 7))
        self.assertEqual(tuple(gt[:, 0].astype(int)), dataset.frame_ids)

    def test_ground_truth_frame_without_detection_counts_as_false_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            gt_path = directory / 'gt.txt'
            result_path = directory / 'result.txt'
            np.savetxt(
                gt_path,
                np.array([[10, 0, 0], [20, 0, 0]], dtype=float),
                fmt='%.1f',
            )
            np.savetxt(
                result_path,
                np.array([[10, 0, 0]], dtype=float),
                fmt='%.1f',
            )

            recall, precision, moda, modp = evaluateDetection_py(
                result_path,
                gt_path,
                'Wildtrack',
            )

        self.assertAlmostEqual(recall, 50.0)
        self.assertAlmostEqual(precision, 100.0)
        self.assertAlmostEqual(moda, 50.0)
        self.assertAlmostEqual(modp, 100.0)


if __name__ == '__main__':
    unittest.main()
