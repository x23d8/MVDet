import unittest

import torch

from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.mirror_focal_heatmap import (
    MirrorFocalHeatmapLoss,
    partial_annotation_defaults,
)


class PartialAnnotationDefaultsTest(unittest.TestCase):
    def test_defaults_follow_missing_label_severity(self):
        pa20 = partial_annotation_defaults(20)
        pa45 = partial_annotation_defaults(45)
        pa60 = partial_annotation_defaults(60)

        self.assertGreater(pa20['threshold'], pa45['threshold'])
        self.assertGreater(pa45['threshold'], pa60['threshold'])
        self.assertLess(pa20['beta'], pa45['beta'])
        self.assertLess(pa45['beta'], pa60['beta'])
        self.assertEqual(pa20['background_weight'], 0.8)
        self.assertEqual(pa45['background_weight'], 0.55)
        self.assertEqual(pa60['background_weight'], 0.4)

    def test_invalid_drop_rate_is_rejected(self):
        with self.assertRaises(ValueError):
            partial_annotation_defaults(100)


class MirrorFocalHeatmapLossTest(unittest.TestCase):
    @staticmethod
    def delta_kernel(channels=1):
        return torch.eye(channels).view(channels, channels, 1, 1)

    def test_gaussian_target_matches_original_mse_transform(self):
        prediction = torch.zeros(1, 1, 5, 7)
        target = torch.zeros(1, 1, 10, 14)
        target[0, 0, 4, 6] = 1
        kernel = torch.tensor([[[[0.25, 0.5, 0.25],
                                 [0.5, 1.0, 0.5],
                                 [0.25, 0.5, 0.25]]]])
        original = GaussianMSE()._traget_transform(prediction, target, kernel)
        adapted = MirrorFocalHeatmapLoss(0.45)._traget_transform(
            prediction, target, kernel
        )

        torch.testing.assert_close(adapted, original.clamp(0, 1))

    def test_selected_unlabeled_peak_receives_positive_gradient(self):
        logits = torch.full((1, 1, 7, 7), -4.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 3, 3] = 2.0
            logits[0, 0, 1, 1] = 2.0
        target = torch.zeros_like(logits)
        target[0, 0, 1, 1] = 1.0
        criterion = MirrorFocalHeatmapLoss(
            0.5,
            threshold=0.6,
            beta=1.0,
            gaussian_weight=1.0,
            focal_weight=1.0,
            warmup_epochs=0,
            ramp_epochs=0,
        )
        criterion.set_epoch(1)

        criterion(logits, target, self.delta_kernel()).backward()

        # Gradient descent raises a mirrored logit and lowers ordinary
        # background logits.
        self.assertLess(logits.grad[0, 0, 3, 3].item(), 0)
        self.assertGreater(logits.grad[0, 0, 6, 6].item(), 0)
        self.assertEqual(criterion.last_stats['pseudo_peaks'], 1.0)

    def test_warmup_does_not_mirror_predictions(self):
        logits = torch.full((1, 1, 5, 5), -4.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 3, 3] = 2.0
        target = torch.zeros_like(logits)
        target[0, 0, 1, 1] = 1.0
        criterion = MirrorFocalHeatmapLoss(
            0.5, threshold=0.6, warmup_epochs=1, ramp_epochs=0
        )
        criterion.set_epoch(1)

        criterion(logits, target, self.delta_kernel()).backward()

        self.assertGreater(logits.grad[0, 0, 3, 3].item(), 0)
        self.assertEqual(criterion.last_stats['pseudo_peaks'], 0.0)

    def test_eval_mode_disables_pseudo_labels(self):
        logits = torch.full((1, 1, 5, 5), -4.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 3, 3] = 2.0
        target = torch.zeros_like(logits)
        target[0, 0, 1, 1] = 1.0
        criterion = MirrorFocalHeatmapLoss(
            0.5, threshold=0.6, warmup_epochs=0, ramp_epochs=0
        )
        criterion.set_epoch(2)
        criterion.eval()

        criterion(logits, target, self.delta_kernel()).backward()

        self.assertGreater(logits.grad[0, 0, 3, 3].item(), 0)
        self.assertEqual(criterion.last_stats['pseudo_peaks'], 0.0)

    def test_pseudo_budget_increases_with_drop_rate(self):
        logits = torch.full((1, 1, 11, 31), -5.0)
        target = torch.zeros_like(logits)
        for column in (1, 5, 9, 13):
            target[0, 0, 1, column] = 1.0
        for column in range(1, 31, 3):
            logits[0, 0, 8, column] = 3.0 - column * 0.001

        counts = []
        for drop_rate in (0.20, 0.45, 0.60):
            criterion = MirrorFocalHeatmapLoss(
                drop_rate,
                threshold=0.6,
                warmup_epochs=0,
                ramp_epochs=0,
                local_max_kernel=3,
            )
            criterion.set_epoch(1)
            loss = criterion(logits, target, self.delta_kernel())
            self.assertTrue(torch.isfinite(loss))
            counts.append(criterion.last_stats['pseudo_peaks'])

        self.assertEqual(counts, [1.0, 3.0, 6.0])

    def test_two_channel_kernel_and_backward_are_supported(self):
        logits = torch.randn(2, 2, 8, 9, requires_grad=True)
        target = torch.zeros(2, 2, 16, 18)
        target[:, :, 4, 6] = 1
        criterion = MirrorFocalHeatmapLoss(0.45)

        loss = criterion(logits, target, self.delta_kernel(channels=2))
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__ == '__main__':
    unittest.main()
