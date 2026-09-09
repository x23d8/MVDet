import unittest

import torch

from multiview_detector.loss.adaptive_brl import (
    AdaptiveBRLLoss,
    MultiViewAdaptiveBRLLoss,
    robust_multiview_consensus,
)
from multiview_detector.loss.heatmap import max_gaussian_target


def gaussian_kernel(size=5):
    axis = torch.arange(size) - size // 2
    yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    kernel = torch.exp(-(xx.square() + yy.square()) / 2.0)
    return kernel.view(1, 1, size, size)


class HeatmapTest(unittest.TestCase):
    def test_max_composition_is_bounded_and_keeps_point_weight(self):
        prediction = torch.zeros(1, 1, 9, 9)
        points = torch.zeros_like(prediction)
        points[0, 0, 4, 4] = 1.0
        points[0, 0, 4, 5] = 0.5
        target = max_gaussian_target(prediction, points, gaussian_kernel())
        self.assertEqual(target.shape, prediction.shape)
        self.assertLessEqual(target.max().item(), 1.0)
        self.assertAlmostEqual(target[0, 0, 4, 4].item(), 1.0)


class ConsensusTest(unittest.TestCase):
    def test_top_two_consensus_is_camera_count_invariant(self):
        scores = torch.tensor([0.9, 0.8, 0.1]).view(1, 3, 1, 1, 1)
        visible = torch.ones_like(scores, dtype=torch.bool)
        three_view, valid = robust_multiview_consensus(scores, visible)
        two_view, _ = robust_multiview_consensus(scores[:, :2], visible[:, :2])
        self.assertTrue(valid.item())
        self.assertTrue(torch.allclose(three_view, two_view))

    def test_consensus_requires_geometric_coverage(self):
        scores = torch.full((1, 3, 1, 2, 2), 0.9)
        visible = torch.zeros_like(scores, dtype=torch.bool)
        visible[:, 0] = True
        _, valid = robust_multiview_consensus(scores, visible, min_views=2)
        self.assertFalse(valid.any().item())


class AdaptiveBRLTest(unittest.TestCase):
    def setUp(self):
        self.kernel = gaussian_kernel(3)
        self.target = torch.zeros(1, 1, 12, 12)
        self.target[0, 0, 2, 2] = 1
        self.target[0, 0, 9, 9] = 1

    def test_propensity_controls_pseudo_positive_budget(self):
        loss_fn = AdaptiveBRLLoss(annotation_probability=0.5, warmup_epochs=0, ramp_epochs=0)
        loss_fn.set_epoch(1)
        logits = torch.zeros(1, 1, 12, 12, requires_grad=True)
        evidence = torch.full_like(logits, 0.1)
        evidence[0, 0, 4, 9] = 0.95
        evidence[0, 0, 7, 3] = 0.90
        evidence[0, 0, 5, 5] = 0.85
        value = loss_fn(logits, self.target, self.kernel, evidence=evidence)
        value.backward()
        self.assertTrue(torch.isfinite(value).item())
        self.assertTrue(torch.isfinite(logits.grad).all().item())
        self.assertEqual(loss_fn.last_stats['pseudo_points'], 2)
        self.assertEqual(loss_fn.last_stats['expected_missing_points'], 2)
        self.assertLess(logits.grad[0, 0, 4, 9].item(), 0)

    def test_warmup_disables_pseudo_supervision(self):
        loss_fn = AdaptiveBRLLoss(annotation_probability=0.5, warmup_epochs=1)
        logits = torch.zeros(1, 1, 12, 12, requires_grad=True)
        evidence = torch.full_like(logits, 0.1)
        evidence[0, 0, 5, 8] = 0.95
        loss_fn.set_epoch(1)
        loss_fn(logits, self.target, self.kernel, evidence=evidence)
        self.assertEqual(loss_fn.last_stats['pseudo_cells'], 0)
        self.assertEqual(loss_fn.last_stats['ramp_factor'], 0)

    def test_full_annotation_has_zero_pseudo_budget(self):
        loss_fn = AdaptiveBRLLoss(annotation_probability=1.0, warmup_epochs=0, ramp_epochs=0)
        loss_fn.set_epoch(1)
        logits = torch.zeros(1, 1, 12, 12)
        loss_fn(logits, self.target, self.kernel)
        self.assertEqual(loss_fn.last_stats['pseudo_points'], 0)
        self.assertEqual(loss_fn.last_stats['expected_missing_points'], 0)

    def test_evidence_is_detached(self):
        loss_fn = AdaptiveBRLLoss(annotation_probability=0.5, warmup_epochs=0, ramp_epochs=0)
        loss_fn.set_epoch(1)
        logits = torch.zeros(1, 1, 12, 12, requires_grad=True)
        evidence = torch.full_like(logits, 0.1, requires_grad=True)
        value = loss_fn(logits, self.target, self.kernel, evidence=evidence)
        value.backward()
        self.assertIsNone(evidence.grad)

    def test_multiview_path_is_finite_and_detaches_camera_evidence(self):
        loss_fn = MultiViewAdaptiveBRLLoss(
            annotation_probability=0.5,
            warmup_epochs=0,
            ramp_epochs=0,
            min_views=2,
        )
        loss_fn.set_epoch(1)
        map_logits = torch.full((1, 1, 12, 12), -2.0, requires_grad=True)
        with torch.no_grad():
            map_logits[0, 0, 6, 6] = 3.0
        view_logits = []
        for _ in range(3):
            logits = torch.full((1, 2, 12, 12), -3.0, requires_grad=True)
            with torch.no_grad():
                logits[0, 1, 6, 6] = 4.0
            view_logits.append(logits)
        identity = [torch.eye(3) for _ in view_logits]
        one_point = torch.zeros(1, 1, 12, 12)
        one_point[0, 0, 2, 2] = 1

        value = loss_fn(
            map_logits,
            one_point,
            self.kernel,
            view_logits=view_logits,
            projection_matrices=identity,
        )
        value.backward()
        self.assertTrue(torch.isfinite(value).item())
        self.assertTrue(torch.isfinite(map_logits.grad).all().item())
        self.assertLess(map_logits.grad[0, 0, 6, 6].item(), 0)
        self.assertEqual(loss_fn.last_stats['pseudo_points'], 1)
        for logits in view_logits:
            self.assertIsNone(logits.grad)


if __name__ == '__main__':
    unittest.main()
