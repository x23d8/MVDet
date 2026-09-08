import unittest
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiview_detector.loss.bev_brl import BEVBRLLoss


def gaussian_kernel(size=5, sigma=1.0):
    radius = size // 2
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing="ij",
    )
    kernel = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma ** 2))
    return kernel.view(1, 1, size, size)


class BEVBRLLossTest(unittest.TestCase):
    def test_max_gaussian_target_is_bounded_and_keeps_peaks(self):
        criterion = BEVBRLLoss(use_consensus=False)
        prediction = torch.zeros(1, 1, 9, 9)
        sparse_target = torch.zeros_like(prediction)
        sparse_target[0, 0, 4, 3] = 1
        sparse_target[0, 0, 4, 5] = 1

        target = criterion.max_gaussian_target(
            prediction,
            sparse_target,
            gaussian_kernel(),
        )

        self.assertEqual(float(target.max()), 1.0)
        self.assertEqual(float(target[0, 0, 4, 3]), 1.0)
        self.assertEqual(float(target[0, 0, 4, 5]), 1.0)
        self.assertTrue(torch.all(target >= 0))
        self.assertTrue(torch.all(target <= 1))

    def test_consensus_gated_loss_has_finite_gradient_and_mirror_peak(self):
        criterion = BEVBRLLoss(
            positive_threshold=0.1,
            ignore_threshold=0.01,
            negative_threshold=0.2,
            view_negative_threshold=0.2,
            bev_threshold=0.6,
            view_threshold=0.6,
            min_views=2,
            consensus_topk=2,
            warmup_epochs=0,
            ramp_epochs=0,
            max_mirror_per_observed=2.0,
        )
        criterion.set_epoch(1)

        map_logits = torch.full((1, 1, 9, 9), -4.0, requires_grad=True)
        with torch.no_grad():
            map_logits[0, 0, 1, 1] = 2.0  # unlabeled, multi-view-supported peak
            map_logits[0, 0, 1, 7] = 2.0  # high BEV score rejected by view consensus
            map_logits[0, 0, 4, 4] = 2.0  # observed positive

        sparse_target = torch.zeros(1, 1, 9, 9)
        sparse_target[0, 0, 4, 4] = 1

        projected_scores = torch.zeros(1, 2, 1, 9, 9)
        projected_scores[:, :, :, 1, 1] = 0.9
        projected_visibility = torch.ones_like(projected_scores, dtype=torch.bool)

        loss = criterion(
            map_logits,
            sparse_target,
            gaussian_kernel(),
            projected_view_scores=projected_scores,
            projected_visibility=projected_visibility,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(map_logits.grad).all())
        self.assertGreaterEqual(criterion.last_stats["positive_cells"], 1)
        self.assertEqual(criterion.last_stats["mirror_cells"], 1)
        self.assertEqual(criterion.last_stats["hard_negative_cells"], 1)
        self.assertGreater(criterion.last_stats["reliable_negative_cells"], 0)

    def test_brl_weight_warmup_and_ramp(self):
        criterion = BEVBRLLoss(brl_weight=0.12, warmup_epochs=2, ramp_epochs=2)
        expected = {1: 0.0, 2: 0.0, 3: 0.06, 4: 0.12, 8: 0.12}
        for epoch, weight in expected.items():
            criterion.set_epoch(epoch)
            self.assertAlmostEqual(criterion.current_brl_weight, weight)


if __name__ == "__main__":
    unittest.main()
