import unittest
from pathlib import Path
import sys
import types
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from multiview_detector.loss import BEVBRLLoss, GaussianMSE, PartialViewBRLLoss
from multiview_detector.trainer import PerspectiveTrainer


def gaussian_kernel(channels=1, size=5, sigma=1.0):
    radius = size // 2
    yy, xx = torch.meshgrid(
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        torch.arange(-radius, radius + 1, dtype=torch.float32),
        indexing='ij',
    )
    gaussian = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma ** 2))
    kernel = torch.zeros(channels, channels, size, size)
    for channel in range(channels):
        kernel[channel, channel] = gaussian
    return kernel


class GaussianTargetTest(unittest.TestCase):
    def test_gaussian_mse_uses_bounded_max_composition(self):
        criterion = GaussianMSE()
        prediction = torch.zeros(1, 1, 9, 9)
        sparse_target = torch.zeros_like(prediction)
        sparse_target[0, 0, 4, 3] = 1
        sparse_target[0, 0, 4, 5] = 1

        target = criterion._traget_transform(
            prediction,
            sparse_target,
            gaussian_kernel(),
        )

        self.assertEqual(float(target.max()), 1.0)
        self.assertTrue(torch.all(target >= 0))
        self.assertTrue(torch.all(target <= 1))


class BEVBRLLossTest(unittest.TestCase):
    def test_positive_bce_does_not_stall_at_one_percent_prior(self):
        criterion = BEVBRLLoss(
            positive_weight=1.0,
            negative_weight=0.0,
            brl_weight=0.0,
            warmup_epochs=0,
            ramp_epochs=0,
        )
        criterion.set_epoch(1)
        prior = 0.01
        prior_logit = torch.logit(torch.tensor(prior))
        logits = torch.full((1, 1, 9, 9), prior_logit.item(), requires_grad=True)
        sparse_target = torch.zeros_like(logits)
        sparse_target[0, 0, 4, 4] = 1
        scores = torch.zeros(1, 2, 1, 9, 9)
        visibility = torch.ones_like(scores, dtype=torch.bool)

        loss = criterion(
            logits,
            sparse_target,
            gaussian_kernel(),
            projected_view_scores=scores,
            projected_visibility=visibility,
        )
        loss.backward()

        positive_count = criterion.last_stats['positive_cells']
        bce_gradient = abs(float(logits.grad[0, 0, 4, 4]))
        mse_gradient = 2 * (1 - prior) * prior * (1 - prior) / positive_count
        self.assertGreater(bce_gradient, 40 * mse_gradient)
        self.assertLess(float(logits.grad[0, 0, 4, 4]), 0.0)

    def test_fully_unlabelled_sample_is_skipped_not_background(self):
        criterion = BEVBRLLoss(warmup_epochs=0, ramp_epochs=0)
        criterion.set_epoch(5)
        logits = torch.full((1, 1, 7, 7), -4.0, requires_grad=True)
        target = torch.zeros_like(logits)
        scores = torch.zeros(1, 2, 1, 7, 7)
        visibility = torch.ones_like(scores, dtype=torch.bool)

        loss = criterion(
            logits,
            target,
            gaussian_kernel(),
            projected_view_scores=scores,
            projected_visibility=visibility,
        )

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(criterion.last_stats['reliable_negative_cells'], 0)
        self.assertEqual(criterion.last_stats['empty_samples_skipped'], 1)

    def test_observed_one_view_gt_stays_valid_and_consensus_gates_unlabelled_peaks(self):
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

        logits = torch.full((1, 1, 9, 9), -4.0, requires_grad=True)
        with torch.no_grad():
            logits[0, 0, 1, 1] = 2.0  # supported hidden pedestrian
            logits[0, 0, 1, 7] = 2.0  # unsupported projection artefact
            logits[0, 0, 4, 4] = 2.0  # observed pedestrian

        target = torch.zeros(1, 1, 9, 9)
        target[0, 0, 4, 4] = 1
        scores = torch.zeros(1, 2, 1, 9, 9)
        scores[:, :, :, 1, 1] = 0.9
        visibility = torch.ones_like(scores, dtype=torch.bool)
        # The observed cell is only visible in one camera and must still be
        # supervised as GT, though it cannot become a consensus pseudo-label.
        visibility[:, 1, :, 4, 4] = False

        loss = criterion(
            logits,
            target,
            gaussian_kernel(),
            projected_view_scores=scores,
            projected_visibility=visibility,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreaterEqual(criterion.last_stats['positive_cells'], 1)
        self.assertEqual(criterion.last_stats['mirror_cells'], 1)
        self.assertEqual(criterion.last_stats['hard_negative_cells'], 1)
        self.assertLess(float(logits.grad[0, 0, 1, 1]), 0.0)
        self.assertGreater(float(logits.grad[0, 0, 1, 7]), 0.0)

    def test_warmup_reduces_negatives_and_disables_mirror(self):
        criterion = BEVBRLLoss(
            negative_weight=0.5,
            brl_weight=0.1,
            warmup_epochs=2,
            ramp_epochs=2,
            negative_warmup_factor=0.25,
        )
        expected = {
            1: (0.125, 0.0),
            2: (0.125, 0.0),
            3: (0.3125, 0.05),
            4: (0.5, 0.1),
        }
        for epoch, (negative_weight, mirror_weight) in expected.items():
            criterion.set_epoch(epoch)
            self.assertAlmostEqual(criterion.current_negative_weight, negative_weight)
            self.assertAlmostEqual(criterion.current_brl_weight, mirror_weight)

    def test_low_consensus_prediction_cannot_escape_negative_loss_after_warmup(self):
        criterion = BEVBRLLoss(
            positive_threshold=0.1,
            ignore_threshold=0.01,
            negative_threshold=0.15,
            view_negative_threshold=0.3,
            hard_negative_threshold=0.4,
            mirror_threshold=0.6,
            warmup_epochs=1,
            ramp_epochs=1,
        )
        criterion.set_epoch(2)
        logits = torch.full((1, 1, 9, 9), -4.0, requires_grad=True)
        target = torch.zeros_like(logits)
        target[0, 0, 4, 4] = 1
        with torch.no_grad():
            # Probability 0.5 is above the detection threshold and above the
            # old easy-negative cutoff, but independent views reject it.
            logits[0, 0, 1, 7] = 0.0
        scores = torch.zeros(1, 2, 1, 9, 9)
        visibility = torch.ones_like(scores, dtype=torch.bool)

        loss = criterion(
            logits,
            target,
            gaussian_kernel(),
            projected_view_scores=scores,
            projected_visibility=visibility,
        )
        loss.backward()

        self.assertGreaterEqual(criterion.last_stats['hard_negative_cells'], 1)
        self.assertGreater(criterion.last_stats['loss_hard_negative'], 0)
        self.assertEqual(criterion.last_stats['hard_negative_weight'], 1.0)
        self.assertGreater(float(logits.grad[0, 0, 1, 7]), 0.0)

    def test_medium_prediction_remains_uncertain_during_warmup(self):
        criterion = BEVBRLLoss(
            negative_threshold=0.15,
            view_negative_threshold=0.3,
            hard_negative_threshold=0.4,
            warmup_epochs=1,
            ramp_epochs=1,
        )
        criterion.set_epoch(1)
        logits = torch.full((1, 1, 9, 9), -4.0, requires_grad=True)
        target = torch.zeros_like(logits)
        target[0, 0, 4, 4] = 1
        with torch.no_grad():
            logits[0, 0, 1, 7] = 0.0
        scores = torch.zeros(1, 2, 1, 9, 9)
        visibility = torch.ones_like(scores, dtype=torch.bool)

        loss = criterion(
            logits,
            target,
            gaussian_kernel(),
            projected_view_scores=scores,
            projected_visibility=visibility,
        )
        loss.backward()

        self.assertEqual(criterion.last_stats['hard_negative_cells'], 0)
        self.assertEqual(float(logits.grad[0, 0, 1, 7]), 0.0)


class PartialViewBRLLossTest(unittest.TestCase):
    def test_head_and_foot_positive_gradients_survive_low_prior(self):
        criterion = PartialViewBRLLoss(
            positive_weight=1.0,
            negative_weight=0.0,
            pseudo_weight=0.0,
            head_weight=1.0,
            foot_weight=1.0,
            warmup_epochs=0,
            ramp_epochs=0,
        )
        criterion.set_epoch(1)
        prior_logit = torch.logit(torch.tensor(0.01)).item()
        logits = [torch.full((1, 2, 9, 9), prior_logit, requires_grad=True) for _ in range(2)]
        targets = [torch.zeros(1, 2, 9, 9) for _ in range(2)]
        supports = [torch.zeros(1, 1, 9, 9) for _ in range(2)]
        validity = [torch.ones(1, 1, 9, 9, dtype=torch.bool) for _ in range(2)]
        for target in targets:
            target[0, :, 4, 4] = 1

        loss = criterion(
            logits,
            targets,
            gaussian_kernel(channels=2),
            foot_support_images=supports,
            foot_support_validity=validity,
        )
        loss.backward()

        for prediction in logits:
            self.assertLess(float(prediction.grad[0, 0, 4, 4]), 0.0)
            self.assertLess(float(prediction.grad[0, 1, 4, 4]), 0.0)

    def test_fully_unlabelled_view_is_skipped_after_warmup(self):
        criterion = PartialViewBRLLoss(warmup_epochs=0, ramp_epochs=0)
        criterion.set_epoch(5)
        logits = [torch.full((1, 2, 7, 7), -4.0, requires_grad=True) for _ in range(2)]
        targets = [torch.zeros(1, 2, 7, 7) for _ in range(2)]
        supports = [torch.zeros(1, 1, 7, 7) for _ in range(2)]
        validity = [torch.ones(1, 1, 7, 7, dtype=torch.bool) for _ in range(2)]

        loss = criterion(
            logits,
            targets,
            gaussian_kernel(channels=2),
            foot_support_images=supports,
            foot_support_validity=validity,
        )

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(criterion.last_stats['foot_negative_cells'], 0)
        self.assertEqual(criterion.last_stats['empty_views_skipped'], 1)

    def test_backprojected_support_excludes_the_camera_being_supervised(self):
        def identity_warp(source, matrix, output_shape):
            self.assertEqual(tuple(source.shape[-2:]), tuple(output_shape))
            self.assertEqual(tuple(matrix.shape[-2:]), (3, 3))
            return source.clone()

        kornia_module = types.ModuleType('kornia')
        geometry_module = types.ModuleType('kornia.geometry')
        transform_module = types.ModuleType('kornia.geometry.transform')
        transform_module.warp_perspective = identity_warp
        fake_modules = {
            'kornia': kornia_module,
            'kornia.geometry': geometry_module,
            'kornia.geometry.transform': transform_module,
        }

        criterion = PartialViewBRLLoss(min_views=2, consensus_topk=2)
        view_logits = [torch.zeros(1, 2, 5, 5) for _ in range(3)]
        map_logits = torch.full((1, 1, 5, 5), -4.0)
        map_logits[0, 0, 1, 1] = 3.0
        scores = torch.zeros(1, 3, 1, 5, 5)
        scores[:, 0, :, 1, 1] = 0.9  # evidence exists only in camera zero
        visibility = torch.ones_like(scores, dtype=torch.bool)
        matrices = [torch.eye(3) for _ in range(3)]

        with patch.dict(sys.modules, fake_modules):
            support, valid = criterion._build_foot_support(
                view_logits,
                map_logits,
                matrices,
                scores,
                visibility,
            )

        # View zero cannot use its own 0.9 score. View one can use view zero as
        # independent evidence, so only view one's support is high.
        self.assertEqual(float(support[0][0, 0, 1, 1]), 0.0)
        self.assertGreater(float(support[1][0, 0, 1, 1]), 0.8)
        self.assertTrue(bool(valid[0][0, 0, 1, 1]))

    def test_missing_head_is_not_background_and_foot_uses_external_support(self):
        criterion = PartialViewBRLLoss(
            positive_threshold=0.1,
            ignore_threshold=0.01,
            negative_threshold=0.2,
            hard_negative_threshold=0.6,
            pseudo_threshold=0.6,
            support_negative_threshold=0.2,
            support_positive_threshold=0.6,
            warmup_epochs=0,
            ramp_epochs=0,
            max_pseudo_per_observed=2.0,
            head_negative_weight=0.0,
        )
        criterion.set_epoch(1)

        logits = [torch.full((1, 2, 9, 9), -4.0, requires_grad=True) for _ in range(2)]
        targets = [torch.zeros(1, 2, 9, 9) for _ in range(2)]
        supports = [torch.zeros(1, 1, 9, 9) for _ in range(2)]
        validity = [torch.ones(1, 1, 9, 9, dtype=torch.bool) for _ in range(2)]
        for view_idx in range(2):
            targets[view_idx][0, :, 4, 4] = 1
            with torch.no_grad():
                logits[view_idx][0, 0, 1, 1] = 2.0  # unlabelled head: must be ignored
                logits[view_idx][0, 1, 1, 1] = 2.0  # externally supported hidden foot
                logits[view_idx][0, 1, 1, 7] = 2.0  # unsupported hard negative
            supports[view_idx][0, 0, 1, 1] = 0.9

        loss = criterion(
            logits,
            targets,
            gaussian_kernel(channels=2),
            foot_support_images=supports,
            foot_support_validity=validity,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(criterion.last_stats['head_negative_cells'], 0)
        self.assertEqual(criterion.last_stats['foot_pseudo_cells'], 1)
        self.assertEqual(criterion.last_stats['foot_hard_negative_cells'], 1)
        for prediction in logits:
            self.assertEqual(float(prediction.grad[0, 0, 1, 1]), 0.0)
            self.assertLess(float(prediction.grad[0, 1, 1, 1]), 0.0)
            self.assertGreater(float(prediction.grad[0, 1, 1, 7]), 0.0)

    def test_warmup_is_observed_positive_only(self):
        criterion = PartialViewBRLLoss(warmup_epochs=2, ramp_epochs=2)
        criterion.set_epoch(1)
        logits = [torch.full((1, 2, 7, 7), -4.0, requires_grad=True) for _ in range(2)]
        targets = [torch.zeros(1, 2, 7, 7) for _ in range(2)]
        supports = [torch.ones(1, 1, 7, 7) for _ in range(2)]
        validity = [torch.ones(1, 1, 7, 7, dtype=torch.bool) for _ in range(2)]

        loss = criterion(
            logits,
            targets,
            gaussian_kernel(channels=2),
            foot_support_images=supports,
            foot_support_validity=validity,
        )

        self.assertEqual(float(loss.detach()), 0.0)
        self.assertEqual(criterion.last_stats['foot_pseudo_cells'], 0)
        self.assertEqual(criterion.last_stats['ramp_factor'], 0.0)

    def test_low_support_foot_prediction_cannot_escape_after_warmup(self):
        criterion = PartialViewBRLLoss(
            negative_threshold=0.15,
            hard_negative_threshold=0.4,
            support_negative_threshold=0.3,
            support_positive_threshold=0.65,
            warmup_epochs=1,
            ramp_epochs=1,
        )
        criterion.set_epoch(2)
        logits = [torch.full((1, 2, 9, 9), -4.0, requires_grad=True) for _ in range(2)]
        targets = [torch.zeros(1, 2, 9, 9) for _ in range(2)]
        supports = [torch.zeros(1, 1, 9, 9) for _ in range(2)]
        validity = [torch.ones(1, 1, 9, 9, dtype=torch.bool) for _ in range(2)]
        for view_idx in range(2):
            targets[view_idx][0, :, 4, 4] = 1
            with torch.no_grad():
                logits[view_idx][0, 1, 1, 7] = 0.0

        loss = criterion(
            logits,
            targets,
            gaussian_kernel(channels=2),
            foot_support_images=supports,
            foot_support_validity=validity,
        )
        loss.backward()

        self.assertGreaterEqual(criterion.last_stats['foot_hard_negative_cells'], 1)
        self.assertGreater(criterion.last_stats['loss_foot_hard_negative'], 0)
        for prediction in logits:
            self.assertGreater(float(prediction.grad[0, 1, 1, 7]), 0.0)


class TrainerIntegrationTest(unittest.TestCase):
    def test_view_loss_can_be_explicitly_disabled_for_image_projection_variant(self):
        criterion = GaussianMSE()
        trainer = PerspectiveTrainer(
            model=object(),
            criterion=criterion,
            logdir='.',
            denormalize=None,
            view_criterion=None,
        )
        self.assertIsNone(trainer.view_criterion)

    def test_legacy_constructor_still_reuses_map_criterion_for_views(self):
        criterion = GaussianMSE()
        trainer = PerspectiveTrainer(
            model=object(),
            criterion=criterion,
            logdir='.',
            denormalize=None,
        )
        self.assertIs(trainer.view_criterion, criterion)


if __name__ == '__main__':
    unittest.main()
