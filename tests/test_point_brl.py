import unittest

import torch

from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.point_brl import PointBRLLoss
from multiview_detector.trainer import PerspectiveTrainer


class PointBRLLossTest(unittest.TestCase):
    def test_mirrored_gradient_reduces_missing_annotation_penalty(self):
        # Ô có p=0.9 nhưng thiếu nhãn phải được đẩy lên sau warm-up.
        logits = torch.tensor([[[[2.1972246, -2.1972246]]]], requires_grad=True)
        target = torch.zeros_like(logits)
        loss_fn = PointBRLLoss(warmup_epochs=1)
        loss_fn.set_epoch(1)
        loss = loss_fn(logits, target)
        loss.backward()
        self.assertLess(logits.grad[0, 0, 0, 0].item(), 0)
        self.assertGreater(logits.grad[0, 0, 0, 1].item(), 0)

    def test_warmup_keeps_high_confidence_negative_in_background(self):
        logits = torch.tensor([[[[2.1972246]]]], requires_grad=True)
        loss_fn = PointBRLLoss(warmup_epochs=2)
        loss_fn.set_epoch(1)
        loss_fn(logits, torch.zeros_like(logits)).backward()
        self.assertGreater(logits.grad.item(), 0)


    def test_known_positive_and_guard_are_not_mirrored(self):
        logits = torch.tensor([[[[2.1972246, 2.1972246, 2.1972246]]]],
                              requires_grad=True)
        target = torch.tensor([[[[1.0, 0.0, 0.0]]]])
        loss_fn = PointBRLLoss(warmup_epochs=0, guard_radius=1)
        loss_fn(logits, target).backward()
        self.assertLess(logits.grad[0, 0, 0, 0].item(), 0)
        self.assertGreater(logits.grad[0, 0, 0, 1].item(), 0)
        self.assertLess(logits.grad[0, 0, 0, 2].item(), 0)

    def test_target_pooling_preserves_annotated_point(self):
        target = torch.zeros(1, 1, 4, 4)
        target[0, 0, 2, 3] = 1
        mask = PointBRLLoss().point_mask(torch.zeros(1, 1, 2, 2), target)
        self.assertEqual(mask.sum().item(), 1)
        self.assertTrue(mask[0, 0, 1, 1].item())

    def test_loss_averages_three_groups_separately(self):
        # gamma=0 cho phép kiểm tra trực tiếp công thức ba nhóm của spec.
        logits = torch.tensor([[[[0.0, 2.1972246, -2.1972246, -2.1972246]]]])
        target = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        loss_fn = PointBRLLoss(gamma=0, confusion_weight=0.25,
                               warmup_epochs=0)
        actual = loss_fn(logits, target)
        expected = (torch.nn.functional.softplus(-logits[0, 0, 0, 0])
                    + 0.25 * torch.nn.functional.softplus(-logits[0, 0, 0, 1])
                    + torch.nn.functional.softplus(logits[0, 0, 0, 2]))
        self.assertTrue(torch.allclose(actual, expected, atol=1e-6))

    def test_empty_groups_and_extreme_logits_have_finite_gradients(self):
        logits = torch.tensor([[[[100.0, -100.0]]]], requires_grad=True)
        loss_fn = PointBRLLoss(warmup_epochs=0)
        loss = loss_fn(logits, torch.ones_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(logits.grad).all().item())

    def test_full_annotation_mode_never_mirrors_negative(self):
        logits = torch.tensor([[[[2.1972246]]]], requires_grad=True)
        loss_fn = PointBRLLoss(warmup_epochs=0, enable_confusion=False)
        loss_fn(logits, torch.zeros_like(logits)).backward()
        self.assertGreater(logits.grad.item(), 0)

    def test_validation_with_complete_labels_disables_mirror(self):
        logits = torch.tensor([[[[2.1972246]]]], requires_grad=True)
        loss_fn = PointBRLLoss(warmup_epochs=0)
        loss_fn.eval()
        loss_fn(logits, torch.zeros_like(logits)).backward()
        self.assertGreater(logits.grad.item(), 0)


class TrainerIntegrationTest(unittest.TestCase):
    def test_train_step_combines_point_bev_and_gaussian_view_loss(self):
        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bev = torch.nn.Parameter(torch.tensor(0.0))
                self.view = torch.nn.Parameter(torch.tensor(0.5))

            def forward(self, data):
                batch = data.shape[0]
                bev = self.bev.expand(batch, 1, 2, 2)
                view = self.view.expand(batch, 2, 2, 2)
                return bev, [view]

        class TinyDataset(torch.utils.data.Dataset):
            img_kernel = torch.eye(2).reshape(2, 2, 1, 1)

            def __len__(self):
                return 1

            def __getitem__(self, index):
                point = torch.zeros(1, 2, 2)
                point[0, 0, 0] = 1
                return torch.zeros(1, 3, 2, 2), point, [torch.zeros(2, 2, 2)], index

        model = TinyModel()
        trainer = PerspectiveTrainer(model, PointBRLLoss(), '.', None,
                                     view_criterion=GaussianMSE())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loader = torch.utils.data.DataLoader(TinyDataset(), batch_size=1)
        trainer.train(1, loader, optimizer)
        self.assertNotEqual(model.bev.item(), 0.0)
        self.assertNotEqual(model.view.item(), 0.5)
        self.assertEqual(trainer.criterion.epoch, 0)


if __name__ == '__main__':
    unittest.main()
