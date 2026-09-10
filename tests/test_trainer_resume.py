import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from multiview_detector.trainer import PerspectiveTrainer


class TinyDataset(Dataset):
    map_kernel = torch.ones(1, 1)
    img_kernel = torch.ones(1, 1)

    def __len__(self):
        return 4

    def __getitem__(self, index):
        value = torch.tensor([[[float(index + 1)]]])
        return value, torch.zeros_like(value), [], index


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.seen = []

    def forward(self, data):
        self.seen.extend(int(value) for value in data[:, 0, 0, 0])
        return data * self.weight, []


class TinyCriterion(nn.Module):
    def forward(self, prediction, target, _kernel):
        return (prediction - target).square().mean()


class TrainerResumeTest(unittest.TestCase):
    def test_start_batch_skips_completed_batches_and_callback_uses_absolute_index(self):
        dataset = TinyDataset()
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        model = TinyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        callbacks = []
        with tempfile.TemporaryDirectory() as logdir:
            trainer = PerspectiveTrainer(
                model,
                TinyCriterion(),
                logdir,
                denormalize=None,
                view_criterion=None,
            )
            loss, _ = trainer.train(
                1,
                loader,
                optimizer,
                log_interval=10,
                start_batch=2,
                checkpoint_callback=lambda epoch, batch: callbacks.append((epoch, batch)),
            )

        self.assertEqual(model.seen, [3, 4])
        self.assertEqual(callbacks, [(1, 3), (1, 4)])
        self.assertTrue(torch.isfinite(torch.tensor(loss)))

    def test_generator_state_recreates_shuffle_before_skipping(self):
        dataset = TinyDataset()
        first_generator = torch.Generator().manual_seed(17)
        epoch_state = first_generator.get_state().clone()
        first_order = [
            int(batch[3])
            for batch in DataLoader(
                dataset, batch_size=1, shuffle=True, generator=first_generator
            )
        ]

        resumed_generator = torch.Generator()
        resumed_generator.set_state(epoch_state)
        resumed_order = [
            int(batch[3])
            for batch in DataLoader(
                dataset, batch_size=1, shuffle=True, generator=resumed_generator
            )
        ]
        self.assertEqual(resumed_order[2:], first_order[2:])


if __name__ == '__main__':
    unittest.main()
