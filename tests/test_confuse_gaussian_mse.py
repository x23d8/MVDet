"""Small numerical checks for the two heatmap target representations."""

import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]


def load_loss(path):
    spec = importlib.util.spec_from_file_location(path.parent.parent.name + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ConfuseGaussianMSE


class ConfuseGaussianMSETest(unittest.TestCase):
    def test_mirror_and_downweight_on_missing_label(self):
        prediction = torch.tensor([[[[0.8, 0.7]]]], requires_grad=True)
        observed = torch.tensor([[[[1.0, 0.0]]]])
        kernel = torch.ones(1, 1, 1, 1)
        for path, needs_kernel in (
            (ROOT / 'multiview_detector/loss/confuse_gaussian_mse.py', True),
            (ROOT / 'SHOTBRL/multiview_detector/loss/confuse_gaussian_mse.py', True),
            (ROOT / 'MVDeTr/multiview_detector/loss/confuse_gaussian_mse.py', False),
        ):
            loss_class = load_loss(path)
            for mirror, background_error in ((True, 0.3 ** 2), (False, 0.7 ** 2)):
                loss = loss_class(beta=0.1, mirror=mirror)
                value = (loss(prediction, observed, kernel) if needs_kernel
                         else loss(prediction, observed))
                expected = (0.2 ** 2 + 0.1 * background_error) / 2
                self.assertAlmostEqual(value.item(), expected, places=6)
                self.assertTrue(torch.isfinite(torch.autograd.grad(value, prediction,
                                                                   retain_graph=True)[0]).all())


if __name__ == '__main__':
    unittest.main()
