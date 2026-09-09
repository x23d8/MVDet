import unittest
from unittest.mock import patch

import torch

from multiview_detector.models.device import resolve_model_devices


class ModelDeviceTest(unittest.TestCase):
    @patch('multiview_detector.models.device.torch.cuda.device_count', return_value=0)
    @patch('multiview_detector.models.device.torch.cuda.is_available', return_value=False)
    def test_cpu_fallback(self, _is_available, _device_count):
        front, fusion = resolve_model_devices()
        self.assertEqual(front, torch.device('cpu'))
        self.assertEqual(fusion, torch.device('cpu'))

    @patch('multiview_detector.models.device.torch.cuda.device_count', return_value=1)
    @patch('multiview_detector.models.device.torch.cuda.is_available', return_value=True)
    def test_single_gpu_uses_cuda_zero(self, _is_available, _device_count):
        front, fusion = resolve_model_devices()
        self.assertEqual(front, torch.device('cuda:0'))
        self.assertEqual(fusion, torch.device('cuda:0'))

    @patch('multiview_detector.models.device.torch.cuda.device_count', return_value=2)
    @patch('multiview_detector.models.device.torch.cuda.is_available', return_value=True)
    def test_two_gpus_preserve_original_split(self, _is_available, _device_count):
        front, fusion = resolve_model_devices()
        self.assertEqual(front, torch.device('cuda:1'))
        self.assertEqual(fusion, torch.device('cuda:0'))


if __name__ == '__main__':
    unittest.main()
