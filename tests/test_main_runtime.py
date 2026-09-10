import tempfile
import os
import unittest
from types import SimpleNamespace

from main import DisabledRun, init_wandb
from multiview_detector.utils.logger import Logger


class MainRuntimeTest(unittest.TestCase):
    def test_disabled_logging_does_not_import_or_require_wandb(self):
        args = SimpleNamespace(wandb_mode='disabled')
        with tempfile.TemporaryDirectory() as logdir:
            run = init_wandb(args, None, None, None, logdir)
        self.assertIsInstance(run, DisabledRun)
        run.define_metric('epoch')
        run.log({'epoch': 0})
        run.finish()

    def test_logger_can_append_when_training_is_resumed(self):
        with tempfile.TemporaryDirectory() as logdir:
            path = os.path.join(logdir, 'log.txt')
            first = Logger(path)
            first.write('epoch 1\n')
            first.close()
            resumed = Logger(path, append=True)
            resumed.write('epoch 2\n')
            resumed.close()
            with open(path, encoding='utf-8') as log_file:
                self.assertEqual(log_file.read(), 'epoch 1\nepoch 2\n')


if __name__ == '__main__':
    unittest.main()
