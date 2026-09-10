import tempfile
import os
import unittest
from types import SimpleNamespace

import torch

from main import (
    DisabledRun,
    current_git_commit,
    init_wandb,
    save_training_checkpoint,
)
from multiview_detector.utils.logger import Logger


class MainRuntimeTest(unittest.TestCase):
    def test_current_git_commit_is_recorded(self):
        commit = current_git_commit()
        self.assertIsNotNone(commit)
        self.assertEqual(len(commit), 40)
        int(commit, 16)

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

    def test_checkpoint_replacement_is_atomic_and_has_no_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'training_checkpoint.pth')
            save_training_checkpoint(path, {'epoch': 1})
            save_training_checkpoint(path, {'epoch': 2})
            self.assertEqual(torch.load(path, weights_only=True), {'epoch': 2})
            self.assertFalse(os.path.exists(f'{path}.tmp'))


if __name__ == '__main__':
    unittest.main()
