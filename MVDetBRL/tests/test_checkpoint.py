import random

import numpy as np
import torch

from multiview_detector.utils.checkpoint import (
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint_atomic,
)


def test_atomic_checkpoint_roundtrip_and_rng_restore(tmp_path):
    random.seed(3)
    np.random.seed(3)
    torch.manual_seed(3)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(1))
    path = tmp_path / "state.pth"
    save_checkpoint_atomic(path, {"epoch": 4, "rng_state": state})
    assert path.exists()
    assert not (tmp_path / "state.pth.tmp").exists()
    loaded = load_checkpoint(path)
    restore_rng_state(loaded["rng_state"])
    actual = (random.random(), np.random.rand(), torch.rand(1))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def _step(model, optimizer, scheduler):
    inputs = torch.randn(4, 3)
    target = torch.randn(4, 1)
    optimizer.zero_grad()
    loss = (model(inputs) - target).square().mean()
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_resumed_optimizer_scheduler_and_rng_match_continuous_training(tmp_path):
    torch.manual_seed(23)
    continuous = torch.nn.Linear(3, 1)
    optimizer = torch.optim.AdamW(continuous.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, total_steps=4
    )
    _step(continuous, optimizer, scheduler)
    _step(continuous, optimizer, scheduler)
    checkpoint = tmp_path / "training_state.pth"
    save_checkpoint_atomic(checkpoint, {
        "model": continuous.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_state": capture_rng_state(),
    })
    _step(continuous, optimizer, scheduler)
    _step(continuous, optimizer, scheduler)

    resumed = torch.nn.Linear(3, 1)
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
    resumed_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        resumed_optimizer, max_lr=1e-3, total_steps=4
    )
    state = load_checkpoint(checkpoint)
    resumed.load_state_dict(state["model"])
    resumed_optimizer.load_state_dict(state["optimizer"])
    resumed_scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng_state"])
    _step(resumed, resumed_optimizer, resumed_scheduler)
    _step(resumed, resumed_optimizer, resumed_scheduler)

    for expected, actual in zip(continuous.parameters(), resumed.parameters()):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
