from types import SimpleNamespace

import pytest
import torch

import main


def test_parse_cuda_devices_rejects_invalid_specs(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    with pytest.raises(ValueError):
        main.parse_cuda_devices("")
    with pytest.raises(ValueError):
        main.parse_cuda_devices("0,0")
    with pytest.raises(ValueError):
        main.parse_cuda_devices("0,2")
    assert main.parse_cuda_devices("0, 1") == [0, 1]


def test_build_pu_criterion_stays_on_requested_device():
    args = SimpleNamespace(
        loss="pu",
        pu_annotation_propensity=None,
        drop_ratio=60,
        pu_pos_thr=0.1,
        pu_class_prior=None,
        pu_beta=0.0,
        pu_gamma=1.0,
    )
    criterion = main.build_criterion(args, torch.device("cpu"))
    assert criterion.annotation_propensity == pytest.approx(0.4)
