"""Tests for smoke_detection.common.seed."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from smoke_detection.common.seed import seed_everything


def _sample_triplet():
    return random.random(), float(np.random.rand()), float(torch.rand(1))


def test_same_seed_produces_same_sequence():
    seed_everything(123)
    a = _sample_triplet()
    seed_everything(123)
    b = _sample_triplet()
    assert a == b


def test_different_seeds_produce_different_sequences():
    seed_everything(1)
    a = _sample_triplet()
    seed_everything(2)
    b = _sample_triplet()
    assert a != b


def test_deterministic_sets_cudnn_flags():
    seed_everything(7, deterministic=True)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False


def test_pythonhashseed_is_set():
    seed_everything(99)
    assert os.environ["PYTHONHASHSEED"] == "99"
