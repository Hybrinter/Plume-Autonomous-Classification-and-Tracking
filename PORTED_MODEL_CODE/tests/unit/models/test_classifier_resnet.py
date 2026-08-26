"""Tests for build_classifier."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from smoke_detection.models.classifier_resnet import build_classifier


def test_returns_nn_module():
    net = build_classifier(pretrained=False)
    assert isinstance(net, nn.Module)


def test_conv1_takes_4_channels():
    net = build_classifier(pretrained=False)
    assert net.conv1.in_channels == 4


def test_fc_outputs_single_logit():
    net = build_classifier(pretrained=False)
    assert net.fc.out_features == 1


@pytest.mark.parametrize("shape", [(2, 4, 90, 90), (1, 4, 120, 120)])
def test_forward_output_shape(shape):
    net = build_classifier(pretrained=False).eval()
    with torch.no_grad():
        out = net(torch.randn(*shape))
    assert out.shape == (shape[0], 1)


@pytest.mark.slow
def test_pretrained_weights_load(cached_resnet50_weights):
    net = build_classifier(pretrained=True)
    # conv1 is replaced after load, so shape[1] must be 4 (not the IN1K 3).
    assert net.conv1.weight.shape[1] == 4
