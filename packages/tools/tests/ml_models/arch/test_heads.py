"""PactNet spatial head, full-frame shapes, and ShuffleNet batch norm."""

import gc

import torch
from tools.ml_models.arch.classifier import build_backbone
from tools.ml_models.arch.compact import PactNet
from tools.ml_models.arch.registry import build
from tools.ml_models.arch.unet import build_segmentor
from torch import nn


def test_default_heads_on_small_tile() -> None:
    """Default pactnet emits one logit; default dilatenet keeps 76x76."""
    tile = torch.zeros(1, 3, 76, 76)
    pact = build("classifier", "pactnet", 3).eval()
    dilate = build("segmentor", "dilatenet", 3).eval()
    with torch.inference_mode():
        assert pact(tile).shape == (1, 1)
        assert dilate(tile).shape == (1, 1, 76, 76)


def test_default_heads_on_flight_frame() -> None:
    """Default pactnet, dilatenet, and U-Net run at 1544x2064."""
    frame = (1, 3, 1544, 2064)
    pact = build("classifier", "pactnet", 3).eval()
    with torch.inference_mode():
        logits = pact(torch.zeros(*frame))
        assert logits.shape == (1, 1)
    del logits
    del pact
    gc.collect()

    dilate = build("segmentor", "dilatenet", 3).eval()
    with torch.inference_mode():
        mask = dilate(torch.zeros(*frame))
        assert mask.shape == (1, 1, 1544, 2064)
    del mask
    del dilate
    gc.collect()

    unet = build_segmentor().eval()
    with torch.inference_mode():
        mask = unet(torch.zeros(*frame))
        assert mask.shape == (1, 1, 1544, 2064)
    del mask
    del unet
    gc.collect()


def test_pactnet_block_logit_exceeds_single_pixel() -> None:
    """A 16x16 support scores higher than one pixel with positive weights."""
    net = PactNet().eval()
    with torch.no_grad():
        for module in net.modules():
            if isinstance(module, nn.Conv2d):
                module.weight.fill_(1e-3)
                if module.bias is not None:
                    module.bias.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    module.weight.fill_(1.0)
                if module.bias is not None:
                    module.bias.zero_()
    single = torch.zeros(1, 3, 64, 64)
    single[0, :, 32, 32] = 1.0
    block = torch.zeros(1, 3, 64, 64)
    block[0, :, 24:40, 24:40] = 1.0
    with torch.inference_mode():
        spatial = net.spatial(single)
        assert spatial.shape[-2] >= 2
        assert spatial.shape[-1] >= 2
        single_logit = net(single)
        block_logit = net(block)
    assert single_logit.shape == (1, 1)
    assert block_logit.shape == (1, 1)
    assert float(block_logit) > float(single_logit)


def test_forward_equals_spatial_amax() -> None:
    """forward returns the max of spatial logits over strided cells."""
    net = PactNet().eval()
    x = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        spatial = net.spatial(x)
        assert spatial.ndim == 4
        assert spatial.shape[0] == 2
        assert spatial.shape[1] == 1
        assert torch.equal(net(x), spatial.amax(dim=(2, 3)))


def test_shufflenet_singleton_batchnorm_accepts_unit_map() -> None:
    """ShuffleNet V2 trains on a 1x1 map; other backbones keep stock batch norm."""
    shuffle = build_backbone("shufflenetv2_x0_5").train()
    norms = [module for module in shuffle.modules() if isinstance(module, nn.BatchNorm2d)]
    assert norms
    assert all(module.__class__.__name__ == "_SingletonSafeBatchNorm2d" for module in norms)
    logits = shuffle(torch.zeros(1, 3, 32, 32))
    assert logits.shape == (1, 1)

    resnet = build_backbone("resnet18")
    resnet_norms = [module for module in resnet.modules() if isinstance(module, nn.BatchNorm2d)]
    assert resnet_norms
    assert all(module.__class__ is nn.BatchNorm2d for module in resnet_norms)
