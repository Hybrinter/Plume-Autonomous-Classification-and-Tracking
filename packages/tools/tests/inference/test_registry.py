"""Architecture registry tests."""

import pytest
import torch
from tools.inference.arch.dilated import DilatedSpec
from tools.inference.arch.registry import (
    EncoderUNetSpec,
    UNetSpec,
    build,
    default_arch,
    known,
    parse_segmentor,
    resolve_arch,
)
from tools.inference.cost import count_params


def test_known_pairs() -> None:
    """known() lists baseline, compact, dilated, and pretrained catalog points."""
    pairs = known()
    assert ("classifier", "resnet50") in pairs
    assert ("segmentor", "unet") in pairs
    assert ("classifier", "mobilenetv3_small_pt") in pairs
    assert ("segmentor", "unet_w16_sep") in pairs
    assert ("segmentor", "runet18_pt") in pairs
    assert ("segmentor", "dilatenet") in pairs
    assert ("segmentor", "dilatenet_w16") in pairs


def test_default_arch() -> None:
    """Defaults are pactnet and dilatenet."""
    assert default_arch("classifier") == "pactnet"
    assert default_arch("segmentor") == "dilatenet"
    with pytest.raises(ValueError, match="unknown train kind"):
        default_arch("nope")


def test_resolve_arch_fills_empty_and_rejects_unknown() -> None:
    """Empty arch selects the default. Names outside the grammar raise."""
    assert resolve_arch("classifier", "") == "pactnet"
    assert resolve_arch("segmentor", "") == "dilatenet"
    assert resolve_arch("segmentor", "unet") == "unet"
    with pytest.raises(ValueError, match="unknown segmentor architecture"):
        resolve_arch("segmentor", "resnet50")
    with pytest.raises(ValueError, match="unknown classifier backbone"):
        resolve_arch("classifier", "unet")
    with pytest.raises(ValueError, match="unknown train kind"):
        resolve_arch("nope", "unet")


def test_parse_segmentor_reads_the_modifier_grammar() -> None:
    """Bare names take the defaults; modifiers override one axis each."""
    assert parse_segmentor("unet") == UNetSpec(base_width=64, depth=4, separable=False)
    assert parse_segmentor("unet_w16_d3_sep") == UNetSpec(base_width=16, depth=3, separable=True)
    assert parse_segmentor("runet18") == EncoderUNetSpec(
        encoder="resnet18", pretrained=False, decoder_width=16, separable=False
    )
    assert parse_segmentor("runet50_pt_x32") == EncoderUNetSpec(
        encoder="resnet50", pretrained=True, decoder_width=32, separable=False
    )
    assert parse_segmentor("dilatenet_w32_s8") == DilatedSpec(
        base_width=32, blocks=4, output_stride=8, separable=True
    )


def test_parse_segmentor_rejects_malformed_modifiers() -> None:
    """A modifier that is not a known token or a positive integer raises."""
    with pytest.raises(ValueError, match="unknown unet modifier"):
        parse_segmentor("unet_zzz")
    with pytest.raises(ValueError, match="malformed width"):
        parse_segmentor("unet_w0")
    with pytest.raises(ValueError, match="malformed width"):
        parse_segmentor("unet_wide")
    with pytest.raises(ValueError, match="malformed depth"):
        parse_segmentor("unet_d")
    with pytest.raises(ValueError, match="unknown segmentor encoder"):
        parse_segmentor("runet99")
    with pytest.raises(ValueError, match="unknown runet modifier"):
        parse_segmentor("runet18_frozen")


def test_build_classifier_and_segmentor() -> None:
    """Registry build returns graphs with the flight I/O ranks."""
    clf = build("classifier", "resnet50", 4).eval()
    seg = build("segmentor", "unet", 4).eval()
    x = torch.zeros(1, 4, 32, 32)
    with torch.no_grad():
        assert clf(x).shape == (1, 1)
        assert seg(x).shape == (1, 1, 32, 32)


def test_build_empty_arch_uses_defaults() -> None:
    """Empty names construct pactnet logits and a dilatenet mask."""
    clf = build("classifier", "", 4).eval()
    seg = build("segmentor", "", 4).eval()
    x = torch.zeros(1, 4, 32, 32)
    with torch.no_grad():
        assert clf(x).shape == (1, 1)
        assert seg(x).shape == (1, 1, 32, 32)


@pytest.mark.parametrize(
    "arch",
    ["unet_w16", "unet_w16_sep", "unet_w16_d3", "runet18", "dilatenet_w16"],
)
def test_build_segmentor_variants_preserve_spatial_size(arch: str) -> None:
    """Every segmentor family returns a logit map at the input resolution."""
    seg = build("segmentor", arch, 4).eval()
    with torch.no_grad():
        assert seg(torch.zeros(1, 4, 64, 64)).shape == (1, 1, 64, 64)


@pytest.mark.parametrize(
    "arch",
    ["resnet18", "mobilenetv3_small", "efficientnet_b0", "shufflenetv2_x0_5"],
)
def test_build_classifier_backbones_emit_one_logit(arch: str) -> None:
    """Every classifier backbone takes four bands and emits a single logit."""
    clf = build("classifier", arch, 4).eval()
    with torch.no_grad():
        assert clf(torch.zeros(1, 4, 64, 64)).shape == (1, 1)


def test_width_and_separable_modifiers_shrink_the_segmentor() -> None:
    """The size knobs are what they claim: each one strictly reduces params."""
    baseline = count_params(build("segmentor", "unet", 4))
    narrow = count_params(build("segmentor", "unet_w16", 4))
    separable = count_params(build("segmentor", "unet_w16_sep", 4))
    shallow = count_params(build("segmentor", "unet_w16_d3", 4))
    assert narrow < baseline
    assert separable < narrow
    assert shallow < narrow
