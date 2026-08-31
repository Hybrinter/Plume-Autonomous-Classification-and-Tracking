"""Compact pactnet classifier tests."""

import pytest
import torch
from tools.inference.arch.classifier import BackboneName, BackboneSpec
from tools.inference.arch.compact import (
    DEFAULT_COMPACT_DEPTH,
    DEFAULT_COMPACT_WIDTH,
    CompactSpec,
    PactNet,
    build_compact_classifier,
    compact_stage_widths,
    parse_compact,
)
from tools.inference.arch.registry import build, known, parse_classifier, resolve_arch
from tools.inference.cost import count_params


def test_parse_compact_defaults() -> None:
    """Bare pactnet uses width 16, depth 4, and separable convolutions."""
    assert parse_compact("pactnet") == CompactSpec(
        base_width=DEFAULT_COMPACT_WIDTH,
        depth=DEFAULT_COMPACT_DEPTH,
        separable=True,
    )


def test_parse_compact_width_modifier() -> None:
    """w<N> overrides the stem width."""
    assert parse_compact("pactnet_w32").base_width == 32


def test_parse_compact_depth_modifier() -> None:
    """d<N> overrides the stage count."""
    assert parse_compact("pactnet_d5").depth == 5


def test_parse_compact_full_modifier() -> None:
    """full selects dense convolutions."""
    assert parse_compact("pactnet_full").separable is False


@pytest.mark.parametrize(
    "name",
    [
        "pactnet_w8_d5_full",
        "pactnet_full_d5_w8",
        "pactnet_d5_w8_full",
    ],
)
def test_parse_compact_modifiers_combine_in_any_order(name: str) -> None:
    """Width, depth, and full modifiers apply regardless of order."""
    spec = parse_compact(name)
    assert spec == CompactSpec(base_width=8, depth=5, separable=False)


def test_parse_compact_rejects_unknown_modifiers() -> None:
    """Unknown modifier tokens raise ValueError."""
    with pytest.raises(ValueError, match="unknown pactnet modifier"):
        parse_compact("pactnet_zzz")


def test_parse_compact_rejects_non_pactnet_head() -> None:
    """A name outside the pactnet family raises ValueError."""
    with pytest.raises(ValueError, match="unknown compact classifier"):
        parse_compact("resnet18")


@pytest.mark.parametrize("token", ["w0", "d0", "wx", "d"])
def test_parse_compact_rejects_malformed_numeric_tokens(token: str) -> None:
    """Zero, empty, or non-numeric width and depth tokens raise ValueError."""
    with pytest.raises(ValueError, match="malformed"):
        parse_compact(f"pactnet_{token}")


def test_compact_stage_widths_doubles_per_stage() -> None:
    """Each stage doubles the channel count up to the ceiling."""
    assert compact_stage_widths(16, 4) == (16, 32, 64, 128)


def test_compact_stage_widths_saturates_at_ceiling() -> None:
    """Wide or deep inputs cap each stage at 256 channels."""
    assert compact_stage_widths(128, 4) == (128, 256, 256, 256)
    assert compact_stage_widths(16, 8) == (16, 32, 64, 128, 256, 256, 256, 256)


def test_compact_stage_widths_returns_depth_entries() -> None:
    """The tuple length matches the requested stage count."""
    assert len(compact_stage_widths(8, 3)) == 3


@pytest.mark.parametrize("size", [128, 256])
def test_pactnet_forward_output_shape(size: int) -> None:
    """Forward maps (N, C, H, W) to (N, 1) logits."""
    net = PactNet(in_channels=4).eval()
    x = torch.zeros(2, 4, size, size)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 1)


def test_pactnet_accepts_non_default_in_channels() -> None:
    """A band count other than four is accepted."""
    net = PactNet(in_channels=6).eval()
    x = torch.zeros(1, 6, 64, 64)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 1)


def test_pactnet_depth_one_works() -> None:
    """A single-stage stack still emits one logit per sample."""
    net = PactNet(depth=1).eval()
    x = torch.zeros(1, 4, 32, 32)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 1)


def test_pactnet_rejects_invalid_depth_and_width() -> None:
    """Depth and base width below one raise ValueError."""
    with pytest.raises(ValueError, match="depth must be at least 1"):
        PactNet(depth=0)
    with pytest.raises(ValueError, match="base_width must be at least 1"):
        PactNet(base_width=0)


def test_build_compact_classifier_honours_spec() -> None:
    """The builder wires width, depth, and separable into the module."""
    spec = CompactSpec(base_width=8, depth=3, separable=False)
    net = build_compact_classifier(spec, in_channels=5)
    assert isinstance(net, PactNet)
    assert net.head.in_features == compact_stage_widths(8, 3)[-1]
    x = torch.zeros(1, 5, 32, 32)
    with torch.no_grad():
        assert net(x).shape == (1, 1)


def test_build_compact_classifier_full_has_more_params_than_separable() -> None:
    """Dense convolutions add parameters at the same width and depth."""
    spec_sep = CompactSpec(base_width=16, depth=4, separable=True)
    spec_full = CompactSpec(base_width=16, depth=4, separable=False)
    sep_params = count_params(build_compact_classifier(spec_sep))
    full_params = count_params(build_compact_classifier(spec_full))
    assert full_params > sep_params


def test_registry_build_pactnet_w8() -> None:
    """Registry build returns a working pactnet classifier."""
    clf = build("classifier", "pactnet_w8", 4).eval()
    with torch.no_grad():
        assert clf(torch.zeros(1, 4, 64, 64)).shape == (1, 1)


def test_parse_classifier_dispatches_by_family() -> None:
    """pactnet names parse to CompactSpec; torchvision names to BackboneSpec."""
    compact = parse_classifier("pactnet_w32")
    backbone = parse_classifier("resnet18_pt")
    assert isinstance(compact, CompactSpec)
    assert compact.base_width == 32
    assert isinstance(backbone, BackboneSpec)
    assert backbone == BackboneSpec(backbone=BackboneName.resnet18, pretrained=True)


def test_known_pactnet_entries_resolve_and_build() -> None:
    """Every catalog pactnet classifier parses and constructs."""
    for kind, name in sorted(known()):
        if kind != "classifier" or not name.startswith("pactnet"):
            continue
        assert resolve_arch(kind, name) == name
        net = build(kind, name, 4).eval()
        with torch.no_grad():
            assert net(torch.zeros(1, 4, 64, 64)).shape == (1, 1)
