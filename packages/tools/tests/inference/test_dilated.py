"""Dilated dilatenet segmentor tests."""

import pytest
import torch
from tools.inference.arch.dilated import (
    DEFAULT_DILATED_BLOCKS,
    DEFAULT_DILATED_WIDTH,
    DEFAULT_OUTPUT_STRIDE,
    DilatedSegmentor,
    DilatedSpec,
    build_dilated_segmentor,
    dilation_rates,
    parse_dilated,
)
from tools.inference.arch.registry import (
    EncoderUNetSpec,
    UNetSpec,
    build,
    known,
    parse_segmentor,
    resolve_arch,
)
from tools.inference.cost import count_params


def test_parse_dilated_defaults() -> None:
    """Bare dilatenet uses width 32, blocks 4, output stride 4, and separable convolutions."""
    assert parse_dilated("dilatenet") == DilatedSpec(
        base_width=DEFAULT_DILATED_WIDTH,
        blocks=DEFAULT_DILATED_BLOCKS,
        output_stride=DEFAULT_OUTPUT_STRIDE,
        separable=True,
    )


def test_parse_dilated_width_modifier() -> None:
    """w<N> overrides the stem width."""
    assert parse_dilated("dilatenet_w16").base_width == 16


def test_parse_dilated_blocks_modifier() -> None:
    """d<N> overrides the dilated-block count."""
    assert parse_dilated("dilatenet_d6").blocks == 6


def test_parse_dilated_output_stride_modifier() -> None:
    """s<N> overrides the output stride."""
    assert parse_dilated("dilatenet_s8").output_stride == 8


def test_parse_dilated_full_modifier() -> None:
    """full selects dense convolutions."""
    assert parse_dilated("dilatenet_full").separable is False


@pytest.mark.parametrize(
    "name",
    [
        "dilatenet_w16_d6_s8_full",
        "dilatenet_full_s8_d6_w16",
        "dilatenet_s8_w16_d6_full",
    ],
)
def test_parse_dilated_modifiers_combine_in_any_order(name: str) -> None:
    """Width, block count, output stride, and full modifiers apply regardless of order."""
    spec = parse_dilated(name)
    assert spec == DilatedSpec(
        base_width=16,
        blocks=6,
        output_stride=8,
        separable=False,
    )


def test_parse_dilated_rejects_non_dilatenet_head() -> None:
    """A name outside the dilatenet family raises ValueError."""
    with pytest.raises(ValueError, match="unknown dilated segmentor"):
        parse_dilated("unet_w16")


def test_parse_dilated_rejects_unknown_modifiers() -> None:
    """Unknown modifier tokens raise ValueError."""
    with pytest.raises(ValueError, match="unknown dilatenet modifier"):
        parse_dilated("dilatenet_zzz")


@pytest.mark.parametrize("token", ["w0", "d0", "wx", "d"])
def test_parse_dilated_rejects_malformed_numeric_tokens(token: str) -> None:
    """Zero, empty, or non-numeric width and block tokens raise ValueError."""
    with pytest.raises(ValueError, match="malformed"):
        parse_dilated(f"dilatenet_{token}")


@pytest.mark.parametrize("name", ["dilatenet_s5", "dilatenet_s16"])
def test_parse_dilated_rejects_unsupported_output_stride(name: str) -> None:
    """Output strides other than 4 or 8 raise ValueError."""
    with pytest.raises(ValueError, match="output stride must be one of"):
        parse_dilated(name)


def test_dilation_rates_doubles_from_one() -> None:
    """Rates are powers of two starting at one."""
    assert dilation_rates(4) == (1, 2, 4, 8)


def test_dilation_rates_returns_blocks_entries() -> None:
    """The tuple length matches the requested block count."""
    assert len(dilation_rates(3)) == 3


def test_dilation_rates_rejects_blocks_below_one() -> None:
    """A block count below one raises ValueError."""
    with pytest.raises(ValueError, match="blocks must be at least 1"):
        dilation_rates(0)


@pytest.mark.parametrize("size", [128, 256])
def test_dilated_segmentor_forward_preserves_spatial_size(size: int) -> None:
    """Forward maps (N, C, H, W) to (N, 1, H, W) at the input resolution."""
    net = DilatedSegmentor(in_channels=4).eval()
    x = torch.zeros(2, 4, size, size)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (2, 1, size, size)


def test_dilated_segmentor_forward_non_square() -> None:
    """Forward preserves non-square height and width."""
    net = DilatedSegmentor(in_channels=4).eval()
    x = torch.zeros(1, 4, 96, 160)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 1, 96, 160)


def test_dilated_segmentor_accepts_non_default_in_channels() -> None:
    """A band count other than four is accepted."""
    net = DilatedSegmentor(in_channels=6).eval()
    x = torch.zeros(1, 6, 64, 64)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 1, 64, 64)


def test_dilated_segmentor_output_stride_eight_preserves_spatial_size() -> None:
    """Output stride 8 still returns a full-resolution logit map."""
    net = DilatedSegmentor(output_stride=8).eval()
    x = torch.zeros(1, 4, 128, 128)
    with torch.no_grad():
        y = net(x)
    assert y.shape == (1, 1, 128, 128)


def test_dilated_segmentor_rejects_invalid_base_width_and_output_stride() -> None:
    """Invalid base width and output stride raise ValueError."""
    with pytest.raises(ValueError, match="base_width must be at least 1"):
        DilatedSegmentor(base_width=0)
    with pytest.raises(ValueError, match="output_stride must be one of"):
        DilatedSegmentor(output_stride=16)


def test_build_dilated_segmentor_honours_spec() -> None:
    """The builder wires width, blocks, output stride, and separable into the module."""
    spec = DilatedSpec(base_width=16, blocks=3, output_stride=8, separable=False)
    net = build_dilated_segmentor(spec, in_channels=5)
    assert isinstance(net, DilatedSegmentor)
    x = torch.zeros(1, 5, 64, 64)
    with torch.no_grad():
        assert net(x).shape == (1, 1, 64, 64)


def test_build_dilated_segmentor_full_has_more_params_than_separable() -> None:
    """Dense convolutions add parameters at the same width and block count."""
    spec_sep = DilatedSpec(base_width=32, blocks=4, output_stride=4, separable=True)
    spec_full = DilatedSpec(base_width=32, blocks=4, output_stride=4, separable=False)
    sep_params = count_params(build_dilated_segmentor(spec_sep))
    full_params = count_params(build_dilated_segmentor(spec_full))
    assert full_params > sep_params


def test_build_dilated_segmentor_wider_has_more_params() -> None:
    """A wider stem yields more parameters at equal depth."""
    narrow = count_params(build_dilated_segmentor(DilatedSpec(16, 4, 4, True)))
    wide = count_params(build_dilated_segmentor(DilatedSpec(64, 4, 4, True)))
    assert wide > narrow


def test_registry_build_dilatenet_w16() -> None:
    """Registry build returns a working dilatenet segmentor."""
    seg = build("segmentor", "dilatenet_w16", 4).eval()
    with torch.no_grad():
        assert seg(torch.zeros(1, 4, 64, 64)).shape == (1, 1, 64, 64)


def test_parse_segmentor_dispatches_by_family() -> None:
    """dilatenet, unet, and runet names parse to their respective specs."""
    dilated = parse_segmentor("dilatenet_w32_d6")
    scratch = parse_segmentor("unet_w16")
    encoder = parse_segmentor("runet18_pt")
    assert isinstance(dilated, DilatedSpec)
    assert dilated == DilatedSpec(32, 6, 4, True)
    assert isinstance(scratch, UNetSpec)
    assert scratch.base_width == 16
    assert isinstance(encoder, EncoderUNetSpec)
    assert encoder == EncoderUNetSpec("resnet18", True, 16, False)


def test_known_dilatenet_entries_resolve_and_build() -> None:
    """Every catalog dilatenet segmentor parses and constructs."""
    for kind, name in sorted(known()):
        if kind != "segmentor" or not name.startswith("dilatenet"):
            continue
        assert resolve_arch(kind, name) == name
        seg = build(kind, name, 4).eval()
        with torch.no_grad():
            assert seg(torch.zeros(1, 4, 64, 64)).shape == (1, 1, 64, 64)
