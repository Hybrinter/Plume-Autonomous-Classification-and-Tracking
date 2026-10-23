"""Tests for the dual-branch VIS+SWIR architecture variants."""

import pytest
import torch
from tools.ml_models.arch.compact import parse_compact
from tools.ml_models.arch.dilated import parse_dilated
from tools.ml_models.arch.dual import (
    DualDilatedSegmentor,
    DualPactNet,
    DualStem,
    shift_planes,
)
from tools.ml_models.arch.registry import build


def test_registry_builds_dual_variants_with_expected_shapes() -> None:
    """pactnet_dual and dilatenet_dual map a 4-plane stack to the usual outputs."""
    x = torch.rand(3, 4, 18, 27)
    gsd = torch.zeros(3, 2)
    classifier = build("classifier", "pactnet_dual", 4)
    segmentor = build("segmentor", "dilatenet_dual_s8", 4)
    assert isinstance(classifier, DualPactNet)
    assert isinstance(segmentor, DualDilatedSegmentor)
    assert classifier(x, gsd).shape == (3, 1)
    assert segmentor(x, gsd).shape == (3, 1, 18, 27)


def test_dual_modifier_parses_with_other_modifiers() -> None:
    """dual combines with width and depth tokens and defaults to off."""
    assert parse_compact("pactnet_dual_w32_d5").dual
    assert parse_compact("pactnet_dual_w32_d5").base_width == 32
    assert not parse_compact("pactnet").dual
    assert parse_dilated("dilatenet_w16_dual").dual


def test_dual_rejects_non_four_plane_input() -> None:
    """A dual graph needs BLUE, GREEN, RED, SWIR."""
    with pytest.raises(ValueError, match="4 input planes"):
        build("classifier", "pactnet_dual", 3)
    with pytest.raises(ValueError, match="4 input planes"):
        build("segmentor", "dilatenet_dual", 1)


def test_eval_mode_is_deterministic() -> None:
    """Jitter and modality dropout are off in eval mode."""
    torch.manual_seed(0)
    model = build("segmentor", "dilatenet_dual", 4).eval()
    x = torch.rand(4, 4, 25, 32)
    gsd = torch.zeros(4, 2)
    torch.testing.assert_close(model(x, gsd), model(x, gsd))


def test_shift_planes_moves_content_and_replicates_edges() -> None:
    """A +1 column shift moves a ramp right and repeats the first column."""
    ramp = torch.arange(5.0).repeat(1, 1, 3, 1)
    shifted = shift_planes(ramp, torch.tensor([[0, 1]]), 2)
    torch.testing.assert_close(shifted[0, 0, 0], torch.tensor([0.0, 0.0, 1.0, 2.0, 3.0]))
    torch.testing.assert_close(shift_planes(ramp, torch.tensor([[0, 0]]), 2), ramp)
    torch.testing.assert_close(shift_planes(ramp, torch.tensor([[1, 1]]), 0), ramp)


def test_modality_dropout_never_drops_both_branches() -> None:
    """At p = 0.5 every sample loses exactly one branch."""
    torch.manual_seed(0)
    stem = DualStem(4, 2, 8, separable=True, jitter_px=0, modality_dropout=0.5).train()
    vis_out: list[torch.Tensor] = []
    swir_out: list[torch.Tensor] = []
    stem.vis_down.register_forward_hook(lambda _m, _i, out: vis_out.append(out))
    stem.swir_down.register_forward_hook(lambda _m, _i, out: swir_out.append(out))
    seen: list[torch.Tensor] = []
    stem.fuse.register_forward_hook(lambda _m, inputs, _o: seen.append(inputs[0]))
    stem(torch.rand(32, 4, 16, 16) + 0.1, torch.zeros(32, 2))
    fused_in = seen[0]
    vis_alive = fused_in[:, : vis_out[0].shape[1]].abs().sum(dim=(1, 2, 3)) > 0
    swir_alive = fused_in[:, vis_out[0].shape[1] :].abs().sum(dim=(1, 2, 3)) > 0
    assert torch.all(vis_alive ^ swir_alive)


def test_stem_rejects_bad_perturbation_settings() -> None:
    """Negative jitter and dropout above 0.5 raise."""
    with pytest.raises(ValueError):
        DualStem(4, 2, 8, separable=True, jitter_px=-1)
    with pytest.raises(ValueError):
        DualStem(4, 2, 8, separable=True, modality_dropout=0.6)
    with pytest.raises(ValueError):
        DualPactNet(16, 1, separable=True)


def test_nodrop_disables_modality_dropout_and_needs_dual() -> None:
    """nodrop sets the dual stem dropout to zero and is rejected without dual."""
    classifier = build("classifier", "pactnet_dual_nodrop", 4)
    segmentor = build("segmentor", "dilatenet_dual_nodrop", 4)
    assert isinstance(classifier, DualPactNet)
    assert isinstance(segmentor, DualDilatedSegmentor)
    assert classifier.stem.modality_dropout == 0.0
    assert segmentor.stem.modality_dropout == 0.0
    assert classifier.stem.jitter_px == 1
    default = build("classifier", "pactnet_dual", 4)
    assert isinstance(default, DualPactNet)
    assert default.stem.modality_dropout == 0.1
    with pytest.raises(ValueError, match="nodrop needs dual"):
        parse_compact("pactnet_nodrop")
    with pytest.raises(ValueError, match="nodrop needs dual"):
        parse_dilated("dilatenet_nodrop")
