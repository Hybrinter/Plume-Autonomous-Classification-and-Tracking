"""Training objective tests."""

import pytest
import torch
from tools.inference.losses import (
    LOSS_NAMES,
    LOSS_SPECS,
    LossSpec,
    PlumeLoss,
    build_loss,
    dice_term,
    focal_term,
)


def test_dice_term_near_zero_on_confident_match() -> None:
    """Confident correct logits yield a Dice loss near zero."""
    targets = torch.ones(2, 1, 8, 8)
    logits = torch.full((2, 1, 8, 8), 12.0)
    assert dice_term(logits, targets).item() < 0.01


def test_dice_term_near_one_on_confident_inversion() -> None:
    """Confident inverted logits yield a Dice loss near one."""
    targets = torch.ones(2, 1, 8, 8)
    logits = torch.full((2, 1, 8, 8), -12.0)
    assert dice_term(logits, targets).item() > 0.98


def test_dice_term_all_negative_smoothing() -> None:
    """All-negative target and prediction stay at zero loss via smoothing."""
    targets = torch.zeros(2, 1, 8, 8)
    logits = torch.full((2, 1, 8, 8), -12.0)
    assert dice_term(logits, targets).item() < 0.01


def test_focal_term_gamma_zero_equals_half_bce() -> None:
    """gamma=0 and alpha=0.5 reduce focal loss to 0.5 times plain BCE."""
    logits = torch.tensor([[2.0, -1.0], [0.5, -2.0]])
    targets = torch.tensor([[1.0, 0.0], [1.0, 1.0]])
    focal = focal_term(logits, targets, gamma=0.0, alpha=0.5)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert torch.allclose(focal, 0.5 * bce)


def test_focal_term_gamma_focuses_confident_correct() -> None:
    """Positive gamma lowers loss on confidently-correct pixels."""
    logits = torch.tensor([[8.0], [8.0], [-8.0], [-8.0]])
    targets = torch.tensor([[1.0], [0.0], [0.0], [1.0]])
    unfocused = focal_term(logits, targets, gamma=0.0, alpha=0.5)
    focused = focal_term(logits, targets, gamma=2.0, alpha=0.5)
    assert focused.item() < unfocused.item()


@pytest.mark.parametrize("name", sorted(LOSS_NAMES))
def test_build_loss_accepts_registered_names(name: str) -> None:
    """Every registered loss name constructs a PlumeLoss."""
    loss = build_loss(name)
    assert isinstance(loss, PlumeLoss)


def test_build_loss_rejects_unknown_name() -> None:
    """An unregistered name raises ValueError."""
    with pytest.raises(ValueError, match="unknown loss"):
        build_loss("nope")


def test_loss_specs_cover_registered_names() -> None:
    """LOSS_SPECS holds one row per registered name."""
    assert set(LOSS_SPECS) == LOSS_NAMES
    assert LOSS_SPECS["focal"] == LossSpec(use_focal=True, dice_weight=0.0, pixel_weight=1.0)
    assert LOSS_SPECS["focal_dice"] == LossSpec(use_focal=True, dice_weight=1.0, pixel_weight=1.0)


def test_build_loss_weight_configuration() -> None:
    """Registered names set pixel and Dice weights as documented."""
    dice = build_loss("dice")
    assert dice.pixel_weight == 0.0
    assert dice.dice_weight == 1.0

    bce = build_loss("bce")
    assert bce.pixel_weight == 1.0
    assert bce.dice_weight == 0.0

    combined = build_loss("bce_dice")
    assert combined.pixel_weight == 1.0
    assert combined.dice_weight == 1.0


def test_plume_loss_forward_scalar_shapes() -> None:
    """Forward returns a scalar for segmentor and classifier tensor ranks."""
    loss_fn = build_loss("bce_dice")
    seg_logits = torch.randn(2, 1, 8, 8)
    seg_targets = torch.randint(0, 2, (2, 1, 8, 8)).float()
    seg_out = loss_fn(seg_logits, seg_targets)
    assert seg_out.ndim == 0
    assert seg_out.shape == torch.Size([])

    clf_logits = torch.randn(4, 1)
    clf_targets = torch.randint(0, 2, (4, 1)).float()
    clf_out = loss_fn(clf_logits, clf_targets)
    assert clf_out.ndim == 0
    assert clf_out.shape == torch.Size([])


def test_plume_loss_pos_weight_raises_positive_miss_loss() -> None:
    """A positive pos_weight penalises missed positives more than zero does."""
    targets = torch.ones(2, 1, 4, 4)
    logits = torch.full((2, 1, 4, 4), -4.0)
    unweighted = build_loss("bce", pos_weight=0.0)(logits, targets)
    weighted = build_loss("bce", pos_weight=5.0)(logits, targets)
    assert weighted.item() > unweighted.item()


def test_plume_loss_backward_propagates_gradients() -> None:
    """Combined loss backpropagates non-zero gradients into logits."""
    logits = torch.randn(2, 1, 4, 4, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 4, 4)).float()
    loss = build_loss("bce_dice")(logits, targets)
    loss.backward()
    assert logits.grad is not None
    assert not torch.all(logits.grad == 0)
