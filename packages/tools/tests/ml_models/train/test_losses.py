"""Independent objective formulas for per-image loss instrumentation."""

import pytest
import torch
from tools.ml_models.train.losses import build_loss
from torch.nn import functional as functional


@pytest.mark.parametrize("name", ["bce", "dice", "bce_dice", "focal", "focal_dice"])
@pytest.mark.parametrize("shape", [(3, 1), (3, 1, 2, 4)])
def test_components_match_existing_configured_objective(name: str, shape: tuple[int, ...]) -> None:
    """Per-image components preserve the uniform-shape legacy objective and gradients."""
    logits = (
        torch.linspace(-2.0, 2.0, steps=torch.Size(shape).numel()).reshape(shape).requires_grad_()
    )
    targets = (torch.arange(logits.numel()).reshape(shape) % 3 == 0).float()
    objective = build_loss(name, pos_weight=2.0, focal_gamma=1.5, focal_alpha=0.3)
    raw = functional.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor([2.0]),
    )
    bce = raw.reshape(shape[0], -1).mean(1)
    probabilities = logits.sigmoid()
    true_probability = probabilities * targets + (1 - probabilities) * (1 - targets)
    weights = 0.3 * targets + 0.7 * (1 - targets)
    focal = (
        (
            weights
            * (1 - true_probability).pow(1.5)
            * functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        )
        .reshape(shape[0], -1)
        .mean(1)
    )
    flat_prob, flat_target = probabilities.reshape(shape[0], -1), targets.reshape(shape[0], -1)
    dice = 1 - (2 * (flat_prob * flat_target).sum(1) + 1) / (
        flat_prob.sum(1) + flat_target.sum(1) + 1
    )
    components = objective.per_sample_components(logits, targets)
    expected = torch.zeros(shape[0])
    if name in ("bce", "bce_dice"):
        assert components.bce is not None
        torch.testing.assert_close(components.bce, bce)
        assert components.focal is None
        expected = expected + bce
    elif name in ("focal", "focal_dice"):
        assert components.focal is not None
        torch.testing.assert_close(components.focal, focal)
        assert components.bce is None
        expected = expected + focal
    else:
        assert components.bce is None and components.focal is None
    if name in ("dice", "bce_dice", "focal_dice"):
        assert components.dice is not None
        torch.testing.assert_close(components.dice, dice)
        expected = expected + dice
    else:
        assert components.dice is None
    torch.testing.assert_close(components.total, expected)
    torch.testing.assert_close(objective(logits, targets), expected.mean())
    expected_gradient = torch.autograd.grad(expected.mean(), logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(objective(logits, targets), logits)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient)
