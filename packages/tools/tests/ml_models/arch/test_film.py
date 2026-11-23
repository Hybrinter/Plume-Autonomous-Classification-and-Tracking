"""Tests for GSD FiLM conditioning and the registry call signature."""

import torch
from tools.ml_models.arch.film import (
    CONDITIONING_ID,
    GsdFilm,
    IgnoreGsd,
    resolve_gsd,
)
from tools.ml_models.arch.registry import build
from torch import nn


def test_film_identity_at_zero_init() -> None:
    """A fresh GsdFilm leaves its features untouched for any GSD."""
    torch.manual_seed(0)
    film = GsdFilm(8)
    features = torch.randn(3, 8, 5, 7)
    gsd = torch.randn(3, 2)
    out = film(features, gsd)
    assert torch.equal(out, features)


def test_resolve_gsd_zeros_on_input_device() -> None:
    """The fallback encoding matches the batch, device, and dtype of x."""
    x = torch.randn(4, 3, 8, 8, dtype=torch.float64)
    gsd = resolve_gsd(x, None)
    assert gsd.shape == (4, 2)
    assert gsd.dtype == x.dtype
    assert gsd.device == x.device
    assert torch.equal(gsd, torch.zeros(4, 2, dtype=x.dtype))
    given = torch.ones(4, 2)
    assert resolve_gsd(x.float(), given) is given


def test_ignore_gsd_forwards_image_only() -> None:
    """The wrapper calls the inner graph without the GSD argument."""
    inner = nn.Sequential(nn.Flatten(), nn.LazyLinear(4))
    wrapped = IgnoreGsd(inner)
    image = torch.randn(2, 3, 8, 8)
    assert wrapped(image, torch.ones(2, 2)).shape == (2, 4)
    assert wrapped(image).shape == (2, 4)


def _train_one_step(model: nn.Module, gsd: torch.Tensor) -> None:
    """Take one optimizer step so the zero-initialised FiLM terms open."""
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    image = torch.randn(gsd.shape[0], 3, 24, 32)
    output: torch.Tensor = model(image, gsd)
    loss = output.square().mean()
    optimizer.zero_grad()
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()


def test_pactnet_output_depends_on_gsd_after_update() -> None:
    """After one step, two different GSD rows give different pactnet logits."""
    torch.manual_seed(0)
    model = build("classifier", "pactnet_w8_d2", 3)
    _train_one_step(model, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
    model.eval()
    image = torch.randn(2, 3, 24, 32)
    with torch.no_grad():
        low: torch.Tensor = model(image, torch.zeros(2, 2))
        high: torch.Tensor = model(image, torch.full((2, 2), 2.0))
    assert not torch.allclose(low, high)


def test_dilatenet_output_depends_on_gsd_after_update() -> None:
    """After one step, two different GSD rows give different dilatenet logits."""
    torch.manual_seed(0)
    model = build("segmentor", "dilatenet_w8_d2", 3)
    _train_one_step(model, torch.tensor([[1.0, 1.0], [1.0, 1.0]]))
    model.eval()
    image = torch.randn(2, 3, 24, 32)
    with torch.no_grad():
        low: torch.Tensor = model(image, torch.zeros(2, 2))
        high: torch.Tensor = model(image, torch.full((2, 2), 2.0))
    assert not torch.allclose(low, high)


def test_conditioned_families_accept_variable_shapes() -> None:
    """Batch and H/W are read from the input, not frozen in the graph."""
    model = build("classifier", "pactnet_w8_d2", 3)
    segmentor = build("segmentor", "dilatenet_w8_d2", 3)
    for batch, height, width in ((1, 32, 48), (3, 40, 56)):
        image = torch.randn(batch, 3, height, width)
        gsd = torch.randn(batch, 2)
        out: torch.Tensor = model(image, gsd)
        assert out.shape == (batch, 1)
        seg: torch.Tensor = segmentor(image, gsd)
        assert seg.shape == (batch, 1, height, width)


def test_legacy_single_input_paths() -> None:
    """``model(image)`` still works and equals the reference-GSD call."""
    torch.manual_seed(0)
    model = build("classifier", "pactnet_w8_d2", 3)
    model.eval()
    image = torch.randn(2, 3, 24, 32)
    legacy: torch.Tensor = model(image)
    reference: torch.Tensor = model(image, torch.zeros(2, 2))
    assert torch.equal(legacy, reference)
    assert legacy.shape == (2, 1)


def test_registry_wraps_unconditioned_families() -> None:
    """Non-conditioned architectures build behind IgnoreGsd; the flight
    families build as real conditioned graphs."""
    torch.manual_seed(0)
    assert not isinstance(build("classifier", "pactnet", 3), IgnoreGsd)
    assert not isinstance(build("segmentor", "dilatenet", 3), IgnoreGsd)
    assert isinstance(build("classifier", "resnet18", 3), IgnoreGsd)
    assert isinstance(build("segmentor", "unet_w8_d3", 3), IgnoreGsd)


def test_conditioning_id_marker() -> None:
    """The checkpoint marker for the conditioned families is stable."""
    assert CONDITIONING_ID == "film-log-gsd-v1"
