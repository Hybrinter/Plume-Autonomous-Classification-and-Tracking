"""Independent dense Conv2d/Linear operation-count examples."""

from typing import cast

import pytest
import torch
from flight.libs.types import Err, Ok
from tools.ml_models.analysis.cost import count_flops, count_params, measure_resources
from torch import nn


class _TwoInput(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(4, 6, 3, groups=2, bias=True)
        self.linear = nn.Linear(2, 6, bias=False)
        self.relu = nn.ReLU()
        self.conv.weight.requires_grad_(False)
        self.gsd: torch.Tensor | None = None

    def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
        self.gsd = gsd.detach().clone()
        return cast(
            torch.Tensor,
            self.relu(self.conv(image)) + self.linear(gsd)[..., None, None],
        )


@pytest.mark.parametrize("batch", [1, 3])
def test_two_input_counter_has_explicit_partial_counts_and_shapes(batch: int) -> None:
    model = _TwoInput()
    model.relu.eval()
    modes = tuple(module.training for module in model.modules())
    rng = torch.get_rng_state().clone()
    result = measure_resources(model, (batch, 4, 5, 7))
    assert isinstance(result, Ok)
    evidence = result.value
    assert evidence.parameters == 126
    assert count_params(model) == 126
    assert evidence.trainable_parameters == 18
    assert evidence.image_shape == (batch, 4, 5, 7)
    assert evidence.gsd_shape == (batch, 2)
    assert evidence.output_shape == (batch, 6, 3, 5)
    assert evidence.counted_flops == batch * (3240 + 24)
    assert tuple(call.flops for call in evidence.calls) == (batch * 3240, batch * 24)
    assert evidence.coverage == "PARTIAL"
    assert evidence.uncounted_leaf_modules == ("relu:ReLU",)
    assert any("latency" in note for note in evidence.limitations)
    assert model.gsd is not None and torch.equal(model.gsd, torch.zeros(batch, 2))
    assert tuple(module.training for module in model.modules()) == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(not module._forward_hooks for module in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())
    counted = count_flops(model, (batch, 4, 5, 7))
    assert isinstance(counted, Ok) and counted.value == evidence.counted_flops


def test_reused_module_calls_are_counted_not_deduplicated() -> None:
    class Twice(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(2, 2)

        def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
            return cast(torch.Tensor, self.linear(gsd) + self.linear(gsd))

    result = measure_resources(Twice(), (1, 1, 3, 3))
    assert isinstance(result, Ok)
    assert result.value.parameters == 6
    assert result.value.counted_flops == 16
    assert len(result.value.calls) == 2


def test_error_restores_mixed_modes_rng_and_removes_hooks() -> None:
    class Failure(_TwoInput):
        def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
            super().forward(image, gsd)
            torch.rand(3)
            raise RuntimeError("injected")

    model = Failure()
    model.relu.eval()
    modes = tuple(module.training for module in model.modules())
    rng = torch.get_rng_state().clone()
    result = measure_resources(model, (1, 4, 5, 7))
    assert isinstance(result, Err) and "injected" in result.error
    assert tuple(module.training for module in model.modules()) == modes
    assert torch.equal(torch.get_rng_state(), rng)
    assert all(not module._forward_hooks for module in model.modules())


def test_functional_only_graph_cannot_be_mislabeled_total_flops() -> None:
    class Functional(nn.Module):
        def forward(self, image: torch.Tensor, gsd: torch.Tensor) -> torch.Tensor:
            return image * gsd[:, :1, None, None].exp()

    result = measure_resources(Functional(), (1, 1, 2, 3))
    assert isinstance(result, Ok)
    assert result.value.counted_flops == 0
    assert result.value.coverage == "PARTIAL"
    assert result.value.uncounted_leaf_modules == ("<root>:Functional",)


@pytest.mark.parametrize("shape", [(1, 3, 2), (1, 0, 2, 2), (True, 4, 5, 7)])
def test_invalid_shapes_are_not_run(shape: tuple[int, ...]) -> None:
    assert isinstance(measure_resources(_TwoInput(), shape), Err)
