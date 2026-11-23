"""Two-input dense Conv2d/Linear operation lower bounds and parameter evidence.

Forward hooks count twice the dense multiply-accumulate count, excluding
bias additions and all other module/functional operations. These partial
counts are not total graph FLOPs or a latency estimate. Zero encoded GSD
represents the dataset reference GSD. Dummy images are unit-domain zeros.
Measurement preserves module modes, RNG state, device and dtype.

Contains:
  - count_params: number of parameters in a module.
  - measure_resources: explicitly partial per-shape resource evidence.
  - count_flops: compatibility accessor for the same partial count.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from flight.libs.types import Err, Ok, Result

if TYPE_CHECKING:
    from torch import nn


@dataclass(frozen=True, slots=True)
class OperationCost:
    """One executed supported module call; reused modules have separate calls."""

    module: str
    operation: str
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    flops: int


@dataclass(frozen=True, slots=True)
class ResourceEvidence:
    """Parameter counts and a dense module-operation lower bound for one batch."""

    parameters: int
    trainable_parameters: int
    image_shape: tuple[int, ...]
    gsd_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    counted_flops: int
    calls: tuple[OperationCost, ...]
    uncounted_leaf_modules: tuple[str, ...]
    coverage: Literal["PARTIAL"] = "PARTIAL"
    method: str = "dense_conv2d_linear_multiply_accumulate_times_two"
    limitations: tuple[str, ...] = (
        "Only executed Conv2d and Linear dense multiply-accumulate terms are counted.",
        "Bias, activation, normalization, pooling, interpolation, FiLM arithmetic, "
        "functional operators and all other operations are excluded.",
        "Dense convolution counts include padded positions; this is not a hardware counter.",
        "Functional operations are not enumerated by module hooks.",
        "Zero encoded GSD is the reference GSD; data-dependent paths may differ.",
        "Forward-only batch count; no optimizer/backward/memory or latency measurement.",
        "Counts are per model and input shape, not paired-pipeline flight qualification.",
    )


def count_params(model: nn.Module) -> int:
    """Return the number of parameters in ``model``.

    Args:
        model: A torch module.

    Returns:
        int: Sum of ``numel()`` over all parameters.
    """
    return int(sum(item.numel() for item in model.parameters()))


def measure_resources(
    model: nn.Module, input_shape: tuple[int, ...]
) -> Result[ResourceEvidence, str]:
    """Call ``model(image, encoded_gsd)`` and count supported dense module operations.

    Args:
        model: Two-input module, measured on its current device/dtype.
        input_shape: Positive ``(batch, channels, height, width)``.

    Returns:
        Result[ResourceEvidence, str]: Partial resource counts or an explicit failure.
    """
    import torch
    from torch import nn

    if len(input_shape) != 4 or any(type(value) is not int or value < 1 for value in input_shape):
        return Err("resource image shape must be positive integer (B,C,H,W)")
    modes = tuple((module, module.training) for module in model.modules())
    parameters = tuple(model.parameters())
    tensors = parameters + tuple(model.buffers())
    device = tensors[0].device if tensors else torch.device("cpu")
    dtypes = {tensor.dtype for tensor in tensors if tensor.is_floating_point()}
    if len({tensor.device for tensor in tensors}) > 1 or len(dtypes) > 1:
        return Err("resource analysis requires a single device and floating dtype")
    dtype = next(iter(dtypes), torch.float32)
    calls: list[OperationCost] = []
    uncounted: set[str] = set()
    names = {module: name or "<root>" for name, module in model.named_modules()}

    def observe(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        """Capture one leaf call, counting only the stated operation subset."""
        if not isinstance(module, nn.Conv2d | nn.Linear):
            uncounted.add(f"{names[module]}:{type(module).__name__}")
            return
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            raise ValueError("supported module has no tensor input")
        if not isinstance(output, torch.Tensor):
            raise ValueError("supported module has no tensor output")
        terms = (
            (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            if isinstance(module, nn.Conv2d)
            else module.in_features
        )
        calls.append(
            OperationCost(
                module=names[module],
                operation=type(module).__name__,
                input_shape=tuple(inputs[0].shape),
                output_shape=tuple(output.shape),
                flops=2 * output.numel() * terms,
            )
        )

    hooks = [
        module.register_forward_hook(observe)
        for module in model.modules()
        if not tuple(module.children())
    ]
    devices = [device.index or 0] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            model.eval()
            image = torch.zeros(input_shape, dtype=dtype, device=device)
            encoded_gsd = torch.zeros((input_shape[0], 2), dtype=dtype, device=device)
            output = model(image, encoded_gsd)
            if not isinstance(output, torch.Tensor) or not bool(torch.isfinite(output).all()):
                return Err("resource forward must return one finite tensor")
            if output.ndim < 1 or output.shape[0] != input_shape[0]:
                return Err("resource output batch disagrees with image batch")
            return Ok(
                ResourceEvidence(
                    parameters=sum(item.numel() for item in parameters),
                    trainable_parameters=sum(
                        item.numel() for item in parameters if item.requires_grad
                    ),
                    image_shape=input_shape,
                    gsd_shape=(input_shape[0], 2),
                    output_shape=tuple(output.shape),
                    counted_flops=sum(call.flops for call in calls),
                    calls=tuple(calls),
                    uncounted_leaf_modules=tuple(sorted(uncounted)),
                )
            )
    except (ValueError, TypeError, RuntimeError, OverflowError) as exc:
        return Err(f"resource measurement failed: {exc}")
    finally:
        for hook in hooks:
            hook.remove()
        for module, training in modes:
            module.training = training


def count_flops(model: nn.Module, input_shape: tuple[int, ...]) -> Result[int, str]:
    """Return only the partial dense Conv2d/Linear count; not total FLOPs or latency."""
    measured = measure_resources(model, input_shape)
    if isinstance(measured, Err):
        return measured
    return Ok(measured.value.counted_flops)
