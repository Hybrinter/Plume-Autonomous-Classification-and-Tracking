"""Preferred onnxruntime execution-provider list for accept and bench.

The preference order is TensorRT, then CUDA, then CPU. The returned list is
the intersection with providers this runtime actually has. Flight session load
does not call this helper; it leaves provider selection to onnxruntime.

Contains:
  - PREFERRED_ORT_PROVIDERS: preference order.
  - resolve_ort_providers: available providers in that order.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

PREFERRED_ORT_PROVIDERS: tuple[str, ...] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
)


def resolve_ort_providers() -> list[str]:
    """Return preferred onnxruntime providers that this install actually has.

    Returns:
        list[str]: Non-empty subset of ``PREFERRED_ORT_PROVIDERS``.

    Raises:
        ImportError: If onnxruntime is not installed.
        RuntimeError: If none of the preferred providers are available.
    """
    try:
        import onnxruntime
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required to resolve execution providers; install pact-tools[export]"
        ) from exc
    available = set(onnxruntime.get_available_providers())
    selected = [name for name in PREFERRED_ORT_PROVIDERS if name in available]
    if not selected:
        raise RuntimeError(
            f"no preferred onnxruntime provider is available (have {sorted(available)})"
        )
    return selected
