"""Tests for the onnxruntime provider preference list."""

import importlib.util

import pytest
from tools.inference.ort_providers import PREFERRED_ORT_PROVIDERS, resolve_ort_providers

_HAS_ORT = importlib.util.find_spec("onnxruntime") is not None


def test_preferred_order_is_trt_cuda_cpu() -> None:
    """Preference is TensorRT, then CUDA, then CPU."""
    assert PREFERRED_ORT_PROVIDERS == (
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    )


@pytest.mark.skipif(not _HAS_ORT, reason="onnxruntime extra not installed")
def test_resolve_ort_providers_cpu_only_in_ci() -> None:
    """CI CPU onnxruntime returns CPUExecutionProvider as the only preferred hit."""
    import onnxruntime

    available = set(onnxruntime.get_available_providers())
    selected = resolve_ort_providers()
    assert selected
    assert all(name in available for name in selected)
    assert selected == [name for name in PREFERRED_ORT_PROVIDERS if name in available]
    if available <= {"CPUExecutionProvider"} or available == {"CPUExecutionProvider"}:
        assert selected == ["CPUExecutionProvider"]
