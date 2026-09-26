"""Preferred onnxruntime execution-provider list for accept and bench.

The implementation lives in ``tools.ml_models.export.ort_providers``. This
module re-exports that public surface so existing callers keep one import path.

Contains:
  - PREFERRED_ORT_PROVIDERS: preference order.
  - resolve_ort_providers: available providers in that order.

Satisfies: REQ-AIML-HIGH-004.
"""

from __future__ import annotations

from tools.ml_models.export.ort_providers import PREFERRED_ORT_PROVIDERS, resolve_ort_providers

__all__ = [
    "PREFERRED_ORT_PROVIDERS",
    "resolve_ort_providers",
]
