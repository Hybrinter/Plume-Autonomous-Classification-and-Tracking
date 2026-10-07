"""Expose sibling test modules to plain imports under --import-mode=importlib."""

import importlib.util
import sys
from pathlib import Path

for _name in ("test_model_measurement", "test_segmentation_figures"):
    _path = Path(__file__).resolve().parent / f"{_name}.py"
    _spec = importlib.util.spec_from_file_location(_name, _path)
    if _spec is not None and _spec.loader is not None:
        _module = importlib.util.module_from_spec(_spec)
        sys.modules.setdefault(_name, _module)
        _spec.loader.exec_module(_module)
