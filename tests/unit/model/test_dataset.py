"""Unit tests for pact.model.dataset.HsgAimlDataset.

Satisfies: §6.2 of PACT_SW_ARCH.md — Model subsystem unit tests.
REQ-AIML-HIGH-001, REQ-AIML-IMAG-001

Note: Most dataset tests require the HSG-AIML dataset to be downloaded from
Zenodo (DOI: 10.5281/zenodo.4250706). Those tests are marked skip until the
dataset is available in CI. Only structural / error-path tests run without data.
"""

from __future__ import annotations

# stdlib
from pathlib import Path

# third-party
import pytest

# module under test
from pact.model.dataset import HsgAimlDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nonexistent_path(tmp_path: Path) -> str:
    """Return a string path that is guaranteed not to exist."""
    return str(tmp_path / "does_not_exist")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_dataset_init_requires_data_dir(tmp_path: Path) -> None:
    """HsgAimlDataset should raise or return len==0 if data_root doesn't exist.

    The dataset must fail gracefully — either raising ValueError/FileNotFoundError
    or returning a dataset with length 0 — when the data root directory does not
    exist. It must not raise an unhandled exception of a different type.
    """
    bad_root = _nonexistent_path(tmp_path)
    try:
        ds = HsgAimlDataset(data_root=bad_root)
        # If constructor succeeds, the dataset must be empty (no files found).
        assert len(ds) == 0, (
            f"Expected empty dataset for non-existent root, got len={len(ds)}"
        )
    except (ValueError, FileNotFoundError, OSError):
        # Raising one of these is also acceptable behaviour.
        pass


@pytest.mark.skip(reason="requires HSG-AIML dataset download (Zenodo 10.5281/zenodo.4250706)")
def test_dataset_len_matches_labelled_count(tmp_path: Path) -> None:
    """HsgAimlDataset.__len__ must return exactly 1,437 (the labelled image count).

    Requires: HSG-AIML dataset downloaded to data/raw/ via scripts/download_dataset.py.
    """
    ds = HsgAimlDataset(data_root="data/raw")
    assert len(ds) == 1437, f"Expected 1437 labelled images, got {len(ds)}"


@pytest.mark.skip(reason="requires HSG-AIML dataset download (Zenodo 10.5281/zenodo.4250706)")
def test_dataset_item_shapes() -> None:
    """HsgAimlDataset[0] must return (tensor, mask) with correct shapes and dtypes.

    tensor: (4, H, W) float32 in [0, 1]
    mask:   (1, H, W) float32 binary

    Requires: HSG-AIML dataset downloaded to data/raw/ via scripts/download_dataset.py.
    """
    import torch  # local import — only needed when dataset is available

    ds = HsgAimlDataset(data_root="data/raw")
    tensor, mask = ds[0]
    assert tensor.shape[0] == 4, f"Expected 4 bands, got {tensor.shape[0]}"
    assert mask.shape[0] == 1, f"Expected 1-channel mask, got {mask.shape[0]}"
    assert tensor.dtype == torch.float32
    assert mask.dtype == torch.float32
    assert float(tensor.min()) >= 0.0
    assert float(tensor.max()) <= 1.0
