"""Tests for Zenodo fetch helpers (no network download)."""

from pathlib import Path

import numpy as np
import pytest
from tools.inference.fetch import (
    file_md5,
    load_dataset_manifest,
    main,
    preprocess_planes,
    preprocess_tree,
    select_pact_bands,
    to_model_domain,
    verify_file,
)


def test_load_dataset_manifest_pins_zenodo_4250706() -> None:
    """The committed manifest names record 4250706 and three files."""
    manifest = load_dataset_manifest()
    assert manifest.record_id == 4250706
    assert "10.5281/zenodo.4250706" in manifest.doi
    assert {item.key for item in manifest.files} == {
        "README.md",
        "segmentation_labels.tar.gz",
        "images.tar.gz",
    }
    assert manifest.pact_band_indices == (1, 2, 3, 7)


def test_verify_file_md5(tmp_path: Path) -> None:
    """verify_file accepts a matching digest and rejects a mismatch."""
    path = tmp_path / "blob.bin"
    path.write_bytes(b"pact")
    digest = file_md5(path)
    assert verify_file(path, digest, expected_size=4)
    assert not verify_file(path, "0" * 32, expected_size=4)
    assert not verify_file(tmp_path / "missing.bin", digest, expected_size=4)


def test_select_pact_bands_takes_b2_b3_b4_b8() -> None:
    """B2/B3/B4/B8 are indices 1, 2, 3, 7 of a 13-band stack."""
    planes = np.zeros((13, 4, 4), dtype=np.float32)
    planes[1] = 0.1
    planes[2] = 0.2
    planes[3] = 0.3
    planes[7] = 0.8
    out = select_pact_bands(planes)
    assert out.shape == (4, 4, 4)
    assert float(out[0].mean()) == pytest.approx(0.1)
    assert float(out[3].mean()) == pytest.approx(0.8)


def test_to_model_domain_resizes_and_scales() -> None:
    """13-band DN 10000 at B2 becomes 1.0 on a 8x8 BLUE plane."""
    planes = np.zeros((13, 4, 4), dtype=np.float32)
    planes[1] = 10000.0
    out = to_model_domain(planes, height=8, width=8, dn_scale=10000.0)
    assert out.shape == (4, 8, 8)
    assert float(out[0].mean()) == pytest.approx(1.0)
    assert float(out[1].mean()) == pytest.approx(0.0)


def test_preprocess_tree_writes_images_npy(tmp_path: Path) -> None:
    """preprocess_tree packs .npy stacks into images.npy."""
    src = tmp_path / "raw"
    src.mkdir()
    stack = np.zeros((13, 4, 4), dtype=np.float32)
    stack[1:4] = 10000.0
    stack[7] = 10000.0
    np.save(src / "a.npy", stack)
    dest = tmp_path / "processed"
    count = preprocess_tree(src, dest, height=8, width=8, indices=(1, 2, 3, 7), dn_scale=10000.0)
    assert count == 1
    images = np.load(dest / "images.npy")
    assert images.shape == (1, 4, 8, 8)
    assert float(images.max()) == pytest.approx(1.0)


def test_cli_without_download_prints_citation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Default CLI prints the citation and does not download."""
    raw = tmp_path / "raw"
    code = main(["--raw-dir", str(raw)])
    assert code == 0
    captured = capsys.readouterr()
    assert "10.5281/zenodo.4250706" in captured.out
    assert "README.md: missing" in captured.out
    assert not any(raw.glob("*")) or list(raw.iterdir()) == []


def test_preprocess_planes_passthrough_four_band() -> None:
    """A 4-band [0, 1] stack resizes without index gather."""
    planes = np.ones((4, 4, 4), dtype=np.float32)
    out = preprocess_planes(planes, height=8, width=8, dn_scale=1.0)
    assert out.shape == (4, 8, 8)
    assert float(out.min()) == pytest.approx(1.0)
