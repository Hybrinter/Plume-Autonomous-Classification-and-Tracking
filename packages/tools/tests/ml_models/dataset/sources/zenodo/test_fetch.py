"""Tests for the Zenodo checksum manifest and file verification helpers."""

from pathlib import Path

import pytest
from tools.ml_models.dataset.sources.zenodo.fetch import (
    file_md5,
    load_dataset_manifest,
    main,
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
