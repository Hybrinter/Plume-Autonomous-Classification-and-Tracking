"""Tests for Zenodo fetch helpers (no network download)."""

import tarfile
from pathlib import Path

import numpy as np
import pytest
from tools.inference.data import load_processed_pack
from tools.inference.fetch import (
    extract_tarball,
    extract_zenodo_archives,
    file_md5,
    load_dataset_manifest,
    load_mask_plane,
    main,
    pair_sample_stems,
    preprocess_pack,
    preprocess_planes,
    preprocess_tree,
    select_pact_bands,
    to_model_domain,
    verify_file,
)
from tools.inference.split import load_dataset_meta


def _write_stack(path: Path, value: float) -> None:
    """Write a 13-band (C, 4, 4) stack with PACT bands set to value."""
    stack = np.zeros((13, 4, 4), dtype=np.float32)
    stack[1:4] = value
    stack[7] = value
    np.save(path, stack)


def _write_mask(path: Path, positive: bool) -> None:
    """Write a (4, 4) binary mask npy."""
    mask = np.zeros((4, 4), dtype=np.float32)
    if positive:
        mask[1:3, 1:3] = 1.0
    np.save(path, mask)


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


def test_extract_tarball_and_pair_stems(tmp_path: Path) -> None:
    """extract_tarball unpacks archives that pair_sample_stems can match."""
    images_src = tmp_path / "img_src"
    labels_src = tmp_path / "lbl_src"
    images_src.mkdir()
    labels_src.mkdir()
    _write_stack(images_src / "s0.npy", 10000.0)
    _write_mask(labels_src / "s0.npy", True)
    images_tar = tmp_path / "images.tar.gz"
    labels_tar = tmp_path / "labels.tar.gz"
    with tarfile.open(images_tar, "w:gz") as handle:
        handle.add(images_src / "s0.npy", arcname="s0.npy")
    with tarfile.open(labels_tar, "w:gz") as handle:
        handle.add(labels_src / "s0.npy", arcname="s0.npy")
    image_root = extract_tarball(images_tar, tmp_path / "extracted_images")
    label_root = extract_tarball(labels_tar, tmp_path / "extracted_labels")
    pairs = pair_sample_stems(image_root, label_root)
    assert len(pairs) == 1
    assert pairs[0][0].stem == "s0"


def test_load_mask_plane_resizes_and_binarizes(tmp_path: Path) -> None:
    """load_mask_plane nearest-neighbor resizes and thresholds to {0, 1}."""
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[0:2, 0:2] = 1.0
    path = tmp_path / "mask.npy"
    np.save(path, mask)
    out = load_mask_plane(path, height=8, width=8)
    assert out.shape == (1, 8, 8)
    assert set(np.unique(out).tolist()) <= {0.0, 1.0}
    assert float(out[0, 0, 0]) == 1.0
    assert float(out[0, 7, 7]) == 0.0


def test_preprocess_pack_writes_labeled_splits(tmp_path: Path) -> None:
    """preprocess_pack writes images, masks, labels, splits, and dataset.json."""
    images_dir = tmp_path / "images"
    labels_dir = tmp_path / "labels"
    images_dir.mkdir()
    labels_dir.mkdir()
    for i in range(4):
        _write_stack(images_dir / f"s{i}.npy", 10000.0)
        _write_mask(labels_dir / f"s{i}.npy", positive=(i % 2 == 0))
    dest = tmp_path / "processed"
    count = preprocess_pack(
        images_dir,
        labels_dir,
        dest,
        height=8,
        width=8,
        indices=(1, 2, 3, 7),
        dn_scale=10000.0,
        source_doi="10.5281/zenodo.4250706",
    )
    assert count == 4
    pack = load_processed_pack(dest)
    assert pack.images.shape == (4, 4, 8, 8)
    assert pack.masks.shape == (4, 1, 8, 8)
    assert pack.labels.shape == (4, 1)
    meta = load_dataset_meta(dest / "dataset.json")
    assert meta.n == 4
    assert meta.source_doi == "10.5281/zenodo.4250706"
    assert meta.dataset_hash == pack.meta.dataset_hash
    assert len(pack.splits.train) + len(pack.splits.val) + len(pack.splits.test) == 4


def test_extract_zenodo_archives_uses_raw_when_missing(tmp_path: Path) -> None:
    """Without tarballs, extract_zenodo_archives returns the raw directory."""
    image_root, label_root = extract_zenodo_archives(tmp_path)
    assert image_root == tmp_path
    assert label_root == tmp_path


def test_cli_preprocess_from_tarballs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--preprocess unpacks tarballs and writes a labeled pack."""
    raw = tmp_path / "raw"
    images_src = raw / "img_src"
    labels_src = raw / "lbl_src"
    images_src.mkdir(parents=True)
    labels_src.mkdir()
    for i in range(3):
        _write_stack(images_src / f"s{i}.npy", 10000.0)
        _write_mask(labels_src / f"s{i}.npy", positive=(i == 0))
    with tarfile.open(raw / "images.tar.gz", "w:gz") as handle:
        for path in sorted(images_src.glob("*.npy")):
            handle.add(path, arcname=path.name)
    with tarfile.open(raw / "segmentation_labels.tar.gz", "w:gz") as handle:
        for path in sorted(labels_src.glob("*.npy")):
            handle.add(path, arcname=path.name)
    dest = tmp_path / "processed"
    code = main(
        [
            "--raw-dir",
            str(raw),
            "--processed-dir",
            str(dest),
            "--preprocess",
            "--height",
            "8",
            "--width",
            "8",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "preprocessed 3 sample(s)" in captured.out
    pack = load_processed_pack(dest)
    assert pack.images.shape[0] == 3
