"""Tests for Zenodo fetch helpers (no network download)."""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from tools.inference.data import load_processed_pack
from tools.inference.fetch import (
    extract_tarball,
    extract_zenodo_archives,
    file_md5,
    index_image_archive,
    load_dataset_manifest,
    load_mask_plane,
    main,
    pair_sample_stems,
    preprocess_pack,
    preprocess_planes,
    preprocess_tree,
    preprocess_zenodo_archives,
    read_annotation_archive,
    select_pact_bands,
    to_model_domain,
    verify_file,
)
from tools.inference.split import compute_dataset_hash, load_dataset_meta

_COLON_STEM = "10003_2019-01-21T10:56:41.330Z_0"
_SQUARE_POINTS = [[25.0, 25.0], [75.0, 25.0], [75.0, 75.0], [25.0, 75.0]]


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


def _annotation_export(
    points: list[list[float]] | None = None,
    *,
    completions: list[object] | None = None,
    polygons: list[list[list[float]]] | None = None,
) -> bytes:
    """Return UTF-8 JSON bytes for one label-studio export."""
    if completions is not None:
        payload: dict[str, object] = {"completions": completions, "data": {}, "id": 1}
    elif polygons is not None:
        payload = {
            "completions": [
                {
                    "result": [
                        {
                            "type": "polygonlabels",
                            "value": {
                                "points": item,
                                "polygonlabels": ["smoke"],
                            },
                            "original_width": 120,
                            "original_height": 120,
                        }
                        for item in polygons
                    ]
                }
            ],
            "data": {},
            "id": 1,
        }
    else:
        payload = {
            "completions": [
                {
                    "result": [
                        {
                            "type": "polygonlabels",
                            "value": {
                                "points": points or _SQUARE_POINTS,
                                "polygonlabels": ["smoke"],
                            },
                            "original_width": 120,
                            "original_height": 120,
                        }
                    ]
                }
            ],
            "data": {},
            "id": 1,
        }
    return json.dumps(payload).encode("utf-8")


def _add_tar_bytes(handle: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    """Add one in-memory member to a tarball without touching the filesystem."""
    info = tarfile.TarInfo(name=arcname)
    info.size = len(payload)
    handle.addfile(info, io.BytesIO(payload))


def _make_geotiff_bytes(height: int = 8, width: int = 8, value: int = 10000) -> bytes:
    """Return 13-band uint16 GeoTIFF bytes for a small tile."""
    from rasterio.io import MemoryFile

    data = np.full((13, height, width), value, dtype=np.uint16)
    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=height,
            width=width,
            count=13,
            dtype="uint16",
        ) as dst:
            dst.write(data)
        return cast(bytes, memfile.read())


def _build_zenodo_fixture_tarballs(tmp_path: Path) -> tuple[Path, Path]:
    """Write miniature images and label tarballs that mirror the Zenodo layout."""
    labels_tar = tmp_path / "segmentation_labels.tar.gz"
    images_tar = tmp_path / "images.tar.gz"
    two_polygons = [
        _SQUARE_POINTS,
        [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]],
    ]
    with tarfile.open(labels_tar, "w:gz") as handle:
        _add_tar_bytes(handle, "alpha_features.json", _annotation_export(_SQUARE_POINTS))
        _add_tar_bytes(
            handle,
            f"{_COLON_STEM}_features.json",
            _annotation_export(polygons=two_polygons),
        )
        _add_tar_bytes(handle, "beta_features.json", _annotation_export(completions=[]))
    with tarfile.open(images_tar, "w:gz") as handle:
        for arcname in (
            "positive/alpha.tif",
            f"positive/{_COLON_STEM}.tif",
            "negative/beta.tif",
            "positive/orphan.tif",
        ):
            _add_tar_bytes(handle, arcname, _make_geotiff_bytes())
    return images_tar, labels_tar


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
    """--preprocess streams tarballs into classifier and segmentor packs."""
    raw = tmp_path / "raw"
    raw.mkdir()
    _build_zenodo_fixture_tarballs(raw)
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
    assert "classifier pack: 4 sample(s)" in captured.out
    assert "segmentor pack: 3 sample(s)" in captured.out
    cls_pack = load_processed_pack(dest / "classifier")
    assert cls_pack.images.shape[0] == 4


def test_read_annotation_archive_keys_by_image_stem(tmp_path: Path) -> None:
    """read_annotation_archive maps image stems and keeps zero-polygon entries."""
    images_tar, labels_tar = _build_zenodo_fixture_tarballs(tmp_path)
    del images_tar
    polygons = read_annotation_archive(labels_tar)
    assert set(polygons) == {"alpha", _COLON_STEM, "beta"}
    assert len(polygons["alpha"]) == 1
    assert len(polygons[_COLON_STEM]) == 2
    assert polygons["beta"] == ()


def test_read_annotation_archive_handles_colon_member_names(tmp_path: Path) -> None:
    """Colon-containing tar member names are readable without extracting to disk."""
    labels_tar = tmp_path / "segmentation_labels.tar.gz"
    with tarfile.open(labels_tar, "w:gz") as handle:
        _add_tar_bytes(
            handle,
            f"{_COLON_STEM}_features.json",
            _annotation_export(_SQUARE_POINTS),
        )
    polygons = read_annotation_archive(labels_tar)
    assert _COLON_STEM in polygons
    assert len(polygons[_COLON_STEM]) == 1


def test_index_image_archive_sorts_and_classifies(tmp_path: Path) -> None:
    """index_image_archive returns sorted stems and positive/annotated subsets."""
    images_tar, labels_tar = _build_zenodo_fixture_tarballs(tmp_path)
    del labels_tar
    polygons = read_annotation_archive(tmp_path / "segmentation_labels.tar.gz")
    index = index_image_archive(images_tar, frozenset(polygons))
    assert index.stems == tuple(sorted(index.stems))
    assert index.positive_stems == frozenset({"alpha", _COLON_STEM, "orphan"})
    assert index.annotated_stems == frozenset({"alpha", _COLON_STEM, "beta"})
    assert "orphan" not in index.annotated_stems


def test_preprocess_zenodo_archives_writes_two_packs(tmp_path: Path) -> None:
    """Streaming preprocess writes classifier and segmentor packs with distinct N."""
    images_tar, labels_tar = _build_zenodo_fixture_tarballs(tmp_path)
    cls_dir = tmp_path / "classifier"
    seg_dir = tmp_path / "segmentor"
    n_cls, n_seg = preprocess_zenodo_archives(
        images_tar,
        labels_tar,
        cls_dir,
        seg_dir,
        height=8,
        width=8,
        indices=(1, 2, 3, 7),
        dn_scale=10000.0,
        source_doi="10.5281/zenodo.4250706",
    )
    assert n_cls == 4
    assert n_seg == 3
    assert n_cls != n_seg
    cls_pack = load_processed_pack(cls_dir)
    seg_pack = load_processed_pack(seg_dir)
    assert cls_pack.images.shape == (4, 4, 8, 8)
    assert seg_pack.images.shape == (3, 4, 8, 8)
    assert float(cls_pack.images.max()) <= 1.0
    assert float(cls_pack.images.min()) >= 0.0


def test_preprocess_zenodo_archives_labels_and_masks(tmp_path: Path) -> None:
    """Positive stems label 1.0; annotated masks follow polygon presence."""
    images_tar, labels_tar = _build_zenodo_fixture_tarballs(tmp_path)
    cls_dir = tmp_path / "classifier"
    preprocess_zenodo_archives(
        images_tar,
        labels_tar,
        cls_dir,
        None,
        height=8,
        width=8,
        indices=(1, 2, 3, 7),
        dn_scale=10000.0,
    )
    stems = json.loads((cls_dir / "stems.json").read_text(encoding="utf-8"))
    labels = np.load(cls_dir / "labels.npy")
    masks = np.load(cls_dir / "masks.npy")
    stem_to_label = {stem: float(labels[idx, 0]) for idx, stem in enumerate(stems)}
    stem_to_mask_max = {stem: float(masks[idx].max()) for idx, stem in enumerate(stems)}
    assert stem_to_label["alpha"] == 1.0
    assert stem_to_label[_COLON_STEM] == 1.0
    assert stem_to_label["beta"] == 0.0
    assert stem_to_label["orphan"] == 1.0
    assert stem_to_mask_max["alpha"] > 0.0
    assert stem_to_mask_max[_COLON_STEM] > 0.0
    assert stem_to_mask_max["beta"] == 0.0
    assert stem_to_mask_max["orphan"] == 0.0


def test_preprocess_zenodo_archives_limit_and_skip(tmp_path: Path) -> None:
    """limit caps the classifier pack; None output dirs skip a pack."""
    images_tar, labels_tar = _build_zenodo_fixture_tarballs(tmp_path)
    limited_dir = tmp_path / "limited"
    n_cls, n_seg = preprocess_zenodo_archives(
        images_tar,
        labels_tar,
        limited_dir,
        None,
        height=8,
        width=8,
        indices=(1, 2, 3, 7),
        dn_scale=10000.0,
        limit=3,
    )
    assert n_cls == 3
    assert n_seg == 0
    stems = json.loads((limited_dir / "stems.json").read_text(encoding="utf-8"))
    assert len(stems) == 3
    only_seg_dir = tmp_path / "segmentor_only"
    n_cls_only, n_seg_only = preprocess_zenodo_archives(
        images_tar,
        labels_tar,
        None,
        only_seg_dir,
        height=8,
        width=8,
        indices=(1, 2, 3, 7),
        dn_scale=10000.0,
    )
    assert n_cls_only == 0
    assert n_seg_only == 3
    assert not (limited_dir / "segmentor").exists()


def test_compute_dataset_hash_is_stable_for_identical_packs(tmp_path: Path) -> None:
    """Identical packs hash to the same digest with chunked file reads."""
    images_tar, labels_tar = _build_zenodo_fixture_tarballs(tmp_path)
    first_dir = tmp_path / "pack_a"
    second_dir = tmp_path / "pack_b"
    for dest in (first_dir, second_dir):
        preprocess_zenodo_archives(
            images_tar,
            labels_tar,
            dest,
            None,
            height=8,
            width=8,
            indices=(1, 2, 3, 7),
            dn_scale=10000.0,
            source_doi="10.5281/zenodo.4250706",
        )
    first_hash = compute_dataset_hash(first_dir)
    second_hash = compute_dataset_hash(second_dir)
    assert first_hash == second_hash
    meta = load_dataset_meta(first_dir / "dataset.json")
    assert meta.dataset_hash == first_hash
