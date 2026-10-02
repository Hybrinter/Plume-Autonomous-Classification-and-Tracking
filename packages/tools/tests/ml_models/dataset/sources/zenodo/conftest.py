"""Zenodo tar fixtures built from stubbed GeoTIFF stacks."""

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from tools.ml_models.dataset.sources.zenodo import archive
from tools.ml_models.dataset.sources.zenodo.bands import ZENODO_BAND_IDS

IMAGE_HW = (119, 121)
STEM_POSITIVE = "10003_2020-01-01T00-00-00.000Z_0"
STEM_NEGATIVE = "10004_2020-01-02T00-00-00.000Z_0"
STEM_BARE = "10005_2020-01-03T00-00-00.000Z_0"
WEIGHT_TABLE_ID = "ap3200t-test"


def patterned_stack() -> np.ndarray:
    """Return a (13, 119, 121) stack with an x ramp, a y ramp, then constants."""
    height, width = IMAGE_HW
    stack = np.zeros((len(ZENODO_BAND_IDS), height, width), dtype=np.float32)
    stack[0] = np.arange(width, dtype=np.float32)[None, :] / width * 10000.0
    stack[1] = np.arange(height, dtype=np.float32)[:, None] / height * 10000.0
    for band in range(2, len(ZENODO_BAND_IDS)):
        stack[band] = np.float32(band * 500.0)
    return stack


def _annotation(points: list[list[float]]) -> bytes:
    payload = {
        "completions": [
            {
                "result": [
                    {
                        "type": "polygonlabels",
                        "value": {
                            "polygonlabels": ["smoke"],
                            "points": points,
                        },
                    }
                ]
            }
        ]
    }
    return json.dumps(payload).encode("utf-8")


def _write_tar(path: Path, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as bundle:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            bundle.addfile(info, io.BytesIO(payload))


def write_archives(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Write the image tar, label tar, and weight table; return their paths."""
    images = tmp_path / "images.tar"
    labels = tmp_path / "labels.tar"
    weights = tmp_path / "weights.toml"
    _write_tar(
        images,
        {
            f"positive/{STEM_POSITIVE}.tif": b"stack0",
            f"negative/{STEM_NEGATIVE}.tif": b"stack1",
            f"positive/{STEM_BARE}.tif": b"stack2",
        },
    )
    _write_tar(
        labels,
        {
            f"{STEM_POSITIVE}_features.json": _annotation(
                [[50.0, 0.0], [100.0, 0.0], [100.0, 100.0], [50.0, 100.0]]
            ),
            f"{STEM_NEGATIVE}_features.json": _annotation(
                [[0.0, 50.0], [100.0, 50.0], [100.0, 100.0], [0.0, 100.0]]
            ),
        },
    )
    weights.write_text(
        f'id = "{WEIGHT_TABLE_ID}"\n[blue]\nB2 = 1.0\n[green]\nB3 = 1.0\n[red]\nB4 = 1.0\n',
        encoding="utf-8",
    )
    return images, labels, weights


@pytest.fixture
def stub_stacks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Stub the GeoTIFF reader and count how many members it decodes."""
    calls: list[str] = []

    def fake(payload: bytes) -> tuple[np.ndarray, tuple[str, ...]]:
        calls.append(payload.decode("utf-8"))
        return patterned_stack(), ZENODO_BAND_IDS

    monkeypatch.setattr(archive, "_stack_from_geotiff", fake)
    return calls


@pytest.fixture
def stems() -> tuple[str, str, str]:
    """Yield the three fixture image stems."""
    return (STEM_POSITIVE, STEM_NEGATIVE, STEM_BARE)


@pytest.fixture
def archives(tmp_path: Path, stub_stacks: list[str]) -> tuple[Path, Path, Path]:
    """Three locations: annotated positive, annotated negative, unannotated."""
    return write_archives(tmp_path)
