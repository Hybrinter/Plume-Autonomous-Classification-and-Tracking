"""Tests for the pure helpers in scripts/prepare_dataset.py.

Import via the file path since `scripts/` is not a package.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_prepare_dataset():
    path = REPO_ROOT / "scripts" / "prepare_dataset.py"
    spec = importlib.util.spec_from_file_location("prepare_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pd_module():
    return _load_prepare_dataset()


def test_site_id_from_stem(pd_module):
    assert pd_module.site_id_from_stem("ghana_2018-01-01") == "ghana"
    assert pd_module.site_id_from_stem("no_underscore") == "no"


@pytest.mark.parametrize(
    "url,expected",
    [
        ("/data/upload/1-site_001.png", "site_001.tif"),
        ("/data/upload/1-site_2024:01:01Z.png", "site_2024_01_01Z.tif"),
    ],
)
def test_json_url_to_tif_basename(pd_module, url, expected):
    assert pd_module.json_url_to_tif_basename(url) == expected


def test_link_or_copy_copy_mode(pd_module, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hello")
    dst = tmp_path / "out" / "dst.txt"
    pd_module.link_or_copy(str(src), str(dst), "copy")
    assert dst.read_text() == "hello"


@pytest.mark.skipif(sys.platform == "win32", reason="hardlinks unreliable on Windows")
def test_link_or_copy_hardlink_mode(pd_module, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("hardlinked")
    dst = tmp_path / "out" / "dst.txt"
    pd_module.link_or_copy(str(src), str(dst), "hardlink")
    # hardlink: the two paths should share inode
    assert dst.stat().st_ino == src.stat().st_ino


def test_link_or_copy_invalid_mode(pd_module, tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("x")
    with pytest.raises(ValueError):
        pd_module.link_or_copy(str(src), str(tmp_path / "dst.txt"), "teleport")
