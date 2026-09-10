"""Provenance and isolation checks for the vendored Xeryon v1.88 package."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import serial
from flight.libs.config import GimbalConfig
from xeryon_vendor import Xeryon

_VENDOR_ROOT = Path(__file__).parents[2] / "xeryon-vendor"
_SOURCE_SHA256 = "6b231f91a57ea60bde1dcf0910155ba5e09e2a30b84bc218a87d2ce703b8ffdb"
_ZIP_SHA256 = "7f3c8a373e66b2fd2cb915035bc2c29f860b59ff76ec4ea1531824b3ca7b8c57"


def test_vendor_source_and_archive_provenance_are_pinned() -> None:
    source = _VENDOR_ROOT / "src" / "xeryon_vendor" / "Xeryon.py"
    notice = (_VENDOR_ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == _SOURCE_SHA256
    assert "version v1.88" in notice
    assert _ZIP_SHA256 in notice
    assert "https://xeryon.com/software/xeryon-python-library/" in notice


def test_selected_stage_and_pyserial_dependency() -> None:
    stage = Xeryon.Stage.XRTU_40_109
    assert stage.name == "XRTU_40_109"
    assert GimbalConfig().encoder_counts_per_rev == 86400
    assert serial.VERSION == "3.5"


def test_simulation_import_does_not_load_vendor_module_or_open_serial() -> None:
    probe = (
        "import sys; import flight.hal.drivers_sim.gimbal; "
        "assert 'xeryon_vendor.Xeryon' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)
