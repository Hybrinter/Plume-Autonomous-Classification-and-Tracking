"""Tests for the shared schema-2 unit-tile product contract."""

import subprocess
import sys
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from flight.libs.types import Err, FaultCode, Ok
from flight.payload.gimbal.footprint import GsdPair
from flight.payload.preprocess.tile_product import (
    UNIT_TILE_SOURCE_SCHEMA,
    UnitTileCapture,
    UnitTileLayout,
    validate_capture,
    validate_layout,
    validate_unit_tile,
)


def _layout(**overrides: object) -> UnitTileLayout:
    """One valid layout with overridable fields."""
    fields: dict[str, object] = {
        "band_names": ("BLUE", "GREEN", "RED"),
        "tile_hw": (4, 7),
        "grid": (2, 2),
        "gsd_reference_m": 15.87,
    }
    fields.update(overrides)
    return UnitTileLayout(**fields)  # type: ignore[arg-type]


def _capture(**overrides: object) -> UnitTileCapture:
    """One valid capture with overridable fields."""
    fields: dict[str, object] = {
        "tile_id": "t0",
        "frame_id": "frame-a",
        "grid_rc": (1, 1),
        "theta_g_deg": 15.0,
        "gsd": GsdPair(16.5, 17.1),
        "gsd_nominal": False,
    }
    fields.update(overrides)
    return UnitTileCapture(**fields)  # type: ignore[arg-type]


def _image() -> np.ndarray:
    """One valid float32 unit tile for ``_layout()``."""
    return np.full((3, 4, 7), 0.5, dtype=np.float32)


def _assert_malformed(result: object) -> None:
    """Assert a validator returned Err(FRAME_MALFORMED)."""
    assert isinstance(result, Err)
    assert result.error == FaultCode.FRAME_MALFORMED


def test_schema_constant() -> None:
    """The unit-tile source schema identifier is 2."""
    assert UNIT_TILE_SOURCE_SCHEMA == 2


def test_types_are_frozen_and_slotted() -> None:
    """Layout and capture instances cannot be mutated or extended."""
    layout = _layout()
    capture = _capture()
    with pytest.raises(FrozenInstanceError):
        layout.tile_hw = (1, 1)  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        capture.gsd_nominal = True  # type: ignore[misc]
    assert not hasattr(layout, "__dict__")
    assert not hasattr(capture, "__dict__")


def test_valid_layout_is_accepted() -> None:
    """A recorded layout with ordered unique bands validates."""
    assert isinstance(validate_layout(_layout()), Ok)
    reordered = _layout(band_names=("RED", "GREEN", "BLUE"))
    assert isinstance(validate_layout(reordered), Ok)


@pytest.mark.parametrize(
    "layout",
    [
        None,
        "layout",
        _layout(band_names=()),
        _layout(band_names=["BLUE", "GREEN"]),
        _layout(band_names=("BLUE", "GREEN", "")),
        _layout(band_names=("BLUE", "GREEN", 3)),
        _layout(band_names=("BLUE", "BLUE")),
        _layout(tile_hw=[4, 7]),
        _layout(tile_hw=(4,)),
        _layout(tile_hw=(4, 7, 1)),
        _layout(tile_hw=(4, True)),
        _layout(tile_hw=(4, 0)),
        _layout(tile_hw=(4, -7)),
        _layout(tile_hw=(4.0, 7)),
        _layout(grid=[2, 2]),
        _layout(grid=(2, 2, 1)),
        _layout(grid=(2, False)),
        _layout(grid=(0, 2)),
        _layout(gsd_reference_m=True),
        _layout(gsd_reference_m="15.87"),
        _layout(gsd_reference_m=0.0),
        _layout(gsd_reference_m=-1.0),
        _layout(gsd_reference_m=float("nan")),
        _layout(gsd_reference_m=float("inf")),
        _layout(gsd_reference_m=10**400),
    ],
    ids=lambda value: repr(value)[:60],
)
def test_invalid_layout_returns_err(layout: object) -> None:
    """Wrong-typed and out-of-domain layouts return Err, not an exception."""
    _assert_malformed(validate_layout(layout))


def test_valid_capture_is_accepted() -> None:
    """Measured and explicitly nominal captures both validate."""
    assert isinstance(validate_capture(_capture(), _layout()), Ok)
    nominal = _capture(gsd_nominal=True)
    assert isinstance(validate_capture(nominal, _layout()), Ok)
    unbounded_theta = _capture(theta_g_deg=200.0)
    assert isinstance(validate_capture(unbounded_theta, _layout()), Ok)


@pytest.mark.parametrize(
    "capture",
    [
        None,
        "capture",
        _capture(tile_id=""),
        _capture(tile_id="."),
        _capture(tile_id=".."),
        _capture(tile_id="a/b"),
        _capture(tile_id="a\\b"),
        _capture(tile_id="a\x00b"),
        _capture(tile_id=12),
        _capture(frame_id=""),
        _capture(frame_id=4.2),
        _capture(grid_rc=[1, 1]),
        _capture(grid_rc=(1,)),
        _capture(grid_rc=(1, 1, 0)),
        _capture(grid_rc=(True, 1)),
        _capture(grid_rc=(1.0, 1)),
        _capture(grid_rc=(-1, 0)),
        _capture(grid_rc=(0, -1)),
        _capture(grid_rc=(2, 0)),
        _capture(grid_rc=(0, 2)),
        _capture(theta_g_deg=True),
        _capture(theta_g_deg="15"),
        _capture(theta_g_deg=float("nan")),
        _capture(theta_g_deg=float("-inf")),
        _capture(gsd=None),
        _capture(gsd=(16.5, 17.1)),
        _capture(gsd=GsdPair(0.0, 17.1)),
        _capture(gsd=GsdPair(16.5, -1.0)),
        _capture(gsd=GsdPair(float("nan"), 17.1)),
        _capture(gsd=GsdPair(16.5, float("inf"))),
        _capture(gsd_nominal=1),
        _capture(gsd_nominal="yes"),
        _capture(gsd_nominal=None),
    ],
    ids=lambda value: repr(value)[:60],
)
def test_invalid_capture_returns_err(capture: object) -> None:
    """Wrong-typed and out-of-domain captures return Err, not an exception."""
    _assert_malformed(validate_capture(capture, _layout()))


@pytest.mark.parametrize(
    "layout",
    [
        None,
        _layout(grid=(0, 2)),
        _layout(band_names=()),
    ],
    ids=lambda value: repr(value)[:60],
)
def test_invalid_layout_rejects_other_validators(layout: object) -> None:
    """Capture and image validators reject a bad layout before field checks."""
    _assert_malformed(validate_capture(_capture(), layout))
    _assert_malformed(validate_unit_tile(_image(), layout))


def test_valid_unit_tile_is_accepted() -> None:
    """A float32 unit tile matching the layout validates at the boundaries."""
    image = _image()
    image[0, 0, 0] = np.float32(0.0)
    image[0, 0, 1] = np.float32(1.0)
    image[2, 3, 6] = np.float32(0.1234567)
    assert isinstance(validate_unit_tile(image, _layout()), Ok)


@pytest.mark.parametrize(
    "image",
    [
        None,
        [[0.5]],
        np.zeros((3, 4, 7), dtype=np.float64),
        np.zeros((3, 4, 7), dtype=np.uint16),
        np.zeros((4, 7), dtype=np.float32),
        np.zeros((3, 4, 8), dtype=np.float32),
        np.zeros((2, 4, 7), dtype=np.float32),
        np.full((3, 4, 7), np.nan, dtype=np.float32),
        np.full((3, 4, 7), -0.25, dtype=np.float32),
        np.full((3, 4, 7), 1.5, dtype=np.float32),
        np.full((3, 4, 7), np.inf, dtype=np.float32),
    ],
    ids=lambda value: repr(getattr(value, "dtype", value))[:60],
)
def test_invalid_unit_tile_returns_err(image: object) -> None:
    """Wrong dtype, shape, or pixel domain returns Err without casting."""
    _assert_malformed(validate_unit_tile(image, _layout()))


def test_validators_do_not_mutate_inputs() -> None:
    """Validation never writes to the image or the metadata."""
    image = _image()
    before = image.tobytes()
    flags_before = image.flags.writeable
    layout = _layout()
    capture = _capture()
    validate_unit_tile(image, layout)
    validate_capture(capture, layout)
    validate_layout(layout)
    assert image.tobytes() == before
    assert image.flags.writeable == flags_before


def test_module_imports_without_torch_or_tools() -> None:
    """The contract module imports without Torch, tools, or hardware SDKs."""
    code = (
        "import sys\n"
        "import flight.payload.preprocess.tile_product\n"
        "assert 'torch' not in sys.modules\n"
        "assert 'tools' not in sys.modules\n"
        "assert 'onnxruntime' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
