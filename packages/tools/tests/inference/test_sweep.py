"""Sweep space expansion and run tests."""

import json
from pathlib import Path

import pytest
from tools.inference.cli import main
from tools.inference.sweep import load_sweep_space, sweep


def test_load_sweep_space_cartesian(tmp_path: Path) -> None:
    """List axes expand in sorted-name cartesian order and honor max_runs."""
    space = tmp_path / "space.toml"
    space.write_text(
        "\n".join(
            [
                'kind = "segmentor"',
                "epochs = 1",
                "learning_rate = [0.01, 0.001]",
                "seed = [0, 1]",
                "max_runs = 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    trials = load_sweep_space(space)
    assert len(trials) == 2
    assert trials[0].learning_rate == 0.01
    assert trials[0].seed == 0
    assert trials[1].learning_rate == 0.01
    assert trials[1].seed == 1


def test_load_sweep_space_unknown_arch(tmp_path: Path) -> None:
    """An unknown architecture in the space raises ValueError."""
    space = tmp_path / "space.toml"
    space.write_text('kind = "segmentor"\narch = "nope"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown segmentor architecture"):
        load_sweep_space(space)


def test_load_sweep_space_rejects_scalar_run_id(tmp_path: Path) -> None:
    """A scalar run_id in the space raises ValueError."""
    space = tmp_path / "space.toml"
    space.write_text('kind = "segmentor"\nrun_id = "fixed"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="omit scalar run_id"):
        load_sweep_space(space)


def test_sweep_writes_jsonl_and_run_dirs(tmp_path: Path) -> None:
    """sweep trains two trials and writes one JSONL record each."""
    space = tmp_path / "space.toml"
    space.write_text(
        "\n".join(
            [
                'kind = "segmentor"',
                "epochs = 1",
                "batch_size = 2",
                "synthetic_samples = 4",
                "input_height_px = 32",
                "input_width_px = 32",
                f'run_dir = "{(tmp_path / "runs").as_posix()}"',
                "learning_rate = [0.01, 0.001]",
                "seed = 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    jsonl = sweep(space)
    assert jsonl.is_file()
    lines = [line for line in jsonl.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert all("val_mean_iou" in item or "best_val_metric" in item for item in records)
    runs = list((tmp_path / "runs").iterdir())
    assert len([path for path in runs if path.is_dir()]) == 2


def _tiny_space(tmp_path: Path, name: str = "space.toml") -> Path:
    """Write a one-trial segmentor space that trains in a moment."""
    space = tmp_path / name
    space.write_text(
        "\n".join(
            [
                'kind = "segmentor"',
                "epochs = 1",
                "batch_size = 2",
                "synthetic_samples = 4",
                "input_height_px = 32",
                "input_width_px = 32",
                f'run_dir = "{(tmp_path / "runs").as_posix()}"',
                "learning_rate = 0.01",
                "seed = 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return space


def test_sweep_refuses_a_locked_output(tmp_path: Path) -> None:
    """A second sweep aimed at a held output refuses rather than interleaving rows."""
    space = _tiny_space(tmp_path)
    out = tmp_path / "sweep.jsonl"
    lock = out.with_suffix(out.suffix + ".lock")
    lock.write_text("pid=1 started=0\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="another sweep is writing"):
        sweep(space, out=str(out))


def test_sweep_releases_its_lock(tmp_path: Path) -> None:
    """The lock is gone once a sweep returns, so the next one can start."""
    space = _tiny_space(tmp_path)
    out = tmp_path / "sweep.jsonl"

    sweep(space, out=str(out))

    assert not out.with_suffix(out.suffix + ".lock").exists()
    assert sweep(space, out=str(out)).is_file()


def test_cli_arches_and_sweep_unknown(tmp_path: Path) -> None:
    """arches prints known pairs. sweep CLI returns 1 on a bad space."""
    assert main(["arches"]) == 0
    space = tmp_path / "bad.toml"
    space.write_text('kind = "segmentor"\narch = "nope"\n', encoding="utf-8")
    assert main(["sweep", "--space", str(space)]) == 1
