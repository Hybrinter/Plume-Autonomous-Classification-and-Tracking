"""CLI surface tests for the single-dataset ml_models commands."""

import re
from collections.abc import Callable
from pathlib import Path

import pytest
from tools.ml_models.cli import app
from typer.testing import CliRunner

# Rich help styles each hyphen on its own, so a color terminal splits ``--dataset``.
_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    """Return help text with terminal color codes removed."""
    return _ANSI_ESCAPE.sub("", text)


def test_train_help_has_no_dataset_weight_option() -> None:
    """The singular workflow exposes ``--dataset`` but no weight option."""
    result = CliRunner().invoke(app, ["train", "--help"])
    assert result.exit_code == 0
    output = _plain(result.output)
    assert "--dataset" in output
    assert "--dataset-weight" not in output


def test_frame_eval_command_is_removed() -> None:
    """No ``frame-eval`` command remains in help or dispatch."""
    runner = CliRunner()
    top = runner.invoke(app, ["--help"])
    assert top.exit_code == 0
    assert "frame-eval" not in _plain(top.output)
    assert runner.invoke(app, ["frame-eval"]).exit_code != 0


def test_accept_rejects_repeated_dataset_before_any_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second ``--dataset`` root fails before sidecar or artifact work."""
    import tools.ml_models.export.accept as accept
    import tools.ml_models.export.manifest as export_manifest

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("reached")

    monkeypatch.setattr(export_manifest, "load_manifest", _boom)
    monkeypatch.setattr(accept, "accept_artifact", _boom)
    missing = str(tmp_path / "missing")
    result = CliRunner().invoke(
        app,
        [
            "accept",
            "--artifact",
            missing,
            "--manifest",
            missing,
            "--dataset",
            missing,
            "--dataset",
            missing,
        ],
    )
    assert result.exit_code != 0
    assert not isinstance(result.exception, AssertionError)


def test_dataset_analyze_help_and_unavailable(tmp_path: Path) -> None:
    """``dataset analyze`` advertises its options and fails closed."""
    runner = CliRunner()
    help_result = runner.invoke(app, ["dataset", "analyze", "--help"])
    assert help_result.exit_code == 0
    output = _plain(help_result.output)
    assert "--dataset" in output
    assert "--out" in output
    out = tmp_path / "analysis"
    result = runner.invoke(
        app,
        ["dataset", "analyze", "--dataset", str(tmp_path / "ds"), "--out", str(out)],
    )
    assert result.exit_code != 0
    assert not out.exists()


def test_dataset_analyze_cli_refuses_valid_dataset(
    tmp_path: Path, build_synthetic_dataset: Callable[..., Path]
) -> None:
    """``dataset analyze`` stays unavailable for a finished dataset until rendering lands."""
    dataset = build_synthetic_dataset(tmp_path / "ds")
    out = tmp_path / "analysis"
    result = CliRunner().invoke(
        app,
        ["dataset", "analyze", "--dataset", str(dataset), "--out", str(out)],
    )
    assert result.exit_code != 0
    assert "unavailable" in result.output
    assert not out.exists()


def test_analyze_help_and_unavailable(tmp_path: Path) -> None:
    """``analyze`` advertises run, checkpoint, and final-test options."""
    runner = CliRunner()
    help_result = runner.invoke(app, ["analyze", "--help"])
    assert help_result.exit_code == 0
    output = _plain(help_result.output)
    for option in ("--run", "--out", "--checkpoint", "--final-test"):
        assert option in output
    out = tmp_path / "model-analysis"
    result = runner.invoke(
        app,
        [
            "analyze",
            "--run",
            str(tmp_path / "run"),
            "--out",
            str(out),
            "--checkpoint",
            "last",
            "--final-test",
        ],
    )
    assert result.exit_code != 0
    assert not out.exists()


def test_render_help_and_unavailable(tmp_path: Path) -> None:
    """``render`` advertises evidence and output options and fails closed."""
    runner = CliRunner()
    help_result = runner.invoke(app, ["render", "--help"])
    assert help_result.exit_code == 0
    output = _plain(help_result.output)
    for option in ("--evidence", "--out"):
        assert option in output
    out = tmp_path / "figures"
    result = runner.invoke(
        app,
        ["render", "--evidence", str(tmp_path / "evidence"), "--out", str(out)],
    )
    assert result.exit_code != 0
    assert not out.exists()


def test_convert_rejects_bad_dataset_counts_before_any_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """INT8 needs exactly one dataset; FP16 takes at most one."""
    import tools.ml_models.export.precision as precision

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("reached")

    monkeypatch.setattr(precision, "quantize_int8", _boom)
    monkeypatch.setattr(precision, "convert_fp16", _boom)
    missing = str(tmp_path / "missing")
    base = [
        "--source",
        missing,
        "--out",
        str(tmp_path / "out.onnx"),
    ]
    for argv in (
        ["convert", "--precision", "int8", *base],
        ["convert", "--precision", "int8", *base, "--dataset", missing, "--dataset", missing],
        ["convert", "--precision", "fp16", *base, "--dataset", missing, "--dataset", missing],
    ):
        result = CliRunner().invoke(app, argv)
        assert result.exit_code != 0, argv
        assert not isinstance(result.exception, AssertionError), argv
        assert not (tmp_path / "out.onnx").exists()
