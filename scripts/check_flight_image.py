#!/usr/bin/env python3
"""Lean-install CI check for the isolated pact-flight package.

Verifies that pact-flight installs without pulling pact-tools, pact-sim, or pact-gse,
declares the expected role extras, and does not expose heavy ML frameworks. Stdlib plus
subprocess calls to ``uv`` so the probe runs in a fresh venv, not the workspace dev env.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
import tomllib
import zipfile
from pathlib import Path

DENIED_DISTRIBUTIONS: frozenset[str] = frozenset({"pact-tools", "pact-sim", "pact-gse"})
DENIED_MODULES: frozenset[str] = frozenset(
    {"torch", "torchvision", "tensorflow", "jax", "jaxlib", "flax", "keras", "mlx"}
)
FLIGHT_EXTRAS: frozenset[str] = frozenset({"inference", "camera", "gimbal"})
TOOLS_EXTRAS: frozenset[str] = frozenset({"train", "export"})
FORBIDDEN_WHEEL_PREFIXES: frozenset[str] = frozenset({"tools/", "sim/", "gse/"})


def load_optional_dependency_keys(pyproject: Path) -> frozenset[str]:
    """Return optional-dependency extra names declared in a pyproject.toml.

    Args:
        pyproject: path to the TOML file to parse.

    Returns:
        Frozenset of extra names under ``[project.optional-dependencies]``.
    """
    with pyproject.open("rb") as handle:
        data = tomllib.load(handle)
    optional = data.get("project", {}).get("optional-dependencies", {})
    return frozenset(optional.keys())


def missing_extras(declared: frozenset[str], required: frozenset[str]) -> frozenset[str]:
    """Return required extras that are not present in ``declared``.

    Args:
        declared: extras found in a pyproject.
        required: extras that must be declared.

    Returns:
        Required extras absent from ``declared``.
    """
    return required - declared


def check_extras_declared(repo: Path) -> list[str]:
    """Verify flight and tools pyprojects declare the expected role extras.

    Args:
        repo: repository root containing ``packages/flight`` and ``packages/tools``.

    Returns:
        Human-readable findings; empty when every required extra is declared.
    """
    findings: list[str] = []
    flight_pyproject = repo / "packages" / "flight" / "pyproject.toml"
    tools_pyproject = repo / "packages" / "tools" / "pyproject.toml"

    flight_missing = missing_extras(
        load_optional_dependency_keys(flight_pyproject),
        FLIGHT_EXTRAS,
    )
    if flight_missing:
        missing = ", ".join(sorted(flight_missing))
        findings.append(
            f"packages/flight/pyproject.toml missing optional-dependencies extras: {missing}"
        )

    tools_missing = missing_extras(
        load_optional_dependency_keys(tools_pyproject),
        TOOLS_EXTRAS,
    )
    if tools_missing:
        missing = ", ".join(sorted(tools_missing))
        findings.append(
            f"packages/tools/pyproject.toml missing optional-dependencies extras: {missing}"
        )

    return findings


def _run_uv(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    """Run a ``uv`` subprocess and return the completed process.

    Args:
        args: arguments to pass after the ``uv`` executable name.
        cwd: optional working directory for the subprocess.

    Returns:
        The completed subprocess result (exit code is not checked).
    """
    return subprocess.run(
        ["uv", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def _fail(message: str, *, detail: str = "") -> int:
    """Print a failure message and return exit code 1.

    Args:
        message: primary failure line printed to stderr.
        detail: optional extra context printed after the primary line.

    Returns:
        Always 1 for use as a process exit code.
    """
    print(message, file=sys.stderr)
    if detail:
        print(detail, file=sys.stderr)
    return 1


def _probe_script() -> str:
    """Return Python source run inside the isolated flight venv."""
    denied_dists = ", ".join(repr(name) for name in sorted(DENIED_DISTRIBUTIONS))
    denied_mods = ", ".join(repr(name) for name in sorted(DENIED_MODULES))
    return f"""\
import importlib.metadata
import importlib.util
import sys

DENIED_DISTRIBUTIONS = {{{denied_dists}}}
DENIED_MODULES = {{{denied_mods}}}
REQUIRED_MODULES = ("numpy", "scipy", "structlog")

errors: list[str] = []

try:
    import flight  # noqa: F401
except ImportError as exc:
    errors.append(f"import flight failed: {{exc}}")

installed = {{dist.name.lower() for dist in importlib.metadata.distributions()}}

if "pact-flight" not in installed:
    errors.append("pact-flight distribution not found after install")

for denied in DENIED_DISTRIBUTIONS:
    if denied in installed:
        errors.append(f"forbidden distribution installed: {{denied}}")

for module_name in DENIED_MODULES:
    if importlib.util.find_spec(module_name) is not None:
        errors.append(f"forbidden module importable: {{module_name}}")

for module_name in REQUIRED_MODULES:
    try:
        __import__(module_name)
    except ImportError as exc:
        errors.append(f"required module {{module_name}} not importable: {{exc}}")

if errors:
    for line in errors:
        print(line, file=sys.stderr)
    sys.exit(1)
"""


def _run_probe(python: Path) -> subprocess.CompletedProcess[str]:
    """Execute the install probe in the given interpreter.

    Args:
        python: path to the isolated venv Python executable.

    Returns:
        The completed subprocess result from running the probe script.
    """
    return subprocess.run(
        [str(python), "-c", _probe_script()],
        capture_output=True,
        text=True,
        check=False,
    )


def _install_flight(repo: Path, python: Path) -> subprocess.CompletedProcess[str] | None:
    """Install pact-flight (base and role extras) into the probe venv.

    Args:
        repo: repository root.
        python: path to the isolated venv Python executable.

    Returns:
        None on success, or the first failing completed process.
    """
    flight_pkg = repo / "packages" / "flight"
    base = _run_uv(["pip", "install", "--python", str(python), str(flight_pkg)])
    if base.returncode != 0:
        return base
    extras = _run_uv(
        [
            "pip",
            "install",
            "--python",
            str(python),
            f"{flight_pkg}[inference,camera,gimbal]",
        ]
    )
    if extras.returncode != 0:
        return extras
    return None


def check_wheel_contents(wheel_path: Path) -> list[str]:
    """Inspect a built wheel for flight-only package layout.

    Args:
        wheel_path: path to the ``.whl`` file to inspect.

    Returns:
        Human-readable findings; empty when the wheel layout is acceptable.
    """
    findings: list[str] = []
    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()

    if not any(name.startswith("flight/") for name in names):
        findings.append(f"{wheel_path.name}: wheel missing flight/ package paths")

    for prefix in FORBIDDEN_WHEEL_PREFIXES:
        package = prefix.rstrip("/")
        if any(name.startswith(prefix) for name in names):
            findings.append(f"{wheel_path.name}: wheel contains forbidden package {package}/")

    return findings


def _build_and_check_wheel(repo: Path, out_dir: Path) -> list[str]:
    """Build pact-flight and verify the wheel contains only flight code.

    Args:
        repo: repository root where ``uv build`` is executed.
        out_dir: directory that receives the built wheel.

    Returns:
        Human-readable findings from the build or wheel inspection.
    """
    build = _run_uv(["build", "--package", "pact-flight", "--out-dir", str(out_dir)], cwd=repo)
    if build.returncode != 0:
        detail = (build.stdout + build.stderr).strip()
        return [f"uv build --package pact-flight failed: {detail}"]

    wheels = sorted(out_dir.glob("*.whl"))
    if not wheels:
        return [f"uv build produced no wheel in {out_dir}"]

    findings: list[str] = []
    for wheel in wheels:
        findings.extend(check_wheel_contents(wheel))
    return findings


def run_full_check(repo: Path) -> int:
    """Create an isolated venv, install pact-flight, and probe the result.

    Args:
        repo: repository root.

    Returns:
        0 on success, 1 on any failure.
    """
    with tempfile.TemporaryDirectory(prefix="check_flight_image_") as tmp:
        tmp_path = Path(tmp)
        venv_dir = tmp_path / "venv"
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()

        venv = _run_uv(["venv", "--python", "3.14", str(venv_dir)])
        if venv.returncode != 0:
            detail = (venv.stdout + venv.stderr).strip()
            return _fail("check_flight_image: failed to create probe venv", detail=detail)

        python = venv_dir / "bin" / "python"
        install_error = _install_flight(repo, python)
        if install_error is not None:
            detail = (install_error.stdout + install_error.stderr).strip()
            return _fail("check_flight_image: pact-flight install failed", detail=detail)

        probe = _run_probe(python)
        if probe.returncode != 0:
            detail = (probe.stdout + probe.stderr).strip()
            return _fail("check_flight_image: install probe failed", detail=detail)

        wheel_findings = _build_and_check_wheel(repo, wheel_dir)
        if wheel_findings:
            for finding in wheel_findings:
                print(finding, file=sys.stderr)
            return 1

    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run checks, and return a process exit code.

    Args:
        argv: optional argument vector (defaults to ``sys.argv[1:]``).

    Returns:
        0 when all requested checks pass, 1 otherwise.
    """
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Verify pact-flight installs as an isolated lean package.",
    )
    parser.add_argument(
        "--extras-only",
        action="store_true",
        help="Only verify role extras are declared in pyproject.toml files.",
    )
    args = parser.parse_args(argv)

    findings = check_extras_declared(repo_root)
    if findings:
        print("check_flight_image: extras check FAILED:", file=sys.stderr)
        for finding in findings:
            print(f"  - {finding}", file=sys.stderr)
        return 1

    if args.extras_only:
        print("check_flight_image: ok")
        return 0

    if run_full_check(repo_root) != 0:
        return 1

    print("check_flight_image: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
