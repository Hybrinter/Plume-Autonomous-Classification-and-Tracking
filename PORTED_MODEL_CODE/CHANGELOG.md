# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.2.0] — 2026-04-17

### Added
- PyTorch Lightning training/eval (`LightningModule`, `LightningDataModule`).
- pydantic v2 + YAML experiment configs under `configs/`.
- Unified CLI: `python -m smoke_detection.cli.train --config <yaml>` and
  `smoke_detection.cli.eval`.
- `ruff` (lint + format), `pytest` scaffolding, `pre-commit`, GitHub Actions
  CI (lint / test matrix 3.11 & 3.12 / build).
- `Makefile` shortcuts.
- `.python-version` pin (3.12 for dev).
- `docs/legacy/` preserving the original 12-channel architecture notes.

### Changed
- Reorganized `src/smoke_detection/` into
  `common/ data/ models/ training/ evaluation/ configs/ cli/`.
- Default data root is now the in-repo `data/` directory (was a sibling
  `dataset_prepared/`). Override via `SMOKEDET_DATA_ROOT` or
  `paths.data_root` YAML field.
- Dependencies consolidated into `pyproject.toml`.
- Python floor raised to 3.11.
- `pyproject.toml` `build-backend` fixed (`setuptools.build_meta`).

### Removed
- `deprecated/` (original 12-channel code). Retained in git history.
- `requirements.txt` (deps now in `pyproject.toml`).
- Task-named subpackages `src/smoke_detection/classification/` and
  `src/smoke_detection/segmentation/`.
- `src/smoke_detection/dataset_paths.py` (replaced by `common/paths.py`).

### Breaking
- Previous entry points `python -m smoke_detection.classification.train`
  and `.segmentation.train` no longer exist. Use
  `python -m smoke_detection.cli.train --config <yaml>`.
- Old `.model` state-dict files from the 12-channel era are not loadable
  into the new 4-channel Lightning modules.
- Metric reporting now uses `torchmetrics`; numeric values may differ
  slightly from the previous hand-rolled aggregation (per-image vs.
  batch-averaged). Paper-accuracy parity has not been re-verified post-port.

## [0.1.0] — Initial release

- Classification (ResNet-50) and segmentation (U-Net) for Sentinel-2 smoke
  plume detection, argparse-based scripts.
