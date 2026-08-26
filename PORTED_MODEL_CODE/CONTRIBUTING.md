# Contributing

Thanks for your interest in contributing.

## Dev Setup

1. Install [`uv`](https://github.com/astral-sh/uv).
2. Clone the repo and install dev dependencies:

   ```bash
   uv sync --extra dev
   ```

3. Install pre-commit hooks:

   ```bash
   uv run pre-commit install
   ```

## Branching

- Branch from `master`: `git checkout -b feat/<short-name>` or `fix/<short-name>`.
- Open a PR against `master`.
- CI (lint + test + build) must pass before merging.

## Code Style

- Formatter + linter: [`ruff`](https://docs.astral.sh/ruff/) (config in `pyproject.toml`).
- Run locally:

  ```bash
  make lint     # check
  make format   # auto-fix
  ```

- Line length 100. Target Python 3.11.

## Tests

- Framework: `pytest`. Tests live under `tests/`.
- Marker tiers:
  - **default** (unmarked) — fast unit + integration; CPU; < 60s on a cold runner.
  - **`slow`** — pretrained ResNet-50 load, overfitting sanity loops.
  - **`gpu`** — requires CUDA; auto-skipped when unavailable.
  - **`e2e`** — CLI end-to-end, fast_dev_run, checkpoint round-trip.
- Make targets:
  - `make test` — default + slow tiers with an 80% coverage gate.
  - `make test-e2e` — e2e tier only.
  - `make test-gpu` — gpu tier only.
  - `make test-all` — everything except `gpu`.
- Pretrained weights are cached under `~/.cache/torch/hub/checkpoints/`. In CI
  they're restored by `actions/cache` keyed on `resnet50-imagenet-v1`. Bump
  the key if the torchvision weights URL changes upstream.
- Tests never hit the network. Synthetic data is generated on the fly by
  `tests/_data.py`.

## Commit Messages

Use [Conventional Commits](https://www.conventionalcommits.org/) prefixes:

- `feat(scope): …` – new functionality
- `fix(scope): …` – bug fix
- `refactor(scope): …` – no behavior change
- `docs: …`, `chore: …`, `ci: …`, `test: …`

## Running Experiments

- Experiment configs live in `configs/<task>/*.yaml`.
- Never hardcode hyperparameters in Python code — add a field to the pydantic
  schema in `src/smoke_detection/configs/` and reference it from the
  LightningModule / DataModule.
- `lightning_logs/` is gitignored. Do not commit checkpoints or TensorBoard
  event files.
