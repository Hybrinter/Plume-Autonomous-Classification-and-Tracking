.PHONY: install lint format test test-e2e test-gpu test-all clean train-cls train-seg eval-cls eval-seg

install:
	uv sync --extra dev

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -m "not slow and not e2e and not gpu"
	uv run pytest -m "slow" --cov-append --cov-fail-under=80

test-e2e:
	uv run pytest -m "e2e" --timeout=300

test-gpu:
	uv run pytest -m "gpu"

test-all:
	uv run pytest -m "not gpu"

CONFIG ?= configs/classification/default.yaml
CKPT ?=

train-cls:
	uv run python -m smoke_detection.cli.train --config configs/classification/default.yaml

train-seg:
	uv run python -m smoke_detection.cli.train --config configs/segmentation/default.yaml

eval-cls:
	uv run python -m smoke_detection.cli.eval --config configs/classification/default.yaml --ckpt $(CKPT)

eval-seg:
	uv run python -m smoke_detection.cli.eval --config configs/segmentation/default.yaml --ckpt $(CKPT)

clean:
	rm -rf lightning_logs/ .pytest_cache/ .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
