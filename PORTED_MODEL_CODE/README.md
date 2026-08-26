# Industrial Smoke Plume Detection

[![CI](https://github.com/ShadHall/IndustrialSmokePlumeDetection/actions/workflows/ci.yml/badge.svg)](https://github.com/ShadHall/IndustrialSmokePlumeDetection/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/ShadHall/IndustrialSmokePlumeDetection/branch/master/graph/badge.svg)](https://codecov.io/gh/ShadHall/IndustrialSmokePlumeDetection)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/downloads/)

This repository implements a two-stage PyTorch Lightning pipeline for detecting
and segmenting industrial smoke plumes from Sentinel-2 multispectral satellite
imagery (4 channels: B2, B3, B4, B8).

Based on the publication *Characterization of Industrial Smoke Plumes from
Remote Sensing Data*, NeurIPS 2020 *Tackling Climate Change with Machine
Learning* workshop.

![segmentation example images](assets/segmentation.png "Segmentation Example Images")

## Project Structure

```
configs/                 experiment YAMLs
data/                    prepared dataset (gitignored)
docs/                    architecture + legacy docs
notebooks/               exploratory notebooks
scripts/                 dataset preparation + smoke tests
src/smoke_detection/
  common/                seed, paths, logging
  data/                  Datasets + LightningDataModules + transforms
  models/                pure nn.Module definitions (ResNet-50, U-Net)
  training/              LightningModules (loss, optimizer, metrics, steps)
  evaluation/            plotting helpers (confusion matrix, ROC, IoU hist)
  configs/               pydantic schemas + YAML loader
  cli/                   train.py, eval.py entry points
tests/                   pytest scaffolding
```

## Installation

Requires Python >= 3.11 (3.12 recommended; see `.python-version`) and
[`uv`](https://github.com/astral-sh/uv).

    uv sync --extra dev

Or from pip/venv:

    pip install -e ".[dev]"

## Data Preparation

1. Download the dataset from [Zenodo](http://doi.org/10.5281/zenodo.4250706)
   and extract it (you should have a `4250706/` directory containing
   `images/` and `segmentation_labels/`).
2. Generate train/val/test splits into the default `data/` location:

```bash
python scripts/prepare_dataset.py --source /path/to/4250706 --output data
```

Override the default root via `SMOKEDET_DATA_ROOT=/some/path`.

The expected layout after preparation:

```
data/
  classification/{train,val,test}/{positive,negative}/*.tif
  segmentation/{train,val,test}/images/{positive,negative}/*.tif
  segmentation/{train,val,test}/labels/*.json
```

## Training

All training goes through a single CLI, parameterized by YAML config:

```bash
# Classification
python -m smoke_detection.cli.train --config configs/classification/default.yaml

# Segmentation
python -m smoke_detection.cli.train --config configs/segmentation/default.yaml

# Override any config field via dotted overrides
python -m smoke_detection.cli.train \
    --config configs/classification/default.yaml \
    --override optim.lr=1e-3 \
    --override trainer.max_epochs=10
```

Or via `make`:

```bash
make train-cls        # classification
make train-seg        # segmentation
```

Logs, checkpoints, and TensorBoard events land in
`lightning_logs/<experiment_name>/version_N/`.

## Evaluation

```bash
python -m smoke_detection.cli.eval \
    --config configs/segmentation/default.yaml \
    --ckpt lightning_logs/segmentation_4ch_unet/version_0/checkpoints/last.ckpt
```

The eval CLI writes confusion matrix + ROC plots (classification) or IoU +
area-ratio histograms (segmentation) into
`lightning_logs/<experiment_name>/eval/`.

## Dev Loop

```bash
make install          # uv sync --extra dev
make lint             # ruff check + ruff format --check
make format           # auto-fix
make test             # pytest
make clean
```

Pre-commit hooks:

```bash
uv run pre-commit install
```

## Citation

    Mommert, M., Sigel, M., Neuhausler, M., Scheibenreif, L., Borth, D.,
    "Characterization of Industrial Smoke Plumes from Remote Sensing Data",
    Tackling Climate Change with Machine Learning Workshop, NeurIPS 2020.

## License

GPL v3 — see `LICENSE`. U-Net code adapted from
[milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) (GPL v3).
