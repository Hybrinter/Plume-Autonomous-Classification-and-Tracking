"""E2E: fast_dev_run for both classification and segmentation CLIs."""

from __future__ import annotations

import pytest

from smoke_detection.cli.train import main as train_main

pytestmark = pytest.mark.e2e


def test_classification_fast_dev_run(classification_yaml_tmp):
    rc = train_main(
        [
            "--config",
            str(classification_yaml_tmp),
            "--override",
            "trainer.fast_dev_run=true",
            "--override",
            "data.batch_size=2",
            "--override",
            "data.num_workers=0",
        ]
    )
    assert rc == 0


def test_segmentation_fast_dev_run(segmentation_yaml_tmp):
    rc = train_main(
        [
            "--config",
            str(segmentation_yaml_tmp),
            "--override",
            "trainer.fast_dev_run=true",
            "--override",
            "data.batch_size=2",
            "--override",
            "data.num_workers=0",
        ]
    )
    assert rc == 0
