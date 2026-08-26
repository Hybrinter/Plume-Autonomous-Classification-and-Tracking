"""E2E: train → save ckpt → load_from_checkpoint → bit-identical forward."""

from __future__ import annotations

import pytest
import torch

from smoke_detection.cli.train import main as train_main
from smoke_detection.training.classification_module import ClassificationModule

pytestmark = pytest.mark.e2e


def test_classification_checkpoint_roundtrip(
    classification_yaml_tmp, sample_classification_config, tmp_path
):
    cfg = sample_classification_config
    rc = train_main(
        [
            "--config",
            str(classification_yaml_tmp),
            "--override",
            "trainer.fast_dev_run=false",
            "--override",
            "trainer.max_epochs=1",
        ]
    )
    assert rc == 0

    ckpt_dir = cfg.paths.output_dir / cfg.paths.experiment_name
    last = next(ckpt_dir.rglob("last.ckpt"))
    loaded = ClassificationModule.load_from_checkpoint(str(last), weights_only=False)
    loaded.eval()

    x = torch.randn(2, 4, 90, 90)
    with torch.no_grad():
        a = loaded(x)
        b = loaded(x)
    torch.testing.assert_close(a, b)
