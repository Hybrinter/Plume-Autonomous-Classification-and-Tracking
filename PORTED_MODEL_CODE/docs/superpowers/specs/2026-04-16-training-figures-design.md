# Training Figure Generation — Design Spec
**Date:** 2026-04-16
**Scope:** `src/smoke_detection/classification/train.py`, `src/smoke_detection/segmentation/train.py`

---

## Context

The train scripts currently log metrics to TensorBoard only. No standalone figures are produced at the end of a run. This spec adds an end-of-training report: metric curves over all epochs plus a 3×3 validation sample prediction grid. Figures are saved to `outputs/{task}/{run_name}/`.

---

## Architecture

**Approach:** Accumulate per-epoch metrics in plain lists inside `train_model()`. After the final epoch, call a `_generate_report()` helper that plots everything and saves to `output_dir`.

`output_dir` is derived from the run name already used for the TensorBoard writer:
```
outputs/classification/4ch_ep{ep}_lr{lr}_bs{bs}_mo{mo}/
outputs/segmentation/4ch_ep{ep}_lr{lr}_bs{bs}_mo{mo}/
```

No new classes, no new files. Two self-contained helper functions added to each train script: `_generate_training_curves()` and `_generate_val_predictions()`, both called from `_generate_report()` at the end of `train_model()`.

---

## Classification `train.py` Changes

### Metric accumulation
Add lists at the top of `train_model()`:
```python
history = {
    'train_loss': [], 'val_loss': [],
    'train_acc': [],  'val_acc': [],
    'lr': [],
}
```
Append to each list at the end of every epoch (values already computed, just need collecting).

### `_generate_training_curves(history, output_dir)`
3-panel matplotlib figure (single row):
- Panel 1: train loss + val loss vs epoch
- Panel 2: train accuracy + val accuracy vs epoch
- Panel 3: learning rate vs epoch (log scale y-axis)

Saved as `{output_dir}/training_curves.png`.

### `_generate_val_predictions(model, device, output_dir)`
- Load 9 samples from the validation set using the standard `create_dataset()` pipeline (transforms applied — normalization required for inference)
- Run model inference
- Display RGB composites with contrast stretching matching eval.py (bands B4, B3, B2 at indices 2, 1, 0 from the 4-channel tensor)
- 3×3 grid of subplots; each cell: RGB composite with title `"True: smoke/clear | Pred: smoke/clear [OK/X]"` (plain ASCII, no unicode)

Saved as `{output_dir}/val_predictions.png`.

---

## Segmentation `train.py` Changes

### Metric accumulation
```python
history = {
    'train_loss': [], 'val_loss': [],
    'train_iou': [],  'val_iou': [],
    'train_acc': [],  'val_acc': [],
    'train_arearatio_mean': [], 'val_arearatio_mean': [],
    'train_arearatio_std': [],  'val_arearatio_std': [],
    'lr': [],
}
```

### `_generate_training_curves(history, output_dir)`
4-panel matplotlib figure (2×2):
- Panel 1: train/val loss vs epoch
- Panel 2: train/val IoU vs epoch
- Panel 3: train/val accuracy vs epoch
- Panel 4: area ratio mean ± std band (train and val) vs epoch

Saved as `{output_dir}/training_curves.png`.

### `_generate_val_predictions(model, device, output_dir)`
- Load 9 samples from the validation set using `create_dataset()` with transforms applied
- Run model inference, threshold at logit ≥ 0
- 9-row × 3-column subplot grid (one row per sample):
  - Col 1: RGB composite (contrast-stretched, bands B4/B3/B2 at indices 2/1/0)
  - Col 2: ground-truth mask (Reds colormap)
  - Col 3: predicted mask (Greens colormap)
- Row label: `"True: smoke/clear | Pred: smoke/clear [OK/X]"` as y-axis label on col 1

Saved as `{output_dir}/val_predictions.png`.

---

## Output Directory

`output_dir` is created with `os.makedirs(output_dir, exist_ok=True)` before saving. The `outputs/` root is already in `.gitignore`.

---

## What Does NOT Change

- TensorBoard logging — unchanged
- Checkpoint saving — unchanged
- `eval.py` scripts — unchanged (they generate their own figures independently)
- `argparse` arguments — no new flags needed; output path is fully derived from existing args

---

## Success Criteria

1. After a completed training run, `outputs/{task}/{run_name}/` contains `training_curves.png` and `val_predictions.png`
2. Figures are readable at default DPI (150)
3. If training is interrupted before the final epoch, no figures are generated (report only runs on clean completion)
4. No import errors — `matplotlib` is already a project dependency
