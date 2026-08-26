> **LEGACY DOCUMENT — preserved for reference.** Slide outline for the
> original 12-channel model. Retained only for historical/reference use.

# Google Slides Content Outline - Industrial Smoke Plume Detection Architecture

## Slide 1 - Title

**Title:** Industrial Smoke Plume Detection from Sentinel-2
**Subtitle:** Architecture of Classification + Segmentation Models
**Presenter fields:** Name, affiliation, date

**Speaker notes (optional):**
- Introduce the goal: detect and characterize industrial smoke plumes from multispectral satellite imagery.
- Mention this is based on a two-stage deep learning pipeline.

---

## Slide 2 - Problem and Motivation

**Main points:**
- Industrial emissions are a major climate driver.
- Need scalable monitoring over large geographies.
- Satellite imagery provides global, repeated observations.

**Visual suggestion:**
- World map / satellite montage + one smoke plume example image.

---

## Slide 3 - Data and Inputs

**Main points:**
- Input source: Sentinel-2 multispectral GeoTIFF image patches.
- Model input uses 12 spectral channels (bands 1-10, 12, 13).
- Band 11 is excluded.
- Images are standardized to `120 x 120` pixels before augmentation/cropping.

**Visual suggestion:**
- Channel list and simple input tensor diagram (`12 x H x W`).

---

## Slide 4 - End-to-End Pipeline

**Main points:**
- Stage 1: Classification (`smoke` vs `no smoke`).
- Stage 2: Segmentation (pixel-wise smoke mask).
- Benefits:
  - fast filtering by classifier,
  - detailed plume extent from segmentation.

**Visual suggestion:**
- Flowchart: Input -> Classifier -> (positive subset) -> U-Net -> mask + area estimate.

---

## Slide 5 - Classification Architecture (Major Section)

**Main points:**
- Backbone: pretrained ResNet-50.
- Modified first conv layer: 3 channels -> 12 channels.
- Modified output head: 1 logit for binary prediction.
- Uses multispectral information rather than RGB-only cues.

**Visual suggestion:**
- ResNet block diagram with highlighted modified input/output layers.

---

## Slide 6 - Classification Data and Training

**Main points:**
- Dataset structure: `positive/` and `negative/` image folders.
- Class balancing by upsampling/downsampling.
- Train transforms: normalize, random crop (`120 -> 90`), random flips/rotations.
- Loss: `BCEWithLogitsLoss`; optimizer: SGD; LR scheduler: ReduceLROnPlateau.
- Primary metric: image-level accuracy.

**Visual suggestion:**
- Compact table: preprocessing, augmentation, loss, optimizer, metric.

---

## Slide 7 - Classification Evaluation and Explainability

**Main points:**
- Predictions thresholded at logit `>= 0`.
- Evaluation computes test accuracy.
- Diagnostic mode can export:
  - RGB view,
  - false-color spectral composite,
  - intermediate activation maps.

**Visual suggestion:**
- 3-panel sample diagnostic image.

---

## Slide 8 - Segmentation Architecture (Major Section)

**Main points:**
- Model: U-Net with encoder-decoder and skip connections.
- Input: 12 channels; output: 1-channel smoke logit map.
- Encoder channel progression: 64 -> 128 -> 256 -> 512 -> bottleneck.
- Decoder upsamples and fuses encoder features for localization.

**Visual suggestion:**
- U-Net "U-shape" figure with skip connections emphasized.

---

## Slide 9 - Segmentation Labels, Training, and Metrics

**Main points:**
- Labels are polygon annotations from JSON.
- Polygons rasterized to binary masks.
- Dataset built with image-level positive/negative balancing.
- Loss: `BCEWithLogitsLoss`; optimizer: SGD; scheduler: ReduceLROnPlateau.
- Metrics:
  - IoU (Jaccard) for overlap quality,
  - image-level smoke/no-smoke accuracy,
  - smoke area ratio (`predicted/true` area).

**Visual suggestion:**
- Example: image + ground-truth mask + prediction overlay.

---

## Slide 10 - Reported Performance and Takeaways

**Main points (from project README):**
- Classification accuracy reported around **94.3%**.
- Segmentation IoU reported around **0.608**.
- Segmentation-based smoke detection accuracy reported around **94.0%**.
- Average smoke-area reproduction error reported around **5.6%**.

**Visual suggestion:**
- Metric cards or bar chart with 4 key numbers.

---

## Slide 11 - Strengths, Limitations, and Risks

**Strengths:**
- Multispectral sensing improves smoke discrimination.
- Two-stage setup combines efficiency and detailed output.
- Segmentation output enables area-based monitoring.

**Limitations noted in project text/code:**
- Potential confusion with some surface objects.
- Difficulty with semi-transparent smoke.
- Path/configuration in scripts is mostly manual.
- Fixed decision threshold may be suboptimal for some deployments.

---

## Slide 12 - Future Improvements

**Main points:**
- Calibrate thresholds and probability outputs.
- Add robust config system (replace hard-coded paths).
- Explore stronger augmentations and modern backbones.
- Add uncertainty estimation and temporal consistency checks.
- Operationalize as monitoring pipeline (batch inference + reporting).

---

## Slide 13 - Closing

**Main points:**
- Recap: architecture, training strategy, and practical monitoring value.
- Emphasize climate-relevant application.
- Invite questions.

**Optional final slide footer:**
- Reference paper + code repository + dataset link.

---

## Appendix Slide Ideas (Optional)

- Detailed channel explanation and why multispectral helps.
- Training hyperparameters and hardware setup.
- Error case gallery: false positives/false negatives.
- Additional qualitative segmentation examples.
