# Results

Measured numbers for the optimization work described in
[`experiments/README.md`](../README.md). Every table here records a sweep or
finalize result. Run directories stay out of git. Reproduce them from the
stage TOML files with `pact-tools inference sweep` and
`pact-tools inference finalize`.

| Stage | Segmentor | Classifier |
| --- | --- | --- |
| 0 baseline, test split, ONNX | done | done |
| 1 recipe sweep | done (30/30) | done (12/12) |
| 2 architecture sweep | done (36/36) | done (21/21) |
| 2.5 extra seeds at the knee | done | done |
| 2.7 refine | done (18/18) | done (12/12) |
| 3 final 256 px + accept | done | done |

Metrics under Stages 0 and 3 are scored on the held-out test split. The split
is the frozen 70/15/15 recipe in `data/manifests/zenodo_4250706_splits.toml` at
seed 0, applied to the sorted paired filenames of Zenodo 4250706. Stages 1 and 2
read the validation split alone.

## Corpus

The two models train on different populations of the same corpus, because the
corpus labels them differently.

| Pack | Samples | Positive | Labels |
| --- | --- | --- | --- |
| Classifier | 21,350 | 3,750 (17.6%) | Plume presence, from the class directory |
| Segmentor | 1,437 | 1,437 (100%) | Label Studio polygons, rasterized to masks |

Test split sizes are 3,202 classifier tiles and 216 segmentor tiles.

## Stage 0: baseline

The architectures the repository shipped before this work, trained from scratch
with BCE under SGD at 128 px, then scored once on the test split.

| | Classifier | Segmentor |
| --- | --- | --- |
| Architecture | ResNet-50 | U-Net (base width 64, depth 4) |
| Parameters | 23,513,217 | 13,391,937 |
| FLOPs | 2.69 G | 15.48 G |
| ONNX bytes | 93,971,104 (89.62 MB) | 53,566,055 (51.08 MB) |
| Accuracy | 0.9750 | -- |
| F1 | 0.9219 | -- |
| Precision | 0.9613 | -- |
| Recall | 0.8856 | -- |
| ROC-AUC | 0.9893 | -- |
| PR-AUC | 0.9673 | -- |
| Brier | 0.0227 | -- |
| Mean IoU | -- | 0.5241 |
| Mean Dice | -- | 0.6513 |
| Blob-gate IoU | -- | 0.5188 |
| BCE | 0.1399 | 0.0442 |

Two readings from the baseline shape the search that follows.

The classifier reaches F1 1.000 on the train split against 0.9219 on test, so it
is memorizing rather than generalizing. A 23.5 M parameter backbone is more
capacity than 21,350 tiles support, which is evidence that a smaller backbone
can hold the metric rather than merely trade against it.

The segmentor clears the 0.5 IoU acceptance gate by 0.024. That is a thin
margin, and it is measured with plain BCE, an objective that scores an empty
mask well when under one percent of pixels are plume. Both facts point at the
objective rather than the architecture as the first thing to change.

## Stage 1: segmentor recipe

Thirty trials over two optimizers, four learning rates, and five objectives,
with the architecture held at `unet_w32`. All thirty completed. Scores are mean
IoU on the validation split, since the test split is reserved for the final
recorded models.

| Optimizer | Rate | Best objective | Mean IoU |
| --- | --- | --- | --- |
| SGD | 0.01 | focal_dice | 0.6010 |
| SGD | 0.01 | bce_dice | 0.5999 |
| AdamW | 0.001 | focal_dice | 0.5961 |
| AdamW | 0.001 | bce_dice | 0.5957 |
| AdamW | 0.001 | bce | 0.5256 |

The objective matters far more than the optimizer. The best composite loss beats
the best plain BCE run by 0.075 mean IoU, which confirms the baseline reading
that an objective scoring an empty mask well on a corpus of under one percent
plume pixels was the first thing to change. Pairing Dice with a pixel loss is
what earns the gain; focal alone is the worst family tried, reaching 0.4258 at
best and peaking at epoch 19 on average before falling away.

The two optimizers differ in robustness rather than in peak score. Across a
tenfold learning-rate change SGD moved 0.07 mean IoU while AdamW moved 0.03, so
AdamW degrades more gently when the rate is not tuned for the architecture at
hand. That is why stage 2 fixes AdamW rather than the nominally winning SGD
cell: the four leading cells sit within 0.005 of each other, which is noise on a
216-image split, and stage 2 must hold one recipe across architectures whose
best rates will differ.

Against the stage 0 baseline on the same validation split, the effect is already
larger than the search that follows it is likely to add.

| | Baseline | Stage 1 best |
| --- | --- | --- |
| Architecture | U-Net width 64 | U-Net width 32 |
| Objective | BCE | focal_dice |
| Parameters | 13,391,937 | 3,350,561 |
| FLOPs | 15.48 G | 3.89 G |
| Validation mean IoU | 0.5509 | 0.6010 |

A model with four times fewer parameters and four times fewer FLOPs scores 0.050
higher. Size and quality are not in tension yet, which means the frontier in
stage 2 should be drawn well inside the baseline's cost.

## Stage 1: classifier recipe

Twelve trials over two optimizers, three learning rates, and two objectives,
with the architecture held at `resnet18_pt`. All twelve completed.

| Optimizer | Rate | Objective | Val F1 |
| --- | --- | --- | --- |
| SGD | 0.01 | bce | 0.9543 |
| AdamW | 0.001 | bce | 0.9525 |
| AdamW | 0.0001 | bce | 0.9518 |
| AdamW | 0.001 | focal | 0.9450 |
| AdamW | 0.0001 | focal | 0.9359 |
| AdamW | 0.01 | bce | 0.9136 |
| SGD | 0.01 | focal | 0.8928 |
| SGD | 0.001 | bce | 0.8895 |
| AdamW | 0.01 | focal | 0.8708 |
| SGD | 0.0001 | bce | 0.6832 |
| SGD | 0.001 | focal | 0.6571 |
| SGD | 0.0001 | focal | 0.2731 |

BCE beats focal at every matched optimizer and rate. The top three cells sit
within 0.003 F1 on a 3,202-image validation split, which is noise, so they are
tied. What separates them is robustness: AdamW holds 0.952 across 0.0001 and
0.001, while SGD drops from 0.954 at 0.01 to 0.890 at 0.001 and 0.683 at
0.0001. Stage 2 therefore holds AdamW at 0.001 with BCE, the centre of the
stable AdamW plateau rather than the nominally winning SGD cell.

Against the stage 0 baseline on the same validation split:

| | Baseline | Stage 1 best |
| --- | --- | --- |
| Architecture | ResNet-50, scratch | ResNet-18, ImageNet |
| Objective | BCE | BCE |
| Parameters | 23,513,217 | 11,180,161 |
| FLOPs | 2.69 G | 1.21 G |
| Validation F1 | 0.9311 | 0.9543 |

Half the parameters, half the FLOPs, and 0.023 higher F1. As with the
segmentor, size and quality are not in tension before the architecture search
has run.

## Stage 2 search space

All eighteen segmentor candidates build and were costed before training, at
128 px with four input bands. The ladder spans a factor of 960 in parameters.

| Architecture | Parameters | FLOPs |
| --- | --- | --- |
| dilatenet_w32 | 23,521 | 0.05 G |
| unet_w8_sep | 29,773 | 0.05 G |
| dilatenet_w64 | 83,905 | 0.18 G |
| unet_w16_sep | 105,973 | 0.15 G |
| unet_w8 | 210,377 | 0.25 G |
| unet_w16_d3 | 211,217 | 0.77 G |
| dilatenet_w128 | 315,265 | 0.66 G |
| unet_w32_sep | 397,765 | 0.51 G |
| dilatenet_w64_full | 667,393 | 1.38 G |
| unet_w16 | 838,929 | 0.98 G |
| unet_w32_d3 | 841,761 | 3.06 G |
| unet_w32 | 3,350,561 | 3.89 G |
| runet18_pt_x8 | 11,747,529 | 1.43 G |
| runet18 / runet18_pt | 12,464,017 | 1.81 G |
| unet (baseline) | 13,391,937 | 15.48 G |
| runet18_pt_x32 | 14,342,817 | 3.10 G |
| runet34_pt | 22,572,177 | 3.02 G |

The two cost axes do not agree, which matters for how the frontier is read. The
baseline U-Net and `runet18_pt_x32` sit within seven percent of each other on
parameters, yet the baseline costs five times the FLOPs, because its parameters
live in wide layers running at high resolution while the encoder's live in deep
layers running at low resolution. Exported bytes track parameters and latency
tracks FLOPs, so an architecture can be good on one axis and poor on the other.
The frontier is therefore drawn twice, once against each cost.

All twenty-one classifier candidates also build. The ladder spans a factor of
2,190 in parameters, from a purpose-built compact network to the ResNet-50
baseline. Each torchvision backbone appears twice with the same parameter count,
once from scratch and once with ImageNet weights, so a pretrained win is
attributable to the weights rather than to capacity.

| Architecture | Parameters | FLOPs |
| --- | --- | --- |
| pactnet_w8 | 10,729 | 0.01 G |
| pactnet | 37,585 | 0.02 G |
| pactnet_w32 | 139,681 | 0.05 G |
| pactnet_w16_d5 | 141,265 | 0.02 G |
| pactnet_w64 | 271,169 | 0.14 G |
| pactnet_w32_d5 | 277,409 | 0.05 G |
| pactnet_w16_full | 291,937 | 0.09 G |
| shufflenetv2_x0_5 / `_pt` | 343,033 | 0.03 G |
| mobilenetv3_small / `_pt` | 1,519,025 | 0.04 G |
| efficientnet_b0 / `_pt` | 4,009,117 | 0.25 G |
| mobilenetv3_large / `_pt` | 4,203,457 | 0.15 G |
| resnet18 / `_pt` | 11,180,161 | 1.21 G |
| resnet34 / `_pt` | 21,288,321 | 2.42 G |
| resnet50 / `_pt` (baseline) | 23,513,217 | 2.69 G |

## Stage 2: segmentor

All thirty-six trials finished. Mean validation IoU, two seeds:

| Architecture | Parameters | Mean IoU |
| --- | --- | --- |
| unet_w32 | 3,350,561 | 0.6193 |
| unet (baseline graph) | 13,391,937 | 0.6182 |
| runet18_pt_x32 | 14,342,817 | 0.6147 |
| runet18_pt_x8 | 11,747,529 | 0.6114 |
| runet34_pt | 22,572,177 | 0.6079 |
| runet18_pt | 12,464,017 | 0.6066 |
| unet_w32_d3 | 841,761 | 0.6033 |
| unet_w16 | 838,929 | 0.6000 |
| runet18 (scratch) | 12,464,017 | 0.5988 |
| unet_w16_d3 | 211,217 | 0.5894 |
| unet_w32_sep | 397,765 | 0.5882 |
| unet_w8 | 210,377 | 0.5853 |
| dilatenet_w64_full | 667,393 | 0.5843 |
| unet_w16_sep | 105,973 | 0.5761 |
| dilatenet_w128 | 315,265 | 0.5712 |
| dilatenet_w64 | 83,905 | 0.5538 |
| unet_w8_sep | 29,773 | 0.5482 |
| dilatenet_w32 | 23,521 | 0.5367 |

The ImageNet encoders do not earn their size. `runet18_pt` beats the same graph
from scratch by 0.008 IoU, and none of the pretrained family beats `unet_w32`.
Retraining the baseline graph under the stage 1 recipe already reaches 0.6182,
so the remaining gain from `unet_w32` is 0.001 and the real result is the size
cut: 3.35 M parameters against 13.4 M at the same score.

The parameter-axis knee is `dilatenet_w64`. Its two-seed mean is 0.5538, which
holds the stage 0 validation floor of 0.5509, at 84 k parameters. That is 22 k
below `unet_w16_sep` (0.5761). `unet_w8_sep` (0.5482) and `dilatenet_w32`
(0.5367) fall under the floor. `dilatenet_w128` scores 0.5712 at 315 k
parameters. `dilatenet_w64_full` scores 0.5843 at 667 k parameters. Both are
larger than the knee, so they cannot cheapen it.

The Pareto command with `--baseline 0.5509` selects `unet_w8_sep`,
`dilatenet_w64`, and `unet_w16_sep` as the knee plus one neighbour on each side.
Stage 2.5 trains those three under seeds 2 and 3. The decoder-free graph is the
size winner unless a later instruction keeps U-Net skips and takes
`unet_w16_sep`.

On FLOPs the knee is `unet_w16_sep` (0.5761 at 0.15 G), not `dilatenet_w64`.
The dilated net is cheaper in parameters (84 k vs 106 k) but costs 0.18 G
FLOPs against 0.15 G for the separable U-Net, and it scores worse, so the
FLOP frontier drops it. The FLOP neighbourhood around the 0.5509 floor is
`unet_w8_sep` (0.05 G, under the floor), `unet_w16_sep`, and `unet_w8`
(0.5853 at 0.25 G). The knee rule still reads the parameter axis first;
exported bytes follow parameters. The FLOP reading is the latency check after
stage 3 measures 256 px.

## Stage 2: classifier

All twenty-one trials finished. Validation F1, one seed except where noted:

| Architecture | Parameters | Val F1 |
| --- | --- | --- |
| efficientnet_b0_pt | 4,009,117 | 0.9615 |
| resnet50_pt | 23,513,217 | 0.9599 |
| resnet18_pt | 11,180,161 | 0.9566 |
| resnet34_pt | 21,288,321 | 0.9509 |
| mobilenetv3_large_pt | 4,203,457 | 0.9489 |
| efficientnet_b0 | 4,009,117 | 0.9412 |
| resnet50 (baseline graph) | 23,513,217 | 0.9369 |
| shufflenetv2_x0_5_pt | 343,033 | 0.9339 |
| resnet34 | 21,288,321 | 0.9337 |
| resnet18 | 11,180,161 | 0.9329 |
| mobilenetv3_large | 4,203,457 | 0.9252 |
| mobilenetv3_small_pt | 1,519,025 | 0.9224 |
| pactnet_w64 | 271,169 | 0.9151 |
| pactnet_w16_full | 291,937 | 0.9080 |
| pactnet_w32_d5 | 277,409 | 0.9013 |
| mobilenetv3_small | 1,519,025 | 0.9011 |
| pactnet_w32 | 139,681 | 0.8887 |
| pactnet_w16_d5 | 141,265 | 0.8715 |
| pactnet | 37,585 | 0.8635 |
| shufflenetv2_x0_5 (scratch) | 343,033 | 0.8592 |
| pactnet_w8 | 10,729 | 0.7982 |

Pretrained weights are worth more than extra capacity. ShuffleNet from scratch
scores 0.859; the same graph with ImageNet weights scores 0.934. EfficientNet-B0
gains 0.020 from the same treatment. The ResNet-50 baseline graph under the
stage 1 recipe reaches 0.9369, 0.006 above its stage 0 validation F1 of 0.9311,
so the remaining quality on the catalog is pretrained compact backbones rather
than a wider ResNet.

The parameter-axis knee against the 0.9311 floor is `shufflenetv2_x0_5_pt`.
It holds the floor at 343 k parameters. `pactnet_w64` (271 k, 0.9151) falls
under it. EfficientNet-B0 ImageNet (0.9615 at 4.0 M) and ResNet-18 ImageNet
(0.9566 at 11.2 M) score higher and sit further up the frontier; they cannot
cheapen the knee. Stage 2.5 therefore repeats ShuffleNet, `pactnet_w64`, and
EfficientNet-B0 ImageNet under three further seeds.

## Post-training quantization

Exported bytes are a size axis of their own, separate from parameter count.
Running the exporter with `--int8` applies post-training quantization calibrated
on the training split, which stores weights as one byte instead of four. The
table below scores each artifact through the acceptance gate against the whole
test split, so the quantized numbers are measured on the quantized graph rather
than inferred from the checkpoint.

| Artifact | ONNX bytes | Quality | Worst latency |
| --- | --- | --- | --- |
| Classifier fp32 | 89.62 MB | Accuracy 0.975 | 69.7 ms |
| Classifier int8 | 22.58 MB | Accuracy 0.974 | 124.8 ms |
| Segmentor fp32 | 51.08 MB | Mean IoU 0.524 | 109.1 ms |
| Segmentor int8 | 12.83 MB | Mean IoU 0.526 | 111.9 ms |

Quantization divides exported bytes by very nearly four on both models while
quality moves by 0.001 or less, which is inside the noise of a single run. It is
therefore a lever the architecture search does not have to pay for, and it
applies to whichever architecture the search selects.

Latency rises rather than falls, because these sessions run on CPU where the
quantized kernels for this operator set are slower than the float ones. Both
artifacts stay far inside the 500 ms budget, so the trade costs nothing that the
gate measures.

## Stage 2.5: extra seeds

Four-seed validation means at the architectures stage 2 placed on or beside
the knee.

| Architecture | Parameters | Seeds | Mean | Range |
| --- | --- | --- | --- | --- |
| Classifier `shufflenetv2_x0_5_pt` | 343,033 | 4 | F1 0.9310 | 0.0110 |
| Classifier `pactnet_w64` | 271,169 | 4 | F1 0.9059 | 0.0230 |
| Classifier `efficientnet_b0_pt` | 4,009,117 | 4 | F1 0.9601 | 0.0116 |
| Segmentor `unet_w8_sep` | 29,773 | 4 | IoU 0.5540 | 0.0152 |
| Segmentor `dilatenet_w64` | 83,905 | 4 | IoU 0.5555 | 0.0154 |
| Segmentor `unet_w16_sep` | 105,973 | 4 | IoU 0.5739 | 0.0116 |
| Segmentor `dilatenet_w32` | 23,521 | 2 | IoU 0.5367 | 0.0098 |

ShuffleNet's four-seed mean is 0.9310 against the 0.9311 floor. That is inside
a 0.011 seed range, so the drop is not a drop. `pactnet_w64` stays under the
floor by 0.025. EfficientNet stays well above it at twelve times the
parameters. The classifier knee therefore stays ShuffleNet.

The extra seeds lifted `unet_w8_sep` from 0.5482 (under the floor) to a
four-seed mean of 0.5540. Auto-spread then subtracted that 0.0152 range from
the 0.5509 floor and produced a new floor of 0.5357. `dilatenet_w32` (two-seed
mean 0.5367, 23,521 parameters) then became the parameter-axis knee. That is a
noisier selection than the two-seed reading that had placed `dilatenet_w64`
(0.5538 / 84 k) or the FLOP knee `unet_w16_sep` (0.5761 / 0.15 G) at the
cut. Stage 2.7 and stage 3 therefore trained `dilatenet_w32`. The 256 px
U-Net retrain in stage 3 is the fair quality comparison against that choice.

## Stage 2.7: refine

Twelve classifier trials on ShuffleNet (weight decay, `pos_weight`,
augmentation) and eighteen segmentor trials on DilateNet-w32 (batch size,
`pos_weight`, weight decay).

The classifier winner is weight decay 0.001, `pos_weight` 4.69, and
augmentation off, at validation F1 0.9467
(`classifier-shufflenetv2_x0_5_pt-0-fd6f054a`). Turning augmentation off and
raising decay both cut the overfitting the baseline showed (train F1 1.000
against test 0.9219). The class-balance weight of 4.69 is the
negative-to-positive ratio of the corpus.

The segmentor winner is batch size 8, `pos_weight` 32, and weight decay
0.0001, at validation IoU 0.5388
(`segmentor-dilatenet_w32-0-0dcda522`). That is a small move on a 23 k
parameter graph whose two-seed mean was already 0.5367. Stage 3 copies those
scalars and retrains at 256 px.

## Stage 3: 256 px test split and acceptance

Both selected architectures, and the original graphs retrained under the same
recipe and resolution, scored once on the held-out test split. The flight
input is 256 px. Stage 0 recorded 128 px, so the 256 px U-Net and ResNet-50
retrains are the fair architecture comparison; the stage 0 numbers remain the
recorded baseline the search had to beat.

### Classifier (test N = 3,202)

| | Stage 0 ResNet-50 @128 | Stage 3 ResNet-50 @256 | ShuffleNetV2-x0.5 ImageNet @256 |
| --- | ---: | ---: | ---: |
| Parameters | 23,513,217 | 23,513,217 | **343,033** |
| FLOPs | 2.69 G | 10.78 G | **0.110 G** |
| Accuracy | 0.9750 | 0.9738 | **0.9803** |
| F1 | 0.9219 | 0.9216 | **0.9398** |
| Precision | 0.9613 | 0.9165 | 0.9572 |
| Recall | 0.8856 | 0.9268 | **0.9231** |
| ROC-AUC | 0.9893 | 0.9903 | **0.9935** |
| PR-AUC | 0.9673 | 0.9752 | **0.9825** |
| Brier | 0.0227 | 0.0214 | **0.0161** |
| ONNX FP32 | 89.62 MB | 89.62 MB | **1.35 MB** (1,419,085 B) |
| ONNX INT8 | 22.58 MB | 22.58 MB | **0.54 MB** (563,825 B) |
| FP32 accept | acc 0.975 / 69.7 ms | acc 0.974 / 111.6 ms | **acc 0.980 / 3.4 ms** |
| INT8 accept | acc 0.974 / 124.8 ms | acc 0.973 / 103.9 ms | acc 0.932 / 28.8 ms |

ShuffleNet clears the 0.9 accuracy gate on FP32 and INT8. It beats the recorded
stage 0 baseline on every reported metric except precision, at 68 times fewer
parameters and 66 times fewer exported bytes. It also beats the fair 256 px
ResNet-50 retrain. INT8 drops accuracy from 0.980 to 0.932 and raises worst
latency from 3.4 ms to 28.8 ms, so the shipped graph is FP32.

Runs: `classifier-resnet50-0-13d58d4b`,
`classifier-shufflenetv2_x0_5_pt-0-bde76f1e`.

### Segmentor (test N = 216)

| | Stage 0 U-Net @128 | Stage 3 U-Net @256 | DilateNet-w32 @256 |
| --- | ---: | ---: | ---: |
| Parameters | 13,391,937 | 13,391,937 | **23,521** |
| FLOPs | 15.48 G | 61.92 G | **0.21 G** |
| Mean IoU | 0.5241 | **0.5780** | 0.5532 |
| Mean Dice | 0.6513 | **0.7071** | 0.6869 |
| Blob-gate IoU | 0.5188 | **0.5778** | 0.5519 |
| ONNX FP32 | 51.08 MB | 51.08 MB (53,566,055 B) | **95.6 KB** (97,868 B) |
| ONNX INT8 | 12.83 MB | 12.83 MB (13,448,253 B) | **47.0 KB** (48,099 B) |
| FP32 accept | IoU 0.524 / 109 ms | IoU 0.578 / 204 ms | **IoU 0.553 / 2.3 ms** |
| INT8 accept | IoU 0.526 / 112 ms | IoU 0.578 / 207 ms | IoU 0.553 / 2.0 ms |

Both stage 3 segmentors clear the 0.5 IoU gate and the 500 ms latency budget.
DilateNet-w32 beats the recorded stage 0 baseline by 0.029 mean IoU at 569
times fewer parameters. It loses 0.025 IoU to the fair 256 px U-Net retrain.
U-Net's 204 ms worst latency sits outside a 100 ms control loop
(`kalman_dt_s = 0.1`); DilateNet does not.

Runs: `segmentor-unet-0-02a955e4`, `segmentor-dilatenet_w32-0-64421cfe`.

### Selected models

- Classifier: ShuffleNetV2-x0.5 with ImageNet stem surgery, FP32 ONNX.
  Test accuracy 0.980, F1 0.940, 343 k parameters, 1.35 MB, 3.4 ms.
  Factory path: `data/models/active_classifier.onnx`.
- Segmentor: DilateNet-w32, FP32 ONNX.
  Test mean IoU 0.553, 23.5 k parameters, 96 KB, 2.3 ms.
  Factory path: `data/models/active_segmentor.onnx`.

`config/default.toml` points `[inference]` at those two paths. Flight loads
them through `OnnxDetector` when `environment.compute` is `real`.
