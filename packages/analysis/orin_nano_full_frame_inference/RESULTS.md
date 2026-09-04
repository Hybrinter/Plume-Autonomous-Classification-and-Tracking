# Orin Nano Super full-frame inference

TEMPORARY ANALYSIS. Not flight software.

This study derives the flight expected-detect latency and the FDIR
inference timeout from Orin Nano Super peaks, model FLOPs, and a named
efficiency policy. It does not use placeholder millisecond targets.

## Headline

| Field | Value |
| --- | --- |
| Board | Jetson Orin Nano Super 8 GB |
| Shipped precision | mixed |
| Detect wall (analytic) | 3.27 ms |
| **Expected latency** | **4 ms** |
| **FDIR timeout** | **20 ms** |
| Timeout policy | `ceil(5 x expected)` |

Write `expected_ms` into `inference.latency_budget_ms` and `timeout_ms`
into `fault.inference_timeout_ms`.

## Board lock

- Ampere 1024 CUDA / 32 tensor cores.
- 17 FP16 TFLOPS, 2.08 FP32 TFLOPS (CUDA cores).
- 33 dense / 67 sparse INT8 TOPS.
- 8 GB LPDDR5 at 102 GB/s.
- Module Super TDP 25 W. Payload-bus FDIR stays 55 W.
- DLA present: False.
- Camera: BFS-U3-50S5 on USB3 (no CSI path in this repo).
- Power mode for a later Nano CSV: `MAXN SUPER` + `jetson_clocks`
  (True), JetPack 6.2+.
- Repo Python 3.14; JetPack system Python 3.10.
  This study does not change `requires-python`.

## Models

| | Classifier | Segmentor |
| --- | --- | --- |
| Arch | `shufflenetv2_x0_5_pt` | `dilatenet_w32` |
| Params | 343 k | 23.5 k |
| FLOPs @ 256x256 | 0.110 G | 0.21 G |
| FLOPs @ 1024x1224 | 2.104 G | 4.02 G |
| Area scale | 19.125 | 19.125 |
| 256-tile FP32 quality | acc 0.980 | IoU 0.553 |
| 256-tile INT8 quality | acc 0.932 | IoU 0.553 |

Re-export at 1024x1224 is a contract fix. It does not create a full-frame
IoU number. Quality remains the 256-tile stage-3 measurement.

## Duty cycle

Classifier runs every frame. Segmentor runs only on a positive presence
decision. `t_search = t_cls`. `t_detect = t_cls + t_seg`.

| Precision | t_search (ms) | t_detect (ms) | cls kernel | seg kernel |
| --- | --- | --- | --- | --- |
| fp32 | 13.49 | 39.23 | 6.74 | 12.87 |
| fp16 | 1.65 | 4.80 | 0.83 | 1.57 |
| int8 | 1.58 | 3.20 | 0.79 | 0.81 |

## Latency model

- Compute bound: `GFLOP / peak_TFLOPS`.
- Memory bound: bytes / (102 GB/s).
- Kernel: `max(compute / 0.15, memory)`.
- Wall: `kernel / 0.5` (ORT + Python wrap).
- `expected_ms = ceil(t_detect)` for the shipped mixed knee.
- `timeout_ms = ceil(5 * expected_ms)`.

## Quantization knee

- Segmentor INT8 keeps 256-tile IoU. INT8 is the segmentor quality knee.
- Classifier INT8 drops accuracy 0.980 to 0.932. FP16 is the classifier
  working knee (same accuracy as FP32, Super tensor cores).
- Mixed knee detect wall (cls FP16 + seg INT8): 3.27 ms.
- Factory ONNX is that pair: classifier FP16, segmentor INT8 QDQ.
  Graph I/O stay float32. `use_int8` stays false because the configured
  paths already point at the quantized graphs, not FP32 siblings.

## Figures

![Duty-cycle latency](outputs/duty_cycle_latency.png)

![Quantization Pareto](outputs/quantization_pareto.png)

![Artifact size](outputs/artifact_size.png)

![FLOPs vs spatial size](outputs/flops_vs_spatial.png)

![Mean time vs plume-positive rate](outputs/mean_time_vs_positive_rate.png)

CPU ORT traces are not plotted as Orin. A laptop-GPU CSV, when present,
is labelled separately from Super estimates.

