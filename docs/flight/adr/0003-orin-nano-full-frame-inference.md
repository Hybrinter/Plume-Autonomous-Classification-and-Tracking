# ADR-FLIGHT-0003: Orin Nano Super full-frame inference budgets

**Status:** Accepted
**Date:** 2026-09-03
**Topic:** interface
**Supersedes:** ADR-FLIGHT-0002 (inference expected/timeout numbers and factory 256 ONNX contract)
**Superseded-by:** none
**Related:** ADR-FLIGHT-0002, ADR-REPO-0004, ADR-TOOLS-0003, ADR-TOOLS-0007

## Context

ADR-FLIGHT-0002 set the demosaiced band plane and inference tensor to 1024 x 1224
and deleted ROI crop. It left factory ONNX at 256 x 256 and named 100 ms expected
latency with a 500 ms FDIR timeout as placeholders. Those numbers were not derived
from the Orin Nano Super 8 GB module. `compute=real` could not load the 256 graphs
against the 1024 x 1224 I/O contract.

Classifier still runs every frame. Segmentor still runs only on a positive
presence decision. The Orin Super study's quality knee is classifier FP16
(INT8 drops 256-tile accuracy) and segmentor INT8 (256-tile IoU holds).

## Decision

- Re-export the factory classifier and segmentor at `(1, 4, 1024, 1224)`. The
  same 256-trained fully convolutional weights are valid at the new spatial size.
  That is a contract fix, not a full-frame quality claim. 256-tile stage-3
  accuracy and IoU remain the published quality numbers.
- Compute target is Jetson Orin Nano Super 8 GB, MAXN SUPER, 25 W module TDP,
  17 FP16 TFLOPS, 33 dense INT8 TOPS, 102 GB/s, no DLA. Camera ingest stays USB3
  Blackfly. Payload-bus FDIR `power_limit_w` stays 55 W and is not module TDP.
- Factory ONNX is the quality knee: classifier FP16 and segmentor INT8 QDQ.
  Graph input and output stay float32. `quantize_knee` overwrites
  `data/models/active_*.onnx` in place. `use_int8` stays false because the
  configured paths already are the quantized graphs, not FP32 files with
  INT8 siblings. Classifier INT8 is not shipped.
- Expected detect latency and FDIR timeout are derived by the
  `orin_nano_full_frame_inference` study: analytic
  `max(compute/eta, memory) / wrap` on the mixed knee, then
  `expected_ms = ceil(t_detect)` and `timeout_ms = ceil(5 * expected_ms)`.
  The live values are `inference.latency_budget_ms = 4` and
  `fault.inference_timeout_ms = 20`. `OnnxDetector` still uses the FDIR timeout.
- onnxruntime provider preference for tools accept/bench is TensorRT, then CUDA,
  then CPU, intersected with what the runtime has. Flight `load_onnx_session`
  passes `providers=None` so a GPU image auto-discovers. The flight extra stays
  CPU onnxruntime.
- `export` `--override-spatial` lets CLI H/W win over checkpoint H/W. Tools
  accept/finalize defaults for latency follow `FaultConfig.inference_timeout_ms`.
  Measure-only call sites keep `max_latency_ms=10_000`.

## Consequences

- `compute=real` can load the factory pair against the band-plane contract.
- Changing Super efficiency constants regenerates the study, then the TOML
  pair, dataclass defaults, and tools CLI defaults must move together.
- On-Nano measurement is still a CSV hook, not a live budget source.
- Repo Python stays 3.14. JetPack 6 system Python stays 3.10. That gap is
  documented; `requires-python` does not change.
- CPU QDQ INT8 is the shipped segmentor graph. Ampere TensorRT may fuse it
  further. FP16 classifier conversion keeps float32 I/O for the flight contract.

## Alternatives considered

- Keep 100/500 ms as live budgets until a Nano bench exists — leaves placeholders
  as if they were derived.
- Ship FP32 ONNX and let TensorRT pick FP16 at session load — leaves the
  segmentor INT8 knee off the factory path.
- Bind providers to `EnvironmentConfig.host` — host is provenance only.
- Retrain at 1024 x 1224 before re-export — quality work, not the I/O contract fix.
- Enable `use_int8` sibling resolution for both nets — classifier INT8 drops
  256-tile accuracy, so a single flag cannot express the mixed knee.
