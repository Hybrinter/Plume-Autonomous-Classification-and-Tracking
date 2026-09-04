# Orin Nano Super full-frame inference

TEMPORARY ANALYSIS. Not flight software. Design study for full-frame
1024x1224 inference on Jetson Orin Nano Super 8 GB.

```text
uv run python -m analysis.studies.orin_nano_full_frame_inference all
uv run python -m analysis.studies.orin_nano_full_frame_inference bench
```

Geometry and budgets: [`RESULTS.md`](RESULTS.md). Figures: [`outputs/`](outputs/).

## Ops lock

- Power mode: `MAXN SUPER` at module TDP 25 W, then `jetson_clocks`.
  HIL procedure records the same mode when Nano traces exist.
- Payload-bus FDIR `power_limit_w` stays 55 W. That is not module TDP.
- Camera is BFS-U3-50S5 over USB3. There is no MIPI/CSI path.
- No DLA on this module. Inference is GPU / CUDA cores + tensor cores.
- Repo Python is 3.14. JetPack 6.2+ system Python is 3.10.
  The flight image does not change `requires-python` for this gap.
- Derived expected 4 ms, timeout 20 ms (mixed knee).
- Factory pair tensors are ~82 MiB of unified LPDDR5 (no discrete VRAM).
- AGX uses the same Cortex-A78AE CPU (8 or 12 cores vs 6). It is not 10x compute; max TDP 60 W exceeds the 55 W payload-bus FDIR cap.

