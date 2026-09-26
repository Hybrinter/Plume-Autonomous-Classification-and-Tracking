"""ONNX export, acceptance, and the flight pair blob.

Contains:
  - onnx: checkpoint export, quantization, and manifest sidecars.
  - accept: hash, I/O contract, and golden-scene gate.
  - finalize: test eval, export, and acceptance for one run.
  - ort_providers: onnxruntime execution-provider preference.
  - pair: flight-promotable check and the classifier plus segmentor blob.

Import each module by name. This package does not re-export names.
"""
