# flight.payload.inference.runtime

**Source:** `packages/flight/src/flight/payload/inference/runtime.py`
**Kind:** module

## Purpose

The runtime module defines the lazy inference-session boundary. A
`RuntimeSession` pairs a `DetectorBackend` with a stable identity and a bounded
warm-up; a `RuntimeFactory` builds a session on demand without doing file or
SDK work at construction. `InferenceRuntime` is the lock-protected holder the
payload app owns: it starts empty on the real path and accepts a session only
through `install_verified`, which the control owner calls after INIT
verification.

## Public interface

| Name | Kind | Description |
| --- | --- | --- |
| `RuntimeSession` | protocol | `identity`, `backend`, `warm_up(cancel)` for one usable session |
| `RuntimeFactory` | protocol | `load(cancel) -> Result[RuntimeSession, FaultCode]` |
| `OnnxRuntimeSession` | dataclass | ONNX session; warm-up runs one synthetic tile through both models |
| `OnnxRuntimeFactory` | class | Config-driven factory; `load` hashes files and builds `OnnxDetector` |
| `ScriptedRuntimeSession` | dataclass | Scripted session; no-op warm-up that consumes no scripted frames |
| `ScriptedRuntimeFactory` | class | Composition-only factory over an injected backend |
| `InferenceRuntime` | class | Lock-protected holder: `snapshot`, `install_verified`, `identity` |

## Inputs and outputs

`OnnxRuntimeFactory.__init__` stores `PactConfig` and does no I/O. `load`
resolves the quantized artifact paths, computes both file digests (unreadable
artifacts return `Err(MODEL_CORRUPT)`), constructs the `OnnxDetector` with
those observed digests as the pinned expectations, and returns a session whose
identity is `onnx:<classifier_sha256>:<segmentor_sha256>` - the ordered digest
pair, not deployment metadata. Broad SDK-boundary exceptions map to
`Err(MODEL_CORRUPT)`; cancellation checkpoints map to `Err(INFERENCE_TIMEOUT)`.
All detector options come from config: confidence and blob area from
`controller.vision`, logit threshold and geometry from `inference`, and the
latency budget from `fault.inference_timeout_ms`.

`InferenceRuntime.from_scripted(backend, identity)` is the explicit sim/test
seam: it installs a scripted session through `install_verified` and keeps a
`ScriptedRuntimeFactory` so the INIT `MODEL_LOAD` effect still exercises a
load path. The real driver selection never calls it.

## Behavior

1. `install_verified` rejects an empty identity with `Err(MODEL_CORRUPT)`.
2. `OnnxRuntimeSession.warm_up` sends a zeroed one-tile float32 image
   `(1, bands, tile_h, tile_w)` and a reference-encoded all-zero `(1, 2)` GSD
   row through `Detector.warm_up`, so warm-up never gates on a production
   frame.
3. The observed digest pair is a fingerprint of the files read at load time,
   not trusted manifest authenticity; it identifies exactly which artifacts
   were loaded and supports same-load consistency checks.

## Errors and faults

`RuntimeFactory.load` returns `Err(MODEL_CORRUPT)` for unreadable artifacts,
failed SDK loads, or contract mismatches, and `Err(INFERENCE_TIMEOUT)` on
cancellation checkpoints. `install_verified` returns `Err(MODEL_CORRUPT)` for
an empty session identity.

## Messages

None.

## Configuration

`OnnxRuntimeFactory` reads `inference` (paths, quantization, geometry,
thresholds), `controller.vision` (confidence gate, blob area), and
`fault.inference_timeout_ms` from `PactConfig`. No new configuration is added.

## Constraints

`onnxruntime` still imports lazily inside session construction; importing this
module does not require the SDK. There is no automatic load, no detect
fallback, and no raw-backend union in `PayloadApp`. The digest identity is a
fingerprint only - it is not an authenticity guarantee.

## Related documents

- [`flight.payload.inference.detector`](detector.md)
- [`flight.payload.inference.verify`](verify.md)
- [`flight.payload.lifecycle`](../lifecycle.md)
- [`flight.core.select_drivers`](../../core/select_drivers.md)
