"""Demo: run one synthetic frame through the PACT segmentation model.

Loads a model from --model-path (or creates a randomly initialized model if the file
does not exist), runs inference on one synthetic 4-band (1,4,256,256) float32 frame,
prints blob detections and wall-clock latency in milliseconds.

Usage
-----
    python scripts/demo_inference.py [--model-path data/models/best.pt]

Satisfies: §7 of PACT_SW_ARCH.md (scripts/demo_inference.py)
"""

from __future__ import annotations

# stdlib
import argparse
import time
from pathlib import Path

# third-party
import numpy as np
import torch

# internal
from pact.model.architecture import build_model


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run one synthetic frame through the PACT segmentation model."
    )
    parser.add_argument(
        "--model-path",
        default="data/models/best.pt",
        help="Path to a saved model checkpoint. If not found, uses random init (default: data/models/best.pt).",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point: build model, run inference, print results."""
    args = parse_args()
    model_path = Path(args.model_path)

    # Load model or fall back to random init
    if model_path.exists():
        print(f"Loading model from: {model_path}")
        model = build_model(encoder_weights=None)
        state_dict = torch.load(str(model_path), map_location="cpu")
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")
    else:
        print(f"Model file not found at '{model_path}'. Using randomly initialized model.")
        model = build_model(encoder_weights=None)

    model.eval()
    device = torch.device("cpu")
    model = model.to(device)

    # Synthetic input: one random 4-band frame
    rng = np.random.default_rng(seed=42)
    raw = rng.random((1, 4, 256, 256), dtype=np.float32)
    tensor = torch.from_numpy(raw).to(device)

    print(f"\nRunning inference on synthetic (1,4,256,256) float32 frame...")
    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)  # (1, 1, 256, 256)
        mask = torch.sigmoid(logits)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    mask_np = mask.squeeze().cpu().numpy()  # (256, 256)
    threshold = 0.5
    binary_mask = (mask_np > threshold).astype(np.uint8)
    n_positive_pixels = int(binary_mask.sum())

    print(f"Inference complete.")
    print(f"  Latency:          {elapsed_ms:.1f} ms")
    print(f"  Mask shape:       {mask_np.shape}")
    print(f"  Max probability:  {float(mask_np.max()):.4f}")
    print(f"  Mean probability: {float(mask_np.mean()):.4f}")
    print(f"  Positive pixels (>{threshold}): {n_positive_pixels} / {256*256}")
    print(f"\nBlob detections (random model — expect near-zero output):")
    if n_positive_pixels == 0:
        print("  No blobs detected above threshold (expected for random model).")
    else:
        print(f"  {n_positive_pixels} pixels above threshold {threshold}.")
        print("  (Run with a trained model for meaningful blob detections.)")


if __name__ == "__main__":
    main()
