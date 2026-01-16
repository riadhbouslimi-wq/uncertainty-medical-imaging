"""Minimal reproduction helper.

Modes:
  - smoke: runs import + forward pass + ECE calculation on synthetic data.
  - (extend) train_public / eval_public once datasets are configured.

Usage:
  python reproduce.py --mode smoke
"""

from __future__ import annotations
import argparse
import numpy as np
import torch

from evaluation.evaluate import expected_calibration_error
from models.model import UGGraphT5, ModelConfig

def smoke():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = ModelConfig()
    model = UGGraphT5(cfg).to(device).eval()

    B = 8
    x = torch.rand(B, 1, 512, 512, device=device)
    y = torch.randint(0, cfg.num_classes, (B,), device=device).cpu().numpy()

    with torch.no_grad():
        probs, entropy, _ = model(x)
    probs_np = probs.cpu().numpy()
    ece = expected_calibration_error(probs_np, y, n_bins=15)
    acc = float((probs_np.argmax(axis=1) == y).mean())

    print("SMOKE TEST OK")
    print(f"ACC={acc:.3f}  ECE={ece:.4f}  mean_entropy={float(entropy.mean().cpu()):.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke"], default="smoke")
    args = ap.parse_args()
    if args.mode == "smoke":
        smoke()

if __name__ == "__main__":
    main()
