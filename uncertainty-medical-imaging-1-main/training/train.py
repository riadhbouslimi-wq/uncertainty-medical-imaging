"""Training entrypoint (skeleton).

This script is intentionally minimal:
  - loads YAML config
  - runs a short synthetic training loop as a template

Replace the dataset + loss with your full implementation.
"""

from __future__ import annotations
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

from models.model import UGGraphT5, ModelConfig

def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    mcfg = ModelConfig(
        image_embed_dim=cfg["model"]["image_embed_dim"],
        kg_embed_dim=cfg["model"]["kg_embed_dim"],
        fused_dim=cfg["model"]["fused_dim"],
        num_classes=cfg["model"]["num_classes"],
        dropout=cfg["model"]["dropout"],
        mc_samples=cfg["model"]["mc_samples"],
        beta=cfg["model"]["beta"],
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UGGraphT5(mcfg).to(device)

    # Synthetic data (smoke template)
    B = cfg["training"]["batch_size"]
    x = torch.rand(B, 1, cfg["model"]["image_size"], cfg["model"]["image_size"], device=device)
    y = torch.randint(0, mcfg.num_classes, (B,), device=device)

    # Simple loss on mean probs (not Bayesian training; template only)
    criterion = nn.NLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"], weight_decay=cfg["training"]["weight_decay"])

    model.train()
    for step in range(cfg["training"]["smoke_steps"]):
        optimizer.zero_grad()
        probs, entropy, _ = model(x)
        loss = criterion((probs.clamp_min(1e-8)).log(), y)
        loss.backward()
        optimizer.step()
        if (step + 1) % 1 == 0:
            print(f"step={step+1} loss={loss.item():.4f} mean_entropy={entropy.mean().item():.4f}")

    print("Done. Replace synthetic loop with real dataloaders for public datasets.")

if __name__ == "__main__":
    main()
