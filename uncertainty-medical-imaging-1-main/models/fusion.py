"""Uncertainty-Guided Multimodal Fusion (UGMF) skeleton."""

from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class FusionConfig:
    dim: int = 512
    beta: float = 1.5  # temperature/sensitivity

class UGMF(nn.Module):
    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg

    def forward(self, z_img: torch.Tensor, z_kg: torch.Tensor, entropy: torch.Tensor) -> torch.Tensor:
        """Fuse embeddings based on normalized entropy.

        entropy: (B,) entropy in nats. We normalize by log(C) outside if needed.
        """
        # Normalize entropy to [0,1] using log(C) estimated from z_img last dim if provided separately.
        # Here we assume binary by default; callers can pass normalized entropy too.
        H = entropy.clamp(min=0.0)
        alpha = torch.exp(-self.cfg.beta * H).unsqueeze(-1)  # (B,1)
        return alpha * z_img + (1.0 - alpha) * z_kg
