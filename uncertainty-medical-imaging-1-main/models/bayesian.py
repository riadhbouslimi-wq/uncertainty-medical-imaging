"""Bayesian head with MC-Dropout and predictive entropy."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class BayesianConfig:
    in_dim: int = 1024
    num_classes: int = 2
    dropout: float = 0.2
    mc_samples: int = 20

class BayesianHead(nn.Module):
    def __init__(self, cfg: BayesianConfig):
        super().__init__()
        self.cfg = cfg
        self.fc1 = nn.Linear(cfg.in_dim, 512)
        self.drop = nn.Dropout(p=cfg.dropout)
        self.fc2 = nn.Linear(512, cfg.num_classes)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

    @torch.no_grad()
    def mc_predict(self, x: torch.Tensor, mc_samples: int | None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return mean probabilities and predictive entropy via MC-Dropout.

        Args:
            x: (B, D)
            mc_samples: number of stochastic forward passes

        Returns:
            probs: (B, C) mean predictive distribution
            entropy: (B,) predictive entropy
        """
        T = mc_samples or self.cfg.mc_samples
        self.train()  # keep dropout active
        probs_T = []
        for _ in range(T):
            logits = self.forward_logits(x)
            probs_T.append(F.softmax(logits, dim=-1))
        probs = torch.stack(probs_T, dim=0).mean(dim=0)  # (B, C)
        entropy = -(probs * (probs.clamp_min(1e-8)).log()).sum(dim=-1)  # (B,)
        self.eval()
        return probs, entropy
