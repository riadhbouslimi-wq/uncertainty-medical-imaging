"""Prompt builder for uncertainty-aware ClinicalT5-style generation (template)."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

@dataclass
class PromptConfig:
    high_uncertainty_threshold: float = 0.6  # example threshold
    medium_uncertainty_threshold: float = 0.25

def hedging_phrase(entropy: float, cfg: PromptConfig) -> str:
    if entropy > cfg.high_uncertainty_threshold:
        return "possible"
    if entropy > cfg.medium_uncertainty_threshold:
        return "suggests"
    return "consistent with"

def build_prompt(findings: str, impression: str, recommendations: str, entropy: float, cfg: PromptConfig) -> str:
    hedge = hedging_phrase(entropy, cfg)
    # Minimal structured prompt; adapt to your exact prompt format in the paper.
    prompt = (
        f"Findings: {findings}.\n"
        f"Impression: {hedge} {impression}.\n"
        f"Recommendations: {recommendations}."
    )
    return prompt
