"""UG-GraphT5 preprocessing utilities.

This module provides:
  - image normalization to 512x512 and [0,1]
  - optional denoising / contrast enhancement hooks
  - placeholder text normalization for report corpora

Replace the placeholders with your pipeline as needed.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

@dataclass
class PreprocessConfig:
    image_size: int = 512
    normalize: bool = True

def preprocess_image(img: np.ndarray, cfg: PreprocessConfig) -> np.ndarray:
    """Preprocess a single grayscale image.

    Args:
        img: HxW or HxWxC numpy array.
        cfg: preprocessing configuration.

    Returns:
        Preprocessed image as float32 in [0,1], shape (image_size, image_size).
    """
    if img.ndim == 3:
        # convert RGB to grayscale simple average (replace with proper conversion if needed)
        img = img.mean(axis=2)

    # Resize placeholder (nearest) to avoid extra deps; replace with cv2/PIL for real runs.
    H, W = img.shape
    s = cfg.image_size
    ys = (np.linspace(0, H - 1, s)).astype(np.int32)
    xs = (np.linspace(0, W - 1, s)).astype(np.int32)
    resized = img[ys][:, xs]

    out = resized.astype(np.float32)
    if cfg.normalize:
        mn, mx = float(out.min()), float(out.max())
        if mx > mn:
            out = (out - mn) / (mx - mn)
        else:
            out = np.zeros_like(out, dtype=np.float32)
    return out

def normalize_report_text(text: str) -> str:
    """Minimal report text normalization (placeholder)."""
    t = text.strip().replace("\r\n", "\n").replace("\t", " ")
    # collapse multiple spaces
    while "  " in t:
        t = t.replace("  ", " ")
    return t.lower()
