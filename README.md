# UG-GraphT5 — Reproducibility Repository (Peer Review)

This repository accompanies the manuscript:

**A Knowledge-Guided and Uncertainty-Calibrated Multimodal Framework for Fracture Diagnosis and Radiology Report Generation**

It provides code and configuration templates to reproduce experiments on **publicly available datasets**
(e.g., MURA, IU-Xray, MIMIC-CXR). Due to institutional/ethical constraints, the **CT-RATE** cohort is not publicly
distributable; however, we provide preprocessing templates and data-format specifications to run the same pipeline on
an approved CT dataset.

## Repository Structure

```text
data/                 Dataset instructions (no data is shipped)
preprocessing/        Preprocessing utilities (image + text)
models/               Model components (vision, Bayesian head, fusion, prompt builder)
training/             Training entrypoints
evaluation/           Evaluation entrypoints (AUC/F1, ECE, BLEU placeholders)
configs/              YAML configs for experiments
checkpoints/          Placeholders for pretrained weights (public-dataset only)
```

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2) Configure datasets
See `data/README.md` for dataset links and expected folder layout.

### 3) Run a minimal smoke test (no real data required)
```bash
python reproduce.py --mode smoke
```

### 4) Train (public datasets)
```bash
python training/train.py --config configs/config.yaml
```

### 5) Evaluate
```bash
python evaluation/evaluate.py --config configs/config.yaml
```

---
**Contact:** riadh.bouslimi@esen.tn
