# Data

This repository does **not** include any dataset files.

## Public datasets
Please download the following datasets according to their official instructions/licensing:

- **MURA**: musculoskeletal radiographs (upper extremities)
- **IU-Xray**: chest x-rays with reports
- **MIMIC-CXR**: chest x-rays with reports

Create the following local layout:

```text
data/
  MURA/
  IU-Xray/
  MIMIC-CXR/
```

## Private dataset (CT-RATE)
CT-RATE is a restricted internal cohort and cannot be redistributed publicly.
If you have approval to use a comparable CT dataset, adapt the `configs/ct_template.yaml` and the preprocessing
templates in `preprocessing/`.

## Minimal smoke test
The `reproduce.py --mode smoke` command runs with synthetic data to verify installation and code wiring.
