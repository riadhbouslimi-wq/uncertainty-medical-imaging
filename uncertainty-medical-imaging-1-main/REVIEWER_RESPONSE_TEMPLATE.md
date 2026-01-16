## Reviewer #2 – Comment 7 (Code availability)

We thank the reviewer for emphasizing the importance of transparency and reproducibility.

In response, we prepared a structured code repository that enables independent inspection and reproduction
of all experiments conducted on **publicly available datasets** (e.g., MURA, IU-Xray, MIMIC-CXR),
while respecting institutional restrictions on proprietary data.

The repository includes:
- complete training and inference code for public datasets;
- preprocessing scripts and configuration files;
- requirements and environment setup instructions;
- example commands for training, evaluation, and uncertainty estimation (MC-Dropout entropy);
- placeholders for public-dataset checkpoints and guidance for hosting large files (e.g., GitHub Releases/LFS).

Due to institutional and ethical constraints, the proprietary CT-RATE dataset cannot be publicly released.
We provide preprocessing templates, data-format specifications, and configuration stubs that allow the pipeline
to be applied to any comparable CT dataset subject to appropriate approval.

**Private peer-review link:** <PASTE_GITHUB_PRIVATE_LINK_HERE>

Upon acceptance, we will make the repository public under an open-source license.
