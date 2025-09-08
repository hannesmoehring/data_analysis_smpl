# Intra-Motion Similarity Analysis

This repository contains supplementary code for analyzing **intra-motion similarity** in motion–language datasets (e.g., [HumanML3D](https://github.com/EricGuo5513/HumanML3D)).  
It is intended as an additional codebase to support experiments and is not a standalone package.

---

## Repository Structure

```
.
├── datasets/           # Place datasets here (unpacked manually)
│   └── humanml3d/      # Example dataset (unpacked)
│      ├── animations/         # Rendered motion animations
│      ├── new_joint_vecs/     # Processed joint vectors
│      ├── new_joints/         # Processed joint data
│      `── texts/              # Textual captions for motions
└── textual/      # Example dataset (unpacked)
      ├── textual_analysis.ipynb       # Main Notebook for referenced Analysis
      ├── util.py                      # Some utility
      `── textual_help.py              # Main Textual Checks
```

---

## Datasets

Datasets must be manually unpacked into `./datasets/`. For example:

```
datasets/
└── humanml3d/
    ├── animations/
    ├── new_joint_vecs/
    ├── new_joints/
    └── texts/
```

> ⚠️ The code will not download datasets automatically. Make sure to unpack the required datasets before running the notebook.

---

## Usage

1. Install requirements:

   ```bash
   pip install -r requirements.txt
   ```

2. Unpack the dataset(s) into `./datasets/`.

3. Open the analysis notebook in the project root:

   ```bash
   cd textual
   jupyter notebook textual_analysis.ipynb
   ```

4. Run cells to compute all metrics such as corpus statistics and intra motion similarity across different embedding backends (MiniLM, MPNet, DistilBERT, TF-IDF).

---

## Notes

- This codebase is **supplementary** — it supports experiments with motion–language similarity but does not include training pipelines for motion models.
- Motion data is assumed to follow the HumanML3D format.
