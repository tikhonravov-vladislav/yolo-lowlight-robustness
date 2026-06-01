# Robustness of YOLO Object Detection Under Low-Light Conditions

Failure-mode analysis of YOLOv8s under controlled low-light degradation, and twotargeted mitigations: **ICCC** (decision-level confidence calibration) and **CBRT** (representation-level curriculum training).

> Research project, Faculty of Computer Science, HSE University.
> Full report: [`coursework.pdf`](coursework.pdf).

## Overview

Object detectors degrade sharply under reduced illumination. This project first **diagnoses** the failure, then **fixes** it:

1. **Analysis.** YOLOv8s is evaluated on COCO val2017 under a controlled syntheticlow-light transform `I_dark = α·I^γ + ε` at three severity levels (Mild, Medium, Severe). Three dominant failure modes emerge: confidence collapse (recall drops
~2× faster than precision), localization instability (mAP50–95 drops more than mAP50), and non-linear sensitivity to severity.
2. **ICCC** — Illumination-Conditioned Confidence Calibration. A post-processing step that lowers the confidence threshold as a function of estimated illumination severity: `τ(γ) = τ₀ − k·(γ−1)`. No retraining, no architecture change.
3. **CBRT** — Curriculum-Based Robustness Training. Staged fine-tuning that exposes the detector to progressively harder degradation (Normal→Mild→Medium→Severe), retaining earlier conditions to avoid forgetting.

## Key results

On **Severe** degradation (COCO val2017), relative to the baseline:

| Method | Recall | mAP50 | mAP50–95 | F1 |
|---|---|---|---|---|
| CBRT | +23.7% | +24.4% | +24.1% | +17.6% |
| ICCC (est. γ) | +17% | — | — | +3.9% |

- ICCC gives consistent recall gains scaling with severity, with negligible cost on clean images.
- Combining the two yields **no** further benefit: the calibrated coefficient for the CBRT model is `k* = 0`, since CBRT already resolves the confidence collapse that ICCC compensates for. The two methods are alternatives, not complements.
- Curriculum ordering does **not** beat uniform augmentation - the driver is the presence of degraded data in training, not the schedule.
- Enhancement-based preprocessing (Zero-DCE, SCI, CLAHE) consistently harms detection under low light within this setup.

## Repository structure

```
.
├── configs/                       # Ultralytics data YAMLs (see note below)
├── docs/
│   └── commands.md                # all training/eval commands + ICCC run order
├── report/
│   └── coursework.pdf
├── results/
│   ├── iccc/                       # gamma estimator + grid-search + ICCC eval tables
│   └── enhancement/               # enhancement comparison table
└── src/
    ├── data_prep/                 # COCO→YOLO conversion, CBRT file lists & configs
    ├── degradation/               # synthetic low-light generation (train / val)
    ├── iccc/                       # split, gamma fit, grid search, ICCC evaluation
    ├── figures/                    # plotting scripts for the report figures
    └── comparison-with-literature/
        ├── enhancement/           # Zero-DCE / SCI / CLAHE preprocessing comparison
        └── MAET/                  # MAET comparison (Colab notebook + notes)
```

**Not included in the repo:** the COCO dataset, its degraded copies, model weight (`*.pt`), and Ultralytics `runs/` outputs. These are large and/or reproducible - see below.

## Setup

```bash
git clone https://github.com/tikhonravov-vladislav/yolo-lowlight-robustness.git
cd yolo-lowlight-robustness
python -m venv .venv && .venv\Scripts\activate     # Windows
pip install ultralytics opencv-python numpy matplotlib pyyaml tqdm
```

## Data preparation

1. Download COCO 2017 (train + val) into `data/coco_raw/`.
2. Convert annotations to YOLO format:
   ```bash
   python src/data_prep/coco2yolo.py
   ```
3. Generate the degraded train and val sets (parameters: report Table 3.1):
   ```bash
   python src/degradation/make_lowlight_train.py
   python src/degradation/make_lowlight_val.py
   ```

Note: throughout the code, the **severe** condition is stored as `*_lowlight (e.g. `val2017_lowlight`, `coco_yolo_lowlight.yaml`).

## Running
All training and evaluation commands (baseline, CBRT, naive augmentation) are documented in [`docs/commands.md`](docs/commands.md). The ICCC pipeline scripts in `src/iccc/` are order-dependent; the required sequence is also in `docs/commands.md`.

Calibrated hyperparameters (from the grid search): `k* = 0.10` for the baseline model, `k* = 0.00` for the CBRT model.

## Comparison with literature

**Enhancement preprocessing** (`src/comparison-with-literature/enhancement/`).
`enhance_images.py` expects the official method repositories cloned under `enhancement_comparison/`, with their pretrained weights, so that the paths inside the script resolve:
- Zero-DCE — https://github.com/Li-Chongyi/Zero-DCE
- SCI — https://github.com/vis-opt-group/SCI

CLAHE needs no external code (OpenCV only). After enhancing, run `evaluate_enhanced.py`.

**MAET** (`src/comparison-with-literature/MAET/`). MAET requires an old software stack and was run in Google Colab; the notebook and a short README documenting the procedure  are in that folder.

## A note on configs and paths

The YAMLs in `configs/` use relative paths and were run from the repository root during the project (e.g. `data=coco_yolo.yaml`). They are kept here as a record of the experimental setup; the per-split ICCC configs (`val_calib_*`, `val_holdout_*`)
are not committed - they are regenerated by the scripts in `src/iccc/`.

## License

MIT — see [`LICENSE`](LICENSE).
