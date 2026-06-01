# MAET comparison

Direct evaluation of the pretrained MAET detector against its YOLOv3 baseline on our four illumination conditions, used for Table 6.5 in the report.

Unlike the rest of the project (YOLOv8s / Ultralytics, run locally), MAET requires an old software stack (Python 3.7, torch 1.6, mmcv-full 1.1.5, MMDetection-era
pycocotools). It was therefore run in Google Colab. The notebook `MAET.ipynb`contains the full run, including the environment setup and several compatibility patches needed to make the original MAET repository run on a modern Colab runtime.

## What the notebook does
- Sets up the conda environment and clones the original MAET repo (`cuiziteng/ICCV_MAET`), with patches for pycocotools / COCO API and device id.
- Loads two pretrained checkpoints: the MAET-COCO model and a standard YOLOv3 baseline from MMDetection.
- Runs `tools/test.py --eval bbox` for both models across all four conditions (Normal, Mild, Medium, Severe), producing the mAP numbers in Table 6.5.

## Notes
- MAET is evaluated zero-shot w.r.t. our degradation model: it was trained on a different physics-based ISP degradation, so this measures transfer, not native performance (see report Section 6.4.2).
- Input images are the same four-condition val2017 sets used elsewhere in the project.
- Checkpoints: MAET-COCO weights from the MAET repo; YOLOv3 baseline from MMDetection.
