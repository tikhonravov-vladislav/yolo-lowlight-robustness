# Commands

All commands are run from the repository root with the virtual environment active.
Final weights (`weights/*.pt`) are not tracked in git; either retrain or obtain
them from Releases.

## Baseline

### Training
Reconstructed from the configuration described in the report (Section 3.1);
the exact command string was not preserved. Fine-tuning: full COCO 2017 train
split, 25 epochs, batch 16, 640x640, AdamW, seed 42, Ultralytics defaults otherwise.

```bash
yolo train model=yolov8s.pt data=coco_yolo.yaml epochs=25 batch=16 imgsz=640 optimizer=AdamW seed=42 device=0 name=baseline_v8s
```

### Evaluation
`iou=0.7` is the Ultralytics default for `val`; it is stated explicitly here for
reproducibility across versions.

```bash
yolo val model=weights/baseline.pt data=coco_yolo.yaml         imgsz=640 batch=16 iou=0.7 device=0 name=baseline_full_normal_eval
yolo val model=weights/baseline.pt data=coco_yolo_mild.yaml    imgsz=640 batch=16 iou=0.7 device=0 name=baseline_full_mild_eval
yolo val model=weights/baseline.pt data=coco_yolo_medium.yaml  imgsz=640 batch=16 iou=0.7 device=0 name=baseline_full_medium_eval
yolo val model=weights/baseline.pt data=coco_yolo_lowlight.yaml imgsz=640 batch=16 iou=0.7 device=0 name=baseline_full_severe_eval
```
