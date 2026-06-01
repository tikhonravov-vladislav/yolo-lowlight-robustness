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

## CBRT (curriculum) and Naive augmentation

`workers=4` was added from stage 2 onward to work around a data-loader crash;
it does not affect results, only loading parallelism.

### CBRT training (3 stages)
Each stage is initialized from the previous stage's `last.pt`. Note: stage 2 loads
from `cbrt_stage12` (not `cbrt_stage1`) — Ultralytics auto-suffixed the run folder
when the name was reused, and that folder holds the actual stage-1 weights used.

```bash
yolo train model=yolov8s.pt data=coco_yolo_cbrt_stage1.yaml epochs=8 imgsz=640 batch=16 device=0 seed=42 patience=0 name=cbrt_stage1
yolo train model=runs/detect/cbrt_stage12/weights/last.pt data=coco_yolo_cbrt_stage2.yaml epochs=8 imgsz=640 batch=16 device=0 seed=42 patience=0 workers=4 name=cbrt_stage2
yolo train model=runs/detect/cbrt_stage2/weights/last.pt data=coco_yolo_cbrt_stage3.yaml epochs=9 imgsz=640 batch=16 device=0 seed=42 patience=0 workers=4 name=cbrt_stage3
```

### Naive augmentation training (single run, 25 epochs)
```bash
yolo train model=yolov8s.pt data=coco_yolo_naive.yaml epochs=25 imgsz=640 batch=16 device=0 seed=42 patience=10 workers=4 name=naive_aug_v8s_full
```

### CBRT evaluation (4 conditions)
```bash
yolo val model=runs/detect/cbrt_stage3/weights/best.pt data=coco_yolo.yaml         imgsz=640 batch=16 device=0 workers=4 name=cbrt_eval_normal
yolo val model=runs/detect/cbrt_stage3/weights/best.pt data=coco_yolo_mild.yaml    imgsz=640 batch=16 device=0 workers=4 name=cbrt_eval_mild
yolo val model=runs/detect/cbrt_stage3/weights/best.pt data=coco_yolo_medium.yaml  imgsz=640 batch=16 device=0 workers=4 name=cbrt_eval_medium
yolo val model=runs/detect/cbrt_stage3/weights/best.pt data=coco_yolo_lowlight.yaml imgsz=640 batch=16 device=0 workers=4 name=cbrt_eval_severe
```

### Naive evaluation (4 conditions)
```bash
yolo val model=runs/detect/naive_aug_v8s_full/weights/best.pt data=coco_yolo.yaml         imgsz=640 batch=16 device=0 workers=4 name=naive_eval_normal
yolo val model=runs/detect/naive_aug_v8s_full/weights/best.pt data=coco_yolo_mild.yaml    imgsz=640 batch=16 device=0 workers=4 name=naive_eval_mild
yolo val model=runs/detect/naive_aug_v8s_full/weights/best.pt data=coco_yolo_medium.yaml  imgsz=640 batch=16 device=0 workers=4 name=naive_eval_medium
yolo val model=runs/detect/naive_aug_v8s_full/weights/best.pt data=coco_yolo_lowlight.yaml imgsz=640 batch=16 device=0 workers=4 name=naive_eval_severe
```
