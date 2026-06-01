# Commands

All commands are run from the repository root with the virtual environment active.
Final weights (`weights/*.pt`, `runs/.../best.pt`) are not tracked in git; either
retrain or obtain them from Releases.

Data-generation scripts (`src/degradation/`, `src/data_prep/`) are plain single-run
scripts - run them directly (e.g. `python src/degradation/make_lowlight_train.py`)
before the steps below. Only commands that are non-obvious are documented here:
Ultralytics CLI calls (which exist nowhere as files) and the ICCC pipeline (whose
run order is mandatory).

## Baseline

### Training
Reconstructed from the configuration described in the report (Section 3.1); the exact
command string was not preserved. Full COCO 2017 train split, 25 epochs, batch 16,
640x640, AdamW, seed 42, Ultralytics defaults otherwise.

```bash
yolo train model=yolov8s.pt data=coco_yolo.yaml epochs=25 batch=16 imgsz=640 optimizer=AdamW seed=42 device=0 name=baseline_v8s
```

### Evaluation (4 conditions)
`iou=0.7` is the Ultralytics default for `val`; stated explicitly for reproducibility.

```bash
yolo val model=weights/baseline.pt data=coco_yolo.yaml          imgsz=640 batch=16 iou=0.7 device=0 name=baseline_full_normal_eval
yolo val model=weights/baseline.pt data=coco_yolo_mild.yaml     imgsz=640 batch=16 iou=0.7 device=0 name=baseline_full_mild_eval
yolo val model=weights/baseline.pt data=coco_yolo_medium.yaml   imgsz=640 batch=16 iou=0.7 device=0 name=baseline_full_medium_eval
yolo val model=weights/baseline.pt data=coco_yolo_lowlight.yaml imgsz=640 batch=16 iou=0.7 device=0 name=baseline_full_severe_eval
```

## CBRT (curriculum) and Naive augmentation

`workers=4` was added from stage 2 onward to work around a data-loader crash; it
affects only loading parallelism, not results.

### CBRT training (3 stages)
Each stage initializes from the previous stage's `last.pt`. Note: stage 2 loads from
`cbrt_stage12` (not `cbrt_stage1`) — Ultralytics auto-suffixed the run folder when the
name was reused, and that folder holds the actual stage-1 weights used.

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
yolo val model=runs/detect/cbrt_stage3/weights/best.pt data=coco_yolo.yaml          imgsz=640 batch=16 device=0 workers=4 name=cbrt_eval_normal
yolo val model=runs/detect/cbrt_stage3/weights/best.pt data=coco_yolo_mild.yaml     imgsz=640 batch=16 device=0 workers=4 name=cbrt_eval_mild
yolo val model=runs/detect/cbrt_stage3/weights/best.pt data=coco_yolo_medium.yaml   imgsz=640 batch=16 device=0 workers=4 name=cbrt_eval_medium
yolo val model=runs/detect/cbrt_stage3/weights/best.pt data=coco_yolo_lowlight.yaml imgsz=640 batch=16 device=0 workers=4 name=cbrt_eval_severe
```

### Naive evaluation (4 conditions)
```bash
yolo val model=runs/detect/naive_aug_v8s_full/weights/best.pt data=coco_yolo.yaml          imgsz=640 batch=16 device=0 workers=4 name=naive_eval_normal
yolo val model=runs/detect/naive_aug_v8s_full/weights/best.pt data=coco_yolo_mild.yaml     imgsz=640 batch=16 device=0 workers=4 name=naive_eval_mild
yolo val model=runs/detect/naive_aug_v8s_full/weights/best.pt data=coco_yolo_medium.yaml   imgsz=640 batch=16 device=0 workers=4 name=naive_eval_medium
yolo val model=runs/detect/naive_aug_v8s_full/weights/best.pt data=coco_yolo_lowlight.yaml imgsz=640 batch=16 device=0 workers=4 name=naive_eval_severe
```

## ICCC

The scripts in `src/iccc/` are order-dependent - each consumes the previous one's
output. Run from the repository root in this sequence:

```bash
python src/iccc/split_val.py              # 1. split val into calib (1000) + holdout (4000)
python src/iccc/make_calib_yamls.py       # 2. per-condition configs for both splits
python src/iccc/make_holdout_yamls.py
python src/iccc/fit_gamma_estimator.py    # 3. fit gamma estimator -> gamma_estimator.json
python src/iccc/grid_search_k.py          # 4. grid-search k: baseline -> k* = 0.10
python src/iccc/grid_search_k_cbrt.py     #    grid-search k: CBRT     -> k* = 0.00
python src/iccc/evaluate_iccc.py          # 5. final eval on holdout
```
