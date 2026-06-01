"""
Evaluates the baseline YOLOv8s detector on enhanced validation images.
Compares: Baseline, Baseline+Zero-DCE, Baseline+SCI, Baseline+CLAHE
across all illumination conditions.

Usage:
  python evaluate_enhanced.py
"""

import os
import csv
import yaml
import shutil
from ultralytics import YOLO

PROJECT_ROOT = os.path.abspath(".")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "coco_yolo")
WEIGHTS = os.path.join(PROJECT_ROOT, "weights", "baseline.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "enhancement_comparison")

VAL_FOLDERS = {
    "normal": "val2017",
    "mild":   "val2017_mild",
    "medium": "val2017_medium",
    "severe": "val2017_lowlight",
}

METHODS = {
    "baseline":  None,
    "zero_dce":  "zerodce",
    "sci":       "sci",
    "clahe":     "clahe",
}

CONDITIONS = ["normal", "mild", "medium", "severe"]


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def create_yaml_for_enhanced(base_yaml_path, enhanced_val_folder, output_yaml_path):
    with open(base_yaml_path, "r") as f:
        config = yaml.safe_load(f)
    config["val"] = f"images/{enhanced_val_folder}"
    with open(output_yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return output_yaml_path


def main():
    print("=" * 75)
    print(" Enhancement Comparison Evaluation")
    print("=" * 75)

    model = YOLO(WEIGHTS)
    print(f"Loaded model from {WEIGHTS}\n")

    base_yaml = os.path.join(PROJECT_ROOT, "coco_yolo.yaml")
    temp_yaml_dir = os.path.join(OUTPUT_DIR, "temp_yamls")
    os.makedirs(temp_yaml_dir, exist_ok=True)

    results = []

    for method_name, suffix in METHODS.items():
        for cond in CONDITIONS:
            base_val_folder = VAL_FOLDERS[cond]

            if suffix is None:
                if cond == "normal":
                    yaml_to_use = base_yaml
                elif cond == "mild":
                    yaml_to_use = os.path.join(PROJECT_ROOT, "coco_yolo_mild.yaml")
                elif cond == "medium":
                    yaml_to_use = os.path.join(PROJECT_ROOT, "coco_yolo_medium.yaml")
                elif cond == "severe":
                    yaml_to_use = os.path.join(PROJECT_ROOT, "coco_yolo_lowlight.yaml")
            else:
                val_folder = f"{base_val_folder}_{suffix}"
                enhanced_path = os.path.join(DATA_ROOT, "images", val_folder)

                if not os.path.exists(enhanced_path):
                    print(f"  SKIP {method_name}/{cond}: folder {val_folder} not found")
                    continue

                yaml_to_use = create_yaml_for_enhanced(
                    base_yaml, val_folder,
                    os.path.join(temp_yaml_dir, f"{method_name}_{cond}.yaml")
                )

            print(f"Evaluating [{method_name}] on [{cond}]...")

            try:
                m = model.val(
                    data=yaml_to_use,
                    imgsz=640,
                    batch=16,
                    device=0,
                    workers=4,
                    verbose=False,
                    save_json=False,
                    plots=False,
                    name=f"enhance_{method_name}_{cond}",
                    project=os.path.join(OUTPUT_DIR, "runs_eval"),
                    exist_ok=True,
                )

                row = {
                    "method": method_name,
                    "condition": cond,
                    "precision": float(m.box.mp),
                    "recall": float(m.box.mr),
                    "mAP50": float(m.box.map50),
                    "mAP50_95": float(m.box.map),
                    "F1": f1_score(float(m.box.mp), float(m.box.mr)),
                }
                results.append(row)

                print(f"    P={row['precision']:.3f}  R={row['recall']:.3f}  "
                      f"mAP50={row['mAP50']:.3f}  mAP50-95={row['mAP50_95']:.3f}  "
                      f"F1={row['F1']:.3f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    #Save results to CSV
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_results.csv")
    fieldnames = ["method", "condition", "precision", "recall",
                  "mAP50", "mAP50_95", "F1"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    print(f"\nResults saved to {csv_path}")

    #Print summary table
    print("\n" + "=" * 85)
    print(f"{'Method':>15} | {'Condition':>8} | {'P':>6} {'R':>6} "
          f"{'mAP50':>7} {'mAP50-95':>9} {'F1':>6}")
    print("-" * 85)
    for row in results:
        print(f"{row['method']:>15} | {row['condition']:>8} | "
              f"{row['precision']:>6.3f} {row['recall']:>6.3f} "
              f"{row['mAP50']:>7.3f} {row['mAP50_95']:>9.3f} "
              f"{row['F1']:>6.3f}")

    #Print relative changes vs baseline
    print("\n" + "=" * 85)
    print("RELATIVE CHANGE vs BASELINE")
    print("=" * 85)
    print(f"{'Method':>15} | {'Condition':>8} | {'dP':>7} {'dR':>7} "
          f"{'dmAP50':>8} {'dmAP50-95':>10} {'dF1':>7}")
    print("-" * 85)

    for row in results:
        if row["method"] == "baseline":
            continue
        base = next(
            (r for r in results
             if r["method"] == "baseline" and r["condition"] == row["condition"]),
            None
        )
        if base is None:
            continue

        dp = row["precision"] - base["precision"]
        dr = row["recall"] - base["recall"]
        dm50 = row["mAP50"] - base["mAP50"]
        dm95 = row["mAP50_95"] - base["mAP50_95"]
        df1 = row["F1"] - base["F1"]

        print(f"{row['method']:>15} | {row['condition']:>8} | "
              f"{dp:>+7.3f} {dr:>+7.3f} {dm50:>+8.3f} "
              f"{dm95:>+10.3f} {df1:>+7.3f}")

    shutil.rmtree(temp_yaml_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
