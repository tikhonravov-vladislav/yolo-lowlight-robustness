import os
import json
import csv
import numpy as np
import cv2
from ultralytics import YOLO


PROJECT_ROOT = os.path.abspath(".")
ICCC_ROOT = os.path.join(PROJECT_ROOT, "iccc")
DATA_ROOT = os.path.abspath("data/coco_yolo")
WEIGHTS = os.path.join(PROJECT_ROOT, "weights", "baseline.pt")

TAU_0 = 0.25
K_STAR = 0.10                                # from grid_search_k.py
GAMMA_ESTIMATOR_PATH = os.path.join(ICCC_ROOT, "gamma_estimator.json")

conditions = {
    "normal": ("val_holdout_normal.yaml", "images/val2017",          1.0),
    "mild":   ("val_holdout_mild.yaml",   "images/val2017_mild",     1.4),
    "medium": ("val_holdout_medium.yaml", "images/val2017_medium",   1.8),
    "severe": ("val_holdout_severe.yaml", "images/val2017_lowlight", 2.2),
}


def mean_intensity(image_path):
    """Compute mean grayscale intensity, normalized to [0, 1]."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(img.mean()) / 255.0


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    # === Load gamma estimator ===
    print(f"Loading gamma estimator from {GAMMA_ESTIMATOR_PATH}")
    with open(GAMMA_ESTIMATOR_PATH, "r") as f:
        estimator = json.load(f)
    coeffs = np.array(estimator["coefficients"])
    print(f"  Polynomial degree: {estimator['polynomial_degree']}")
    print(f"  Coefficients: {coeffs}")
    print(f"  Calibration MAE: {estimator['fit_quality']['MAE']:.4f}\n")

    # === Load held-out filenames ===
    holdout_txt = os.path.join(ICCC_ROOT, "val_holdout_filenames.txt")
    with open(holdout_txt, "r") as f:
        holdout_filenames = [line.strip() for line in f if line.strip()]
    print(f"Held-out set: {len(holdout_filenames)} images\n")

    # === Estimate average gamma per condition (on held-out data) ===
    print("Estimating average gamma per condition on held-out data...")
    avg_predicted_gamma = {}
    for cond_name, (_, folder, true_gamma) in conditions.items():
        folder_abs = os.path.join(DATA_ROOT, folder)
        intensities = []
        for filename in holdout_filenames:
            img_path = os.path.join(folder_abs, filename)
            mi = mean_intensity(img_path)
            if mi is not None:
                intensities.append(mi)

        intensities = np.array(intensities)
        predicted_gammas = np.polyval(coeffs, intensities)
        avg_gamma = float(np.mean(predicted_gammas))
        avg_predicted_gamma[cond_name] = avg_gamma

        print(f"  [{cond_name}] true γ={true_gamma}, "
              f"predicted γ_hat (avg) = {avg_gamma:.4f}")

    print(f"\nLoading model from {WEIGHTS}")
    model = YOLO(WEIGHTS)

    results = []

    for cond_name, (yaml_name, _, true_gamma) in conditions.items():
        yaml_path = os.path.join(ICCC_ROOT, yaml_name)
        gamma_hat = avg_predicted_gamma[cond_name]

        # Adaptive threshold for ICCC
        tau_iccc = max(0.01, min(1.0, TAU_0 - K_STAR * (gamma_hat - 1)))

        print(f"\n=== Condition: {cond_name} ===")
        print(f"  true γ = {true_gamma}, predicted γ_hat = {gamma_hat:.4f}")
        print(f"  baseline τ = {TAU_0}, ICCC τ = {tau_iccc:.4f}")

        # --- Baseline run (τ = τ₀, no ICCC) ---
        print(f"  Running baseline...")
        m_base = model.val(
            data=yaml_path,
            conf=TAU_0,
            iou=0.7,
            workers=4,
            verbose=False,
            save_json=False,
            plots=False,
            name=f"holdout_baseline_{cond_name}",
            project=os.path.join(ICCC_ROOT, "runs_eval"),
            exist_ok=True,
        )
        base_metrics = {
            "method": "baseline",
            "condition": cond_name,
            "true_gamma": true_gamma,
            "tau": TAU_0,
            "precision": float(m_base.box.mp),
            "recall": float(m_base.box.mr),
            "mAP50": float(m_base.box.map50),
            "mAP50_95": float(m_base.box.map),
        }
        base_metrics["F1"] = f1_score(base_metrics["precision"], base_metrics["recall"])
        print(f"    [baseline] P={base_metrics['precision']:.3f}  "
              f"R={base_metrics['recall']:.3f}  "
              f"mAP50={base_metrics['mAP50']:.3f}  "
              f"mAP50-95={base_metrics['mAP50_95']:.3f}  "
              f"F1={base_metrics['F1']:.3f}")
        results.append(base_metrics)

        # --- ICCC run (τ = adaptive) ---
        print(f"  Running ICCC...")
        m_iccc = model.val(
            data=yaml_path,
            conf=tau_iccc,
            iou=0.7,
            workers=4,
            verbose=False,
            save_json=False,
            plots=False,
            name=f"holdout_iccc_{cond_name}",
            project=os.path.join(ICCC_ROOT, "runs_eval"),
            exist_ok=True,
        )
        iccc_metrics = {
            "method": "ICCC",
            "condition": cond_name,
            "true_gamma": true_gamma,
            "predicted_gamma": gamma_hat,
            "tau": tau_iccc,
            "precision": float(m_iccc.box.mp),
            "recall": float(m_iccc.box.mr),
            "mAP50": float(m_iccc.box.map50),
            "mAP50_95": float(m_iccc.box.map),
        }
        iccc_metrics["F1"] = f1_score(iccc_metrics["precision"], iccc_metrics["recall"])
        print(f"    [ICCC]     P={iccc_metrics['precision']:.3f}  "
              f"R={iccc_metrics['recall']:.3f}  "
              f"mAP50={iccc_metrics['mAP50']:.3f}  "
              f"mAP50-95={iccc_metrics['mAP50_95']:.3f}  "
              f"F1={iccc_metrics['F1']:.3f}")
        results.append(iccc_metrics)
        # ICCC Oracle run (τ computed from true γ) 
        tau_oracle = max(0.01, min(1.0, TAU_0 - K_STAR * (true_gamma - 1)))
        print(f"  Running ICCC (oracle)... τ = {tau_oracle:.4f}")
        m_oracle = model.val(
            data=yaml_path,
            conf=tau_oracle,
            iou=0.7,
            workers=4,
            verbose=False,
            save_json=False,
            plots=False,
            name=f"holdout_iccc_oracle_{cond_name}",
            project=os.path.join(ICCC_ROOT, "runs_eval"),
            exist_ok=True,
        )       
        oracle_metrics = {
            "method": "ICCC_oracle",
            "condition": cond_name,
            "true_gamma": true_gamma,
            "predicted_gamma": true_gamma,
            "tau": tau_oracle,
            "precision": float(m_oracle.box.mp),
            "recall": float(m_oracle.box.mr),
            "mAP50": float(m_oracle.box.map50),
            "mAP50_95": float(m_oracle.box.map),
        }
        oracle_metrics["F1"] = f1_score(oracle_metrics["precision"], oracle_metrics["recall"])
        results.append(oracle_metrics)


    csv_path = os.path.join(ICCC_ROOT, "iccc_evaluation_results.csv")
    fieldnames = ["method", "condition", "true_gamma", "predicted_gamma",
                  "tau", "precision", "recall", "mAP50", "mAP50_95", "F1"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Fill missing predicted_gamma for baseline rows
            if "predicted_gamma" not in row:
                row["predicted_gamma"] = ""
            writer.writerow(row)
    print(f"\nFull results saved to {csv_path}")

    #Summary table
    print("\n" + "=" * 75)
    print("FINAL EVALUATION SUMMARY (held-out 4000 images)")
    print("=" * 75)
    print(f"{'condition':>10} | {'method':>10} | {'P':>6} {'R':>6} {'mAP50':>7} {'mAP50-95':>9} {'F1':>6}")
    print("-" * 75)
    for cond_name in conditions:
        for method in ["baseline", "ICCC"]:
            row = next(r for r in results if r["condition"] == cond_name and r["method"] == method)
            print(f"{cond_name:>10} | {method:>10} | "
                  f"{row['precision']:>6.3f} {row['recall']:>6.3f} "
                  f"{row['mAP50']:>7.3f} {row['mAP50_95']:>9.3f} "
                  f"{row['F1']:>6.3f}")

    #Per-condition improvement table
    print("\n" + "=" * 75)
    print("IMPROVEMENT (ICCC vs Baseline)")
    print("=" * 75)
    print(f"{'condition':>10} | {'ΔP':>7} {'ΔR':>7} {'ΔmAP50':>9} {'ΔmAP50-95':>11} {'ΔF1':>7}")
    print("-" * 75)
    for cond_name in conditions:
        base = next(r for r in results if r["condition"] == cond_name and r["method"] == "baseline")
        iccc = next(r for r in results if r["condition"] == cond_name and r["method"] == "ICCC")
        dP  = iccc["precision"] - base["precision"]
        dR  = iccc["recall"]    - base["recall"]
        dM5 = iccc["mAP50"]     - base["mAP50"]
        dM9 = iccc["mAP50_95"]  - base["mAP50_95"]
        dF1 = iccc["F1"]        - base["F1"]
        print(f"{cond_name:>10} | "
              f"{dP:>+7.4f} {dR:>+7.4f} {dM5:>+9.4f} {dM9:>+11.4f} {dF1:>+7.4f}")


if __name__ == "__main__":
    main()
