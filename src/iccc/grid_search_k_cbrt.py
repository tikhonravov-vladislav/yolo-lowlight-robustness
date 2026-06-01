import os
import json
import csv
from ultralytics import YOLO


PROJECT_ROOT = os.path.abspath(".")
ICCC_ROOT = os.path.join(PROJECT_ROOT, "iccc")

WEIGHTS = os.path.join(PROJECT_ROOT, "runs", "detect", "cbrt_stage3", "weights", "best.pt")

# Standard YOLO inference confidence threshold
TAU_0 = 0.25

conditions = {
    "normal": ("val_calib_normal.yaml", 1.0),
    "mild":   ("val_calib_mild.yaml",   1.4),
    "medium": ("val_calib_medium.yaml", 1.8),
    "severe": ("val_calib_severe.yaml", 2.2),
}

# Grid of k values to search over
K_VALUES = [0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15]


def compute_tau(k, gamma):
    """Adaptive threshold τ(γ) = τ₀ - k·(γ - 1), clipped to [0.01, 1.0]."""
    tau = TAU_0 - k * (gamma - 1)
    return max(0.01, min(1.0, tau))


def f1_score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def main():
    print(f"Loading CBRT model from {WEIGHTS}")
    model = YOLO(WEIGHTS)

    results_rows = []

    for k in K_VALUES:
        print(f"\n=== k = {k} ===")
        for cond_name, (yaml_name, gamma) in conditions.items():
            tau = compute_tau(k, gamma)
            yaml_path = os.path.join(ICCC_ROOT, yaml_name)

            print(f"  [{cond_name}] γ={gamma}, τ={tau:.4f}")

            run_name = f"grid_cbrt_k{k:.2f}_{cond_name}".replace(".", "p")
            metrics = model.val(
                data=yaml_path,
                conf=tau,
                iou=0.7,
                workers=4,
                verbose=False,
                save_json=False,
                plots=False,
                name=run_name,
                project=os.path.join(ICCC_ROOT, "runs_grid_cbrt"),
                exist_ok=True,
            )

            p = float(metrics.box.mp)
            r = float(metrics.box.mr)
            map50 = float(metrics.box.map50)
            map5095 = float(metrics.box.map)
            f1 = f1_score(p, r)

            print(f"    P={p:.3f}  R={r:.3f}  mAP50={map50:.3f}  mAP50-95={map5095:.3f}  F1={f1:.3f}")

            results_rows.append({
                "k": k,
                "condition": cond_name,
                "gamma": gamma,
                "tau": tau,
                "precision": p,
                "recall": r,
                "mAP50": map50,
                "mAP50_95": map5095,
                "F1": f1,
            })

    csv_path = os.path.join(ICCC_ROOT, "grid_search_k_cbrt_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results_rows[0].keys())
        writer.writeheader()
        writer.writerows(results_rows)
    print(f"\nFull results saved to {csv_path}")

    print("\n=== Average F1 across conditions per k (CBRT model) ===")
    print(f"{'k':>6} | {'avg F1':>8} | {'normal':>8} {'mild':>8} {'medium':>8} {'severe':>8}")
    print("-" * 60)

    summary = {}
    for k in K_VALUES:
        f1s = {row["condition"]: row["F1"] for row in results_rows if row["k"] == k}
        avg_f1 = sum(f1s.values()) / len(f1s)
        summary[k] = {"avg_F1": avg_f1, **f1s}
        print(f"{k:>6.2f} | {avg_f1:>8.4f} | "
              f"{f1s['normal']:>8.4f} {f1s['mild']:>8.4f} {f1s['medium']:>8.4f} {f1s['severe']:>8.4f}")

    best_k = max(summary.keys(), key=lambda k: summary[k]["avg_F1"])
    print(f"\nBest k_CBRT = {best_k} (avg F1 = {summary[best_k]['avg_F1']:.4f})")

    summary_path = os.path.join(ICCC_ROOT, "grid_search_k_cbrt_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"best_k": best_k, "summary": summary}, f, indent=2)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
