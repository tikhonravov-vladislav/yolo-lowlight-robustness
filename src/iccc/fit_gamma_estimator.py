import os
import json
import numpy as np
import cv2

PROJECT_ROOT = os.path.abspath(".")
ICCC_ROOT = os.path.join(PROJECT_ROOT, "iccc")
DATA_ROOT = os.path.abspath("data/coco_yolo")

conditions = {
    "normal": ("images/val2017",          1.0),
    "mild":   ("images/val2017_mild",     1.4),
    "medium": ("images/val2017_medium",   1.8),
    "severe": ("images/val2017_lowlight", 2.2),
}

calib_txt = os.path.join(ICCC_ROOT, "val_calib_filenames.txt")
with open(calib_txt, "r") as f:
    calib_filenames = [line.strip() for line in f if line.strip()]
print(f"Loaded {len(calib_filenames)} calibration filenames\n")


def mean_intensity(image_path):
    """Compute the mean grayscale intensity of an image, normalized to [0, 1]."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    return float(img.mean()) / 255.0


# Collect (mean_intensity, true_gamma) pairs across all 4 conditions
print("Computing mean intensities across all conditions...")
intensities = []
gammas = []
per_condition_stats = {}

for cond_name, (folder, true_gamma) in conditions.items():
    folder_abs = os.path.join(DATA_ROOT, folder)
    cond_intensities = []

    for filename in calib_filenames:
        img_path = os.path.join(folder_abs, filename)
        mi = mean_intensity(img_path)
        if mi is not None:
            cond_intensities.append(mi)
            intensities.append(mi)
            gammas.append(true_gamma)

    avg = np.mean(cond_intensities)
    std = np.std(cond_intensities)
    per_condition_stats[cond_name] = {
        "gamma": true_gamma,
        "mean_intensity_avg": float(avg),
        "mean_intensity_std": float(std),
        "n_samples": len(cond_intensities),
    }
    print(f"  [{cond_name}] γ={true_gamma}, "
          f"mean_intensity = {avg:.4f} ± {std:.4f}, "
          f"n = {len(cond_intensities)}")

print(f"\nTotal data points: {len(intensities)}")

# Fit quadratic polynomial: gamma = a*I^2 + b*I + c
intensities = np.array(intensities)
gammas = np.array(gammas)

coeffs = np.polyfit(intensities, gammas, deg=2)
a, b, c = coeffs
print(f"\nFitted polynomial: γ = {a:.4f} * I^2 + {b:.4f} * I + {c:.4f}")

# Evaluate the fit
predicted_gammas = np.polyval(coeffs, intensities)
residuals = predicted_gammas - gammas
mae = float(np.mean(np.abs(residuals)))
rmse = float(np.sqrt(np.mean(residuals ** 2)))

print(f"\nFit quality on calibration data:")
print(f"  MAE  = {mae:.4f}")
print(f"  RMSE = {rmse:.4f}")

# Per-condition prediction quality
print(f"\nPer-condition prediction quality:")
print(f"{'condition':>10} | {'true γ':>7} | {'pred γ (avg)':>14} | {'MAE':>7}")
print("-" * 50)
for cond_name, (folder, true_gamma) in conditions.items():
    mask = gammas == true_gamma
    preds_for_cond = predicted_gammas[mask]
    avg_pred = float(np.mean(preds_for_cond))
    cond_mae = float(np.mean(np.abs(preds_for_cond - true_gamma)))
    print(f"{cond_name:>10} | {true_gamma:>7.2f} | {avg_pred:>14.4f} | {cond_mae:>7.4f}")

# Save fit parameters
estimator_data = {
    "polynomial_degree": 2,
    "coefficients": [float(c) for c in coeffs],
    "formula": "gamma = c[0]*I^2 + c[1]*I + c[2]",
    "fit_quality": {
        "MAE": mae,
        "RMSE": rmse,
    },
    "per_condition_stats": per_condition_stats,
}

out_path = os.path.join(ICCC_ROOT, "gamma_estimator.json")
with open(out_path, "w") as f:
    json.dump(estimator_data, f, indent=2)
print(f"\nGamma estimator saved to {out_path}")
