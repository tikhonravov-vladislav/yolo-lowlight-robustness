# split_val.py
import os
import random

random.seed(42)

CALIB_RATIO = 0.20

# Use ABSOLUTE paths to avoid Ultralytics path resolution issues
DATA_ROOT = os.path.abspath("data/coco_yolo")
ICCC_ROOT = os.path.abspath("iccc")

VAL_FOLDER = "images/val2017"

def list_images(folder):
    full_path = os.path.join(DATA_ROOT, folder)
    files = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return sorted(files)

print("Scanning val2017 folder...")
all_val_images = list_images(VAL_FOLDER)
print(f"  val2017: {len(all_val_images)} images")

shuffled = all_val_images.copy()
random.shuffle(shuffled)

n_calib = int(len(shuffled) * CALIB_RATIO)
calib_files = shuffled[:n_calib]
holdout_files = shuffled[n_calib:]

print(f"\nCalibration split: {len(calib_files)} images ({CALIB_RATIO*100:.0f}%)")
print(f"Held-out split:    {len(holdout_files)} images ({(1-CALIB_RATIO)*100:.0f}%)")

configs = {
    "val_calib_filenames.txt":   calib_files,
    "val_holdout_filenames.txt": holdout_files,
}

os.makedirs(ICCC_ROOT, exist_ok=True)

for filename, file_list in configs.items():
    out_path = os.path.join(ICCC_ROOT, filename)
    with open(out_path, "w") as f:
        for name in file_list:
            f.write(name + "\n")
    print(f"Created {filename}: {len(file_list)} filenames")

# Sanity check
assert set(calib_files).isdisjoint(set(holdout_files)), "Splits overlap!"
assert len(calib_files) + len(holdout_files) == len(all_val_images), "Lost images!"
print("\nSanity checks passed.")
