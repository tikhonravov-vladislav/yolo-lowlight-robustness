import os
import random

random.seed(42)

TOTAL_PER_EPOCH = 118287

# Use ABSOLUTE paths to avoid Ultralytics path resolution issues
DATA_ROOT = os.path.abspath("data/coco_yolo")

sources = {
    "normal": "images/train2017",
    "mild":   "images/train2017_mild",
    "medium": "images/train2017_medium",
    "severe": "images/train2017_severe",
}

def list_images(folder):
    full_path = os.path.join(DATA_ROOT, folder)
    files = [f for f in os.listdir(full_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    return sorted(files)

print("Scanning source folders...")
all_images = {}
for name, folder in sources.items():
    all_images[name] = list_images(folder)
    print(f"  {name}: {len(all_images[name])} images")

def sample_mix(level_names, total):
    n_levels = len(level_names)
    per_level = total // n_levels
    remainder = total - per_level * n_levels

    paths = []
    for i, name in enumerate(level_names):
        n_take = per_level + (1 if i < remainder else 0)
        sampled = random.sample(all_images[name], n_take)
        for filename in sampled:
            # ABSOLUTE path with forward slashes (Ultralytics-friendly)
            abs_path = os.path.join(DATA_ROOT, sources[name], filename).replace("\\", "/")
            paths.append(abs_path)

    random.shuffle(paths)
    return paths

configs = {
    "train_stage1.txt":  ["normal", "mild"],
    "train_stage2.txt":  ["normal", "mild", "medium"],
    "train_stage3.txt":  ["normal", "mild", "medium", "severe"],
    "train_naive.txt":   ["normal", "mild", "medium", "severe"],
}

for filename, levels in configs.items():
    paths = sample_mix(levels, TOTAL_PER_EPOCH)
    with open(os.path.join(DATA_ROOT, filename), "w") as f:
        for p in paths:
            f.write(p + "\n")
    print(f"Created {filename}: {len(paths)} images ({', '.join(levels)})")

print("\nAll train list files created.")
