# Generates mild, medium, and severe low-light versions of COCO train2017.
import os
import cv2
import numpy as np
from tqdm import tqdm

np.random.seed(42)

src_dir = "data/coco_yolo/images/train2017"

levels = {
    "mild":   {"gamma": 1.4, "scale": 0.75, "noise": 5},
    "medium": {"gamma": 1.8, "scale": 0.55, "noise": 8},
    "severe": {"gamma": 2.2, "scale": 0.40, "noise": 10},
}

def apply_lowlight(img, gamma, scale, noise_std):
    img = img.astype(np.float32) / 255.0
    img = np.power(img, gamma)
    img = img * scale

    noise = np.random.normal(0, noise_std/255.0, img.shape)
    img = img + noise

    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)

    return img

for level_name, params in levels.items():
    dst_dir = f"data/coco_yolo/images/train2017_{level_name}"
    os.makedirs(dst_dir, exist_ok=True)

    print(f"\nGenerating {level_name} level for train2017...")

    for filename in tqdm(os.listdir(src_dir)):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        img = cv2.imread(src_path)
        if img is None:
            continue

        low_img = apply_lowlight(
            img,
            params["gamma"],
            params["scale"],
            params["noise"]
        )
        cv2.imwrite(dst_path, low_img)

print("\nAll train low-light levels created.")
