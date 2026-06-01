import json
from pathlib import Path
from collections import defaultdict

# Paths
COCO_ROOT = Path("data/coco_raw")
OUT_ROOT = Path("data/coco_yolo")

ANN_TRAIN = COCO_ROOT / "annotations" / "instances_train2017.json"
ANN_VAL   = COCO_ROOT / "annotations" / "instances_val2017.json"

def convert(ann_path: Path, split: str):
    out_labels_dir = OUT_ROOT / "labels" / split
    out_labels_dir.mkdir(parents=True, exist_ok=True)

    with ann_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: (img["file_name"], img["width"], img["height"]) for img in coco["images"]}

    # category_id -> 0..79 (COCO has 80 classes
    cats_sorted = sorted(coco["categories"], key=lambda x: x["id"])
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(cats_sorted)}

    # group annotations per image
    ann_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        if ann.get("iscrowd", 0) == 1:
            continue
        ann_by_image[ann["image_id"]].append(ann)

    # write one .txt per image
    written = 0
    for image_id, anns in ann_by_image.items():
        file_name, w, h = images[image_id]
        lines = []
        for ann in anns:
            x, y, bw, bh = ann["bbox"]  # COCO bbox: top-left x,y + width,height (pixels)

            # convert to YOLO normalized: x_center, y_center, width, height in [0,1]
            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h

            cls = cat_id_to_idx[ann["category_id"]]
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            (out_labels_dir / Path(file_name).with_suffix(".txt").name).write_text("\n".join(lines), encoding="utf-8")
            written += 1

    print(f"{split}: wrote labels for {written} images (skipped empty images)")

if __name__ == "__main__":
    convert(ANN_TRAIN, "train2017")
    convert(ANN_VAL, "val2017")
