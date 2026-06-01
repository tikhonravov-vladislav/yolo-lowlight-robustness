import yaml

with open("coco_yolo.yaml", "r") as f:
    base = yaml.safe_load(f)

names = base["names"]

configs = {
    "coco_yolo_cbrt_stage1.yaml": "train_stage1.txt",
    "coco_yolo_cbrt_stage2.yaml": "train_stage2.txt",
    "coco_yolo_cbrt_stage3.yaml": "train_stage3.txt",
    "coco_yolo_naive.yaml":       "train_naive.txt",
}

for filename, train_list in configs.items():
    data = {
        "path": "data/coco_yolo",
        "train": train_list,
        "val": "images/val2017",
        "names": names,
    }
    with open(filename, "w") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=False)
    print(f"Created {filename}")

print("\nAll yaml files updated.")
