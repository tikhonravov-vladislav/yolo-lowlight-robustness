import os
import yaml

DATA_ROOT = os.path.abspath("data/coco_yolo")
ICCC_ROOT = os.path.abspath("iccc")

CALIB_FILENAMES_TXT = os.path.join(ICCC_ROOT, "val_calib_filenames.txt")

with open("coco_yolo.yaml", "r") as f:
    names = yaml.safe_load(f)["names"]
names_block = "names:\n" + "".join(f"  {i}: {n}\n" for i, n in names.items())

sources = {
    "normal": "images/val2017",
    "mild":   "images/val2017_mild",
    "medium": "images/val2017_medium",
    "severe": "images/val2017_lowlight",
}

print("Loading calibration filenames...")
with open(CALIB_FILENAMES_TXT, "r") as f:
    calib_filenames = [line.strip() for line in f if line.strip()]
print(f"  Loaded {len(calib_filenames)} filenames\n")

for condition, folder in sources.items():
    folder_abs = os.path.join(DATA_ROOT, folder)

    paths = []
    for filename in calib_filenames:
        abs_path = os.path.join(folder_abs, filename).replace("\\", "/")
        paths.append(abs_path)

    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        print(f"  WARNING [{condition}]: {len(missing)} images missing in {folder_abs}")
        print(f"    Example: {missing[0]}")
        continue

    txt_path = os.path.join(ICCC_ROOT, f"val_calib_{condition}.txt")
    with open(txt_path, "w") as f:
        for p in paths:
            f.write(p + "\n")

    yaml_path = os.path.join(ICCC_ROOT, f"val_calib_{condition}.yaml")
    yaml_content = (
        f"path: {DATA_ROOT.replace(os.sep, '/')}\n"
        f"train: images/train2017\n"
        f"val: {txt_path.replace(os.sep, '/')}\n"
        f"{names_block}"
    )
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"  [{condition}] {len(paths)} images")
    print(f"    txt:  val_calib_{condition}.txt")
    print(f"    yaml: val_calib_{condition}.yaml")

print("\nAll calibration YAML configs created.")
