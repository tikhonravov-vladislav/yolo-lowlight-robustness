"""
Runs COCO validation images through three enhancement methods:
  1) Zero-DCE   (curve estimation, CVPR 2020)
  2) SCI         (self-calibrated illumination, CVPR 2022)
  3) CLAHE       (classical adaptive histogram equalization)
 
Usage:
  python enhance_images.py --method zero_dce
  python enhance_images.py --method sci
  python enhance_images.py --method clahe
  python enhance_images.py --method all
"""
 
import os
import sys
import argparse
import glob
import numpy as np
import cv2
import torch
from tqdm import tqdm
 
PROJECT_ROOT = os.path.abspath(".")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data", "coco_yolo", "images")
 
ZERODCE_ROOT = os.path.join(PROJECT_ROOT, "enhancement_comparison", "Zero-DCE", "Zero-DCE_code")
SCI_ROOT = os.path.join(PROJECT_ROOT, "enhancement_comparison", "SCI", "CVPR")
 
VAL_FOLDERS = {
    "normal": "val2017",
    "mild":   "val2017_mild",
    "medium": "val2017_medium",
    "severe": "val2017_lowlight",
}
 
 
#Zero-DCE
def load_zero_dce():
    sys.path.insert(0, ZERODCE_ROOT)
    from model import enhance_net_nopool as DCENet
    model = DCENet()
    weights_path = os.path.join(ZERODCE_ROOT, "snapshots", "Epoch99.pth")
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    model.cuda()
    print(f"[Zero-DCE] Loaded weights from {weights_path}")
    sys.path.pop(0)
    return model
 
 
def enhance_zero_dce(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        _, enhanced, _ = model(img_t)
    enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    return enhanced_bgr
 
 
#SCI

def load_sci():
    sys.path.insert(0, SCI_ROOT)
    from model import Finetunemodel
    model = Finetunemodel(os.path.join(SCI_ROOT, "weights", "difficult.pt"))
    model.eval()
    model.cuda()
    print(f"[SCI] Loaded weights from {SCI_ROOT}/weights/difficult.pt")
    sys.path.pop(0)
    return model
 
 
def enhance_sci(model, img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_t = torch.from_numpy(img_rgb.astype(np.float32) / 255.0)
    img_t = img_t.permute(2, 0, 1).unsqueeze(0).cuda()
    with torch.no_grad():
        result = model(img_t)
        if isinstance(result, tuple):
            enhanced = result[-1]
        else:
            enhanced = result
    enhanced = enhanced.squeeze(0).permute(1, 2, 0).cpu().numpy()
    enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    return enhanced_bgr
 
#CLAHE 

def load_clahe():
    print("[CLAHE] No model to load (OpenCV method)")
    return None
 
 
def enhance_clahe(model, img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced_bgr = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced_bgr
 
 
# Main processing loop
def process_folder(model, enhance_fn, src_folder, dst_folder, method_name):
    os.makedirs(dst_folder, exist_ok=True)
    image_files = sorted(glob.glob(os.path.join(src_folder, "*.jpg")))
    if not image_files:
        image_files = sorted(glob.glob(os.path.join(src_folder, "*.png")))
 
    print(f"\n[{method_name}] Processing {len(image_files)} images")
    print(f"  From: {src_folder}")
    print(f"  To:   {dst_folder}")
 
    for img_path in tqdm(image_files, desc=f"{method_name}"):
        filename = os.path.basename(img_path)
        dst_path = os.path.join(dst_folder, filename)
 
        if os.path.exists(dst_path):
            continue
 
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Cannot read {img_path}, skipping")
            continue
 
        try:
            enhanced = enhance_fn(model, img)
            cv2.imwrite(dst_path, enhanced)
        except Exception as e:
            print(f"  ERROR on {filename}: {e}")
            continue
 
 
def main():
    parser = argparse.ArgumentParser(description="Enhance validation images")
    parser.add_argument("--method", type=str, required=True,
                        choices=["zero_dce", "sci", "clahe", "all"],
                        help="Which enhancement method to run")
    parser.add_argument("--conditions", type=str, nargs="+",
                        default=["normal", "mild", "medium", "severe"],
                        help="Which conditions to process")
    args = parser.parse_args()
 
    if args.method == "all":
        methods_to_run = ["zero_dce", "sci", "clahe"]
    else:
        methods_to_run = [args.method]
 
    for method in methods_to_run:
        print(f"\n{'='*60}")
        print(f" Loading {method}")
        print(f"{'='*60}")
 
        if method == "zero_dce":
            model = load_zero_dce()
            enhance_fn = enhance_zero_dce
            suffix = "zerodce"
        elif method == "sci":
            model = load_sci()
            enhance_fn = enhance_sci
            suffix = "sci"
        elif method == "clahe":
            model = load_clahe()
            enhance_fn = enhance_clahe
            suffix = "clahe"
 
        for cond_name in args.conditions:
            src_folder = os.path.join(DATA_ROOT, VAL_FOLDERS[cond_name])
            dst_folder = os.path.join(DATA_ROOT, f"{VAL_FOLDERS[cond_name]}_{suffix}")
            process_folder(model, enhance_fn, src_folder, dst_folder, f"{method}/{cond_name}")
 
        if model is not None:
            del model
            torch.cuda.empty_cache()
 
    print("\n" + "="*60)
    print(" All done! Enhanced images saved.")
    print("="*60)
 
 
if __name__ == "__main__":
    main()
