import os
import time
import math
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from ultralytics import YOLO

# ─── Configuration ─────────────────────────────────────────────────────────
class CFG:
    train_dir = '../data/hecto/train'  # adjust if needed

cfg = CFG()


# ─── 1) Pad helper (no OpenCV) ───────────────────────────────────────────────
def pad_mask(pil_img: Image.Image, patch_size: int):
    """
    Pad the input PIL image up to multiples of patch_size, then zero out padded margins.
    Returns:
      - img_masked: np.ndarray (h_pad, w_pad, 3)
      - (h_pad, w_pad)
      - (orig_w, orig_h)
      - (x1, y1, x2, y2) for the original region (always (0,0,orig_w,orig_h))
    """
    img_np = np.array(pil_img)
    orig_h, orig_w = img_np.shape[:2]
    ps = patch_size

    # Compute padded dims
    w_pad = math.ceil(orig_w / ps) * ps
    h_pad = math.ceil(orig_h / ps) * ps
    pad_right  = w_pad - orig_w
    pad_bottom = h_pad - orig_h

    # Pad with zeros on right and bottom
    img_padded = np.pad(
        img_np,
        ((0, pad_bottom), (0, pad_right), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Build mask for original region
    x1, y1, x2, y2 = 0, 0, orig_w, orig_h
    mask2d = np.zeros((h_pad, w_pad), dtype=np.uint8)
    mask2d[y1:y2, x1:x2] = 1
    mask3c = np.stack([mask2d] * 3, axis=2)

    img_masked = img_padded * mask3c
    return img_masked, (h_pad, w_pad), (orig_w, orig_h), (x1, y1, x2, y2)


# ─── 2) FullImageLoader ─────────────────────────────────────────────────────
class FullImageLoader:
    """
    Scans a training directory with subfolders per class and builds:
      - self.paths:  List[str] of full image paths
      - self.labels: List[int] of integer labels (not used here, but collected)
      - self.classes: sorted List[str] of class names
    """
    def __init__(self, train_dir: str):
        self.train_dir = Path(train_dir)
        self.classes = sorted([d.name for d in self.train_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.paths: list[str] = []
        self.labels: list[int] = []

        for cls_name in self.classes:
            cls_idx = self.class_to_idx[cls_name]
            cls_folder = self.train_dir / cls_name
            for img_file in sorted(cls_folder.iterdir()):
                if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue
                self.paths.append(str(img_file))
                self.labels.append(cls_idx)

        assert len(self.paths) == len(self.labels), "Mismatch between image paths and labels"


# ─── Main precompute routine ─────────────────────────────────────────────────
def main():
    # ─── STEP 0: Load YOLO model ───────────────────────────────────────────────
    yolo_weights_path = "../models/yolo11n.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        yolo_model = YOLO(yolo_weights_path)
        yolo_model.to(device).eval()
        print("✅ YOLO11n loaded successfully\n")
    except Exception as e:
        raise RuntimeError(f"❌ Could not load YOLO11n weights:\n  {e}")

    # ─── STEP 0.1: Warm‐up CUDA ────────────────────────────────────────────────
    with torch.no_grad():
        dummy = torch.zeros((1, 3, 640, 640), device=device, dtype=torch.float32)
        _ = yolo_model(dummy)[0]
    del dummy
    print("🚀 CUDA warmed up with dummy forward-pass\n")

    # ─── STEP 1: Prepare FullImageLoader & parameters ────────────────────────
    full_loader = FullImageLoader(cfg.train_dir)
    conf_thresh  = 0.25
    img_size_inf = 640   # letterbox size for YOLO
    patch_size   = 224   # for efficientnet_b0 variant

    variant     = "b0"
    out_base    = Path("/data") / "hecto" / f"precomputed_{variant}"
    stage1_base = out_base / "stage1_masked_resized"
    stage2_base = out_base / "stage2_masked_patchify"
    stage1_base.mkdir(parents=True, exist_ok=True)
    stage2_base.mkdir(parents=True, exist_ok=True)

    total_images = len(full_loader.paths)
    print(f"=== Precomputing variant {variant} (patch_size = {patch_size}) ===")
    start_all = time.time()

    for idx, img_path in enumerate(full_loader.paths, start=1):
        if idx % 100 == 1:
            elapsed = time.time() - start_all
            print(f"  [{variant}] Image {idx}/{total_images} — elapsed {elapsed:.1f}s")

        rel      = Path(img_path).relative_to(cfg.train_dir)
        cls_name = rel.parent.name
        stem     = rel.stem

        # Make class subfolders
        (stage1_base / cls_name).mkdir(parents=True, exist_ok=True)
        (stage2_base / cls_name).mkdir(parents=True, exist_ok=True)

        pil = Image.open(img_path).convert("RGB")
        img0 = np.array(pil)

        # YOLO inference
        results = yolo_model(pil, imgsz=img_size_inf)
        dets    = results[0].boxes

        boxes_xyxy = dets.xyxy.cpu().numpy()
        confs      = dets.conf.cpu().numpy()
        cls_idxs   = dets.cls.cpu().numpy()

        # Filter for class index 2 (car) and confidence threshold
        car_mask = (cls_idxs == 2) & (confs >= conf_thresh)
        if car_mask.sum() == 0:
            masked = pil
        else:
            car_boxes = boxes_xyxy[car_mask]
            areas     = (car_boxes[:, 2] - car_boxes[:, 0]) * (car_boxes[:, 3] - car_boxes[:, 1])
            best_i    = int(np.argmax(areas))
            x1, y1, x2, y2 = car_boxes[best_i].astype(int)

            H, W, _ = img0.shape
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(0, min(x2, W))
            y2 = max(0, min(y2, H))

            masked = Image.new("RGB", pil.size, (0, 0, 0))
            car_crop = pil.crop((x1, y1, x2, y2))
            masked.paste(car_crop, (x1, y1))

        # Stage 1: resize the masked image to (patch_size, patch_size) and save
        resized_masked = masked.resize((patch_size, patch_size))
        arr_s1 = np.array(resized_masked, dtype=np.uint8)
        np.save(str(stage1_base / cls_name / f"{stem}.npy"), arr_s1)

        # Stage 2: pad → extract non-overlapping patches → save
        padded_np, (h_pad, w_pad), (_, _), (_, _, _, _) = pad_mask(
            masked, patch_size
        )
        patches_m = []
        for top in range(0, h_pad, patch_size):
            for left in range(0, w_pad, patch_size):
                tile = padded_np[top : top + patch_size, left : left + patch_size, :]
                patches_m.append(tile)
        arr_s2 = np.stack(patches_m, axis=0).astype(np.uint8)
        np.save(str(stage2_base / cls_name / f"{stem}.npy"), arr_s2)

    elapsed_total = time.time() - start_all
    print(f"\n→ Finished variant {variant} in {elapsed_total:.1f}s, saved under {out_base}")


if __name__ == "__main__":
    main()
