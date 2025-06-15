#!/usr/bin/env python3
# inference_with_meta.py
# Multi-scale TTA ensemble with ConvNeXt-V2 Stage1/Stage2 + meta-learner stacking

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageFilter
import timm
from timm.data import resolve_model_data_config
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
import joblib

# ─── CONFIG ────────────────────────────────────────────────────────────────
TEST_DIR    = Path("/data/hecto/test")
MODELS_DIR  = Path("./models")
SAMPLE_CSV  = Path("/data/hecto/sample_submission.csv")
OUTPUT_CSV  = Path("submission_with_meta.csv")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE  = 8
NUM_WORKERS = 4
MODEL_NAME  = "convnextv2_large.fcmae_ft_in22k_in1k_384"

# scales and their blending weights
SCALES = [
  (320,  .08),
  (352,  .08),
  (384,  .40),
  (416,  .12),
  (448,  .20),
  (480,  .12),
]

# globals
paths = None
ckpts = None
n_classes = None

# ─── TTA TRANSFORMS ────────────────────────────────────────────────────────
def build_tta_transforms(dc):
    size = dc['input_size'][-1]
    mean, std = dc['mean'], dc['std']
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    def zoom_out(img, zoom=0.9):
        new = int(size * zoom)
        img2 = TF.resize(img, new)
        pad = (size - new) // 2
        img2 = TF.pad(img2, [pad]*4, fill=0)
        return TF.center_crop(img2, size)

    tta = []
    # center + flip
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), normalize]))
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.Lambda(TF.hflip), normalize]))
    # zoom-outs
    tta.append(transforms.Compose([transforms.Lambda(lambda x: zoom_out(x, 0.9)), normalize]))
    tta.append(transforms.Compose([transforms.Lambda(lambda x: zoom_out(x, 0.8)), normalize]))
    # sharpen, autocontrast, equalize
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.Lambda(lambda img: img.filter(ImageFilter.SHARPEN)), normalize]))
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.Lambda(ImageOps.autocontrast), normalize]))
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.Lambda(ImageOps.equalize), normalize]))
    return tta

class TestDataset(Dataset):
    def __init__(self, image_paths, transforms_list):
        self.paths = image_paths
        self.transforms = transforms_list
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        views = [tf(img) for tf in self.transforms]
        return torch.stack(views), idx

# ─── MODEL LOADER & INFERENCE ───────────────────────────────────────────────
def load_model(ckpt_file, model_name, num_classes):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(state)
    return model.to(DEVICE).eval()

def infer_for_dc(dc):
    tta = build_tta_transforms(dc)
    ds = TestDataset(paths, tta)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    preds = torch.zeros(len(ds), n_classes, device=DEVICE)
    for ckpt_file in ckpts:
        model = load_model(ckpt_file, MODEL_NAME, n_classes)
        for views, idxs in tqdm(dl, desc=f"{dc['input_size'][-1]}px:{Path(ckpt_file).name}", leave=False):
            b, t, c, h, w = views.shape
            x = views.view(b*t, c, h, w).to(DEVICE)
            with torch.no_grad(), autocast(device_type='cuda', enabled=True):
                logits = model(x)
                prob = F.softmax(logits, dim=1)
            prob = prob.view(b, t, -1).mean(1)
            preds[idxs] += prob
        torch.cuda.empty_cache()
    preds.div_(len(ckpts))
    # now do smoothing + sharpening to match meta-training
    arr = preds.cpu().numpy()
    eps, T = 1e-3, 0.9
    arr = arr * (1 - eps) + (eps / n_classes)
    arr = np.power(arr, 1.0 / T)
    arr = arr / arr.sum(axis=1, keepdims=True)
    return arr

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    global paths, ckpts, n_classes
    # load ids and setup
    df = pd.read_csv(SAMPLE_CSV)
    img_ids = df.iloc[:,0].astype(str).tolist()
    paths = [TEST_DIR/f"{i}.jpg" for i in img_ids]
    n_classes = df.shape[1] - 1

    # Stage 1
    ckpts = sorted(MODELS_DIR.glob("convnext_stage1_fold*.pth"))
    if not ckpts:
        raise FileNotFoundError("No Stage1 checkpoints found")
    print("===> Stage1 inference")
    dummy = timm.create_model(MODEL_NAME, pretrained=False)
    dc_base = resolve_model_data_config(dummy)
    preds_s1 = None
    for size, w in SCALES:
        dc = dict(dc_base)
        dc['input_size'] = (3, size, size)
        dc['crop_pct'] = size / dc_base['input_size'][-1]
        block = infer_for_dc(dc) * w
        preds_s1 = block if preds_s1 is None else preds_s1 + block

    # Stage 2
    ckpts = sorted(MODELS_DIR.glob("convnext_stage2_fold*.pth"))
    if not ckpts:
        raise FileNotFoundError("No Stage2 checkpoints found")
    print("===> Stage2 inference")
    preds_s2 = None
    for size, w in SCALES:
        dc = dict(dc_base)
        dc['input_size'] = (3, size, size)
        dc['crop_pct'] = size / dc_base['input_size'][-1]
        block = infer_for_dc(dc) * w
        preds_s2 = block if preds_s2 is None else preds_s2 + block

    # Meta-learner
    print("===> Applying meta-learner")
    meta = joblib.load("meta_data/meta_lr.pkl")
    X_test = np.concatenate([preds_s1, preds_s2], axis=1)
    final = meta.predict_proba(X_test)

    # Save submission
    df.iloc[:,1:] = final
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved → {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
