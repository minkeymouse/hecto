#!/usr/bin/env python3
# inference_convnextv2_two_stage_multiscale.py
# Ensemble inference with multi-scale TTA + two-stage fine-tuned ConvNeXt-V2

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps, ImageFilter
import pandas as pd
import timm
from timm.data import resolve_model_data_config
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
import numpy as np

# ─── CONFIG ────────────────────────────────────────────────────────────────
TEST_DIR    = Path("/data/hecto/test")
MODELS_DIR  = Path("./models")
SAMPLE_CSV  = Path("/data/hecto/sample_submission.csv")
OUTPUT_CSV  = Path("submission_convnext_multiscale.csv")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE  = 8
NUM_WORKERS = 4
MODEL_NAME  = "convnextv2_large.fcmae_ft_in22k_in1k_384"

# scales and their blending weights
SCALES = [
    (320,  0.08),
    (352,  0.08),
    (384,  0.40),
    (416,  0.12),  # new mid-point
    (448,  0.20),
    (480,  0.12),  # another mid-point
]

# global placeholders filled in main()
paths = None
ckpts = None
n_classes = None

# ─── BUILD TTA TRANSFORMS ─────────────────────────────────────────────────
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

    tta_list = []
    # center
    tta_list.append(transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        normalize
    ]))
    # horizontal flip
    tta_list.append(transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.Lambda(TF.hflip), normalize
    ]))
    # zoom outs
    tta_list.append(transforms.Compose([transforms.Lambda(lambda x: zoom_out(x, 0.9)), normalize]))
    tta_list.append(transforms.Compose([transforms.Lambda(lambda x: zoom_out(x, 0.8)), normalize]))
    # sharpen, autocontrast, equalize
    tta_list.append(transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.Lambda(lambda img: img.filter(ImageFilter.SHARPEN)), normalize
    ]))
    tta_list.append(transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.Lambda(ImageOps.autocontrast), normalize
    ]))
    tta_list.append(transforms.Compose([
        transforms.Resize(size), transforms.CenterCrop(size),
        transforms.Lambda(ImageOps.equalize), normalize
    ]))
    return tta_list

# ─── DATASET ───────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, image_paths, transforms_list):
        self.paths = image_paths
        self.transforms = transforms_list
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        views = [tf(img) for tf in self.transforms]
        return torch.stack(views), p.name

# ─── MODEL LOADER ─────────────────────────────────────────────────────────
def load_model(ckpt_file, model_name, num_classes):
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(ckpt_file, map_location="cpu")
    model.load_state_dict(state)
    return model.to(DEVICE).eval()

# ─── MULTI-SCALE INFERENCE ─────────────────────────────────────────────────
def infer_for_dc(dc):
    tta_transforms = build_tta_transforms(dc)
    ds = TestDataset(paths, tta_transforms)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

    preds = torch.zeros(len(ds), n_classes, device=DEVICE)
    for ckpt in ckpts:
        model = load_model(ckpt, MODEL_NAME, n_classes)
        model.eval()
        idx = 0
        for views, _ in tqdm(dl, desc=f"{dc['input_size'][-1]}px:{ckpt.name}"):
            b, t, c, h, w = views.shape
            x = views.view(b*t, c, h, w).to(DEVICE)
            with torch.no_grad(), autocast(device_type='cuda', enabled=True):
                logits = model(x)
                probs  = F.softmax(logits, dim=1)
            probs = probs.view(b, t, -1).mean(dim=1)
            preds[idx:idx+b] += probs
            idx += b
        torch.cuda.empty_cache()
    preds /= len(ckpts)
    return preds.cpu().numpy()

# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    global paths, ckpts, n_classes

    # load submission template and paths
    df       = pd.read_csv(SAMPLE_CSV)
    img_ids  = df.iloc[:,0].astype(str)
    paths    = [TEST_DIR/f"{i}.jpg" for i in img_ids]
    n_classes = df.shape[1] - 1

    # prepare checkpoints
    ckpts = sorted(MODELS_DIR.glob("convnext_fold*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {MODELS_DIR}")

    # base config for default scale
    dummy = timm.create_model(MODEL_NAME, pretrained=False)
    dc_base = resolve_model_data_config(dummy)

    # run multi-scale inference
    combined = None
    for size, weight in SCALES:
        dc_i = dict(dc_base)
        dc_i['input_size'] = (3, size, size)
        dc_i['crop_pct']   = size / dc_base['input_size'][-1]
        print(f"Running inference at {size}x{size} (weight {weight})")
        preds_i = infer_for_dc(dc_i)
        arr_i = weight * preds_i
        combined = arr_i if combined is None else combined + arr_i

    arr = combined
    # uniform-prior smoothing
    eps = 1e-3
    arr = arr * (1 - eps) + (eps / n_classes)
    # temperature sharpening
    T = 0.9
    arr = np.power(arr, 1.0 / T)
    arr = arr / arr.sum(axis=1, keepdims=True)

    df.iloc[:,1:] = arr
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved multi-scale ensemble to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
