#!/usr/bin/env python3
# inference_pipeline.py
# Single-stage deep TTA + temperature scaling inference tailored for cars with MC-Dropout

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import numpy as np
import timm
from timm.data.config import resolve_model_data_config
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.amp.autocast_mode import autocast
from torch import nn

# ─── CONFIG ────────────────────────────────────────────────────────────────
TEST_DIR    = Path("/data/hecto/test")
MODELS_DIR  = Path("./models")
SAMPLE_CSV  = Path("/data/hecto/sample_submission.csv")
OUTPUT_CSV  = Path("submission_convnext.csv")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# inference settings
BATCH_SIZE      = 1                # minimal to allow max TTA
NUM_WORKERS     = 4
MODEL_NAME      = "convnextv2_large.fcmae_ft_in22k_in1k_384"
TRAIN_RES       = 448              # must match training stage-2 size
best_T          = 1.7              # temperature from validation calibration
N_MC            = 5                # MC-Dropout passes

# ─── HELPERS ─────────────────────────────────────────────────────────────────
def pad_to_base(img, base_size):
    w, h   = img.size
    pad_w  = max(0, base_size - w)
    pad_h  = max(0, base_size - h)
    left   = pad_w // 2
    right  = pad_w - left
    top    = pad_h // 2
    bottom = pad_h - top
    return TF.pad(img, [left, top, right, bottom], fill=0)

def enable_mc_dropout(model):
    # Turn on dropout layers, keep batchnorm in eval
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
        elif isinstance(m, nn.BatchNorm2d):
            m.eval()

# build deep TTA at TRAIN_RES without vertical flips
# for each scale: center + hflip + FiveCrop + TenCrop when available
def build_deep_transforms(dc,
                          base_size=TRAIN_RES,
                          scales=(0.6, 0.8, 1.0, 1.2, 1.4, 1.6)):
    mean, std = dc['mean'], dc['std']
    normalize = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    tta = []
    for s in scales:
        crop = int(base_size * s)
        # center
        tta.append(transforms.Compose([
            transforms.Resize(crop),
            transforms.Lambda(lambda img: pad_to_base(img, base_size)),
            transforms.CenterCrop(base_size),
            normalize
        ]))
        # hflip
        tta.append(transforms.Compose([
            transforms.Resize(crop),
            transforms.Lambda(lambda img: pad_to_base(img, base_size)),
            transforms.CenterCrop(base_size),
            transforms.Lambda(lambda x: TF.hflip(x)),
            normalize
        ]))
        # five-crop
        if crop >= base_size:
            tta.append(transforms.Compose([
                transforms.Resize(crop),
                transforms.FiveCrop(base_size),
                transforms.Lambda(lambda vs: torch.stack([normalize(v) for v in vs]))
            ]))
            # ten-crop
            tta.append(transforms.Compose([
                transforms.Resize(crop),
                transforms.TenCrop(base_size),
                transforms.Lambda(lambda vs: torch.stack([normalize(v) for v in vs]))
            ]))
    return tta

# custom dataset applying multiple views
class TestDataset(Dataset):
    def __init__(self, image_paths, transforms_list):
        self.paths = image_paths
        self.transforms = transforms_list
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        views = []
        for tfm in self.transforms:
            out = tfm(img)
            # unpack multi-crop outputs
            if torch.is_tensor(out) and out.ndim == 4:
                for v in out:
                    views.append(v)
            else:
                views.append(out)
        return torch.stack(views), p.name

# load a single checkpoint
def load_model(ckpt_file, model_name, num_classes, dropout_p=0.2):
    """
    Load checkpoint and insert a dropout layer before the classification head for MC-Dropout.
    """
    model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)
    state = torch.load(ckpt_file, map_location='cpu')
    model.load_state_dict(state)
    # Insert dropout before head to enable MC-Dropout
    if hasattr(model, 'head'):
        model.head = nn.Sequential(
            nn.Dropout(dropout_p),
            model.head
        )
    model.eval()  # set batchnorm to eval
    return model.to(DEVICE)

# ─── MAIN ───────────────────────────────────────────────────────────────────
def main():
    df = pd.read_csv(SAMPLE_CSV)
    df[df.columns[1:]] = df[df.columns[1:]].astype("float64")

    img_ids = df.iloc[:,0].astype(str)
    paths   = [TEST_DIR/f"{i}.jpg" for i in img_ids]
    n_classes = df.shape[1] - 1

    # prepare TTA
    dummy = timm.create_model(MODEL_NAME, pretrained=False)
    dc = resolve_model_data_config(dummy)
    transforms_list = build_deep_transforms(dc)

    ds = TestDataset(paths, transforms_list)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

    # load checkpoints
    ckpts = sorted(MODELS_DIR.glob("*.pth"))
    if not ckpts:
        raise FileNotFoundError(f"No .pth files in {MODELS_DIR}")
    state0 = torch.load(ckpts[0], map_location='cpu')
    head_key = next(k for k in state0 if k.endswith("head.fc.weight"))
    ckpt_classes = state0[head_key].shape[0]
    if ckpt_classes != n_classes:
        raise ValueError(f"Checkpoint classes={ckpt_classes} vs CSV classes={n_classes}")

    preds = torch.zeros(len(ds), n_classes, device=DEVICE)

    # inference loop with MC-Dropout & cleanup
    for ckpt in ckpts:
        print(f"Loading {ckpt.name}...")
        model = load_model(ckpt, MODEL_NAME, ckpt_classes)
        enable_mc_dropout(model)

        idx = 0
        for views, _ in tqdm(dl, desc=f"Fold {ckpt.name}"):
            b, t, c, h, w = views.shape
            x = views.view(b*t, c, h, w).to(DEVICE)

            # MC-Dropout ensemble
            mc_probs = []
            with torch.no_grad(), autocast(device_type= 'cuda', enabled=True):
                for _ in range(N_MC):
                    logits = model(x)
                    logits = logits / best_T
                    mc_probs.append(F.softmax(logits, dim=1))
            probs = torch.stack(mc_probs, dim=0).mean(0)
            probs = probs.view(b, t, -1).mean(dim=1)

            preds[idx:idx+b] += probs
            idx += b

            # free memory
            del x, mc_probs, probs
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()

    preds /= len(ckpts)
    arr = preds.cpu().numpy()

    df.iloc[:,1:] = arr
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved submission to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
