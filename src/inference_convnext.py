#!/usr/bin/env python3
# inference_convnextv2_two_stage.py
# Ensemble inference with two-stage fine-tuned ConvNeXt-V2 and TTA, with progress tracking

from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import timm
from timm.data import resolve_model_data_config, create_transform
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.amp import autocast

# ─── CONFIG ────────────────────────────────────────────────────────────────
TEST_DIR    = Path("/data/hecto/test")
MODELS_DIR  = Path("./models/convnext")
SAMPLE_CSV  = Path("/data/hecto/sample_submission.csv")
OUTPUT_CSV  = Path("submission_convnext.csv")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# On a 16GB GPU using 4 TTA views, start with batch size ~8
BATCH_SIZE  = 24
NUM_WORKERS = 4
MODEL_NAME  = "convnextv2_large.fcmae_ft_in22k_in1k_384"

# ─── BUILD TTA TRANSFORMS (no crop ops) ────────────────────────────────────
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
    # original
    tta_list.append(transforms.Compose([
        transforms.Resize(int(size * dc['crop_pct'])),
        transforms.CenterCrop(size), normalize
    ]))
    # horizontal flip
    tta_list.append(transforms.Compose([
        transforms.Resize(int(size * dc['crop_pct'])),
        transforms.CenterCrop(size),
        transforms.Lambda(lambda x: TF.hflip(x)), normalize
    ]))
    # zoom outs
    tta_list.append(transforms.Compose([
        transforms.Lambda(lambda x: zoom_out(x, 0.9)), normalize
    ]))
    tta_list.append(transforms.Compose([
        transforms.Lambda(lambda x: zoom_out(x, 0.8)), normalize
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

# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    # read submission template
    df = pd.read_csv(SAMPLE_CSV)
    img_ids = df.iloc[:,0].astype(str)
    paths = [TEST_DIR/f"{i}.jpg" for i in img_ids]
    n_classes = df.shape[1] - 1

    # setup TTA
    dummy = timm.create_model(MODEL_NAME, pretrained=False)
    dc = resolve_model_data_config(dummy)
    tta_transforms = build_tta_transforms(dc)
    n_views = len(tta_transforms)

    # dataset & loader
    ds = TestDataset(paths, tta_transforms)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

    # infer class count from checkpoint head
    ckpt0 = MODELS_DIR / "convnext_fold0.pth"
    if not ckpt0.exists(): raise FileNotFoundError(ckpt0)
    state0 = torch.load(ckpt0, map_location="cpu")
    head_key = next(k for k in state0 if k.endswith("head.fc.weight"))
    ckpt_classes = state0[head_key].shape[0]
    if ckpt_classes != n_classes:
        raise ValueError(f"Checkpoint classes={ckpt_classes} vs CSV classes={n_classes}")

    preds = torch.zeros(len(ds), n_classes, device=DEVICE)
    ckpts = sorted(MODELS_DIR.glob("convnext_fold*.pth"))
    if not ckpts: raise FileNotFoundError(MODELS_DIR)

    # loop folds
    for ckpt in ckpts:
        print(f"Loading {ckpt.name}...")
        model = load_model(ckpt, MODEL_NAME, ckpt_classes)
        idx = 0
        # track per-fold progress
        for views, _ in tqdm(dl, desc=f"Fold {ckpt.name}"):
            b, t, c, h, w = views.shape
            # effective batch = BATCH_SIZE * n_views
            x = views.view(b*t, c, h, w).to(DEVICE)
            with torch.no_grad(), autocast(device_type='cuda', enabled=True):
                logits = model(x)
                probs = F.softmax(logits, dim=1)
            # average over TTA
            probs = probs.view(b, t, -1).mean(dim=1)
            preds[idx:idx+b] += probs
            idx += b
        torch.cuda.empty_cache()

    # average across folds
    preds /= len(ckpts)
    arr = preds.cpu().numpy()

    # fill submission
    df.iloc[:,1:] = arr
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved ensemble to {OUTPUT_CSV}")

if __name__ == "__main__": main()
