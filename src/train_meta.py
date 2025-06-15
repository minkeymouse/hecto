#!/usr/bin/env python3
"""
meta_learner.py

End-to-end stacking pipeline using a CFG object:
1. Generate stratified K-fold splits (train_folds.csv) from your train directory
2. Generate OOF predictions for each base model (Stage1, Stage2)
3. Generate test-set predictions for each base model
4. Assemble meta-features and train a logistic regression meta-learner

Run the script directly; configuration is in CFG.
"""
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
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import joblib
from tqdm import tqdm

# ─── CONFIG ────────────────────────────────────────────────────────────────
class CFG:
    # Paths
    train_dir        = Path("/data/hecto/train")
    test_dir         = Path("/data/hecto/test")
    sample_csv       = Path("/data/hecto/sample_submission.csv")
    models_dir       = Path("./models")
    output_dir       = Path("./meta_data")
    train_folds_csv  = Path("train_folds.csv")

    # Inference settings
    n_folds          = 4
    seed             = 42
    batch_size       = 8
    num_workers      = 4
    device           = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name       = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    scales           = [(320,0.08),(352,0.08),(384,0.40),(416,0.12),(448,0.20),(480,0.12)]

cfg = CFG()
cfg.output_dir.mkdir(exist_ok=True, parents=True)

# ─── FOLD GENERATION ────────────────────────────────────────────────────────
def create_folds():
    classes = sorted([d.name for d in cfg.train_dir.iterdir() if d.is_dir()])
    ids, labels = [], []
    for idx, cls in enumerate(classes):
        for img in (cfg.train_dir/cls).glob("*.jpg"):
            ids.append(img.stem)
            labels.append(idx)
    df = pd.DataFrame({"id": ids, "label": labels})
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    df["fold"] = -1
    for fold, (_, vidx) in enumerate(skf.split(df.id, df.label)):
        df.loc[vidx, "fold"] = fold
    df.to_csv(cfg.train_folds_csv, index=False)
    print(f"[folds] saved → {cfg.train_folds_csv}")

# ─── TRANSFORMS & DATASET ───────────────────────────────────────────────────
def build_tta_transforms(dc):
    size = dc['input_size'][-1]
    mean, std = dc['mean'], dc['std']
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    def zoom_out(img, zoom=0.9):
        new = int(size * zoom)
        img2 = TF.resize(img, new)
        pad = (size-new)//2
        img2 = TF.pad(img2, [pad]*4, fill=0)
        return TF.center_crop(img2, size)
    tta = []
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), normalize]))
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.Lambda(TF.hflip), normalize]))
    for z in (0.9, 0.8):
        tta.append(transforms.Compose([transforms.Lambda(lambda x: zoom_out(x, z)), normalize]))
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.Lambda(lambda img: img.filter(ImageFilter.SHARPEN)), normalize]))
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.Lambda(ImageOps.autocontrast), normalize]))
    tta.append(transforms.Compose([transforms.Resize(size), transforms.CenterCrop(size), transforms.Lambda(ImageOps.equalize), normalize]))
    return tta

class InferenceDataset(Dataset):
    def __init__(self, ids, base_dir, dc):
        self.ids = ids
        self.base_dir = base_dir
        self.transforms = build_tta_transforms(dc)
    def __len__(self): return len(self.ids)
    def __getitem__(self, i):
        img = Image.open(self.base_dir/f"{self.ids[i]}.jpg").convert("RGB")
        views = torch.stack([tf(img) for tf in self.transforms])
        return views, i

# ─── INFERENCE ──────────────────────────────────────────────────────────────
def load_model(ckpt, num_classes):
    m = timm.create_model(cfg.model_name, pretrained=False, num_classes=num_classes)
    sd = torch.load(ckpt, map_location='cpu')
    m.load_state_dict(sd)
    return m.to(cfg.device).eval()

def infer_multi_scale(ids, base_dir, ckpts):
    dummy = timm.create_model(cfg.model_name, pretrained=False)
    dc_base = resolve_model_data_config(dummy)
    preds = None
    for size, w in cfg.scales:
        dc = dict(dc_base)
        dc['input_size'] = (3, size, size)
        dc['crop_pct'] = size/dc_base['input_size'][-1]
        ds = InferenceDataset(ids, base_dir, dc)
        dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
        scale_pred = torch.zeros(len(ds), dummy.num_classes, device=cfg.device)
        for ckpt in ckpts:
            model = load_model(ckpt, dummy.num_classes)
            for views, idxs in tqdm(dl, desc=f"{size}px {ckpt.name}", leave=False):
                b,t,c,h,w = views.shape
                x = views.view(b*t,c,h,w).to(cfg.device)
                with torch.no_grad(), torch.cuda.amp.autocast():
                    logits = model(x)
                    prob = F.softmax(logits,1)
                prob = prob.view(b,t,-1).mean(1)
                scale_pred[idxs] += prob
            scale_pred.div_(len(ckpts))
        arr = (w*scale_pred).cpu().numpy()
        preds = arr if preds is None else preds+arr
    eps, T = 1e-3, 0.9
    preds = preds*(1-eps)+(eps/preds.shape[1])
    preds = np.power(preds,1.0/T)
    preds /= preds.sum(1,keepdims=True)
    return preds

# ─── PIPELINE STEPS ─────────────────────────────────────────────────────────
def generate_oof():
    df = pd.read_csv(cfg.train_folds_csv)
    for fold in range(cfg.n_folds):
        val = df[df.fold==fold]; ids = val.id.astype(str).tolist(); y = val.label.values
        o1 = infer_multi_scale(ids, cfg.train_dir, sorted(cfg.models_dir.glob(f"convnext_stage1_fold{fold}*.pth")))
        o2 = infer_multi_scale(ids, cfg.train_dir, sorted(cfg.models_dir.glob(f"convnext_stage2_fold{fold}*.pth")))
        np.save(cfg.output_dir/f"oof_s1_fold{fold}.npy", o1)
        np.save(cfg.output_dir/f"oof_s2_fold{fold}.npy", o2)
        np.save(cfg.output_dir/f"labels_fold{fold}.npy", y)
        print(f"[OOF] fold{fold} → {o1.shape}")

def generate_test():
    sub = pd.read_csv(cfg.sample_csv); ids = sub.iloc[:,0].astype(str).tolist()
    t1 = infer_multi_scale(ids, cfg.test_dir, sorted(cfg.models_dir.glob("convnext_stage1_fold*.pth")))
    t2 = infer_multi_scale(ids, cfg.test_dir, sorted(cfg.models_dir.glob("convnext_stage2_fold*.pth")))
    np.save(cfg.output_dir/"test_s1.npy", t1); np.save(cfg.output_dir/"test_s2.npy", t2)
    print(f"[TEST] → test_s1.npy, test_s2.npy")

def train_meta():
    X, y = [], []
    for fold in range(cfg.n_folds):
        s1 = np.load(cfg.output_dir/f"oof_s1_fold{fold}.npy");
        s2 = np.load(cfg.output_dir/f"oof_s2_fold{fold}.npy");
        lbl= np.load(cfg.output_dir/f"labels_fold{fold}.npy")
        X.append(np.concatenate([s1, s2],1)); y.append(lbl)
    X = np.vstack(X); y = np.concatenate(y)
    print(f"[META] X={X.shape} y={y.shape}")
    meta = LogisticRegression(multi_class="multinomial",solver="lbfgs",C=1.0,max_iter=1000)
    meta.fit(X,y); joblib.dump(meta, cfg.output_dir/"meta_lr.pkl")
    print("[META] saved meta_lr.pkl")

# ─── EXECUTE ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    create_folds()
    generate_oof()
    generate_test()
    train_meta()
