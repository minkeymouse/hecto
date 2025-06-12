#!/usr/bin/env python3
# train_convnextv2_two_stage.py
# Two-stage fine-tune of ConvNeXt-V2 with class-aware geometric augmentations

import argparse, random, math
from pathlib import Path
import contextlib

import pickle
import numpy as np
import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

import timm
from timm.data import create_transform
from timm.loss import SoftTargetCrossEntropy
from timm.data import resolve_model_data_config
from timm.utils import ModelEmaV2

# ─────────────────────────────────────────────────────────────────────────────
class CFG:
    # — unchanged hardware & I/O
    train_dir    = Path("data/train")
    out_dir      = Path("outputs/convnext2"); out_dir.mkdir(exist_ok=True)
    num_workers  = 8
    fp16         = True

    # — unchanged CV split
    model_name   = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    folds        = 5
    seed         = 42

    # — Stage 1: easy crops at 336px (keep bs=88)
    size_s1      = 336
    bs_s1        = 88
    ep_s1        = 30      # fewer epochs—most gains happen early
    lr_s1        = 2e-4    # ↑ from 8e-5 to give a stronger signal
    crop_prob_s1 = 0.7

    # — Stage 2: full images at 448px (keep bs=48)
    size_s2      = 448
    bs_s2        = 48
    ep_s2        = 60       # fine-tune only needs a handful
    lr_s2        = 5e-5    # ↑ from 1e-5
    crop_prob_s2 = 0.3

    # — optimization
    wd           = 0.01    # ↓ from 0.05
    eta_min      = 1e-6
    early_stop   = 5       # stop sooner if no val-loss gain

    # — regularization
    mixup        = 0.0     # ↓ from 0.2
    cutmix       = 0.0     # ↓ from 0.4


cfg = CFG()

# ─── Helpers ────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def detect_car_crop(img, pad=15):
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cnts,_ = cv2.findContours(cv2.Canny(gray,50,150),
                              cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return img
    x,y,w,h = cv2.boundingRect(max(cnts,key=cv2.contourArea))
    area = bgr.shape[0]*bgr.shape[1]
    if w*h < 0.05*area:
        return img
    y1 = max(y-pad,0)
    y2 = min(y+h+pad, img.height)
    return img.crop((0,y1,img.width,y2))

def effective_weights(labels, beta=0.9999):
    counts = np.bincount(labels, minlength=cfg.C)
    eff    = (1 - np.power(beta, counts)) / (1 - beta)
    eff[eff==0] = 1
    return (1.0/eff)[labels]

# Build geometric-only augment pipelines
_conf = resolve_model_data_config(timm.create_model(cfg.model_name, pretrained=True))
_mean,_std = _conf["mean"], _conf["std"]

def make_geom(size):
    return T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomAffine(degrees=15, translate=(0,0), scale=(0.9,1.1), shear=15),
        T.RandomResizedCrop(size, scale=(0.8,1.0)),
        T.ToTensor(),
        T.Normalize(_mean,_std),
    ])

tf_good_s1    = make_geom(cfg.size_s1)
tf_default_s1 = make_geom(cfg.size_s1)  # can lighten if desired
tf_good_s2    = make_geom(cfg.size_s2)
tf_default_s2 = make_geom(cfg.size_s2)

# ─── Data Preparation ───────────────────────────────────────────────────────
# 1) Scan paths & preload
classes   = sorted([d.name for d in cfg.train_dir.iterdir() if d.is_dir()])
all_paths = [p for cls in classes for p in (cfg.train_dir/cls).glob("*.jpg")]
cfg.C     = len(classes)

# Preload
CACHE_FILE = "ram_cache.pkl"

print(f"Found {cfg.C} classes • {len(all_paths)} images")

if Path(CACHE_FILE).exists():
    print("Loading RAM cache…")
    with open(CACHE_FILE, "rb") as f:
        ram = pickle.load(f)
else:
    print("Building RAM cache…")
    ram = {p: Image.open(p).convert("RGB") for p in tqdm(all_paths, desc="RAM preload")}
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(ram, f)

# 2) Compute per-path filesize & top-60% flags
path2size = {p: p.stat().st_size for p in all_paths}
class2thr = {
    cls: np.percentile([path2size[p] for p in all_paths if p.parent.name==cls], 40)
    for cls in classes
}
path2good = {p: path2size[p]>=class2thr[p.parent.name] for p in all_paths}

# 3) Build items list
items = [(p, classes.index(p.parent.name)) for p in all_paths]
labels = np.array([lbl for _,lbl in items])

# ─── Dataset Classes ────────────────────────────────────────────────────────
class ConditionalClassSizeDataset(Dataset):
    """
    Top-60% per-class (by filesize) → tf_good; else tf_default.
    Applies vertical-only contour crop with prob.
    """
    def __init__(self, items, ram, path2good, tf_good, tf_default, crop_prob):
        self.items      = items
        self.ram        = ram
        self.path2good  = path2good
        self.tf_good    = tf_good
        self.tf_default = tf_default
        self.crop_prob  = crop_prob

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path,label = self.items[idx]
        img = self.ram[path]
        if self.crop_prob>0 and random.random()<self.crop_prob:
            img = detect_car_crop(img)
        tf = self.tf_good if self.path2good[path] else self.tf_default
        return tf(img), label

class PreloadDataset(Dataset):
    """Simple RAM dataset for validation (no good/small logic)."""
    def __init__(self, items, ram, transform, crop_prob=0.0):
        self.items     = items
        self.ram       = ram
        self.transform = transform
        self.crop_prob = crop_prob

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        path,label = self.items[idx]
        img = self.ram[path]
        if self.crop_prob>0 and random.random()<self.crop_prob:
            img = detect_car_crop(img)
        return self.transform(img), label

# ─── DataLoader Factory ────────────────────────────────────────────────────
def make_loaders(train_items, val_items,
                 tf_good, tf_default, tf_val,
                 batch_size, crop_prob):
    # train on good subset only, oversample via effective_weights
    ds = ConditionalClassSizeDataset(
        train_items, ram, path2good, tf_good, tf_default, crop_prob
    )
    w  = effective_weights(np.array([lbl for _,lbl in train_items]))

    # sampler = WeightedRandomSampler(w, len(w), replacement=True)
    train_loader = DataLoader(
        ds, batch_size=batch_size,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True
    )
    
    # val on full set
    ds_val = PreloadDataset(val_items, ram, tf_val, crop_prob=0.0)
    val_loader = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )
    return train_loader, val_loader

# ─── Training Loop Utilities ────────────────────────────────────────────────
def run_epoch(model, loader, optimizer, scaler, loss_fn,
              train=True, mixup_fn=None):
    if train:
        model.train()
    else:
        model.eval()
        # free any leftover cache
        torch.cuda.empty_cache()

    all_preds, all_trues = [], []

    # choose the right context
    no_grad_ctx = torch.no_grad() if not train else contextlib.nullcontext()
    for imgs, targets in tqdm(loader, desc="Train" if train else "Val ", leave=False):
        imgs, targets = imgs.cuda(), targets.cuda()
        batch_idxes   = targets.clone()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with no_grad_ctx, autocast(enabled=cfg.fp16):
            if train and mixup_fn:
                imgs_m, targets_m = mixup_fn(imgs, targets)
                logits = model(imgs_m)
                loss   = loss_fn(logits, targets_m)
            else:
                logits = model(imgs)
                one_hot = F.one_hot(targets, cfg.C).float()
                loss = loss_fn(logits, one_hot)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        preds_cpu = logits.detach().cpu().float()
        all_preds.append(preds_cpu)
        all_trues.append(batch_idxes.cpu())

    preds = torch.cat(all_preds).softmax(dim=1).numpy()
    trues = torch.cat(all_trues).numpy()
    return log_loss(trues, preds, labels=list(range(cfg.C)))


def train_fold(fold, train_items, val_items,
               size, bs, epochs, lr, crop_prob, tf_good, tf_default, tf_val, stage_name, init_state=None):

    log_dir = cfg.out_dir / f"fold{fold}" / stage_name
    writer  = SummaryWriter(log_dir)

    sample_loader = DataLoader(
        ConditionalClassSizeDataset(train_items, ram, path2good, tf_good, tf_default, crop_prob),
        batch_size=16, shuffle=True, num_workers=0
    )

    sample_imgs, _ = next(iter(sample_loader))
    writer.add_images("augmented_samples", sample_imgs, 0, dataformats="NCHW")

    print(f"\n=== Fold {fold} {stage_name} ===")
    # setup mixup/cutmix
    mixup_fn = timm.data.mixup.Mixup(
        mixup_alpha = cfg.mixup,
        cutmix_alpha = cfg.cutmix,
        label_smoothing = 0.0,
        num_classes = cfg.C)
    # model & EMA
    model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.C).cuda()
    if init_state: model.load_state_dict(init_state, strict=True)
    ema    = ModelEmaV2(model, decay=0.9999, device="cuda")
    loss_fn = SoftTargetCrossEntropy()
    opt     = AdamW(model.parameters(), lr=lr, weight_decay=cfg.wd)
    sched   = CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.eta_min)
    scaler  = GradScaler(enabled=cfg.fp16)

    train_loader, val_loader = make_loaders(
        train_items, val_items,
        tf_good, tf_default, tf_val,
        batch_size=bs, crop_prob=crop_prob
    )

    best, noimp = 1e9, 0
    ckpt = cfg.out_dir/f"fold{fold}_{stage_name}.pth"
    for ep in range(1, epochs+1):
        print(f"Epoch {ep}/{epochs}")
        trn = run_epoch(model, train_loader, opt, scaler, loss_fn, train=True, mixup_fn=mixup_fn)
        print(f"  [DEBUG] Stage {stage_name} Fold {fold} Epoch {ep}: train_loss={trn:.4f}")
        ema.update(model)
        val = run_epoch(ema.module, val_loader, opt, scaler, loss_fn, train=False)
        sched.step()
        print(f"{stage_name} F{fold}: trn={trn:.4f} val={val:.4f}")

        writer.add_scalars(
            "loss",
            {"train": trn, "val": val},
            ep
        )
        # record current LR
        writer.add_scalar("learning_rate", opt.param_groups[0]["lr"], ep)

        if val < best:
            best, noimp = val, 0
            torch.save(ema.module.state_dict(), ckpt)
        else:
            noimp += 1
            if noimp >= cfg.early_stop:
                print("Early stopping")
                break
    torch.cuda.empty_cache()
    writer.close()
    return ckpt

# ─── Main ───────────────────────────────────────────────────────────────────
def main(use_smoke=False, only_fold=None):

    if use_smoke:
        global items, labels, path2good
        items = items[:1000]
        labels = labels[:1000]
        # If you want to keep path2good in sync, you can:
        path2good = {p: path2good[p] for p,_ in items}
        print(f"Smoke‐mode: using {len(items)} samples")

    set_seed(cfg.seed)
    labels = np.array([lbl for _,lbl in items])
    skf = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)

    for fold, (tr,vl) in enumerate(skf.split(items, labels)):
        if only_fold is not None and fold != only_fold:
            continue
        train_it, val_it = [items[i] for i in tr], [items[i] for i in vl]

        # Stage 1: only good images
        good = [(p,l) for p,l in train_it if path2good[p]]
        print(f"Stage1: {len(good)} good samples")
        ckpt1 = train_fold(
            fold, good, val_it,
            cfg.size_s1, cfg.bs_s1, cfg.ep_s1, cfg.lr_s1, cfg.crop_prob_s1,
            tf_good_s1, tf_default_s1,  # train pipelines
            create_transform(          # val pipeline
              input_size=(3,cfg.size_s1,cfg.size_s1),
              is_training=False,
              interpolation=_conf["interpolation"],
              mean=_mean, std=_std
            ),
            'Stage1'
        )

        # Stage 2: all images
        state = torch.load(ckpt1, map_location='cpu')
        train2, val2 = train_it, val_it
        _ = train_fold(
            fold, train2, val2,
            cfg.size_s2, cfg.bs_s2, cfg.ep_s2, cfg.lr_s2, cfg.crop_prob_s2,
            tf_good_s2, tf_default_s2,
            create_transform(
              input_size=(3,cfg.size_s2,cfg.size_s2),
              is_training=False,
              interpolation=_conf["interpolation"],
              mean=_mean, std=_std
            ),
            'Stage2',
            init_state=state
        )

    print("All done.")

if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--smoke-test', action='store_true')
    p.add_argument('--smoke-data', action='store_true')
    p.add_argument('--fold', type=int, default=None,
                    help="If set, only train this fold index (0-based).")
    args = p.parse_args()
    if args.smoke_test:
        m = timm.create_model(cfg.model_name, pretrained=True).cuda()
        s = resolve_model_data_config(m)['input_size'][-1]
        print("OK", m(torch.randn(1,3,s,s).cuda()).shape)
    else:
        main(use_smoke=args.smoke_data, only_fold=args.fold)
