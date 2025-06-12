#!/usr/bin/env python3
# train_regnety_160_two_stage.py

import os, random, math
from pathlib import Path

import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import timm
from timm.data import resolve_model_data_config, create_transform

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# ─── CONFIG ─────────────────────────────────────────────────────────────────
class CFG:
    train_dir      = '/data/hecto/train'
    models_dir     = Path('./two_stage_models_regnety160'); models_dir.mkdir(exist_ok=True)
    log_dir_base   = Path('./runs_regnety160');      log_dir_base.mkdir(exist_ok=True)
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name     = 'regnety_320.swag_ft_in1k'
    n_folds        = 5
    seed           = 42

    s1_epochs      = 30
    s2_epochs      = 60
    batch_size     = 24
    num_workers    = 4

    lr             = 2e-3
    weight_decay   = 1e-4
    scheduler      = 'CosineAnnealingLR'
    eta_min        = 1e-6

    mixup_alpha    = 0.5
    cutmix_alpha   = 1.0

    early_stop     = 10
    use_weighted_sampler = True

cfg = CFG()
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─── HELPERS ────────────────────────────────────────────────────────────────
def detect_car_by_largest_contour(img_bgr, t1=50, t2=150):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges= cv2.Canny(blur, t1, t2)
    cnts,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return img_bgr
    c = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    area = img_bgr.shape[0]*img_bgr.shape[1]
    if w*h < 0.05 * area: return img_bgr
    pad = 10
    x1,y1 = max(x-pad,0), max(y-pad,0)
    x2,y2 = min(x+w+pad,img_bgr.shape[1]-1), min(y+h+pad,img_bgr.shape[0]-1)
    return img_bgr[y1:y2, x1:x2]

def compute_class_weights(labels, n_cls):
    cnt = np.bincount(labels, minlength=n_cls)
    cnt[cnt==0] = 1
    inv = 1.0 / cnt
    return [inv[l] for l in labels]

def mixup_data(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        idx = torch.randperm(x.size(0), device=x.device)
        return lam*x + (1-lam)*x[idx], y, y[idx], lam
    return x, y, y, 1.0

def cutmix_data(x, y, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        bs,_,H,W = x.size()
        idx = torch.randperm(bs, device=x.device)
        rx,ry = np.random.randint(W), np.random.randint(H)
        rw, rh = int(W*math.sqrt(1-lam)), int(H*math.sqrt(1-lam))
        x1,x2 = np.clip(rx-rw//2,0,W), np.clip(rx+rw//2,0,W)
        y1,y2 = np.clip(ry-rh//2,0,H), np.clip(ry+rh//2,0,H)
        x[:,:,y1:y2,x1:x2] = x[idx,:,y1:y2,x1:x2]
        lam = 1 - ((x2-x1)*(y2-y1)/(W*H))
        return x, y, y[idx], lam
    return x, y, y, 1.0

# ─── DATASET ─────────────────────────────────────────────────────────────────
class PreloadDataset(Dataset):
    def __init__(self, items, ram_dict, transform):
        self.items = items
        self.ram   = ram_dict
        self.tf    = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        p,l = self.items[idx]
        return self.tf(self.ram[p]), l

# ─── MODEL WRAPPER ───────────────────────────────────────────────────────────
class HectoNet(nn.Module):
    def __init__(self, model_name, num_classes, mixup_alpha):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=num_classes)
    def forward(self, x, y=None):
        if self.training and self.mixup_alpha>0 and y is not None:
            x_m, y1, y2, lam = mixup_data(x, y, self.mixup_alpha)
            logits = self.backbone(x_m)
            loss   = lam*F.cross_entropy(logits, y1) + (1-lam)*F.cross_entropy(logits, y2)
            return logits, loss
        logits = self.backbone(x)
        if y is not None:
            return logits, F.cross_entropy(logits, y)
        return logits

# ─── TRAINING PIPELINE ──────────────────────────────────────────────────────
def run_two_stage():
    # 1) gather all paths & labels
    classes = sorted(d.name for d in Path(cfg.train_dir).iterdir() if d.is_dir())
    cls2idx = {c:i for i,c in enumerate(classes)}
    all_items = [(p, cls2idx[p.parent.name])
                 for c in classes
                 for p in (Path(cfg.train_dir)/c).glob('*')
                 if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    cfg.num_classes = len(classes)

    # 2) preload images (raw & cropped)
    print("RAM preload…")
    ram_raw, ram_crop = {}, {}
    tmp = timm.create_model(cfg.model_name, pretrained=True)
    dc  = resolve_model_data_config(tmp)
    resize = transforms.Compose([
        transforms.Resize(int(dc['input_size'][-1]*1.05)),
        transforms.CenterCrop(dc['input_size'][-1])
    ])
    for p,_ in tqdm(all_items, desc="Preload"):
        img = Image.open(p).convert('RGB')
        ram_raw[p] = resize(img)
        crop = detect_car_by_largest_contour(cv2.imread(str(p)))
        ram_crop[p] = resize(Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)))

    # 3) stratified K-fold
    labels = np.array([lbl for _,lbl in all_items])
    skf    = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)

    for fold, (tr_idx, vl_idx) in enumerate(skf.split(all_items, labels)):
        print(f"\n=== Fold {fold} ===")
        tr_items = [all_items[i] for i in tr_idx]
        vl_items = [all_items[i] for i in vl_idx]

        # get transforms
        m2    = timm.create_model(cfg.model_name, pretrained=True)
        dc2   = resolve_model_data_config(m2)
        tf_tr = create_transform(**dc2, is_training=True)
        tf_vl = create_transform(**dc2, is_training=False)

        # ── Stage 1: cropped ────────────────────────────
        ds_tr1 = PreloadDataset(tr_items, ram_crop, tf_tr)
        ds_vl1 = PreloadDataset(vl_items, ram_crop, tf_vl)
        if cfg.use_weighted_sampler:
            w1       = compute_class_weights([l for _,l in tr_items], cfg.num_classes)
            sampler1 = WeightedRandomSampler(w1, len(w1), replacement=True)
            ld_tr1   = DataLoader(ds_tr1, batch_size=cfg.batch_size,
                                  sampler=sampler1, num_workers=cfg.num_workers, pin_memory=True)
        else:
            ld_tr1 = DataLoader(ds_tr1, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=True)
        ld_vl1 = DataLoader(ds_vl1, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

        model = HectoNet(cfg.model_name, cfg.num_classes, cfg.mixup_alpha).to(cfg.device)
        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        scheduler = (lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.s1_epochs, eta_min=cfg.eta_min)
                     if cfg.scheduler=='CosineAnnealingLR'
                     else lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.s1_epochs//3), gamma=0.5))
        scaler    = GradScaler()
        writer    = SummaryWriter(cfg.log_dir_base/f'fold{fold}/stage1')
        best_ll, no_imp = float('inf'), 0
        ckpt1 = cfg.models_dir/f'fold{fold}_stage1.pth'

        for ep in range(cfg.s1_epochs):
            # — train —
            model.train()
            preds, trues = [], []
            for x,y in tqdm(ld_tr1, desc=f"S1 F{fold} Ep{ep+1}"):
                x,y = x.to(cfg.device), y.to(cfg.device)
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    logits, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                p = F.softmax(logits, 1)
                p = torch.clamp(p, 1e-7, 1-1e-7)
                p = torch.nan_to_num(p, nan=1e-7, posinf=1-1e-7, neginf=1e-7)
                preds.append(p.detach().cpu().numpy())
                trues.append(y.cpu().numpy())

            # — step scheduler once per epoch —
            scheduler.step()

            # — compute train log-loss —
            train_ll = log_loss(
                np.concatenate(trues),
                np.vstack(preds).clip(1e-7,1-1e-7),
                labels=list(range(cfg.num_classes))
            )

            # — validate —
            model.eval()
            vpreds, vtrues = [], []
            with torch.no_grad():
                for x,y in ld_vl1:
                    x,y = x.to(cfg.device), y.to(cfg.device)
                    with autocast(device_type='cuda'):
                        logits = model(x)
                    p = F.softmax(logits, 1)
                    p = torch.clamp(p, 1e-7, 1-1e-7)
                    p = torch.nan_to_num(p, nan=1e-7, posinf=1-1e-7, neginf=1e-7)
                    vpreds.append(p.detach().cpu().numpy())
                    vtrues.append(y.cpu().numpy())
            val_ll = log_loss(
                np.concatenate(vtrues),
                np.vstack(vpreds).clip(1e-7,1-1e-7),
                labels=list(range(cfg.num_classes))
            )

            writer.add_scalars('loss', {'train':train_ll, 'val':val_ll}, ep)
            print(f"[S1][F{fold} Ep{ep+1}] Tr={train_ll:.4f} Val={val_ll:.4f}")

            if val_ll < best_ll:
                best_ll, no_imp = val_ll, 0
                torch.save(model.state_dict(), ckpt1)
            else:
                no_imp += 1
                if no_imp >= cfg.early_stop:
                    break

        writer.close()
        model.load_state_dict(torch.load(ckpt1, map_location=cfg.device))
        torch.cuda.empty_cache()

        # ── Stage 2: raw ────────────────────────────────
        ds_tr2 = PreloadDataset(tr_items, ram_raw, tf_tr)
        ds_vl2 = PreloadDataset(vl_items, ram_raw, tf_vl)
        if cfg.use_weighted_sampler:
            w2       = compute_class_weights([l for _,l in tr_items], cfg.num_classes)
            sampler2 = WeightedRandomSampler(w2, len(w2), replacement=True)
            ld_tr2   = DataLoader(ds_tr2, batch_size=cfg.batch_size,
                                  sampler=sampler2, num_workers=cfg.num_workers, pin_memory=True)
        else:
            ld_tr2 = DataLoader(ds_tr2, batch_size=cfg.batch_size, shuffle=True,
                                num_workers=cfg.num_workers, pin_memory=True)
        ld_vl2 = DataLoader(ds_vl2, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr/5, weight_decay=cfg.weight_decay)
        scheduler2 = (lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.s2_epochs, eta_min=cfg.eta_min)
                      if cfg.scheduler=='CosineAnnealingLR'
                      else lr_scheduler.StepLR(optimizer, step_size=max(1, cfg.s2_epochs//3), gamma=0.5))
        scaler     = GradScaler()
        writer     = SummaryWriter(cfg.log_dir_base/f'fold{fold}/stage2')
        best_ll, no_imp = float('inf'), 0
        ckpt2 = cfg.models_dir/f'fold{fold}_stage2.pth'

        for ep in range(cfg.s2_epochs):
            model.train()
            preds, trues = [], []
            for x,y in tqdm(ld_tr2, desc=f"S2 F{fold} Ep{ep+1}"):
                x,y = x.to(cfg.device), y.to(cfg.device)
                optimizer.zero_grad()
                with autocast(device_type='cuda'):
                    if random.random() < 0.5:
                        x2,y1,y2,lam = cutmix_data(x, y, cfg.cutmix_alpha)
                        logits = model.backbone(x2)
                        loss   = lam*F.cross_entropy(logits, y1) + (1-lam)*F.cross_entropy(logits, y2)
                    else:
                        logits, loss = model(x, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                preds.append(F.softmax(logits,1).detach().cpu().numpy())
                trues.append(y.cpu().numpy())

            # scheduler step
            scheduler2.step()

            train_ll = log_loss(
                np.concatenate(trues),
                np.vstack(preds).clip(1e-7,1-1e-7),
                labels=list(range(cfg.num_classes))
            )

            model.eval()
            vpreds, vtrues = [], []
            with torch.no_grad():
                for x,y in ld_vl2:
                    x,y = x.to(cfg.device), y.to(cfg.device)
                    with autocast(device_type='cuda'):
                        logits = model(x)
                    vpreds.append(F.softmax(logits,1).cpu().numpy())
                    vtrues.append(y.cpu().numpy())
            val_ll = log_loss(
                np.concatenate(vtrues),
                np.vstack(vpreds).clip(1e-7,1-1e-7),
                labels=list(range(cfg.num_classes))
            )

            writer.add_scalars('loss', {'train':train_ll, 'val':val_ll}, ep)
            print(f"[S2][F{fold} Ep{ep+1}] Tr={train_ll:.4f} Val={val_ll:.4f}")

            if val_ll < best_ll:
                best_ll, no_imp = val_ll, 0
                torch.save(model.state_dict(), ckpt2)
            else:
                no_imp += 1
                if no_imp >= cfg.early_stop:
                    break

        writer.close()
        torch.cuda.empty_cache()

    print("All folds done.")

if __name__=='__main__':
    run_two_stage()
