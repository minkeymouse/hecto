#!/usr/bin/env python3
# inference_ensemble_three_archs.py
# Python ≥3.9, Torch ≥1.13, timm ≥0.9

import unicodedata
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import timm
from timm.data import resolve_model_data_config, create_transform

# ─── CONFIG ────────────────────────────────────────────────────────────────────
class CFG:
    train_dir           = Path("./data/hecto/train")
    test_dir            = Path("./data/hecto/test")
    sample_csv          = Path("./data/hecto/sample_submission.csv")

    efficientnet_models = Path("./models/efficientnet")
    convnext_models     = Path("./models/convnext")
    regnety_models      = Path("./models/regnety")

    model_names         = [
        "tf_efficientnetv2_m.in21k_ft_in1k",
        "convnext_small.in12k_ft_in1k_384",
        "regnety_320.swag_ft_in1k",
    ]
    arch_weights        = [0.2, 0.3, 0.5]        # will be re-normalised

    in_chans            = 3
    batch_size          = 64
    num_workers         = 4
    device              = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── SEED ───────────────────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# ─── UNICODE NORMALIZATION ─────────────────────────────────────────────────────
def norm(s: str) -> str:
    """Normalize to NFC, strip whitespace."""
    return unicodedata.normalize("NFC", s).strip()

# ─── MODEL WRAPPER ─────────────────────────────────────────────────────────────
class HectoNet(nn.Module):
    def __init__(self, model_name: str, in_chans: int, n_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            in_chans=in_chans,
            num_classes=n_classes
        )
    def forward(self, x):
        return self.backbone(x)

# ─── DATASET ───────────────────────────────────────────────────────────────────
class TestDataset(Dataset):
    def __init__(self, img_paths):
        self.img_paths = img_paths
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx):
        p   = self.img_paths[idx]
        img = Image.open(p).convert("RGB")
        return img, p.stem

# ─── TRANSFORM FACTORY ─────────────────────────────────────────────────────────
def get_val_transform(model_name):
    dummy = timm.create_model(model_name, pretrained=False)
    dc    = resolve_model_data_config(dummy)
    tf    = create_transform(**dc, is_training=False)
    return tf, dc["input_size"]

# ─── ALIAS LABELS ──────────────────────────────────────────────────────────────
ALIAS_PAIRS = [
    ('K5_3세대_하이브리드_2020_2022', 'K5_하이브리드_3세대_2020_2023'),
    ('디_올뉴니로_2022_2025',        '디_올_뉴_니로_2022_2025'),
    ('718_박스터_2017_2024',        '박스터_718_2017_2024'),
    ('RAV4_2016_2018',             '라브4_4세대_2013_2018'),
    ('RAV4_5세대_2019_2024',       '라브4_5세대_2019_2024'),
]

# ─── INFERENCE ─────────────────────────────────────────────────────────────────
def main() -> None:
    cfg = CFG()

    # 1) submission skeleton & basic meta
    sub      = pd.read_csv(cfg.sample_csv)
    id_col   = sub.columns[0]
    cls_cols = sub.columns[1:].tolist()
    ids      = sub[id_col].astype(str).tolist()
    N        = len(ids)
    C        = len(cls_cols)

    # 2) normalize & sanity-check classes
    #    CSV labels → normalized set
    cls_norm       = [norm(c) for c in cls_cols]
    cls_norm_set   = set(cls_norm)
    #    train folders → normalized set
    raw_train_dirs = [d.name for d in cfg.train_dir.iterdir() if d.is_dir()]
    train_norm     = [norm(d) for d in raw_train_dirs]
    train_norm_set = set(train_norm)

    missing = cls_norm_set - train_norm_set
    extra   = train_norm_set - cls_norm_set
    if missing or extra:
        print("→ In CSV but not in train dirs:", missing)
        print("→ In train dirs but not in CSV:", extra)
        raise RuntimeError("Classes mismatch—even after normalization")

    #    Build ordered list of *actual* folder names matching CSV order
    train_dirs_ordered = [
        next(d for d in raw_train_dirs if norm(d) == n)
        for n in cls_norm
    ]

    # 3) id → path map
    test_map = {p.stem: p for p in cfg.test_dir.glob("*")}
    if missing := set(ids) - set(test_map.keys()):
        raise RuntimeError(f"Missing test images: {missing}")
    test_paths = [test_map[i] for i in ids]

    # 4) loader (keeps submission order)
    dl_test = DataLoader(
        TestDataset(test_paths),
        batch_size   = cfg.batch_size,
        shuffle      = False,
        num_workers  = cfg.num_workers,
        pin_memory   = True,
        collate_fn   = lambda batch: ([img for img, _ in batch], None)
    )

    def softmax_np(x: np.ndarray) -> np.ndarray:
        x = x - x.max(axis=1, keepdims=True)
        np.exp(x, out=x)
        x /= x.sum(axis=1, keepdims=True)
        return x

    # 5) per-architecture loop
    arch_logits = []
    for model_name, arch_w in zip(cfg.model_names, cfg.arch_weights):
        if "efficientnet" in model_name:
            model_dir, prefix = cfg.efficientnet_models, "efficientnet"
        elif "convnext" in model_name:
            model_dir, prefix = cfg.convnext_models,     "convx"
        else:
            model_dir, prefix = cfg.regnety_models,      "regnety"

        ckpts = sorted(model_dir.glob(f"{prefix}_fold*.pth"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints '{prefix}_fold*.pth' in {model_dir}")

        val_tf, (C_in, H, W) = get_val_transform(model_name)
        tta_fns = [
            lambda img, tf=val_tf: tf(img),
            lambda img, tf=val_tf: tf(TF.hflip(img)),
            lambda img, tf=val_tf,H=H,W=W:
                tf(TF.center_crop(TF.resize(img, (int(H*1.15), int(W*1.15))),
                                 (int(H*0.875), int(W*0.875)))),
            lambda img, tf=val_tf: tf(TF.rotate(img,  5)),
            lambda img, tf=val_tf: tf(TF.rotate(img, -5)),
            lambda img, tf=val_tf: tf(TF.adjust_brightness(img, 1.15)),
        ]
        tta_w = 1.0 / len(tta_fns)

        fold_log_probs = []
        for ckpt in ckpts:
            net = HectoNet(model_name, cfg.in_chans, C)
            net.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
            net.to(cfg.device).eval()

            logp = np.empty((N, C), dtype=np.float32)
            ptr  = 0
            with torch.no_grad():
                for imgs, _ in tqdm(dl_test, desc=f"{prefix}|{ckpt.name}", leave=False):
                    bs = len(imgs)
                    acc = torch.zeros(bs, C, device=cfg.device)
                    for tfm in tta_fns:
                        batch = torch.stack([tfm(img) for img in imgs]).to(cfg.device)
                        acc += tta_w * net(batch)
                    lp = F.log_softmax(acc, dim=1).cpu().numpy()
                    logp[ptr:ptr+bs] = lp
                    ptr += bs

            fold_log_probs.append(logp)

        arch_logits.append(np.mean(fold_log_probs, axis=0))

    # 6) architecture-level weighted ensemble
    aw = np.array(cfg.arch_weights, dtype=np.float32)
    aw /= aw.sum()
    ens_logits = sum(w * l for w, l in zip(aw, arch_logits))
    final_p    = softmax_np(ens_logits.copy())

    # 7) merge alias classes
    for a, b in ALIAS_PAIRS:
        if a in cls_cols and b in cls_cols:
            ia, ib = cls_cols.index(a), cls_cols.index(b)
            final_p[:, ia] += final_p[:, ib]
            final_p[:, ib]  = final_p[:, ia]

    # 8) save
    sub.loc[:, cls_cols] = final_p
    out_path = Path("submission.csv")
    sub.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] saved → {out_path.resolve()}")

if __name__ == "__main__":
    main()
