#!/usr/bin/env python3
"""
train_meta.py

End-to-end stacking pipeline using a CFG object:
1. Generate stratified K-fold splits (train_folds.csv) from your train directory
2. Generate OOF predictions for each base model (Stage1, Stage2)
3. Generate test-set predictions for each base model
4. Assemble meta-features and train a meta-learner (LogisticRegression or LightGBM)

Usage:
  python train_meta.py --stage all [--meta-model {lr,lgb}]
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
torch
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

# if using LightGBM
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

# ─── CONFIG ────────────────────────────────────────────────────────────────
class CFG:
    train_dir       = Path("./data/hecto/train")
    test_dir        = Path("./data/hecto/test")
    sample_csv      = Path("./data/hecto/sample_submission.csv")
    models_dir      = Path("./models")
    output_dir      = Path("meta_data")
    train_folds_csv = Path("train_folds.csv")

    n_folds     = 5
    seed        = 42
    batch_size  = 8
    num_workers = 4
    device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name  = "convnextv2_large.fcmae_ft_in22k_in1k_384"
    scales      = [(320,0.08),(352,0.08),(384,0.40),(416,0.12),(448,0.20),(480,0.12)]

    # Meta-model params
    lgb_params = {
        "objective": "multiclass",
        "num_class": None,  # set later
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }
    lgb_rounds = 500
    lgb_early_stop = 50

cfg = CFG()
cfg.output_dir.mkdir(exist_ok=True, parents=True)

# ─── FOLD GENERATION ────────────────────────────────────────────────────────
def create_folds():
    classes = sorted([d.name for d in cfg.train_dir.iterdir() if d.is_dir()])
    ids, labels = [], []
    for cls in classes:
        for img_path in (cfg.train_dir/cls).glob("*.jpg"):
            rel = img_path.relative_to(cfg.train_dir).as_posix()
            ids.append(rel)
            labels.append(classes.index(cls))
    df = pd.DataFrame({"id": ids, "label": labels})
    skf = StratifiedKFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(df, df.label)):
        df.loc[val_idx, "fold"] = fold
    df.to_csv(cfg.train_folds_csv, index=False)
    print(f"[folds] saved → {cfg.train_folds_csv}")

# ─── TRANSFORMS & DATASET (same as inference) ─────────────────────────────────
# ... (build_tta_transforms, InferenceDataset, load_model, infer_multi_scale) ...
# omitted for brevity; assume imported or defined above as before

# ─── PIPELINE STEPS ─────────────────────────────────────────────────────────

def generate_oof():
    df = pd.read_csv(cfg.train_folds_csv)
    for fold in range(cfg.n_folds):
        val = df[df.fold == fold]
        ids = val.id.tolist()
        y   = val.label.values
        o1 = infer_multi_scale(ids, cfg.train_dir,
                               sorted(cfg.models_dir.glob(f"convnext_stage1_fold{fold}*.pth")))
        o2 = infer_multi_scale(ids, cfg.train_dir,
                               sorted(cfg.models_dir.glob(f"convnext_stage2_fold{fold}*.pth")))
        np.save(cfg.output_dir/f"oof_s1_fold{fold}.npy", o1)
        np.save(cfg.output_dir/f"oof_s2_fold{fold}.npy", o2)
        np.save(cfg.output_dir/f"labels_fold{fold}.npy", y)
        print(f"[OOF] fold{fold} → {o1.shape}")


def generate_test():
    sub = pd.read_csv(cfg.sample_csv)
    ids = [f"{fn}.jpg" for fn in sub.iloc[:,0].astype(str)]
    t1 = infer_multi_scale(ids, cfg.test_dir,
                           sorted(cfg.models_dir.glob("convnext_stage1_fold*.pth")))
    t2 = infer_multi_scale(ids, cfg.test_dir,
                           sorted(cfg.models_dir.glob("convnext_stage2_fold*.pth")))
    np.save(cfg.output_dir/"test_s1.npy", t1)
    np.save(cfg.output_dir/"test_s2.npy", t2)
    print("[TEST] → test_s1.npy, test_s2.npy")


def train_meta(meta_model: str = "lr"):
    # load OOF features
    X_parts, y_parts = [], []
    for fold in range(cfg.n_folds):
        s1  = np.load(cfg.output_dir/f"oof_s1_fold{fold}.npy")
        s2  = np.load(cfg.output_dir/f"oof_s2_fold{fold}.npy")
        lbl = np.load(cfg.output_dir/f"labels_fold{fold}.npy")
        # meta-features: entropy & max-gap
        ent1 = -np.sum(s1 * np.log(s1 + 1e-12), axis=1, keepdims=True)
        ent2 = -np.sum(s2 * np.log(s2 + 1e-12), axis=1, keepdims=True)
        gap1 = np.diff(-np.partition(-s1, 1)[:, :2], axis=1)
        gap2 = np.diff(-np.partition(-s2, 1)[:, :2], axis=1)
        Xf = np.concatenate([s1, s2, ent1, ent2, gap1, gap2], axis=1)
        X_parts.append(Xf)
        y_parts.append(lbl)
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    print(f"[META] X={X.shape}, y={y.shape}   model={meta_model}")

    if meta_model == "lr":
        meta = LogisticRegression(multi_class="multinomial", solver="lbfgs",
                                  C=1.0, max_iter=1000)
        meta.fit(X, y)
        joblib.dump(meta, cfg.output_dir/"meta_lr.pkl")
        print("[META] saved → meta_lr.pkl")

    elif meta_model == "lgb":
        if lgb is None:
            raise ImportError("LightGBM not installed")
        cfg.lgb_params['num_class'] = X.shape[1]//2
        dtrain = lgb.Dataset(X, y)
        bst = lgb.train(
            cfg.lgb_params, dtrain,
            num_boost_round=cfg.lgb_rounds,
            valid_sets=[dtrain],
            early_stopping_rounds=cfg.lgb_early_stop,
            verbose_eval=False
        )
        bst.save_model(str(cfg.output_dir/"meta_lgb.txt"))
        print("[META] saved → meta_lgb.txt")

    else:
        raise ValueError("Unknown meta_model: choose 'lr' or 'lgb'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["folds", "oof", "test", "meta", "all"],
        default="all",
        help="where to start (default: all)"
    )
    parser.add_argument(
        "--meta-model", choices=["lr","lgb"], default="lr",
        help="meta learner: logistic regression or lightgbm"
    )
    args = parser.parse_args()

    if args.stage in ("folds", "all"):
        create_folds()
    if args.stage in ("oof", "all"):
        generate_oof()
    if args.stage in ("test", "all"):
        generate_test()
    if args.stage in ("meta", "all"):
        train_meta(args.meta_model)
