# train_efficientnetv2_two_stage.py

import os, random, math
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
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
    models_dir     = Path('./two_stage_models'); models_dir.mkdir(exist_ok=True)
    log_dir_base   = Path('./runs')
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name     = 'tf_efficientnetv2_m.in21k_ft_in1k'
    in_chans       = 3
    num_classes    = 0  # set at runtime

    n_folds        = 5
    selected_folds = list(range(5))
    seed           = 42

    s1_epochs      = 30
    s2_epochs      = 60
    batch_size     = 16
    num_workers    = 4

    optimizer      = 'AdamW'
    lr             = 2e-3
    weight_decay   = 1e-4
    scheduler      = 'CosineAnnealingLR'
    eta_min        = 1e-6

    mixup_alpha    = 0.5
    cutmix_alpha   = 1.0

    early_stop     = 10
    use_weighted_sampler = True

cfg = CFG()
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)
random.seed(cfg.seed)
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
    if w*h < 0.05*img_bgr.shape[0]*img_bgr.shape[1]:
        return img_bgr
    pad=10
    x1,y1 = max(x-pad,0), max(y-pad,0)
    x2,y2 = min(x+w+pad,img_bgr.shape[1]-1), min(y+h+pad,img_bgr.shape[0]-1)
    return img_bgr[y1:y2, x1:x2]

def compute_class_weights(labels, n_cls):
    cnt = np.bincount(labels, minlength=n_cls)
    cnt[cnt==0]=1
    inv = 1.0/cnt
    return [inv[l] for l in labels]

def cutmix(x,y,alpha=1.0):
    if alpha<=0: return x,y,None,1.0
    lam = np.random.beta(alpha,alpha)
    bs,_,H,W = x.size()
    idx = torch.randperm(bs, device=x.device)
    y2  = y[idx]
    rx,ry = np.random.randint(W),np.random.randint(H)
    rw,rh = int(W*math.sqrt(1-lam)), int(H*math.sqrt(1-lam))
    x1,x2 = np.clip(rx-rw//2,0,W), np.clip(rx+rw//2,0,W)
    y1,y2_ = np.clip(ry-rh//2,0,H), np.clip(ry+rh//2,0,H)
    x[:,:,y1:y2_,x1:x2] = x[idx,:,y1:y2_,x1:x2]
    lam = 1 - ((x2-x1)*(y2_-y1)/(W*H))
    return x,y,y2,lam

class HectoNet(nn.Module):
    def __init__(self,model_name,in_chans,num_classes,mixup_alpha=0.0):
        super().__init__()
        self.mixup_alpha = mixup_alpha
        self.backbone = timm.create_model(model_name,pretrained=True,
                                          in_chans=in_chans,
                                          num_classes=num_classes)
    def forward(self,x,y=None):
        if self.training and self.mixup_alpha>0 and y is not None:
            lam = np.random.beta(self.mixup_alpha,self.mixup_alpha)
            idx = torch.randperm(x.size(0),device=x.device)
            x_m = lam*x + (1-lam)*x[idx]
            y1,y2 = y,y[idx]
            logits = self.backbone(x_m)
            loss = lam*F.cross_entropy(logits,y1)+(1-lam)*F.cross_entropy(logits,y2)
            return logits, loss
        return self.backbone(x)

# ─── MAIN ───────────────────────────────────────────────────────────────────
def run_two_stage():
    # 1) build dataset listing
    classes = sorted([d.name for d in Path(cfg.train_dir).iterdir() if d.is_dir()])
    cfg.num_classes = len(classes)
    cls2idx = {c:i for i,c in enumerate(classes)}
    all_items=[]
    for c in classes:
        for f in sorted((Path(cfg.train_dir)/c).glob('*')):
            if f.suffix.lower() in {'.jpg','.jpeg','.png'}:
                all_items.append((f,cls2idx[c]))

    # 2) preload into RAM
    print("Preloading images into RAM…")
    ram_raw  = {}
    ram_crop = {}
    m = timm.create_model(cfg.model_name, pretrained=True)
    dc = resolve_model_data_config(m)
    # derive test‐resize
    resize_tf = transforms.Compose([
        transforms.Resize(int(dc['input_size'][-1]*1.05)),
        transforms.CenterCrop(dc['input_size'][-1])
    ])
    for p,_ in tqdm(all_items,desc="RAM preload"):
        img = Image.open(p).convert('RGB')
        ram_raw[p] = resize_tf(img)
        bgr = cv2.imread(str(p))
        crop = detect_car_by_largest_contour(bgr)
        crop_rgb = cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        ram_crop[p] = resize_tf(Image.fromarray(crop_rgb))

    # 3) build folds
    labels = np.array([lbl for _,lbl in all_items])
    folds=[]
    skf = StratifiedKFold(n_splits=cfg.n_folds,shuffle=True,random_state=cfg.seed)
    for fi,(tr,vl) in enumerate(skf.split(labels,labels)):
        if fi in cfg.selected_folds:
            folds.append(([all_items[i] for i in tr],[all_items[i] for i in vl]))

    # 4) train per‐fold
    for fold,(tr_items,vl_items) in enumerate(folds):
        print(f"\n=== Fold {fold} ===")
        # transforms
        m = timm.create_model(cfg.model_name, pretrained=True)
        dc = resolve_model_data_config(m)
        t_train = create_transform(**dc,is_training=True)
        t_val   = create_transform(**dc,is_training=False)

        # datasets
        class PreloadCrop(Dataset):
            def __init__(self,items,tf): self.items, self.tf = items,tf
            def __len__(self): return len(self.items)
            def __getitem__(self,i):
                p,l = self.items[i]
                return self.tf(ram_crop[p]), l

        class PreloadRaw(Dataset):
            def __init__(self,items,tf): self.items, self.tf = items,tf
            def __len__(self): return len(self.items)
            def __getitem__(self,i):
                p,l = self.items[i]
                return self.tf(ram_raw[p]), l

        ds_tr1 = PreloadCrop(tr_items, t_train)
        ds_vl1 = PreloadCrop(vl_items, t_val)
        if cfg.use_weighted_sampler:
            w1 = compute_class_weights([l for _,l in tr_items], cfg.num_classes)
            sampler1 = WeightedRandomSampler(w1,len(w1),replacement=True)
            ld_tr1 = DataLoader(ds_tr1,batch_size=cfg.batch_size,
                                sampler=sampler1,
                                num_workers=cfg.num_workers,pin_memory=True)
        else:
            ld_tr1 = DataLoader(ds_tr1,batch_size=cfg.batch_size,shuffle=True,
                                num_workers=cfg.num_workers,pin_memory=True)
        ld_vl1 = DataLoader(ds_vl1,batch_size=cfg.batch_size,shuffle=False,
                            num_workers=cfg.num_workers,pin_memory=True)

        # model, optim, sched
        model = HectoNet(cfg.model_name,cfg.in_chans,cfg.num_classes,
                         mixup_alpha=cfg.mixup_alpha).to(cfg.device)
        opt   = getattr(optim,cfg.optimizer)(model.parameters(),
                                             lr=cfg.lr,weight_decay=cfg.weight_decay)
        if cfg.scheduler=='CosineAnnealingLR':
            sched = lr_scheduler.CosineAnnealingLR(opt,
                        T_max=cfg.s1_epochs,eta_min=cfg.eta_min)
        else:
            sched = lr_scheduler.StepLR(opt,step_size=max(1,cfg.s1_epochs//3),gamma=0.5)

        writer   = SummaryWriter(cfg.log_dir_base/f'fold{fold}/stage1')
        best_ll1 = float('inf'); no_imp1=0
        best_p1  = cfg.models_dir/f'fold{fold}_stage1_best.pth'

        for ep in range(cfg.s1_epochs):
            model.train()
            acc_p,acc_t=[],[]
            for x,y in tqdm(ld_tr1,desc=f"S1 F{fold} Ep{ep+1}"):
                x,y = x.to(cfg.device), y.to(cfg.device)
                opt.zero_grad()
                logits,loss = model(x,y)
                loss.backward(); opt.step()
                acc_p.append(F.softmax(logits,1).detach().cpu().numpy())
                acc_t.append(y.cpu().numpy())
            train_ll = log_loss(np.concatenate(acc_t),
                                np.vstack(acc_p).clip(1e-7,1-1e-7),
                                labels=list(range(cfg.num_classes)))

            # val
            model.eval()
            vp,vt=[],[]
            with torch.no_grad():
                for x,y in ld_vl1:
                    x,y=x.to(cfg.device),y.to(cfg.device)
                    out = model(x)
                    vp.append(F.softmax(out,1).cpu().numpy())
                    vt.append(y.cpu().numpy())
            val_ll = log_loss(np.concatenate(vt),
                              np.vstack(vp).clip(1e-7,1-1e-7),
                              labels=list(range(cfg.num_classes)))

            writer.add_scalars('loss',{'train':train_ll,'val':val_ll},ep)
            print(f"[S1][F{fold} Ep{ep+1}] Tr={train_ll:.4f} Val={val_ll:.4f}")

            if val_ll<best_ll1:
                best_ll1,no_imp1=val_ll,0
                torch.save(model.state_dict(),best_p1)
            else:
                no_imp1+=1
                if no_imp1>=cfg.early_stop: break
            sched.step()

        writer.close()
        torch.cuda.empty_cache()
        model.load_state_dict(torch.load(best_p1,map_location=cfg.device))

        # ─── Stage2 ───────────────────────────────────────────────────
        ds_tr2 = PreloadRaw(tr_items, t_train)
        ds_vl2 = PreloadRaw(vl_items, t_val)
        if cfg.use_weighted_sampler:
            w2 = compute_class_weights([l for _,l in tr_items], cfg.num_classes)
            sampler2 = WeightedRandomSampler(w2,len(w2),replacement=True)
            ld_tr2 = DataLoader(ds_tr2,batch_size=cfg.batch_size,
                                sampler=sampler2,
                                num_workers=cfg.num_workers,pin_memory=True)
        else:
            ld_tr2 = DataLoader(ds_tr2,batch_size=cfg.batch_size,shuffle=True,
                                num_workers=cfg.num_workers,pin_memory=True)
        ld_vl2 = DataLoader(ds_vl2,batch_size=cfg.batch_size,shuffle=False,
                            num_workers=cfg.num_workers,pin_memory=True)

        opt   = getattr(optim,cfg.optimizer)(model.parameters(),
                                             lr=cfg.lr/5,weight_decay=cfg.weight_decay)
        if cfg.scheduler=='CosineAnnealingLR':
            sched = lr_scheduler.CosineAnnealingLR(opt,
                        T_max=cfg.s2_epochs,eta_min=cfg.eta_min)
        else:
            sched = lr_scheduler.StepLR(opt,step_size=max(1,cfg.s2_epochs//3),gamma=0.5)

        writer   = SummaryWriter(cfg.log_dir_base/f'fold{fold}/stage2')
        best_ll2 = float('inf'); no_imp2=0
        best_p2  = cfg.models_dir/f'fold{fold}_stage2_best.pth'

        for ep in range(cfg.s2_epochs):
            model.train()
            acc_p,acc_t=[],[]
            for x,y in tqdm(ld_tr2,desc=f"S2 F{fold} Ep{ep+1}"):
                x,y = x.to(cfg.device), y.to(cfg.device)
                opt.zero_grad()
                if random.random()<0.5:
                    x2,y1,y2,lam = cutmix(x,y,cfg.cutmix_alpha)
                    logits = model.backbone(x2)
                    loss   = lam*F.cross_entropy(logits,y1)+(1-lam)*F.cross_entropy(logits,y2)
                else:
                    logits,loss = model(x,y)
                loss.backward(); opt.step()
                acc_p.append(F.softmax(logits,1).detach().cpu().numpy())
                acc_t.append(y.cpu().numpy())
            train_ll = log_loss(np.concatenate(acc_t),
                                np.vstack(acc_p).clip(1e-7,1-1e-7),
                                labels=list(range(cfg.num_classes)))

            # val
            model.eval()
            vp,vt=[],[]
            with torch.no_grad():
                for x,y in ld_vl2:
                    x,y=x.to(cfg.device),y.to(cfg.device)
                    out = model(x)
                    vp.append(F.softmax(out,1).cpu().numpy())
                    vt.append(y.cpu().numpy())
            val_ll = log_loss(np.concatenate(vt),
                              np.vstack(vp).clip(1e-7,1-1e-7),
                              labels=list(range(cfg.num_classes)))

            writer.add_scalars('loss',{'train':train_ll,'val':val_ll},ep)
            print(f"[S2][F{fold} Ep{ep+1}] Tr={train_ll:.4f} Val={val_ll:.4f}")

            if val_ll<best_ll2:
                best_ll2,no_imp2=val_ll,0
                torch.save(model.state_dict(),best_p2)
            else:
                no_imp2+=1
                if no_imp2>=cfg.early_stop: break
            sched.step()

        writer.close()
        torch.cuda.empty_cache()

    print("All folds done.")

if __name__=='__main__':
    run_two_stage()
