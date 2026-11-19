"""Training loop with mixed precision, class weights, best-model saving and auto-resume."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Config
from .data import read_targets
from .model import build_model, make_dataset, save_checkpoint


def _confusion(pred, tgt, n=3):
    import torch

    k = (tgt >= 0) & (tgt < n)
    return torch.bincount(n * tgt[k] + pred[k], minlength=n * n).reshape(n, n)


def _iou(cm):
    cm = cm.double()
    inter = cm.diag()
    return (inter / (cm.sum(0) + cm.sum(1) - inter).clamp(min=1)).cpu().numpy()


def train(cfg: Config, meta: pd.DataFrame, log=print) -> dict:
    import segmentation_models_pytorch as smp
    import torch
    import torch.nn as nn

    tc, sc = cfg.train, cfg.segmentation
    torch.manual_seed(tc.seed)
    np.random.seed(tc.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    out = Path(tc.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / f"{sc.arch}_{sc.encoder}_3class.pt"
    last_path = out / f"{sc.arch}_{sc.encoder}_3class_last.pt"

    tr_df, va_df = meta[meta.split == "train"], meta[meta.split == "val"]
    kw = dict(batch_size=tc.batch_size, num_workers=tc.num_workers, pin_memory=amp)
    tr_dl = torch.utils.data.DataLoader(make_dataset(tr_df, sc, True), shuffle=True, drop_last=True, **kw)
    va_dl = torch.utils.data.DataLoader(make_dataset(va_df, sc), shuffle=False, **kw)

    cnt = np.zeros(3)
    for p in tr_df.msk.sample(min(100, len(tr_df)), random_state=tc.seed):
        cnt += np.bincount(read_targets(p)[0].ravel(), minlength=3)
    w = torch.tensor((cnt.sum() / (3 * np.maximum(cnt, 1))) ** 0.5, dtype=torch.float32, device=device)
    log(f"device={device} | train={len(tr_df)} val={len(va_df)} | class share={np.round(cnt / cnt.sum(), 3)}")

    model = build_model(sc).to(device)
    if tc.init_checkpoint:
        ck0 = torch.load(tc.init_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck0["model"])
        log(f"fine-tuning from {tc.init_checkpoint}")
    ce, dice = nn.CrossEntropyLoss(weight=w), smp.losses.DiceLoss("multiclass")
    opt = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=tc.lr, total_steps=tc.epochs * max(1, len(tr_dl)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, hist = 0, -1.0, []
    if tc.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if ck["epoch"] + 1 < tc.epochs:
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
        start, best, hist = ck["epoch"] + 1, ck["best"], ck["history"]
        log(f"resumed at epoch {start} (best mIoU {best:.3f})")

    for ep in range(start, tc.epochs):
        model.train()
        t0, tl = time.time(), 0.0
        for x, y in tr_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
                o = model(x)
                loss = ce(o, y) + dice(o, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tl += loss.item()
        model.eval()
        cm = torch.zeros(3, 3, dtype=torch.long, device=device)
        vl = 0.0
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            for x, y in va_dl:
                x, y = x.to(device), y.to(device)
                o = model(x).float()
                vl += (ce(o, y) + dice(o, y)).item()
                cm += _confusion(o.argmax(1).flatten(), y.flatten())
        ious = _iou(cm)
        miou = float(ious.mean())
        hist.append(
            {
                "epoch": ep,
                "train_loss": tl / max(1, len(tr_dl)),
                "val_loss": vl / max(1, len(va_dl)),
                "iou_bg": float(ious[0]),
                "iou_field": float(ious[1]),
                "iou_boundary": float(ious[2]),
                "miou": miou,
            }
        )
        is_best = miou > best
        best = max(best, miou)
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "epoch": ep,
                "best": best,
                "history": hist,
            },
            last_path,
        )
        if is_best:
            save_checkpoint(best_path, model, sc, {"best_miou": best, "epoch": ep})
        log(
            f"ep {ep + 1}/{tc.epochs} loss {hist[-1]['train_loss']:.3f}/{hist[-1]['val_loss']:.3f} | IoU field "
            f"{ious[1]:.3f} boundary {ious[2]:.3f} | mIoU {miou:.3f}{' *' if is_best else ''} | {time.time() - t0:.0f}s"
        )
    pd.DataFrame(hist).to_csv(out / "training_history.csv", index=False)
    (out / "training_summary.json").write_text(json.dumps({"best_miou": best, "checkpoint": str(best_path)}, indent=2))
    return {"best_miou": best, "checkpoint": str(best_path), "history": hist}
