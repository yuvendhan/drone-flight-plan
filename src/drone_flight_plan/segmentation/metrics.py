"""Segmentation metrics: pixel, boundary, field-instance; threshold tuning on validation."""

from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from skimage.morphology import disk

from ..geo.io import read_raster_rgb01
from .data import read_targets
from .postprocess import probs_to_instances


def match_instances(pred, gt, thr=0.5):
    g_ids = [g for g in np.unique(gt) if g > 0]
    p_ids = [p for p in np.unique(pred) if p > 0]
    if not g_ids or not p_ids:
        return 0, len(p_ids), len(g_ids), []
    inter = np.zeros((max(p_ids) + 1, max(g_ids) + 1))
    np.add.at(inter, (pred.ravel(), gt.ravel()), 1)
    ps, gs = inter.sum(1), inter.sum(0)
    tp, ious, used = 0, [], set()
    for g in g_ids:
        col = inter[:, g].copy()
        col[0] = 0
        p = int(col.argmax())
        if p == 0 or p in used:
            continue
        iou = col[p] / (ps[p] + gs[g] - col[p])
        if iou >= thr:
            tp += 1
            ious.append(iou)
            used.add(p)
    return tp, len(p_ids), len(g_ids), ious


def boundary_f1(pred_b, gt_b, tol=2):
    if gt_b.sum() == 0 and pred_b.sum() == 0:
        return 1.0
    prec = (pred_b & binary_dilation(gt_b, structure=disk(tol))).sum() / max(pred_b.sum(), 1)
    rec = (gt_b & binary_dilation(pred_b, structure=disk(tol))).sum() / max(gt_b.sum(), 1)
    return 2 * prec * rec / max(prec + rec, 1e-9)


def predict_split(predictor, df, limit=None):
    """Cache predictions for a split: list of (file_id, rgb, transform, crs, target, ids, prob, seconds)."""
    out = []
    for _, r in df.head(limit or len(df)).iterrows():
        rgb, tr, crs = read_raster_rgb01(r.img)
        tgt, enum = read_targets(r.msk)
        t0 = time.time()
        prob = predictor.predict_probs(rgb)
        out.append((r.file_id, rgb, tr, crs, tgt, enum, prob, time.time() - t0))
    return out


def evaluate(cache, seed_thr=0.5, boundary_thr=0.4, min_px=150):
    cm = np.zeros((3, 3), np.int64)
    bf, TP, NP, NG, mi, ts = [], 0, 0, 0, [], []
    for _, _, _, _, tgt, enum, prob, sec in cache:
        pred = prob.argmax(0)
        cm += np.bincount(3 * tgt.ravel() + pred.ravel(), minlength=9).reshape(3, 3)
        bf.append(boundary_f1(pred == 2, tgt == 2))
        lab = probs_to_instances(prob, seed_thr, boundary_thr, min_px)
        tp, n_p, n_g, ious = match_instances(lab, enum * (tgt > 0))
        TP, NP, NG = TP + tp, NP + n_p, NG + n_g
        mi += ious
        ts.append(sec)
    inter = np.diag(cm)
    iou = inter / (cm.sum(0) + cm.sum(1) - inter).clip(1)
    pixel = pd.DataFrame(
        {
            "IoU": iou,
            "Dice": 2 * inter / (cm.sum(0) + cm.sum(1)).clip(1),
            "Precision": inter / cm.sum(0).clip(1),
            "Recall": inter / cm.sum(1).clip(1),
        },
        index=["background", "field", "boundary"],
    ).round(3)
    ext_i, ext_t, ext_p = cm[1:, 1:].sum(), cm[1:, :].sum(), cm[:, 1:].sum()
    p, r = TP / max(NP, 1), TP / max(NG, 1)
    summary = {
        "field_extent_IoU": round(ext_i / max(ext_t + ext_p - ext_i, 1), 3),
        "mIoU_3class": round(iou.mean(), 3),
        "boundary_F1@2px": round(float(np.mean(bf)), 3),
        "instance_precision": round(p, 3),
        "instance_recall": round(r, 3),
        "instance_F1": round(2 * p * r / max(p + r, 1e-9), 3),
        "matched_field_mean_IoU": round(float(np.mean(mi)) if mi else 0.0, 3),
        "inference_s_per_tile": round(float(np.mean(ts)), 3),
    }
    return pixel, summary


def tune_thresholds(val_cache, seeds=(0.4, 0.5, 0.6, 0.7), bounds=(0.3, 0.4, 0.5)):
    """Grid-search watershed thresholds on validation for best field-instance F1."""
    rows = []
    for s, b in itertools.product(seeds, bounds):
        _, summ = evaluate(val_cache, s, b)
        rows.append({"seed_thr": s, "boundary_thr": b, "instance_F1": summ["instance_F1"]})
    df = pd.DataFrame(rows).sort_values("instance_F1", ascending=False)
    return df.iloc[0].to_dict(), df
