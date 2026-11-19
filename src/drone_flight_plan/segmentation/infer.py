"""Run a trained model on any georeferenced image → field polygons."""

from __future__ import annotations

import numpy as np

from ..config import SegmentationConfig
from ..geo.crs import WGS84, reproject_geom
from ..geo.io import read_raster_rgb01
from .model import MEAN, STD, load_checkpoint
from .postprocess import instances_to_polygons, probs_to_instances


class Predictor:
    def __init__(self, checkpoint: str, sc: SegmentationConfig | None = None, device: str | None = None):
        import torch

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, trained_sc, self.ckpt = load_checkpoint(checkpoint, self.device)
        # architecture comes from the checkpoint; thresholds/resolution can be overridden by config
        self.sc = (
            trained_sc if sc is None else sc.model_copy(update={"arch": trained_sc.arch, "encoder": trained_sc.encoder})
        )

    def predict_probs(self, rgb01: np.ndarray, tta: bool = True) -> np.ndarray:
        """(3,H,W) in 0..1 → (3,H,W) class probabilities."""
        torch = self.torch
        x = torch.from_numpy((rgb01 - MEAN) / STD).float()[None].to(self.device)
        H, W = x.shape[-2:]
        ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
        x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        amp = self.device == "cuda"
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            p = self.model(x).float().softmax(1)
            n = 1
            if tta:
                p += self.model(x.flip(-1)).float().softmax(1).flip(-1)
                p += self.model(x.flip(-2)).float().softmax(1).flip(-2)
                n = 3
        return (p / n)[0, :, :H, :W].cpu().numpy()

    def predict_large(self, rgb01: np.ndarray) -> np.ndarray:
        """Sliding window with centre-weighted blending (no seams)."""
        T, ov = self.sc.tile, self.sc.overlap
        _, H, W = rgb01.shape
        prob = np.zeros((3, H, W), np.float32)
        wsum = np.zeros((H, W), np.float32)
        step = max(32, T - ov)
        ys = list(range(0, max(H - T, 0) + 1, step)) or [0]
        xs = list(range(0, max(W - T, 0) + 1, step)) or [0]
        if ys[-1] + T < H:
            ys.append(H - T)
        if xs[-1] + T < W:
            xs.append(W - T)
        for y in ys:
            for x in xs:
                crop = rgb01[:, y : y + T, x : x + T]
                h, w = crop.shape[1:]
                win = np.maximum(np.outer(np.hanning(h + 2)[1:-1], np.hanning(w + 2)[1:-1]), 1e-3).astype(np.float32)
                prob[:, y : y + h, x : x + w] += self.predict_probs(crop) * win
                wsum[y : y + h, x : x + w] += win
        return prob / np.maximum(wsum, 1e-6)

    def segment_array(self, rgb01, transform, px_m=None):
        sc = self.sc
        prob = self.predict_large(rgb01)
        lab = probs_to_instances(prob, sc.seed_thr, sc.boundary_thr, sc.min_field_px)
        polys = instances_to_polygons(lab, transform, sc.simplify_m, sc.min_field_area_m2, px_m=px_m)
        return polys, prob, lab

    def segment_geotiff(self, path) -> dict:
        """GeoTIFF (any CRS / resolution) → fields. Image is resampled to the training resolution."""
        rgb, tr, crs = read_raster_rgb01(path, self.sc.target_res_m)
        polys, prob, lab = self.segment_array(rgb, tr, px_m=self.sc.target_res_m)
        return {
            "polygons": polys,
            "polygons_wgs84": [reproject_geom(p, crs, WGS84) for p in polys],
            "crs": crs,
            "transform": tr,
            "rgb": rgb,
            "prob": prob,
            "labels": lab,
        }
