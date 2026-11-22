"""Run a trained model on any georeferenced image → field polygons.

Works with either a PyTorch checkpoint (.pt, needs torch) or an ONNX model (.onnx, needs only onnxruntime).
"""

from __future__ import annotations

import json

import numpy as np

from ..config import SegmentationConfig
from ..geo.crs import WGS84, reproject_geom
from ..geo.io import read_raster_rgb01
from .model import MEAN, STD
from .postprocess import instances_to_polygons, probs_to_instances


def _softmax(x, axis=1):
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


class _TorchRunner:
    def __init__(self, path, device=None):
        import torch

        from .model import load_checkpoint

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.sc, self.meta = load_checkpoint(path, self.device)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        amp = self.device == "cuda"
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp):
            return self.model(torch.from_numpy(x).to(self.device)).float().cpu().numpy()


class _OnnxRunner:
    def __init__(self, path):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.meta = dict(self.sess.get_modelmeta().custom_metadata_map)
        self.sc = SegmentationConfig.model_validate(json.loads(self.meta.get("seg_config", "{}")))

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input: x.astype(np.float32)})[0]


class Predictor:
    def __init__(self, checkpoint: str, sc: SegmentationConfig | None = None, device: str | None = None):
        self.path = str(checkpoint)
        self.runner = _OnnxRunner(checkpoint) if self.path.endswith(".onnx") else _TorchRunner(checkpoint, device)
        trained = self.runner.sc
        # architecture/resolution come from training; thresholds etc. may be overridden by config
        self.sc = (
            trained
            if sc is None
            else sc.model_copy(
                update={"arch": trained.arch, "encoder": trained.encoder, "target_res_m": trained.target_res_m}
            )
        )
        self.backend = "onnx" if isinstance(self.runner, _OnnxRunner) else "torch"

    def predict_probs(self, rgb01: np.ndarray, tta: bool = True) -> np.ndarray:
        """(3,H,W) in 0..1 → (3,H,W) class probabilities."""
        x = ((rgb01 - MEAN) / STD).astype(np.float32)[None]
        H, W = x.shape[-2:]
        ph, pw = (32 - H % 32) % 32, (32 - W % 32) % 32
        if ph or pw:
            x = np.pad(x, ((0, 0), (0, 0), (0, ph), (0, pw)), mode="reflect")
        p = _softmax(self.runner(x))
        n = 1
        if tta:
            p += _softmax(self.runner(np.ascontiguousarray(x[..., ::-1])))[..., ::-1]
            p += _softmax(self.runner(np.ascontiguousarray(x[..., ::-1, :])))[..., ::-1, :]
            n = 3
        return (p / n)[0, :, :H, :W].astype(np.float32)

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
