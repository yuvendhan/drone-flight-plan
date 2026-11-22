"""Export the trained PyTorch model to ONNX (and optionally int8) for light, fast inference.

Training needs PyTorch; running the model does not. ONNX Runtime is a much smaller install and is
usually faster on CPUs, so the app / free hosting only needs `onnxruntime`.

    export_onnx("models/unet_resnet34_3class.pt", "models/model.onnx")
    quantize_int8("models/model.onnx", "models/model_int8.onnx", calibration_images)
    compare_models("models/model.onnx", "models/model_int8.onnx", test_meta)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geo.io import read_raster_rgb01
from .model import MEAN, STD, load_checkpoint


def export_onnx(checkpoint, out_path, tile: int = 512, opset: int = 17, extra_meta: dict | None = None) -> Path:
    """PyTorch checkpoint (.pt) → ONNX with dynamic height/width. Training settings are embedded as metadata."""
    import onnx
    import torch

    model, sc, ck = load_checkpoint(checkpoint, "cpu")
    x = torch.randn(1, 3, tile, tile)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    kw = dict(input_names=["image"], output_names=["logits"], opset_version=opset)
    try:  # classic exporter: stable dynamic axes
        torch.onnx.export(
            model,
            x,
            str(out),
            dynamo=False,
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch", 2: "height", 3: "width"},
            },
            **kw,
        )
    except TypeError:  # very old torch without the `dynamo` argument
        torch.onnx.export(
            model,
            x,
            str(out),
            dynamic_axes={
                "image": {0: "batch", 2: "height", 3: "width"},
                "logits": {0: "batch", 2: "height", 3: "width"},
            },
            **kw,
        )
    m = onnx.load(str(out))
    meta = {
        "seg_config": json.dumps(sc.model_dump()),
        "source": Path(checkpoint).name,
        "best_miou": str(ck.get("best_miou", "")),
        "format": "fp32",
        **(extra_meta or {}),
    }
    _set_meta(m, meta)
    onnx.save(m, str(out))
    return out


def _set_meta(model, meta: dict):
    existing = {p.key: p for p in model.metadata_props}
    for k, v in meta.items():
        if k in existing:
            existing[k].value = str(v)
        else:
            p = model.metadata_props.add()
            p.key, p.value = k, str(v)


def read_meta(onnx_path) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    return dict(sess.get_modelmeta().custom_metadata_map)


def quantize_int8(onnx_path, out_path, calibration_images: list, tile: int = 512, max_images: int = 32) -> Path:
    """Static int8 quantisation (≈4× smaller). Calibrates on a few real tiles. Always check accuracy afterwards."""
    import onnx
    from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.items = iter(calibration_images[:max_images])

        def get_next(self):
            p = next(self.items, None)
            if p is None:
                return None
            rgb, _, _ = read_raster_rgb01(p)
            rgb = rgb[:, :tile, :tile]
            return {"image": ((rgb - MEAN) / STD)[None].astype(np.float32)}

    out = Path(out_path)
    pre = out.with_suffix(".pre.onnx")
    try:
        quant_pre_process(str(onnx_path), str(pre))
        src = pre
    except Exception:
        src = Path(onnx_path)
    quantize_static(
        str(src),
        str(out),
        Reader(),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )
    if pre.exists():
        pre.unlink()
    m = onnx.load(str(out))
    meta = read_meta(onnx_path)
    meta["format"] = "int8"
    _set_meta(m, meta)
    onnx.save(m, str(out))
    return out


def compare_models(model_a, model_b, meta_df, limit: int = 50) -> dict:
    """Same test tiles through two models → mIoU of each vs ground truth + pixel agreement between them."""
    from .data import read_targets
    from .infer import Predictor

    pa, pb = Predictor(str(model_a)), Predictor(str(model_b))
    cm_a, cm_b = np.zeros((3, 3), np.int64), np.zeros((3, 3), np.int64)
    agree, total = 0, 0
    for _, r in meta_df.head(limit).iterrows():
        rgb, _, _ = read_raster_rgb01(r.img)
        tgt, _ = read_targets(r.msk)
        a = pa.predict_probs(rgb, tta=False).argmax(0)
        b = pb.predict_probs(rgb, tta=False).argmax(0)
        cm_a += np.bincount(3 * tgt.ravel() + a.ravel(), minlength=9).reshape(3, 3)
        cm_b += np.bincount(3 * tgt.ravel() + b.ravel(), minlength=9).reshape(3, 3)
        agree += int((a == b).sum())
        total += a.size

    def miou(cm):
        i = np.diag(cm)
        return float((i / np.maximum(cm.sum(0) + cm.sum(1) - i, 1)).mean())

    size = lambda p: round(Path(p).stat().st_size / 1e6, 2)  # noqa: E731
    return {
        "miou_a": round(miou(cm_a), 4),
        "miou_b": round(miou(cm_b), 4),
        "miou_drop": round(miou(cm_a) - miou(cm_b), 4),
        "pixel_agreement": round(agree / max(total, 1), 4),
        "size_mb_a": size(model_a),
        "size_mb_b": size(model_b),
    }
