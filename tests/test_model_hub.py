import hashlib

import numpy as np
import pytest
from conftest import require_torch_stack

from drone_flight_plan.config import SegmentationConfig
from drone_flight_plan.hub import ModelNotAvailable, model_card, resolve_model


def test_resolve_local_first(tmp_path):
    p = tmp_path / "model.onnx"
    p.write_bytes(b"x")
    assert resolve_model(SegmentationConfig(checkpoint=str(p), hf_repo=None)) == p


def test_resolve_url_with_checksum(tmp_path, monkeypatch):
    data = b"fake-model-bytes" * 100

    class Resp:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

        def iter_content(self, n):
            yield data

    import requests

    monkeypatch.setattr(requests, "get", Resp)
    sc = SegmentationConfig(
        checkpoint=str(tmp_path / "none.onnx"),
        hf_repo=None,
        model_url="https://example.com/m/model.onnx",
        model_sha256=hashlib.sha256(data).hexdigest(),
    )
    p = resolve_model(sc, cache_dir=tmp_path / "cache")
    assert p.read_bytes() == data
    assert resolve_model(sc, cache_dir=tmp_path / "cache") == p  # cached, no re-download
    bad = sc.model_copy(update={"model_sha256": "0" * 64, "model_url": "https://example.com/x/other.onnx"})
    with pytest.raises(ModelNotAvailable):
        resolve_model(bad, cache_dir=tmp_path / "cache")


def test_resolve_hf(tmp_path, monkeypatch):
    import huggingface_hub

    target = tmp_path / "hf.onnx"
    target.write_bytes(b"m")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, fn, **k: str(target))
    sc = SegmentationConfig(checkpoint=str(tmp_path / "none.onnx"), hf_repo="u/r")
    assert resolve_model(sc) == target


def test_nothing_available(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(ModelNotAvailable):
        resolve_model(SegmentationConfig(checkpoint=str(tmp_path / "none.onnx"), hf_repo="u/r"))


def test_model_card():
    card = model_card("u/r", {"instance_F1": 0.81}, SegmentationConfig())
    assert card.startswith("---") and "instance_F1 | 0.81" in card and "u/r" in card


def test_onnx_export_matches_torch(tmp_path):
    require_torch_stack()
    pytest.importorskip("onnxruntime")
    from drone_flight_plan.config import load_config
    from drone_flight_plan.geo.io import read_raster_rgb01
    from drone_flight_plan.segmentation.data import make_synthetic_dataset
    from drone_flight_plan.segmentation.infer import Predictor
    from drone_flight_plan.segmentation.onnx_export import compare_models, export_onnx, quantize_int8
    from drone_flight_plan.segmentation.train import train

    cfg = load_config(
        "default",
        {
            "train": {
                "data_dir": str(tmp_path / "d"),
                "out_dir": str(tmp_path / "m"),
                "n_train": 4,
                "n_val": 2,
                "n_test": 2,
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
            },
            "segmentation": {"encoder_weights": None, "tile": 128, "encoder": "resnet18"},
        },
    )
    meta = make_synthetic_dataset(cfg.train, size=128)
    ck = train(cfg, meta, log=lambda *a: None)["checkpoint"]
    onnx_path = export_onnx(ck, tmp_path / "model.onnx", tile=128)
    rgb = read_raster_rgb01(meta.img.iloc[-1])[0]
    a = Predictor(ck).predict_probs(rgb)
    b = Predictor(str(onnx_path))
    assert b.backend == "onnx" and b.sc.encoder == "resnet18"
    assert np.abs(a - b.predict_probs(rgb)).max() < 1e-3
    q = quantize_int8(onnx_path, tmp_path / "model_int8.onnx", list(meta.img), tile=128)
    res = compare_models(onnx_path, q, meta[meta.split == "test"])
    assert res["size_mb_b"] < res["size_mb_a"] / 2


def test_upload_model_calls(tmp_path, monkeypatch):
    import huggingface_hub

    from drone_flight_plan.hub import upload_model

    calls = []

    class FakeApi:
        def __init__(self, token=None):
            calls.append(("token", token))

        def create_repo(self, repo_id, **k):
            calls.append(("create", repo_id))

        def upload_file(self, path_or_fileobj, path_in_repo, repo_id):
            calls.append(("upload", path_in_repo))

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    m = tmp_path / "model.onnx"
    m.write_bytes(b"x")
    url = upload_model(m, "u/r", "tok", {"instance_F1": 0.8}, SegmentationConfig())
    assert url == "https://huggingface.co/u/r"
    assert ("create", "u/r") in calls
    assert {c[1] for c in calls if c[0] == "upload"} == {"model.onnx", "metrics.json", "README.md"}
