import numpy as np
import pytest
from rasterio.transform import from_origin

from drone_flight_plan.segmentation.postprocess import instances_to_polygons, probs_to_instances


def test_touching_fields_are_separated():
    H = W = 200
    prob = np.zeros((3, H, W), np.float32)
    prob[0] = 1
    prob[:, 20:180, 20:180] = 0
    prob[1, 20:180, 20:180] = 1
    prob[2, 20:180, 98:102] = 1  # boundary line in the middle
    prob[1, 20:180, 98:102] = 0
    lab = probs_to_instances(prob)
    assert len(np.unique(lab)) - 1 == 2
    polys = instances_to_polygons(lab, from_origin(0, 200, 1, 1), min_area_m2=100)
    assert len(polys) == 2
    assert all(abs(p.area - 80 * 160) / (80 * 160) < 0.1 for p in polys)


def test_train_and_predict_tiny(tmp_path):
    pytest.importorskip("segmentation_models_pytorch")
    from drone_flight_plan.config import load_config
    from drone_flight_plan.segmentation.data import make_synthetic_dataset
    from drone_flight_plan.segmentation.infer import Predictor
    from drone_flight_plan.segmentation.train import train

    cfg = load_config(
        "default",
        {
            "train": {
                "data_dir": str(tmp_path / "d"),
                "out_dir": str(tmp_path / "m"),
                "n_train": 4,
                "n_val": 2,
                "n_test": 1,
                "epochs": 1,
                "batch_size": 2,
                "num_workers": 0,
            },
            "segmentation": {"encoder_weights": None, "tile": 128, "encoder": "resnet18"},
        },
    )
    meta = make_synthetic_dataset(cfg.train, size=128)
    res = train(cfg, meta, log=lambda *a: None)
    pred = Predictor(res["checkpoint"], cfg.segmentation)
    out = pred.segment_geotiff(meta.img.iloc[-1])
    assert out["prob"].shape[0] == 3 and np.allclose(out["prob"].sum(0), 1, atol=1e-3)


def test_meta_from_folder(tmp_path):
    from drone_flight_plan.segmentation.data import make_synthetic_tile, meta_from_folder

    (tmp_path / "images").mkdir()
    (tmp_path / "masks").mkdir()
    for i in range(10):
        make_synthetic_tile(i, tmp_path / "images" / f"t{i}.tif", tmp_path / "masks" / f"t{i}.tif", size=64)
    df = meta_from_folder(tmp_path)
    assert len(df) == 10 and set(df.split) == {"train", "val", "test"}
