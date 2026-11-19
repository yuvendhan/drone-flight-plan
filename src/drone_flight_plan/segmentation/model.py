"""Model building, dataset, checkpoint I/O (PyTorch; install with `pip install drone-flight-plan[ml]`)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import SegmentationConfig
from ..geo.io import read_raster_rgb01

MEAN = np.array([0.485, 0.456, 0.406], np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], np.float32)[:, None, None]


def build_model(sc: SegmentationConfig, pretrained: bool = True):
    import segmentation_models_pytorch as smp

    weights = sc.encoder_weights if pretrained else None
    if sc.arch == "deeplabv3plus":
        return smp.DeepLabV3Plus(sc.encoder, encoder_weights=weights, in_channels=3, classes=3)
    return smp.Unet(sc.encoder, encoder_weights=weights, in_channels=3, classes=3)


def save_checkpoint(path, model, sc: SegmentationConfig, extra: dict | None = None):
    import torch

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seg_config": sc.model_dump(), **(extra or {})}, path)


def load_checkpoint(path, device="cpu"):
    """→ (model in eval mode, SegmentationConfig used for training, raw checkpoint dict)."""
    import torch

    ck = torch.load(path, map_location=device, weights_only=False)
    sc = SegmentationConfig.model_validate(ck.get("seg_config", {}))
    model = build_model(sc, pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), sc, ck


def make_dataset(df, sc: SegmentationConfig, augment=False):
    import torch

    from .data import read_targets

    class FieldDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.df = df.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, i):
            img, _, _ = read_raster_rgb01(self.df.img[i])
            tgt, _ = read_targets(self.df.msk[i])
            T = sc.tile
            img, tgt = img[:, :T, :T], tgt[:T, :T]
            ph, pw = T - img.shape[1], T - img.shape[2]
            if ph > 0 or pw > 0:
                img = np.pad(img, ((0, 0), (0, ph), (0, pw)))
                tgt = np.pad(tgt, ((0, ph), (0, pw)))
            if augment:
                k = np.random.randint(4)
                img, tgt = np.rot90(img, k, (1, 2)), np.rot90(tgt, k)
                if np.random.rand() < 0.5:
                    img, tgt = img[:, :, ::-1], tgt[:, ::-1]
                if np.random.rand() < 0.5:
                    img = np.clip(img * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08), 0, 1)
                if np.random.rand() < 0.3:  # colour shift: helps generalise to other regions/seasons
                    img = np.clip(img * np.random.uniform(0.85, 1.15, (3, 1, 1)), 0, 1)
            img = (img - MEAN) / STD
            return (
                torch.from_numpy(np.ascontiguousarray(img)).float(),
                torch.from_numpy(np.ascontiguousarray(tgt)).long(),
            )

    return FieldDataset()
