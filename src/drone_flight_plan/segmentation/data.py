"""Training data: AI4Boundaries download, synthetic tiles (smoke tests), label reading."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.transform import Affine
from scipy import ndimage as ndi
from scipy.ndimage import binary_dilation
from shapely import affinity
from shapely.geometry import Polygon
from skimage.measure import label as sk_label
from skimage.morphology import disk
from skimage.segmentation import find_boundaries

from ..config import TrainConfig

AI4B_CSV = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/AI4BOUNDARIES/orthophoto/"
    "ai4boundaries_ftp_urls_orthophoto_split.csv"
)


def read_targets(path):
    """Label file → (3-class target [0 bg, 1 field, 2 boundary], field-ID map).
    Handles AI4Boundaries 4-band labels (extent, boundary, distance, enumeration) or a 1-band ID map."""
    with rasterio.open(path) as s:
        m = np.nan_to_num(s.read().astype(np.float32))
    if m.shape[0] >= 2:
        extent, boundary = m[0] > 0.5, m[1] > 0.5
        enum = m[3].astype(np.int32) if m.shape[0] >= 4 else None
    else:
        enum = m[0].astype(np.int32)
        extent = enum > 0
        boundary = find_boundaries(enum, mode="inner") & extent
    if enum is None or enum.max() == 0:
        enum = sk_label(extent & ~boundary)
    boundary = binary_dilation(boundary, structure=disk(1)) & (extent | boundary)
    tgt = np.zeros(extent.shape, np.int64)
    tgt[extent] = 1
    tgt[boundary] = 2
    return tgt, enum


def _download(url, dst: Path, tries=3) -> bool:
    import requests

    if dst.exists() and dst.stat().st_size > 1000:
        return True
    for _ in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200 and len(r.content) > 1000:
                dst.write_bytes(r.content)
                return True
        except Exception:
            time.sleep(2)
    return False


def download_ai4boundaries(tc: TrainConfig, workers=16) -> pd.DataFrame:
    """Download a country-balanced sample. Returns meta table: file_id, split, img, msk."""
    root = Path(tc.data_dir)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    idx = pd.read_csv(AI4B_CSV)
    idx["country"] = idx.file_id.str.split("_").str[0]
    idx = idx[idx.country.isin(tc.countries)]
    tr = idx[idx.split == "train"].sample(frac=1, random_state=tc.seed)
    va = idx[idx.split.isin(["val", "validation"])]
    te = idx[idx.split == "test"]
    if len(va) == 0:
        va, tr = tr.iloc[: tc.n_val], tr.iloc[tc.n_val :]
    if len(te) == 0:
        te, tr = tr.iloc[: tc.n_test], tr.iloc[tc.n_test :]
    k = max(1, len(tc.countries))

    def pick(d, n):
        return (
            d.groupby("country", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, n // k)), random_state=tc.seed))
            .head(n)
        )

    sel = pd.concat(
        [
            pick(tr, tc.n_train).assign(split="train"),
            pick(va, tc.n_val).assign(split="val"),
            pick(te, tc.n_test).assign(split="test"),
        ]
    )
    sel["img"] = [str(root / "images" / Path(u).name) for u in sel.orthophoto_images_file_url]
    sel["msk"] = [str(root / "masks" / Path(u).name) for u in sel.orthophoto_masks_file_url]
    jobs = list(zip(sel.orthophoto_images_file_url, sel.img)) + list(zip(sel.orthophoto_masks_file_url, sel.msk))
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda j: _download(j[0], Path(j[1])), jobs))
    sel = sel[[Path(a).exists() and Path(b).exists() for a, b in zip(sel.img, sel.msk)]]
    meta = sel[["file_id", "split", "img", "msk"]].reset_index(drop=True)
    meta.to_csv(root / "meta.csv", index=False)
    return meta


def make_synthetic_tile(seed: int, out_img, out_msk, size: int = 512, res: float = 1.0):
    """Fake but structured farm tile with 4-band labels like AI4Boundaries (for tests / smoke runs)."""
    rng = np.random.default_rng(seed)
    x0, y0 = 4_000_000 + rng.integers(0, 50_000), 3_000_000 + rng.integers(0, 50_000)
    tr = Affine(res, 0, x0, 0, -res, y0 + size * res)
    boxes, work = [], [(-150, -150, size + 150, size + 150)]
    while work:
        b = work.pop()
        w, h = b[2] - b[0], b[3] - b[1]
        if max(w, h) < rng.integers(60, 200):
            boxes.append(b)
            continue
        if w > h:
            c = b[0] + w * rng.uniform(0.35, 0.65)
            work += [(b[0], b[1], c, b[3]), (c, b[1], b[2], b[3])]
        else:
            c = b[1] + h * rng.uniform(0.35, 0.65)
            work += [(b[0], b[1], b[2], c), (b[0], c, b[2], b[3])]
    ang, gap = rng.uniform(0, 90), rng.uniform(1.5, 3.5)
    polys = [
        affinity.rotate(Polygon([(a, b), (c, b), (c, d), (a, d)]).buffer(-gap), ang, origin=(size / 2, size / 2))
        for a, b, c, d in boxes
    ]
    enum = np.zeros((size, size), np.int32)
    img = np.zeros((size, size, 3), np.float32)
    img[:] = rng.uniform(0.25, 0.4, 3)
    for k, p in enumerate(polys, 1):
        if p.is_empty:
            continue
        m = features.rasterize([(p, 1)], out_shape=(size, size), transform=Affine.identity()).astype(bool)
        if rng.random() < 0.12:
            img[m] = (0.1, 0.25, 0.1)
            continue
        enum[m] = k
        base = rng.choice([[0.45, 0.35, 0.2], [0.3, 0.55, 0.2], [0.6, 0.55, 0.3], [0.2, 0.4, 0.15]])
        img[m] = np.clip(np.array(base) + rng.normal(0, 0.05, 3), 0, 1)
    img += rng.normal(0, 0.03, img.shape)
    img += 0.02 * np.sin(np.arange(size)[:, None] / rng.uniform(2, 5))[..., None]
    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    extent = enum > 0
    boundary = find_boundaries(enum, mode="inner") & extent
    prof = dict(driver="GTiff", width=size, height=size, crs="EPSG:3035", transform=tr)
    with rasterio.open(out_img, "w", count=3, dtype="uint8", **prof) as dst:
        dst.write(img.transpose(2, 0, 1))
    with rasterio.open(out_msk, "w", count=4, dtype="float32", **prof) as dst:
        dst.write(np.stack([extent, boundary, ndi.distance_transform_edt(extent), enum]).astype(np.float32))


def make_synthetic_dataset(tc: TrainConfig, size: int = 512) -> pd.DataFrame:
    root = Path(tc.data_dir)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    n = tc.n_train + tc.n_val + tc.n_test
    for i in range(n):
        ip, mp = root / "images" / f"SYN_{i}.tif", root / "masks" / f"SYN_{i}.tif"
        if not ip.exists():
            make_synthetic_tile(i, ip, mp, size)
        split = "train" if i < tc.n_train else ("val" if i < tc.n_train + tc.n_val else "test")
        rows.append({"file_id": f"SYN_{i}", "split": split, "img": str(ip), "msk": str(mp)})
    meta = pd.DataFrame(rows)
    meta.to_csv(root / "meta.csv", index=False)
    return meta


def meta_from_folder(root, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 42) -> pd.DataFrame:
    """Your own labelled tiles → meta table.  root/images/<id>.tif + root/masks/<id>.tif
    (mask = 1-band field-ID raster: 0 background, 1..N one number per field — e.g. rasterised from QGIS polygons)."""
    root = Path(root)
    rows = [
        {"file_id": p.stem, "img": str(p), "msk": str(root / "masks" / p.name)}
        for p in sorted((root / "images").glob("*.tif"))
        if (root / "masks" / p.name).exists()
    ]
    if not rows:
        raise FileNotFoundError(f"No matching images/masks .tif pairs in {root}")
    df = pd.DataFrame(rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_te, n_va = max(1, int(n * test_frac)), max(1, int(n * val_frac))
    df["split"] = ["test"] * n_te + ["val"] * n_va + ["train"] * (n - n_te - n_va)
    df.to_csv(root / "meta.csv", index=False)
    return df
