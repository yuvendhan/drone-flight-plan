"""Put a plain picture (screenshot, photo) on the map using a few known GPS points.

A screenshot from Google Maps / Earth / Apple Maps has no coordinates. If you know the GPS of
2+ spots in the picture (right-click → copy coordinates in Google Maps), we can compute where
every pixel is on Earth.

- 2 points, north-up=True  → scale only (picture not rotated; best for map screenshots)
- 2 points                  → scale + rotation
- 3+ points                 → full affine (handles slight skew), with an error report
Web maps use Web Mercator, so we fit in EPSG:3857.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from rasterio.transform import Affine

from .crs import WEB_MERCATOR, WGS84, transformer
from .io import write_geotiff


@dataclass
class ControlPoint:
    col: float  # pixel x (from left)
    row: float  # pixel y (from top)
    lat: float
    lon: float


@dataclass
class GeorefResult:
    path: Path
    transform: Affine
    rms_error_m: float
    ground_res_m: float
    method: str


def fit_transform(points: list[ControlPoint], north_up: bool = False) -> tuple[Affine, float, str]:
    if len(points) < 2:
        raise ValueError("Need at least 2 points with known GPS")
    t = transformer(WGS84, WEB_MERCATOR)
    XY = np.array([t.transform(p.lon, p.lat) for p in points])
    CR = np.array([(p.col, p.row) for p in points], float)
    n = len(points)
    if north_up and n >= 2:
        A = np.c_[CR[:, 0], np.ones(n)]
        (sx, c), *_ = np.linalg.lstsq(A, XY[:, 0], rcond=None)
        A = np.c_[CR[:, 1], np.ones(n)]
        (sy, f), *_ = np.linalg.lstsq(A, XY[:, 1], rcond=None)
        aff, method = Affine(sx, 0, c, 0, sy, f), "north-up scale"
    elif n == 2:
        # X = a*c + b*r + tx ; Y = b*c - a*r + ty  (similarity with image y pointing down)
        A, B = [], []
        for (c_, r_), (x, y) in zip(CR, XY):
            A += [[c_, r_, 1, 0], [-r_, c_, 0, 1]]
            B += [x, y]
        a, b, tx, ty = np.linalg.lstsq(np.array(A), np.array(B), rcond=None)[0]
        aff, method = Affine(a, b, tx, b, -a, ty), "similarity"
    else:
        A = np.c_[CR, np.ones(n)]
        px, *_ = np.linalg.lstsq(A, XY[:, 0], rcond=None)
        py, *_ = np.linalg.lstsq(A, XY[:, 1], rcond=None)
        aff, method = Affine(*px, *py), "affine"
    pred = np.c_[aff.a * CR[:, 0] + aff.b * CR[:, 1] + aff.c, aff.d * CR[:, 0] + aff.e * CR[:, 1] + aff.f]
    lat0 = math.radians(np.mean([p.lat for p in points]))
    rms = float(np.sqrt(np.mean(np.sum((pred - XY) ** 2, axis=1)))) * math.cos(lat0)
    if n == 2 and not north_up:
        rms = float("nan")  # exact fit, no redundancy → unknown error
    return aff, rms, method


def georeference_image(image_path, points: list[ControlPoint], out_path, north_up: bool = False) -> GeorefResult:
    img = np.asarray(Image.open(image_path).convert("RGB"))
    aff, rms, method = fit_transform(points, north_up)
    lat0 = math.radians(np.mean([p.lat for p in points]))
    res = math.hypot(aff.a, aff.d) * math.cos(lat0)
    path = write_geotiff(out_path, img.transpose(2, 0, 1), aff, WEB_MERCATOR)
    return GeorefResult(Path(path), aff, rms, res, method)


def georeference_by_corners(image_path, top_left_latlon, bottom_right_latlon, out_path) -> GeorefResult:
    """Simplest option: GPS of the top-left and bottom-right corners of a north-up screenshot."""
    w, h = Image.open(image_path).size
    pts = [ControlPoint(0, 0, *top_left_latlon), ControlPoint(w, h, *bottom_right_latlon)]
    return georeference_image(image_path, pts, out_path, north_up=True)
