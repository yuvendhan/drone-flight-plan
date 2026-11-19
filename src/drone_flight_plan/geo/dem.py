"""Ground height (DEM) so the drone can keep a constant height above sloping / terraced land.

Sources:
- your own DSM/DTM GeoTIFF (e.g. from WebODM — best, centimetre detail)
- Copernicus GLO-30 (free, global, 30 m) read straight from the public AWS bucket
- Flat (no terrain) fallback
"""

from __future__ import annotations

import math

import numpy as np
import rasterio
from scipy.ndimage import map_coordinates

from .crs import WGS84, transformer

COP30_URL = (
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{ns}{lat:02d}_00_{ew}{lon:03d}_00_DEM.tif"
)


class Terrain:
    """Ground elevation lookup. elevation(xy, crs) → metres above sea level."""

    def __init__(self, array=None, transform=None, crs=None, name="flat"):
        self.array, self.transform, self.crs, self.name = array, transform, crs, name

    @property
    def is_flat(self) -> bool:
        return self.array is None

    @classmethod
    def flat(cls) -> "Terrain":
        return cls()

    @classmethod
    def from_geotiff(cls, path, bounds_lonlat=None, pad_deg=0.002) -> "Terrain":
        with rasterio.open(path) as s:
            if bounds_lonlat is not None:
                w, so, e, n = bounds_lonlat
                t = transformer(WGS84, s.crs)
                xs, ys = zip(
                    *[
                        t.transform(x, y)
                        for x, y in [
                            (w - pad_deg, so - pad_deg),
                            (e + pad_deg, n + pad_deg),
                            (w - pad_deg, n + pad_deg),
                            (e + pad_deg, so - pad_deg),
                        ]
                    ]
                )
                win = rasterio.windows.from_bounds(min(xs), min(ys), max(xs), max(ys), s.transform)
                win = win.round_offsets().round_lengths().intersection(rasterio.windows.Window(0, 0, s.width, s.height))
                arr = s.read(1, window=win, masked=True).astype(np.float32)
                tr = s.window_transform(win)
            else:
                arr, tr = s.read(1, masked=True).astype(np.float32), s.transform
            arr = arr.filled(np.nan) if np.ma.isMaskedArray(arr) else arr
            if np.isnan(arr).all():
                raise ValueError("DEM has no data over this area")
            arr = np.where(np.isnan(arr), np.nanmedian(arr), arr)
            return cls(arr, tr, s.crs, name=str(path))

    @classmethod
    def copernicus30(cls, bounds_lonlat) -> "Terrain":
        """Free global 30 m DEM (needs internet). Single 1°×1° tile covering the field centre."""
        w, s, e, n = bounds_lonlat
        lat, lon = math.floor((s + n) / 2), math.floor((w + e) / 2)
        url = COP30_URL.format(ns="N" if lat >= 0 else "S", lat=abs(lat), ew="E" if lon >= 0 else "W", lon=abs(lon))
        t = cls.from_geotiff("/vsicurl/" + url, bounds_lonlat)
        t.name = "Copernicus GLO-30"
        return t

    def elevation(self, xy, crs) -> np.ndarray:
        xy = np.atleast_2d(np.asarray(xy, float))
        if self.is_flat:
            return np.zeros(len(xy))
        t = transformer(crs, self.crs)
        X, Y = t.transform(xy[:, 0], xy[:, 1])
        inv = ~self.transform
        X, Y = np.asarray(X), np.asarray(Y)
        cols, rows = inv.a * X + inv.b * Y + inv.c, inv.d * X + inv.e * Y + inv.f
        # bilinear; transform maps pixel corners, so shift by 0.5 to pixel centres
        return map_coordinates(self.array, [np.asarray(rows) - 0.5, np.asarray(cols) - 0.5], order=1, mode="nearest")

    def slope_and_contour(self, field_m, crs, n=12) -> dict:
        """Fit a plane over the field → slope (%) and contour direction (degrees, 0–179, in metric frame)."""
        if self.is_flat:
            return {"slope_pct": 0.0, "contour_deg": None, "relief_m": 0.0}
        minx, miny, maxx, maxy = field_m.bounds
        gx, gy = np.meshgrid(np.linspace(minx, maxx, n), np.linspace(miny, maxy, n))
        pts = np.c_[gx.ravel(), gy.ravel()]
        from shapely import contains_xy

        inside = contains_xy(field_m, pts[:, 0], pts[:, 1])
        pts = pts[inside] if inside.sum() >= 6 else pts
        z = self.elevation(pts, crs)
        A = np.c_[pts, np.ones(len(pts))]
        (a, b, _), *_ = np.linalg.lstsq(A, z, rcond=None)
        slope = math.hypot(a, b)
        grad_deg = math.degrees(math.atan2(b, a))
        return {
            "slope_pct": round(100 * slope, 2),
            "contour_deg": round((grad_deg + 90) % 180, 1),
            "relief_m": round(float(z.max() - z.min()), 2),
        }
