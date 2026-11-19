"""Read/write rasters and field outlines (GeoTIFF, KML, GeoJSON)."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

from .crs import WGS84


def pixel_size_m(src: rasterio.DatasetReader) -> float:
    res = abs(src.transform.a)
    if src.crs is None:
        raise ValueError("Image has no coordinate system (not georeferenced). Georeference it first.")
    if src.crs.is_geographic:
        lat = src.xy(src.height // 2, src.width // 2)[1]
        res *= 111_320 * math.cos(math.radians(lat))
    elif src.crs.to_epsg() == 3857:  # web mercator metres are stretched by 1/cos(lat)
        from .crs import transformer

        lon, lat = transformer(src.crs, WGS84).transform(*src.xy(src.height // 2, src.width // 2))
        res *= math.cos(math.radians(lat))
    return res


def to_rgb01(a: np.ndarray) -> np.ndarray:
    """(C,H,W) any dtype → (3,H,W) float32 in 0..1."""
    a = a[:3] if a.shape[0] >= 3 else np.repeat(a[:1], 3, 0)
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    a = a.astype(np.float32)
    lo, hi = np.nanpercentile(a, (2, 98), axis=(1, 2), keepdims=True)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)


def read_raster_rgb01(path, target_res_m: float | None = None):
    """GeoTIFF → (rgb float32 (3,H,W), transform, crs). Optionally resample to target_res_m metres/pixel."""
    with rasterio.open(path) as s:
        crs, tr = s.crs, s.transform
        if target_res_m:
            scale = pixel_size_m(s) / target_res_m
            H, W = max(32, round(s.height * scale)), max(32, round(s.width * scale))
            a = s.read(out_shape=(s.count, H, W), resampling=Resampling.bilinear)
            sx, sy = s.width / W, s.height / H
            tr = Affine(tr.a * sx, tr.b * sy, tr.c, tr.d * sx, tr.e * sy, tr.f)
        else:
            a = s.read()
    return to_rgb01(a), tr, crs


def write_geotiff(path, rgb: np.ndarray, transform, crs) -> Path:
    """rgb: (3,H,W) uint8 or float 0..1."""
    if rgb.dtype != np.uint8:
        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    path = Path(path)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=rgb.shape[2],
        height=rgb.shape[1],
        count=3,
        dtype="uint8",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(rgb)
    return path


def parse_kml_polygons(text: str) -> list[Polygon]:
    """All polygons in a KML (Google Earth / Google My Maps), as WGS84 lon/lat."""
    out = []
    for block in re.findall(r"<Polygon.*?</Polygon>", text, flags=re.S | re.I):
        rings = re.findall(r"<coordinates>(.*?)</coordinates>", block, flags=re.S | re.I)
        if not rings:
            continue

        def parse(r):
            return [tuple(map(float, c.split(",")[:2])) for c in r.split()]

        outer, holes = parse(rings[0]), [parse(r) for r in rings[1:]]
        if len(outer) >= 3:
            out.append(Polygon(outer, [h for h in holes if len(h) >= 3]).buffer(0))
    if not out:  # a plain LineString/LinearRing outline
        for r in re.findall(r"<coordinates>(.*?)</coordinates>", text, flags=re.S | re.I):
            pts = [tuple(map(float, c.split(",")[:2])) for c in r.split()]
            if len(pts) >= 3:
                out.append(Polygon(pts).buffer(0))
    return out


def read_field_file(path) -> Polygon:
    """KML or GeoJSON → largest polygon (WGS84)."""
    path = Path(path)
    text = path.read_text(errors="ignore")
    if path.suffix.lower() in (".kml", ".xml"):
        polys = parse_kml_polygons(text)
    else:
        gj = json.loads(text)
        feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
        polys = []
        for f in feats:
            g = shape(f["geometry"] if "geometry" in f else f)
            polys += list(g.geoms) if isinstance(g, MultiPolygon) else [g] if isinstance(g, Polygon) else []
    if not polys:
        raise ValueError(f"No polygon found in {path.name}")
    return max(polys, key=lambda p: p.area)


def polygons_to_geojson(polys, crs, props: list[dict] | None = None) -> dict:
    from .crs import reproject_geom

    feats = []
    for i, p in enumerate(polys):
        feats.append(
            {
                "type": "Feature",
                "properties": (props[i] if props else {"id": i}),
                "geometry": mapping(reproject_geom(p, crs, WGS84)),
            }
        )
    return {"type": "FeatureCollection", "features": feats}
