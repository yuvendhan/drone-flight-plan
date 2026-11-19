"""Download satellite map tiles for an area and save them as a georeferenced GeoTIFF.

Because we know exactly which tiles we downloaded, the result already has GPS —
no screenshot, no manual georeferencing.

⚠️ Licensing: public basemaps (Esri World Imagery, Google, Apple) are for viewing. Use them for
personal testing only; for a product use your own drone orthomosaics or licensed imagery.
"""

from __future__ import annotations

import io
import math
from typing import Callable

import numpy as np
from PIL import Image
from rasterio.transform import from_origin

from .io import write_geotiff

ORIGIN = 20037508.342789244
TILE_PX = 256


def lonlat_to_tile(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2**z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(max(min(lat, 85.0511), -85.0511))
    y = (1.0 - math.asinh(math.tan(lat_r)) / math.pi) / 2.0 * n
    return x, y


def tile_size_m(z: int) -> float:
    return 2 * ORIGIN / 2**z


def ground_res_m(lat: float, z: int) -> float:
    """Metres per pixel on the ground at this latitude and zoom."""
    return tile_size_m(z) / TILE_PX * math.cos(math.radians(lat))


def _http_fetch(url: str) -> bytes:
    import requests

    r = requests.get(url, timeout=30, headers={"User-Agent": "drone-flight-plan/0.2 (research)"})
    r.raise_for_status()
    return r.content


def fetch_imagery(
    bbox_lonlat,
    out_path,
    zoom: int = 18,
    url_template: str = "",
    fetcher: Callable[[str], bytes] | None = None,
    max_tiles: int = 400,
):
    """
    bbox_lonlat: (west, south, east, north) in degrees.
    Returns path of a GeoTIFF in EPSG:3857.
    """
    fetcher = fetcher or _http_fetch
    w, s, e, n = bbox_lonlat
    x0, y0 = lonlat_to_tile(w, n, zoom)
    x1, y1 = lonlat_to_tile(e, s, zoom)
    tx = range(int(math.floor(x0)), int(math.floor(x1)) + 1)
    ty = range(int(math.floor(y0)), int(math.floor(y1)) + 1)
    if len(tx) * len(ty) > max_tiles:
        raise ValueError(
            f"Area too big: {len(tx) * len(ty)} tiles at zoom {zoom} (max {max_tiles}). "
            "Zoom in or lower the zoom level."
        )
    mosaic = np.zeros((len(ty) * TILE_PX, len(tx) * TILE_PX, 3), np.uint8)
    for j, y in enumerate(ty):
        for i, x in enumerate(tx):
            raw = fetcher(url_template.format(z=zoom, x=x, y=y))
            img = Image.open(io.BytesIO(raw)).convert("RGB").resize((TILE_PX, TILE_PX))
            mosaic[j * TILE_PX : (j + 1) * TILE_PX, i * TILE_PX : (i + 1) * TILE_PX] = np.asarray(img)
    size = tile_size_m(zoom)
    left = -ORIGIN + tx[0] * size
    top = ORIGIN - ty[0] * size
    transform = from_origin(left, top, size / TILE_PX, size / TILE_PX)
    return write_geotiff(out_path, mosaic.transpose(2, 0, 1), transform, "EPSG:3857")
