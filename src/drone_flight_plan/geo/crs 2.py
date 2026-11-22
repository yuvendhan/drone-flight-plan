"""Coordinate helpers: pixels ↔ map ↔ metres (UTM) ↔ GPS (WGS84)."""

from __future__ import annotations

from functools import lru_cache

import shapely
from pyproj import CRS, Transformer
from shapely.geometry import Polygon

WGS84 = "EPSG:4326"
WEB_MERCATOR = "EPSG:3857"


def utm_crs_for(lon: float, lat: float) -> CRS:
    zone = int((lon + 180) // 6) + 1
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


@lru_cache(maxsize=64)
def _transformer(src: str, dst: str) -> Transformer:
    return Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)


def transformer(src, dst) -> Transformer:
    return _transformer(CRS.from_user_input(src).to_string(), CRS.from_user_input(dst).to_string())


def reproject_geom(geom, src, dst):
    if CRS.from_user_input(src) == CRS.from_user_input(dst):
        return geom
    return shapely.ops.transform(transformer(src, dst).transform, geom)


def to_metric(geom, src_crs):
    """Any geometry → (geometry in local UTM metres, that UTM CRS)."""
    c = reproject_geom(geom.centroid, src_crs, WGS84)
    utm = utm_crs_for(c.x, c.y)
    return reproject_geom(geom, src_crs, utm), utm


def latlon_polygon(coords_latlon) -> Polygon:
    """[(lat, lon), ...] → valid WGS84 polygon (lon, lat order inside)."""
    if len(coords_latlon) < 3:
        raise ValueError("A field needs at least 3 corners")
    poly = Polygon([(lo, la) for la, lo in coords_latlon]).buffer(0)
    if poly.is_empty or poly.area == 0:
        raise ValueError("Corners do not form an area — check the order (go around the field)")
    return poly


def xy_to_latlon(points_xy, crs) -> list[tuple[float, float]]:
    t = transformer(crs, WGS84)
    return [(lat, lon) for lon, lat in (t.transform(x, y) for x, y in points_xy)]


def latlon_to_xy(lat: float, lon: float, crs) -> tuple[float, float]:
    return transformer(WGS84, crs).transform(lon, lat)
