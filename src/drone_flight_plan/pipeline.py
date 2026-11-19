"""High-level API used by the CLI and the app.

from drone_flight_plan import pipeline, load_config
cfg = load_config("india_spraying")
mission = pipeline.plan_field(field_wgs84, cfg, home_latlon=(12.97, 77.59), dem="copernicus")
"""

from __future__ import annotations

from shapely.geometry import Point
from shapely.ops import unary_union

from .config import Config
from .geo.crs import WGS84, latlon_to_xy, reproject_geom, to_metric
from .geo.dem import Terrain
from .planning.coverage import plan_coverage
from .planning.mission import Mission, build_mission


def load_terrain(dem, field_wgs84) -> Terrain:
    """dem: None/'flat' | 'copernicus' | path to GeoTIFF."""
    if dem in (None, "", "flat"):
        return Terrain.flat()
    bounds = field_wgs84.bounds
    if dem == "copernicus":
        return Terrain.copernicus30(bounds)
    return Terrain.from_geotiff(dem, bounds)


def plan_field(field_wgs84, cfg: Config, home_latlon=None, no_fly_wgs84=None, dem=None, angle=None) -> Mission:
    field_m, crs = to_metric(field_wgs84, WGS84)
    no_fly_m = None
    if no_fly_wgs84:
        geoms = no_fly_wgs84 if isinstance(no_fly_wgs84, (list, tuple)) else [no_fly_wgs84]
        no_fly_m = reproject_geom(unary_union(geoms), WGS84, crs)
    terrain = dem if isinstance(dem, Terrain) else load_terrain(dem, field_wgs84)
    home_xy = latlon_to_xy(*home_latlon, crs) if home_latlon else None
    route = plan_coverage(field_m, cfg, crs, no_fly_m, terrain, angle=angle, start=home_xy)
    return build_mission(route, cfg, home_xy, terrain)


def pick_field(polys_wgs84, lat: float, lon: float):
    """Field containing the pin, else the nearest one. Returns (polygon, contained: bool)."""
    if not polys_wgs84:
        raise ValueError("No fields detected")
    pin = Point(lon, lat)
    hit = [p for p in polys_wgs84 if p.contains(pin)]
    if hit:
        return min(hit, key=lambda p: p.area), True
    return min(polys_wgs84, key=lambda p: p.distance(pin)), False
