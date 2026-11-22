"""High-level API used by the CLI and the app.

from drone_flight_plan import pipeline, load_config
cfg = load_config("india_spraying")
mission = pipeline.plan_field(field_wgs84, cfg, home_latlon=(12.97, 77.59),
                              trees=[(12.9712, 77.5945)], power_lines=[line_wgs84], dem="copernicus")
"""

from __future__ import annotations

from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .config import Config
from .geo.crs import WGS84, latlon_to_xy, reproject_geom, to_metric
from .geo.dem import Terrain
from .planning.coverage import plan_coverage
from .planning.mission import Mission, build_mission
from .units import all_units, format_area


def load_terrain(dem, field_wgs84) -> Terrain:
    """dem: None/'flat' | 'copernicus' | path to GeoTIFF."""
    if dem in (None, "", "flat"):
        return Terrain.flat()
    bounds = field_wgs84.bounds
    if dem == "copernicus":
        return Terrain.copernicus30(bounds)
    return Terrain.from_geotiff(dem, bounds)


def tree_zone(lat: float, lon: float, radius_m: float):
    """Circle of radius_m metres around a tree, as a WGS84 polygon."""
    p, crs = to_metric(Point(lon, lat), WGS84)
    return reproject_geom(p.buffer(radius_m, 24), crs, WGS84)


def _as_list(x):
    if not x:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def plan_field(
    field_wgs84,
    cfg: Config,
    home_latlon=None,
    no_fly_wgs84=None,
    dem=None,
    angle=None,
    trees=None,
    power_lines=None,
) -> Mission:
    """
    field_wgs84 : field polygon (lon/lat)
    no_fly_wgs84: polygons to avoid (buildings, wells, ponds...)
    trees       : [(lat, lon) or (lat, lon, radius_m)] — avoided with canopy radius + clearance
    power_lines : LineStrings (lon/lat) — avoided by powerline_clearance_m + clearance
    """
    field_m, crs = to_metric(field_wgs84, WGS84)
    zones = [reproject_geom(g, WGS84, crs) for g in _as_list(no_fly_wgs84)]
    for t in _as_list(trees):
        lat, lon, *r = t
        zones.append(reproject_geom(tree_zone(lat, lon, r[0] if r else cfg.mission.tree_radius_m), WGS84, crs))
    lines_m = [reproject_geom(ln, WGS84, crs) for ln in _as_list(power_lines)]
    zones += [ln.buffer(max(cfg.mission.powerline_clearance_m, 0.1)) for ln in lines_m]
    no_fly_m = unary_union(zones) if zones else None

    terrain = dem if isinstance(dem, Terrain) else load_terrain(dem, field_wgs84)
    home_xy = latlon_to_xy(*home_latlon, crs) if home_latlon else None
    route = plan_coverage(field_m, cfg, crs, no_fly_m, terrain, angle=angle, start=home_xy)
    mission = build_mission(route, cfg, home_xy, terrain)

    # power lines crossed by travel legs (take-off → field, field → home)
    if lines_m:
        danger = unary_union([ln.buffer(cfg.mission.powerline_clearance_m) for ln in lines_m])
        crossings = 0
        for s in mission.sorties:
            for a, b in zip(s.waypoints[:-1], s.waypoints[1:]):
                travel = a.kind in ("takeoff", "transit") and b.kind == "transit"
                if travel and (a.x, a.y) != (b.x, b.y) and LineString([(a.x, a.y), (b.x, b.y)]).intersects(danger):
                    crossings += 1
        if crossings:
            mission.warnings.append(
                f"{crossings} travel leg(s) cross a power line zone at travel height "
                f"{cfg.mission.transit_agl_m:.0f} m. Make sure that is well above the wires, or move the home point."
            )

    # areas in every unit + the chosen display unit
    work_area = route.workable.area
    mission.metrics["field_area_m2"] = round(field_m.area, 1)
    mission.metrics["field_area"] = all_units(field_m.area)
    mission.metrics["field_area_text"] = format_area(field_m.area, cfg.display.area_unit)
    mission.metrics["sprayable_area_text"] = format_area(work_area, cfg.display.area_unit)
    mission.metrics["no_fly_zones"] = {
        "polygons": len(_as_list(no_fly_wgs84)),
        "trees": len(_as_list(trees)),
        "power_lines": len(lines_m),
    }
    return mission


def pick_field(polys_wgs84, lat: float, lon: float):
    """Field containing the pin, else the nearest one. Returns (polygon, contained: bool)."""
    if not polys_wgs84:
        raise ValueError("No fields detected")
    pin = Point(lon, lat)
    hit = [p for p in polys_wgs84 if p.contains(pin)]
    if hit:
        return min(hit, key=lambda p: p.area), True
    return min(polys_wgs84, key=lambda p: p.distance(pin)), False
