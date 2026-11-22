import pytest
from shapely.geometry import LineString

from drone_flight_plan.geo.crs import WGS84, latlon_to_xy, reproject_geom, to_metric
from drone_flight_plan.pipeline import plan_field, tree_zone
from drone_flight_plan.planning.mission import check_home

LINE = LineString([(77.5938, 12.97145), (77.5956, 12.97145)])  # crosses the test field west→east


def _work_segments(m):
    for s in m.sorties:
        for a, b in zip(s.waypoints[:-1], s.waypoints[1:]):
            if a.kind == "work" and b.kind == "work":
                yield LineString([(a.x, a.y), (b.x, b.y)])


def test_tree_zone_radius(field_wgs84):
    z = tree_zone(12.9714, 77.5946, 5.0)
    zm, _ = to_metric(z, WGS84)
    assert zm.area == pytest.approx(3.1416 * 25, rel=0.02)


def test_power_line_clearance_at_work_height(spray_cfg, field_wgs84):
    m = plan_field(field_wgs84, spray_cfg, home_latlon=(12.9708, 77.5938), power_lines=[LINE])
    line_m = reproject_geom(LINE, WGS84, m.crs)
    need = spray_cfg.mission.powerline_clearance_m
    assert min(seg.distance(line_m) for seg in _work_segments(m)) >= need - 0.1
    assert m.metrics["travel_hops"] >= 1  # field split by the wire → hop at travel height
    assert any("power line" in w for w in m.warnings)


def test_trees_are_avoided(spray_cfg, field_wgs84):
    tree = (12.9714, 77.5946, 4.0)
    m = plan_field(field_wgs84, spray_cfg, home_latlon=(12.9708, 77.5938), trees=[tree])
    tz = reproject_geom(tree_zone(*tree), WGS84, m.crs)
    assert min(seg.distance(tz) for seg in _work_segments(m)) >= spray_cfg.no_fly_buffer_m - 0.1
    assert m.metrics["no_fly_zones"]["trees"] == 1


def test_home_checks(cfg, field_wgs84):
    fm, crs = to_metric(field_wgs84, WGS84)
    c = field_wgs84.centroid
    inside = check_home(fm, latlon_to_xy(c.y, c.x, crs), cfg)
    assert inside["inside"] and not inside["ok"]
    ok = check_home(fm, latlon_to_xy(12.9708, 77.5938, crs), cfg)  # ~30 m outside
    assert ok["ok"] and 10 < ok["distance_m"] < 60
    far = check_home(fm, latlon_to_xy(12.9690, 77.5920, crs), cfg)  # ~300 m away
    assert not far["ok"] and "farther" in far["warnings"][0]


def test_area_text_in_metrics(spray_cfg, field_wgs84):
    m = plan_field(field_wgs84, spray_cfg, home_latlon=(12.9708, 77.5938))
    assert m.metrics["field_area_text"].endswith("acres")
    assert m.metrics["field_area"]["acre"] == pytest.approx(m.metrics["field_area_m2"] / 4046.8564224, rel=1e-3)
