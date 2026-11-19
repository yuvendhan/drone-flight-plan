import math

import pytest
from shapely.geometry import Polygon

from drone_flight_plan.geo.dem import Terrain
from drone_flight_plan.pipeline import plan_field
from drone_flight_plan.planning.coverage import plan_coverage
from drone_flight_plan.planning.mission import build_mission


def test_spray_litres_and_refills(spray_cfg, field_wgs84):
    m = plan_field(field_wgs84, spray_cfg, home_latlon=(12.9708, 77.5938))
    area_ha = m.metrics["field_ha"]
    expected = area_ha * spray_cfg.spray.rate_l_per_ha
    assert m.metrics["spray_litres"] == pytest.approx(expected, rel=0.15)
    assert all(s.spray_l <= spray_cfg.spray.tank_l + 1e-6 for s in m.sorties)
    assert m.metrics["refills"] == len(m.sorties) - 1 >= math.ceil(expected / spray_cfg.spray.tank_l) - 1


def test_flights_fit_battery(cfg):
    f = Polygon([(0, 0), (1500, 0), (1500, 900), (0, 900)])
    m = build_mission(plan_coverage(f, cfg), cfg, home_xy=(-30, -30))
    assert len(m.sorties) > 1
    assert all(s.flight_time_s <= cfg.drone.battery_min * 60 for s in m.sorties)


def test_takeoff_and_landing_at_home(cfg, field_wgs84):
    m = plan_field(field_wgs84, cfg, home_latlon=(12.9705, 77.5935))
    for s in m.sorties:
        assert s.waypoints[0].kind == "takeoff" and s.waypoints[-1].kind == "land"
        assert (s.waypoints[0].x, s.waypoints[0].y) == pytest.approx(m.home_xy)
        assert (s.waypoints[-1].x, s.waypoints[-1].y) == pytest.approx(m.home_xy)
    assert not any("INSIDE" in w for w in m.warnings)


def test_home_inside_field_warns(cfg, field_wgs84):
    c = field_wgs84.centroid
    m = plan_field(field_wgs84, cfg, home_latlon=(c.y, c.x))
    assert any("INSIDE" in w for w in m.warnings)


def test_terrain_following_keeps_agl(spray_cfg, field_wgs84, sloped_dem):
    m = plan_field(field_wgs84, spray_cfg, home_latlon=(12.9708, 77.5938), dem=str(sloped_dem))
    work = [w for s in m.sorties for w in s.waypoints if w.kind == "work"]
    rel = [w.alt_rel_m for w in work]
    assert max(rel) - min(rel) > 5  # heights change with the slope (~13 m relief across field)...
    assert all(abs(w.agl_m - spray_cfg.mission.work_agl_m) < 1e-6 for w in work)  # ...AGL stays constant
    assert m.metrics["climb_m"] > 0


def test_terrain_contour(field_m, sloped_dem):
    geom, crs = field_m
    t = Terrain.from_geotiff(sloped_dem)
    info = t.slope_and_contour(geom, crs)
    assert info["slope_pct"] == pytest.approx(10, abs=0.5)
    assert info["contour_deg"] == pytest.approx(90, abs=2)  # slope rises east → contours run north-south
