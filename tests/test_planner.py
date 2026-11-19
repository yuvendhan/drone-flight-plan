import pytest
from shapely import affinity
from shapely.geometry import Point, Polygon

from drone_flight_plan.planning.coverage import compare_strategies, plan_coverage


def test_rectangle_picks_edge_direction(cfg):
    f = affinity.rotate(Polygon([(0, 0), (650, 0), (650, 260), (0, 260)]), 25)
    r = plan_coverage(f, cfg)
    assert r.angle == 25
    assert r.metrics["path_outside_m"] == 0
    assert r.metrics["coverage_pct"] > 95


def test_u_shape_with_pond_stays_inside(cfg, u_field):
    r = plan_coverage(u_field, cfg)
    assert r.metrics["path_outside_m"] == 0
    assert r.metrics["coverage_pct"] > 93


def test_no_fly_zone_avoided(cfg):
    f = Polygon([(0, 0), (400, 0), (400, 300), (0, 300)])
    tree = Point(200, 150).buffer(15)
    r = plan_coverage(f, cfg, no_fly_m=tree)
    from shapely.geometry import LineString

    assert LineString(r.pts).distance(tree) >= cfg.no_fly_buffer_m - 0.1


def test_optimised_never_worse_than_baselines(cfg, u_field):
    df = compare_strategies(u_field, cfg)
    opt = df[df.strategy == "Drone Flight Plan optimised"].iloc[0]
    assert opt.outside_m == 0
    assert opt.time_min <= df.time_min.max()


def test_too_small_field_raises(cfg):
    with pytest.raises(ValueError):
        plan_coverage(Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]), cfg)


def test_spraying_fine_spacing(spray_cfg, field_m):
    geom, crs = field_m
    r = plan_coverage(geom, spray_cfg, crs)
    assert r.metrics["spacing_m"] < 5
    assert r.metrics["coverage_pct"] > 90
