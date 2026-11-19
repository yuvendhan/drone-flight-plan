import math

import pytest
from pydantic import ValidationError

from drone_flight_plan.config import available_profiles, load_config


def test_profiles_load():
    for p in available_profiles():
        load_config(p)
    assert {"default", "india_mapping", "india_spraying"} <= set(available_profiles())


def test_mapping_spacing(cfg):
    fp = 2 * 40 * math.tan(math.radians(73.7) / 2)
    assert cfg.footprint_m == pytest.approx(fp)
    assert cfg.spacing_m == pytest.approx(fp * 0.7)


def test_spraying_derived(spray_cfg):
    assert spray_cfg.footprint_m == spray_cfg.spray.swath_m
    # flow (L/s) = dose(L/ha) * spacing(m) * speed(m/s) / 10000
    exp = spray_cfg.spray.rate_l_per_ha * spray_cfg.spacing_m * spray_cfg.mission.speed_mps / 1e4
    assert spray_cfg.flow_l_per_s == pytest.approx(exp)


def test_overrides_and_validation():
    c = load_config("default", {"mission": {"speed_mps": 3, "transit_agl_m": 10}})
    assert c.mission.speed_mps == 3
    assert c.mission.transit_agl_m == c.mission.work_agl_m  # raised to work height
    with pytest.raises(ValidationError):
        load_config("default", {"mission": {"speed_mps": -1}})
    with pytest.raises(FileNotFoundError):
        load_config("nope")
