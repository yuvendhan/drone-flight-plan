import pytest

from drone_flight_plan.config import drone_catalog, load_config
from drone_flight_plan.units import all_units, convert_area, format_area


def test_area_units():
    assert convert_area(10_000, "ha") == pytest.approx(1)
    assert convert_area(4046.8564224, "acre") == pytest.approx(1)
    assert convert_area(4046.8564224, "cent") == pytest.approx(100)
    assert convert_area(4046.8564224, "guntha") == pytest.approx(40)
    assert format_area(10_000, "acre") == "2.47 acres"
    assert format_area(4046.8564224, "acre") == "1.00 acre"
    assert set(all_units(1)) == {"ha", "acre", "cent", "guntha"}
    with pytest.raises(ValueError):
        convert_area(1, "bigha")


def test_default_unit_is_acre():
    assert load_config("default").display.area_unit == "acre"


def test_every_drone_preset_loads():
    for name, d in drone_catalog().items():
        cfg = load_config("default", drone=name)
        assert cfg.mission.mode in d["jobs"]
        assert cfg.spacing_m > 0


def test_drone_preset_then_override():
    cfg = load_config("default", {"spray": {"rate_l_per_ha": 40}}, drone="Agri sprayer 16 L class")
    assert cfg.spray.tank_l == 16 and cfg.spray.rate_l_per_ha == 40 and cfg.mission.mode == "spraying"
    with pytest.raises(KeyError):
        load_config("default", drone="Imaginary X1")
