import io
import json
import zipfile

from drone_flight_plan.export.writers import bundle_zip, export_all, qgc_plan
from drone_flight_plan.pipeline import plan_field


def test_qgc_plan_structure(cfg, field_wgs84, tmp_path):
    m = plan_field(field_wgs84, cfg, home_latlon=(12.9705, 77.5935))
    plan = qgc_plan(m, cfg, 0)
    cmds = [i["command"] for i in plan["mission"]["items"]]
    assert cmds[0] == 22 and cmds[1] == 178 and cmds[-1] == 21  # takeoff, speed, ..., land
    assert plan["mission"]["plannedHomePosition"][0] == __import__("pytest").approx(12.9705, abs=1e-6)
    files = export_all(m, cfg, tmp_path)
    assert all(p.exists() for p in files["plan"]) and files["kml"].exists()
    json.loads(files["geojson"].read_text())


def test_spray_pump_toggles(spray_cfg, field_wgs84):
    m = plan_field(field_wgs84, spray_cfg, home_latlon=(12.9708, 77.5938))
    items = qgc_plan(m, spray_cfg, 0)["mission"]["items"]
    servo = [i["params"][1] for i in items if i["command"] == 183]
    assert servo and servo[0] == spray_cfg.spray.pwm_on and servo[-1] == spray_cfg.spray.pwm_off
    # pump must be off before landing
    land_idx = max(k for k, i in enumerate(items) if i["command"] == 21)
    last_servo = max(k for k, i in enumerate(items) if i["command"] == 183)
    assert last_servo < land_idx


def test_zip_bundle(spray_cfg, field_wgs84):
    m = plan_field(field_wgs84, spray_cfg, home_latlon=(12.9708, 77.5938))
    z = zipfile.ZipFile(io.BytesIO(bundle_zip(m, spray_cfg, "f")))
    names = z.namelist()
    assert "f.kml" in names and "f_summary.json" in names
    assert sum(n.endswith(".plan") for n in names) == len(m.sorties)
