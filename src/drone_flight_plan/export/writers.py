"""Mission exporters: QGroundControl .plan, KML (Google Earth), GeoJSON, CSV, and a zip bundle."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import simplekml
from shapely.geometry import mapping

from ..config import Config
from ..geo.crs import WGS84, reproject_geom, transformer
from ..planning.geometry import largest_polygon
from ..planning.mission import Mission

# MAVLink command ids
NAV_WAYPOINT, NAV_LAND, NAV_TAKEOFF = 16, 21, 22
DO_CHANGE_SPEED, DO_SET_SERVO = 178, 183
FRAME_GLOBAL_REL_ALT, FRAME_MISSION = 3, 2


def _ll(mission: Mission, x, y):
    lon, lat = transformer(mission.crs, WGS84).transform(x, y)
    return lat, lon


def qgc_plan(mission: Mission, cfg: Config, sortie_idx: int) -> dict:
    """QGroundControl plan for one flight (PX4 / ArduPilot). Heights are relative to home
    and already include terrain, so the drone keeps its height above sloping ground."""
    s = mission.sorties[sortie_idx]
    home_lat, home_lon = _ll(mission, *mission.home_xy)
    items, jid = [], [1]

    def item(cmd, params, frame=FRAME_GLOBAL_REL_ALT, alt=None):
        it = {
            "type": "SimpleItem",
            "autoContinue": True,
            "command": cmd,
            "doJumpId": jid[0],
            "frame": frame,
            "params": params,
            "AMSLAltAboveTerrain": None,
            "Altitude": alt if alt is not None else 0,
            "AltitudeMode": 1,
        }
        jid[0] += 1
        items.append(it)

    spraying = cfg.mission.mode == "spraying"
    pump_on = False
    for w in s.waypoints:
        lat, lon = _ll(mission, w.x, w.y)
        alt = round(w.alt_rel_m, 2)
        if w.kind == "takeoff":
            item(NAV_TAKEOFF, [0, 0, 0, None, lat, lon, alt], alt=alt)
            item(DO_CHANGE_SPEED, [1, cfg.mission.speed_mps, -1, 0, 0, 0, 0], frame=FRAME_MISSION)
        elif w.kind == "land":
            if pump_on:
                item(DO_SET_SERVO, [cfg.spray.servo_channel, cfg.spray.pwm_off, 0, 0, 0, 0, 0], frame=FRAME_MISSION)
                pump_on = False
            item(NAV_LAND, [0, 0, 0, None, lat, lon, 0], alt=0)
        else:
            if spraying and w.spray != pump_on:
                pwm = cfg.spray.pwm_on if w.spray else cfg.spray.pwm_off
                item(DO_SET_SERVO, [cfg.spray.servo_channel, pwm, 0, 0, 0, 0, 0], frame=FRAME_MISSION)
                pump_on = w.spray
            item(NAV_WAYPOINT, [0, 0, 0, None, lat, lon, alt], alt=alt)

    fpoly = largest_polygon(reproject_geom(mission.route.field, mission.crs, WGS84))
    fence = [[la, lo] for lo, la in list(fpoly.exterior.coords)[:-1]]
    return {
        "fileType": "Plan",
        "version": 1,
        "groundStation": "QGroundControl",
        "geoFence": {"circles": [], "polygons": [], "version": 2},
        "rallyPoints": {"points": [], "version": 2},
        "mission": {
            "version": 2,
            "firmwareType": 12,
            "vehicleType": 2,
            "cruiseSpeed": cfg.mission.speed_mps,
            "hoverSpeed": cfg.mission.speed_mps,
            "globalPlanAltitudeMode": 1,
            "plannedHomePosition": [home_lat, home_lon, 0],
            "items": items,
        },
        "drone_flight_plan": {
            "field_polygon_latlon": fence,
            "flight": sortie_idx + 1,
            "of": len(mission.sorties),
            "note": "Field outline included for reference; geofence left empty because take-off/landing "
            "is outside the field.",
        },
    }


def kml_text(mission: Mission, cfg: Config) -> str:
    kml = simplekml.Kml()
    fpoly = largest_polygon(reproject_geom(mission.route.field, mission.crs, WGS84))
    pg = kml.newpolygon(
        name="Field",
        outerboundaryis=list(fpoly.exterior.coords),
        innerboundaryis=[list(r.coords) for r in fpoly.interiors],
    )
    pg.style.polystyle.color = simplekml.Color.changealphaint(60, simplekml.Color.green)
    pg.style.linestyle.color = simplekml.Color.green
    pg.style.linestyle.width = 3
    hl = _ll(mission, *mission.home_xy)
    kml.newpoint(name="Home (take-off / landing)", coords=[(hl[1], hl[0])])
    colors = [
        simplekml.Color.red,
        simplekml.Color.blue,
        simplekml.Color.orange,
        simplekml.Color.purple,
        simplekml.Color.cyan,
        simplekml.Color.yellow,
    ]
    for i, s in enumerate(mission.sorties):
        coords = []
        for w in s.waypoints:
            lat, lon = _ll(mission, w.x, w.y)
            coords.append((lon, lat, max(w.agl_m, 0)))
        ls = kml.newlinestring(name=f"Flight {i + 1} ({s.flight_time_s / 60:.1f} min)", coords=coords)
        ls.altitudemode = simplekml.AltitudeMode.relativetoground
        ls.extrude = 0
        ls.style.linestyle.color = colors[i % len(colors)]
        ls.style.linestyle.width = 2
    return kml.kml()


def geojson(mission: Mission) -> dict:
    feats = [
        {
            "type": "Feature",
            "properties": {
                "name": "field",
                **{k: v for k, v in mission.metrics.items() if isinstance(v, (int, float, str))},
            },
            "geometry": mapping(reproject_geom(mission.route.field, mission.crs, WGS84)),
        }
    ]
    hl = _ll(mission, *mission.home_xy)
    feats.append(
        {
            "type": "Feature",
            "properties": {"name": "home"},
            "geometry": {"type": "Point", "coordinates": [hl[1], hl[0]]},
        }
    )
    for i, s in enumerate(mission.sorties):
        coords = [[_ll(mission, w.x, w.y)[1], _ll(mission, w.x, w.y)[0]] for w in s.waypoints]
        feats.append(
            {
                "type": "Feature",
                "properties": {
                    "name": f"flight {i + 1}",
                    "minutes": round(s.flight_time_s / 60, 2),
                    "spray_l": round(s.spray_l, 2),
                },
                "geometry": {"type": "LineString", "coordinates": coords},
            }
        )
    return {"type": "FeatureCollection", "features": feats}


def waypoints_table(mission: Mission) -> pd.DataFrame:
    rows = []
    for i, s in enumerate(mission.sorties):
        for j, w in enumerate(s.waypoints):
            lat, lon = _ll(mission, w.x, w.y)
            rows.append(
                {
                    "flight": i + 1,
                    "seq": j,
                    "kind": w.kind,
                    "lat": round(lat, 8),
                    "lon": round(lon, 8),
                    "alt_rel_home_m": round(w.alt_rel_m, 2),
                    "agl_m": round(w.agl_m, 2),
                    "spray": w.spray,
                }
            )
    return pd.DataFrame(rows)


def export_all(mission: Mission, cfg: Config, out_dir, name="mission") -> dict:
    """Write every format to out_dir. Returns {format: path or [paths]}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files = {"plan": []}
    for i in range(len(mission.sorties)):
        p = out / f"{name}_flight{i + 1}.plan"
        p.write_text(json.dumps(qgc_plan(mission, cfg, i), indent=2))
        files["plan"].append(p)
    files["kml"] = out / f"{name}.kml"
    files["kml"].write_text(kml_text(mission, cfg))
    files["geojson"] = out / f"{name}.geojson"
    files["geojson"].write_text(json.dumps(geojson(mission)))
    files["csv"] = out / f"{name}_waypoints.csv"
    waypoints_table(mission).to_csv(files["csv"], index=False)
    files["summary"] = out / f"{name}_summary.json"
    files["summary"].write_text(
        json.dumps(
            {"metrics": mission.metrics, "warnings": mission.warnings, "config": cfg.model_dump()},
            indent=2,
            default=str,
        )
    )
    return files


def bundle_zip(mission: Mission, cfg: Config, name="mission") -> bytes:
    """All files in one zip (for downloads in the app)."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i in range(len(mission.sorties)):
            z.writestr(f"{name}_flight{i + 1}.plan", json.dumps(qgc_plan(mission, cfg, i), indent=2))
        z.writestr(f"{name}.kml", kml_text(mission, cfg))
        z.writestr(f"{name}.geojson", json.dumps(geojson(mission)))
        z.writestr(f"{name}_waypoints.csv", waypoints_table(mission).to_csv(index=False))
        z.writestr(
            f"{name}_summary.json",
            json.dumps(
                {"metrics": mission.metrics, "warnings": mission.warnings, "config": cfg.model_dump()},
                indent=2,
                default=str,
            ),
        )
    return buf.getvalue()
