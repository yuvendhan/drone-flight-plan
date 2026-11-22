"""Drone Flight Plan web app.

    pip install -e ".[app,onnx]"
    streamlit run app/streamlit_app.py

Flow: 1) find the field (map search / draw / GeoTIFF / screenshot) → 2) home + obstacles (trees, power lines)
→ 3) plan → 4) download mission files.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import streamlit as st
from folium.plugins import Draw
from shapely.geometry import LineString, Polygon, mapping, shape
from streamlit_folium import st_folium

from drone_flight_plan import load_config
from drone_flight_plan.config import _deep_merge, drone_catalog
from drone_flight_plan.export.writers import bundle_zip
from drone_flight_plan.geo.crs import WGS84, latlon_to_xy, to_metric, transformer
from drone_flight_plan.geo.georef import ControlPoint, georeference_by_corners, georeference_image
from drone_flight_plan.geo.io import read_field_file
from drone_flight_plan.geo.tiles import fetch_imagery
from drone_flight_plan.pipeline import pick_field, plan_field, tree_zone
from drone_flight_plan.planning.mission import check_home
from drone_flight_plan.units import format_area
from drone_flight_plan.viz import plot_mission

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TMP = Path(tempfile.gettempdir()) / "dfp_app"
TMP.mkdir(exist_ok=True)
COLORS = ["#e41a1c", "#377eb8", "#ff7f00", "#984ea3", "#00a6a6", "#a65628"]
UNITS = {"acre": "Acres", "ha": "Hectares", "cent": "Cents", "guntha": "Guntha"}

st.set_page_config(page_title="Drone Flight Plan", page_icon="🚁", layout="wide")
ss = st.session_state
DEFAULTS = {
    "fields": [],
    "field": None,
    "home": None,
    "no_fly": [],
    "trees": [],
    "lines": [],
    "mission": None,
    "center": (12.9716, 77.5946),
    "image_bounds": None,
    "want_location": False,
    "last_click": None,
}
for k, v in DEFAULTS.items():
    ss.setdefault(k, v)


# ------------------------------------------------------------------ helpers
def parse_latlon(text: str):
    try:
        a, b = (float(x) for x in text.replace(" ", "").split(","))
        return a, b
    except Exception:
        return None


def base_map(center, zoom=17):
    m = folium.Map(location=center, zoom_start=zoom, max_zoom=21, tiles=None, control_scale=True)
    folium.TileLayer(ESRI, attr="Esri World Imagery", name="Satellite", max_zoom=21, max_native_zoom=19).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Map").add_to(m)
    return m


def add_poly(m, poly, color, tooltip, weight=2, fill=0.15):
    folium.GeoJson(
        mapping(poly),
        style_function=lambda _x, c=color, w=weight, f=fill: {"color": c, "weight": w, "fillOpacity": f},
        tooltip=tooltip,
    ).add_to(m)


def area_text(poly_wgs84, unit):
    return format_area(to_metric(poly_wgs84, WGS84)[0].area, unit)


@st.cache_data(show_spinner=False, ttl=3600)
def search_place(q: str):
    """Village / town search with OpenStreetMap Nominatim."""
    import requests

    r = requests.get(
        "https://nominatim.openstreetmap.org/search",
        timeout=15,
        params={"q": q, "format": "json", "limit": 5},
        headers={"User-Agent": "drone-flight-plan/0.3 (github.com/yuvendhan/drone-flight-plan)"},
    )
    r.raise_for_status()
    return [(d["display_name"], float(d["lat"]), float(d["lon"])) for d in r.json()]


@st.cache_resource(show_spinner="Getting the AI model ready (first time only)…")
def get_predictor(seg_json: str):
    from drone_flight_plan.config import SegmentationConfig
    from drone_flight_plan.hub import resolve_model
    from drone_flight_plan.segmentation.infer import Predictor

    sc = SegmentationConfig.model_validate_json(seg_json)
    return Predictor(str(resolve_model(sc)), sc)


def predictor_or_none(cfg):
    try:
        return get_predictor(cfg.segmentation.model_dump_json()), None
    except Exception as e:
        return None, str(e)


def detect(tif_path, cfg):
    pred, err = predictor_or_none(cfg)
    if pred is None:
        st.warning(
            "The AI model is not available, so fields can't be found automatically. "
            "Use **✏️ Draw / KML** to outline the field yourself."
        )
        st.caption(err)
        return
    with st.spinner("Finding fields…"):
        res = pred.segment_geotiff(tif_path)
    ss.fields, ss.field, ss.mission = res["polygons_wgs84"], None, None
    import rasterio

    with rasterio.open(tif_path) as s:
        t = transformer(s.crs, WGS84)
        w, so, e, n = s.bounds
        (lo1, la1), (lo2, la2) = t.transform(w, so), t.transform(e, n)
    ss.image_bounds = [[la1, lo1], [la2, lo2]]
    ss.center = ((la1 + la2) / 2, (lo1 + lo2) / 2)
    st.success(f"Found {len(ss.fields)} fields. Click your field on the map below.")


def reset_plan():
    ss.mission = None


# ------------------------------------------------------------------ sidebar: settings
with st.sidebar:
    st.title("🚁 Drone Flight Plan")
    cat = drone_catalog()
    drone_names = list(cat) + ["Custom"]
    drone = st.selectbox(
        "Your drone",
        drone_names,
        index=drone_names.index("Agri sprayer 10 L class"),
        help="Generic drone classes with typical values. Pick 'Custom' to type your own.",
    )
    if drone != "Custom":
        st.caption(f"ℹ️ {cat[drone]['note']} Example values — check your drone's manual.")
    jobs = cat[drone]["jobs"] if drone != "Custom" else ["mapping", "spraying"]
    mode = st.radio(
        "Job", jobs, horizontal=True, format_func=lambda j: {"mapping": "📷 Mapping", "spraying": "💧 Spraying"}[j]
    )
    unit = st.selectbox("Area unit", list(UNITS), format_func=UNITS.get)
    preset = None if drone == "Custom" else drone
    profile = "default" if preset else ("india_mapping" if mode == "mapping" else "india_spraying")
    preset_mode = cat[drone]["settings"]["mission"]["mode"] if preset else mode
    mode_fix = (
        {}
        if mode == preset_mode
        else {
            "mission": (
                {"mode": "mapping", "work_agl_m": 30, "transit_agl_m": 45, "side_overlap": 0.7, "speed_mps": 5}
                if mode == "mapping"
                else {
                    "mode": "spraying",
                    "work_agl_m": 2.5,
                    "transit_agl_m": 15,
                    "side_overlap": 0.1,
                    "speed_mps": 4.5,
                    "edge_margin_m": 0.5,
                }
            )
        }
    )
    base = load_config(profile, mode_fix, drone=preset)

    over = {"display": {"area_unit": unit}, "mission": {"mode": mode}}
    if mode == "spraying":
        dose = st.number_input(
            "Dose (litres per hectare)", 1.0, 500.0, float(base.spray.rate_l_per_ha), help="From the product label."
        )
        over["spray"] = {"rate_l_per_ha": dose}
    with st.expander("More options", expanded=drone == "Custom"):
        m_ = over["mission"]
        m_["work_agl_m"] = st.number_input("Work height above ground (m)", 1.0, 120.0, float(base.mission.work_agl_m))
        m_["transit_agl_m"] = st.number_input(
            "Travel height (m), above trees/wires", 5.0, 120.0, float(base.mission.transit_agl_m)
        )
        m_["speed_mps"] = st.number_input("Speed (m/s)", 1.0, 15.0, float(base.mission.speed_mps))
        m_["edge_margin_m"] = st.number_input("Gap from field edge (m)", 0.0, 20.0, float(base.mission.edge_margin_m))
        m_["powerline_clearance_m"] = st.number_input(
            "Distance from power lines (m)", 3.0, 50.0, float(base.mission.powerline_clearance_m)
        )
        over["drone"] = {
            "battery_min": st.number_input("Battery flight time (min)", 3.0, 60.0, float(base.drone.battery_min)),
            "span_m": st.number_input("Drone width arm-to-arm (m)", 0.2, 5.0, float(base.drone.span_m)),
        }
        if mode == "spraying":
            over["spray"].update(
                {
                    "swath_m": st.number_input("Spray width (m)", 1.0, 15.0, float(base.spray.swath_m)),
                    "tank_l": st.number_input("Tank (L)", 1.0, 100.0, float(base.spray.tank_l)),
                }
            )
            m_["side_overlap"] = st.slider("Swath overlap", 0.0, 0.5, float(base.mission.side_overlap))
        else:
            over["camera"] = {
                "hfov_deg": st.number_input("Camera field of view (°)", 20.0, 120.0, float(base.camera.hfov_deg))
            }
            m_["side_overlap"] = st.slider("Photo side overlap", 0.0, 0.85, float(base.mission.side_overlap))
        dem_choice = st.radio("Ground height", ["Flat", "Copernicus 30 m (internet)", "Upload DEM/DSM"])
        dem = None
        if dem_choice.startswith("Copernicus"):
            dem = "copernicus"
        elif dem_choice.startswith("Upload"):
            up = st.file_uploader("DEM GeoTIFF", type=["tif", "tiff"], key="dem")
            if up:
                p = TMP / up.name
                p.write_bytes(up.getvalue())
                dem = str(p)
        hf_repo = st.text_input("AI model (Hugging Face repo)", base.segmentation.hf_repo or "")
        over["segmentation"] = {"hf_repo": hf_repo or None}
    cfg = load_config(profile, _deep_merge(mode_fix, over), drone=preset)
    st.caption(
        f"Stripe spacing **{cfg.spacing_m:.1f} m**"
        + (f" · pump flow **{cfg.flow_l_per_s * 60:.2f} L/min**" if mode == "spraying" else "")
    )
    pred, _ = predictor_or_none(cfg)
    st.caption(f"AI model: {'✅ ready (' + pred.backend + ')' if pred else '⚠️ not available — draw fields manually'}")

# ------------------------------------------------------------------ step 1: field
st.header("1 · Find your field")
tab_map, tab_draw, tab_tif, tab_shot = st.tabs(
    ["🛰️ Search & pick on map", "✏️ Draw / KML", "🗺️ Drone photo (GeoTIFF)", "📸 Screenshot (other)"]
)

with tab_map:
    c1, c2 = st.columns([3, 1])
    q = c1.text_input("Search village / town, or type lat, lon", placeholder="e.g. Thanjavur  or  10.787, 79.137")
    if q:
        ll = parse_latlon(q)
        if ll:
            ss.center = ll
        else:
            try:
                hits = search_place(q)
                if hits:
                    pick = c2.selectbox("Results", range(len(hits)), format_func=lambda i: hits[i][0][:60])
                    ss.center = hits[pick][1:]
                else:
                    st.info("No place found. Try another spelling, a nearby town, or lat, lon.")
            except Exception:
                st.info("Place search needs internet. Type lat, lon instead.")
    st.caption(
        "Zoom to your field, draw a **rectangle** around it (toolbar on the left), then press the button. "
        "⚠️ Satellite basemap: personal testing only."
    )
    m = base_map(ss.center)
    Draw(
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
        edit_options={"edit": False},
    ).add_to(m)
    out = st_folium(m, height=450, use_container_width=True, key="pickmap", returned_objects=["last_active_drawing"])
    rect = (out or {}).get("last_active_drawing")
    if st.button("🔍 Find fields in this box", disabled=rect is None, type="primary"):
        w, s_, e, n = shape(rect["geometry"]).bounds
        try:
            with st.spinner("Downloading satellite picture…"):
                tif = fetch_imagery(
                    (w, s_, e, n), TMP / "area.tif", cfg.imagery.zoom, ESRI, max_tiles=cfg.imagery.max_tiles
                )
            detect(tif, cfg)
        except Exception as ex:
            st.error(str(ex))

with tab_draw:
    st.caption("Draw the outline yourself (polygon tool) or upload a KML/GeoJSON. Use this to **fix** an AI outline.")
    kf = st.file_uploader("KML / GeoJSON", type=["kml", "geojson", "json"], key="kml")
    if kf:
        p = TMP / kf.name
        p.write_bytes(kf.getvalue())
        try:
            ss.field = read_field_file(p)
            ss.center = (ss.field.centroid.y, ss.field.centroid.x)
            reset_plan()
            st.success(f"Field loaded: {area_text(ss.field, unit)}")
        except Exception as ex:
            st.error(str(ex))
    m = base_map(ss.center)
    if ss.field is not None:
        add_poly(m, ss.field, "lime", "current field", 3)
    Draw(
        draw_options={
            "polyline": False,
            "rectangle": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": True,
        }
    ).add_to(m)
    out = st_folium(m, height=450, use_container_width=True, key="drawmap", returned_objects=["last_active_drawing"])
    d = (out or {}).get("last_active_drawing")
    if d and d["geometry"]["type"] == "Polygon" and st.button("Use drawn outline"):
        ss.field = Polygon(d["geometry"]["coordinates"][0]).buffer(0)
        reset_plan()

with tab_tif:
    st.caption("Best accuracy: an orthomosaic from your own drone photos (WebODM / OpenDroneMap), or any GeoTIFF.")
    tf = st.file_uploader("GeoTIFF", type=["tif", "tiff"], key="tif")
    if tf and st.button("Find fields"):
        p = TMP / tf.name
        p.write_bytes(tf.getvalue())
        detect(p, cfg)

with tab_shot:
    st.caption(
        "A **straight-down, north-up** screenshot from Google Maps / Earth. In Google Maps, right-click a spot "
        "→ click the numbers to copy its GPS."
    )
    img = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"], key="shot")
    how = st.radio("Place it using", ["Corner GPS (top-left + bottom-right)", "3+ known points"], horizontal=True)
    if img:
        ip = TMP / img.name
        ip.write_bytes(img.getvalue())
        st.image(img, use_container_width=True)
        if how.startswith("Corner"):
            c1, c2 = st.columns(2)
            tl = parse_latlon(c1.text_input("Top-left corner lat, lon", ""))
            br = parse_latlon(c2.text_input("Bottom-right corner lat, lon", ""))
            if st.button("Place screenshot & find fields", disabled=not (tl and br)):
                r = georeference_by_corners(ip, tl, br, TMP / "shot.tif")
                st.info(f"Placed: {r.ground_res_m:.2f} m per pixel")
                detect(r.path, cfg)
        else:
            import pandas as pd

            pts = st.data_editor(
                pd.DataFrame({"pixel_x": [0.0] * 3, "pixel_y": [0.0] * 3, "lat": [0.0] * 3, "lon": [0.0] * 3}),
                num_rows="dynamic",
            )
            good = pts[(pts.lat != 0) & (pts.lon != 0)]
            if st.button("Place screenshot & find fields", disabled=len(good) < 2):
                cps = [ControlPoint(r.pixel_x, r.pixel_y, r.lat, r.lon) for r in good.itertuples()]
                r = georeference_image(ip, cps, TMP / "shot.tif")
                st.info(f"Placed with {r.method}: {r.ground_res_m:.2f} m/pixel, error ≈ {r.rms_error_m:.1f} m")
                detect(r.path, cfg)

if ss.fields:
    st.subheader("Click your field")
    m = base_map(ss.center, 18)
    if ss.image_bounds:
        m.fit_bounds(ss.image_bounds)
    for i, p in enumerate(ss.fields):
        add_poly(m, p, "cyan", f"field {i + 1} · {area_text(p, unit)}", 1, 0.05)
    if ss.field is not None:
        add_poly(m, ss.field, "lime", "selected", 3, 0.2)
    out = st_folium(m, height=450, use_container_width=True, key="choosemap", returned_objects=["last_clicked"])
    clk = (out or {}).get("last_clicked")
    if clk:
        f, inside = pick_field(ss.fields, clk["lat"], clk["lng"])
        if ss.field is None or not f.equals(ss.field):
            ss.field = f
            reset_plan()
            if not inside:
                st.warning("Click was not inside a detected field — picked the closest one.")
            st.rerun()

if ss.field is None:
    st.info("No field selected yet.")
    st.stop()
st.success(f"Field: **{area_text(ss.field, unit)}**")

# ------------------------------------------------------------------ step 2: home + obstacles
st.header("2 · Take-off spot and obstacles")
c1, c2 = st.columns([3, 1])
with c1:
    st.caption(
        "**Click** the map, then use the buttons on the right. Draw **lines** over power lines and "
        "**polygons** over houses, wells or ponds."
    )
    m = base_map((ss.field.centroid.y, ss.field.centroid.x), 18)
    add_poly(m, ss.field, "lime", "field", 3)
    for z in ss.no_fly:
        add_poly(m, z, "red", "no-fly", 2, 0.4)
    for t in ss.trees:
        add_poly(m, tree_zone(*t), "darkgreen", "tree", 2, 0.5)
    for ln in ss.lines:
        folium.PolyLine(
            [(y, x) for x, y in ln.coords], color="yellow", weight=4, dash_array="8", tooltip="power line"
        ).add_to(m)
    if ss.home:
        folium.Marker(ss.home, tooltip="home", icon=folium.Icon(color="black", icon="home")).add_to(m)
    Draw(
        draw_options={
            "polyline": True,
            "rectangle": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": True,
        }
    ).add_to(m)
    out = st_folium(
        m, height=500, use_container_width=True, key="homemap", returned_objects=["last_clicked", "all_drawings"]
    )
with c2:
    clk = (out or {}).get("last_clicked")
    if clk:
        ss.last_click = (clk["lat"], clk["lng"])
    st.markdown("**Home (take-off / landing)**")
    if st.button("📍 Use my location"):
        ss.want_location = True
    if ss.want_location:
        from streamlit_js_eval import get_geolocation

        loc = get_geolocation()
        if loc:
            coords = loc.get("coords", loc)
            ss.home = (float(coords["latitude"]), float(coords["longitude"]))
            ss.want_location = False
            reset_plan()
            st.rerun()
        st.caption("Allow location access in your browser… (works on localhost or https)")
    if st.button("🏠 Home at clicked spot", disabled=ss.last_click is None):
        ss.home = ss.last_click
        reset_plan()
    if ss.home:
        fm, crs = to_metric(ss.field, WGS84)
        chk = check_home(fm, latlon_to_xy(*ss.home, crs), cfg)
        (st.success if chk["ok"] else st.warning)(
            f"{chk['distance_m']:.0f} m from the field" if not chk["inside"] else "Inside the field!"
        )
        for w in chk["warnings"]:
            st.caption(w)

    st.markdown("**Obstacles**")
    radius = st.number_input("Tree canopy radius (m)", 1.0, 20.0, float(cfg.mission.tree_radius_m))
    if st.button("🌳 Tree at clicked spot", disabled=ss.last_click is None):
        ss.trees.append((*ss.last_click, radius))
        reset_plan()
        st.rerun()
    drawings = (out or {}).get("all_drawings") or []
    polys = [shape(g["geometry"]).buffer(0) for g in drawings if g["geometry"]["type"] == "Polygon"]
    lines = [LineString(g["geometry"]["coordinates"]) for g in drawings if g["geometry"]["type"] == "LineString"]
    if (polys or lines) and st.button(f"⛔ Save drawn: {len(lines)} power line(s), {len(polys)} area(s)"):
        ss.no_fly += polys
        ss.lines += lines
        reset_plan()
        st.rerun()
    st.write(f"🌳 {len(ss.trees)} trees · ⚡ {len(ss.lines)} power lines · ⛔ {len(ss.no_fly)} areas")
    if (ss.trees or ss.lines or ss.no_fly) and st.button("Clear obstacles"):
        ss.trees, ss.lines, ss.no_fly = [], [], []
        reset_plan()
        st.rerun()

# ------------------------------------------------------------------ step 3: plan
st.header("3 · Plan")
if st.button("🚁 Plan mission", type="primary"):
    try:
        with st.spinner("Planning…"):
            ss.mission = plan_field(
                ss.field, cfg, ss.home, ss.no_fly or None, dem, trees=ss.trees or None, power_lines=ss.lines or None
            )
            ss.mission_cfg = cfg
    except Exception as ex:
        st.error(f"Could not plan: {ex}")

mission = ss.mission
if mission is None:
    st.stop()
mt = mission.metrics
cols = st.columns(6)
cols[0].metric("Field", mt["field_area_text"])
cols[1].metric("Coverage", f"{mt['coverage_pct']:.1f} %")
cols[2].metric("Flights", mt["flights"])
cols[3].metric("Total flight time", f"{mt['total_flight_min']} min")
cols[4].metric("Stripe direction", f"{mt['angle_deg']}°")
if mt.get("spray_litres") is not None:
    cols[5].metric("Spray needed", f"{mt['spray_litres']:.1f} L ({mt['refills']} refills)")
else:
    cols[5].metric("Max height", f"{mt['max_agl_m']} m")
for w in mission.warnings:
    st.warning(w)

c1, c2 = st.columns(2)
with c1:
    m = base_map((ss.field.centroid.y, ss.field.centroid.x), 18)
    add_poly(m, ss.field, "lime", "field", 3, 0.05)
    for ln in ss.lines:
        folium.PolyLine([(y, x) for x, y in ln.coords], color="yellow", weight=4, dash_array="8").add_to(m)
    t = transformer(mission.crs, WGS84)
    for i, s in enumerate(mission.sorties):
        pts = [(la, lo) for lo, la in (t.transform(w.x, w.y) for w in s.waypoints)]
        folium.PolyLine(
            pts, color=COLORS[i % len(COLORS)], weight=2, tooltip=f"flight {i + 1}: {s.flight_time_s / 60:.1f} min"
        ).add_to(m)
    lo, la = t.transform(*mission.home_xy)
    folium.Marker((la, lo), tooltip="home", icon=folium.Icon(color="black", icon="home")).add_to(m)
    st_folium(m, height=480, use_container_width=True, key="resultmap", returned_objects=[])
with c2:
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_mission(mission, ax)
    st.pyplot(fig)

with st.expander("Details"):
    st.json(mt)

# ------------------------------------------------------------------ step 4: download
st.header("4 · Download")
st.download_button(
    "⬇️ Mission files (.zip: QGroundControl .plan per flight, KML, GeoJSON, CSV, summary)",
    bundle_zip(mission, ss.mission_cfg, "dfp_mission"),
    "dfp_mission.zip",
    "application/zip",
)
st.download_button(
    "⬇️ Field outline (GeoJSON)",
    json.dumps({"type": "Feature", "properties": {}, "geometry": mapping(ss.field)}),
    "field.geojson",
    "application/geo+json",
)
st.caption(
    "Always check the route before flying. It only avoids the trees, wires and areas you marked. "
    "Follow DGCA rules (Digital Sky zones, drone registration, Remote Pilot Certificate)."
)
