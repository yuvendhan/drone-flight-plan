"""Drone Flight Plan web app.

    pip install -e ".[app,ml]"
    streamlit run app/streamlit_app.py

Flow: 1) get the field (map pick / screenshot / GeoTIFF / draw or KML) → 2) home point + no-fly zones
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
from shapely.geometry import Polygon, mapping, shape
from streamlit_folium import st_folium

from drone_flight_plan import available_profiles, load_config
from drone_flight_plan.export.writers import bundle_zip
from drone_flight_plan.geo.crs import WGS84, to_metric, transformer
from drone_flight_plan.geo.georef import ControlPoint, georeference_by_corners, georeference_image
from drone_flight_plan.geo.io import read_field_file
from drone_flight_plan.geo.tiles import fetch_imagery
from drone_flight_plan.pipeline import pick_field, plan_field
from drone_flight_plan.viz import plot_mission

ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TMP = Path(tempfile.gettempdir()) / "dfp_app"
TMP.mkdir(exist_ok=True)
COLORS = ["#e41a1c", "#377eb8", "#ff7f00", "#984ea3", "#00a6a6", "#a65628"]

st.set_page_config(page_title="Drone Flight Plan — drone field planner", page_icon="🚁", layout="wide")
ss = st.session_state
for k, v in {
    "fields": [],
    "field": None,
    "home": None,
    "no_fly": [],
    "mission": None,
    "center": (12.9716, 77.5946),
    "image_bounds": None,
}.items():
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


@st.cache_resource(show_spinner="Loading AI model…")
def get_predictor(path: str, profile: str):
    from drone_flight_plan.segmentation.infer import Predictor

    return Predictor(path, load_config(profile).segmentation)


def detect(tif_path, ckpt, profile):
    if not ckpt or not Path(ckpt).exists():
        st.warning(
            "No trained model found. Train one (Colab notebook / `drone-flight-plan train`) and set its path in the "
            "sidebar — or use the **Draw / KML** tab to outline the field yourself."
        )
        return
    with st.spinner("Finding fields…"):
        res = get_predictor(ckpt, profile).segment_geotiff(tif_path)
    ss.fields = res["polygons_wgs84"]
    ss.field, ss.mission = None, None
    with __import__("rasterio").open(tif_path) as s:
        t = transformer(s.crs, WGS84)
        w, so, e, n = s.bounds
        (lo1, la1), (lo2, la2) = t.transform(w, so), t.transform(e, n)
    ss.image_bounds = [[la1, lo1], [la2, lo2]]
    ss.center = ((la1 + la2) / 2, (lo1 + lo2) / 2)
    st.success(f"Found {len(ss.fields)} fields. Click your field on the map below.")


# ------------------------------------------------------------------ sidebar: settings
with st.sidebar:
    st.title("🚁 Drone Flight Plan")
    profile = st.selectbox(
        "Profile",
        available_profiles(),
        index=available_profiles().index("india_spraying") if "india_spraying" in available_profiles() else 0,
    )
    base = load_config(profile)
    mode = st.radio("Job", ["mapping", "spraying"], index=0 if base.mission.mode == "mapping" else 1, horizontal=True)
    st.subheader("Flight")
    work = st.number_input("Work height above ground (m)", 1.0, 120.0, float(base.mission.work_agl_m))
    transit = st.number_input("Travel height (m) — above trees/wires", 5.0, 120.0, float(base.mission.transit_agl_m))
    speed = st.number_input("Speed (m/s)", 1.0, 15.0, float(base.mission.speed_mps))
    margin = st.number_input("Gap from field edge (m)", 0.0, 20.0, float(base.mission.edge_margin_m))
    st.subheader("Drone")
    battery = st.number_input("Battery flight time (min)", 3.0, 60.0, float(base.drone.battery_min))
    span = st.number_input("Drone width arm-to-arm (m)", 0.2, 5.0, float(base.drone.span_m))
    over = {
        "mission": {
            "mode": mode,
            "work_agl_m": work,
            "transit_agl_m": transit,
            "speed_mps": speed,
            "edge_margin_m": margin,
        },
        "drone": {"battery_min": battery, "span_m": span},
    }
    if mode == "spraying":
        st.subheader("Sprayer")
        over["spray"] = {
            "swath_m": st.number_input("Spray width (m)", 1.0, 15.0, float(base.spray.swath_m)),
            "tank_l": st.number_input("Tank (L)", 1.0, 100.0, float(base.spray.tank_l)),
            "rate_l_per_ha": st.number_input("Dose (L/ha)", 1.0, 500.0, float(base.spray.rate_l_per_ha)),
        }
        over["mission"]["side_overlap"] = st.slider("Swath overlap", 0.0, 0.5, float(base.mission.side_overlap))
    else:
        over["camera"] = {
            "hfov_deg": st.number_input("Camera field of view (°)", 20.0, 120.0, float(base.camera.hfov_deg))
        }
        over["mission"]["side_overlap"] = st.slider("Photo side overlap", 0.0, 0.85, float(base.mission.side_overlap))
    cfg = load_config(profile, over)
    st.caption(
        f"Stripe spacing **{cfg.spacing_m:.1f} m**"
        + (f" · pump flow **{cfg.flow_l_per_s * 60:.2f} L/min**" if mode == "spraying" else "")
    )
    st.subheader("Terrain")
    dem_choice = st.radio("Ground height", ["Flat", "Copernicus 30 m (internet)", "Upload DEM/DSM"], index=0)
    dem = None
    if dem_choice.startswith("Copernicus"):
        dem = "copernicus"
    elif dem_choice.startswith("Upload"):
        up = st.file_uploader("DEM GeoTIFF", type=["tif", "tiff"], key="dem")
        if up:
            p = TMP / up.name
            p.write_bytes(up.getvalue())
            dem = str(p)
    st.subheader("AI model")
    ckpt = st.text_input("Checkpoint path", base.segmentation.checkpoint)

# ------------------------------------------------------------------ step 1: field
st.header("1 · Get the field")
tab_map, tab_shot, tab_tif, tab_draw = st.tabs(["🛰️ Pick on map", "📸 Screenshot", "🗺️ GeoTIFF", "✏️ Draw / KML"])

with tab_map:
    st.caption(
        "Zoom to the area, draw a **rectangle** around it (toolbar on the left), then press the button. "
        "Imagery is downloaded with GPS already attached. ⚠️ Basemap imagery: personal testing only."
    )
    q = st.text_input("Go to lat, lon", f"{ss.center[0]:.5f}, {ss.center[1]:.5f}", key="goto")
    if parse_latlon(q):
        ss.center = parse_latlon(q)
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
    if st.button("Fetch imagery & find fields", disabled=rect is None):
        w, s_, e, n = shape(rect["geometry"]).bounds
        try:
            with st.spinner("Downloading imagery…"):
                tif = fetch_imagery(
                    (w, s_, e, n), TMP / "area.tif", cfg.imagery.zoom, ESRI, max_tiles=cfg.imagery.max_tiles
                )
            detect(tif, ckpt, profile)
        except Exception as ex:
            st.error(str(ex))

with tab_shot:
    st.caption(
        "Upload a **straight-down, north-up** screenshot (Google Maps / Earth / Apple Maps). In Google Maps, "
        "right-click a spot → click the numbers to copy its GPS."
    )
    img = st.file_uploader("Screenshot", type=["png", "jpg", "jpeg"], key="shot")
    how = st.radio(
        "How to place it on the map", ["Corner GPS (top-left + bottom-right)", "3+ known points"], horizontal=True
    )
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
                detect(r.path, ckpt, profile)
        else:
            st.caption("Pixel x/y: hover the image in any viewer (or open it in Paint) to read pixel positions.")
            import pandas as pd

            pts = st.data_editor(
                pd.DataFrame(
                    {
                        "pixel_x": [0.0, 0.0, 0.0],
                        "pixel_y": [0.0, 0.0, 0.0],
                        "lat": [0.0, 0.0, 0.0],
                        "lon": [0.0, 0.0, 0.0],
                    }
                ),
                num_rows="dynamic",
            )
            good = pts[(pts.lat != 0) & (pts.lon != 0)]
            if st.button("Place screenshot & find fields", disabled=len(good) < 2):
                cps = [ControlPoint(r.pixel_x, r.pixel_y, r.lat, r.lon) for r in good.itertuples()]
                r = georeference_image(ip, cps, TMP / "shot.tif")
                st.info(f"Placed with {r.method}: {r.ground_res_m:.2f} m/pixel, error ≈ {r.rms_error_m:.1f} m")
                detect(r.path, ckpt, profile)

with tab_tif:
    st.caption("Best option: an orthomosaic from your own drone photos (WebODM/OpenDroneMap), or any GeoTIFF.")
    tf = st.file_uploader("GeoTIFF", type=["tif", "tiff"], key="tif")
    if tf and st.button("Find fields"):
        p = TMP / tf.name
        p.write_bytes(tf.getvalue())
        detect(p, ckpt, profile)

with tab_draw:
    st.caption(
        "Draw the field outline yourself (polygon tool), or upload a KML/GeoJSON. Also use this to **fix** an "
        "AI outline: redraw it here."
    )
    kf = st.file_uploader("KML / GeoJSON", type=["kml", "geojson", "json"], key="kml")
    if kf:
        p = TMP / kf.name
        p.write_bytes(kf.getvalue())
        try:
            ss.field = read_field_file(p)
            ss.center = (ss.field.centroid.y, ss.field.centroid.x)
            st.success("Field loaded.")
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
        ss.mission = None

# detected fields → choose
if ss.fields:
    st.subheader("Choose your field")
    m = base_map(ss.center, 18)
    if ss.image_bounds:
        m.fit_bounds(ss.image_bounds)
    for i, p in enumerate(ss.fields):
        add_poly(m, p, "cyan", f"field {i + 1} · {to_metric(p, WGS84)[0].area / 1e4:.2f} ha", 1, 0.05)
    if ss.field is not None:
        add_poly(m, ss.field, "lime", "selected", 3, 0.2)
    out = st_folium(m, height=450, use_container_width=True, key="choosemap", returned_objects=["last_clicked"])
    clk = (out or {}).get("last_clicked")
    if clk:
        f, inside = pick_field(ss.fields, clk["lat"], clk["lng"])
        if ss.field is None or not f.equals(ss.field):
            ss.field, ss.mission = f, None
            if not inside:
                st.warning("Click was not inside a detected field — picked the closest one.")
            st.rerun()

if ss.field is None:
    st.info("No field selected yet.")
    st.stop()

# ------------------------------------------------------------------ step 2: home + no-fly
st.header("2 · Home point and no-fly zones")
st.caption(
    "**Click** the map to set the take-off/landing spot (outside the field: road, bund, open ground). "
    "Draw **polygons/circles** over trees, poles, wells or houses to avoid them."
)
c1, c2 = st.columns([3, 1])
with c1:
    m = base_map((ss.field.centroid.y, ss.field.centroid.x), 18)
    add_poly(m, ss.field, "lime", "field", 3)
    for z in ss.no_fly:
        add_poly(m, z, "red", "no-fly", 2, 0.4)
    if ss.home:
        folium.Marker(ss.home, tooltip="home", icon=folium.Icon(color="black", icon="home")).add_to(m)
    Draw(
        draw_options={
            "polyline": False,
            "rectangle": True,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "polygon": True,
        }
    ).add_to(m)
    out = st_folium(
        m, height=480, use_container_width=True, key="homemap", returned_objects=["last_clicked", "all_drawings"]
    )
with c2:
    clk = (out or {}).get("last_clicked")
    if clk and st.button("📍 Set home here"):
        ss.home, ss.mission = (clk["lat"], clk["lng"]), None
    drawings = (out or {}).get("all_drawings") or []
    if drawings and st.button(f"⛔ Add {len(drawings)} drawn no-fly zone(s)"):
        ss.no_fly += [shape(g["geometry"]).buffer(0) for g in drawings if g["geometry"]["type"] == "Polygon"]
        ss.mission = None
    if ss.no_fly and st.button("Clear no-fly zones"):
        ss.no_fly, ss.mission = [], None
    st.metric("Home", f"{ss.home[0]:.6f}, {ss.home[1]:.6f}" if ss.home else "not set")
    st.metric("No-fly zones", len(ss.no_fly))

# ------------------------------------------------------------------ step 3: plan
st.header("3 · Plan")
if st.button("🚁 Plan mission", type="primary"):
    try:
        with st.spinner("Planning…"):
            ss.mission = plan_field(ss.field, cfg, ss.home, ss.no_fly or None, dem)
            ss.mission_cfg = cfg
    except Exception as ex:
        st.error(f"Could not plan: {ex}")

mission = ss.mission
if mission is None:
    st.stop()
mt = mission.metrics
cols = st.columns(6)
cols[0].metric("Field", f"{mt['field_ha']:.2f} ha")
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
    st.json({k: v for k, v in mt.items()})

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
    "Always check the route on the map before flying. This plan does not know about wires, trees or "
    "airspace unless you marked them. Follow DGCA rules (Digital Sky zones, registration, pilot certificate)."
)
