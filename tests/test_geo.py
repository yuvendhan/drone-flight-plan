import io

import numpy as np
import pytest
import rasterio
from PIL import Image

from drone_flight_plan.geo.crs import utm_crs_for
from drone_flight_plan.geo.georef import ControlPoint, georeference_by_corners, georeference_image
from drone_flight_plan.geo.io import parse_kml_polygons, read_raster_rgb01
from drone_flight_plan.geo.tiles import fetch_imagery, ground_res_m, lonlat_to_tile


def test_utm_zone():
    assert utm_crs_for(77.59, 12.97).to_epsg() == 32643
    assert utm_crs_for(-6.26, 53.35).to_epsg() == 32629


def test_tile_math():
    x, y = lonlat_to_tile(0, 0, 1)
    assert (x, y) == pytest.approx((1, 1))
    assert ground_res_m(0, 18) == pytest.approx(0.597, abs=0.01)


def test_fetch_imagery_with_fake_server(tmp_path):
    def fake(url):
        b = io.BytesIO()
        Image.new("RGB", (256, 256), (10, 120, 30)).save(b, "PNG")
        return b.getvalue()

    p = fetch_imagery((77.594, 12.971, 77.5955, 12.972), tmp_path / "a.tif", 18, "x/{z}/{y}/{x}", fetcher=fake)
    with rasterio.open(p) as s:
        assert s.crs.to_epsg() == 3857 and s.count == 3
    rgb, tr, crs = read_raster_rgb01(p, target_res_m=1.0)
    assert rgb.shape[0] == 3 and abs(rgb[1].mean() - 120 / 255) < 0.02


def test_georef_corners_roundtrip(tmp_path):
    img = tmp_path / "shot.png"
    Image.fromarray(np.zeros((400, 600, 3), np.uint8)).save(img)
    r = georeference_by_corners(img, (12.98, 77.58), (12.97, 77.60), tmp_path / "shot.tif")
    with rasterio.open(r.path) as s:
        from pyproj import Transformer

        t = Transformer.from_crs(s.crs, 4326, always_xy=True)
        lon, lat = t.transform(*s.xy(0, 0, offset="ul"))
        assert (lat, lon) == pytest.approx((12.98, 77.58), abs=1e-6)
        lon, lat = t.transform(*s.xy(400, 600, offset="ul"))
        assert (lat, lon) == pytest.approx((12.97, 77.60), abs=1e-6)


def test_georef_affine_error_report(tmp_path):
    img = tmp_path / "p.png"
    Image.fromarray(np.zeros((100, 100, 3), np.uint8)).save(img)
    pts = [
        ControlPoint(0, 0, 12.98, 77.58),
        ControlPoint(100, 0, 12.98, 77.59),
        ControlPoint(0, 100, 12.97, 77.58),
        ControlPoint(100, 100, 12.97, 77.59),
    ]
    r = georeference_image(img, pts, tmp_path / "p.tif")
    assert r.method == "affine" and r.rms_error_m < 25


def test_parse_kml():
    kml = """<kml><Placemark><Polygon><outerBoundaryIs><LinearRing><coordinates>
    77.1,12.1,0 77.2,12.1,0 77.2,12.2,0 77.1,12.2,0 77.1,12.1,0</coordinates></LinearRing></outerBoundaryIs>
    </Polygon></Placemark></kml>"""
    (p,) = parse_kml_polygons(kml)
    assert p.bounds == pytest.approx((77.1, 12.1, 77.2, 12.2))
