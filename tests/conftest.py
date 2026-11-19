import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon

from drone_flight_plan.config import load_config
from drone_flight_plan.geo.crs import to_metric


@pytest.fixture
def cfg():
    return load_config("default")


@pytest.fixture
def spray_cfg():
    return load_config("india_spraying")


@pytest.fixture
def field_wgs84():
    # ~1.2 ha field near Bengaluru (lon, lat)
    return Polygon([(77.5940, 12.9710), (77.5952, 12.9710), (77.5953, 12.9718), (77.5941, 12.9719)])


@pytest.fixture
def field_m(field_wgs84):
    return to_metric(field_wgs84, "EPSG:4326")


@pytest.fixture
def u_field():
    return Polygon([(0, 0), (600, 0), (600, 450), (420, 450), (420, 170), (180, 170), (180, 450), (0, 450)]).difference(
        Point(300, 85).buffer(35)
    )


@pytest.fixture
def sloped_dem(tmp_path, field_m):
    """DEM in the field's UTM CRS rising 10 m per 100 m eastwards."""
    geom, crs = field_m
    minx, miny, maxx, maxy = geom.buffer(300).bounds
    res = 5.0
    W, H = int((maxx - minx) / res), int((maxy - miny) / res)
    xs = minx + (np.arange(W) + 0.5) * res
    z = np.tile(100 + 0.1 * (xs - minx), (H, 1)).astype("float32")
    p = tmp_path / "dem.tif"
    with rasterio.open(
        p,
        "w",
        driver="GTiff",
        width=W,
        height=H,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(minx, maxy, res, res),
    ) as d:
        d.write(z, 1)
    return p
