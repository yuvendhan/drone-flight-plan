"""Model probabilities → one polygon per field."""

from __future__ import annotations

import numpy as np
from rasterio import features
from scipy.ndimage import binary_erosion
from shapely.geometry import shape
from shapely.ops import unary_union
from skimage.measure import label as sk_label
from skimage.morphology import disk
from skimage.segmentation import watershed

from ..planning.geometry import polygons_of


def probs_to_instances(prob: np.ndarray, seed_thr=0.5, boundary_thr=0.4, min_px=150) -> np.ndarray:
    """prob: (3,H,W) [bg, field, boundary]. Seeds = confident field insides; watershed grows them and stops
    at predicted boundaries, so touching fields stay separate."""
    interior, boundary = prob[1], prob[2]
    field = (interior + boundary) > 0.5
    seeds = binary_erosion((interior > seed_thr) & (boundary < boundary_thr), structure=disk(2))
    markers = sk_label(seeds)
    sizes = np.bincount(markers.ravel())
    markers[np.isin(markers, np.where(sizes < max(1, min_px // 3))[0])] = 0
    lab = watershed(boundary, markers, mask=field)
    sizes = np.bincount(lab.ravel())
    lab[np.isin(lab, np.where(sizes < min_px)[0])] = 0
    return lab


def instances_to_polygons(lab: np.ndarray, transform, simplify_m=2.0, min_area_m2=300.0, px_m=None):
    """Label image → list of shapely polygons in the raster CRS (pixel staircase removed, simplified)."""
    groups: dict[int, list] = {}
    for geom, v in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=transform):
        groups.setdefault(int(v), []).append(shape(geom).buffer(0))
    px = abs(transform.a)
    scale = (px_m / px) if px_m else 1.0  # convert metre settings into CRS units (e.g. web mercator)
    out = []
    for gs in groups.values():
        polys = polygons_of(unary_union(gs))
        if not polys:
            continue
        g = max(polys, key=lambda p: p.area)
        g = g.buffer(px, join_style="mitre").buffer(-px, join_style="mitre")
        g = g.simplify(simplify_m / scale).buffer(0)
        polys = polygons_of(g)
        if not polys:
            continue
        g = max(polys, key=lambda p: p.area)
        if g.area * scale * scale >= min_area_m2:
            out.append(g)
    return out
