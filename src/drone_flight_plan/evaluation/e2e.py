"""End-to-end score: plan on the AI's polygon, judge the path against the REAL field."""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import shapely

from ..config import Config
from ..planning.coverage import Route, evaluate_route, plan_coverage
from ..segmentation.postprocess import instances_to_polygons, probs_to_instances


def _complete_largest(polys, tgt_shape, transform):
    H, W = tgt_shape
    box = shapely.box(*rasterio.transform.array_bounds(H, W, transform)).buffer(-3)
    inside = [p for p in polys if box.contains(p)] or polys
    return max(inside, key=lambda p: p.area) if inside else None


def end_to_end(cache, cfg: Config, extra_margins=(0,)) -> pd.DataFrame:
    sc = cfg.segmentation
    rows = []
    for fid, _rgb, tr, _crs, tgt, enum, prob, _ in cache:
        gts = instances_to_polygons((enum * (tgt > 0)).astype(np.int32), tr, simplify_m=0.5, min_area_m2=500)
        lab = probs_to_instances(prob, sc.seed_thr, sc.boundary_thr, sc.min_field_px)
        preds = instances_to_polygons(lab, tr, sc.simplify_m, sc.min_field_area_m2)
        gt = _complete_largest(gts, tgt.shape, tr)
        if gt is None or not preds:
            continue
        pr = max(preds, key=lambda p: p.intersection(gt).area / p.union(gt).area)
        piou = pr.intersection(gt).area / pr.union(gt).area
        for extra in extra_margins:
            c = cfg.model_copy(deep=True)
            c.mission.edge_margin_m = cfg.mission.edge_margin_m + extra
            c.mission.coarse_step_deg = max(cfg.mission.coarse_step_deg, 10)
            try:
                r_ai = plan_coverage(pr, c)
                r_gt = plan_coverage(gt, cfg)
            except ValueError:
                continue
            judged = evaluate_route(Route(r_ai.pts, r_ai.work, r_ai.angle, gt, r_ai.flyable, gt), cfg, against=gt)
            rows.append(
                {
                    "tile": fid,
                    "extra_margin_m": extra,
                    "field_ha": round(gt.area / 1e4, 3),
                    "polygon_IoU": round(piou, 3),
                    "hausdorff_m": round(gt.hausdorff_distance(pr), 2),
                    "coverage_gt_plan": r_gt.metrics["coverage_pct"],
                    "coverage_ai_plan": judged["coverage_pct"],
                    "outside_ai_plan_m": judged["path_outside_m"],
                    "fully_inside": judged["path_outside_m"] < 1,
                }
            )
    return pd.DataFrame(rows)
