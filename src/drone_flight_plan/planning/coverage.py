"""Coverage path planning: lawnmower stripes over any field shape.

Rules for choosing the stripe direction:
  1. never leave the safe area, 2. cover (almost) as much as the best plan,
  3. then the shortest flight time (distance + turns + climbing on slopes).
  Near-ties go to an angle parallel to a field edge or to the hill contour.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely import affinity
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from ..config import Config
from .geometry import (
    connect,
    count_turns,
    edge_angles,
    lines_of,
    outside_length,
    polygons_of,
    rotate_points,
)


@dataclass
class Route:
    """Flight path in metric coordinates. work[i] is True if segment (i-1 → i) is a working pass."""

    pts: np.ndarray
    work: np.ndarray
    angle: float | list
    field: object  # field polygon (metres)
    flyable: object  # where the drone may fly
    workable: object  # field minus no-fly zones (what must be covered)
    crs: object = None
    metrics: dict = field(default_factory=dict)

    def work_lines(self) -> list[LineString]:
        lines, cur = [], []
        for i in range(1, len(self.pts)):
            if self.work[i]:
                if not cur:
                    cur = [self.pts[i - 1]]
                cur.append(self.pts[i])
            elif cur:
                lines.append(LineString(cur))
                cur = []
        if cur:
            lines.append(LineString(cur))
        return [ln for ln in lines if ln.length > 0]

    @property
    def length_m(self) -> float:
        return float(LineString(self.pts).length) if len(self.pts) > 1 else 0.0


# ---------------------------------------------------------------- stripes


def _sweep_rows(poly, angle, spacing, origin):
    r = affinity.rotate(poly, -angle, origin=origin)
    minx, miny, maxx, maxy = r.bounds
    n = max(1, int(math.ceil((maxy - miny) / spacing)))
    y0 = (miny + maxy) / 2 - (n - 1) * spacing / 2
    segs = []
    for i in range(n):
        y = y0 + i * spacing
        for g in lines_of(r.intersection(LineString([(minx - 1, y), (maxx + 1, y)]))):
            if g.length < 0.5:
                continue
            xa, xb = g.coords[0][0], g.coords[-1][0]
            segs.append((min(xa, xb), max(xa, xb), y, i))
    return segs


def decompose_cells(segs):
    """Boustrophedon cell decomposition on stripe pieces.

    Rows are grouped into 'cells': a cell continues while the next row has the same number of pieces and
    each piece overlaps the one before it. Where a tree/pond splits the rows, or a field edge bends
    (L/U shapes), a new cell starts. Each cell can then be mowed cleanly back and forth."""
    rows: dict[int, list] = {}
    for sg in segs:
        rows.setdefault(sg[3], []).append(sg)
    cells, open_cells, prev = [], [], None
    for r in sorted(rows):
        row = sorted(rows[r], key=lambda q: q[0])
        cont = (
            prev is not None
            and r == prev[0] + 1
            and len(row) == len(prev[1])
            and all(a[0] < b[1] and b[0] < a[1] for a, b in zip(row, prev[1]))
        )
        if cont:
            for c, sg in zip(open_cells, row):
                c.append(sg)
        else:
            open_cells = [[sg] for sg in row]
            cells += open_cells
        prev = (r, row)
    return cells


def _cell_paths(cell):
    """The 4 ways to mow a cell: start bottom/top × left/right. Each = list of (p, q) passes."""
    out = []
    for rows in (cell, cell[::-1]):
        for first_left in (True, False):
            passes, left = [], first_left
            for x0, x1, y, _ in rows:
                passes.append(((x0, y), (x1, y)) if left else ((x1, y), (x0, y)))
                left = not left
            out.append(passes)
    return out


def _order_segments(segs, start):
    """Mow cell by cell: go to the nearest unvisited cell, enter at its closest corner, lawnmower through it."""
    cells = decompose_cells(segs)
    order, cur = [], start
    remaining = list(range(len(cells)))
    while remaining:
        best = None
        for ci in remaining:
            for passes in _cell_paths(cells[ci]):
                d = math.dist(cur, passes[0][0])
                if best is None or d < best[0]:
                    best = (d, ci, passes)
        _, ci, passes = best
        remaining.remove(ci)
        order += passes
        cur = passes[-1][1]
    return order


def plan_part(fly_poly, angle: float, spacing: float, headland: bool, start=None):
    """One connected area at one angle → (points, work flags) in map coordinates."""
    origin = np.array([fly_poly.centroid.x, fly_poly.centroid.y])
    rfly = affinity.rotate(fly_poly, -angle, origin=tuple(origin))
    pts, work = [], []

    def add(path, is_work):
        for k, p in enumerate(path):
            if pts and math.dist(p, pts[-1]) < 1e-3:
                continue
            pts.append(tuple(p))
            work.append(bool(is_work) and (k > 0 or len(path) == 1) and len(pts) > 1)

    sweep_poly = fly_poly
    if headland:
        laps = sorted(polygons_of(fly_poly.buffer(-spacing / 2)), key=lambda p: -p.area)
        rings = []
        for lp in laps:
            rlp = affinity.rotate(lp, -angle, origin=tuple(origin))
            rings += [rlp.exterior] + list(rlp.interiors)
        for r in rings:
            ring = [tuple(c) for c in r.coords]
            if pts:
                k = int(np.argmin([math.dist(pts[-1], c) for c in ring[:-1]]))
                ring = ring[k:-1] + ring[: k + 1]
                add(connect(pts[-1], ring[0], rfly), False)
            add(ring[:1], False)
            add(ring, True)
        sweep_poly = fly_poly.buffer(-spacing)

    segs = _sweep_rows(sweep_poly, angle, spacing, tuple(origin)) if not sweep_poly.is_empty else []
    if not segs and not pts:
        return None, None
    if segs:
        if pts:
            st = pts[-1]
        elif start is not None:
            st = tuple(rotate_points([start], -angle, origin)[0])
        else:
            st = (segs[0][0], segs[0][2])
        for p, q in _order_segments(segs, st):
            if pts:
                add(connect(pts[-1], p, rfly), False)
            else:
                add([p], False)
            add([p, q], True)
    P = rotate_points(pts, angle, origin)
    W = np.asarray(work, bool)
    if len(W):
        W[0] = False
    return P, W


# ---------------------------------------------------------------- scoring


def work_footprint(lines, width):
    return unary_union([ln.buffer(width / 2, cap_style="flat") for ln in lines]) if lines else Point().buffer(0)


def _work_lines(P, W):
    out, cur = [], []
    for i in range(1, len(P)):
        if W[i]:
            if not cur:
                cur = [P[i - 1]]
            cur.append(P[i])
        elif cur:
            out.append(LineString(cur))
            cur = []
    if cur:
        out.append(LineString(cur))
    return [ln for ln in out if ln.length > 0]


def climb_m(P, terrain, crs, step=10.0) -> float:
    if terrain is None or terrain.is_flat or len(P) < 2:
        return 0.0
    from .geometry import densify

    D = densify(P, step)
    z = terrain.elevation(D, crs)
    dz = np.diff(z)
    return float(dz[dz > 0].sum())


def score(P, W, part, cfg: Config, terrain=None, crs=None):
    """(time_s, missed_m2, outside_m)."""
    line = LineString(P)
    t = line.length / cfg.mission.speed_mps + count_turns(P) * cfg.drone.turn_time_s
    t += climb_m(P, terrain, crs) / cfg.drone.climb_rate_mps
    missed = max(part.area - work_footprint(_work_lines(P, W), cfg.spacing_m).intersection(part).area, 0)
    return t, missed, outside_length(P, part)


def pick_best(cands, area, preferred=()):
    ok = [c for c in cands if c[0][2] < 1.0] or [min(cands, key=lambda c: c[0][2])]
    best_missed = min(c[0][1] for c in ok)
    ok = [c for c in ok if c[0][1] <= best_missed + 0.003 * area]
    best = min(ok, key=lambda c: c[0][0])
    near = [
        c for c in ok if c[0][0] <= best[0][0] * 1.01 and any(abs((c[1] - e + 90) % 180 - 90) < 0.5 for e in preferred)
    ]
    return min(near, key=lambda c: c[0][0]) if near else best


# ---------------------------------------------------------------- public API


def flyable_area(field_m, cfg: Config, no_fly_m=None):
    fly = field_m.buffer(-cfg.mission.edge_margin_m)
    if no_fly_m is not None and not no_fly_m.is_empty:
        fly = fly.difference(no_fly_m.buffer(cfg.no_fly_buffer_m))
    return fly


def plan_coverage(field_m, cfg: Config, crs=None, no_fly_m=None, terrain=None, angle=None, start=None) -> Route:
    """
    field_m : shapely Polygon in metres (UTM).
    no_fly_m: optional (Multi)Polygon of trees/poles/wells/houses to avoid (metres).
    terrain : optional Terrain for climb cost + contour-aligned stripes.
    angle   : force a stripe direction (degrees), else search.
    """
    if field_m.geom_type not in ("Polygon", "MultiPolygon") or field_m.area <= 0:
        raise ValueError("Field must be a polygon with area")
    sp = cfg.spacing_m
    fly = flyable_area(field_m, cfg, no_fly_m)
    workable = field_m if no_fly_m is None else field_m.difference(no_fly_m)
    parts = [p for p in sorted(polygons_of(fly), key=lambda p: -p.area) if p.area > sp * sp * 0.25]
    if not parts:
        raise ValueError(
            f"Field too small for stripe spacing {sp:.1f} m and edge margin "
            f"{cfg.mission.edge_margin_m} m. Fly lower / reduce margin."
        )
    contour = None
    if terrain is not None and not terrain.is_flat:
        contour = terrain.slope_and_contour(field_m, crs).get("contour_deg")

    def best_for(part, st):
        if angle is not None:
            P, W = plan_part(part, angle, sp, cfg.mission.headland, st)
            return angle, P, W
        preferred = edge_angles(part) + ([round(contour)] if contour is not None else [])
        cands = []

        def trial(a):
            a = a % 180
            P, W = plan_part(part, a, sp, cfg.mission.headland, st)
            if P is not None and len(P) > 1:
                cands.append((score(P, W, part, cfg, terrain, crs), a, P, W))

        for a in sorted(set(range(0, 180, cfg.mission.coarse_step_deg)) | set(preferred)):
            trial(a)
        if not cands:
            return None, None, None
        a0 = pick_best(cands, part.area, preferred)[1]
        tried = {c[1] for c in cands}
        for a in range(
            int(a0) - cfg.mission.coarse_step_deg + 1, int(a0) + cfg.mission.coarse_step_deg, cfg.mission.fine_step_deg
        ):
            if a % 180 not in tried:
                trial(a)
        _, a, P, W = pick_best(cands, part.area, preferred)
        return a, P, W

    allP, allW, angles, cur = [], [], [], start
    remaining = parts[:]
    while remaining:
        if cur is not None:
            remaining.sort(key=lambda p: p.distance(Point(cur)))
        part = remaining.pop(0)
        a, P, W = best_for(part, cur)
        if P is None:
            continue
        if allP:
            W = W.copy()
            W[0] = False  # transit from previous piece
        allP.append(P)
        allW.append(W)
        angles.append(a)
        cur = tuple(P[-1])
    if not allP:
        raise ValueError("Could not plan any stripes for this field")
    route = Route(
        np.vstack(allP), np.concatenate(allW), angles[0] if len(angles) == 1 else angles, field_m, fly, workable, crs
    )
    route.metrics = evaluate_route(route, cfg, terrain)
    if contour is not None:
        route.metrics["contour_deg"] = contour
    return route


def evaluate_route(route: Route, cfg: Config, terrain=None, against=None) -> dict:
    """Coverage is measured with the overlap-guaranteed swath (= stripe spacing) on work passes only.
    against: optionally a different 'true' field polygon (end-to-end evaluation)."""
    target = against if against is not None else route.workable
    flyable = flyable_area(against, cfg) if against is not None else route.flyable
    lines = route.work_lines()
    swath = work_footprint(lines, cfg.spacing_m)
    foot = work_footprint(lines, cfg.footprint_m)
    covered = swath.intersection(target).area
    P = route.pts
    return {
        "field_ha": round(target.area / 1e4, 4),
        "distance_m": round(route.length_m, 1),
        "work_distance_m": round(sum(ln.length for ln in lines), 1),
        "turns": count_turns(P),
        "climb_m": round(climb_m(P, terrain, route.crs), 1),
        "path_time_min": round(
            (route.length_m / cfg.mission.speed_mps + count_turns(P) * cfg.drone.turn_time_s) / 60, 2
        ),
        "coverage_pct": round(100 * covered / target.area, 2),
        "footprint_cover_pct": round(100 * foot.intersection(target).area / target.area, 2),
        "missed_m2": round(target.area - covered, 1),
        "spill_m2": round(foot.difference(target).area, 1),
        "path_outside_m": round(outside_length(P, flyable), 1),
        "angle_deg": route.angle,
        "spacing_m": round(cfg.spacing_m, 2),
    }


def longest_edge_angle(poly) -> float:
    c = list(poly.minimum_rotated_rectangle.exterior.coords)
    _, a, b = max((math.dist(c[i], c[i + 1]), c[i], c[i + 1]) for i in range(4))
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180


def compare_strategies(field_m, cfg: Config, crs=None, no_fly_m=None, terrain=None):
    """Baselines vs optimised stripe direction (for reports)."""
    import pandas as pd

    rows = []
    for name, ang in [
        ("Fixed 0°", 0),
        ("Longest edge (manual habit)", round(longest_edge_angle(field_m))),
        ("Drone Flight Plan optimised", None),
    ]:
        r = plan_coverage(field_m, cfg, crs, no_fly_m, terrain, angle=ang)
        m = r.metrics
        rows.append(
            {
                "strategy": name,
                "angle": m["angle_deg"],
                "distance_m": m["distance_m"],
                "turns": m["turns"],
                "climb_m": m["climb_m"],
                "time_min": m["path_time_min"],
                "coverage_pct": m["coverage_pct"],
                "outside_m": m["path_outside_m"],
            }
        )
    return pd.DataFrame(rows)
