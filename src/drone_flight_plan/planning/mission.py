"""Turn a coverage Route into flyable missions:

home (outside the field) → take off → climb to travel height → fly to the work start →
descend to work height → stripes (terrain-following, pump on/off) → climb → fly home → land.

Long jobs are split into several flights ("sorties"): a flight ends when the battery (with reserve)
or — when spraying — the tank would run out. The drone returns home, you swap battery / refill,
and the next flight continues exactly where the last one stopped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from shapely.geometry import LineString, Point

from ..config import Config
from .coverage import Route
from .geometry import densify


@dataclass
class Waypoint:
    x: float
    y: float
    alt_rel_m: float  # height above HOME ground (what PX4/ArduPilot "relative" frame uses)
    agl_m: float  # height above the ground under the drone
    kind: str  # takeoff | transit | work | land
    spray: bool = False


@dataclass
class Sortie:
    waypoints: list[Waypoint]
    flight_time_s: float
    work_m: float
    spray_l: float
    distance_m: float


@dataclass
class Mission:
    sorties: list[Sortie]
    home_xy: tuple
    crs: object
    route: Route
    metrics: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------- splitting into flights


def _split_route(route: Route, cfg: Config, home):
    """Cut the route into chunks that fit battery and tank. Returns list of (points, work_flags)."""
    v = cfg.mission.speed_mps
    T = cfg.usable_flight_s
    spraying = cfg.mission.mode == "spraying"
    tank = cfg.spray.tank_l if spraying else math.inf
    l_per_m = cfg.flow_l_per_s / v if spraying else 0.0
    overhead = 2 * cfg.mission.transit_agl_m / cfg.drone.climb_rate_mps + 10  # climb + descend + settle

    def ret(p):
        return math.dist(p, home) / v

    P, W, J = route.pts, route.work, route.jump
    chunks, cur_pts, cur_w, cur_j = [], [tuple(P[0])], [False], [False]
    t = overhead + ret(P[0])
    vol = 0.0
    prev_heading = None
    i = 1
    while i < len(P):
        a, b, w, jmp = np.asarray(cur_pts[-1]), np.asarray(P[i]), bool(W[i]), bool(J[i])
        seg = math.dist(a, b)
        if seg < 1e-6:
            i += 1
            continue
        heading = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
        sharp = prev_heading is not None and abs((heading - prev_heading + 180) % 360 - 180) > 30
        dt = seg / v + (cfg.drone.turn_time_s if sharp else 0.0)
        dvol = seg * l_per_m if w else 0.0
        if t + dt + ret(b) <= T and vol + dvol <= tank + 1e-9:
            cur_pts.append(tuple(b))
            cur_w.append(w)
            cur_j.append(jmp)
            t += dt
            vol += dvol
            i += 1
            prev_heading = heading
            continue
        # how far along this segment can we still go?
        lo, hi = 0.0, 1.0
        for _ in range(30):
            mid = (lo + hi) / 2
            pm = a + (b - a) * mid
            ok = t + seg * mid / v + ret(pm) <= T and vol + (seg * mid * l_per_m if w else 0) <= tank + 1e-9
            lo, hi = (mid, hi) if ok else (lo, mid)
        if lo > 0.02:
            cut = tuple(a + (b - a) * lo)
            cur_pts.append(cut)
            cur_w.append(w)
            cur_j.append(jmp)
        elif len(cur_pts) <= 1:
            raise ValueError(
                "One flight cannot even reach the field and come back. "
                "Move the home point closer or check battery settings."
            )
        chunks.append((cur_pts, cur_w, cur_j))
        start = cur_pts[-1]
        cur_pts, cur_w, cur_j = [start], [False], [False]
        t, vol, prev_heading = overhead + ret(start), 0.0, None
    if len(cur_pts) > 1:
        chunks.append((cur_pts, cur_w, cur_j))
    return chunks


# ---------------------------------------------------------------- waypoints + altitudes


def _waypoints_for(chunk, cfg: Config, home, terrain, crs, home_z):
    pts, work, jumps = chunk
    m = cfg.mission
    follow = terrain is not None and not terrain.is_flat and m.terrain_follow

    def z(xy):
        return terrain.elevation(np.atleast_2d(xy), crs) if follow else np.zeros(len(np.atleast_2d(xy)))

    def transit_alt(a, b):
        d = densify([a, b], 20.0)
        return float(m.transit_agl_m + z(d).max() - home_z)

    wps = [Waypoint(*home, m.transit_agl_m, m.transit_agl_m, "takeoff")]
    start = pts[0]
    alt = transit_alt(home, start)
    wps.append(Waypoint(*start, alt, alt + home_z - float(z(start)[0]), "transit"))
    # descend and work
    zs = float(z(start)[0])
    wps.append(Waypoint(*start, m.work_agl_m + zs - home_z, m.work_agl_m, "work", False))
    for i in range(1, len(pts)):
        a, b, w = pts[i - 1], pts[i], work[i]
        if jumps[i]:  # hop to another piece: climb, cross at travel height, descend
            alt = transit_alt(a, b)
            za, zb = float(z(a)[0]), float(z(b)[0])
            wps.append(Waypoint(float(a[0]), float(a[1]), alt, alt + home_z - za, "transit"))
            wps.append(Waypoint(float(b[0]), float(b[1]), alt, alt + home_z - zb, "transit"))
            wps.append(Waypoint(float(b[0]), float(b[1]), m.work_agl_m + zb - home_z, m.work_agl_m, "work", False))
            continue
        seg = densify([a, b], m.terrain_wp_spacing_m)[1:] if follow else np.asarray([b])
        zz = z(seg)
        for p, zp in zip(seg, zz):
            wps.append(
                Waypoint(
                    float(p[0]),
                    float(p[1]),
                    m.work_agl_m + float(zp) - home_z,
                    m.work_agl_m,
                    "work",
                    bool(w) and cfg.mission.mode == "spraying",
                )
            )
    end = pts[-1]
    alt = transit_alt(end, home)
    wps.append(Waypoint(*end, alt, alt + home_z - float(z(end)[0]), "transit"))
    wps.append(Waypoint(*home, alt, alt, "transit"))
    wps.append(Waypoint(*home, 0.0, 0.0, "land"))
    return _simplify_work(wps)


def _rdp_keep(P: np.ndarray, eps: float) -> np.ndarray:
    """Douglas–Peucker: indices to keep so that EVERY dropped point is within eps of the kept polyline."""
    keep = np.zeros(len(P), bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(P) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = P[i], P[j]
        ab = b - a
        L2 = float(ab @ ab)
        seg = P[i + 1 : j] - a
        t = np.clip(seg @ ab / L2, 0, 1) if L2 > 0 else np.zeros(len(seg))
        d = np.linalg.norm(seg - np.outer(t, ab), axis=1)
        k = int(np.argmax(d))
        if d[k] > eps:
            keep[i + 1 + k] = True
            stack += [(i, i + 1 + k), (i + 1 + k, j)]
    return keep


def _simplify_work(wps, tol=0.25, alt_tol=0.3):
    """Drop redundant work waypoints (straight, same height, same pump state) without cutting corners:
    every removed point stays within `tol` m sideways and `alt_tol` m in height of the simplified path."""
    out, i = [], 0
    while i < len(wps):
        j = i
        if wps[i].kind == "work":
            while j + 1 < len(wps) and wps[j + 1].kind == "work" and wps[j + 1].spray == wps[i].spray:
                j += 1
        run = wps[i : j + 1]
        if len(run) > 2:
            P = np.array([(w.x, w.y, w.alt_rel_m * tol / alt_tol) for w in run])
            run = [w for w, k in zip(run, _rdp_keep(P, tol)) if k]
        out += run
        i = j + 1
    return out


def _flight_time(wps, cfg: Config):
    t, v = 0.0, cfg.mission.speed_mps
    for a, b in zip(wps[:-1], wps[1:]):
        horiz = math.dist((a.x, a.y), (b.x, b.y))
        vert = abs(b.alt_rel_m - a.alt_rel_m)
        t += max(horiz / v, vert / cfg.drone.climb_rate_mps)
    t += cfg.mission.transit_agl_m / cfg.drone.climb_rate_mps  # initial climb
    pts = [(w.x, w.y) for w in wps if w.kind == "work"]
    from .geometry import count_turns

    return t + count_turns(pts) * cfg.drone.turn_time_s


# ---------------------------------------------------------------- public API


def check_home(field_m, home_xy, cfg: Config) -> dict:
    """Is the take-off spot sensible? Outside the field, not too close, not too far."""
    p = Point(home_xy)
    inside = field_m.buffer(-0.5).contains(p)
    dist = (
        0.0
        if inside
        else round(float(field_m.exterior.distance(p) if field_m.geom_type == "Polygon" else field_m.distance(p)), 1)
    )
    m = cfg.mission
    w = []
    if inside:
        w.append("Home point is INSIDE the field. Take-off/landing should be outside (road, bund, open ground).")
    elif dist < m.home_min_m:
        w.append(
            f"Home is only {dist:.0f} m from the field edge — keep at least {m.home_min_m:.0f} m so the "
            "drone does not take off over the crop."
        )
    elif dist > m.home_max_m:
        w.append(
            f"Home is {dist:.0f} m from the field — farther than {m.home_max_m:.0f} m. Keep the drone in "
            "sight and radio range; move closer if possible."
        )
    return {"inside": inside, "distance_m": dist, "ok": not w, "warnings": w}


def build_mission(route: Route, cfg: Config, home_xy=None, terrain=None) -> Mission:
    """home_xy: take-off/landing point in the route's metric CRS. Default = nearest point just outside the field."""
    crs = route.crs
    warnings = []
    if home_xy is None:
        b = route.field.exterior if route.field.geom_type == "Polygon" else route.field.convex_hull.exterior
        p0 = Point(route.pts[0])
        near = b.interpolate(b.project(p0))
        vec = np.array([near.x - route.field.centroid.x, near.y - route.field.centroid.y])
        vec = vec / (np.linalg.norm(vec) + 1e-9)
        home_xy = (near.x + vec[0] * 5, near.y + vec[1] * 5)
        warnings.append("Home point not set: using a spot 5 m outside the field edge. Set a real, safe take-off spot.")
    home_xy = (float(home_xy[0]), float(home_xy[1]))
    home_check = check_home(route.field, home_xy, cfg)
    warnings += home_check["warnings"]
    home_z = float(terrain.elevation([home_xy], crs)[0]) if terrain is not None and not terrain.is_flat else 0.0

    chunks = _split_route(route, cfg, home_xy)
    sorties = []
    l_per_m = cfg.flow_l_per_s / cfg.mission.speed_mps if cfg.mission.mode == "spraying" else 0.0
    for ch in chunks:
        wps = _waypoints_for(ch, cfg, home_xy, terrain, crs, home_z)
        pts, work, _ = ch
        work_m = sum(math.dist(pts[i - 1], pts[i]) for i in range(1, len(pts)) if work[i])
        dist = sum(math.dist((a.x, a.y), (b.x, b.y)) for a, b in zip(wps[:-1], wps[1:]))
        sorties.append(Sortie(wps, _flight_time(wps, cfg), work_m, work_m * l_per_m, dist))

    m = cfg.mission
    max_agl = max(w.agl_m for s in sorties for w in s.waypoints)
    if max_agl > m.max_agl_m:
        warnings.append(f"Highest point {max_agl:.0f} m above ground exceeds the legal ceiling {m.max_agl_m:.0f} m.")
    over = [i + 1 for i, s in enumerate(sorties) if s.flight_time_s > cfg.drone.battery_min * 60]
    if over:
        warnings.append(f"Flights {over} may exceed battery time — check speed/battery settings.")
    if m.mode == "spraying" and m.work_agl_m > 5:
        warnings.append("Spray height > 5 m: high drift risk.")
    if terrain is None or terrain.is_flat:
        warnings.append("No terrain data: heights assume flat ground. Add a DEM for slopes/terraces.")
    transit_out = sum(
        LineString([(s.waypoints[0].x, s.waypoints[0].y), (s.waypoints[1].x, s.waypoints[1].y)])
        .difference(route.field)
        .length
        for s in sorties
    )
    if transit_out > 0:
        warnings.append(
            f"Travel legs cross ~{transit_out:.0f} m outside the field at travel height "
            f"{m.transit_agl_m:.0f} m — make sure that airspace/land is OK to overfly."
        )

    total_s = sum(s.flight_time_s for s in sorties)
    metrics = dict(route.metrics)
    metrics.update(
        {
            "mode": m.mode,
            "flights": len(sorties),
            "total_flight_min": round(total_s / 60, 1),
            "flight_min_each": [round(s.flight_time_s / 60, 1) for s in sorties],
            "mission_distance_m": round(sum(s.distance_m for s in sorties), 1),
            "waypoints_each": [len(s.waypoints) for s in sorties],
            "max_agl_m": round(max_agl, 1),
            "home_distance_m": home_check["distance_m"],
            "terrain": "flat" if terrain is None or terrain.is_flat else terrain.name,
        }
    )
    if m.mode == "spraying":
        litres = sum(s.spray_l for s in sorties)
        metrics.update(
            {
                "spray_litres": round(litres, 2),
                "refills": max(0, len(sorties) - 1),
                "pump_flow_l_min": round(cfg.flow_l_per_s * 60, 3),
                "dose_l_per_ha": cfg.spray.rate_l_per_ha,
            }
        )
    return Mission(sorties, home_xy, crs, route, metrics, warnings)
