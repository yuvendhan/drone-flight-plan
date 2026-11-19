"""Low-level geometry used by the planner."""

from __future__ import annotations

import heapq
import math

import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import substring


def lines_of(g) -> list[LineString]:
    if g is None or g.is_empty:
        return []
    if g.geom_type == "LineString":
        return [g]
    if hasattr(g, "geoms"):
        return [x for p in g.geoms for x in lines_of(p)]
    return []


def polygons_of(g) -> list:
    if g is None or g.is_empty:
        return []
    if g.geom_type == "Polygon":
        return [g]
    if hasattr(g, "geoms"):
        return [x for p in g.geoms for x in polygons_of(p)]
    return []


def largest_polygon(g):
    ps = polygons_of(g)
    return max(ps, key=lambda p: p.area) if ps else None


def rotate_points(pts, angle_deg: float, origin) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    P = np.asarray(pts, float).reshape(-1, 2) - origin
    return np.c_[P[:, 0] * c - P[:, 1] * s, P[:, 0] * s + P[:, 1] * c] + origin


def _ring_arc(ring, a, b) -> list:
    """Shortest walk from a to b ALONG a ring."""
    L = ring.length
    da, db = ring.project(Point(a)), ring.project(Point(b))

    def fwd(s, e):
        if s <= e:
            return list(substring(ring, s, e).coords)
        return list(substring(ring, s, L).coords) + list(substring(ring, 0, e).coords)

    f, r = fwd(da, db), fwd(db, da)[::-1]
    lf = LineString(f).length if len(f) > 1 else 0.0
    lr = LineString(r).length if len(r) > 1 else 0.0
    return f if lf <= lr else r


def _visibility_path(a, b, poly, safe):
    """Shortest path through polygon corners (Dijkstra on a visibility graph)."""
    nodes = [tuple(a), tuple(b)]
    for r in [poly.exterior] + list(poly.interiors):
        nodes += [tuple(c) for c in list(r.coords)[:-1]]
    n = len(nodes)
    dist, prev, done = [math.inf] * n, [-1] * n, [False] * n
    dist[0] = 0.0
    pq = [(0.0, 0)]
    while pq:
        d, i = heapq.heappop(pq)
        if done[i]:
            continue
        done[i] = True
        if i == 1:
            break
        for j in range(n):
            if done[j]:
                continue
            nd = d + math.dist(nodes[i], nodes[j])
            if nd < dist[j] and safe.covers(LineString([nodes[i], nodes[j]])):
                dist[j], prev[j] = nd, i
                heapq.heappush(pq, (nd, j))
    if prev[1] == -1:
        return None
    out, k = [], 1
    while k != -1:
        out.append(nodes[k])
        k = prev[k]
    return out[::-1]


def connect(a, b, poly) -> list:
    """Path a→b inside poly: straight, else along an edge, else around corners."""
    if math.dist(a, b) < 1e-6:
        return [tuple(a)]
    safe = poly.buffer(0.05)
    if safe.covers(LineString([a, b])):
        return [tuple(a), tuple(b)]
    best = None
    for ring in [poly.exterior] + list(poly.interiors):
        path = [tuple(a)] + [tuple(p) for p in _ring_arc(ring, a, b)] + [tuple(b)]
        ls = LineString(path)
        if safe.covers(ls) and (best is None or ls.length < best[0]):
            best = (ls.length, path)
    if best:
        return best[1]
    vp = _visibility_path(a, b, poly, safe)
    return vp if vp else [tuple(a), tuple(b)]


def count_turns(pts, thr_deg: float = 30) -> int:
    if len(pts) < 3:
        return 0
    P = np.asarray(LineString(pts).simplify(0.3).coords)
    if len(P) < 3:
        return 0
    v = np.diff(P, axis=0)
    h = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
    d = np.abs((np.diff(h) + 180) % 360 - 180)
    return int((d > thr_deg).sum())


def outside_length(pts, poly) -> float:
    """Metres of path outside poly, per segment (robust to self-overlap)."""
    P = np.asarray(pts, float)
    if len(P) < 2:
        return 0.0
    segs = shapely.linestrings(np.stack([P[:-1], P[1:]], axis=1))
    return float(np.nansum(shapely.length(shapely.difference(segs, poly.buffer(0.1)))))


def edge_angles(poly, min_len: float = 15) -> list[int]:
    out = set()
    for r in [poly.exterior] + list(poly.interiors):
        c = list(r.coords)
        for a, b in zip(c[:-1], c[1:]):
            if math.dist(a, b) >= min_len:
                out.add(round(math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180) % 180)
    return sorted(out)


def densify(pts, step: float) -> np.ndarray:
    """Insert points so no segment is longer than `step`."""
    P = np.asarray(pts, float)
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(math.ceil(math.dist(a, b) / step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * k / n)
    return np.asarray(out)
