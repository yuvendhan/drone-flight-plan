"""Plots and animations (matplotlib)."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from .planning.geometry import polygons_of

COLORS = ["#e41a1c", "#377eb8", "#ff7f00", "#984ea3", "#00a6a6", "#a65628"]


def plot_mission(mission, ax=None, image=None, extent=None, title=None):
    import matplotlib.pyplot as plt

    ax = ax or plt.subplots(figsize=(8, 8))[1]
    if image is not None:
        ax.imshow(image, extent=extent)
    r = mission.route if hasattr(mission, "route") else mission
    for p in polygons_of(r.field):
        ax.plot(*p.exterior.xy, color="lime", lw=2.5, label="field")
        for h in p.interiors:
            ax.fill(*h.xy, color="#5dade2", alpha=0.8, label="no-fly")
    for p in polygons_of(r.flyable):
        ax.plot(*p.exterior.xy, color="gold", lw=1, ls="--", label="safe area")
    if hasattr(mission, "sorties"):
        for i, s in enumerate(mission.sorties):
            P = np.array([(w.x, w.y) for w in s.waypoints])
            c = COLORS[i % len(COLORS)]
            work = [w.kind == "work" for w in s.waypoints]
            ax.plot(P[:, 0], P[:, 1], color=c, lw=1.2, alpha=0.35, ls=":")
            Pw = P[np.array(work)]
            if len(Pw) > 1:
                ax.plot(Pw[:, 0], Pw[:, 1], color=c, lw=1.4, label=f"flight {i + 1} ({s.flight_time_s / 60:.1f} min)")
        ax.plot(*mission.home_xy, marker="^", color="black", ms=12, ls="", label="home (take-off/land)")
    else:
        P = r.pts
        ax.plot(P[:, 0], P[:, 1], color="red", lw=1.2, label="path")
    h, lab = ax.get_legend_handles_labels()
    u = dict(zip(lab, h))
    ax.legend(u.values(), u.keys(), loc="upper right", fontsize=8, framealpha=0.85)
    ax.set_aspect("equal")
    m = getattr(mission, "metrics", r.metrics)
    ax.set_title(
        title
        or f"{m.get('mode', '')} | angle {m.get('angle_deg')}° | cover {m.get('coverage_pct')}% | "
        f"{m.get('flights', 1)} flight(s), {m.get('total_flight_min', m.get('path_time_min'))} min",
        fontsize=9,
    )
    return ax


def animate_route(route, out_path, frames=120):
    import matplotlib.pyplot as plt
    from matplotlib import animation

    line = LineString(route.pts)
    fig, ax = plt.subplots(figsize=(6, 6))
    for p in polygons_of(route.field):
        ax.fill(*p.exterior.xy, color="#cfe8b5")
        ax.plot(*p.exterior.xy, color="green", lw=2)
        for h in p.interiors:
            ax.fill(*h.xy, color="#5dade2")
    ax.set_aspect("equal")
    ax.axis("off")
    (trail,) = ax.plot([], [], color="red", lw=1.5)
    (dot,) = ax.plot([], [], "o", color="black", ms=8)
    ds = np.linspace(0, line.length, frames)

    def upd(i):
        pts = [line.interpolate(d) for d in ds[: i + 1]]
        trail.set_data([p.x for p in pts], [p.y for p in pts])
        dot.set_data([pts[-1].x], [pts[-1].y])
        return trail, dot

    animation.FuncAnimation(fig, upd, frames=frames, interval=60).save(out_path, writer=animation.PillowWriter(fps=15))
    plt.close(fig)
    return out_path
