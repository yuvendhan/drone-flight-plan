"""Command line:  drone-flight-plan --help

drone-flight-plan profiles
drone-flight-plan plan field.kml --profile india_spraying --home 12.9716,77.5946 --dem copernicus -o out/
drone-flight-plan georef shot.png --tl 12.98,77.58 --br 12.97,77.60 -o shot.tif
drone-flight-plan fetch-imagery --bbox 77.58,12.96,77.60,12.98 -o area.tif
drone-flight-plan segment area.tif --checkpoint models/unet_resnet34_3class.pt -o fields.geojson
drone-flight-plan train --profile default [--synthetic]
drone-flight-plan evaluate --checkpoint models/unet_resnet34_3class.pt
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .config import available_profiles, load_config

app = typer.Typer(add_completion=False, help="Drone Flight Plan: AI field detection + drone mission planning")


def _latlon(s: str):
    lat, lon = (float(v) for v in s.split(","))
    return lat, lon


@app.command()
def profiles():
    """List packaged config profiles."""
    for p in available_profiles():
        typer.echo(p)


@app.command()
def plan(
    field: Path = typer.Argument(..., help="KML or GeoJSON with the field outline"),
    profile: str = typer.Option("default", "--profile", "-p"),
    home: Optional[str] = typer.Option(None, help="lat,lon of take-off/landing spot (outside the field)"),
    dem: Optional[str] = typer.Option(None, help="'copernicus' or path to DEM/DSM GeoTIFF"),
    no_fly: Optional[Path] = typer.Option(None, help="KML/GeoJSON of trees/poles/wells to avoid"),
    out: Path = typer.Option(Path("out"), "--out", "-o"),
    name: str = typer.Option("mission"),
):
    """Plan a mission for a field outline and export all formats."""
    from .export.writers import export_all
    from .geo.io import parse_kml_polygons, read_field_file
    from .pipeline import plan_field

    cfg = load_config(profile)
    f = read_field_file(field)
    nf = None
    if no_fly:
        nf = parse_kml_polygons(no_fly.read_text()) if no_fly.suffix.lower() == ".kml" else [read_field_file(no_fly)]
    m = plan_field(f, cfg, _latlon(home) if home else None, nf, dem)
    files = export_all(m, cfg, out, name)
    typer.echo(json.dumps(m.metrics, indent=2, default=str))
    for w in m.warnings:
        typer.secho(f"⚠ {w}", fg=typer.colors.YELLOW)
    typer.echo(f"Saved to {out}: {[str(p) for v in files.values() for p in (v if isinstance(v, list) else [v])]}")


@app.command()
def georef(
    image: Path,
    tl: str = typer.Option(..., help="lat,lon of top-left corner"),
    br: str = typer.Option(..., help="lat,lon of bottom-right corner"),
    out: Path = typer.Option(..., "--out", "-o"),
):
    """Georeference a north-up screenshot from its corner GPS."""
    from .geo.georef import georeference_by_corners

    r = georeference_by_corners(image, _latlon(tl), _latlon(br), out)
    typer.echo(f"{r.path} | {r.ground_res_m:.2f} m/pixel | method {r.method}")


@app.command("fetch-imagery")
def fetch_imagery_cmd(
    bbox: str = typer.Option(..., help="west,south,east,north (degrees)"),
    out: Path = typer.Option(..., "--out", "-o"),
    zoom: Optional[int] = None,
    profile: str = "default",
):
    """Download basemap imagery for an area as GeoTIFF (personal testing only — check licence)."""
    from .geo.tiles import fetch_imagery

    cfg = load_config(profile)
    b = tuple(float(v) for v in bbox.split(","))
    p = fetch_imagery(b, out, zoom or cfg.imagery.zoom, cfg.imagery.tile_url, max_tiles=cfg.imagery.max_tiles)
    typer.echo(str(p))


@app.command()
def segment(
    image: Path,
    checkpoint: Path = typer.Option(...),
    profile: str = "default",
    out: Path = typer.Option(Path("fields.geojson"), "--out", "-o"),
):
    """Detect fields in a GeoTIFF → GeoJSON of polygons."""
    from .geo.crs import WGS84
    from .geo.io import polygons_to_geojson
    from .segmentation.infer import Predictor

    cfg = load_config(profile)
    res = Predictor(str(checkpoint), cfg.segmentation).segment_geotiff(image)
    gj = polygons_to_geojson(res["polygons_wgs84"], WGS84)
    out.write_text(json.dumps(gj))
    typer.echo(f"{len(res['polygons'])} fields → {out}")


@app.command()
def train(profile: str = "default", synthetic: bool = typer.Option(False, help="Use synthetic tiles (smoke test)")):
    """Download AI4Boundaries (or make synthetic data) and train the segmentation model."""
    from .segmentation.data import download_ai4boundaries, make_synthetic_dataset
    from .segmentation.train import train as run

    cfg = load_config(profile)
    meta = make_synthetic_dataset(cfg.train, cfg.segmentation.tile) if synthetic else download_ai4boundaries(cfg.train)
    typer.echo(meta.split.value_counts().to_string())
    res = run(cfg, meta)
    typer.echo(f"best mIoU {res['best_miou']:.3f} → {res['checkpoint']}")


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(...),
    profile: str = "default",
    tune: bool = True,
    out: Path = typer.Option(Path("reports"), "--out", "-o"),
):
    """Test-set metrics (+ threshold tuning on validation and end-to-end planning score)."""
    import pandas as pd

    from .evaluation.e2e import end_to_end
    from .segmentation.infer import Predictor
    from .segmentation.metrics import evaluate as ev
    from .segmentation.metrics import predict_split, tune_thresholds

    cfg = load_config(profile)
    meta = pd.read_csv(Path(cfg.train.data_dir) / "meta.csv")
    pred = Predictor(str(checkpoint), cfg.segmentation)
    out.mkdir(parents=True, exist_ok=True)
    if tune:
        best, grid = tune_thresholds(predict_split(pred, meta[meta.split == "val"], 100))
        cfg.segmentation.seed_thr, cfg.segmentation.boundary_thr = best["seed_thr"], best["boundary_thr"]
        grid.to_csv(out / "threshold_grid.csv", index=False)
        typer.echo(f"tuned thresholds: {best}")
    cache = predict_split(pred, meta[meta.split == "test"])
    pixel, summ = ev(cache, cfg.segmentation.seed_thr, cfg.segmentation.boundary_thr, cfg.segmentation.min_field_px)
    pixel.to_csv(out / "pixel_metrics.csv")
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    e2e = end_to_end(cache[:40], cfg, extra_margins=(0, 2, 4))
    e2e.to_csv(out / "end_to_end.csv", index=False)
    typer.echo(pixel.to_string())
    typer.echo(json.dumps(summ, indent=2))
    if len(e2e):
        typer.echo(
            e2e.groupby("extra_margin_m")[["coverage_ai_plan", "outside_ai_plan_m", "fully_inside"]]
            .mean()
            .round(2)
            .to_string()
        )


if __name__ == "__main__":
    app()
