"""Where the trained model lives and how the app gets it.

resolve_model(): local file → Hugging Face Hub → direct URL (e.g. GitHub Release), first that works.
upload_model(): push a model + model card to your Hugging Face repo (run from Colab with HF_TOKEN).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .config import SegmentationConfig

CACHE = Path.home() / ".cache" / "drone_flight_plan"


class ModelNotAvailable(RuntimeError):
    pass


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_url(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if expected_sha256 and sha256(tmp) != expected_sha256.lower():
        tmp.unlink(missing_ok=True)
        raise ModelNotAvailable("Downloaded model is corrupted (checksum mismatch). Try again.")
    tmp.replace(dest)
    return dest


def resolve_model(sc: SegmentationConfig, cache_dir: Path | None = None) -> Path:
    """Return a local path to the model, downloading it once if needed."""
    errors = []
    local = Path(sc.checkpoint)
    if local.exists():
        return local
    if sc.hf_repo:
        try:
            from huggingface_hub import hf_hub_download

            return Path(
                hf_hub_download(
                    sc.hf_repo, sc.hf_filename, revision=sc.hf_revision, cache_dir=str(cache_dir) if cache_dir else None
                )
            )
        except Exception as e:  # repo missing, offline, not public yet...
            errors.append(f"Hugging Face ({sc.hf_repo}): {type(e).__name__}: {e}")
    if sc.model_url:
        dest = (cache_dir or CACHE) / Path(sc.model_url.split("?")[0]).name
        if dest.exists() and (not sc.model_sha256 or sha256(dest) == sc.model_sha256.lower()):
            return dest
        try:
            return _download_url(sc.model_url, dest, sc.model_sha256)
        except Exception as e:
            errors.append(f"URL: {type(e).__name__}: {e}")
    raise ModelNotAvailable(
        "No AI model available. "
        + (" | ".join(errors) if errors else "Set segmentation.hf_repo or model_url, or put a model at " + str(local))
    )


def model_card(repo_id: str, metrics: dict, sc: SegmentationConfig, extra: dict | None = None) -> str:
    """README.md for the Hugging Face model page."""
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items())
    ex = "\n".join(f"- **{k}:** {v}" for k, v in (extra or {}).items())
    return f"""---
license: mit
library_name: onnx
pipeline_tag: image-segmentation
tags: [agriculture, field-boundaries, remote-sensing, semantic-segmentation, drones, onnx]
---

# Field boundary detection — Drone Flight Plan

Finds agricultural fields in aerial / satellite images: every pixel is classified as **background**, **field
interior** or **field boundary**; a watershed step then splits touching fields into one polygon each.
Used by [Drone Flight Plan](https://github.com/yuvendhan/drone-flight-plan) to plan drone mapping and spraying
missions.

## Model
- Architecture: **{sc.arch}** with **{sc.encoder}** encoder, 3 classes, exported to **ONNX**
- Input: RGB, normalised with ImageNet mean/std, any size (multiple of 32 after padding),
  **{sc.target_res_m} m per pixel**
- Output: logits `(1, 3, H, W)` → softmax → [background, field, boundary]
{ex}

## Test results
| Metric | Value |
|---|---|
{rows}

## Training data
AI4Boundaries (European Commission JRC, open data): 1 m aerial orthophotos with field labels from several EU
countries.

## Limitations
- Trained on European fields. Small, irregular fields (e.g. India, separated by bunds) may need fine-tuning on
  local imagery.
- Expects straight-down imagery near {sc.target_res_m} m/pixel. Cloud, haze and very different seasons reduce
  accuracy.
- Always check the outline before planning a flight.

## Use
```python
from huggingface_hub import hf_hub_download
from drone_flight_plan.segmentation.infer import Predictor
model = Predictor(hf_hub_download("{repo_id}", "model.onnx"))
fields = model.segment_geotiff("area.tif")["polygons_wgs84"]
```
"""


def upload_model(
    model_path,
    repo_id: str,
    token: str | None = None,
    metrics: dict | None = None,
    sc: SegmentationConfig | None = None,
    private: bool = False,
    extra: dict | None = None,
) -> str:
    """Create the repo if needed and upload model.onnx + README.md (model card) + metrics.json."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_file(path_or_fileobj=str(model_path), path_in_repo="model.onnx", repo_id=repo_id)
    if metrics is not None:
        api.upload_file(
            path_or_fileobj=json.dumps(metrics, indent=2).encode(), path_in_repo="metrics.json", repo_id=repo_id
        )
    if sc is not None:
        card = model_card(repo_id, metrics or {}, sc, extra)
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id)
    return f"https://huggingface.co/{repo_id}"
