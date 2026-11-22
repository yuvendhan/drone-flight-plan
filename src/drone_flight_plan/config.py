"""Settings for everything. Defaults live here; YAML profiles (configs/*.yaml) override them.

cfg = load_config("india_spraying")                 # packaged profile
cfg = load_config("my.yaml", {"mission": {"speed_mps": 4}})
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

PACKAGED = Path(__file__).parent / "configs"


class DroneProfile(BaseModel):
    name: str = "generic-quad"
    span_m: float = Field(1.2, gt=0, description="Arm-to-arm width incl. propellers (safety clearance)")
    max_speed_mps: float = Field(12.0, gt=0)
    battery_min: float = Field(18.0, gt=0, description="Usable flight time per battery")
    battery_reserve_pct: float = Field(20.0, ge=0, lt=90, description="Always land with this much left")
    climb_rate_mps: float = Field(2.5, gt=0)
    turn_time_s: float = Field(4.0, ge=0, description="Time lost per sharp turn")


class CameraProfile(BaseModel):
    hfov_deg: float = Field(73.7, gt=0, lt=180, description="Horizontal field of view")


class SprayProfile(BaseModel):
    swath_m: float = Field(5.0, gt=0, description="Spray width on the ground")
    tank_l: float = Field(10.0, gt=0)
    rate_l_per_ha: float = Field(20.0, gt=0, description="Dose")
    servo_channel: int = Field(9, description="Pump servo/relay channel for DO_SET_SERVO")
    pwm_on: int = 1900
    pwm_off: int = 1100


class MissionConfig(BaseModel):
    mode: Literal["mapping", "spraying"] = "mapping"
    work_agl_m: float = Field(40.0, gt=0, description="Height above ground while working")
    transit_agl_m: float = Field(50.0, gt=0, description="Height above ground while travelling (above trees)")
    speed_mps: float = Field(6.0, gt=0)
    side_overlap: float = Field(0.3, ge=0, lt=0.9, description="Mapping: photo side overlap. Spraying: swath overlap")
    edge_margin_m: float = Field(3.0, ge=0, description="Gap from field edge to the flight path area")
    obstacle_clearance_m: float = Field(2.0, ge=0, description="Extra gap around no-fly zones (+ half drone span)")
    headland: bool = True
    coarse_step_deg: int = Field(5, ge=1, le=45)
    fine_step_deg: int = Field(1, ge=1, le=10)
    terrain_follow: bool = True
    terrain_wp_spacing_m: float = Field(
        15.0, gt=1, description="Waypoint every N m along stripes when following terrain"
    )
    max_agl_m: float = Field(120.0, description="Legal ceiling (India green zone / EU open category ≈ 120 m)")
    powerline_clearance_m: float = Field(
        10.0, ge=0, description="Keep this far from power lines (plus obstacle clearance)"
    )
    tree_radius_m: float = Field(4.0, gt=0, description="Default canopy radius for a clicked tree")
    home_min_m: float = Field(5.0, ge=0, description="Warn if home is closer than this to the field edge")
    home_max_m: float = Field(150.0, gt=0, description="Warn if home is farther than this (radio link / line of sight)")

    @model_validator(mode="after")
    def _transit_above_work(self):
        if self.transit_agl_m < self.work_agl_m:
            self.transit_agl_m = self.work_agl_m
        return self


class SegmentationConfig(BaseModel):
    arch: Literal["unet", "deeplabv3plus"] = "unet"
    encoder: str = "resnet34"
    encoder_weights: Optional[str] = "imagenet"
    tile: int = 512
    overlap: int = 64
    target_res_m: float = Field(1.0, gt=0, description="Model input resolution (m/pixel)")
    seed_thr: float = 0.5
    boundary_thr: float = 0.4
    min_field_px: int = 150
    simplify_m: float = 2.0
    min_field_area_m2: float = 300.0
    checkpoint: str = "models/model.onnx"
    hf_repo: Optional[str] = Field(
        "yuvendhan/drone-flight-plan-fields", description="Hugging Face model repo (change to your own username/repo)"
    )
    hf_filename: str = "model.onnx"
    hf_revision: str = "main"
    model_url: Optional[str] = Field(None, description="Fallback direct download URL (e.g. GitHub Release asset)")
    model_sha256: Optional[str] = Field(None, description="Expected SHA-256 of the downloaded model file")


class TrainConfig(BaseModel):
    data_dir: str = "data/ai4boundaries"
    countries: list[str] = ["AT", "ES", "FR", "LU", "NL", "SE", "SI"]
    n_train: int = 1500
    n_val: int = 250
    n_test: int = 250
    epochs: int = 25
    batch_size: int = 8
    lr: float = 3e-4
    num_workers: int = 2
    seed: int = 42
    out_dir: str = "models"
    resume: bool = True
    init_checkpoint: Optional[str] = Field(None, description="Start from these weights (fine-tuning)")


class ImageryConfig(BaseModel):
    tile_url: str = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    zoom: int = Field(18, ge=10, le=20)
    max_tiles: int = 400


class DisplayConfig(BaseModel):
    area_unit: Literal["acre", "ha", "cent", "guntha"] = "acre"


class Config(BaseModel):
    display: DisplayConfig = DisplayConfig()
    drone: DroneProfile = DroneProfile()
    camera: CameraProfile = CameraProfile()
    spray: SprayProfile = SprayProfile()
    mission: MissionConfig = MissionConfig()
    segmentation: SegmentationConfig = SegmentationConfig()
    train: TrainConfig = TrainConfig()
    imagery: ImageryConfig = ImageryConfig()

    # ---------- derived values ----------
    @property
    def footprint_m(self) -> float:
        """Ground width covered by one pass (camera width when mapping, spray width when spraying)."""
        if self.mission.mode == "spraying":
            return self.spray.swath_m
        return 2 * self.mission.work_agl_m * math.tan(math.radians(self.camera.hfov_deg) / 2)

    @property
    def spacing_m(self) -> float:
        """Distance between neighbouring stripes."""
        return self.footprint_m * (1 - self.mission.side_overlap)

    @property
    def usable_flight_s(self) -> float:
        return self.drone.battery_min * 60 * (1 - self.drone.battery_reserve_pct / 100)

    @property
    def flow_l_per_s(self) -> float:
        """Pump flow so that dose (L/ha) is met at this speed and swath."""
        return self.spray.rate_l_per_ha * self.spacing_m * self.mission.speed_mps / 10_000

    @property
    def no_fly_buffer_m(self) -> float:
        return self.mission.obstacle_clearance_m + self.drone.span_m / 2


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def available_profiles() -> list[str]:
    return sorted(p.stem for p in PACKAGED.glob("*.yaml") if p.stem != "drones")


def drone_catalog() -> dict[str, dict]:
    """Drone presets from configs/drones.yaml: {name: {"jobs": [...], "note": str, "settings": {...}}}."""
    return yaml.safe_load((PACKAGED / "drones.yaml").read_text())["drones"]


def load_config(profile: str | Path | None = None, overrides: dict | None = None, drone: str | None = None) -> Config:
    """Load defaults ← packaged/explicit YAML ← drone preset ← overrides dict."""
    data: dict = {}
    if profile:
        p = Path(profile)
        if not p.exists():
            p = PACKAGED / f"{profile}.yaml"
        if not p.exists():
            raise FileNotFoundError(f"Config '{profile}' not found. Available: {available_profiles()}")
        data = yaml.safe_load(p.read_text()) or {}
    if drone:
        cat = drone_catalog()
        if drone not in cat:
            raise KeyError(f"Unknown drone '{drone}'. Available: {list(cat)}")
        data = _deep_merge(data, cat[drone].get("settings", {}))
    data = _deep_merge(data, overrides or {})
    return Config.model_validate(data)


def save_config(cfg: Config, path: str | Path) -> None:
    Path(path).write_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False))
