from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .physics_pue import pue_physics
    from .lstm_pue_residual import ModelConfig, PueResidualPredictorLSTM
except ImportError:
    # Fallback for direct script execution.
    from physics_pue import pue_physics
    from lstm_pue_residual import ModelConfig, PueResidualPredictorLSTM


class PueHistoryInput(BaseModel):
    room_temperature: list[float]
    cooling_setpoint: list[float]
    cooling_fan_speed_pct: list[float]
    outside_temperature: list[float]
    humidity: list[float]
    wind_speed: list[float]
    avg_exhaust_temp: list[float]
    servers_power_total: list[float]
    pue_real: list[float]


class PueFutureInput(BaseModel):
    outside_temperature: list[float]
    avg_exhaust_temp: list[float]
    servers_power_total: list[float]
    cooling_setpoint: list[float]
    cooling_fan_speed_pct: list[float]


class PueHybridRecommendRequest(BaseModel):
    history: PueHistoryInput
    future: PueFutureInput
    delta_grid_c: Optional[list[float]] = Field(
        default=None, description="Setpoint delta grid in Celsius (e.g. [-2,-1,0,1,2])"
    )


class PueScenarioResponse(BaseModel):
    delta_c: float
    pue_pred_future_mean: list[float]
    cooling_energy_kwh_pred: float
    savings_pct_vs_baseline: float


class PueHybridRecommendResponse(BaseModel):
    input_hours: int
    horizon_hours: int
    baseline_cooling_energy_kwh: float
    best: PueScenarioResponse


app = FastAPI(
    title="Hybrid PUE API",
    description="Simple FastAPI wrapper for physics+ML PUE recommender",
    version="1.0.0",
)


def _repo_root() -> Path:
    # models/predict_pue/api.py -> repo root
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def _load_runtime_assets() -> dict:
    model_dir = Path(__file__).resolve().parent
    model_path = model_dir / "pue_residual_predictor.pt"
    meta_path = model_dir / "pue_residual_meta.json"
    config_path = _repo_root() / "DC_digital_twin" / "config" / "config_google_300s.yaml"

    if not model_path.exists():
        raise RuntimeError(f"Missing model file: {model_path}")
    if not meta_path.exists():
        raise RuntimeError(f"Missing meta file: {meta_path}")
    if not config_path.exists():
        raise RuntimeError(f"Missing config file: {config_path}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    input_hours = int(meta["input_hours"])
    horizon_hours = int(meta["horizon_hours"])
    feature_cols = list(meta["feature_cols"])
    X_mean = np.array(meta["X_mean"], dtype=np.float32).reshape(1, 1, -1)
    X_std = np.array(meta["X_std"], dtype=np.float32).reshape(1, 1, -1)

    model_cfg = ModelConfig(**meta["model_config"])
    device = torch.device("cpu")
    model = PueResidualPredictorLSTM(model_cfg).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    with open(config_path, "r", encoding="utf-8") as f:
        dc_cfg = yaml.safe_load(f)
    crac = dc_cfg["cooling"]["crac"]

    return {
        "model": model,
        "device": device,
        "input_hours": input_hours,
        "horizon_hours": horizon_hours,
        "feature_cols": feature_cols,
        "X_mean": X_mean,
        "X_std": X_std,
        "cop_curve": crac.get("cop_curve", [0.002, -0.15, 4.0]),
        "capacity": float(crac["capacity"]),
        "fan_law": dc_cfg["cooling"]["fans"].get("law", "cubic"),
        "fan_max_power": float(dc_cfg["cooling"]["fans"]["max_power"]),
    }


def _validate_len(name: str, arr: list[float], expected: int) -> None:
    if len(arr) != expected:
        raise ValueError(f"{name} length must be {expected}, got {len(arr)}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/pue/hybrid/recommend", response_model=PueHybridRecommendResponse)
def recommend_pue_hybrid(payload: PueHybridRecommendRequest) -> PueHybridRecommendResponse:
    try:
        assets = _load_runtime_assets()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model/assets: {e}")

    input_hours = int(assets["input_hours"])
    horizon_hours = int(assets["horizon_hours"])

    history = payload.history
    future = payload.future

    for field_name, arr in [
        ("history.room_temperature", history.room_temperature),
        ("history.cooling_setpoint", history.cooling_setpoint),
        ("history.cooling_fan_speed_pct", history.cooling_fan_speed_pct),
        ("history.outside_temperature", history.outside_temperature),
        ("history.humidity", history.humidity),
        ("history.wind_speed", history.wind_speed),
        ("history.avg_exhaust_temp", history.avg_exhaust_temp),
        ("history.servers_power_total", history.servers_power_total),
        ("history.pue_real", history.pue_real),
    ]:
        _validate_len(field_name, arr, input_hours)

    for field_name, arr in [
        ("future.outside_temperature", future.outside_temperature),
        ("future.avg_exhaust_temp", future.avg_exhaust_temp),
        ("future.servers_power_total", future.servers_power_total),
        ("future.cooling_setpoint", future.cooling_setpoint),
        ("future.cooling_fan_speed_pct", future.cooling_fan_speed_pct),
    ]:
        _validate_len(field_name, arr, horizon_hours)

    room_temperature = np.asarray(history.room_temperature, dtype=np.float32)
    cooling_setpoint_hist = np.asarray(history.cooling_setpoint, dtype=np.float32)
    cooling_fan_speed_hist = np.asarray(history.cooling_fan_speed_pct, dtype=np.float32) / 100.0
    outside_temperature_hist = np.asarray(history.outside_temperature, dtype=np.float32)
    humidity_hist = np.asarray(history.humidity, dtype=np.float32)
    wind_speed_hist = np.asarray(history.wind_speed, dtype=np.float32)
    avg_exhaust_temp_hist = np.asarray(history.avg_exhaust_temp, dtype=np.float32)
    servers_power_total_hist = np.asarray(history.servers_power_total, dtype=np.float32)
    pue_real_hist = np.asarray(history.pue_real, dtype=np.float32)

    outside_temperature_future = np.asarray(future.outside_temperature, dtype=np.float32)
    avg_exhaust_temp_future = np.asarray(future.avg_exhaust_temp, dtype=np.float32)
    servers_power_total_future = np.asarray(future.servers_power_total, dtype=np.float32)
    cooling_setpoint_future_base = np.asarray(future.cooling_setpoint, dtype=np.float32)
    cooling_fan_speed_future = np.asarray(future.cooling_fan_speed_pct, dtype=np.float32) / 100.0

    pue_physics_hist, _ = pue_physics(
        servers_power=servers_power_total_hist,
        return_temperature=avg_exhaust_temp_hist,
        setpoint=cooling_setpoint_hist,
        fan_speed=cooling_fan_speed_hist,
        outside_temperature=outside_temperature_hist,
        cop_curve=assets["cop_curve"],
        capacity=assets["capacity"],
        fan_max_power=assets["fan_max_power"],
        fan_law=assets["fan_law"],
    )
    residual_hist = pue_real_hist - np.asarray(pue_physics_hist, dtype=np.float32)

    features_map = {
        "room_temperature": room_temperature,
        "cooling_setpoint": cooling_setpoint_hist,
        "cooling_fan_speed": cooling_fan_speed_hist,
        "outside_temperature": outside_temperature_hist,
        "humidity": humidity_hist,
        "wind_speed": wind_speed_hist,
        "avg_exhaust_temp": avg_exhaust_temp_hist,
        "servers_power_total": servers_power_total_hist,
        "pue_physics": np.asarray(pue_physics_hist, dtype=np.float32),
        "residual": residual_hist,
    }
    feature_cols = list(assets["feature_cols"])
    missing = [k for k in feature_cols if k not in features_map]
    if missing:
        raise HTTPException(status_code=500, detail=f"Feature columns missing: {missing}")

    X_in = np.stack([features_map[c] for c in feature_cols], axis=1).astype(np.float32)[None, :, :]
    Xn = (X_in - assets["X_mean"]) / assets["X_std"]
    xb = torch.tensor(Xn, dtype=torch.float32, device=assets["device"])
    with torch.no_grad():
        residual_mean_pred, _ = assets["model"](xb)
    residual_mean_pred = residual_mean_pred[0].cpu().numpy().astype(np.float32)

    def _scenario_energy(setpoint_future: np.ndarray) -> tuple[np.ndarray, float]:
        pue_ph_future, _ = pue_physics(
            servers_power=servers_power_total_future,
            return_temperature=avg_exhaust_temp_future,
            setpoint=setpoint_future,
            fan_speed=cooling_fan_speed_future,
            outside_temperature=outside_temperature_future,
            cop_curve=assets["cop_curve"],
            capacity=assets["capacity"],
            fan_max_power=assets["fan_max_power"],
            fan_law=assets["fan_law"],
        )
        pue_pred = np.maximum(np.asarray(pue_ph_future, dtype=np.float32) + residual_mean_pred, 1.0)
        cooling_energy_kwh_pred = float(np.sum(servers_power_total_future * (pue_pred - 1.0)) / 1000.0)
        return pue_pred, cooling_energy_kwh_pred

    _, baseline_energy_kwh = _scenario_energy(cooling_setpoint_future_base)

    delta_grid = payload.delta_grid_c or [-2.0, -1.0, 0.0, 1.0, 2.0]
    setpoint_min_c, setpoint_max_c = 18.0, 27.0

    best: Optional[PueScenarioResponse] = None
    best_savings = -1e18

    for delta_c in delta_grid:
        scenario_setpoint = np.clip(cooling_setpoint_future_base + float(delta_c), setpoint_min_c, setpoint_max_c)
        pue_pred, energy_kwh = _scenario_energy(scenario_setpoint)
        savings = float((baseline_energy_kwh - energy_kwh) / baseline_energy_kwh * 100.0) if baseline_energy_kwh else 0.0
        scenario = PueScenarioResponse(
            delta_c=float(delta_c),
            pue_pred_future_mean=[float(x) for x in pue_pred.tolist()],
            cooling_energy_kwh_pred=energy_kwh,
            savings_pct_vs_baseline=savings,
        )
        if savings > best_savings:
            best_savings = savings
            best = scenario

    if best is None:
        raise HTTPException(status_code=500, detail="No scenarios produced.")

    return PueHybridRecommendResponse(
        input_hours=input_hours,
        horizon_hours=horizon_hours,
        baseline_cooling_energy_kwh=baseline_energy_kwh,
        best=best,
    )

