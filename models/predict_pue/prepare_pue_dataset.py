#!/usr/bin/env python3
"""
Строит pue_dataset.npz из *_servers.csv и *_summary.csv (результаты DC_digital_twin).

Признаки на каждый час согласованы с models/predict_pue/api.py (feature_cols в meta).
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import requests
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.predict_pue.physics_pue import pue_physics


def _hour_from_step(step_series: pd.Series, time_step_seconds: int) -> pd.Series:
    step0 = step_series.astype(np.int64) - 1
    return ((step0 * int(time_step_seconds)) // 3600).astype(np.int64)


def _openmeteo_hourly(
    base_date: str, total_hours: int, lat: float, lon: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = datetime.strptime(base_date, "%Y-%m-%d")
    days_needed = int(math.ceil(total_hours / 24.0))
    end_date = (base + timedelta(days=days_needed - 1)).strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": base.strftime("%Y-%m-%d"),
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
        "timezone": "Europe/Moscow",
    }
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    temps = np.array(data["hourly"]["temperature_2m"][:total_hours], dtype=np.float32)
    hums = np.array(data["hourly"]["relative_humidity_2m"][:total_hours], dtype=np.float32)
    winds = np.array(data["hourly"]["wind_speed_10m"][:total_hours], dtype=np.float32)
    return temps, hums, winds


def _list_run_files(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("*_servers.csv"))


def _find_summary(servers_path: Path) -> Path:
    return Path(str(servers_path).replace("_servers.csv", "_summary.csv"))


def build_hourly_frame(
    servers_csv: Path,
    summary_csv: Path,
    *,
    time_step_seconds: int,
    base_date: str,
    lat: float,
    lon: float,
) -> pd.DataFrame:
    servers = pd.read_csv(servers_csv)
    summary = pd.read_csv(summary_csv)
    for c in ["step"]:
        servers[c] = servers[c].astype(np.int64)
        summary[c] = summary[c].astype(np.int64)

    servers["hour"] = _hour_from_step(servers["step"], time_step_seconds)
    summary["hour"] = _hour_from_step(summary["step"], time_step_seconds)

    srv_h = (
        servers.groupby("hour", as_index=False)
        .agg(
            servers_power_total=("power", "sum"),
            avg_exhaust_temp=("t_out", "mean"),
        )
    )

    sum_h = (
        summary.groupby("hour", as_index=False)
        .agg(
            room_temperature=("room_temperature", "mean"),
            pue=("pue", "mean"),
            cooling_setpoint=("cooling_setpoint", "mean"),
            cooling_fan_speed=("cooling_fan_speed", "mean"),
        )
    )

    df = sum_h.merge(srv_h, on="hour", how="inner")
    df = df.sort_values("hour").reset_index(drop=True)
    total_hours = int(df["hour"].max()) + 1

    temps, hums, winds = _openmeteo_hourly(base_date, total_hours, lat, lon)
    weather = pd.DataFrame(
        {
            "hour": np.arange(total_hours, dtype=np.int64),
            "outside_temperature": temps,
            "humidity": hums,
            "wind_speed": winds,
        }
    )
    df = df.merge(weather, on="hour", how="left")
    return df


def build_dataset(
    results_dir: Path,
    dc_yaml: Path,
    *,
    input_hours: int,
    horizon_hours: int,
    time_step_seconds: int,
    base_date: str,
) -> Tuple[np.ndarray, np.ndarray]:
    cfg = yaml.safe_load(dc_yaml.read_text(encoding="utf-8"))
    crac = cfg["cooling"]["crac"]
    fans = cfg["cooling"]["fans"]
    cop_curve = crac.get("cop_curve", [0.002, -0.15, 4.0])
    capacity = float(crac["capacity"])
    fan_max_power = float(fans["max_power"])
    fan_law = fans.get("law", "cubic")
    lat = float(cfg.get("weather", {}).get("lat", 55.7558))
    lon = float(cfg.get("weather", {}).get("lon", 37.6173))

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []

    for servers_csv in _list_run_files(results_dir):
        summary_csv = _find_summary(servers_csv)
        if not summary_csv.exists():
            continue
        df = build_hourly_frame(
            servers_csv,
            summary_csv,
            time_step_seconds=time_step_seconds,
            base_date=base_date,
            lat=lat,
            lon=lon,
        )
        if df.empty:
            continue

        total_hours = int(df["hour"].max()) + 1
        if total_hours < input_hours + horizon_hours:
            continue

        # Заполняем по часам без дыр
        full = df.set_index("hour").reindex(range(total_hours))
        if full.isnull().any().any():
            continue

        sp = full["servers_power_total"].to_numpy(dtype=np.float32)
        rt = full["avg_exhaust_temp"].to_numpy(dtype=np.float32)
        setp = full["cooling_setpoint"].to_numpy(dtype=np.float32)
        fan = (full["cooling_fan_speed"].to_numpy(dtype=np.float32) / 100.0).clip(0.0, 1.0)
        outside = full["outside_temperature"].to_numpy(dtype=np.float32)
        room = full["room_temperature"].to_numpy(dtype=np.float32)
        hum = full["humidity"].to_numpy(dtype=np.float32)
        wind = full["wind_speed"].to_numpy(dtype=np.float32)
        pue_real = full["pue"].to_numpy(dtype=np.float32)

        pue_phy, _ = pue_physics(
            servers_power=sp,
            return_temperature=rt,
            setpoint=setp,
            fan_speed=fan,
            outside_temperature=outside,
            cop_curve=cop_curve,
            capacity=capacity,
            fan_max_power=fan_max_power,
            fan_law=fan_law,
        )
        pue_phy = np.asarray(pue_phy, dtype=np.float32)
        residual = pue_real - pue_phy

        # [T, 10] features — порядок как в pue_residual_meta.json
        feat = np.stack(
            [
                room,
                setp,
                fan,
                outside,
                hum,
                wind,
                rt,
                sp,
                pue_phy,
                residual,
            ],
            axis=1,
        ).astype(np.float32)

        max_start = total_hours - (input_hours + horizon_hours) + 1
        for start in range(max_start):
            x_win = feat[start : start + input_hours]
            y_win = residual[start + input_hours : start + input_hours + horizon_hours]
            if y_win.shape[0] != horizon_hours:
                continue
            X_list.append(x_win)
            y_list.append(y_win)

    if not X_list:
        raise RuntimeError("No PUE windows built; check results_dir and run length.")
    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.stack(y_list, axis=0).astype(np.float32)
    return X, y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument(
        "--dc_config",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "DC_digital_twin" / "config" / "config_google_300s.yaml"),
    )
    ap.add_argument("--out", type=str, default="models/predict_pue/pue_dataset.npz")
    ap.add_argument("--input_hours", type=int, default=24)
    ap.add_argument("--horizon_hours", type=int, default=6)
    ap.add_argument("--time_step_seconds", type=int, default=300)
    ap.add_argument("--base_date", type=str, default="2019-05-01")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    results = Path(args.results_dir)
    if not results.is_absolute():
        results = root / results

    X, y = build_dataset(
        results,
        Path(args.dc_config),
        input_hours=args.input_hours,
        horizon_hours=args.horizon_hours,
        time_step_seconds=args.time_step_seconds,
        base_date=args.base_date,
    )
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X, y=y, input_hours=args.input_hours, horizon_hours=args.horizon_hours)
    print(f"Saved {out} X={X.shape} y={y.shape}")


if __name__ == "__main__":
    main()
