#!/usr/bin/env python3
"""
Инференс для обученной LSTM-модели.

Пример:
  python models/predict_server_temp/predict_temp_predictor.py \
    --servers_csv DC_digital_twin/results/sim_xxx_servers.csv \
    --summary_csv DC_digital_twin/results/sim_xxx_summary.csv \
    --model_path temp_predictor.pt \
    --meta_path temp_predictor_meta.json \
    --out_csv preds.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import requests

from datetime import datetime, timedelta
import math

from lstm_temp_predictor import ModelConfig, TempPredictorLSTM


def _hour_from_step(step_series: pd.Series, time_step_seconds: int) -> pd.Series:
    step0 = step_series.astype(np.int64) - 1
    return ((step0 * int(time_step_seconds)) // 3600).astype(np.int64)


def _sin_cos_time(hour: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * hour / period
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


MOSCOW_LAT = 55.7558
MOSCOW_LON = 37.6173


def _openmeteo_hourly(base_date: str, total_hours: int, lat: float, lon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    outside_temp (°C), humidity (%), длины total_hours.
    """
    base = datetime.strptime(base_date, "%Y-%m-%d")
    days_needed = int(math.ceil(total_hours / 24.0))
    end_date = (base + timedelta(days=days_needed - 1)).strftime("%Y-%m-%d")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": base.strftime("%Y-%m-%d"),
        "end_date": end_date,
        "hourly": ["temperature_2m", "relative_humidity_2m"],
        "timezone": "Europe/Moscow",
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()

    temps = np.array(data["hourly"]["temperature_2m"][:total_hours], dtype=np.float32)
    hums = np.array(data["hourly"]["relative_humidity_2m"][:total_hours], dtype=np.float32)
    return temps, hums


def main():
    parser = argparse.ArgumentParser(description="Predict server temperatures with trained model.")
    parser.add_argument("--servers_csv", type=str, required=True)
    parser.add_argument("--summary_csv", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="temp_predictor.pt")
    parser.add_argument("--meta_path", type=str, default="temp_predictor_meta.json")
    parser.add_argument("--out_csv", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    meta = json.loads(Path(args.meta_path).read_text(encoding="utf-8"))
    input_hours = int(meta["input_hours"])
    horizon_hours = int(meta["horizon_hours"])
    feature_count = int(meta["feature_count"])
    time_step_seconds = int(meta.get("time_step_seconds", 300))
    base_date = str(meta.get("base_date", "2019-05-01"))
    feature_cols_str = meta.get("feature_cols")
    if not feature_cols_str:
        raise RuntimeError("В meta json нет feature_cols. Нужно пересобрать модель новым prepare_temp_dataset.")
    feature_cols = tuple(str(feature_cols_str).split(","))

    X_mean = np.array(meta["X_mean"], dtype=np.float32).reshape(1, 1, feature_count)
    X_std = np.array(meta["X_std"], dtype=np.float32).reshape(1, 1, feature_count)

    servers = pd.read_csv(args.servers_csv)
    summary = pd.read_csv(args.summary_csv)

    servers["step"] = servers["step"].astype(np.int64)
    servers["server_id"] = servers["server_id"].astype(np.int64)
    summary["step"] = summary["step"].astype(np.int64)

    servers["hour"] = _hour_from_step(servers["step"], time_step_seconds)
    summary["hour"] = _hour_from_step(summary["step"], time_step_seconds)

    srv_hour = (
        servers.groupby(["hour", "server_id"], as_index=False)
        .agg(
            utilization=("utilization", "mean"),
            t_chip=("t_chip", "mean"),
            t_in=("t_in", "mean"),
            server_fan_speed=("fan_speed", "mean"),
            power=("power", "mean"),
        )
    )

    sum_hour = (
        summary.groupby(["hour"], as_index=False)
        .agg(
            setpoint=("cooling_setpoint", "mean"),
        )
    )

    merged = srv_hour.merge(sum_hour, on="hour", how="inner")
    if merged.empty:
        raise RuntimeError("Нет данных после merge servers/hour и summary/hour")

    # position: нормализованный server_id
    num_servers = int(merged["server_id"].max()) + 1
    denom = max(1, num_servers - 1)
    merged["position"] = (merged["server_id"].astype(np.float32) / float(denom)).astype(np.float32)

    # Погода по hour
    merged["hour_of_day"] = (merged["hour"] % 24).astype(np.int64)
    hour_sin, hour_cos = _sin_cos_time(merged["hour_of_day"].to_numpy(), 24.0)
    merged["hour_sin"] = hour_sin
    merged["hour_cos"] = hour_cos

    total_hours = int(merged["hour"].max()) + 1
    outside_temp, humidity = _openmeteo_hourly(base_date, total_hours, MOSCOW_LAT, MOSCOW_LON)
    weather_df = pd.DataFrame({
        "hour": np.arange(total_hours, dtype=np.int64),
        "outside_temp": outside_temp.astype(np.float32),
        "humidity": humidity.astype(np.float32),
    })
    merged = merged.merge(weather_df, on="hour", how="left")

    # feature_cols берём из meta (порядок важен)
    if total_hours < input_hours:
        raise RuntimeError("Недостаточно истории (hours) для input window")

    start_hour = total_hours - input_hours

    merged = merged.sort_values(["server_id", "hour"])
    samples = []
    server_ids = []

    for server_id, g in merged.groupby("server_id"):
        g = g.sort_values("hour").set_index("hour")
        full = g.reindex(range(start_hour, total_hours))
        if full[list(feature_cols)].isnull().any().any() or full[list(feature_cols)].empty:
            continue
        X_win = full[list(feature_cols)].to_numpy(dtype=np.float32)
        samples.append(X_win)
        server_ids.append(int(server_id))

    X = np.stack(samples, axis=0)  # [N, input_hours, F]
    Xn = (X - X_mean) / X_std

    device = torch.device(args.device)
    cfg = ModelConfig(input_size=feature_count, horizon_hours=horizon_hours)
    model = TempPredictorLSTM(cfg).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        xb = torch.tensor(Xn, dtype=torch.float32).to(device)
        mean_pred, std_pred, p_over_pred, _ = model(xb)

    # Печать: для каждого сервера вероятность перегрева на любом из horizon часов
    p_any = p_over_pred.max(dim=1).values.cpu().numpy()
    mean_first = mean_pred[:, 0].cpu().numpy()
    std_first = std_pred[:, 0].cpu().numpy()

    print(f"Прогноз выполнен для {len(server_ids)} серверов.")
    for i, sid in enumerate(server_ids):
        print(
            f"server_id={sid} mean(t+1h)={mean_first[i]:.2f}C std={std_first[i]:.2f} "
            f"p_overheat_any={p_any[i]:.3f}"
        )

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, sid in enumerate(server_ids):
            row = {"server_id": sid}
            for k in range(horizon_hours):
                row[f"mean_h{k+1}"] = float(mean_pred[i, k].cpu().numpy())
                row[f"std_h{k+1}"] = float(std_pred[i, k].cpu().numpy())
                row[f"p_overheat_h{k+1}"] = float(p_over_pred[i, k].cpu().numpy())
            rows.append(row)

        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Сохранено: {out_path}")


if __name__ == "__main__":
    main()

