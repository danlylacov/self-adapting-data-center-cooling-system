#!/usr/bin/env python3
"""
Подготовка датасета для модели прогноза t_chip серверов.

Источник:
- DC_digital_twin/results_google_300s/*_servers.csv
- DC_digital_twin/results_google_300s/*_summary.csv

Шаги:
1) агрегируем по server_id и hour
2) добавляем погоду (Open-Meteo): outside_temp, humidity
3) строим окна:
   X: [input_hours, num_features]
   y_mean: [horizon_hours] (будущая t_chip)
   overheat: [horizon_hours] (t_chip > threshold)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from datetime import datetime, timedelta
import math

import requests

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetConfig:
    input_hours: int = 24
    horizon_hours: int = 6
    threshold_c: float = 85.0
    results_dir: str = "DC_digital_twin/results_google_300s"

    # В симуляторе каждый шаг соответствует time_step_seconds
    time_step_seconds: int = 300

    # Маппинг "индекс часов симуляции" -> календарное время для Open-Meteo
    base_date: str = "2019-05-01"

    use_openmeteo: bool = True

    feature_cols: Tuple[str, ...] = (
        "utilization",
        "t_chip",
        "t_in",
        "setpoint",
        "server_fan_speed",
        "power",
        "outside_temp",
        "humidity",
        "position",
        "hour_sin",
        "hour_cos",
    )


def _hour_from_step(step_series: pd.Series, time_step_seconds: int) -> pd.Series:
    """
    step начинается с 1. Время t = (step-1) * time_step_seconds.
    hour_index = t / 3600.
    """
    step0 = step_series.astype(np.int64) - 1
    return ((step0 * int(time_step_seconds)) // 3600).astype(np.int64)


def _sin_cos_time(hour: np.ndarray, period: float) -> Tuple[np.ndarray, np.ndarray]:
    angle = 2.0 * np.pi * hour / period
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


MOSCOW_LAT = 55.7558
MOSCOW_LON = 37.6173


def _openmeteo_hourly(base_date: str, total_hours: int, lat: float, lon: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает массивы длины total_hours:
    outside_temp (°C), humidity (%)
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

    temps = data["hourly"]["temperature_2m"][:total_hours]
    hums = data["hourly"]["relative_humidity_2m"][:total_hours]

    temps = np.array(temps, dtype=np.float32)
    hums = np.array(hums, dtype=np.float32)
    return temps, hums


def _list_run_files(results_dir: Path) -> List[Path]:
    return sorted(list(results_dir.glob("*_servers.csv")))


def _find_summary_for_servers(servers_path: Path) -> Path:
    # sim_..._servers.csv -> sim_..._summary.csv
    return Path(str(servers_path).replace("_servers.csv", "_summary.csv"))


def _read_and_hourly_aggregate(cfg: DatasetConfig, servers_csv: Path, summary_csv: Path) -> pd.DataFrame:
    servers = pd.read_csv(servers_csv)
    summary = pd.read_csv(summary_csv)

    # Нормализуем типы
    for c in ["step", "server_id"]:
        if c in servers.columns:
            servers[c] = servers[c].astype(np.int64)
        if c in summary.columns:
            summary[c] = summary[c].astype(np.int64)

    servers["hour"] = _hour_from_step(servers["step"], cfg.time_step_seconds)
    summary["hour"] = _hour_from_step(summary["step"], cfg.time_step_seconds)

    # Почасовая агрегация по серверам
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

    # Почасовая агрегация по помещению/охлаждению
    sum_hour = (
        summary.groupby(["hour"], as_index=False)
        .agg(
            setpoint=("cooling_setpoint", "mean"),
        )
    )

    # Присоединяем охлаждение к каждому серверу
    merged = srv_hour.merge(sum_hour, on="hour", how="inner")

    # position: нормализованный server_id
    num_servers = int(merged["server_id"].max()) + 1
    denom = max(1, num_servers - 1)
    merged["position"] = (merged["server_id"].astype(np.float32) / float(denom)).astype(np.float32)

    # Погода по hour
    total_hours = int(merged["hour"].max()) + 1
    if cfg.use_openmeteo:
        outside_temp, humidity = _openmeteo_hourly(cfg.base_date, total_hours, MOSCOW_LAT, MOSCOW_LON)
        weather_df = pd.DataFrame({
            "hour": np.arange(total_hours, dtype=np.int64),
            "outside_temp": outside_temp.astype(np.float32),
            "humidity": humidity.astype(np.float32),
        })
    else:
        weather_df = pd.DataFrame({
            "hour": np.arange(total_hours, dtype=np.int64),
            "outside_temp": np.full(total_hours, merged["setpoint"].mean(), dtype=np.float32),
            "humidity": np.full(total_hours, 50.0, dtype=np.float32),
        })

    merged = merged.merge(weather_df, on="hour", how="left")

    # Время суток
    merged["hour_of_day"] = (merged["hour"] % 24).astype(np.int64)
    hour_sin, hour_cos = _sin_cos_time(merged["hour_of_day"].to_numpy(), period=24.0)
    merged["hour_sin"] = hour_sin
    merged["hour_cos"] = hour_cos

    return merged


def build_samples(cfg: DatasetConfig, results_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        X: [N, input_hours, num_features]
        y_mean: [N, horizon_hours]
        overheat: [N, horizon_hours] float32
    """
    run_servers_files = _list_run_files(results_dir)
    if not run_servers_files:
        raise RuntimeError(f"Не найдено *_servers.csv в {results_dir}")

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    overheat_list: List[np.ndarray] = []
    window_start_list: List[int] = []
    server_id_list: List[int] = []

    for servers_csv in run_servers_files:
        summary_csv = _find_summary_for_servers(servers_csv)
        if not summary_csv.exists():
            continue

        merged = _read_and_hourly_aggregate(cfg, servers_csv, summary_csv)
        if merged.empty:
            continue

        total_hours = int(merged["hour"].max()) + 1
        if total_hours < (cfg.input_hours + cfg.horizon_hours):
            # Не хватает длины по времени для окон.
            continue

        # Чтобы было проще: гарантируем сортировку
        merged = merged.sort_values(["server_id", "hour"]).reset_index(drop=True)

        # Превращаем в "плоский" тензор по server_id:
        # Для каждого server_id строим последовательность по hour
        for server_id, g in merged.groupby("server_id"):
            g = g.sort_values("hour")
            g = g.set_index("hour")
            full = g.reindex(range(total_hours))
            # Если где-то в шкале есть пропуски — пропускаем run/server
            if full[list(cfg.feature_cols)].isnull().any().any() or full["t_chip"].isnull().any():
                continue

            # Берём вектор по признакам на каждый час
            seq_features = full[list(cfg.feature_cols)].to_numpy(dtype=np.float32)
            # Целевая t_chip на каждый час
            seq_t_chip = full["t_chip"].to_numpy(dtype=np.float32)

            # Скользящие окна
            max_start = total_hours - (cfg.input_hours + cfg.horizon_hours) + 1
            for start in range(max_start):
                x_win = seq_features[start : start + cfg.input_hours]  # [input_hours, F]
                y_win = seq_t_chip[start + cfg.input_hours : start + cfg.input_hours + cfg.horizon_hours]  # [horizon]

                overheat = (y_win > cfg.threshold_c).astype(np.float32)

                X_list.append(x_win)
                y_list.append(y_win)
                overheat_list.append(overheat)
                window_start_list.append(int(start))
                server_id_list.append(int(server_id))

    if not X_list:
        raise RuntimeError(
            "Не удалось построить обучающие окна. "
            "Проверь input_hours/horizon_hours и длину симуляций в results_dir."
        )

    X = np.stack(X_list, axis=0).astype(np.float32)  # [N, input_hours, F]
    y_mean = np.stack(y_list, axis=0).astype(np.float32)  # [N, horizon]
    overheat = np.stack(overheat_list, axis=0).astype(np.float32)  # [N, horizon]
    window_start_hour = np.array(window_start_list, dtype=np.int64)  # [N]
    server_id_arr = np.array(server_id_list, dtype=np.int64)  # [N]
    return X, y_mean, overheat, window_start_hour, server_id_arr


def main():
    parser = argparse.ArgumentParser(description="Подготовка датасета для LSTM прогнозов.")
    parser.add_argument("--results_dir", type=str, default="DC_digital_twin/results_google_300s")
    parser.add_argument("--output_npz", type=str, default="temp_dataset.npz")
    parser.add_argument("--input_hours", type=int, default=24)
    parser.add_argument("--horizon_hours", type=int, default=6)
    parser.add_argument("--threshold_c", type=float, default=85.0)
    parser.add_argument("--time_step_seconds", type=int, default=300)
    parser.add_argument("--base_date", type=str, default="2019-05-01")
    parser.add_argument("--no_openmeteo", action="store_true", help="Не использовать Open-Meteo.")
    args = parser.parse_args()

    cfg = DatasetConfig(
        input_hours=args.input_hours,
        horizon_hours=args.horizon_hours,
        threshold_c=args.threshold_c,
        results_dir=args.results_dir,
        time_step_seconds=args.time_step_seconds,
        base_date=args.base_date,
        use_openmeteo=not args.no_openmeteo,
    )

    results_dir = Path(args.results_dir)
    X, y_mean, overheat, window_start_hour, server_id_arr = build_samples(cfg, results_dir)

    # Простое сохранение: дальше обучаем уже из NPZ
    output_npz = Path(args.output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        X=X,
        y_mean=y_mean,
        overheat=overheat,
        window_start_hour=window_start_hour,
        server_id_arr=server_id_arr,
        input_hours=cfg.input_hours,
        horizon_hours=cfg.horizon_hours,
        threshold_c=cfg.threshold_c,
        time_step_seconds=cfg.time_step_seconds,
        base_date=cfg.base_date,
        feature_cols=",".join(cfg.feature_cols),
    )
    print(f"Сохранено: {output_npz}")
    print(f"X: {X.shape}, y_mean: {y_mean.shape}, overheat: {overheat.shape}")


if __name__ == "__main__":
    main()

