#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_date_list(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def to_epoch_seconds(ts: pd.Timestamp) -> int:
    return int(ts.value // 10**9)


def from_epoch_seconds(seconds: int) -> datetime:
    return datetime.utcfromtimestamp(int(seconds))


def hour_sin_cos(hour_of_day: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hour_of_day = np.asarray(hour_of_day, dtype=np.float32)
    angle = 2.0 * np.pi * hour_of_day / 24.0
    return np.sin(angle).astype(np.float32), np.cos(angle).astype(np.float32)


def build_time_features(
    ds: pd.Series,
    peak_event_by_hour_of_week: Optional[Dict[int, int]] = None,
    maintenance_event: Optional[pd.Series] = None,
    include_holidays: bool = True,
) -> pd.DataFrame:
    """
    Build calendar/time features for each timestamp.

    ds: datetime-like Series (timezone-naive is assumed)
    """
    dt = pd.to_datetime(ds)
    hour_of_day = dt.dt.hour.astype(int)
    day_of_week = dt.dt.dayofweek.astype(int)  # Mon=0
    month = dt.dt.month.astype(int)
    hour_sin, hour_cos = hour_sin_cos(hour_of_day)

    df = pd.DataFrame(
        {
            "ds": dt,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "month": month,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
        }
    )

    if include_holidays:
        try:
            import holidays

            years = sorted(set(dt.dt.year.tolist()))
            ru_holidays = holidays.RU(years=years)
            is_holiday = dt.dt.date.astype(object).map(lambda d: int(d in ru_holidays))
        except Exception:
            # If holidays package is missing or RU calendars fail, fall back to zeros.
            is_holiday = np.zeros(len(df), dtype=np.int64)
    else:
        is_holiday = np.zeros(len(df), dtype=np.int64)

    df["is_holiday"] = is_holiday.astype(np.int64)

    if maintenance_event is None:
        df["maintenance_event"] = np.zeros(len(df), dtype=np.int64)
    else:
        df["maintenance_event"] = maintenance_event.astype(np.int64).values

    if peak_event_by_hour_of_week is None:
        df["peak_event"] = np.zeros(len(df), dtype=np.int64)
    else:
        hour_of_week = (df["day_of_week"] * 24 + df["hour_of_day"]).astype(int)
        df["peak_event"] = hour_of_week.map(lambda k: int(peak_event_by_hour_of_week.get(int(k), 0))).astype(
            np.int64
        )

    return df


def make_maintenance_event(
    ds: pd.Series,
    weekdays: Sequence[int],
    hours: Sequence[int],
    dates: Sequence[str],
) -> pd.Series:
    """
    maintenance_event = 1 for timestamps that match:
      - weekday in `weekdays` AND hour in `hours`, OR
      - date in `dates` (YYYY-MM-DD) and hour in `hours` (or all hours if hours empty).
    """
    dt = pd.to_datetime(ds)
    weekday = dt.dt.dayofweek.astype(int)
    hour = dt.dt.hour.astype(int)
    date_str = dt.dt.strftime("%Y-%m-%d")

    weekdays_set = set(int(x) for x in weekdays)
    hours_set = set(int(x) for x in hours)
    dates_set = set(str(x) for x in dates)

    mask_week = np.array([(int(w) in weekdays_set and int(h) in hours_set) for w, h in zip(weekday, hour)])

    if dates_set:
        if hours_set:
            mask_dates = np.array([(d in dates_set and int(h) in hours_set) for d, h in zip(date_str, hour)])
        else:
            mask_dates = np.array([d in dates_set for d in date_str])
    else:
        mask_dates = np.zeros(len(dt), dtype=bool)

    mask = mask_week | mask_dates
    return pd.Series(mask.astype(np.int64), index=ds.index)


def compute_peak_event_by_hour_of_week(
    df_hourly: pd.DataFrame,
    train_end_idx: int,
    peak_quantile: float,
    peak_event_fraction_threshold: float,
) -> Dict[int, int]:
    """
    Compute known peak schedule purely from TRAIN history:
      - for each hour_of_week (0..167), compute threshold = quantile(y_sum, peak_quantile)
      - peak_event_flag = 1 if fraction(y_sum > threshold) >= peak_event_fraction_threshold
    """
    if train_end_idx <= 0:
        raise ValueError("train_end_idx must be > 0")

    df_train = df_hourly.iloc[:train_end_idx].copy()

    # Be robust: caller may provide only `ds` (and `y_sum`) without derived calendar columns.
    if "day_of_week" in df_train.columns and "hour_of_day" in df_train.columns:
        day_of_week = df_train["day_of_week"].astype(int)
        hour_of_day = df_train["hour_of_day"].astype(int)
    else:
        dt = pd.to_datetime(df_train["ds"])
        day_of_week = dt.dt.dayofweek.astype(int)
        hour_of_day = dt.dt.hour.astype(int)

    df_train["hour_of_week"] = (day_of_week * 24 + hour_of_day).astype(int)

    flags: Dict[int, int] = {}
    for how in range(7 * 24):
        g = df_train[df_train["hour_of_week"] == how]["y_sum"].to_numpy(dtype=np.float64)
        if len(g) == 0:
            flags[how] = 0
            continue
        thr = float(np.quantile(g, peak_quantile))
        frac = float(np.mean(g > thr))
        flags[how] = 1 if frac >= peak_event_fraction_threshold else 0
    return flags


def train_val_test_split_indices(
    n: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[int, int, int]:
    """
    Returns (train_end_idx, val_end_idx, test_start_idx).
    Indices follow Python slicing semantics:
      train: [0, train_end_idx)
      val: [train_end_idx, val_end_idx)
      test: [val_end_idx, n)
    """
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("train_ratio must be in (0,1)")
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be in (0,1)")
    train_end_idx = int(n * train_ratio)
    val_end_idx = int(n * (train_ratio + val_ratio))
    val_end_idx = min(val_end_idx, n)
    test_start_idx = val_end_idx
    return train_end_idx, val_end_idx, test_start_idx


def save_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class PeakInfo:
    peak_quantile: float
    peak_event_fraction_threshold: float
    peak_event_by_hour_of_week: Dict[int, int]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9))) * 100.0)
    return {"mae": mae, "rmse": rmse, "mape_pct": mape}


def build_future_times(last_ds: pd.Timestamp, horizon_hours: int) -> pd.DatetimeIndex:
    return pd.date_range(
        start=last_ds + pd.Timedelta(hours=1),
        periods=horizon_hours,
        freq="1h",
    )

