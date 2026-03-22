#!/usr/bin/env python3
"""
Строит load_hourly.csv (ds, y_sum, календарные признаки) из CSV симуляции.
Формат согласован с models/predict_load/artifacts/load_hourly.csv и load_meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.predict_load.utils import build_time_features, train_val_test_split_indices


def _hour_from_step(step_series: pd.Series, time_step_seconds: int) -> pd.Series:
    step0 = step_series.astype(np.int64) - 1
    return ((step0 * int(time_step_seconds)) // 3600).astype(np.int64)


def _list_servers(results_dir: Path) -> List[Path]:
    return sorted(results_dir.glob("*_servers.csv"))


def _summary_path(p: Path) -> Path:
    return Path(str(p).replace("_servers.csv", "_summary.csv"))


def build_hourly_y_sum(servers_csv: Path, summary_csv: Path, time_step_seconds: int) -> pd.DataFrame:
    servers = pd.read_csv(servers_csv)
    summary = pd.read_csv(summary_csv)
    servers["hour"] = _hour_from_step(servers["step"], time_step_seconds)
    agg = servers.groupby("hour", as_index=False).agg(y_sum_w=("power", "sum"))
    # y_sum в репозитории ~ total kW по стойке (масштаб как в load_hourly)
    agg["y_sum"] = agg["y_sum_w"].astype(np.float64) / 1000.0
    summary["hour"] = _hour_from_step(summary["step"], time_step_seconds)
    # load_base как средняя утилизация * масштаб (упрощённо — средняя утилизация по серверам за час)
    util_h = servers.groupby("hour", as_index=False).agg(load_base=("utilization", "mean"))
    out = agg.merge(util_h, on="hour", how="inner")
    return out[["hour", "y_sum", "load_base"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", type=str, required=True)
    ap.add_argument("--out_csv", type=str, default="models/predict_load/artifacts/load_hourly.csv")
    ap.add_argument("--out_meta", type=str, default="models/predict_load/artifacts/load_meta.json")
    ap.add_argument("--time_step_seconds", type=int, default=300)
    ap.add_argument("--base_date", type=str, default="2019-05-01")
    ap.add_argument("--origin_date", type=str, default=None, help="Переопределить origin_date в meta")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.is_absolute():
        results_dir = _REPO_ROOT / results_dir

    rows = []
    for srv in _list_servers(results_dir):
        sm = _summary_path(srv)
        if not sm.exists():
            continue
        part = build_hourly_y_sum(srv, sm, args.time_step_seconds)
        rows.append(part)

    if not rows:
        raise RuntimeError(f"No data in {results_dir}")

    df = pd.concat(rows, ignore_index=True)
    df = df.groupby("hour", as_index=False).agg({"y_sum": "mean", "load_base": "mean"})
    df = df.sort_values("hour").reset_index(drop=True)

    base = datetime.strptime(args.base_date, "%Y-%m-%d")
    ds_list = [base + timedelta(hours=int(h)) for h in df["hour"]]
    ds = pd.Series(ds_list)

    peak_quantile = 0.9
    how = (df["hour"].astype(int) % 168).values
    y = df["y_sum"].to_numpy()
    thr = np.quantile(y, peak_quantile)
    peak_by_how = {}
    for k in np.unique(how):
        mask = how == k
        peak_by_how[int(k)] = int(np.mean(y[mask]) >= thr * 0.95)

    tf = build_time_features(
        ds=ds,
        peak_event_by_hour_of_week=peak_by_how,
        maintenance_event=None,
        include_holidays=True,
    )
    out_df = pd.DataFrame(
        {
            "ds": ds,
            "load_base": df["load_base"].values,
            "y_sum": df["y_sum"].values,
            "hour_of_day": tf["hour_of_day"].values,
            "day_of_week": tf["day_of_week"].values,
            "month": tf["month"].values,
            "hour_sin": tf["hour_sin"].values,
            "hour_cos": tf["hour_cos"].values,
            "is_holiday": tf["is_holiday"].values,
            "maintenance_event": tf["maintenance_event"].values,
            "peak_event": tf["peak_event"].values,
        }
    )

    out_csv = Path(args.out_csv)
    if not out_csv.is_absolute():
        out_csv = _REPO_ROOT / out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    n = len(out_df)
    te, ve, _ts = train_val_test_split_indices(n, train_ratio=0.7, val_ratio=0.15)
    origin = args.origin_date or args.base_date
    meta = {
        "csv_out_path": (
            str(out_csv.relative_to(_REPO_ROOT))
            if str(out_csv.resolve()).startswith(str(_REPO_ROOT.resolve()))
            else str(out_csv)
        ),
        "npz_out_path": "models/predict_load/artifacts/load_dataset.npz",
        "history_days": 30,
        "horizon_hours": 48,
        "num_servers": 42,
        "stride_seconds": args.time_step_seconds,
        "origin_date": origin,
        "total_hours": n,
        "train_end_idx": int(te),
        "val_end_idx": int(ve),
        "peak_quantile": peak_quantile,
        "peak_event_fraction_threshold": 0.5,
        "peak_event_by_hour_of_week": {str(k): int(v) for k, v in peak_by_how.items()},
        "maintenance_weekdays": [],
        "maintenance_hours": [],
        "maintenance_dates": [],
        "include_holidays": True,
    }
    out_meta = Path(args.out_meta)
    if not out_meta.is_absolute():
        out_meta = _REPO_ROOT / out_meta
    out_meta.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv} ({n} rows) and {out_meta}")


if __name__ == "__main__":
    main()
