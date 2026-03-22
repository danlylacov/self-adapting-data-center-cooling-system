#!/usr/bin/env python3
"""
Обучение Prophet на load_hourly.csv; сохраняет prophet_model_*.pkl и prophet_meta_*.json.
DeepAR не обучается здесь (тяжёлые зависимости и время); при необходимости — отдельный ноутбук 02.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.predict_load.utils import compute_metrics, load_json, train_val_test_split_indices


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--load_csv", type=str, default="models/predict_load/artifacts/load_hourly.csv")
    ap.add_argument("--load_meta", type=str, default="models/predict_load/artifacts/load_meta.json")
    ap.add_argument(
        "--run_id",
        type=str,
        default="prophet_full48",
        help="Имя артефакта (api.py использует prophet_full48 по умолчанию)",
    )
    ap.add_argument("--artifacts_dir", type=str, default="models/predict_load/artifacts")
    args = ap.parse_args()

    art = Path(args.artifacts_dir)
    if not art.is_absolute():
        art = _REPO_ROOT / art
    load_csv = Path(args.load_csv)
    if not load_csv.is_absolute():
        load_csv = _REPO_ROOT / load_csv
    load_meta = load_json(Path(args.load_meta) if Path(args.load_meta).is_absolute() else _REPO_ROOT / args.load_meta)

    df = pd.read_csv(load_csv, parse_dates=["ds"])
    regressors = ["is_holiday", "maintenance_event", "peak_event"]
    for r in regressors:
        if r not in df.columns:
            df[r] = 0

    n = len(df)
    te, ve, _ = train_val_test_split_indices(n, train_ratio=0.7, val_ratio=0.15)
    df_train = df.iloc[:te].copy()
    df_val = df.iloc[te:ve].copy()
    df_test = df.iloc[ve:].copy()

    try:
        from prophet import Prophet
    except ImportError as e:
        raise RuntimeError("pip install prophet") from e

    prophet_df = df_train.rename(columns={"y_sum": "y"})[["ds", "y"] + regressors]
    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
        changepoint_prior_scale=0.05,
    )
    for r in regressors:
        m.add_regressor(r, mode="additive")
    m.fit(prophet_df)

    def _eval(sub: pd.DataFrame, name: str) -> dict:
        fc_in = sub.rename(columns={"y_sum": "y"})[["ds", "y"] + regressors]
        pred = m.predict(fc_in)
        y_true = sub["y_sum"].to_numpy(dtype=np.float64)
        y_hat = pred["yhat"].to_numpy(dtype=np.float64)
        return compute_metrics(y_true, y_hat)

    metrics = {"val": _eval(df_val, "val"), "test": _eval(df_test, "test")}

    run_id = args.run_id
    pkl_path = art / f"prophet_model_{run_id}.pkl"
    meta_path = art / f"prophet_meta_{run_id}.json"

    with open(pkl_path, "wb") as f:
        pickle.dump(m, f)

    meta = {
        "train_end_idx": te,
        "val_end_idx": ve,
        "horizon_hours": int(load_meta.get("horizon_hours", 48)),
        "regressors": regressors,
        "prophet_settings": {
            "changepoint_prior_scale": 0.05,
            "seasonality_mode": "additive",
            "daily_seasonality": True,
            "weekly_seasonality": True,
            "yearly_seasonality": True,
        },
        "metrics": metrics,
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {pkl_path} and {meta_path}")
    print("metrics:", metrics)


if __name__ == "__main__":
    main()
