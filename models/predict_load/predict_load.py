#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .utils import (
        build_time_features,
        compute_metrics,
        ensure_dir,
        load_json,
        make_maintenance_event,
    )
except ImportError:
    from utils import (
        build_time_features,
        compute_metrics,
        ensure_dir,
        load_json,
        make_maintenance_event,
    )


def _default_run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


_GROUP_NORMALIZER_GET_PARAMETERS_ORIG = None


def _patch_group_normalizer_get_parameters() -> None:
    """Work around pytorch_forecasting GroupNormalizer + pandas .loc[group] edge cases (e.g. wrong tuple length)."""
    global _GROUP_NORMALIZER_GET_PARAMETERS_ORIG
    from pytorch_forecasting.data.encoders import GroupNormalizer

    if _GROUP_NORMALIZER_GET_PARAMETERS_ORIG is None:
        _GROUP_NORMALIZER_GET_PARAMETERS_ORIG = GroupNormalizer.get_parameters

    def _safe(self, groups, group_names=None):
        import torch

        if isinstance(groups, torch.Tensor):
            groups = tuple(groups.detach().cpu().reshape(-1).tolist())
        elif isinstance(groups, list):
            groups = tuple(groups)
        elif isinstance(groups, np.ndarray):
            groups = tuple(groups.reshape(-1).tolist())

        ng = len(self._groups)
        if ng == 1 and len(groups) > 1:
            groups = (groups[0],)
        elif ng > 0 and len(groups) > ng:
            groups = tuple(groups[:ng])

        if ng == 1 and len(groups) == 1:
            try:
                idx = self.norm_.index
                dt = getattr(idx, "dtype", None)
                if dt is not None and getattr(dt, "kind", None) in ("i", "u"):
                    groups = (int(groups[0]),)
            except Exception:
                pass

        return _GROUP_NORMALIZER_GET_PARAMETERS_ORIG(self, groups, group_names)

    GroupNormalizer.get_parameters = _safe


def _restore_group_normalizer_get_parameters() -> None:
    global _GROUP_NORMALIZER_GET_PARAMETERS_ORIG
    if _GROUP_NORMALIZER_GET_PARAMETERS_ORIG is None:
        return
    from pytorch_forecasting.data.encoders import GroupNormalizer

    GroupNormalizer.get_parameters = _GROUP_NORMALIZER_GET_PARAMETERS_ORIG


def _try_predictive_components(prophet_model, df: pd.DataFrame) -> pd.DataFrame:
    if hasattr(prophet_model, "predictive_components"):
        return prophet_model.predictive_components(df)
    # Fallback for other Prophet versions
    if hasattr(prophet_model, "predict_components"):
        return prophet_model.predict_components(df)
    raise AttributeError("Prophet model has no predictive_components/predict_components methods")


def _make_future_covariates(
    history_last_ds: pd.Timestamp,
    horizon_hours: int,
    load_meta: Dict[str, object],
    maintenance_weekdays: List[int],
    maintenance_hours: List[int],
    maintenance_dates: List[str],
) -> pd.DataFrame:
    # Future ds points
    future_ds = pd.date_range(
        start=history_last_ds + pd.Timedelta(hours=1),
        periods=int(horizon_hours),
        freq="1h",
    )

    # Peak event schedule is computed from TRAIN in prepare_load_dataset.py
    peak_event_by_hour_of_week = load_meta.get("peak_event_by_hour_of_week") or {}

    maintenance_event = make_maintenance_event(
        ds=pd.Series(future_ds),
        weekdays=maintenance_weekdays,
        hours=maintenance_hours,
        dates=maintenance_dates,
    )

    time_features = build_time_features(
        ds=pd.Series(future_ds),
        peak_event_by_hour_of_week=peak_event_by_hour_of_week,
        maintenance_event=maintenance_event,
        include_holidays=bool(load_meta.get("include_holidays", True)),
    )
    return pd.concat(
        [pd.DataFrame({"ds": future_ds}), time_features.drop(columns=["ds"])],
        axis=1,
    )


def predict_with_prophet(
    prophet_model_path: Path,
    load_csv_path: Path,
    load_meta: Dict[str, object],
    prophet_meta: Dict[str, object],
    run_id: str,
    horizon_hours: int,
    history_hours: Optional[int],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    try:
        import prophet  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "Prophet is not installed. Install from models/predict_load/requirements_model_load.txt"
        ) from e

    with open(prophet_model_path, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(load_csv_path, parse_dates=["ds"])

    regressors: List[str] = list(prophet_meta.get("regressors") or ["is_holiday", "maintenance_event", "peak_event"])
    for r in regressors:
        if r not in df.columns:
            raise KeyError(f"Missing regressor column {r} in {load_csv_path}")

    if history_hours is None:
        history_hours = int(load_meta.get("history_days", 30)) * 24
    history_hours = int(history_hours)

    history = df.iloc[-history_hours:].copy()
    last_ds = history["ds"].iloc[-1]

    # Build future input for Prophet regressors.
    maintenance_weekdays = load_meta.get("maintenance_weekdays") or []
    maintenance_hours = load_meta.get("maintenance_hours") or []
    maintenance_dates = load_meta.get("maintenance_dates") or []

    future_covariates = _make_future_covariates(
        history_last_ds=last_ds,
        horizon_hours=horizon_hours,
        load_meta=load_meta,
        maintenance_weekdays=maintenance_weekdays,
        maintenance_hours=maintenance_hours,
        maintenance_dates=maintenance_dates,
    )

    future_input = future_covariates[["ds"] + regressors].copy()

    forecast_future = model.predict(future_input)
    # Forecast includes ds and yhat/yhat_lower/yhat_upper

    # Components (trend + daily/weekly/yearly).
    # Prophet versions differ: some don't expose predictive_components(), but `predict()`
    # typically returns these columns directly.
    comp_future = forecast_future.copy()

    # Peaks: max yhat on horizon
    yhat = forecast_future["yhat"].to_numpy(dtype=np.float64)
    peak_idx = int(np.argmax(yhat))
    peak_ds = pd.to_datetime(forecast_future["ds"].iloc[peak_idx])
    peak_val = float(yhat[peak_idx])

    forecast_out = pd.DataFrame(
        {
            "ds": pd.to_datetime(forecast_future["ds"]),
            "yhat_mean": forecast_future["yhat"].astype(np.float64),
            "yhat_lower": forecast_future.get("yhat_lower", pd.Series([np.nan] * len(forecast_future))).astype(
                np.float64
            ),
            "yhat_upper": forecast_future.get("yhat_upper", pd.Series([np.nan] * len(forecast_future))).astype(
                np.float64
            ),
        }
    )

    # Keep only those decomposition columns that exist in this Prophet version.
    component_cols = [c for c in ["trend", "daily", "weekly", "yearly"] if c in comp_future.columns]
    if "ds" not in comp_future.columns:
        comp_future = comp_future.copy()
        comp_future["ds"] = future_input["ds"].values[: len(comp_future)]
    components_out = comp_future[["ds"] + component_cols].copy()

    peaks_out = {
        "peak_value": peak_val,
        "peak_time": str(peak_ds),
        "peak_idx": peak_idx,
        "horizon_hours": int(horizon_hours),
        "model_type": "prophet",
        "run_id": run_id,
    }

    ensure_dir(out_dir)
    forecast_csv_path = out_dir / f"forecast_prophet_{run_id}.csv"
    components_csv_path = out_dir / f"components_prophet_{run_id}.csv"
    peaks_json_path = out_dir / f"peaks_prophet_{run_id}.json"

    forecast_out.to_csv(forecast_csv_path, index=False)
    components_out.to_csv(components_csv_path, index=False)
    peaks_json_path.write_text(json.dumps(peaks_out, ensure_ascii=False, indent=2), encoding="utf-8")

    return forecast_out, components_out, peaks_out


def predict_with_deepar(
    deepar_model_ckpt: Path,
    deepar_dataset_pkl: Path,
    load_csv_path: Path,
    load_meta: Dict[str, object],
    deepar_meta: Dict[str, object],
    run_id: str,
    horizon_hours: int,
    history_hours: Optional[int],
    out_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    # Import ML libs only at runtime.
    try:
        import lightning.pytorch as pl
        from pytorch_forecasting import DeepAR, TimeSeriesDataSet
    except Exception as e:
        raise RuntimeError(
            "DeepAR dependencies are not installed. Install from models/predict_load/requirements_model_load.txt"
        ) from e

    with open(deepar_dataset_pkl, "rb") as f:
        training_dataset = pickle.load(f)

    # Load DeepAR model from checkpoint
    model = DeepAR.load_from_checkpoint(str(deepar_model_ckpt))

    df = pd.read_csv(load_csv_path, parse_dates=["ds"])

    # DeepAR dataset requires group ids and monotonically increasing time index.
    if "time_idx" not in df.columns:
        df = df.copy()
        df["time_idx"] = np.arange(len(df), dtype=np.int64)
    if "series_id" not in df.columns:
        df = df.copy()
        df["series_id"] = 0

    # Choose encoder window.
    encoder_length_hours = int(deepar_meta.get("encoder_length_hours", horizon_hours * 2))
    lags_hours = deepar_meta.get("lags_hours") or []
    max_lag = max([int(x) for x in lags_hours] or [0])

    if history_hours is None:
        history_hours = int(load_meta.get("history_days", 30)) * 24
    history_hours = min(int(history_hours), len(df))
    # Ensure we have enough history to compute lagged target features for the encoder.
    min_required_history = int(encoder_length_hours + max_lag)
    if history_hours < min_required_history:
        history_hours = int(len(df)) if len(df) >= min_required_history else int(history_hours)

    history = df.iloc[-history_hours:].copy()
    last_ds = history["ds"].iloc[-1]
    last_time_idx = int(history["time_idx"].iloc[-1])

    # DeepAR expects categorical cols as strings.
    for col in ["hour_of_day", "day_of_week", "month"]:
        if col in history.columns:
            history[col] = history[col].astype(int).astype(str)

    maintenance_weekdays = load_meta.get("maintenance_weekdays") or []
    maintenance_hours = load_meta.get("maintenance_hours") or []
    maintenance_dates = load_meta.get("maintenance_dates") or []

    future_covariates = _make_future_covariates(
        history_last_ds=last_ds,
        horizon_hours=horizon_hours,
        load_meta=load_meta,
        maintenance_weekdays=maintenance_weekdays,
        maintenance_hours=maintenance_hours,
        maintenance_dates=maintenance_dates,
    )

    # Build extended dataframe for prediction.
    # Note: in this pytorch-forecasting version, target values in decoder must be finite
    # even in prediction mode. We fill future y_sum with a placeholder (last observed value).
    future_covariates_ext = future_covariates.copy()
    last_y = float(history["y_sum"].iloc[-1])
    future_covariates_ext["y_sum"] = np.float32(last_y)
    future_covariates_ext["series_id"] = 0

    start_future_idx = last_time_idx + 1
    future_time_idx = np.arange(start_future_idx, start_future_idx + int(horizon_hours), dtype=np.int64)
    future_covariates_ext["time_idx"] = future_time_idx

    # Ensure categorical columns for future.
    for col in ["hour_of_day", "day_of_week", "month"]:
        if col in future_covariates_ext.columns:
            future_covariates_ext[col] = future_covariates_ext[col].astype(int).astype(str)

    extended = pd.concat([history, future_covariates_ext], axis=0, ignore_index=True)

    # Align group id dtype with the fitted GroupNormalizer index (avoids pandas loc edge cases).
    try:
        tn0 = training_dataset.target_normalizer
        if hasattr(tn0, "norm_") and hasattr(tn0.norm_, "index"):
            idx_dtype = tn0.norm_.index.dtype
            if extended["series_id"].dtype != idx_dtype:
                extended = extended.copy()
                extended["series_id"] = extended["series_id"].astype(idx_dtype)
    except Exception:
        pass

    # Build predict dataset and run prediction.
    predict_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        extended,
        predict=True,
        stop_randomization=True,
    )

    predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        inference_mode=True,
    )

    _patch_group_normalizer_get_parameters()
    try:
        preds = model.predict(predict_dataloader)
    finally:
        _restore_group_normalizer_get_parameters()
    preds_np = np.asarray(preds)

    # Expected output for NormalDistributionLoss is typically mean predictions:
    # shapes can be [n_samples, pred_len] or [pred_len] depending on batching.
    if preds_np.ndim == 2:
        # Average over batch dimension for a stable single trajectory.
        deepar_yhat = preds_np.mean(axis=0).astype(np.float64)
    elif preds_np.ndim == 1:
        deepar_yhat = preds_np.astype(np.float64)
    elif preds_np.ndim == 3:
        # Fallback: if output still includes distribution/quantile-like dimensions, take the first component.
        deepar_yhat = preds_np[..., 0].mean(axis=0).astype(np.float64)
    else:
        raise RuntimeError(f"Unexpected predictions shape from DeepAR: {preds_np.shape}")
    future_ds = future_covariates["ds"].to_numpy()

    # Peaks
    peak_idx = int(np.argmax(deepar_yhat))
    peak_ds = pd.to_datetime(future_ds[peak_idx])
    peak_val = float(deepar_yhat[peak_idx])

    forecast_out = pd.DataFrame(
        {
            "ds": pd.to_datetime(future_ds),
            "yhat_mean": deepar_yhat,
            "yhat_lower": np.nan,
            "yhat_upper": np.nan,
        }
    )

    # DeepAR components: take Prophet decomposition and adjust trend to match DeepAR mean.
    # This keeps the trend/daily/weekly/yearly decomposition consistent with plan.
    try:
        from prophet import Prophet
    except Exception as e:
        raise RuntimeError("Prophet is required for DeepAR components decomposition") from e

    prophet_regressors = ["is_holiday", "maintenance_event", "peak_event"]
    hist_for_prophet = history.copy()
    for r in prophet_regressors:
        if r not in hist_for_prophet.columns:
            hist_for_prophet[r] = 0
    prophet_df = hist_for_prophet.rename(columns={"y_sum": "y"})[["ds", "y"] + prophet_regressors]

    m = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode="additive",
    )
    for r in prophet_regressors:
        m.add_regressor(r, mode="additive")
    m.fit(prophet_df)

    future_input = future_covariates[["ds"] + prophet_regressors].copy()
    prophet_forecast = m.predict(future_input)
    prophet_yhat = prophet_forecast["yhat"].to_numpy(dtype=np.float64)

    components_cols = [c for c in ["trend", "daily", "weekly", "yearly"] if c in prophet_forecast.columns]
    components_out = prophet_forecast[["ds"] + components_cols].copy()
    if "trend" in components_out.columns:
        delta = deepar_yhat - prophet_yhat
        components_out["trend"] = components_out["trend"].astype(np.float64) + delta

    peaks_out = {
        "peak_value": peak_val,
        "peak_time": str(peak_ds),
        "peak_idx": peak_idx,
        "horizon_hours": int(horizon_hours),
        "model_type": "deepar",
        "run_id": run_id,
    }

    ensure_dir(out_dir)
    forecast_csv_path = out_dir / f"forecast_deepar_{run_id}.csv"
    components_csv_path = out_dir / f"components_deepar_{run_id}.csv"
    peaks_json_path = out_dir / f"peaks_deepar_{run_id}.json"

    forecast_out.to_csv(forecast_csv_path, index=False)
    components_out.to_csv(components_csv_path, index=False)
    peaks_json_path.write_text(json.dumps(peaks_out, ensure_ascii=False, indent=2), encoding="utf-8")

    return forecast_out, components_out, peaks_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified load forecasting inference (Prophet / DeepAR).")
    parser.add_argument("--model_type", type=str, choices=["prophet", "deepar"], required=True)
    parser.add_argument("--horizon_hours", type=int, default=48)
    parser.add_argument("--history_hours", type=int, default=None)
    parser.add_argument("--run_id", type=str, default=_default_run_id())
    parser.add_argument("--artifacts_dir", type=str, default="models/predict_load/artifacts")
    parser.add_argument("--outputs_dir", type=str, default="models/predict_load/outputs")

    parser.add_argument("--model_run_id", type=str, default="default")

    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    out_dir = Path(args.outputs_dir)

    load_meta = load_json(artifacts_dir / "load_meta.json")
    load_csv_path = Path(load_meta["csv_out_path"]) if "csv_out_path" in load_meta else (artifacts_dir / "load_hourly.csv")

    if args.model_type == "prophet":
        prophet_model_path = artifacts_dir / f"prophet_model_{args.model_run_id}.pkl"
        prophet_meta_path = artifacts_dir / f"prophet_meta_{args.model_run_id}.json"
        prophet_meta = load_json(prophet_meta_path)
        predict_with_prophet(
            prophet_model_path=prophet_model_path,
            load_csv_path=load_csv_path,
            load_meta=load_meta,
            prophet_meta=prophet_meta,
            run_id=args.run_id,
            horizon_hours=args.horizon_hours,
            history_hours=args.history_hours,
            out_dir=out_dir,
        )
    else:
        deepar_meta_path = artifacts_dir / f"deepar_meta_{args.model_run_id}.json"
        deepar_meta = load_json(deepar_meta_path)
        predict_with_deepar(
            deepar_model_ckpt=artifacts_dir / f"deepar_model_{args.model_run_id}.ckpt",
            deepar_dataset_pkl=artifacts_dir / f"deepar_dataset_{args.model_run_id}.pkl",
            load_csv_path=load_csv_path,
            load_meta=load_meta,
            deepar_meta=deepar_meta,
            run_id=args.run_id,
            horizon_hours=args.horizon_hours,
            history_hours=args.history_hours,
            out_dir=out_dir,
        )

    print(f"Saved outputs to: {out_dir}")


if __name__ == "__main__":
    main()

