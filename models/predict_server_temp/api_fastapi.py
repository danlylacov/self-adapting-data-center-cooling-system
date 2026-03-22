#!/usr/bin/env python3
"""
FastAPI wrapper для LSTM прогнозов температуры серверов.

Вход:
  X: [servers, 24, features] в том же порядке признаков, что и в meta["feature_cols"]

Выход:
  mean: [servers, 6]
  std:  [servers, 6]
  p_overheat: [servers, 6]  (вероятность t_chip > threshold_c)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .lstm_temp_predictor import ModelConfig, TempPredictorLSTM
except ImportError:
    # Fallback for direct script execution.
    from lstm_temp_predictor import ModelConfig, TempPredictorLSTM


app = FastAPI(title="Server temperature predictor")


class HealthResponse(BaseModel):
    status: str = "ok"


class PredictRequest(BaseModel):
    # X: [servers, 24, features]
    X: List[List[List[float]]] = Field(..., description="Tensor-like nested list")


class PredictResponse(BaseModel):
    mean: List[List[float]]
    std: List[List[float]]
    p_overheat: List[List[float]]


def _resolve_path(env_name: str, default_rel: str) -> Path:
    base_dir = Path(__file__).resolve().parent
    p = os.environ.get(env_name, str(base_dir / default_rel))
    return Path(p)


MODEL_PATH = _resolve_path("MODEL_PATH", "temp_predictor_mdot002_75_300s.pt")
META_PATH = _resolve_path("META_PATH", "temp_predictor_mdot002_75_300s_meta.json")


def _load_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
    if not META_PATH.exists():
        raise FileNotFoundError(f"META_PATH not found: {META_PATH}")

    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    feature_count = int(meta["feature_count"])
    horizon_hours = int(meta["horizon_hours"])

    X_mean = np.array(meta["X_mean"], dtype=np.float32).reshape(1, 1, feature_count)
    X_std = np.array(meta["X_std"], dtype=np.float32).reshape(1, 1, feature_count)

    model = TempPredictorLSTM(ModelConfig(input_size=feature_count, horizon_hours=horizon_hours))
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    return {
        "meta": meta,
        "feature_count": feature_count,
        "horizon_hours": horizon_hours,
        "X_mean": X_mean,
        "X_std": X_std,
        "model": model,
    }


try:
    bundle = _load_bundle()
except Exception as e:
    bundle = None
    # Важно: не падаем на импорт, чтобы /health жил.
    _init_error = str(e)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(status="ok" if bundle is not None else f"error: {globals().get('_init_error','unknown')}")


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if bundle is None:
        raise HTTPException(status_code=500, detail=f"Model bundle not loaded: {globals().get('_init_error','unknown')}")

    X = np.array(req.X, dtype=np.float32)
    if X.ndim != 3:
        raise HTTPException(status_code=400, detail="X must be rank-3: [servers, 24, features]")

    feature_count = bundle["feature_count"]
    horizon_hours = bundle["horizon_hours"]

    if X.shape[-1] != feature_count:
        raise HTTPException(
            status_code=400,
            detail=f"Bad feature dimension: got {X.shape[-1]}, expected {feature_count} (from meta.feature_cols).",
        )
    if X.shape[1] != int(bundle["meta"]["input_hours"]):
        raise HTTPException(
            status_code=400,
            detail=f"Bad time dimension: got {X.shape[1]}, expected input_hours={bundle['meta']['input_hours']}.",
        )

    # Normalize using training stats
    X_mean = bundle["X_mean"]
    X_std = bundle["X_std"]
    Xn = (X - X_mean) / X_std

    xb = torch.tensor(Xn, dtype=torch.float32)
    with torch.no_grad():
        mean_pred, std_pred, p_over_pred, _ = bundle["model"](xb)

    # mean_pred/std_pred/p_over_pred: [servers, horizon]
    mean_out = mean_pred.cpu().numpy().tolist()
    std_out = std_pred.cpu().numpy().tolist()
    p_out = p_over_pred.cpu().numpy().tolist()

    # Sanity: horizon must match
    if len(mean_out) > 0 and len(mean_out[0]) != horizon_hours:
        raise HTTPException(status_code=500, detail="Output horizon mismatch.")

    return PredictResponse(mean=mean_out, std=std_out, p_overheat=p_out)  # type: ignore[arg-type]

