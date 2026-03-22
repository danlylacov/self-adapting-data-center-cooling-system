#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Literal

from fastapi import FastAPI, HTTPException, Query

try:
    from .predict_load import predict_with_deepar, predict_with_prophet
    from .utils import load_json
except ImportError:
    # Fallback for direct script execution.
    from predict_load import predict_with_deepar, predict_with_prophet
    from utils import load_json


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Fixed production-like model runs (best available in this repo state).
PROPHET_RUN_ID = "prophet_full48"
DEEPAR_RUN_ID = "deepar_serving"


app = FastAPI(title="Predict Load API", version="1.0.0")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"status": "ok"}


@app.get("/forecast")
def forecast(
    model_type: Literal["prophet", "deepar"] = Query(default="prophet"),
    horizon_hours: int = Query(default=48, ge=24, le=48),
) -> Dict[str, Any]:
    load_meta_path = ARTIFACTS_DIR / "load_meta.json"
    load_csv_path = ARTIFACTS_DIR / "load_hourly.csv"
    if not load_meta_path.exists() or not load_csv_path.exists():
        raise HTTPException(status_code=500, detail="Missing core artifacts: load_meta.json/load_hourly.csv")

    load_meta = load_json(load_meta_path)
    run_id = f"api_{model_type}_{horizon_hours}h"

    try:
        if model_type == "prophet":
            prophet_meta = load_json(ARTIFACTS_DIR / f"prophet_meta_{PROPHET_RUN_ID}.json")
            forecast_df, components_df, peaks = predict_with_prophet(
                prophet_model_path=ARTIFACTS_DIR / f"prophet_model_{PROPHET_RUN_ID}.pkl",
                load_csv_path=load_csv_path,
                load_meta=load_meta,
                prophet_meta=prophet_meta,
                run_id=run_id,
                horizon_hours=horizon_hours,
                history_hours=None,
                out_dir=OUTPUTS_DIR,
            )
        else:
            deepar_meta = load_json(ARTIFACTS_DIR / f"deepar_meta_{DEEPAR_RUN_ID}.json")
            deepar_horizon = int(deepar_meta.get("horizon_hours", 48))
            if int(horizon_hours) != deepar_horizon:
                raise HTTPException(
                    status_code=400,
                    detail=f"DeepAR supports fixed horizon_hours={deepar_horizon} for current checkpoint",
                )
            forecast_df, components_df, peaks = predict_with_deepar(
                deepar_model_ckpt=ARTIFACTS_DIR / f"deepar_model_{DEEPAR_RUN_ID}.ckpt",
                deepar_dataset_pkl=ARTIFACTS_DIR / f"deepar_dataset_{DEEPAR_RUN_ID}.pkl",
                load_csv_path=load_csv_path,
                load_meta=load_meta,
                deepar_meta=deepar_meta,
                run_id=run_id,
                horizon_hours=horizon_hours,
                history_hours=None,
                out_dir=OUTPUTS_DIR,
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}") from e

    return {
        "model_type": model_type,
        "horizon_hours": horizon_hours,
        "peak": peaks,
        "forecast": forecast_df.to_dict(orient="records"),
        "components": components_df.to_dict(orient="records"),
    }

