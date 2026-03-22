# Load Forecast Model (Prophet + DeepAR)

Simple inference service for total data-center load forecasting.

## What this model does

- Input:
  - historical server load (hourly)
  - calendar features (hour/day/month/holidays)
  - event features (maintenance / known peak periods)
- Output:
  - hourly total-load forecast for the next 24-48 hours
  - decomposition: trend, daily, weekly, yearly
  - peak value and peak time on the forecast horizon

## Implemented models

- `Prophet`
- `DeepAR` (pytorch-forecasting)

The API allows choosing `model_type=prophet|deepar`.

## Current quality (val/test)

### Prophet (`prophet_full48`)

- val: `MAE=1.064`, `RMSE=1.317`, `MAPE=5.51%`
- test: `MAE=1.233`, `RMSE=1.544`, `MAPE=6.15%`

### DeepAR (`deepar_serving`)

- val: `MAE=1.113`, `RMSE=1.327`, `MAPE=5.47%`
- test: `MAE=1.348`, `RMSE=1.749`, `MAPE=6.79%`

Conclusion: Prophet is currently better on test in this repository state.

## Minimal API

File: `models/predict_load/api.py`

### Run

```bash
uvicorn models.predict_load.api:app --host 0.0.0.0 --port 8010
```

### Endpoints

- `GET /health`
- `GET /forecast?model_type=prophet&horizon_hours=48`
- `GET /forecast?model_type=deepar&horizon_hours=24`

## Files kept for serving

- `api.py` - FastAPI wrapper
- `predict_load.py`, `utils.py` - inference helpers
- `artifacts/load_meta.json`, `artifacts/load_hourly.csv`
- `artifacts/prophet_model_prophet_full48.pkl`, `artifacts/prophet_meta_prophet_full48.json`
- `artifacts/deepar_model_deepar_serving.ckpt`,
  `artifacts/deepar_dataset_deepar_serving.pkl`,
  `artifacts/deepar_meta_deepar_serving.json`

