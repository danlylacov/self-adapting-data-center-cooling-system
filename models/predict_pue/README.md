# Hybrid PUE Model (Physics + ML)

Этот модуль добавляет гибридную модель прогноза **PUE**:
- **physics baseline** считает теоретический PUE по упрощенным термодинамическим формулам (`physics_pue.py`);
- **ML** (LSTM, `lstm_pue_residual.py`) предсказывает `residual = pue_real - pue_physics`;
- **API-инференс** (`api.py`) складывает physics + ML и выдает рекомендацию по `cooling_setpoint`.

## Что нужно на вход

Для подготовки датасета используйте результаты симуляции:
- `*_servers.csv`
- `*_summary.csv`

Ожидаемые колонки (из текущего `ResultSaver`):
- в `*_servers.csv`: `step`, `power`, `t_out`
- в `*_summary.csv`: `step`, `room_temperature`, `pue`, `cooling_setpoint`, `cooling_power`, `cooling_fan_speed`
- `outside_temperature` в `*_summary.csv` желательно (добавлено в симуляторе). Если её нет, `prepare_pue_dataset.py` подтянет outside temp из Open-Meteo.

Скрипты ходят в интернет за погодой Open-Meteo (humidity и wind всегда).

## Запуск API (prod inference)

Запуск:

```bash
uvicorn models.predict_pue.api:app --host 0.0.0.0 --port 8010
```

Эндпоинты:
- `GET /health`
- `POST /pue/hybrid/recommend`

`POST /pue/hybrid/recommend` принимает JSON с `history` (длина `input_hours`) и `future` (длина `horizon_hours`) и возвращает:
- baseline energy (`baseline_cooling_energy_kwh`)
- лучший сценарий `best` (delta setpoint, прогноз PUE по горизонту и ожидаемая экономия).

Для работы API нужны артефакты в этой же папке:
- `pue_residual_predictor.pt`
- `pue_residual_meta.json`

## Замечания по интерпретации

- ML residual прогнозируется **только по истории** `input_hours` (будущие `outside_temperature/humidity/wind` не влияют на residual напрямую, но влияют на physics baseline).
- Экономия в рекомендациях оценивается по **ожидаемой мощности охлаждения** за горизонт как:
  `P_cooling ≈ P_servers * (PUE - 1)`

## Оценка качества

Модель обучается предсказывать `residual = pue_real - pue_physics`, поэтому качество можно оценивать в терминах ошибки residual (и, эквивалентно, ошибки `pue_pred` относительно физического baseline при фиксированном `pue_physics`).

На тестовой выборке (split как в `train_pue_residual_predictor.py`, окна: `input_hours=24`, `horizon_hours=6`):
- baseline (без ML, residual=0): `MAE=0.406023`, `RMSE=0.406485`
- hybrid (physics + LSTM residual): `MAE=0.013889`, `RMSE=0.017436`
- улучшение:
  - MAE уменьшилась примерно в `29.3x` (около `+96.58%` по формуле “baseline-hybrid”)
  - RMSE уменьшилась примерно в `23.3x` (около `+95.71%`)

