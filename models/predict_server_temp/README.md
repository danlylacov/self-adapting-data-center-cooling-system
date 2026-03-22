# Server temperature predictor (LSTM + uncertainty)

Модель прогнозирует температуру чипа сервера `t_chip` на горизонте **6 часов** вперёд, а также выдаёт:
1) **неопределённость** (через предсказанный `std`)
2) **вероятность overheat** `P(t_chip > threshold_c)` для каждого часа горизонта.

## Вход / выход API

FastAPI-эндпоинт: `POST /predict`

### Вход
`X` — тензор формы **[servers, 24, features]**

Порядок признаков в последней размерности `features` (11 штук):
1. `utilization`
2. `t_chip` (история)
3. `t_in`
4. `setpoint`
5. `server_fan_speed`
6. `power`
7. `outside_temp`
8. `humidity`
9. `position` (server_id, нормализованный к [0..1])
10. `hour_sin`
11. `hour_cos`

Важно: модель ожидает, что 24 временных шага идут подряд (input window).

### Выход
Возвращается:
- `mean`: `[servers, 6]` прогноз `t_chip` (°C)
- `std`: `[servers, 6]` неопределённость
- `p_overheat`: `[servers, 6]` вероятность `t_chip > threshold_c`

## Запуск FastAPI

### Модель / meta
По умолчанию wrapper пытается загрузить:
- `temp_predictor_mdot002_75_300s.pt`
- `temp_predictor_mdot002_75_300s_meta.json`

Можно переопределить через env:
- `MODEL_PATH`
- `META_PATH`

### Команда
```bash
cd /home/daniil/Рабочий\ стол/bonch/Self-adapting-data-center-cooling-system
.venv/bin/uvicorn models.predict_server_temp.api_fastapi:app --host 0.0.0.0 --port 8002
```

## Пример запроса
```bash
curl -X POST http://localhost:8002/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "X": [
      [
        [0.5, 30.0, 22.0, 22.0, 0.5, 300.0, 18.0, 50.0, 0.1, 0.0, 1.0],
        ...
      ]
    ]
  }'
```
Пример демонстрационный: фактическая форма должна быть **[servers, 24, 11]**, где `...` — остальные 23 временных шага.

## Оценка качества (сценарий с `t_chip > 75°C`)
Метрики на `temp_dataset_mdot002_75_300s.npz` (time-based split):

VAL:
- MAE: **2.2946°C**
- RMSE: **2.8485°C**
- NLL: **1.5441**
- Brier: **0.083718**
- Acc: **0.883**
- overheat positive rate: **0.1206**

TEST:
- MAE: **2.7410°C**
- RMSE: **3.5040°C**
- NLL: **1.8198**
- Brier: **0.068845**
- Acc: **0.918**
- overheat positive rate: **0.0820**

## Важное замечание по интерпретации `Acc`
`P(t_chip > threshold_c)` — редкое событие. Поэтому **accuracy** может быть “хорошей” даже при слабом качестве калибровки.
Ориентироваться лучше на `Brier` и на качество вероятностей по порогам.

