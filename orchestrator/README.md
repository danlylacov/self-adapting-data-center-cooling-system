# Оркестратор (ML-прогон)

FastAPI-сервис, который выполняет прогон цифрового двойника (`DC_digital_twin`) и на каждом **тике управления** вызывает:

- **predict_load** — почасовой прогноз нагрузки (`GET /forecast`, минимум `horizon_hours=24`; для PUE берутся первые 6 часов).
- **predict_pue** — `POST /pue/hybrid/recommend` (история длины `input_hours` = **24 последних шага симуляции**, горизонт 6).
- **predict_server_temp** — `POST /predict` с тензором `X` формы `[servers, 24, 11]`; при агрегированном `p_overheat` выше порога — разгон вентиляторов (по умолчанию до 100%).

## Важно: 24 «часа» vs шаги симуляции

Буферы истории для PUE и температурной модели — это **24 последних шага** симуляции, а не обязательно 24 часа физического времени. Шаг задаётся `deltaTime` (сек). Для демо это нормально; качество моделей может отличаться от обучения на часовых рядах.

## Порты и переменные окружения

| Переменная   | По умолчанию              | Назначение        |
|-------------|---------------------------|-------------------|
| `TWIN_BASE` | `http://127.0.0.1:8000`   | API двойника      |
| `PUE_BASE`  | `http://127.0.0.1:8011`   | predict_pue       |
| `TEMP_BASE` | `http://127.0.0.1:8002`   | predict_server_temp |
| `LOAD_BASE` | `http://127.0.0.1:8010`   | predict_load      |
| `GA_BASE`   | `http://127.0.0.1:8013`   | predict_ga (генетическая политика) |

Оркестратор слушает **8030** (см. `__main__.py` и Docker).

## GA-прогон (`POST /run/ga`)

Отдельный режим без ML: на каждом тике управления оркестратор вызывает сервис **predict_ga** (`POST /recommend` с `avg_chip_temp` и `setpoint_c`), применяет возвращённые уставку и скорость вентилятора. Прогноз нагрузки, PUE и temp-модель не используются.

- Сервис: каталог [`models/predict_ga/`](../models/predict_ga/), образ собирается из корня `v2` (см. `Dockerfile`).
- Веса политики: `GA/tuned_params.json` копируется в образ; переменная **`TUNED_PARAMS_PATH`** переопределяет путь.
- Локальный запуск predict_ga (из корня `v2`):

```bash
PYTHONPATH=. uvicorn models.predict_ga.app:app --host 127.0.0.1 --port 8013
```

- `GET /health` оркестратора дополнительно опрашивает `GA_BASE` и возвращает поле **`ga`** (как ответ `/health` predict_ga).

Страница UI: **`/orch-ga`** в web-приложении (`DC_digital_twin/web`).

## Масштаб прогноза нагрузки → `servers_power_total`

Почасовые значения из `forecast` (`yhat_mean`) приводятся к ваттам так: якорь — текущая суммарная мощность стойки `rack.total_power` (Вт); ряд прогноза масштабируется относительно среднего по первым 6 точкам, чтобы сохранить форму кривой и опереться на текущий уровень мощности.

## Поведение при недоступности ML

По умолчанию (`failOnMlUnavailable: false`) оркестратор **не падает**: пропускает рекомендацию PUE или предсказание температуры и продолжает прогон; в ответе `meta.mlFallbackUsed` может быть `true`. При `failOnMlUnavailable: true` вернётся ошибка клиента ML.

## Temp-aware PUE (приоритет температуры чипа)

Сервис PUE выбирает сдвиг уставки `delta_c` по прогнозу энергии охлаждения. Чтобы не ослаблять охлаждение при перегреве и не усиливать его при избытке холода, после ответа ML применяется постобработка по средней температуре чипа `avg_chip_temp` из телеметрии двойника:

| Параметр `POST /run` | По умолчанию | Смысл |
|---------------------|--------------|--------|
| `tempAwarePue` | `true` | Включить гейт по температуре |
| `chipTempTargetC` | `62` | Целевая температура чипа (°C) |
| `chipTempDeadbandC` | `3` | Половина коридора: при `t > target + deadband` запрещены положительные `delta_c` (не поднимать уставку); при `t < target - deadband` запрещены отрицательные (не опускать уставку). В коридоре `delta_c` от ML не меняется. |

В ответе: `meta.pueDeltaRaw` — рекомендация ML, `meta.pueDeltaApplied` / `meta.pueDeltaRecommended` — после гейта (фактически применённый сдвиг). В точках `points[]` есть поля `pueDeltaRaw` и `pueDeltaApplied` на тиках управления.

Разгон вентиляторов по `p_overheat` (`safetyMaxPOverheat`, `fanBoostSpeed`) не отключается.

## Порядок запуска

1. **DC Digital Twin** API (порт 8000).
2. Сервисы ML (8010, 8011, 8002) — по необходимости; без них возможен fallback.
3. **Оркестратор** (8030).
4. Web (`npm run dev` в `DC_digital_twin/web`), страница `/orch`. В dev по умолчанию UI ходит на **прокси** ` /orchestrator` → `127.0.0.1:8030` (см. `DC_digital_twin/web/vite.config.js`), чтобы не ловить CORS. Если в `.env` задан `VITE_ORCHESTRATOR_BASE`, запросы идут напрямую на этот URL — оркестратор должен быть доступен с этого origin (см. CORS в `app.py`).

## Запуск локально

Из корня репозитория (`v2`):

```bash
pip install -r orchestrator/requirements.txt
PYTHONPATH=. uvicorn orchestrator.app:app --host 0.0.0.0 --port 8030
```

Или:

```bash
PYTHONPATH=. python3 -m orchestrator
```

## Docker

```bash
cd orchestrator
docker compose up --build
```

## API

- `GET /health` — доступность двойника, опрос ML (`pue`, `temp`, `load`) и **predict_ga** (`ga`).
- `POST /run` — тело см. `RunRequest` в `app.py` (сценарий как на главной странице + `controlIntervalSteps`, `safetyMaxPOverheat`, и т.д.).
- `POST /run/ga` — тело см. `RunGaRequest` в `app.py` (сценарий + `realism` + `controlIntervalSteps`, опционально `seed`, `failOnGaUnavailable`).

## Смоук-тест

```bash
./scripts/smoke_run.sh
```

(требуется запущенный twin; ML опционально.)

## Несколько прогонов и метрики

При запущенных twin + оркестратор + ML:

```bash
# из корня репозитория v2:
python3 orchestrator/scripts/benchmark_ml_runs.py
```

Печатает 5 сценариев (разный `controlIntervalSteps`, `safetyMaxPOverheat`, `seed`, длина прогона) и сводку по PUE, температуре, риску, уставке/вентилятору.

В корне репозитория: **[`twin_and_ml_runs.ipynb`](../twin_and_ml_runs.ipynb)** — Jupyter-ноутбук: прямой прогон двойника и прогон через оркестратор с подробными графиками и сравнением.

## Тесты

```bash
cd /path/to/v2
PYTHONPATH=. python3 -m unittest orchestrator.tests.test_temp_policy
```

## Устранение: `[404] Not Found` на `POST /run/ga` в UI

Оркестратор нужно **перезапустить** после обновления кода (старый процесс не содержит маршрута `/run/ga`). Проверка:

```bash
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8030/run/ga -H "Content-Type: application/json" -d '{}'
```

Ожидается **422** (нет поля `scenario`), а не **404**. Сервис **predict_ga** должен слушать `GA_BASE` (по умолчанию порт **8013**); локально: `PYTHONPATH=. uvicorn models.predict_ga.app:app --host 127.0.0.1 --port 8013`.
