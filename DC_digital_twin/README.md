# Self-Adapting Data Center Cooling System

Цифровой двойник стойки ЦОД с адаптивным охлаждением. Симулятор моделирует тепловые процессы в серверной стойке, помещении и системе охлаждения (CRAC) с учётом законов физики.

## Возможности

- **Тепловая модель серверов** — чипы с тепловой инерцией, теплообмен воздухом (epsilon heat exchanger)
- **Модель стойки** — рециркуляция воздуха (hot/cold aisle), температура на входе/выходе
- **Модель помещения** — энергетический баланс, теплообмен через стены
- **CRAC** — мощность охлаждения, COP от температуры, setpoint, скорость вентиляторов
- **Режимы симуляции** — realtime (с масштабированием времени), fast (ускоренно), interactive (пошагово)
- **REST API** — FastAPI-обёртка для удалённого управления и интеграции

## Установка

```bash
pip install -r requirements.txt
```

UI (React + Vite):

```bash
cd web
npm install
cp .env.example .env   # опционально: VITE_API_BASE=http://127.0.0.1:8000
```

Переменная **`VITE_API_BASE`** (файл `web/.env`) задаёт URL FastAPI без завершающего слэша; по умолчанию `http://127.0.0.1:8000`. После изменения `.env` перезапусти `npm run dev`.

## Конфигурация

Конфиг **не хранится в `config.yaml`**. Значения по умолчанию — в `src/default_config.py`.

| Способ | Описание |
|--------|----------|
| **Web UI** | Вкладка **«Конфигурация»** (`/config`) — полная форма или редактирование JSON, кнопка «Сохранить на сервер» |
| **REST API** | `GET /config` — текущий конфиг, `PUT /config` — полная замена (симулятор пересоздаётся), `GET /config/defaults` — шаблон |
| **Переменная** | `CONFIG_PATH` — опциональный путь к YAML при старте API/CLI, если файл есть |

Ключевые секции: `simulator`, `realism`, `rack`, `servers`, `cooling`, `room`, `load_generator`, `mqtt`, `output`, `logging`, опционально `weather`.

### Режим реалистичности

- `realism.use_dynamic_crac_power`: учитывать фактическую `return_temperature`
- `realism.room_temp_clip_min/max`: клип температуры помещения
- `realism.chip_temp_clip_multiplier`: множитель для верхней границы чипа (метаданные/UI)

## Запуск CLI

```bash
# Ускоренный режим (50 шагов)
python main.py --mode fast --steps 50

# Реальное время (3600 сек симуляции)
python main.py --mode realtime --duration 3600

# Интерактивный режим
python main.py --mode interactive
```

Параметры: `--config` (опциональный YAML), `--mode`, `--duration`, `--steps`, `--output`.

### Батч: 100 прогонов и оценка реализма

Скрипт варирует seed нагрузки, профиль сервера, CRAC, зал и среду; после каждого прогона выводит **балл реалистичности** (0–100) и сохраняет таблицу в `results/realism_benchmark.csv`.

```bash
python scripts/benchmark_realism.py              # 100 прогонов × 1800 шагов
python scripts/benchmark_realism.py --runs 20 --steps 900 --seed 42
```

## REST API

Запуск сервера:

```bash
python run_api.py
# или
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Документация: http://localhost:8000/docs

Web UI:

```bash
cd web
npm run dev
```

По умолчанию UI берёт адрес API из `VITE_API_BASE` (см. `web/.env.example`).

### Симуляция

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/simulation/start` | Запуск: `mode` (realtime/fast/interactive), `duration` или `steps` |
| POST | `/simulation/stop` | Остановка |
| POST | `/simulation/step` | Выполнить N шагов |
| GET | `/simulation/status` | Статус: running, step, time |
| POST | `/simulation/reset` | Сброс (опционально `seed`) |
| GET | `/simulation/state` | Полное состояние |
| GET | `/simulation/telemetry` | Телеметрия |
| GET | `/state` | Состояние (корневой алиас) |
| GET | `/telemetry` | Телеметрия (корневой алиас) |

### Полная конфигурация

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/config` | Текущий JSON-конфиг симулятора |
| PUT | `/config` | Полная замена конфига (пересоздание симулятора; не во время прогона) |
| GET | `/config/defaults` | Встроенный конфиг по умолчанию |

### Охлаждение

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/cooling/setpoint` | Температура подачи 18–27°C |
| POST | `/cooling/fanspeed` | Скорость вентиляторов 0–100% |
| POST | `/cooling/mode` | Режим: free / chiller / mixed |
| GET | `/cooling` | Текущее состояние охлаждения |

### Нагрузка и среда

| Метод | Путь | Описание |
|-------|------|----------|
| POST | `/load/mode` | Режим нагрузки: random / periodic / dataset |
| POST | `/load/params` | Параметры random/periodic (`mean_load`, `std_load`, ...) |
| POST | `/load/dataset` | Выбрать датасет нагрузки по пути |
| POST | `/environment/outside` | Ручная установка внешней среды |
| POST | `/environment/weather-mode` | Режим погоды: manual / openmeteo / dataset |
| POST | `/datasets/load/upload` | Загрузка CSV датасета нагрузки |
| POST | `/datasets/weather/upload` | Загрузка CSV погодного датасета |
| GET | `/datasets` | Список доступных CSV датасетов |
| WS | `/ws/telemetry` | Live поток состояния и телеметрии |

### Примеры

```bash
# Запуск fast-симуляции на 100 шагов
curl -X POST http://localhost:8000/simulation/start \
  -H "Content-Type: application/json" \
  -d '{"mode": "fast", "steps": 100}'

# Установить setpoint 20°C
curl -X POST http://localhost:8000/cooling/setpoint \
  -H "Content-Type: application/json" \
  -d '{"temperature": 20}'

# Выполнить 5 шагов
curl -X POST http://localhost:8000/simulation/step \
  -H "Content-Type: application/json" \
  -d '{"steps": 5}'

# Получить телеметрию
curl http://localhost:8000/telemetry
```

## Структура проекта

```
├── api/                    # FastAPI-обёртка
│   ├── main.py             # Приложение, lifespan, CORS
│   ├── deps.py             # DI, SimulatorService
│   ├── schemas.py          # Pydantic-модели
│   └── routers/
│       ├── simulation.py   # Эндпоинты симуляции
│       ├── full_config.py  # GET/PUT полного конфига
│       └── cooling.py      # Эндпоинты охлаждения
├── config/
│   └── README.md           # Как задаётся конфиг (без YAML по умолчанию)
├── src/
│   ├── default_config.py   # Встроенный конфиг по умолчанию
│   ├── core/
│   │   ├── simulator.py    # Главный симулятор
│   │   └── load_generator.py
│   ├── models/
│   │   ├── server.py       # Модель сервера
│   │   ├── rack.py         # Модель стойки
│   │   ├── room.py         # Модель помещения
│   │   └── cooling.py      # CRAC, Chiller
│   ├── output/             # Сохранение результатов
│   └── mqtt/               # Публикация в MQTT
├── main.py                 # CLI-точка входа
├── run_api.py              # Запуск API
└── requirements.txt
```

## Входные данные

### Конфигурация (структура JSON)

| Секция | Поле | Тип | Описание |
|--------|------|-----|----------|
| `simulator` | `time_step` | float | Шаг симуляции (сек) |
| | `realtime_factor` | float | Ускорение в realtime |
| `rack` | `height`, `width`, `depth` | float | Размеры стойки (м) |
| | `num_units` | int | Количество юнитов |
| | `containment` | str | hot_aisle / cold_aisle / none |
| `servers` | `p_idle`, `p_max` | float | Мощность (Вт) |
| | `t_max` | float | Макс. температура чипа (°C) |
| | `c_thermal` | float | Теплоёмкость (Дж/К) |
| | `m_dot` | float | Расход воздуха (кг/с) |
| `cooling` | `capacity` | float | Мощность CRAC (Вт) |
| | `cop_curve` | [a,b,c] | COP(T) = aT² + bT + c |
| | `default_setpoint` | float | Температура подачи (°C) |
| `room` | `volume` | float | Объём (м³) |
| | `wall_heat_transfer` | float | Теплообмен стен (Вт/К) |
| | `initial_temperature` | float | Начальная температура (°C) |
| `load_generator` | `type` | str | random / dataset / periodic |
| | `dataset_path` | str | Путь к CSV (для dataset) |
| | `mean_load`, `std_load` | float | Для random |
| | `random_seed` | int | Seed для воспроизводимости |

### Датасет нагрузки (для type: dataset)

CSV-файл с колонкой `load` (0–1):

```csv
load
0.45
0.62
0.58
...
```

### REST API — входные тела запросов

| Эндпоинт | Тело запроса |
|----------|--------------|
| `POST /simulation/start` | `{"mode": "realtime"\|"fast"\|"interactive", "duration": 3600, "steps": 100}` |
| `POST /simulation/step` | `{"steps": 1, "delta_time": null}` |
| `POST /simulation/reset` | `{"seed": 42}` |
| `POST /cooling/setpoint` | `{"temperature": 20}` (18–27) |
| `POST /cooling/fanspeed` | `{"speed": 80}` (0–100) |
| `POST /cooling/mode` | `{"mode": "free"\|"chiller"\|"mixed"}` |

---

## Выходные данные

### GET /state — полное состояние

```json
{
  "time": 100.0,
  "step": 100,
  "rack": {
    "supply_temperature": 22.0,
    "total_power": 12345.0,
    "avg_exhaust_temp": 28.5,
    "servers": [
      {
        "server_id": 0,
        "utilization": 0.65,
        "t_in": 22.1,
        "t_out": 28.3,
        "t_chip": 45.2,
        "power": 312.5,
        "fan_speed": 0.5
      }
    ]
  },
  "room": {
    "temperature": 24.2,
    "outside_temperature": 20.0,
    "thermal_capacity": 120600.0
  },
  "cooling": {
    "mode": "mixed",
    "crac": {
      "setpoint": 22.0,
      "fan_speed": 0.5,
      "capacity": 30000
    }
  }
}
```

### GET /telemetry — телеметрия (датчики)

```json
{
  "timestamp": 1733456789.123,
  "step": 100,
  "servers": [
    {
      "server_id": 0,
      "utilization": 0.65,
      "t_in": 22.1,
      "t_out": 28.3,
      "t_chip": 45.2,
      "power": 312.5,
      "fan_speed": 0.5
    }
  ],
  "cooling": {
    "supply_temperature": 22.0,
    "power_consumption": 2500.0,
    "setpoint": 22.0,
    "fan_speed": 50.0
  },
  "room": { "temperature": 24.2 },
  "pue": 1.18
}
```

### Файлы результатов (results/)

При `output.enabled: true` сохраняются:

**`sim_YYYYMMDD_HHMMSS_summary.csv`** — сводка по шагам:

| Поле | Тип | Единица | Описание |
|------|-----|---------|----------|
| `step` | int | — | Номер шага симуляции |
| `timestamp` | float | Unix (сек) | Время записи |
| `room_temperature` | float | °C | Температура воздуха в помещении |
| `pue` | float | — | Power Usage Effectiveness: (P_серверы + P_охлаждение) / P_серверы |
| `cooling_setpoint` | float | °C | Целевая температура подачи CRAC (18–27) |
| `cooling_power` | float | Вт | Потребление охлаждения (компрессор + вентиляторы) |
| `cooling_fan_speed` | float | % | Скорость вентиляторов CRAC (0–100) |

**`sim_YYYYMMDD_HHMMSS_servers.csv`** — по каждому серверу на каждом шаге:

| Поле | Тип | Единица | Описание |
|------|-----|---------|----------|
| `step` | int | — | Номер шага симуляции |
| `timestamp` | float | Unix (сек) | Время записи |
| `server_id` | int | — | Идентификатор сервера |
| `utilization` | float | 0–1 | Загрузка (доля от max) |
| `t_in` | float | °C | Температура воздуха на входе |
| `t_out` | float | °C | Температура воздуха на выходе |
| `t_chip` | float | °C | Температура чипа |
| `power` | float | Вт | Потребляемая мощность (тепловыделение) |
| `fan_speed` | float | 0–1 | Скорость вентилятора сервера (авто по температуре) |

Формат задаётся в конфиге: `output.format` — csv / json / parquet.

---

## Физика модели

### Сервер

**Мощность (тепловыделение):**
$$
P = P_{\mathrm{idle}} + (P_{\mathrm{max}} - P_{\mathrm{idle}}) \cdot u
$$
где \(u \in [0, 1]\) — загрузка.

**Теплообменник (epsilon-NTU):** температура выхлопа
$$
T_{\mathrm{out}} = T_{\mathrm{in}} + \varepsilon (T_{\mathrm{chip}} - T_{\mathrm{in}})
$$
где \(\varepsilon \approx 0.8\) — эффективность теплообменника.

**Тепловая инерция чипа:**
$$
C \frac{dT_{\mathrm{chip}}}{dt} = P - \dot{m} \, c_p \, \varepsilon (T_{\mathrm{chip}} - T_{\mathrm{in}})
$$
где \(C\) — теплоёмкость (Дж/К), \(\dot{m}\) — расход воздуха (кг/с), \(c_p ≈ 1005\) Дж/(кг·К).

**Интегрирование:** явный метод Эйлера, \(T_{\mathrm{chip}}^{n+1} = T_{\mathrm{chip}}^n + \Delta t \cdot dT/dt\).

### Стойка (рециркуляция)

Температура на входе сервера \(i\) с учётом смешивания:
$$
T_{\mathrm{in},i} = T_{\mathrm{supply}} + \sum_j R_{ij} (T_{\mathrm{out},j} - T_{\mathrm{supply}})
$$
где \(R_{ij}\) — матрица рециркуляции (доля тепла от сервера \(j\), попадающая на вход \(i\)). Зависит от `containment`: hot_aisle, cold_aisle, none.

### Помещение

Энергетический баланс:
$$
C_{\mathrm{room}} \frac{dT}{dt} = Q_{\mathrm{racks}} - Q_{\mathrm{cooling}} + Q_{\mathrm{wall}}
$$
где \(Q_{\mathrm{wall}} = k_{\mathrm{wall}} (T_{\mathrm{outside}} - T)\) — теплообмен через стены, \(C_{\mathrm{room}} = V \cdot \rho \cdot c_p\) (объём × плотность × теплоёмкость воздуха).

### CRAC (охлаждение)

**Мощность охлаждения:**
$$
Q_{\mathrm{cooling}} = \min\left( C_{\mathrm{max}} \frac{\Delta T}{5}, C_{\mathrm{max}} \right) \cdot f_{\mathrm{fan}}
$$
при \(\Delta T = T_{\mathrm{return}} - T_{\mathrm{setpoint}} > 0\); иначе 0. \(T_{\mathrm{return}}\) — средняя температура выхлопа стойки (горячий коридор), \(f_{\mathrm{fan}} \in [0, 1]\) — скорость вентилятора.

**COP (коэффициент эффективности):**
$$
\mathrm{COP}(T) = a T^2 + b T + c
$$
где \(T\) — внешняя температура. Параметры \([a, b, c]\) в `cop_curve`.

**Потребление охлаждения:**
$$
P_{\mathrm{cooling}} = \frac{Q_{\mathrm{cooling}}}{\mathrm{COP}} + P_{\mathrm{fan,max}} \cdot f_{\mathrm{fan}}^3
$$
(кубический закон для вентиляторов).

### PUE

$$
\mathrm{PUE} = \frac{P_{\mathrm{servers}} + P_{\mathrm{cooling}}}{P_{\mathrm{servers}}}
$$

---

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `CONFIG_PATH` | Опциональный путь к YAML при старте | не задано (встроенный `default_config`) |
| `API_HOST` | Хост API | `0.0.0.0` |
| `API_PORT` | Порт API | `8000` |
