<div align="center">

# Цифровой двойник ЦОД и ML-оркестрация

**Физико-ориентированный симулятор серверной стойки, REST API, веб-интерфейс и контур машинного обучения для прогноза нагрузки, PUE, температуры серверов и замкнутого управления охлаждением.**

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](https://react.dev/)
[![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white)](https://vitejs.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)
[![Nginx](https://img.shields.io/badge/Nginx-009639?style=flat&logo=nginx&logoColor=white)](https://nginx.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)
[![Recharts](https://img.shields.io/badge/Recharts-8884d8?style=flat)](https://recharts.org/)

</div>

---

## Описание проекта

Репозиторий объединяет:

- **Цифровой двойник** — дискретная тепловая модель стойки, серверов, помещения и CRAC; шаг симуляции по API или из UI.
- **ML-сервисы** — отдельные HTTP-сервисы для прогноза нагрузки (Prophet / DeepAR), гибридного PUE и прогноза температуры серверов (LSTM), а также сервис **генетической политики** охлаждения (GA).
- **Оркестратор** — сценарные прогоны с периодическим управлением уставками CRAC по данным ML и правилам безопасности (в т.ч. режимы ML-прогона и GA-прогона).
- **Веб-приложение** — страницы быстрого прогона, полной конфигурации, ML- и GA-прогонов с графиками.

Подробная математическая формализация: [`DC_digital_twin/docs/mathematical_model.tex`](DC_digital_twin/docs/mathematical_model.tex). Блок-схема двойника в LaTeX: [`DC_digital_twin/docs/twin_block_diagram.tex`](DC_digital_twin/docs/twin_block_diagram.tex).

---

## Архитектура

На верхнем уровне данные идут **от сценария и конфигурации** в **симулятор (двойник)**; **оркестратор** на шагах управления вызывает **ML-сервисы** и задаёт уставки охлаждения; **веб-клиент** обращается к API двойника и к API оркестратора.

```mermaid
flowchart LR
  subgraph client["Клиент"]
    UI["Веб-UI\n(React + Vite)"]
  end

  subgraph twin_svc["Двойник"]
    API["FastAPI\n:8000"]
    SIM["Симулятор ЦОД"]
    API --> SIM
  end

  subgraph ml["ML / политика"]
    PL["predict_load\n:8010"]
    PP["predict_pue\n:8011"]
    PT["predict_server_temp\n:8002"]
    GA["predict_ga\n:8013"]
  end

  subgraph orch_svc["Оркестратор"]
    ORCH["FastAPI\n:8030"]
  end

  UI -->|REST| API
  UI -->|REST| ORCH
  ORCH -->|шаги симуляции| API
  ORCH --> PL
  ORCH --> PP
  ORCH --> PT
  ORCH --> GA
```

В **Docker Compose** все сервисы подключены к одной сети; оркестратор обращается к двойнику и ML по внутренним именам (`twin:8000`, `predict-load:8010`, …). Сборка фронта задаёт URL API через переменные `VITE_*` (куда ходит **браузер** пользователя).

---

## Сервисы

| Сервис | Порт (по умолчанию) | Назначение |
|--------|---------------------|------------|
| **twin** (`DC_digital_twin`) | **8000** | FastAPI: симуляция, конфиг `/config`, охлаждение, нагрузка, датасеты, телеметрия. |
| **predict-load** | **8010** | Прогноз временного ряда нагрузки (Prophet / DeepAR). |
| **predict-pue** | **8011** | Гибридная модель PUE (физика + нейросеть). |
| **predict-server-temp** | **8002** | Прогноз температуры серверов и риска перегрева (LSTM). |
| **predict-ga** | **8013** | Политика охлаждения на основе GA (`GA/`, `tuned_params.json`). |
| **orchestrator** | **8030** | Сценарные прогоны с вызовами ML и управлением уставками; `POST /run`, `POST /run/ga`. |
| **web** | **8080** → 80 в контейнере | Статическая сборка UI + nginx. |

Дополнительно в репозитории: каталог `GA/` (обучение и параметры политики), `models/rl/` (офлайн RL, по желанию), скрипты в `scripts/`.

---

## Запуск

### Вариант 1: Docker Compose (рекомендуется)

Из **корня репозитория**:

```bash
docker compose up --build -d
```

После старта:

- Веб-интерфейс: **http://127.0.0.1:8080**
- API двойника: **http://127.0.0.1:8000** (`GET /health`)
- Оркестратор: **http://127.0.0.1:8030** (`GET /health`)

Если UI открывается с другого хоста или порта, пересоберите фронт с нужными URL:

```bash
VITE_API_BASE=http://<хост>:8000 VITE_ORCHESTRATOR_BASE=http://<хост>:8030 docker compose build web
docker compose up -d web
```

Остановка: `docker compose down`.

---

### Вариант 2: Локальная разработка без Docker

1. **Виртуальное окружение Python** (в корне или в `DC_digital_twin`):

   ```bash
   cd DC_digital_twin
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   python run_api.py
   ```

   API слушает хост/порт из переменных `API_HOST` / `API_PORT` (по умолчанию `0.0.0.0:8000`).

2. **ML-сервисы и оркестратор** — отдельные терминалы, из корня репозитория с `PYTHONPATH=.`:

   ```bash
   uvicorn models.predict_load.api:app --host 127.0.0.1 --port 8010
   uvicorn models.predict_pue.api:app --host 127.0.0.1 --port 8011
   uvicorn models.predict_server_temp.api_fastapi:app --host 127.0.0.1 --port 8002
   PYTHONPATH=. uvicorn models.predict_ga.app:app --host 127.0.0.1 --port 8013
   PYTHONPATH=. uvicorn orchestrator.app:app --host 0.0.0.0 --port 8030
   ```

   Переменные окружения оркестратора по умолчанию указывают на `127.0.0.1` (см. `orchestrator/config.py`).

3. **Фронтенд**:

   ```bash
   cd DC_digital_twin/web
   npm install
   cp .env.example .env   # при необходимости поправьте VITE_API_BASE
   npm run dev
   ```

   В dev для страниц оркестратора Vite проксирует `/orchestrator` на `127.0.0.1:8030` (см. `vite.config.js`).

---

## Структура каталогов (кратко)

| Каталог | Содержимое |
|---------|------------|
| `DC_digital_twin/` | Симулятор, FastAPI, веб (`web/`) |
| `models/` | `predict_load`, `predict_pue`, `predict_server_temp`, `predict_ga`, ноутбуки-шаблоны в `notebooks/` |
| `orchestrator/` | Оркестратор прогонов |
| `GA/` | GA-ядро, обучение, `tuned_params.json` |
| `docker-compose.yml` | Единый compose всего стека |

---

## Лицензия и заметки

Используйте конфигурацию и эндпоинты в соответствии с политикой вашей среды. При публикации репозитория не коммитьте секреты и локальные `.env`.
