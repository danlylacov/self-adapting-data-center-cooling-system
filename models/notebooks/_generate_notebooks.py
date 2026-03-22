#!/usr/bin/env python3
"""One-off generator for ipynb files (run from repo root optional)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _find_repo_root_lines() -> list:
    """Locate v2 root (folder that contains models/predict_load/artifacts/load_meta.json)."""
    return [
        "def _find_repo_root() -> Path:\n",
        "    p = Path.cwd().resolve()\n",
        "    for _ in range(12):\n",
        "        marker = p / 'models' / 'predict_load' / 'artifacts' / 'load_meta.json'\n",
        "        if marker.is_file():\n",
        "            return p\n",
        "        if p.parent == p:\n",
        "            break\n",
        "        p = p.parent\n",
        "    raise RuntimeError('Не найден корень v2 (ожидается models/predict_load/artifacts/load_meta.json)')\n\n",
        "REPO_ROOT = _find_repo_root()\n",
    ]


def nb(cells, title: str) -> dict:
    out = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": [],
    }
    for c in cells:
        if c["t"] == "md":
            out["cells"].append({"cell_type": "markdown", "metadata": {}, "source": c["s"]})
        else:
            out["cells"].append(
                {
                    "cell_type": "code",
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": c["s"],
                }
            )
    return out


def write(name: str, data: dict) -> None:
    p = Path(__file__).parent / name
    p.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", p)


# --- 01 Prophet ---
write(
    "01_prophet_load_forecast.ipynb",
    nb(
        [
            {
                "t": "md",
                "s": [
                    "# Прогноз нагрузки: Prophet\n\n",
                    "Ноутбук демонстрирует **инференс** сохранённой модели Prophet и визуализацию истории/прогноза.\n\n",
                    "**Запуск:** из корня репозитория `v2` задайте `PYTHONPATH=.` и установите зависимости из `models/predict_load/requirements_model_load.txt` (в т.ч. `prophet`, `pandas`).\n",
                ],
            },
            {
                "t": "code",
                "s": [
                    "import sys\n",
                    "from pathlib import Path\n\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "import pandas as pd\n\n",
                    *_find_repo_root_lines(),
                    "ART = REPO_ROOT / 'models' / 'predict_load' / 'artifacts'\n",
                    "OUT = REPO_ROOT / 'models' / 'notebooks' / '_outputs'\n",
                    "OUT.mkdir(parents=True, exist_ok=True)\n",
                    "sys.path.insert(0, str(REPO_ROOT))\n\n",
                    "from models.predict_load.predict_load import predict_with_prophet\n",
                    "from models.predict_load.utils import load_json\n\n",
                    "print('REPO_ROOT =', REPO_ROOT)\n",
                ],
            },
            {
                "t": "md",
                "s": ["## Загрузка артефактов и прогноз на 48 ч\n"],
            },
            {
                "t": "code",
                "s": [
                    "load_meta = load_json(ART / 'load_meta.json')\n",
                    "prophet_meta = load_json(ART / 'prophet_meta_prophet_full48.json')\n",
                    "model_path = ART / 'prophet_model_prophet_full48.pkl'\n",
                    "load_csv = ART / 'load_hourly.csv'\n\n",
                    "forecast_df, components_df, peaks = predict_with_prophet(\n",
                    "    prophet_model_path=model_path,\n",
                    "    load_csv_path=load_csv,\n",
                    "    load_meta=load_meta,\n",
                    "    prophet_meta=prophet_meta,\n",
                    "    run_id='notebook_prophet',\n",
                    "    horizon_hours=48,\n",
                    "    history_hours=24 * 14,\n",
                    "    out_dir=OUT,\n",
                    ")\n",
                    "peaks\n",
                ],
            },
            {
                "t": "md",
                "s": ["## Визуализация: история и прогноз `yhat_mean`\n"],
            },
            {
                "t": "code",
                "s": [
                    "hist = pd.read_csv(load_csv, parse_dates=['ds']).tail(24 * 14)\n",
                    "fc = forecast_df.copy()\n",
                    "fc['ds'] = pd.to_datetime(fc['ds'])\n\n",
                    "fig, ax = plt.subplots(figsize=(12, 4))\n",
                    "ax.plot(hist['ds'], hist['y_sum'], label='История y_sum', color='C0')\n",
                    "ax.plot(fc['ds'], fc['yhat_mean'], label='Прогноз yhat_mean', color='C1')\n",
                    "if 'yhat_lower' in fc.columns and fc['yhat_lower'].notna().any():\n",
                    "    ax.fill_between(fc['ds'], fc['yhat_lower'], fc['yhat_upper'], alpha=0.2, color='C1', label='Интервал')\n",
                    "ax.axvline(hist['ds'].iloc[-1], color='k', ls='--', alpha=0.5, label='Конец истории')\n",
                    "ax.set_title('Prophet: история и прогноз нагрузки')\n",
                    "ax.legend()\n",
                    "ax.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "**Что на графике:** по оси X — время (часовой ряд `ds`). **Синяя** кривая — фактическая суммарная нагрузка дата-центра **`y_sum`** за последние 14 суток. **Оранжевая** — прогноз Prophet, точечная оценка **`yhat_mean`** на выбранный горизонт (здесь 48 ч). **Вертикальная пунктирная линия** — последний момент истории: правее неё только прогноз. Если видна **полупрозрачная заливка**, это интервал **`yhat_lower` … `yhat_upper`** (неопределённость модели).\n",
                ],
            },
            {
                "t": "md",
                "s": ["## Компоненты (тренд / сезонность), если есть в `components_df`\n"],
            },
            {
                "t": "code",
                "s": [
                    "comp_cols = [c for c in components_df.columns if c not in ('ds',)]\n",
                    "if comp_cols:\n",
                    "    components_df['ds'] = pd.to_datetime(components_df['ds'])\n",
                    "    n = min(4, len(comp_cols))\n",
                    "    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)\n",
                    "    if n == 1:\n",
                    "        axes = [axes]\n",
                    "    for ax, col in zip(axes, comp_cols[:n]):\n",
                    "        ax.plot(components_df['ds'], components_df[col])\n",
                    "        ax.set_ylabel(col)\n",
                    "        ax.grid(True, alpha=0.3)\n",
                    "    plt.suptitle('Компоненты Prophet (фрагмент)')\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print('Нет числовых компонент для отрисовки')\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "**Расшифровка:** Prophet разлагает ряд на **тренд** и **сезонности** (суточная `daily`, недельная `weekly`, годовая `yearly`, если они есть в `components_df`). По вертикали — вклад компоненты в аддитивную модель; по горизонтали — те же даты, что и у прогноза. Это помогает понять, за счёт чего формируется кривая `yhat` (например, суточные пики vs долгий тренд).\n",
                ],
            },
        ],
        "prophet",
    ),
)

# --- 02 DeepAR ---
write(
    "02_deepar_load_forecast.ipynb",
    nb(
        [
            {
                "t": "md",
                "s": [
                    "# Прогноз нагрузки: DeepAR (PyTorch Forecasting)\n\n",
                    "Инференс чекпоинта `deepar_model_deepar_serving.ckpt`. Требуется `pytorch-forecasting`, `lightning`.\n\n",
                    "Горизонт фиксирован метаданными модели (часто 48 ч).\n",
                ],
            },
            {
                "t": "code",
                "s": [
                    "import sys\n",
                    "from pathlib import Path\n\n",
                    "import matplotlib.pyplot as plt\n",
                    "import pandas as pd\n\n",
                    *_find_repo_root_lines(),
                    "ART = REPO_ROOT / 'models' / 'predict_load' / 'artifacts'\n",
                    "OUT = REPO_ROOT / 'models' / 'notebooks' / '_outputs'\n",
                    "OUT.mkdir(parents=True, exist_ok=True)\n",
                    "sys.path.insert(0, str(REPO_ROOT))\n\n",
                    "from models.predict_load.predict_load import predict_with_deepar\n",
                    "from models.predict_load.utils import load_json\n\n",
                    "load_meta = load_json(ART / 'load_meta.json')\n",
                    "deepar_meta = load_json(ART / 'deepar_meta_deepar_serving.json')\n",
                    "horizon = int(deepar_meta.get('horizon_hours', 48))\n\n",
                    "forecast_df, components_df, peaks = predict_with_deepar(\n",
                    "    deepar_model_ckpt=ART / 'deepar_model_deepar_serving.ckpt',\n",
                    "    deepar_dataset_pkl=ART / 'deepar_dataset_deepar_serving.pkl',\n",
                    "    load_csv_path=ART / 'load_hourly.csv',\n",
                    "    load_meta=load_meta,\n",
                    "    deepar_meta=deepar_meta,\n",
                    "    run_id='notebook_deepar',\n",
                    "    horizon_hours=horizon,\n",
                    "    history_hours=24 * 21,\n",
                    "    out_dir=OUT,\n",
                    ")\n",
                    "print(peaks)\n\n",
                    "# Визуализация в той же ячейке — чтобы «Запустить всё» и nbconvert не ловили NameError\n",
                    "load_csv = ART / 'load_hourly.csv'\n",
                    "hist = pd.read_csv(load_csv, parse_dates=['ds']).tail(24 * 21)\n",
                    "fc = forecast_df.copy()\n",
                    "fc['ds'] = pd.to_datetime(fc['ds'])\n\n",
                    "fig, ax = plt.subplots(figsize=(12, 4))\n",
                    "ax.plot(hist['ds'], hist['y_sum'], label='История', color='C0')\n",
                    "ax.plot(fc['ds'], fc['yhat_mean'], label='DeepAR yhat_mean', color='C2')\n",
                    "ax.axvline(hist['ds'].iloc[-1], color='k', ls='--', alpha=0.5)\n",
                    "ax.set_title('DeepAR: прогноз нагрузки')\n",
                    "ax.legend()\n",
                    "ax.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "**Описание графика:** **Синяя** линия — фактическая нагрузка **`y_sum`** (последние 21 суток, как в ноутбуке). **Зелёная** — прогноз DeepAR, среднее **`yhat_mean`** на горизонт **`horizon_hours`** из `deepar_meta` (часто 48 ч). **Серая вертикаль** — момент «сейчас»: слева только история, справа — модельная оценка будущего. DeepAR учитывает лаги и ковариаты, заданные при обучении; сравнение с Prophet из ноутбука 01 — по тому же `load_hourly.csv`, но другая архитектура.\n",
                ],
            },
        ],
        "deepar",
    ),
)

# --- 03 PUE ---
write(
    "03_pue_hybrid_residual.ipynb",
    nb(
        [
            {
                "t": "md",
                "s": [
                    "# Гибридный PUE: физика + LSTM по остатку\n\n",
                    "1. Базовая **физическая** модель `pue_physics`.\n",
                    "2. Загрузка датасета `pue_dataset.npz` и весов `pue_residual_predictor.pt`.\n",
                    "3. Сравнение предсказанного остатка с целевым `y`, метрики MAE/RMSE.\n",
                    "4. Короткое **дообучение** (несколько эпох) с кривой loss — опционально, для демонстрации «обучения».\n",
                ],
            },
            {
                "t": "code",
                "s": [
                    "import json\n",
                    "import sys\n",
                    "from pathlib import Path\n\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "import torch\n",
                    "import torch.nn as nn\n\n",
                    *_find_repo_root_lines(),
                    "sys.path.insert(0, str(REPO_ROOT))\n\n",
                    "from models.predict_pue.physics_pue import pue_physics\n",
                    "from models.predict_pue.lstm_pue_residual import ModelConfig, PueResidualPredictorLSTM\n",
                ],
            },
            {
                "t": "md",
                "s": ["## 1) Пример физического PUE (скаляр и массивы)\n"],
            },
            {
                "t": "code",
                "s": [
                    "cop_curve = [0.002, -0.15, 4.0]\n",
                    "pue, p_cool = pue_physics(\n",
                    "    servers_power=120_000.0,\n",
                    "    return_temperature=32.0,\n",
                    "    setpoint=22.0,\n",
                    "    fan_speed=0.65,\n",
                    "    outside_temperature=24.0,\n",
                    "    cop_curve=cop_curve,\n",
                    "    capacity=30_000.0,\n",
                    "    fan_max_power=2000.0,\n",
                    "    fan_law='cubic',\n",
                    ")\n",
                    "print('PUE_physics =', float(pue), 'P_cooling_W ≈', float(p_cool))\n",
                ],
            },
            {
                "t": "md",
                "s": ["## 2) Датасет и оценка обученной LSTM\n"],
            },
            {
                "t": "code",
                "s": [
                    "PUE_DIR = REPO_ROOT / 'models' / 'predict_pue'\n",
                    "ds = np.load(PUE_DIR / 'pue_dataset.npz', allow_pickle=True)\n",
                    "X = ds['X']  # [N, T, F]\n",
                    "y = ds['y']  # residual targets [N, H]\n",
                    "meta_path = PUE_DIR / 'pue_residual_meta.json'\n",
                    "meta = json.loads(meta_path.read_text(encoding='utf-8'))\n",
                    "X_mean = np.array(meta['X_mean'], dtype=np.float32).reshape(1, 1, -1)\n",
                    "X_std = np.array(meta['X_std'], dtype=np.float32).reshape(1, 1, -1)\n",
                    "cfg = ModelConfig(**meta['model_config'])\n",
                    "model = PueResidualPredictorLSTM(cfg)\n",
                    "model.load_state_dict(torch.load(PUE_DIR / 'pue_residual_predictor.pt', map_location='cpu'))\n",
                    "model.eval()\n\n",
                    "Xn = (X.astype(np.float32) - X_mean) / X_std\n",
                    "with torch.no_grad():\n",
                    "    mean_pred, _ = model(torch.from_numpy(Xn))\n",
                    "mean_pred = mean_pred.numpy()\n",
                    "err = mean_pred - y\n",
                    "mae = np.mean(np.abs(err))\n",
                    "rmse = np.sqrt(np.mean(err ** 2))\n",
                    "print(f'MAE={mae:.5f} RMSE={rmse:.5f} (на всём датасете)')\n",
                ],
            },
            {
                "t": "code",
                "s": [
                    "# Распределение ошибки по последнему шагу горизонта\n",
                    "h = -1\n",
                    "plt.figure(figsize=(8, 3))\n",
                    "plt.hist(err[:, h], bins=40, alpha=0.85)\n",
                    "plt.title(f'Ошибка residual (горизонт idx={h})')\n",
                    "plt.xlabel('pred - true')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "**Гистограмма ошибки:** по оси X — разность **`pred − true`** для **остаточного PUE** (то, что предсказывает LSTM поверх физической базы). Последний индекс горизонта **`h = -1`** — финальный шаг окна. Узкий пик около нуля означает малую ошибку; смещение влево/вправо — систематическое занижение или завышение остатка.\n",
                ],
            },
            {
                "t": "code",
                "s": [
                    "plt.figure(figsize=(5, 5))\n",
                    "plt.scatter(y[:, h], mean_pred[:, h], s=3, alpha=0.4)\n",
                    "lims = [y[:, h].min(), y[:, h].max()]\n",
                    "plt.plot(lims, lims, 'r--', label='идеал')\n",
                    "plt.xlabel('true residual')\n",
                    "plt.ylabel('pred residual')\n",
                    "plt.legend()\n",
                    "plt.title('Предсказание vs истина (последний час горизонта)')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "**Диаграмма рассеяния:** каждая точка — одно окно из датасета. По **оси X** — истинный остаток **`y`**, по **Y** — предсказание **`mean_pred`**. Красная пунктирная линия **`y = x`** — идеальное совпадение. Точки вдоль диагонали означают хорошую калибровку; облако выше/ниже диагонали — смещение модели на этом горизонте.\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "## 3) Мини-обучение (несколько эпох) — кривая loss\n\n",
                    "Берём подвыборку и упрощённый MSE по остатку. Полное обучение см. в проекте отдельным скриптом (если добавлен).\n",
                ],
            },
            {
                "t": "code",
                "s": [
                    "torch.manual_seed(0)\n",
                    "n = min(256, X.shape[0])\n",
                    "idx = np.random.choice(X.shape[0], size=n, replace=False)\n",
                    "Xb = torch.from_numpy(Xn[idx])\n",
                    "yb = torch.from_numpy(y[idx].astype(np.float32))\n",
                    "m = PueResidualPredictorLSTM(cfg)\n",
                    "opt = torch.optim.Adam(m.parameters(), lr=1e-3)\n",
                    "losses = []\n",
                    "for ep in range(1, 16):\n",
                    "    m.train()\n",
                    "    pred, _ = m(Xb)\n",
                    "    loss = torch.mean((pred - yb) ** 2)\n",
                    "    opt.zero_grad()\n",
                    "    loss.backward()\n",
                    "    opt.step()\n",
                    "    losses.append(loss.item())\n",
                    "plt.figure(figsize=(8, 3))\n",
                    "plt.plot(losses, marker='o')\n",
                    "plt.xlabel('epoch')\n",
                    "plt.ylabel('MSE train (subset)')\n",
                    "plt.title('Демо: сходимость при дообучении с нуля на подвыборке')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "**Кривая loss:** по горизонтали — номер **эпохи** (1…15) на малой подвыборке; по вертикали — **MSE** между предсказанным и истинным остатком. Модель каждый раз инициализируется заново для демонстрации убывания ошибки; это не полноценное обучение на всём датасете, а иллюстрация шага оптимизации.\n",
                ],
            },
        ],
        "pue",
    ),
)

# --- 04 Temp ---
write(
    "04_server_temperature_lstm.ipynb",
    nb(
        [
            {
                "t": "md",
                "s": [
                    "# LSTM: температура серверов и p_overheat\n\n",
                    "Загрузка весов `temp_predictor_mdot002_75_300s.pt` и метаданных. Если локального `temp_dataset.npz` нет, генерируется **синтетический** вход для демонстрации прохода сети.\n",
                ],
            },
            {
                "t": "code",
                "s": [
                    "import json\n",
                    "import sys\n",
                    "from pathlib import Path\n\n",
                    "import matplotlib.pyplot as plt\n",
                    "import numpy as np\n",
                    "import torch\n\n",
                    *_find_repo_root_lines(),
                    "sys.path.insert(0, str(REPO_ROOT))\n\n",
                    "from models.predict_server_temp.lstm_temp_predictor import ModelConfig, TempPredictorLSTM\n\n",
                    "TDIR = REPO_ROOT / 'models' / 'predict_server_temp'\n",
                    "meta = json.loads((TDIR / 'temp_predictor_mdot002_75_300s_meta.json').read_text(encoding='utf-8'))\n",
                    "fc = int(meta['feature_count'])\n",
                    "ih = int(meta['input_hours'])\n",
                    "hh = int(meta['horizon_hours'])\n",
                    "cfg = ModelConfig(input_size=fc, horizon_hours=hh)\n",
                    "model = TempPredictorLSTM(cfg)\n",
                    "model.load_state_dict(torch.load(TDIR / 'temp_predictor_mdot002_75_300s.pt', map_location='cpu'))\n",
                    "model.eval()\n",
                    "X_mean = np.array(meta['X_mean'], dtype=np.float32).reshape(1, 1, -1)\n",
                    "X_std = np.array(meta['X_std'], dtype=np.float32).reshape(1, 1, -1)\n",
                    "print('input_hours', ih, 'horizon', hh, 'features', fc)\n",
                ],
            },
            {
                "t": "md",
                "s": ["## Синтетический батч: 4 сервера\n"],
            },
            {
                "t": "code",
                "s": [
                    "B, T, F = 4, ih, fc\n",
                    "X_raw = np.random.randn(B, T, F).astype(np.float32) * 0.5\n",
                    "Xn = (X_raw - X_mean) / X_std\n",
                    "xb = torch.from_numpy(Xn)\n",
                    "with torch.no_grad():\n",
                    "    mean, std, p_over, _ = model(xb)\n\n",
                    "fig, axes = plt.subplots(1, 3, figsize=(12, 3))\n",
                    "for ax, arr, title in zip(axes, [mean, std, p_over], ['mean t_chip', 'std', 'p_overheat']):\n",
                    "    im = ax.imshow(arr.numpy(), aspect='auto', cmap='viridis')\n",
                    "    ax.set_title(title)\n",
                    "    ax.set_xlabel('horizon')\n",
                    "    ax.set_ylabel('server')\n",
                    "    plt.colorbar(im, ax=ax)\n",
                    "plt.suptitle('Выходы LSTM (синтетический вход)')\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "**Три панели (синтетический вход):** **mean** — предсказанная температура чипа по каждому **серверу** (строка) и **шагу горизонта** (столбец); цвет по шкале справа. **std** — оценка разброса/неопределённости на выходе головы. **p_overheat** — вероятность события перегрева в [0, 1] (как настроено в метаданных модели). Данные **случайные**, график показывает только форму тензоров после прохода сети.\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "## Если есть `temp_dataset.npz` — оценка на реальных окнах\n\n",
                    "Положите файл рядом с весами или сгенерируйте `prepare_temp_dataset.py`.\n",
                ],
            },
            {
                "t": "code",
                "s": [
                    "npz_path = TDIR / 'temp_dataset.npz'\n",
                    "if npz_path.exists():\n",
                    "    d = np.load(npz_path, allow_pickle=True)\n",
                    "    X = d['X'][:64]\n",
                    "    y = d['y_mean'][:64]\n",
                    "    Xn = (X.astype(np.float32) - X_mean) / X_std\n",
                    "    with torch.no_grad():\n",
                    "        pred, _, _, _ = model(torch.from_numpy(Xn))\n",
                    "    pred = pred.numpy()\n",
                    "    mae = np.mean(np.abs(pred - y))\n",
                    "    print(f'MAE t_chip (первые 64 окна): {mae:.4f}')\n",
                    "    plt.figure(figsize=(8, 3))\n",
                    "    plt.plot(y[:, 0], label='true t_chip h+1', alpha=0.8)\n",
                    "    plt.plot(pred[:, 0], label='pred', alpha=0.8)\n",
                    "    plt.legend()\n",
                    "    plt.title('Первый шаг горизонта: сравнение (если npz есть)')\n",
                    "    plt.grid(True, alpha=0.3)\n",
                    "    plt.tight_layout()\n",
                    "    plt.show()\n",
                    "else:\n",
                    "    print('temp_dataset.npz не найден — пропуск оценки на данных.')\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "**Если график построился:** по оси X — номер окна (последовательность первых 64 примеров), по Y — **`t_chip`** на **первом шаге горизонта** (через один час вперёд от конца окна, в зависимости от разметки датасета). Сравниваются **истина** (`y`) и **предсказание** LSTM. **Если файла нет** — сообщение в выводе ячейки; тогда смотрите только блок с синтетическими теплокартами выше.\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "## Демо «обучения»: несколько шагов на случайных данных\n\n",
                    "Упрощённый цикл для визуализации убывания loss (как в `train_temp_predictor.py`).\n",
                ],
            },
            {
                "t": "code",
                "s": [
                    "torch.manual_seed(1)\n",
                    "m2 = TempPredictorLSTM(ModelConfig(input_size=fc, horizon_hours=hh))\n",
                    "opt = torch.optim.Adam(m2.parameters(), lr=1e-3)\n",
                    "yb = torch.randn(16, hh) * 5 + 60\n",
                    "xb = torch.randn(16, ih, fc)\n",
                    "losses = []\n",
                    "for _ in range(30):\n",
                    "    m2.train()\n",
                    "    mean_p, std_p, p_o, _ = m2(xb)\n",
                    "    loss = torch.mean((mean_p - yb) ** 2) + 0.1 * torch.mean((p_o - 0.1) ** 2)\n",
                    "    opt.zero_grad()\n",
                    "    loss.backward()\n",
                    "    opt.step()\n",
                    "    losses.append(loss.item())\n",
                    "plt.figure(figsize=(8, 3))\n",
                    "plt.plot(losses)\n",
                    "plt.xlabel('step')\n",
                    "plt.ylabel('loss')\n",
                    "plt.title('Случайные данные: сходимость упрощённого loss')\n",
                    "plt.grid(True, alpha=0.3)\n",
                    "plt.tight_layout()\n",
                    "plt.show()\n",
                ],
            },
            {
                "t": "md",
                "s": [
                    "**Кривая loss:** по X — шаг градиентного обновления (0…29), по Y — скалярный **loss** (MSE по средней температуре плюс штраф по `p_overheat`). Входы и цели **случайные**, веса — новая модель: цель ячейки показать, что градиенты считаются и ошибка может убывать, а не воспроизвести качество на реальных данных.\n",
                ],
            },
        ],
        "temp",
    ),
)

print('done')
