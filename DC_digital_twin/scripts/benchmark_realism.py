#!/usr/bin/env python3
"""
100 (или N) прогонов симулятора с разными параметрами и оценкой «реалистичности» после каждого.

Запуск из корня проекта DC_digital_twin:
  python scripts/benchmark_realism.py
  python scripts/benchmark_realism.py --runs 100 --steps 1800

Результат: results/realism_benchmark.csv и краткая сводка в stdout.
"""
from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.simulator import DataCenterSimulator  # noqa: E402
from src.default_config import get_default_config_copy  # noqa: E402


def load_base_config() -> dict:
    return get_default_config_copy()


def evaluate_realism(states: list[dict], delta_time: float) -> tuple[float, dict[str, float]]:
    """
    Оценка 0..100: насколько траектории похожи на правдоподобное поведение стойки.
    Учитывает диапазоны T_chip, T_in, T_room, PUE, гладкость и отсутствие «ломаных» скачков.
    """
    if len(states) < 10:
        return 0.0, {"reason": 0.0}

    chips: list[float] = []
    inlets: list[float] = []
    rooms: list[float] = []
    pues: list[float] = []

    for s in states:
        rack = s.get("rack") or {}
        for srv in rack.get("servers") or []:
            chips.append(float(srv.get("t_chip", 0.0)))
            inlets.append(float(srv.get("t_in", 0.0)))
        rooms.append(float((s.get("room") or {}).get("temperature", 0.0)))
        pues.append(float((s.get("telemetry") or {}).get("pue_real", 1.0)))

    chips_arr = np.asarray(chips, dtype=np.float64)
    inlets_arr = np.asarray(inlets, dtype=np.float64)
    rooms_arr = np.asarray(rooms, dtype=np.float64)
    pues_arr = np.asarray(pues, dtype=np.float64)

    if not np.all(np.isfinite(chips_arr)):
        return 0.0, {"nan": 1.0}

    # --- Диапазоны (мягкие штрафы) ---
    def band_score(x: np.ndarray, lo: float, hi: float) -> float:
        if x.size == 0:
            return 0.0
        below = np.maximum(0.0, lo - x)
        above = np.maximum(0.0, x - hi)
        penalty = float(np.mean(below + above))
        # ~1°C средний выход за границу -> заметный штраф
        return max(0.0, 100.0 - penalty * 20.0)

    # Диапазон чипа: учитываем простой/низкую нагрузку (20–35°C) и горячий режим
    s_chip = band_score(chips_arr, 20.0, 98.0)
    s_inlet = band_score(inlets_arr, 16.0, 35.0)
    s_room = band_score(rooms_arr, 17.0, 36.0)

    # PUE: типично 1.05–2.2 для малой установки
    pue_mid = np.clip(pues_arr, 1.0, 5.0)
    pue_pen = float(np.mean(np.maximum(0.0, pue_mid - 2.8) + np.maximum(0.0, 1.02 - pue_mid)))
    s_pue = max(0.0, 100.0 - pue_pen * 40.0)

    # Гладкость: макс. скорость изменения T_chip (°C/с)
    dchip = np.abs(np.diff(chips_arr)) / max(delta_time, 1e-6)
    max_rate = float(np.max(dchip)) if dchip.size else 0.0
    # >3 °C/с на чипе за шаг — подозрительно для 1-сек шага
    s_smooth = max(0.0, 100.0 - max(0.0, max_rate - 2.5) * 15.0)

    # Стабильность на хвосте (последние 15% шагов): низкая дисперсия лучше
    tail = max(5, len(chips_arr) // 6)
    tail_chips = chips_arr[-tail:]
    sigma = float(np.std(tail_chips))
    # слишком дребезжащий или слишком плоский после прогрева — слегка штрафуем крайности
    if sigma < 0.05:
        s_stab = 70.0
    elif sigma > 8.0:
        s_stab = max(0.0, 100.0 - (sigma - 8.0) * 5.0)
    else:
        s_stab = 95.0

    weights = {
        "chip_range": 0.28,
        "inlet_range": 0.12,
        "room_range": 0.12,
        "pue": 0.18,
        "smooth": 0.18,
        "stability": 0.12,
    }
    total = (
        weights["chip_range"] * s_chip
        + weights["inlet_range"] * s_inlet
        + weights["room_range"] * s_room
        + weights["pue"] * s_pue
        + weights["smooth"] * s_smooth
        + weights["stability"] * s_stab
    )
    breakdown = {
        "score": round(total, 2),
        "s_chip": round(s_chip, 2),
        "s_inlet": round(s_inlet, 2),
        "s_room": round(s_room, 2),
        "s_pue": round(s_pue, 2),
        "s_smooth": round(s_smooth, 2),
        "s_stab": round(s_stab, 2),
        "max_dchip_dt": round(max_rate, 4),
        "tail_sigma_chip": round(sigma, 4),
    }
    return round(total, 2), breakdown


def randomize_run(cfg: dict, rng: np.random.Generator, _run_id: int) -> dict:
    c = copy.deepcopy(cfg)
    prof = c["servers"]["profiles"][c["servers"]["default_profile"]]

    # Физика сервера: разумный разброс 1–2U класса
    prof["p_idle"] = int(rng.integers(100, 180))
    prof["p_max"] = int(rng.integers(320, 520))
    prof["c_thermal"] = int(rng.integers(5000, 12000))
    prof["m_dot"] = round(float(rng.uniform(0.04, 0.12)), 4)
    prof["t_max"] = 85

    c["load_generator"]["type"] = "random"
    c["load_generator"]["random_seed"] = int(rng.integers(0, 2**31 - 1))
    c["load_generator"]["mean_load"] = round(float(rng.uniform(0.35, 0.82)), 4)
    c["load_generator"]["std_load"] = round(float(rng.uniform(0.06, 0.22)), 4)

    c["room"]["initial_temperature"] = round(float(rng.uniform(20.5, 25.5)), 2)
    c["room"]["thermal_mass_factor"] = round(float(rng.uniform(6.0, 18.0)), 2)
    c["room"]["wall_heat_transfer"] = int(rng.integers(35, 85))

    crac = c["cooling"]["crac"]
    crac["default_setpoint"] = round(float(rng.uniform(18.0, 23.5)), 2)
    crac["capacity"] = int(rng.uniform(0.7, 1.15) * 28000)
    crac["airflow_m_dot"] = round(float(rng.uniform(1.2, 2.8)), 3)

    # Клипы ближе к реальности (без «космоса»)
    c["realism"]["room_temp_clip_min"] = 8.0
    c["realism"]["room_temp_clip_max"] = 48.0
    c["realism"]["chip_temp_clip_multiplier"] = round(float(rng.uniform(1.08, 1.22)), 3)

    c["simulator"]["time_step"] = 1.0
    c["output"]["enabled"] = False
    c["logging"] = {**(c.get("logging") or {}), "level": "ERROR"}

    return c


def run_one(
    cfg: dict,
    steps: int,
    rng: np.random.Generator,
) -> tuple[float, dict, dict]:
    sim = DataCenterSimulator(config=cfg)

    mode = str(rng.choice(["mixed", "chiller", "free"]))
    sim.set_cooling_mode(mode)
    sim.set_weather_mode("manual")
    outside = float(rng.uniform(-5.0, 34.0))
    sim.set_outside_environment(
        temperature=outside,
        humidity=float(rng.uniform(30.0, 70.0)),
        wind_speed=float(rng.uniform(0.0, 12.0)),
    )
    # Параметры, не в YAML
    sim.set_cooling_setpoint(float(cfg["cooling"]["crac"]["default_setpoint"]))
    sim.set_fan_speed(float(rng.uniform(35.0, 95.0)))

    sim.run_fast(steps)
    states = sim.state_history
    dt = float(sim.time_step)
    score, breakdown = evaluate_realism(states, dt)

    meta = {
        "cooling_mode": mode,
        "outside_c": round(outside, 2),
        "fan_speed_pct": round(sim.cooling.crac.fan_speed * 100.0, 1),
        "mean_load": cfg["load_generator"]["mean_load"],
        "std_load": cfg["load_generator"]["std_load"],
        "seed": cfg["load_generator"]["random_seed"],
    }
    return score, breakdown, meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Батч-прогоны с оценкой реалистичности")
    parser.add_argument("--runs", type=int, default=100, help="Число прогонов")
    parser.add_argument("--steps", type=int, default=1800, help="Шагов на прогон")
    parser.add_argument("--seed", type=int, default=2026, help="Seed для воспроизводимости набора сценариев")
    args = parser.parse_args()

    base = load_base_config()
    rng = np.random.default_rng(args.seed)

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "realism_benchmark.csv"

    rows: list[dict] = []
    scores: list[float] = []

    print(f"Прогонов: {args.runs}, шагов: {args.steps}, seed сценариев: {args.seed}\n")

    for i in range(args.runs):
        cfg = randomize_run(base, rng, i)
        score, breakdown, meta = run_one(cfg, args.steps, rng)
        scores.append(score)

        row = {
            "run": i + 1,
            "realism_score": score,
            **{f"m_{k}": v for k, v in meta.items()},
            **{f"b_{k}": v for k, v in breakdown.items() if k != "score"},
        }
        rows.append(row)

        print(
            f"[{i+1:3d}/{args.runs}] реализм: {score:5.1f} | "
            f"T_out={meta['outside_c']:5.1f}°C mode={meta['cooling_mode']:<7} "
            f"load={meta['mean_load']:.2f}±{meta['std_load']:.2f} | "
            f"max dTchip/dt={breakdown.get('max_dchip_dt', 0):.3f}°C/s"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    arr = np.asarray(scores, dtype=np.float64)
    print("\n=== Итог ===")
    print(f"Средний балл реализма: {arr.mean():.2f} (σ={arr.std():.2f})")
    print(f"Min / max: {arr.min():.2f} / {arr.max():.2f}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
