#!/usr/bin/env python3
"""
Множественные прогоны через HTTP API: режимы охлаждения × наружная температура.
Пишет сводку в stdout и опционально CSV.

Использование:
  export API_BASE=http://127.0.0.1:8000
  python3 scripts/benchmark_api_runs.py

Требуется запущенный API (run_api.py / uvicorn).
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from statistics import mean, pstdev
from typing import Any, Dict, List, Optional, Tuple

try:
    import requests
except ImportError:
    print("pip install requests", file=sys.stderr)
    raise

BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT = 120


def post(path: str, payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    r = requests.post(f"{BASE}{path}", json=payload or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def get(path: str) -> Dict[str, Any]:
    r = requests.get(f"{BASE}{path}", timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


@dataclass
class RunResult:
    seed: int
    scenario: str
    temps: List[float] = field(default_factory=list)  # T_room после каждого чанка
    final_room: float = 0.0
    telemetry_last: Optional[Dict[str, Any]] = None

    @property
    def delta_final(self) -> float:
        if len(self.temps) < 2:
            return 0.0
        return self.temps[-1] - self.temps[0]

    @property
    def max_temp(self) -> float:
        return max(self.temps) if self.temps else 0.0

    @property
    def min_temp(self) -> float:
        return min(self.temps) if self.temps else 0.0

    @property
    def strictly_non_decreasing(self) -> bool:
        """Все шаги dT >= 0 (монотонный рост или плато)."""
        if len(self.temps) < 2:
            return True
        return all(self.temps[i + 1] >= self.temps[i] - 1e-9 for i in range(len(self.temps) - 1))

    @property
    def has_cooling_dip(self) -> bool:
        """Было ли снижение T_room (охлаждение реально работает)."""
        if len(self.temps) < 3:
            return False
        return self.min_temp < self.temps[0] - 0.01 or any(
            self.temps[i + 1] < self.temps[i] - 0.01 for i in range(len(self.temps) - 1)
        )


def one_run(
    scenario_name: str,
    cooling_mode: str,
    outside_c: float,
    seed: int,
    chunks: int,
    steps_per_chunk: int,
    delta_time: float,
) -> RunResult:
    post("/simulation/reset", {"seed": seed})
    post("/environment/weather-mode", {"mode": "manual"})
    post("/environment/outside", {"temperature": outside_c, "humidity": 40.0, "wind_speed": 0.0})
    post("/cooling/mode", {"mode": cooling_mode})
    post("/cooling/setpoint", {"temperature": 22.0})
    post("/cooling/fanspeed", {"speed": 65.0})
    post("/load/mode", {"mode": "random"})
    post("/load/params", {"mean_load": 0.55, "std_load": 0.12})

    res = RunResult(seed=seed, scenario=scenario_name)
    for _ in range(chunks):
        st = post(
            "/simulation/step",
            {"steps": steps_per_chunk, "delta_time": delta_time},
        )
        tr = float(st.get("room", {}).get("temperature", float("nan")))
        res.temps.append(tr)
    res.final_room = res.temps[-1] if res.temps else float("nan")
    try:
        res.telemetry_last = get("/telemetry")
    except Exception:
        res.telemetry_last = None
    return res


def summarize_runs(name: str, runs: List[RunResult]) -> Dict[str, Any]:
    finals = [r.final_room for r in runs]
    deltas = [r.delta_final for r in runs]
    mono = sum(1 for r in runs if r.strictly_non_decreasing)
    dips = sum(1 for r in runs if r.has_cooling_dip)
    maxes = [r.max_temp for r in runs]
    return {
        "scenario": name,
        "n": len(runs),
        "final_T_mean": mean(finals),
        "final_T_std": pstdev(finals) if len(finals) > 1 else 0.0,
        "delta_T_mean": mean(deltas),
        "max_T_over_runs_mean": mean(maxes),
        "runs_monotonic_non_decreasing": mono,
        "runs_with_cooling_dip": dips,
        "fraction_monotonic": mono / len(runs) if runs else 0.0,
        "fraction_with_dip": dips / len(runs) if runs else 0.0,
    }


def health_check() -> bool:
    try:
        r = requests.get(f"{BASE}/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def sanity_check_free_hot_outside() -> Tuple[bool, str]:
    """
    После фикса free+жара процесс API должен грузить актуальный cooling.py.
    Старый код: T_room ~24 после 50 шагов; новый: ~20.7 (охлаждение работает).
    """
    post("/simulation/reset", {"seed": 0})
    post("/environment/weather-mode", {"mode": "manual"})
    post("/environment/outside", {"temperature": 35.0, "humidity": 40.0, "wind_speed": 0.0})
    post("/cooling/mode", {"mode": "free"})
    post("/cooling/setpoint", {"temperature": 22.0})
    post("/cooling/fanspeed", {"speed": 65.0})
    post("/load/mode", {"mode": "random"})
    post("/load/params", {"mean_load": 0.55, "std_load": 0.12})
    st = post("/simulation/step", {"steps": 50, "delta_time": 10.0})
    t = float(st.get("room", {}).get("temperature", 0.0))
    # Новый код: зал охлаждается от 22 к ~20.7; старый: рост к ~24+
    ok = t < 23.0
    msg = f"sanity free+35°C, 50×dt=10: T_room={t:.3f} (ожидание < 23 при актуальном коде)"
    return ok, msg


def main() -> int:
    if not health_check():
        print(f"API недоступен: {BASE}/health", file=sys.stderr)
        print("Запустите: cd DC_digital_twin && PYTHONPATH=. python3 -m uvicorn api.main:app --host 127.0.0.1 --port 8000", file=sys.stderr)
        return 2

    print(f"API: {BASE}\n")
    ok_s, msg_s = sanity_check_free_hot_outside()
    print(msg_s)
    if not ok_s:
        print(
            "ОШИБКА: похоже, API использует старый код (free при T_out≥T_ret давал Q=0). "
            "Перезапустите uvicorn из каталога DC_digital_twin с PYTHONPATH=.",
            file=sys.stderr,
        )
        return 3
    print()

    # Сценарии: (имя, режим, T_out)
    scenarios: List[Tuple[str, str, float]] = [
        ("free_Tout35", "free", 35.0),
        ("free_Tout40", "free", 40.0),
        ("free_Tout24", "free", 24.0),
        ("mixed_Tout35", "mixed", 35.0),
        ("chiller_Tout35", "chiller", 35.0),
        ("free_Tout18", "free", 18.0),
    ]

    seeds = list(range(25))  # 25 прогонов на сценарий
    chunks = 20
    steps_per_chunk = 50
    delta_time = 10.0
    total_steps = chunks * steps_per_chunk

    all_summaries: List[Dict[str, Any]] = []

    for name, mode, tout in scenarios:
        runs: List[RunResult] = []
        t0 = time.perf_counter()
        for s in seeds:
            runs.append(
                one_run(
                    name,
                    mode,
                    tout,
                    s,
                    chunks=chunks,
                    steps_per_chunk=steps_per_chunk,
                    delta_time=delta_time,
                )
            )
        elapsed = time.perf_counter() - t0
        summ = summarize_runs(name, runs)
        summ["wall_time_s"] = round(elapsed, 2)
        summ["steps_per_run"] = total_steps
        all_summaries.append(summ)

        # Пример одного прогона (первый seed): траектория
        ex = runs[0]
        print(f"=== {name} ({mode}, T_out={tout}°C) — {len(seeds)} прогонов × {total_steps} шагов, dt={delta_time}s ===")
        print(f"    время: {elapsed:.1f}s | пример seed=0: T_room: {ex.temps[0]:.3f} → {ex.temps[-1]:.3f} (min {ex.min_temp:.3f}, max {ex.max_temp:.3f})")
        if ex.telemetry_last:
            pue = ex.telemetry_last.get("pue")
            cp = ex.telemetry_last.get("cooling", {}).get("power_consumption")
            print(f"    телеметрия (последний шаг seed=0): PUE≈{pue:.3f}, P_cool≈{cp:.0f} W")
        print(
            f"    сводка: final_T_mean={summ['final_T_mean']:.3f}±{summ['final_T_std']:.3f} | "
            f"ΔT(конец-нач) mean={summ['delta_T_mean']:.3f}"
        )
        print(
            f"    монотонно неубывающих траекторий: {summ['runs_monotonic_non_decreasing']}/{summ['n']} "
            f"({100*summ['fraction_monotonic']:.1f}%)"
        )
        print(
            f"    прогонов со снижением T_room (охлаждение): {summ['runs_with_cooling_dip']}/{summ['n']} "
            f"({100*summ['fraction_with_dip']:.1f}%)"
        )
        print()

    # Сравнение: «опасный» режим — только монотонный рост без единого спада
    print("--- Итог ---")
    print(
        "Ожидание после фикса free: при высокой T_out доля прогонов с cooling_dip > 0 и "
        "fraction_monotonic < 1 (зал не только растёт)."
    )
    for s in all_summaries:
        flag = "OK" if s["fraction_with_dip"] > 0.8 or s["delta_T_mean"] < 2.0 else "проверить"
        if s["scenario"].startswith("free_Tout35") or s["scenario"].startswith("free_Tout40"):
            flag = "OK" if s["fraction_with_dip"] > 0.5 else "FAIL"
        print(
            f"  {s['scenario']}: dip={s['fraction_with_dip']*100:.0f}% mono={s['fraction_monotonic']*100:.0f}% "
            f"ΔT_mean={s['delta_T_mean']:.2f} [{flag}]"
        )

    out_csv = os.environ.get("BENCHMARK_CSV", "")
    if out_csv:
        import csv

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
            w.writeheader()
            w.writerows(all_summaries)
        print(f"\nCSV: {out_csv}")

    out_md = os.environ.get("BENCHMARK_MD", "")
    if out_md and all_summaries:
        lines = [
            "# Результаты benchmark_api_runs.py",
            "",
            f"- API: `{BASE}`",
            "- Прогон: 25 сидов × 20 чанков × 50 шагов × dt=10 с → **1000 шагов** на прогон.",
            "",
            "## Сводная таблица",
            "",
            "| Сценарий | final_T_mean | final_T_std | ΔT_mean | монотонно ↑ | со спадом (охлаждение) |",
            "|----------|--------------|-------------|---------|-------------|-------------------------|",
        ]
        for s in all_summaries:
            lines.append(
                f"| {s['scenario']} | {s['final_T_mean']:.3f} | {s['final_T_std']:.3f} | "
                f"{s['delta_T_mean']:.3f} | {s['fraction_monotonic']*100:.0f}% | {s['fraction_with_dip']*100:.0f}% |"
            )
        lines.extend(
            [
                "",
                "## Интерпретация",
                "",
                "- **со спадом** — траектория T_room не только росла: было снижение (работает отбор тепла).",
                "- При **free** и жаркой наружке после фикса ожидается доля со спадом > 0 и разумный ΔT.",
                "",
            ]
        )
        with open(out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"Markdown: {out_md}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
