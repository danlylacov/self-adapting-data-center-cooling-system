#!/usr/bin/env python3
"""Несколько прогонов POST /run и сводка метрик (для оценки ML-оркестратора)."""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List


ORCH = "http://127.0.0.1:8030"


def post_run(body: Dict[str, Any], timeout: int = 600) -> Dict[str, Any]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{ORCH}/run",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


@dataclass
class RunSummary:
    name: str
    steps: int
    n_points: int
    pue_mean: float
    pue_min: float
    pue_max: float
    room_end: float
    avg_chip_end: float
    overheat_pct_end: float
    setpoint_start: float
    setpoint_end: float
    fan_start: float
    fan_end: float
    ml_fallback: bool
    max_p_overheat_reported: Any
    pue_deltas_nonnull: int


def summarize(name: str, resp: Dict[str, Any]) -> RunSummary:
    pts: List[Dict[str, Any]] = resp.get("points") or []
    meta = resp.get("meta") or {}
    fs = resp.get("finalState") or {}
    tel = fs.get("telemetry") or {}

    pues = [float(p.get("pue") or 0) for p in pts]
    first = pts[0] if pts else {}
    last = pts[-1] if pts else {}

    d_nonnull = sum(1 for p in pts if p.get("pueDeltaRecommended") is not None)

    return RunSummary(
        name=name,
        steps=len(pts),
        n_points=len(pts),
        pue_mean=sum(pues) / len(pues) if pues else 0.0,
        pue_min=min(pues) if pues else 0.0,
        pue_max=max(pues) if pues else 0.0,
        room_end=float(fs.get("room", {}).get("temperature") or last.get("room") or 0),
        avg_chip_end=float(tel.get("avg_chip_temp") or last.get("avgChip") or 0),
        overheat_pct_end=float(tel.get("overheat_risk") or 0) * 100.0,
        setpoint_start=float(first.get("appliedSetpoint") or 0),
        setpoint_end=float(last.get("appliedSetpoint") or 0),
        fan_start=float(first.get("appliedFanSpeed") or 0),
        fan_end=float(last.get("appliedFanSpeed") or 0),
        ml_fallback=bool(meta.get("mlFallbackUsed")),
        max_p_overheat_reported=meta.get("maxPOverheat"),
        pue_deltas_nonnull=d_nonnull,
    )


def base_scenario(steps: int, delta_time: int = 10) -> Dict[str, Any]:
    return {
        "steps": steps,
        "deltaTime": delta_time,
        "setpoint": 22,
        "fanSpeed": 65,
        "coolingMode": "mixed",
        "meanLoad": 0.55,
        "stdLoad": 0.12,
        "outsideTemp": 24,
        "useDatasetLoad": False,
    }


def main() -> None:
    scenarios: List[tuple[str, Dict[str, Any]]] = [
        (
            "A: baseline (interval=5, steps=80)",
            {
                "scenario": base_scenario(80),
                "controlIntervalSteps": 5,
                "safetyMaxPOverheat": 0.2,
                "loadForecastModel": "prophet",
                "failOnMlUnavailable": False,
                "seed": 42,
            },
        ),
        (
            "B: чаще управление (interval=3, steps=80)",
            {
                "scenario": base_scenario(80),
                "controlIntervalSteps": 3,
                "safetyMaxPOverheat": 0.2,
                "loadForecastModel": "prophet",
                "failOnMlUnavailable": False,
                "seed": 42,
            },
        ),
        (
            "C: выше порог перегрева (0.35), steps=80",
            {
                "scenario": base_scenario(80),
                "controlIntervalSteps": 5,
                "safetyMaxPOverheat": 0.35,
                "loadForecastModel": "prophet",
                "failOnMlUnavailable": False,
                "seed": 42,
            },
        ),
        (
            "D: другой seed, steps=80",
            {
                "scenario": base_scenario(80),
                "controlIntervalSteps": 5,
                "safetyMaxPOverheat": 0.2,
                "loadForecastModel": "prophet",
                "failOnMlUnavailable": False,
                "seed": 999,
            },
        ),
        (
            "E: длинный прогон (steps=150, interval=5)",
            {
                "scenario": base_scenario(150),
                "controlIntervalSteps": 5,
                "safetyMaxPOverheat": 0.2,
                "loadForecastModel": "prophet",
                "failOnMlUnavailable": False,
                "seed": 7,
            },
        ),
    ]

    summaries: List[RunSummary] = []

    for name, body in scenarios:
        print(f"\n>>> {name} ...", flush=True)
        try:
            resp = post_run(body)
        except urllib.error.HTTPError as e:
            print(f"HTTP {e.code}: {e.read().decode()[:500]}", file=sys.stderr)
            continue
        except Exception as e:
            print(f"Ошибка: {e}", file=sys.stderr)
            continue
        s = summarize(name, resp)
        summaries.append(s)
        print(
            f"    PUE mean={s.pue_mean:.3f} [{s.pue_min:.3f}–{s.pue_max:.3f}], "
            f"room_end={s.room_end:.2f}°C, chip={s.avg_chip_end:.2f}°C, "
            f"risk={s.overheat_pct_end:.1f}%, "
            f"setpoint {s.setpoint_start:.1f}→{s.setpoint_end:.1f}, fan {s.fan_start:.0f}→{s.fan_end:.0f}%, "
            f"ml_fallback={s.ml_fallback}, ticks_with_ΔPUE={s.pue_deltas_nonnull}",
            flush=True,
        )

    if not summaries:
        print("Нет успешных прогонов.", file=sys.stderr)
        sys.exit(1)

    print("\n=== Сводная таблица ===")
    print(
        f"{'Сценарий':<40} {'PUĒ':>7} {'PUE max':>8} {'T зал':>7} {'T чип':>7} {'риск%':>6} {'Δsetpt':>8} {'fan%':>6}"
    )
    for s in summaries:
        dsp = s.setpoint_end - s.setpoint_start
        print(
            f"{s.name[:40]:<40} {s.pue_mean:7.3f} {s.pue_max:8.3f} {s.room_end:7.2f} {s.avg_chip_end:7.2f} "
            f"{s.overheat_pct_end:6.1f} {dsp:+8.2f} {s.fan_end:6.0f}"
        )

    print("\n=== Краткая оценка ===")
    means = [s.pue_mean for s in summaries]
    print(f"Средний PUE по сценариям: min={min(means):.3f}, max={max(means):.3f}, разброс={max(means)-min(means):.3f}")
    risks = [s.overheat_pct_end for s in summaries]
    print(f"Финальный риск перегрева (%): min={min(risks):.1f}, max={max(risks):.1f}")
    if any(s.ml_fallback for s in summaries):
        print("Внимание: в каких-то прогонах срабатывал ML fallback (сервис недоступен или ошибка).")
    else:
        print("ML fallback не использовался — все вызовы PUE/temp/load отработали.")
    print(
        "Интервал управления 3 (B) vs 5 (A): чаще тики — больше шансов изменить setpoint/fan по рекомендациям; "
        "порог 0.35 (C) реже поднимает вентилятор до 100% при тех же предсказаниях p_overheat."
    )


if __name__ == "__main__":
    main()
