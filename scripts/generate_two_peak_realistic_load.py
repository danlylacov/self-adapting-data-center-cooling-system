#!/usr/bin/env python3
"""Два плато нагрузки (как в v1), но с плавными фронтами и шумом → data/two_peak_realistic_load.csv."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def smoothstep(edge0: float, edge1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def add_plateau(
    t: np.ndarray,
    n: int,
    center: float,
    half_width: float,
    ramp_in: float,
    ramp_out: float,
    level: float,
    rng: np.random.Generator,
) -> np.ndarray:
    a = center - half_width
    b = center + half_width
    m0 = smoothstep(a - ramp_in, a, t)
    m1 = 1.0 - smoothstep(b, b + ramp_out, t)
    plate = np.clip(np.minimum(m0, m1), 0.0, 1.0)
    jitter = 0.025 * np.sin(2 * np.pi * (t - a) / max(b - a, 1)) + rng.normal(0, 0.008, n)
    return level * plate * (1.0 + jitter)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=1440)
    p.add_argument("--seed", type=int, default=20250322)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    n = args.steps
    t = np.arange(n, dtype=float)

    base = 0.30 + 0.05 * np.sin(2 * np.pi * t / (n * 0.85))
    eta = np.zeros(n)
    for i in range(1, n):
        eta[i] = 0.92 * eta[i - 1] + rng.normal(0, 0.012)
    load = base + eta

    load += add_plateau(t, n, 380.0, 95.0, 55.0, 48.0, 0.66, rng)
    load += add_plateau(t, n, 1020.0, 110.0, 62.0, 55.0, 0.62, rng)

    for _ in range(18):
        c = float(rng.integers(40, n - 40))
        w = float(rng.uniform(4.0, 14.0))
        amp = float(rng.uniform(0.05, 0.14))
        load += amp * np.exp(-0.5 * ((t - c) / w) ** 2)

    load = np.clip(load, 0.08, 1.0)
    load = np.round(load, 5)

    root = Path(__file__).resolve().parents[1]
    for rel in (
        root / "data" / "two_peak_realistic_load.csv",
        root / "DC_digital_twin" / "data" / "two_peak_realistic_load.csv",
    ):
        rel.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"step_index": np.arange(n), "load": load}).to_csv(rel, index=False)
        print("Wrote", rel, "min/max/mean:", float(load.min()), float(load.max()), float(load.mean()))


if __name__ == "__main__":
    main()
