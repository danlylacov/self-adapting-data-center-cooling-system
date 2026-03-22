#!/usr/bin/env python3
"""
Прогон генетического алгоритма для подбора параметров политики охлаждения.
Результат: GA/tuned_params.json + импорт GaCoolingPolicy из ga_policy.

Запуск из корня репозитория v2:
  PYTHONPATH=. python -m GA.train_ga

Переменная окружения TWIN_BASE (по умолчанию http://127.0.0.1:8000).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Корень v2 в sys.path
_V2_ROOT = Path(__file__).resolve().parent.parent
if str(_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(_V2_ROOT))

import numpy as np

from GA.ga_core import (
    EpisodeConfig,
    TwinApi,
    mix_ga_rng_seed,
    run_ga,
    save_tuned_params,
)


def main() -> None:
    p = argparse.ArgumentParser(description="GA tuning for cooling policy via digital twin API")
    p.add_argument("--twin-base", default=os.environ.get("TWIN_BASE", "http://127.0.0.1:8000"))
    p.add_argument("--out", default=str(_V2_ROOT / "GA" / "tuned_params.json"))
    p.add_argument("--population", type=int, default=14)
    p.add_argument("--generations", type=int, default=18)
    p.add_argument("--seed", type=int, default=42, help="Сценарий симуляции (нагрузка)")
    p.add_argument("--rng-seed", type=int, default=123, help="Базовый сид ГА")
    p.add_argument(
        "--mix-rng-with-scenario",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Смешивать --rng-seed с --seed и --outside-temp (разная популяция при разных сценариях)",
    )
    p.add_argument("--n-steps", type=int, default=120)
    p.add_argument("--delta-time", type=float, default=10.0)
    p.add_argument("--control-interval", type=int, default=5)
    p.add_argument("--outside-temp", type=float, default=28.0)
    args = p.parse_args()

    episode = EpisodeConfig(
        n_steps=args.n_steps,
        delta_time=args.delta_time,
        control_interval_steps=args.control_interval,
        seed=args.seed,
        setpoint0=22.0,
        fan0=65.0,
        cooling_mode="mixed",
        mean_load=0.55,
        std_load=0.12,
        outside_temp=args.outside_temp,
    )

    chip_limit = 72.0
    penalty_w = 500.0

    twin = TwinApi(args.twin_base)
    try:
        twin.get("/health")
    except Exception as e:
        print("Не удаётся достучаться до двойника:", e, file=sys.stderr)
        sys.exit(1)

    rng_seed_eff = (
        mix_ga_rng_seed(args.rng_seed, args.seed, args.outside_temp)
        if args.mix_rng_with_scenario
        else args.rng_seed
    )
    rng = np.random.default_rng(rng_seed_eff)
    print(
        f"GA: population={args.population} generations={args.generations} "
        f"episode_steps={args.n_steps} twin={args.twin_base} "
        f"rng_seed={args.rng_seed} effective={rng_seed_eff} mix_scenario={args.mix_rng_with_scenario}",
        flush=True,
    )

    try:
        best_chrom, fitness_hist, _ = run_ga(
            twin,
            episode,
            population_size=args.population,
            n_generations=args.generations,
            chip_temp_limit_c=chip_limit,
            temp_penalty_weight=penalty_w,
            rng=rng,
        )
    finally:
        twin.close()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_tuned_params(
        str(out_path),
        chrom=best_chrom,
        episode=episode,
        fitness_history=fitness_hist,
        chip_temp_limit_c=chip_limit,
        temp_penalty_weight=penalty_w,
        rng_seed=args.rng_seed,
        rng_seed_effective=rng_seed_eff if args.mix_rng_with_scenario else None,
        extra={"cli": vars(args)},
    )

    print("Готово. Лучший fitness:", min(fitness_hist))
    print("Параметры сохранены:", out_path.resolve())


if __name__ == "__main__":
    main()
