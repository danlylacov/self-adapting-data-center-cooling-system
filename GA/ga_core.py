"""
Общая логика ГА и оценки на API цифрового двойника (используется train_ga.py и ноутбуком).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

import httpx
import numpy as np

SP_MIN_C = 18.0
SP_MAX_C = 27.0

# [target_chip, kp, fan_base, k_fan, max_delta_sp]
GENE_BOUNDS = np.array(
    [
        [58.0, 68.0],
        [0.02, 0.45],
        [35.0, 95.0],
        [0.0, 8.0],
        [0.15, 2.5],
    ]
)

GENE_NAMES = ("target_chip_c", "kp", "fan_base", "k_fan", "max_delta_sp")


def mix_ga_rng_seed(base: int, episode_seed: int, outside_temp: float) -> int:
    """
    Детерминированно смешивает базовый сид ГА с параметрами сценария, чтобы при
    одинаковом --rng-seed разные (seed, outside_temp) давали разную инициализацию
    популяции и не сливались лучшие хромосомы только из-за совпадения RNG.
    """
    t = int(round(float(outside_temp) * 1000.0)) & 0xFFFFFFFF
    s = (int(episode_seed) & 0xFFFFFFFF) * 0x9E3779B1
    mixed = (int(base) ^ s ^ t) & 0xFFFFFFFF
    return int(mixed) if mixed != 0 else 1


def clip_chromosome(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    for i in range(len(out)):
        lo, hi = GENE_BOUNDS[i]
        out[i] = float(np.clip(out[i], lo, hi))
    return out


def random_chromosome(rng: np.random.Generator) -> np.ndarray:
    low = GENE_BOUNDS[:, 0]
    high = GENE_BOUNDS[:, 1]
    return clip_chromosome(rng.uniform(low, high))


@dataclass
class EpisodeConfig:
    n_steps: int
    delta_time: float
    control_interval_steps: int
    seed: int
    setpoint0: float
    fan0: float
    cooling_mode: str
    mean_load: float
    std_load: float
    outside_temp: float


class TwinApi:
    def __init__(self, base_url: str, timeout: float = 120.0):
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _json(self, method: str, path: str, body: Optional[dict] = None) -> Any:
        r = self._client.request(method, path, json=body)
        r.raise_for_status()
        if r.status_code == 204 or not r.content:
            return None
        return r.json()

    def post(self, path: str, body: Optional[dict] = None) -> Any:
        return self._json("POST", path, body or {})

    def get(self, path: str) -> Any:
        return self._json("GET", path)

    def stop(self) -> None:
        self.post("/simulation/stop", {})

    def reset(self, seed: Optional[int] = None) -> None:
        self.post("/simulation/reset", {} if seed is None else {"seed": seed})

    def step(self, delta_time: float, steps: int = 1) -> Any:
        return self.post("/simulation/step", {"steps": steps, "delta_time": delta_time})

    def state(self) -> dict:
        return self.get("/simulation/state")

    def telemetry(self) -> dict:
        return self.get("/simulation/telemetry")

    def set_cooling_mode(self, mode: str) -> Any:
        return self.post("/cooling/mode", {"mode": mode})

    def set_setpoint(self, temperature: float) -> Any:
        return self.post("/cooling/setpoint", {"temperature": float(temperature)})

    def set_fanspeed(self, speed_pct: float) -> Any:
        return self.post("/cooling/fanspeed", {"speed": float(speed_pct)})

    def setup_scenario(
        self,
        *,
        seed: Optional[int],
        setpoint0: float,
        fan0: float,
        cooling_mode: str,
        mean_load: float,
        std_load: float,
        outside_temp: float,
    ) -> None:
        self.stop()
        self.reset(seed=seed)
        self.set_cooling_mode(cooling_mode)
        self.set_setpoint(setpoint0)
        self.set_fanspeed(fan0)
        self.post("/load/mode", {"mode": "random"})
        self.post("/load/params", {"mean_load": mean_load, "std_load": std_load})
        self.post("/environment/weather-mode", {"mode": "manual"})
        self.post(
            "/environment/outside",
            {"temperature": outside_temp, "humidity": 40.0, "wind_speed": 0.0},
        )


def evaluate_chromosome(
    twin: TwinApi,
    chrom: np.ndarray,
    cfg: EpisodeConfig,
    *,
    chip_temp_limit_c: float,
    temp_penalty_weight: float,
) -> float:
    g = clip_chromosome(chrom)
    target_chip, kp, fan_base, k_fan, max_delta_sp = g

    twin.setup_scenario(
        seed=cfg.seed,
        setpoint0=cfg.setpoint0,
        fan0=cfg.fan0,
        cooling_mode=cfg.cooling_mode,
        mean_load=cfg.mean_load,
        std_load=cfg.std_load,
        outside_temp=cfg.outside_temp,
    )

    total_energy_j = 0.0
    temp_penalty = 0.0

    for step_idx in range(1, cfg.n_steps + 1):
        twin.step(cfg.delta_time, steps=1)
        tel = twin.telemetry()
        st = twin.state()

        p_cool = float(tel.get("cooling", {}).get("power_consumption", 0.0))
        total_energy_j += p_cool * cfg.delta_time

        avg_chip = float(st.get("telemetry", {}).get("avg_chip_temp", 70.0))
        temp_penalty += max(0.0, avg_chip - chip_temp_limit_c) * cfg.delta_time

        if step_idx % cfg.control_interval_steps != 0:
            continue

        err = avg_chip - target_chip
        delta_sp = -kp * err
        delta_sp = float(np.clip(delta_sp, -max_delta_sp, max_delta_sp))

        crac = (st.get("cooling") or {}).get("crac") or {}
        cur_sp = float(crac.get("setpoint", cfg.setpoint0))
        new_sp = float(np.clip(cur_sp + delta_sp, SP_MIN_C, SP_MAX_C))
        twin.set_setpoint(new_sp)

        fan = fan_base + k_fan * max(0.0, err)
        fan = float(np.clip(fan, 0.0, 100.0))
        twin.set_fanspeed(fan)

    return total_energy_j + temp_penalty_weight * temp_penalty


def tournament_select(rng: np.random.Generator, pop: np.ndarray, fitness: np.ndarray, k: int = 3) -> int:
    idx = rng.choice(len(pop), size=k, replace=False)
    best = idx[np.argmin(fitness[idx])]
    return int(best)


def crossover(rng: np.random.Generator, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    alpha = float(rng.uniform(0.2, 0.8))
    child = alpha * p1 + (1.0 - alpha) * p2
    return clip_chromosome(child)


def mutate(rng: np.random.Generator, x: np.ndarray, sigma: float = 0.08, p_mutate: float = 0.15) -> np.ndarray:
    y = x.copy()
    for i in range(len(y)):
        if rng.random() < p_mutate:
            lo, hi = GENE_BOUNDS[i]
            scale = (hi - lo) * sigma
            y[i] += float(rng.normal(0.0, scale))
    return clip_chromosome(y)


def run_ga(
    twin: TwinApi,
    episode_cfg: EpisodeConfig,
    *,
    population_size: int = 12,
    n_generations: int = 15,
    chip_temp_limit_c: float = 72.0,
    temp_penalty_weight: float = 500.0,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, list[float], list[np.ndarray]]:
    if rng is None:
        rng = np.random.default_rng(7)

    pop = np.array([random_chromosome(rng) for _ in range(population_size)])
    best_fitness_hist: list[float] = []
    best_chrom_hist: list[np.ndarray] = []
    best_ever_f = float("inf")
    best_ever_chrom: Optional[np.ndarray] = None

    for _gen in range(n_generations):
        fitness = np.array(
            [
                evaluate_chromosome(
                    twin,
                    pop[i],
                    episode_cfg,
                    chip_temp_limit_c=chip_temp_limit_c,
                    temp_penalty_weight=temp_penalty_weight,
                )
                for i in range(population_size)
            ]
        )
        best_i = int(np.argmin(fitness))
        best_f = float(fitness[best_i])
        best_ind = pop[best_i].copy()
        best_fitness_hist.append(best_f)
        best_chrom_hist.append(best_ind)
        if best_f < best_ever_f:
            best_ever_f = best_f
            best_ever_chrom = best_ind.copy()

        new_pop = [best_ind.copy()]
        while len(new_pop) < population_size:
            i1 = tournament_select(rng, pop, fitness)
            i2 = tournament_select(rng, pop, fitness)
            child = crossover(rng, pop[i1], pop[i2])
            child = mutate(rng, child, sigma=0.08)
            new_pop.append(child)
        pop = np.array(new_pop)

    assert best_ever_chrom is not None
    return best_ever_chrom, best_fitness_hist, best_chrom_hist


def chrom_to_dict(chrom: np.ndarray) -> dict[str, float]:
    g = clip_chromosome(chrom)
    return {name: float(g[i]) for i, name in enumerate(GENE_NAMES)}


def save_tuned_params(
    path: str,
    *,
    chrom: np.ndarray,
    episode: EpisodeConfig,
    fitness_history: list[float],
    chip_temp_limit_c: float,
    temp_penalty_weight: float,
    rng_seed: int,
    rng_seed_effective: Optional[int] = None,
    extra: Optional[dict[str, Any]] = None,
) -> None:
    training: dict[str, Any] = {
        "episode": asdict(episode),
        "chip_temp_limit_c": chip_temp_limit_c,
        "temp_penalty_weight": temp_penalty_weight,
        "rng_seed": rng_seed,
    }
    if rng_seed_effective is not None:
        training["rng_seed_effective"] = rng_seed_effective
    payload = {
        "version": 1,
        "genes": chrom_to_dict(chrom),
        "fitness_best": float(min(fitness_history)) if fitness_history else None,
        "fitness_by_generation": [float(x) for x in fitness_history],
        "training": training,
    }
    if extra:
        payload["extra"] = extra
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
