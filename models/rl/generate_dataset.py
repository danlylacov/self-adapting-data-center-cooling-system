#!/usr/bin/env python3
"""
Generate an offline RL dataset by rolling out `DC_digital_twin` with a safe/baseline controller.

Dataset format (saved as .npz):
  - obs:        [N, D]  float32
  - actions:   [N, 2]  float32  (setpoint, fan_speed_pct)
  - rewards:   [N]     float32
  - next_obs:  [N, D]  float32
  - dones:     [N]     bool   (terminal either unsafe or time limit)
  - unsafe:    [N]     bool   (terminal because overheat happened)
Plus a sidecar JSON with `meta` and `obs_keys`.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np


def _add_repo_root_to_sys_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    return repo_root


def _add_digital_twin_src_to_sys_path(repo_root: Path) -> None:
    sys.path.insert(0, str(repo_root / "DC_digital_twin" / "src"))


def _obs_to_vector(obs: dict[str, float], obs_keys: list[str]) -> np.ndarray:
    return np.array([float(obs.get(k, 0.0)) for k in obs_keys], dtype=np.float32)


def _build_observation_from_twin_state(state: dict[str, Any], *, price_eur_per_kwh: float, carbon_kg_per_kwh: float) -> dict[str, float]:
    cooling = state.get("cooling", {}) or {}
    telemetry = state.get("telemetry", {}) or {}
    overheat_physics = float(telemetry.get("overheat_risk", 0.0))

    return {
        "avg_power_kw": float(telemetry.get("total_power_kw", 0.0)),
        "avg_chip_temp": float(telemetry.get("avg_chip_temp", 70.0)),
        "avg_inlet_temp": float(telemetry.get("avg_inlet_temp", 24.0)),
        # Keep the same semantics as orchestrator: physical simulator risk here.
        "overheat_risk": overheat_physics,
        "overheat_risk_physics": overheat_physics,
        # No temp predictor during offline dataset generation (unless caller wires it in later).
        "ml_overheat_risk": overheat_physics,
        "delta_time_sec": float(telemetry.get("delta_time_sec", 300.0)),
        "setpoint": float(cooling.get("setpoint", 22.0)),
        "fan_speed_pct": float(cooling.get("fan_speed_pct", 50.0)),
        "price_eur_per_kwh": float(price_eur_per_kwh),
        "carbon_kg_per_kwh": float(carbon_kg_per_kwh),
    }


def _apply_runtime_constraints(
    *,
    candidate_action: Any,
    prev_action: Any,
    max_setpoint_delta: float,
    max_fan_delta: float,
    setpoint_min: float,
    setpoint_max: float,
) -> Any:
    # Mimics `RuntimeManager._apply_action()` logic: clamp deltas relative to prev applied action.
    setpoint_delta = float(candidate_action.setpoint - prev_action.setpoint)
    setpoint_delta = max(-max_setpoint_delta, min(max_setpoint_delta, setpoint_delta))
    setpoint = float(prev_action.setpoint + setpoint_delta)
    setpoint = min(max(setpoint, setpoint_min), setpoint_max)

    fan_delta = float(candidate_action.fan_speed - prev_action.fan_speed)
    fan_delta = max(-max_fan_delta, min(max_fan_delta, fan_delta))
    fan = float(prev_action.fan_speed + fan_delta)
    fan = min(max(fan, 0.0), 100.0)

    # Preserve the Action type (from orchestrator controllers).
    return type(candidate_action)(setpoint=setpoint, fan_speed=fan)


def generate_dataset(
    *,
    config_path: Path,
    episodes: int,
    steps_per_episode: int,
    seed: int | None,
    out_path: Path,
    price_eur_per_kwh: float,
    carbon_kg_per_kwh: float,
    setpoint_min: float,
    setpoint_max: float,
    max_setpoint_delta: float,
    max_fan_delta: float,
) -> None:
    repo_root = _add_repo_root_to_sys_path()

    # Imports after sys.path tweaking.
    # Import with a full package path so that `..models.*` relative imports inside
    # `DC_digital_twin/src/core/simulator.py` resolve correctly.
    from DC_digital_twin.src.core.simulator import DataCenterSimulator  # type: ignore
    from services.orchestrator_service.app.controllers import Action, BaselineController  # type: ignore
    from services.orchestrator_service.app.reward import compute_reward  # type: ignore

    obs_keys = [
        "avg_power_kw",
        "avg_chip_temp",
        "avg_inlet_temp",
        "overheat_risk",
        "overheat_risk_physics",
        "ml_overheat_risk",
        "delta_time_sec",
        "setpoint",
        "fan_speed_pct",
        "price_eur_per_kwh",
        "carbon_kg_per_kwh",
    ]

    baseline = BaselineController()

    # Storage
    obs_list: list[np.ndarray] = []
    next_obs_list: list[np.ndarray] = []
    actions_list: list[np.ndarray] = []
    rewards_list: list[float] = []
    dones_list: list[bool] = []
    unsafe_list: list[bool] = []
    episode_boundaries: list[dict[str, Any]] = []

    sim = DataCenterSimulator(str(config_path))
    overheat_threshold_c = float(sim.config.get("overheat_threshold_c", 75.0))

    global_step = 0
    for ep in range(episodes):
        ep_seed = None if seed is None else int(seed + ep)
        sim.reset(ep_seed)

        # Align simulator cooling with orchestrator env initial `last_action` values.
        prev_candidate_action = Action(setpoint=22.0, fan_speed=50.0)
        sim.set_cooling_setpoint(prev_candidate_action.setpoint)
        sim.set_fan_speed(prev_candidate_action.fan_speed)

        state = sim.get_state()
        obs = _build_observation_from_twin_state(
            state,
            price_eur_per_kwh=price_eur_per_kwh,
            carbon_kg_per_kwh=carbon_kg_per_kwh,
        )

        start_idx = global_step
        for t in range(steps_per_episode):
            candidate_action = baseline.decide(obs)

            applied_action = _apply_runtime_constraints(
                candidate_action=candidate_action,
                prev_action=prev_candidate_action,
                max_setpoint_delta=max_setpoint_delta,
                max_fan_delta=max_fan_delta,
                setpoint_min=setpoint_min,
                setpoint_max=setpoint_max,
            )

            sim.set_cooling_setpoint(applied_action.setpoint)
            sim.set_fan_speed(applied_action.fan_speed)
            sim.step()

            state_after = sim.get_state()
            next_obs_dict = _build_observation_from_twin_state(
                state_after,
                price_eur_per_kwh=price_eur_per_kwh,
                carbon_kg_per_kwh=carbon_kg_per_kwh,
            )
            next_obs_vec = _obs_to_vector(next_obs_dict, obs_keys=obs_keys)

            delta_setpoint = float(candidate_action.setpoint - prev_candidate_action.setpoint)
            delta_fan = float(candidate_action.fan_speed - prev_candidate_action.fan_speed)

            avg_power_kw = float(next_obs_dict["avg_power_kw"])
            delta_time_sec = float(next_obs_dict["delta_time_sec"])
            energy_kwh = avg_power_kw * (delta_time_sec / 3600.0)
            cost = float(price_eur_per_kwh) * energy_kwh
            co2 = float(carbon_kg_per_kwh) * energy_kwh
            overheat_risk = float(next_obs_dict["overheat_risk_physics"])

            reward, _breakdown = compute_reward(
                energy_kwh=energy_kwh,
                cost=cost,
                co2=co2,
                overheat_risk=overheat_risk,
                delta_setpoint=delta_setpoint,
                delta_fan=delta_fan,
            )

            unsafe = overheat_risk > 0.0
            time_limit = t == (steps_per_episode - 1)
            done = bool(unsafe or time_limit)

            obs_list.append(_obs_to_vector(obs, obs_keys=obs_keys))
            next_obs_list.append(next_obs_vec)
            actions_list.append(np.array([candidate_action.setpoint, candidate_action.fan_speed], dtype=np.float32))
            rewards_list.append(float(reward))
            dones_list.append(done)
            unsafe_list.append(bool(unsafe))

            global_step += 1

            prev_candidate_action = candidate_action
            obs = next_obs_dict

            if unsafe:
                break

        episode_boundaries.append(
            {
                "episode": ep,
                "start_index": start_idx,
                "end_index_exclusive": global_step,
                "unsafe_in_episode": any(unsafe_list[start_idx:global_step]),
            }
        )

    obs_arr = np.stack(obs_list, axis=0) if obs_list else np.zeros((0, len(obs_keys)), dtype=np.float32)
    next_obs_arr = np.stack(next_obs_list, axis=0) if next_obs_list else np.zeros((0, len(obs_keys)), dtype=np.float32)
    actions_arr = np.stack(actions_list, axis=0) if actions_list else np.zeros((0, 2), dtype=np.float32)
    rewards_arr = np.asarray(rewards_list, dtype=np.float32)
    dones_arr = np.asarray(dones_list, dtype=bool)
    unsafe_arr = np.asarray(unsafe_list, dtype=bool)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        obs=obs_arr,
        actions=actions_arr,
        rewards=rewards_arr,
        next_obs=next_obs_arr,
        dones=dones_arr,
        unsafe=unsafe_arr,
    )

    meta = {
        "config_path": str(config_path),
        "episodes": episodes,
        "steps_per_episode": steps_per_episode,
        "seed": seed,
        "price_eur_per_kwh": price_eur_per_kwh,
        "carbon_kg_per_kwh": carbon_kg_per_kwh,
        "threshold_c": overheat_threshold_c,
        "constraints": {
            "setpoint_min": setpoint_min,
            "setpoint_max": setpoint_max,
            "max_setpoint_delta": max_setpoint_delta,
            "max_fan_delta": max_fan_delta,
        },
        "obs_keys": obs_keys,
        "episode_boundaries": episode_boundaries,
        "N_transitions": int(obs_arr.shape[0]),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved dataset: {out_path} (N={obs_arr.shape[0]})")
    print(f"Saved meta: {meta_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline RL dataset generator (twin rollouts).")
    parser.add_argument("--config-path", type=str, required=True, help="Path to DC_digital_twin YAML config")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps-per-episode", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out-path", type=str, required=True, help="Output .npz path")

    parser.add_argument("--price-eur-per-kwh", type=float, default=0.2)
    parser.add_argument("--carbon-kg-per-kwh", type=float, default=0.4)

    parser.add_argument("--setpoint-min", type=float, default=18.0)
    parser.add_argument("--setpoint-max", type=float, default=27.0)
    parser.add_argument("--max-setpoint-delta", type=float, default=1.0)
    parser.add_argument("--max-fan-delta", type=float, default=10.0)

    args = parser.parse_args()

    generate_dataset(
        config_path=Path(args.config_path),
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        seed=args.seed,
        out_path=Path(args.out_path),
        price_eur_per_kwh=args.price_eur_per_kwh,
        carbon_kg_per_kwh=args.carbon_kg_per_kwh,
        setpoint_min=args.setpoint_min,
        setpoint_max=args.setpoint_max,
        max_setpoint_delta=args.max_setpoint_delta,
        max_fan_delta=args.max_fan_delta,
    )


if __name__ == "__main__":
    main()

