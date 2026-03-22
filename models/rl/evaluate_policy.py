#!/usr/bin/env python3
"""
Evaluate an offline-trained policy artifact on `DC_digital_twin` rollouts.

Metrics:
  - never_overheat_step_fraction
  - never_overheat_episode_fraction
  - max_avg_chip_temp
  - total_energy_kwh / total_cost_eur / total_co2_kg
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


def _add_repo_root_to_sys_path() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    return repo_root


def _build_observation_from_twin_state(state: dict[str, Any], *, price_eur_per_kwh: float, carbon_kg_per_kwh: float) -> dict[str, float]:
    cooling = state.get("cooling", {}) or {}
    telemetry = state.get("telemetry", {}) or {}
    overheat_physics = float(telemetry.get("overheat_risk", 0.0))

    return {
        "avg_power_kw": float(telemetry.get("total_power_kw", 0.0)),
        "avg_chip_temp": float(telemetry.get("avg_chip_temp", 70.0)),
        "avg_inlet_temp": float(telemetry.get("avg_inlet_temp", 24.0)),
        "overheat_risk": overheat_physics,
        "overheat_risk_physics": overheat_physics,
        "ml_overheat_risk": overheat_physics,
        "delta_time_sec": float(telemetry.get("delta_time_sec", 300.0)),
        "setpoint": float(cooling.get("setpoint", 22.0)),
        "fan_speed_pct": float(cooling.get("fan_speed_pct", 50.0)),
        "price_eur_per_kwh": float(price_eur_per_kwh),
        "carbon_kg_per_kwh": float(carbon_kg_per_kwh),
    }


def _obs_to_vector(obs: dict[str, float], obs_keys: list[str]) -> np.ndarray:
    return np.array([float(obs.get(k, 0.0)) for k in obs_keys], dtype=np.float32)


class MLP(nn.Module):
    def __init__(self, *, in_dim: int, out_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, *, obs_dim: int, hidden_dims: list[int], setpoint_min: float, setpoint_max: float) -> None:
        super().__init__()
        self.setpoint_min = float(setpoint_min)
        self.setpoint_max = float(setpoint_max)
        self.body = MLP(in_dim=obs_dim, out_dim=2, hidden_dims=hidden_dims)

    def forward(self, obs_norm: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.body(obs_norm))

    def to_action(self, actor_out_norm: torch.Tensor) -> torch.Tensor:
        setpoint_norm = actor_out_norm[:, 0]
        fan_norm = actor_out_norm[:, 1]
        setpoint = self.setpoint_min + (setpoint_norm + 1.0) * 0.5 * (self.setpoint_max - self.setpoint_min)
        fan_speed = (fan_norm + 1.0) * 0.5 * 100.0
        return torch.stack([setpoint, fan_speed], dim=1)


def _apply_runtime_constraints(
    *,
    candidate_action: Any,
    prev_action: Any,
    max_setpoint_delta: float,
    max_fan_delta: float,
    setpoint_min: float,
    setpoint_max: float,
) -> Any:
    setpoint_delta = float(candidate_action.setpoint - prev_action.setpoint)
    setpoint_delta = max(-max_setpoint_delta, min(max_setpoint_delta, setpoint_delta))
    setpoint = float(prev_action.setpoint + setpoint_delta)
    setpoint = min(max(setpoint, setpoint_min), setpoint_max)

    fan_delta = float(candidate_action.fan_speed - prev_action.fan_speed)
    fan_delta = max(-max_fan_delta, min(max_fan_delta, fan_delta))
    fan = float(prev_action.fan_speed + fan_delta)
    fan = min(max(fan, 0.0), 100.0)

    return type(candidate_action)(setpoint=setpoint, fan_speed=fan)


@dataclass(frozen=True)
class EvalConfig:
    device: str
    episodes: int
    steps_per_episode: int
    seed: int | None
    price_eur_per_kwh: float
    carbon_kg_per_kwh: float
    max_setpoint_delta: float
    max_fan_delta: float
    setpoint_min: float
    setpoint_max: float


def evaluate_policy(
    *,
    policy_path: Path,
    config_path: Path,
    out_json_path: Path,
    cfg: EvalConfig,
) -> None:
    repo_root = _add_repo_root_to_sys_path()
    from services.orchestrator_service.app.controllers import Action  # type: ignore
    from services.orchestrator_service.app.reward import compute_reward  # type: ignore
    from DC_digital_twin.src.core.simulator import DataCenterSimulator  # type: ignore

    # PyTorch >=2.6 defaults to `weights_only=True`, but our artifact is a dict
    # containing numpy arrays + metadata. This must be loaded as a full object.
    policy = torch.load(policy_path, map_location="cpu", weights_only=False)
    obs_mean = policy["obs_mean"].astype(np.float32)
    obs_std = policy["obs_std"].astype(np.float32)
    obs_dim = int(obs_mean.shape[1])

    setpoint_min_policy = float(policy["setpoint_min"])
    setpoint_max_policy = float(policy["setpoint_max"])
    actor_hidden = policy.get("actor_hidden", [256, 256])

    # Use policy action bounds as defaults.
    setpoint_min = cfg.setpoint_min if cfg.setpoint_min is not None else setpoint_min_policy  # type: ignore[comparison-overlap]
    setpoint_max = cfg.setpoint_max if cfg.setpoint_max is not None else setpoint_max_policy  # type: ignore[comparison-overlap]

    # Observation keys must match dataset/training.
    obs_keys = policy.get("obs_keys") or [
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

    device = torch.device(cfg.device)
    actor = Actor(obs_dim=obs_dim, hidden_dims=list(actor_hidden), setpoint_min=setpoint_min_policy, setpoint_max=setpoint_max_policy).to(device)
    actor.load_state_dict(policy["actor_state_dict"])
    actor.eval()

    sim = DataCenterSimulator(str(config_path))

    total_steps = 0
    safe_steps = 0
    total_episodes = cfg.episodes
    safe_episodes = 0
    max_avg_chip_temp = -float("inf")

    total_energy_kwh = 0.0
    total_cost_eur = 0.0
    total_co2_kg = 0.0

    for ep in range(cfg.episodes):
        ep_seed = None if cfg.seed is None else int(cfg.seed + ep)
        sim.reset(ep_seed)

        prev_candidate_action = Action(setpoint=22.0, fan_speed=50.0)
        sim.set_cooling_setpoint(prev_candidate_action.setpoint)
        sim.set_fan_speed(prev_candidate_action.fan_speed)

        episode_overheat = False
        state = sim.get_state()
        obs_dict = _build_observation_from_twin_state(
            state,
            price_eur_per_kwh=cfg.price_eur_per_kwh,
            carbon_kg_per_kwh=cfg.carbon_kg_per_kwh,
        )

        for _ in range(cfg.steps_per_episode):
            obs_vec = _obs_to_vector(obs_dict, obs_keys=obs_keys)
            obs_norm = (obs_vec - obs_mean[0]) / obs_std[0]
            obs_norm_t = torch.tensor(obs_norm, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                out_norm = actor(obs_norm_t)  # [1,2] in [-1,1]
                action_abs = actor.to_action(out_norm)[0]  # [2]

            candidate_action = Action(setpoint=float(action_abs[0].item()), fan_speed=float(action_abs[1].item()))

            applied_action = _apply_runtime_constraints(
                candidate_action=candidate_action,
                prev_action=prev_candidate_action,
                max_setpoint_delta=cfg.max_setpoint_delta,
                max_fan_delta=cfg.max_fan_delta,
                setpoint_min=setpoint_min,
                setpoint_max=setpoint_max,
            )

            sim.set_cooling_setpoint(applied_action.setpoint)
            sim.set_fan_speed(applied_action.fan_speed)
            sim.step()

            state_after = sim.get_state()
            obs_after_dict = _build_observation_from_twin_state(
                state_after,
                price_eur_per_kwh=cfg.price_eur_per_kwh,
                carbon_kg_per_kwh=cfg.carbon_kg_per_kwh,
            )

            avg_chip_temp = float(obs_after_dict["avg_chip_temp"])
            max_avg_chip_temp = max(max_avg_chip_temp, avg_chip_temp)

            overheat_risk = float(obs_after_dict["overheat_risk_physics"])
            is_safe = overheat_risk <= 0.0
            safe_steps += 1 if is_safe else 0
            total_steps += 1

            unsafe = overheat_risk > 0.0
            if unsafe:
                episode_overheat = True

            # Energy/cost/CO2
            avg_power_kw = float(obs_after_dict["avg_power_kw"])
            delta_time_sec = float(obs_after_dict["delta_time_sec"])
            energy_kwh = avg_power_kw * (delta_time_sec / 3600.0)
            cost_eur = float(cfg.price_eur_per_kwh) * energy_kwh
            co2_kg = float(cfg.carbon_kg_per_kwh) * energy_kwh
            total_energy_kwh += energy_kwh
            total_cost_eur += cost_eur
            total_co2_kg += co2_kg

            # Optional: keep reward computation consistent with orchestrator shaping.
            delta_setpoint = float(candidate_action.setpoint - prev_candidate_action.setpoint)
            delta_fan = float(candidate_action.fan_speed - prev_candidate_action.fan_speed)
            _reward, _ = compute_reward(
                energy_kwh=energy_kwh,
                cost=cost_eur,
                co2=co2_kg,
                overheat_risk=overheat_risk,
                delta_setpoint=delta_setpoint,
                delta_fan=delta_fan,
            )

            prev_candidate_action = candidate_action
            obs_dict = obs_after_dict

            if unsafe:
                break

        if not episode_overheat:
            safe_episodes += 1

    report = {
        "policy_path": str(policy_path),
        "config_path": str(config_path),
        "episodes": cfg.episodes,
        "steps_per_episode": cfg.steps_per_episode,
        "safe_episodes": safe_episodes,
        "never_overheat_episode_fraction": safe_episodes / total_episodes if total_episodes > 0 else 0.0,
        "safe_steps": safe_steps,
        "never_overheat_step_fraction": safe_steps / total_steps if total_steps > 0 else 0.0,
        "max_avg_chip_temp": max_avg_chip_temp,
        "total_energy_kwh": total_energy_kwh,
        "total_cost_eur": total_cost_eur,
        "total_co2_kg": total_co2_kg,
    }

    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved eval report: {out_json_path}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate offline-trained policy on the twin.")
    parser.add_argument("--policy-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--out-json", type=str, required=True)

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--steps-per-episode", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)

    parser.add_argument("--price-eur-per-kwh", type=float, default=0.2)
    parser.add_argument("--carbon-kg-per-kwh", type=float, default=0.4)

    parser.add_argument("--setpoint-min", type=float, default=18.0)
    parser.add_argument("--setpoint-max", type=float, default=27.0)
    parser.add_argument("--max-setpoint-delta", type=float, default=1.0)
    parser.add_argument("--max-fan-delta", type=float, default=10.0)

    args = parser.parse_args()

    cfg = EvalConfig(
        device=args.device,
        episodes=args.episodes,
        steps_per_episode=args.steps_per_episode,
        seed=args.seed,
        price_eur_per_kwh=args.price_eur_per_kwh,
        carbon_kg_per_kwh=args.carbon_kg_per_kwh,
        max_setpoint_delta=args.max_setpoint_delta,
        max_fan_delta=args.max_fan_delta,
        setpoint_min=args.setpoint_min,
        setpoint_max=args.setpoint_max,
    )

    evaluate_policy(
        policy_path=Path(args.policy_path),
        config_path=Path(args.config_path),
        out_json_path=Path(args.out_json),
        cfg=cfg,
    )


if __name__ == "__main__":
    main()

