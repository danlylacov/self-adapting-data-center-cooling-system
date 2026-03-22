"""Общая настройка двойника перед прогоном (ML и GA)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from orchestrator.clients import TwinClient


def setup_twin_scenario(
    twin: TwinClient,
    *,
    setpoint0: float,
    fan0: float,
    cooling_mode: str,
    mean_load: float,
    std_load: float,
    outside_temp: float,
    use_dataset: bool,
    load_dataset_path: str,
    realism: Dict[str, Any],
    seed: Optional[int],
) -> None:
    twin.post("/simulation/stop", {})
    twin.post("/simulation/reset", {"seed": seed} if seed is not None else {})
    twin.post("/cooling/mode", {"mode": cooling_mode})
    twin.post("/cooling/setpoint", {"temperature": setpoint0})
    twin.post("/cooling/fanspeed", {"speed": fan0})

    if use_dataset and load_dataset_path:
        twin.post("/datasets/load/select", {"path": load_dataset_path})
        twin.post("/load/mode", {"mode": "dataset"})
    else:
        twin.post("/load/mode", {"mode": "random"})
        twin.post("/load/params", {"mean_load": mean_load, "std_load": std_load})

    twin.post("/environment/weather-mode", {"mode": "manual"})
    twin.post("/environment/outside", {"temperature": outside_temp, "humidity": 40, "wind_speed": 0})

    twin.post("/realism/mode", {"mode": realism.get("mode", "realistic")})
    twin.post(
        "/realism/params",
        {
            "use_dynamic_crac_power": bool(realism.get("use_dynamic_crac_power", True)),
            "room_temp_clip_min": float(realism.get("room_temp_clip_min", 10)),
            "room_temp_clip_max": float(realism.get("room_temp_clip_max", 42)),
            "chip_temp_clip_multiplier": float(realism.get("chip_temp_clip_multiplier", 1.18)),
        },
    )
