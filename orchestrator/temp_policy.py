"""Temperature-aware gating for PUE setpoint deltas (orchestrator)."""

from __future__ import annotations

import math
from typing import Any


def adjust_pue_delta_for_chip_temp(
    delta_c: float,
    avg_chip_temp_c: Any,
    *,
    target_c: float,
    deadband_c: float,
    enabled: bool,
) -> float:
    """
    Post-process ML-recommended setpoint delta using avg chip temperature deadband.

    - If t > target + deadband (hot): do not relax cooling — delta_c := min(delta_c, 0).
    - If t < target - deadband (cold): do not strengthen cooling — delta_c := max(delta_c, 0).
    - In-band: leave delta_c unchanged.

    Non-finite avg_chip_temp skips adjustment and returns delta_c unchanged.
    """
    if not enabled:
        return float(delta_c)
    try:
        t = float(avg_chip_temp_c)
    except (TypeError, ValueError):
        return float(delta_c)
    if not math.isfinite(t):
        return float(delta_c)

    lo = float(target_c) - float(deadband_c)
    hi = float(target_c) + float(deadband_c)
    d = float(delta_c)

    if t > hi:
        return min(d, 0.0)
    if t < lo:
        return max(d, 0.0)
    return d
