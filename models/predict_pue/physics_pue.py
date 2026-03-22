#!/usr/bin/env python3
"""
Упрощенная физическая (baseline) модель PUE для гибридного прогнозирования.

Мотивация:
- COP чиллера/CRAC зависит от внешней температуры (как в `DC_digital_twin/src/models/cooling.py`)
- тепловая нагрузка охлаждения зависит от (T_return - setpoint) и скорости вентиляторов
- итоговая электрическая мощность охлаждения = охлаждающая тепловая нагрузка / COP + мощность вентиляторов
"""

from __future__ import annotations

from typing import Literal, Sequence, Tuple, Union

import numpy as np

Number = Union[float, np.ndarray]


def cop_from_curve(cop_curve: Sequence[float], outside_temperature: Number) -> Number:
    """
    COP(T) = a*T^2 + b*T + c, с отсечением снизу (COP >= 1).
    """
    a, b, c = float(cop_curve[0]), float(cop_curve[1]), float(cop_curve[2])
    t = np.asarray(outside_temperature, dtype=np.float32)
    cop = a * t**2 + b * t + c
    cop = np.maximum(cop, 1.0)
    return cop if isinstance(outside_temperature, np.ndarray) else float(cop)


def fan_power(
    fan_max_power: float,
    fan_speed: Number,
    fan_law: Literal["cubic", "linear"] = "cubic",
) -> Number:
    """
    Мощность вентиляторов CRAC.

    В симуляторе:
    - cubic: P_fan = fan_max_power * (fan_speed^3)
    - linear: P_fan = fan_max_power * fan_speed
    """
    s = np.asarray(fan_speed, dtype=np.float32)
    s = np.clip(s, 0.0, 1.0)
    if fan_law == "cubic":
        power = float(fan_max_power) * (s**3)
    else:
        power = float(fan_max_power) * s
    return power if isinstance(fan_speed, np.ndarray) else float(power)


def crac_cooling_load(
    capacity: float,
    return_temperature: Number,
    setpoint: Number,
    fan_speed: Number,
) -> Number:
    """
    Упрощенная тепловая нагрузка охлаждения (Вт), аналог `CRAC.compute_cooling_power`.

    В симуляторе (в `DC_digital_twin/src/models/cooling.py`):
      delta_t = return_temperature - setpoint
      power = min(capacity * (delta_t/5), capacity)
      power *= fan_speed
    """
    rt = np.asarray(return_temperature, dtype=np.float32)
    fs = np.asarray(fan_speed, dtype=np.float32)
    fs = np.clip(fs, 0.0, 1.0)
    sp = np.asarray(setpoint, dtype=np.float32)

    delta_t = rt - sp
    # delta_t <= 0 => 0 Вт (уже достаточно холодно)
    base = np.minimum(float(capacity) * (delta_t / 5.0), float(capacity))
    base = np.where(delta_t <= 0.0, 0.0, base)
    power = base * fs
    is_array = (
        isinstance(return_temperature, np.ndarray)
        or isinstance(fan_speed, np.ndarray)
        or isinstance(setpoint, np.ndarray)
    )
    return power if is_array else float(power)


def pue_physics(
    servers_power: Number,
    return_temperature: Number,
    setpoint: Number,
    fan_speed: Number,
    outside_temperature: Number,
    cop_curve: Sequence[float],
    capacity: float,
    fan_max_power: float,
    fan_law: Literal["cubic", "linear"] = "cubic",
) -> Tuple[Number, Number]:
    """
    Расчет baseline PUE и электрической мощности охлаждения.

    cooling_electric = cooling_load / COP(outside_temperature) + fan_power
    pue_physics = (servers_power + cooling_electric) / servers_power
    """
    sp = np.asarray(servers_power, dtype=np.float32)
    if np.any(sp < 0):
        raise ValueError("servers_power must be non-negative")

    cop = cop_from_curve(cop_curve, outside_temperature)
    cooling_load = crac_cooling_load(capacity, return_temperature, setpoint, fan_speed)
    compressor_power = cooling_load / np.maximum(np.asarray(cop, dtype=np.float32), 1e-6)
    fans_power = fan_power(fan_max_power, fan_speed, fan_law=fan_law)

    cooling_electric = compressor_power + np.asarray(fans_power, dtype=np.float32)

    pue = np.where(sp > 0.0, (sp + cooling_electric) / sp, 1.0).astype(np.float32)

    is_array = (
        isinstance(servers_power, np.ndarray)
        or isinstance(return_temperature, np.ndarray)
        or isinstance(setpoint, np.ndarray)
        or isinstance(fan_speed, np.ndarray)
        or isinstance(outside_temperature, np.ndarray)
    )
    if is_array:
        return pue, cooling_electric
    return float(pue), float(np.asarray(cooling_electric))

