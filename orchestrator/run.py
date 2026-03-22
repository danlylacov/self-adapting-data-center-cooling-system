"""Core simulation loop: twin steps, rolling buffers, PUE + load + temp safety."""

from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from orchestrator.clients import HttpError, MlClient, TwinClient
from orchestrator.config import Settings
from orchestrator.run_common import setup_twin_scenario
from orchestrator.temp_policy import adjust_pue_delta_for_chip_temp


DELTA_TIME_MIN = 1
DELTA_TIME_MAX = 300
SETPOINT_MIN_C = 18.0
SETPOINT_MAX_C = 27.0


def _f(v: Any, default: float = 0.0) -> float:
    """Безопасное float: в JSON часто приходит null → float(None) давал бы TypeError и 500."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _apply_fan_safety_from_p_overheat(
    twin: TwinClient,
    *,
    max_p: Optional[float],
    safety_hi: float,
    release_ratio: float,
    fan_boost_speed: float,
    fan0: float,
) -> None:
    """
    Разгон вентилятора при p_overheat выше порога; возврат к сценарной скорости при
    достаточном снижении (гистерезис), чтобы на графиках было видно «пик — max, потом спад».
    """
    if max_p is None:
        return
    rr = clamp(float(release_ratio), 0.0, 0.999)
    release_lo = float(safety_hi) * rr
    if max_p > safety_hi:
        twin.post("/cooling/fanspeed", {"speed": fan_boost_speed})
    elif max_p < release_lo:
        twin.post("/cooling/fanspeed", {"speed": fan0})


def _pad_series(values: List[float], target_len: int) -> List[float]:
    if len(values) >= target_len:
        return values[-target_len:]
    if not values:
        return [0.0] * target_len
    pad = values[0]
    return [pad] * (target_len - len(values)) + values


def _merge_histories(buf: Deque[Dict[str, Any]], n: int) -> Dict[str, List[float]]:
    rows = list(buf)[-n:]
    if len(rows) < n:
        first = rows[0] if rows else {}
        pad = {k: _f(first.get(k), 0.0) for k in (
            "room_temperature",
            "cooling_setpoint",
            "cooling_fan_speed_pct",
            "outside_temperature",
            "humidity",
            "wind_speed",
            "avg_exhaust_temp",
            "servers_power_total",
            "pue_real",
        )}
        while len(rows) < n:
            rows.insert(0, pad)
    out: Dict[str, List[float]] = {}
    for key in (
        "room_temperature",
        "cooling_setpoint",
        "cooling_fan_speed_pct",
        "outside_temperature",
        "humidity",
        "wind_speed",
        "avg_exhaust_temp",
        "servers_power_total",
        "pue_real",
    ):
        out[key] = [_f(r.get(key), 0.0) for r in rows]
    return out


def _sample_pue_row(state: Dict[str, Any], telemetry: Dict[str, Any]) -> Dict[str, Any]:
    cooling = state.get("cooling", {}) or {}
    crac = cooling.get("crac", {}) or {}
    room = state.get("room", {}) or {}
    rack = state.get("rack", {}) or {}
    tel = state.get("telemetry", {}) or {}

    sp_raw = crac.get("setpoint")
    if sp_raw is None:
        sp_raw = cooling.get("setpoint", 22.0)
    setpoint = _f(sp_raw, 22.0)
    fs = crac.get("fan_speed")
    if fs is None:
        fs = 0.5
    fan_pct = _f(cooling.get("fan_speed_pct"), _f(fs, 0.5) * 100.0)

    pue_raw = tel.get("pue_real")
    if pue_raw is None:
        pue_raw = telemetry.get("pue", 1.0)
    pue_real = _f(pue_raw, 1.0)

    return {
        "room_temperature": _f(room.get("temperature"), 0.0),
        "cooling_setpoint": setpoint,
        "cooling_fan_speed_pct": fan_pct,
        "outside_temperature": _f(tel.get("outside_temperature"), _f(room.get("outside_temperature"), 20.0)),
        "humidity": _f(tel.get("humidity"), 50.0),
        "wind_speed": _f(tel.get("wind_speed"), 0.0),
        "avg_exhaust_temp": _f(tel.get("avg_exhaust_temp"), _f(rack.get("avg_exhaust_temp"), 0.0)),
        "servers_power_total": _f(rack.get("total_power"), 0.0),
        "pue_real": pue_real,
    }


def _server_feature_vector(
    state: Dict[str, Any],
    server: Dict[str, Any],
    server_index: int,
    num_servers: int,
) -> List[float]:
    cooling = state.get("cooling", {}) or {}
    crac = cooling.get("crac", {}) or {}
    room = state.get("room", {}) or {}
    tel = state.get("telemetry", {}) or {}

    sim_time = _f(state.get("time"), 0.0)
    hour = (sim_time / 3600.0) % 24.0

    pos = float(server_index / max(num_servers - 1, 1)) if num_servers > 1 else 0.0

    return [
        _f(server.get("utilization"), 0.0),
        _f(server.get("t_chip"), 0.0),
        _f(server.get("t_in"), 0.0),
        _f(crac.get("setpoint"), 22.0),
        _f(server.get("fan_speed"), 0.5),
        _f(server.get("power"), 0.0),
        _f(room.get("outside_temperature"), 20.0),
        _f(tel.get("humidity"), 50.0),
        pos,
        math.sin(2.0 * math.pi * hour / 24.0),
        math.cos(2.0 * math.pi * hour / 24.0),
    ]


def _build_X_from_buffers(
    buffers: List[Deque[List[float]]],
    input_hours: int,
) -> List[List[List[float]]]:
    """Shape [servers, input_hours, 11]."""
    X: List[List[List[float]]] = []
    for buf in buffers:
        seq = list(buf)
        if len(seq) < input_hours:
            pad = seq[0] if seq else [0.0] * 11
            seq = [pad] * (input_hours - len(seq)) + seq
        else:
            seq = seq[-input_hours:]
        X.append(seq)
    return X


def _max_p_overheat(resp: Dict[str, Any]) -> float:
    p = resp.get("p_overheat") or []
    if not p:
        return 0.0
    vals: List[float] = []
    for row in p:
        if row:
            vals.append(max(_f(x, 0.0) for x in row))
    return max(vals) if vals else 0.0


def execute_run(
    *,
    twin: TwinClient,
    pue: MlClient,
    temp: MlClient,
    load: MlClient,
    settings: Settings,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    scenario = payload.get("scenario") or {}
    realism = payload.get("realism") or {}

    steps = int(scenario.get("steps", 360))
    delta_time = float(scenario.get("deltaTime", 10))
    delta_time = clamp(delta_time, DELTA_TIME_MIN, DELTA_TIME_MAX)

    setpoint0 = float(scenario.get("setpoint", 22))
    fan0 = float(scenario.get("fanSpeed", 65))
    cooling_mode = str(scenario.get("coolingMode", "mixed"))
    mean_load = float(scenario.get("meanLoad", 0.55))
    std_load = float(scenario.get("stdLoad", 0.12))
    outside_temp = float(scenario.get("outsideTemp", 24))
    use_dataset = bool(scenario.get("useDatasetLoad", False))
    load_dataset_path = scenario.get("loadDatasetPath") or ""

    control_interval = max(1, int(payload.get("controlIntervalSteps", 5)))
    safety_max_p = float(payload.get("safetyMaxPOverheat", 0.2))
    load_model = str(payload.get("loadForecastModel", "prophet"))
    if load_model not in ("prophet", "deepar"):
        load_model = "prophet"
    load_every_ticks = max(1, int(payload.get("loadForecastEveryNTicks", 1)))
    fan_boost_speed = float(payload.get("fanBoostSpeed", 100.0))
    safety_release_ratio = float(payload.get("safetyReleaseRatio", 0.68))
    seed = payload.get("seed")

    delta_grid = payload.get("deltaGridC")
    if delta_grid is not None and not isinstance(delta_grid, list):
        delta_grid = None

    fail_on_ml = bool(payload.get("failOnMlUnavailable", False))

    temp_aware_pue = bool(payload.get("tempAwarePue", True))
    chip_temp_target_c = float(payload.get("chipTempTargetC", 62.0))
    chip_temp_deadband_c = float(payload.get("chipTempDeadbandC", 3.0))

    points: List[Dict[str, Any]] = []
    pue_delta_recommended: Optional[float] = None
    pue_delta_raw_last: Optional[float] = None
    max_p_overheat_last: Optional[float] = None
    ml_fallback_used = False
    last_forecast_anchor: Optional[float] = None

    hist_buf: Deque[Dict[str, Any]] = deque(maxlen=settings.pue_input_hours)
    server_buffers: List[Deque[List[float]]] = []
    load_tick_counter = 0
    cached_forecast: Optional[List[Dict[str, Any]]] = None

    setup_twin_scenario(
        twin,
        setpoint0=setpoint0,
        fan0=fan0,
        cooling_mode=cooling_mode,
        mean_load=mean_load,
        std_load=std_load,
        outside_temp=outside_temp,
        use_dataset=use_dataset,
        load_dataset_path=load_dataset_path,
        realism=realism,
        seed=int(seed) if seed is not None else None,
    )

    for step_idx in range(1, steps + 1):
        twin.post("/simulation/step", {"steps": 1, "delta_time": delta_time})
        state = twin.get("/simulation/state")
        telemetry = twin.get("/simulation/telemetry")

        hist_buf.append(_sample_pue_row(state, telemetry))

        servers = (state.get("rack") or {}).get("servers") or []
        if not server_buffers or len(server_buffers) != len(servers):
            server_buffers = [deque(maxlen=settings.temp_input_hours) for _ in range(len(servers))]
        for i, srv in enumerate(servers):
            server_buffers[i].append(_server_feature_vector(state, srv, i, len(servers)))

        crac_live = ((state.get("cooling") or {}).get("crac") or {}) if state else {}
        applied_sp = _f(crac_live.get("setpoint"), setpoint0)
        applied_fan = _f((telemetry.get("cooling") or {}).get("fan_speed"), fan0)

        tel_s = state.get("telemetry") or {}
        room_s = state.get("room") or {}
        cool_t = telemetry.get("cooling") or {}
        point_row: Dict[str, Any] = {
            "step": state.get("step", step_idx),
            "room": _f(room_s.get("temperature"), 0.0),
            "outside": _f(room_s.get("outside_temperature"), 0.0),
            "pue": _f(telemetry.get("pue"), 0.0),
            "avgChip": _f(tel_s.get("avg_chip_temp"), 0.0),
            "overheatRisk": _f(tel_s.get("overheat_risk"), 0.0) * 100.0,
            "coolingPower": _f(cool_t.get("power_consumption"), 0.0),
            "totalPowerKw": _f(tel_s.get("total_power_kw"), 0.0),
            "appliedSetpoint": applied_sp,
            "appliedFanSpeed": applied_fan,
        }
        points.append(point_row)

        # Control tick: after recording this step, optionally run ML
        if step_idx % control_interval != 0:
            continue

        hist_merged = _merge_histories(hist_buf, settings.pue_input_hours)

        # Load forecast (optional)
        forecast_rows: List[Dict[str, Any]] = []
        if load_tick_counter % load_every_ticks == 0:
            try:
                load_resp = load.get(
                    "/forecast",
                    params={"model_type": load_model, "horizon_hours": 24},
                )
                forecast_rows = list(load_resp.get("forecast") or [])
                cached_forecast = forecast_rows
            except HttpError:
                if fail_on_ml:
                    raise
                ml_fallback_used = True
                forecast_rows = cached_forecast or []
        else:
            forecast_rows = cached_forecast or []
        load_tick_counter += 1

        anchor_power = _f((state.get("rack") or {}).get("total_power"), 0.0) or 1.0
        last_forecast_anchor = anchor_power

        # First 6 hours mapped to servers_power_total (shape relative to anchor)
        yhats: List[float] = []
        for row in forecast_rows[: settings.pue_horizon_hours]:
            y = row.get("yhat_mean")
            if y is None:
                y = row.get("yhat")
            yhats.append(_f(y, anchor_power))
        if len(yhats) < settings.pue_horizon_hours:
            yhats = _pad_series(yhats, settings.pue_horizon_hours)
        mean_y = sum(yhats) / max(len(yhats), 1)
        mean_y = mean_y if abs(mean_y) > 1e-9 else 1.0
        future_power = [anchor_power * (y / mean_y) for y in yhats]
        # Без прогноза нагрузки (все нули) или нулевая мощность — плоский ряд в якоре, иначе PUE/физика может дать ошибку
        if not future_power or max(abs(x) for x in future_power) < 1e-6:
            future_power = [float(anchor_power)] * settings.pue_horizon_hours

        tel = state.get("telemetry") or {}
        room = state.get("room") or {}
        outside_f = _f(tel.get("outside_temperature"), _f(room.get("outside_temperature"), outside_temp))
        exhaust_f = _f(tel.get("avg_exhaust_temp"), _f((state.get("rack") or {}).get("avg_exhaust_temp"), 25.0))

        current_sp = _f(((state.get("cooling") or {}).get("crac") or {}).get("setpoint"), applied_sp)
        current_fan_pct = _f((telemetry.get("cooling") or {}).get("fan_speed"), applied_fan)

        future_base = {
            "outside_temperature": [outside_f] * settings.pue_horizon_hours,
            "avg_exhaust_temp": [exhaust_f] * settings.pue_horizon_hours,
            "servers_power_total": future_power,
            "cooling_setpoint": [current_sp] * settings.pue_horizon_hours,
            "cooling_fan_speed_pct": [current_fan_pct] * settings.pue_horizon_hours,
        }

        pue_body = {
            "history": hist_merged,
            "future": future_base,
        }
        if delta_grid is not None:
            pue_body["delta_grid_c"] = delta_grid

        tick_delta: Optional[float] = None
        tick_delta_raw: Optional[float] = None
        try:
            pue_resp = pue.post("/pue/hybrid/recommend", pue_body)
            best = pue_resp.get("best") or {}
            tick_delta_raw = _f(best.get("delta_c"), 0.0)
            avg_chip = _f(tel.get("avg_chip_temp"), 70.0)
            tick_delta = adjust_pue_delta_for_chip_temp(
                tick_delta_raw,
                avg_chip,
                target_c=chip_temp_target_c,
                deadband_c=chip_temp_deadband_c,
                enabled=temp_aware_pue,
            )
            pue_delta_recommended = tick_delta
            pue_delta_raw_last = tick_delta_raw
            new_sp = clamp(current_sp + tick_delta, SETPOINT_MIN_C, SETPOINT_MAX_C)
            twin.post("/cooling/setpoint", {"temperature": new_sp})
        except HttpError:
            if fail_on_ml:
                raise
            ml_fallback_used = True

        # Temp predict (uses buffers up to current step)
        max_p_overheat_last = None
        if server_buffers:
            X = _build_X_from_buffers(server_buffers, settings.temp_input_hours)
            try:
                temp_resp = temp.post("/predict", {"X": X})
                max_p_overheat_last = _max_p_overheat(temp_resp)
                _apply_fan_safety_from_p_overheat(
                    twin,
                    max_p=max_p_overheat_last,
                    safety_hi=safety_max_p,
                    release_ratio=safety_release_ratio,
                    fan_boost_speed=fan_boost_speed,
                    fan0=fan0,
                )
            except HttpError:
                if fail_on_ml:
                    raise
                ml_fallback_used = True
                max_p_overheat_last = None

        st_after = twin.get("/simulation/state")
        tel_after = twin.get("/simulation/telemetry")
        points[-1]["pueDeltaRecommended"] = tick_delta
        points[-1]["pueDeltaRaw"] = tick_delta_raw
        points[-1]["pueDeltaApplied"] = tick_delta
        points[-1]["maxPOverheat"] = max_p_overheat_last
        st_cool = (st_after.get("cooling") or {}) if st_after else {}
        st_crac = st_cool.get("crac") or {}
        points[-1]["appliedSetpoint"] = _f(st_crac.get("setpoint"), applied_sp)
        points[-1]["appliedFanSpeed"] = _f((tel_after.get("cooling") or {}).get("fan_speed"), applied_fan)

    # Final state for aggregates
    final_state = twin.get("/simulation/state")

    return {
        "points": points,
        "finalState": final_state,
        "meta": {
            "pueDeltaRecommended": pue_delta_recommended,
            "pueDeltaRaw": pue_delta_raw_last,
            "pueDeltaApplied": pue_delta_recommended,
            "maxPOverheat": max_p_overheat_last,
            "mlFallbackUsed": ml_fallback_used,
            "forecastAnchorPowerW": last_forecast_anchor,
            "tempAwarePue": temp_aware_pue,
        },
    }
