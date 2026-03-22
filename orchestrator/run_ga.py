"""Прогон двойника с управлением через GA-сервис (уставка); вентилятор как в ML-прогоне."""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, List, Optional

from orchestrator.clients import HttpError, MlClient, TwinClient
from orchestrator.config import Settings
from orchestrator.run import (
    DELTA_TIME_MAX,
    DELTA_TIME_MIN,
    _apply_fan_safety_from_p_overheat,
    _build_X_from_buffers,
    _f,
    _max_p_overheat,
    _server_feature_vector,
    clamp,
)
from orchestrator.run_common import setup_twin_scenario


def execute_run_ga(
    *,
    twin: TwinClient,
    ga: MlClient,
    temp: MlClient,
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
    safety_release_ratio = float(payload.get("safetyReleaseRatio", 0.68))
    fan_boost_speed = float(payload.get("fanBoostSpeed", 100.0))
    seed = payload.get("seed")
    fail_on_ga = bool(payload.get("failOnGaUnavailable", False))
    fail_on_ml = bool(payload.get("failOnMlUnavailable", False))
    ga_setpoint_max_c = payload.get("gaSetpointMaxC")
    ga_setpoint_bias_c = float(payload.get("gaSetpointBiasC", 0.0))
    ga_override_setpoint_c = payload.get("gaOverrideSetpointC")
    ga_min_fan_speed_pct = payload.get("gaMinFanSpeedPct")

    points: list[Dict[str, Any]] = []
    ga_fallback_used = False
    temp_fallback_used = False
    ga_setpoint_last: Optional[float] = None
    ga_fan_last: Optional[float] = None
    server_buffers: List[Deque[List[float]]] = []

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

        if step_idx % control_interval != 0:
            continue

        tel = state.get("telemetry") or {}
        avg_chip = _f(tel.get("avg_chip_temp"), 70.0)
        current_sp = _f(((state.get("cooling") or {}).get("crac") or {}).get("setpoint"), applied_sp)

        tick_ga_sp: Optional[float] = None
        tick_ga_fan: Optional[float] = None
        ga_raw_setpoint: Optional[float] = None
        try:
            rec = ga.post(
                "/recommend",
                {"avg_chip_temp": avg_chip, "setpoint_c": current_sp},
            )
            ga_raw_setpoint = _f(rec.get("setpoint_c"), current_sp)
            tick_ga_fan = _f(rec.get("fan_speed_pct"), applied_fan)
            if ga_override_setpoint_c is not None:
                tick_ga_sp = clamp(float(ga_override_setpoint_c), 18.0, 27.0)
            else:
                tick_ga_sp = ga_raw_setpoint
                tick_ga_sp -= ga_setpoint_bias_c
                if ga_setpoint_max_c is not None:
                    tick_ga_sp = min(tick_ga_sp, float(ga_setpoint_max_c))
                tick_ga_sp = clamp(tick_ga_sp, 18.0, 27.0)
            twin.post("/cooling/setpoint", {"temperature": tick_ga_sp})
            # Рекомендацию GA по вентилятору не применяем — как в ML: сценарный fan,
            # разгон только при p_overheat > safetyMaxPOverheat (см. ниже).
            ga_setpoint_last = tick_ga_sp
            ga_fan_last = tick_ga_fan
        except HttpError:
            if fail_on_ga:
                raise
            ga_fallback_used = True

        max_p_overheat_last: Optional[float] = None
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
                temp_fallback_used = True
                max_p_overheat_last = None

        if ga_min_fan_speed_pct is not None:
            tel_fan = twin.get("/simulation/telemetry")
            cur_fan = _f((tel_fan.get("cooling") or {}).get("fan_speed"), fan0)
            lo = float(ga_min_fan_speed_pct)
            if cur_fan < lo:
                twin.post("/cooling/fanspeed", {"speed": lo})

        st_after = twin.get("/simulation/state")
        tel_after = twin.get("/simulation/telemetry")
        points[-1]["gaPolicySetpointRaw"] = ga_raw_setpoint
        points[-1]["gaSetpointRecommended"] = tick_ga_sp
        points[-1]["gaFanRecommended"] = tick_ga_fan
        points[-1]["maxPOverheat"] = max_p_overheat_last
        st_cool = (st_after.get("cooling") or {}) if st_after else {}
        st_crac = st_cool.get("crac") or {}
        points[-1]["appliedSetpoint"] = _f(st_crac.get("setpoint"), applied_sp)
        points[-1]["appliedFanSpeed"] = _f((tel_after.get("cooling") or {}).get("fan_speed"), applied_fan)

    final_state = twin.get("/simulation/state")

    return {
        "points": points,
        "finalState": final_state,
        "meta": {
            "gaFallbackUsed": ga_fallback_used,
            "tempFallbackUsed": temp_fallback_used,
            "gaSetpointRecommended": ga_setpoint_last,
            "gaFanRecommended": ga_fan_last,
            "gaFanApplied": False,
            "fanControlLikeMl": True,
            "safetyMaxPOverheat": safety_max_p,
            "fanBoostSpeed": fan_boost_speed,
            "controlMode": "ga",
            "gaOverrideSetpointC": ga_override_setpoint_c,
            "gaMinFanSpeedPct": ga_min_fan_speed_pct,
        },
    }
