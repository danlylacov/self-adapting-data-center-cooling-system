"""Полная конфигурация симулятора (замена YAML)."""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends, HTTPException

from api.deps import SimulatorService, get_simulator_service
from src.default_config import get_default_config_copy

router = APIRouter(prefix="/config", tags=["config"])

REQUIRED_SECTIONS = (
    "simulator",
    "realism",
    "rack",
    "servers",
    "cooling",
    "room",
    "load_generator",
    "mqtt",
    "output",
    "logging",
)


def _validate_config(cfg: Dict[str, Any]) -> None:
    if not isinstance(cfg, dict):
        raise HTTPException(status_code=400, detail="Config must be a JSON object")
    for key in REQUIRED_SECTIONS:
        if key not in cfg:
            raise HTTPException(status_code=400, detail=f"Missing section: {key}")
    servers = cfg.get("servers") or {}
    profiles = servers.get("profiles") or {}
    dp = servers.get("default_profile")
    if dp and dp not in profiles:
        raise HTTPException(
            status_code=400,
            detail=f"servers.profiles must include default_profile '{dp}'",
        )


@router.get("/")
async def get_full_config(service: SimulatorService = Depends(get_simulator_service)):
    """Текущая конфигурация симулятора."""
    return deepcopy(service.sim.config)


@router.get("/defaults")
async def get_default_config():
    """Значения по умолчанию (как при первом запуске без CONFIG_PATH)."""
    return get_default_config_copy()


@router.put("/")
async def put_full_config(
    payload: Dict[str, Any] = Body(...),
    service: SimulatorService = Depends(get_simulator_service),
):
    """
    Полная замена конфигурации. Симулятор пересоздаётся (история шагов сбрасывается).
    Нельзя применять во время активного realtime/fast прогона.
    """
    _validate_config(payload)
    await service.reconfigure(payload)
    return {
        "ok": True,
        "name": service.sim.config.get("simulator", {}).get("name"),
        "message": "Configuration applied; simulator reinitialized",
    }
