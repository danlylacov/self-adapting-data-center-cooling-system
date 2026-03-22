"""Environment configuration for the orchestrator service."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


@dataclass(frozen=True)
class Settings:
    twin_base: str
    pue_base: str
    temp_base: str
    load_base: str
    ga_base: str
    pue_input_hours: int = 24
    pue_horizon_hours: int = 6
    temp_input_hours: int = 24


def load_settings() -> Settings:
    return Settings(
        twin_base=_env("TWIN_BASE", "http://127.0.0.1:8000").rstrip("/"),
        pue_base=_env("PUE_BASE", "http://127.0.0.1:8011").rstrip("/"),
        temp_base=_env("TEMP_BASE", "http://127.0.0.1:8002").rstrip("/"),
        load_base=_env("LOAD_BASE", "http://127.0.0.1:8010").rstrip("/"),
        ga_base=_env("GA_BASE", "http://127.0.0.1:8013").rstrip("/"),
    )
