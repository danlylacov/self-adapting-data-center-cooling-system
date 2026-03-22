"""HTTP-сервис рекомендаций охлаждения по обученной генетической политике."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# PYTHONPATH должен включать корень v2 (родитель GA), в Docker: /app
from GA.ga_policy import GaCoolingPolicy


def _default_tuned_params_path() -> Path:
    """Рядом с репозиторием: v2/GA/tuned_params.json (локально). В Docker задайте TUNED_PARAMS_PATH=/app/tuned_params.json."""
    return Path(__file__).resolve().parent.parent.parent / "GA" / "tuned_params.json"


TUNED_PARAMS_PATH = os.environ.get("TUNED_PARAMS_PATH") or str(_default_tuned_params_path())

_policy: GaCoolingPolicy | None = None

app = FastAPI(title="predict_ga", version="1.0.0")


def _load_policy() -> GaCoolingPolicy:
    global _policy
    path = Path(TUNED_PARAMS_PATH)
    if not path.is_file():
        raise RuntimeError(f"Tuned params not found: {path.resolve()}")
    _policy = GaCoolingPolicy.from_json_file(path)
    return _policy


@app.on_event("startup")
def startup() -> None:
    _load_policy()


class RecommendRequest(BaseModel):
    avg_chip_temp: float = Field(..., description="Средняя температура чипа, °C")
    setpoint_c: float = Field(..., ge=18.0, le=27.0, description="Текущая уставка подачи CRAC, °C")


class RecommendResponse(BaseModel):
    setpoint_c: float
    fan_speed_pct: float


@app.get("/health")
def health() -> dict:
    try:
        p = _policy or _load_policy()
        return {
            "ok": True,
            "tuned_params_path": str(Path(TUNED_PARAMS_PATH).resolve()),
            "policy": p.to_dict(),
        }
    except Exception as e:
        return {"ok": False, "error": str(e), "tuned_params_path": TUNED_PARAMS_PATH}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(body: RecommendRequest) -> RecommendResponse:
    p = _policy
    if p is None:
        try:
            p = _load_policy()
        except Exception as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
    new_sp, fan = p.recommend(body.avg_chip_temp, body.setpoint_c)
    return RecommendResponse(setpoint_c=new_sp, fan_speed_pct=fan)
