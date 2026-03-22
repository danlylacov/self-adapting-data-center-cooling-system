"""FastAPI entry: health + POST /run."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from orchestrator.clients import HttpError, MlClient, TwinClient
from orchestrator.config import load_settings
from orchestrator.run import execute_run
from orchestrator.run_ga import execute_run_ga


class ScenarioModel(BaseModel):
    steps: int = Field(ge=1, le=100_000)
    deltaTime: float = Field(ge=1, le=300)
    setpoint: float = Field(default=22, ge=18, le=27)
    fanSpeed: float = Field(default=65, ge=0, le=100)
    coolingMode: Literal["free", "chiller", "mixed"] = "mixed"
    meanLoad: float = 0.55
    stdLoad: float = 0.12
    outsideTemp: float = 24.0
    useDatasetLoad: bool = False
    loadDatasetPath: Optional[str] = None


class RealismModel(BaseModel):
    mode: str = "realistic"
    use_dynamic_crac_power: bool = True
    room_temp_clip_min: float = 10.0
    room_temp_clip_max: float = 42.0
    chip_temp_clip_multiplier: float = 1.18


class RunRequest(BaseModel):
    scenario: ScenarioModel
    realism: RealismModel = Field(default_factory=RealismModel)
    controlIntervalSteps: int = Field(default=5, ge=1, le=10_000)
    safetyMaxPOverheat: float = Field(default=0.2, ge=0.0, le=1.0)
    loadForecastModel: Literal["prophet", "deepar"] = "prophet"
    loadForecastEveryNTicks: int = Field(default=1, ge=1, le=1000)
    fanBoostSpeed: float = Field(default=100.0, ge=0.0, le=100.0)
    safetyReleaseRatio: float = Field(
        default=0.68,
        ge=0.0,
        le=0.999,
        description="Доля от safetyMaxPOverheat: ниже — возврат вентилятора к сценарной скорости (гистерезис).",
    )
    seed: Optional[int] = None
    deltaGridC: Optional[List[float]] = None
    failOnMlUnavailable: bool = False
    tempAwarePue: bool = True
    chipTempTargetC: float = Field(default=62.0, ge=30.0, le=95.0)
    chipTempDeadbandC: float = Field(default=3.0, ge=0.1, le=20.0)


class RunGaRequest(BaseModel):
    """Прогон с GA по уставке; вентилятор как в POST /run (сценарий + boost по temp ML)."""

    scenario: ScenarioModel
    realism: RealismModel = Field(default_factory=RealismModel)
    controlIntervalSteps: int = Field(default=5, ge=1, le=10_000)
    safetyMaxPOverheat: float = Field(default=0.2, ge=0.0, le=1.0)
    fanBoostSpeed: float = Field(default=100.0, ge=0.0, le=100.0)
    safetyReleaseRatio: float = Field(
        default=0.68,
        ge=0.0,
        le=0.999,
        description="Гистерезис возврата fan к сценарной скорости (как в POST /run).",
    )
    seed: Optional[int] = None
    failOnGaUnavailable: bool = False
    failOnMlUnavailable: bool = False
    gaSetpointMaxC: Optional[float] = Field(
        default=None,
        ge=18.0,
        le=27.0,
        description="Верхняя граница уставки после GA: не даём политике «разогревать» до 27 °C ради низкого PUE.",
    )
    gaSetpointBiasC: float = Field(
        default=0.0,
        ge=-3.0,
        le=3.0,
        description="Сдвиг уставки в сторону охлаждения (вычитается из ответа GA).",
    )
    gaOverrideSetpointC: Optional[float] = Field(
        default=None,
        ge=18.0,
        le=27.0,
        description="Если задано — игнорировать уставку из GA и всегда подавать это значение (демо «консервативного» канала).",
    )
    gaMinFanSpeedPct: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Нижняя граница скорости вентилятора после safety (поднимает PUE при откате).",
    )


app = FastAPI(title="DC Orchestrator", version="1.0.0")

# allow_credentials=False — можно allow_origins=["*"] (удобно для деплоя за любым хостом/портом).
# Для dev (Vite) и прод (nginx на :8080 и т.д.) достаточно wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, Any]:
    settings = load_settings()
    twin_ok = False
    twin_err: Optional[str] = None
    ml: Dict[str, Any] = {}
    try:
        tc = TwinClient(settings.twin_base)
        try:
            tc.health()
            twin_ok = True
        finally:
            tc.close()
    except Exception as e:
        twin_ok = False
        twin_err = str(e)

    for name, base in (
        ("pue", settings.pue_base),
        ("temp", settings.temp_base),
        ("load", settings.load_base),
    ):
        try:
            c = MlClient(base)
            try:
                ml[name] = c.health()
            finally:
                c.close()
        except Exception as e:
            ml[name] = {"error": str(e)}

    ga_health: Dict[str, Any] = {}
    try:
        gc = MlClient(settings.ga_base)
        try:
            ga_health = gc.health()
        finally:
            gc.close()
    except Exception as e:
        ga_health = {"error": str(e)}

    status = "ok" if twin_ok else "degraded"
    out: Dict[str, Any] = {"status": status, "twin": twin_ok, "ml": ml, "ga": ga_health}
    if twin_err:
        out["twin_error"] = twin_err
    return out


@app.post("/run")
def run(payload: RunRequest) -> Dict[str, Any]:
    settings = load_settings()
    twin = TwinClient(settings.twin_base)
    pue = MlClient(settings.pue_base)
    temp = MlClient(settings.temp_base)
    load = MlClient(settings.load_base)
    try:
        body: Dict[str, Any] = payload.model_dump()
        result = execute_run(
            twin=twin,
            pue=pue,
            temp=temp,
            load=load,
            settings=settings,
            payload=body,
        )
        return result
    except HttpError as e:
        if e.status_code == 503:
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        # Любая другая ошибка (TypeError, KeyError, …) — отдаём текст в detail, чтобы было видно в UI
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    finally:
        twin.close()
        pue.close()
        temp.close()
        load.close()


@app.post("/run/ga")
def run_ga_endpoint(payload: RunGaRequest) -> Dict[str, Any]:
    settings = load_settings()
    twin = TwinClient(settings.twin_base)
    ga = MlClient(settings.ga_base)
    temp = MlClient(settings.temp_base)
    try:
        body: Dict[str, Any] = payload.model_dump()
        result = execute_run_ga(
            twin=twin,
            ga=ga,
            temp=temp,
            settings=settings,
            payload=body,
        )
        return result
    except HttpError as e:
        if e.status_code == 503:
            raise HTTPException(status_code=503, detail=str(e)) from e
        raise HTTPException(status_code=502, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {e}") from e
    finally:
        twin.close()
        ga.close()
        temp.close()
