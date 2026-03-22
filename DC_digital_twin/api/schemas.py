from typing import Optional, Literal
from pydantic import BaseModel, Field


class StartSimulationRequest(BaseModel):
    mode: Literal["realtime", "fast", "interactive"] = "realtime"
    duration: Optional[float] = None
    steps: Optional[int] = None


class StepRequest(BaseModel):
    steps: int = Field(default=1, ge=1, le=100000)
    delta_time: Optional[float] = Field(default=None, gt=0, le=300)


class ResetRequest(BaseModel):
    seed: Optional[int] = None


class SetpointRequest(BaseModel):
    temperature: float = Field(ge=18.0, le=27.0)


class FanSpeedRequest(BaseModel):
    speed: float = Field(ge=0.0, le=100.0)


class CoolingModeRequest(BaseModel):
    mode: Literal["free", "chiller", "mixed"]


class TimeFactorRequest(BaseModel):
    value: float = Field(gt=0.0)


class LoadModeRequest(BaseModel):
    mode: Literal["random", "periodic", "dataset", "constant"]


class LoadParamsRequest(BaseModel):
    mean_load: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    std_load: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    day_base: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    night_base: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    constant_load: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class DatasetSelectRequest(BaseModel):
    path: str


class OutsideEnvironmentRequest(BaseModel):
    temperature: float
    humidity: float = Field(default=50.0, ge=0.0, le=100.0)
    wind_speed: float = Field(default=0.0, ge=0.0)


class WeatherModeRequest(BaseModel):
    mode: Literal["manual", "openmeteo", "dataset"]


class RealismModeRequest(BaseModel):
    mode: Literal["realistic"]


class RealismParamsRequest(BaseModel):
    use_dynamic_crac_power: Optional[bool] = None
    room_temp_clip_min: Optional[float] = None
    room_temp_clip_max: Optional[float] = None
    chip_temp_clip_multiplier: Optional[float] = Field(default=None, gt=1.0, le=3.0)
