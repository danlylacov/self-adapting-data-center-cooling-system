from fastapi import APIRouter, Depends

from api.deps import SimulatorService, get_simulator_service
from api.schemas import DatasetSelectRequest, OutsideEnvironmentRequest, WeatherModeRequest

router = APIRouter(prefix="/environment", tags=["environment"])


@router.post("/outside")
async def set_outside(payload: OutsideEnvironmentRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_outside_environment(
        temperature=payload.temperature,
        humidity=payload.humidity,
        wind_speed=payload.wind_speed,
    )
    return service.sim.room.get_state()


@router.post("/weather-mode")
async def set_weather_mode(payload: WeatherModeRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_weather_mode(payload.mode)
    return {"mode": service.sim.weather_mode}


@router.post("/weather-dataset")
async def set_weather_dataset(
    payload: DatasetSelectRequest, service: SimulatorService = Depends(get_simulator_service)
):
    service.sim.set_weather_dataset(payload.path)
    return {"mode": service.sim.weather_mode, "path": payload.path}


@router.get("")
async def get_environment(service: SimulatorService = Depends(get_simulator_service)):
    return {
        "mode": service.sim.weather_mode,
        "room": service.sim.room.get_state(),
    }
