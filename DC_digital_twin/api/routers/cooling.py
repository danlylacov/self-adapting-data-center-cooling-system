from fastapi import APIRouter, Depends

from api.deps import SimulatorService, get_simulator_service
from api.schemas import CoolingModeRequest, FanSpeedRequest, SetpointRequest

router = APIRouter(prefix="/cooling", tags=["cooling"])


@router.get("")
async def cooling_state(service: SimulatorService = Depends(get_simulator_service)):
    return service.sim.cooling.get_state()


@router.post("/setpoint")
async def set_setpoint(payload: SetpointRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_cooling_setpoint(payload.temperature)
    return service.sim.cooling.get_state()


@router.post("/fanspeed")
async def set_fanspeed(payload: FanSpeedRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_fan_speed(payload.speed)
    return service.sim.cooling.get_state()


@router.post("/mode")
async def set_mode(payload: CoolingModeRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_cooling_mode(payload.mode)
    return service.sim.cooling.get_state()
