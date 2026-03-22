from fastapi import APIRouter, Depends

from api.deps import SimulatorService, get_simulator_service
from api.schemas import ResetRequest, StartSimulationRequest, StepRequest, TimeFactorRequest

router = APIRouter(prefix="/simulation", tags=["simulation"])


@router.post("/start")
async def start_simulation(
    payload: StartSimulationRequest, service: SimulatorService = Depends(get_simulator_service)
):
    return await service.start(mode=payload.mode, duration=payload.duration, steps=payload.steps)


@router.post("/stop")
async def stop_simulation(service: SimulatorService = Depends(get_simulator_service)):
    return await service.stop()


@router.post("/step")
async def step_simulation(payload: StepRequest, service: SimulatorService = Depends(get_simulator_service)):
    return await service.run_steps(steps=payload.steps, delta_time=payload.delta_time)


@router.get("/status")
async def status(service: SimulatorService = Depends(get_simulator_service)):
    return service.status()


@router.post("/reset")
async def reset(payload: ResetRequest, service: SimulatorService = Depends(get_simulator_service)):
    return await service.reset(seed=payload.seed)


@router.get("/state")
async def state(service: SimulatorService = Depends(get_simulator_service)):
    return service.sim.get_state()


@router.get("/telemetry")
async def telemetry(service: SimulatorService = Depends(get_simulator_service)):
    return service.sim.get_telemetry()


@router.get("/time-factor")
async def get_time_factor(service: SimulatorService = Depends(get_simulator_service)):
    return {"value": service.sim.realtime_factor}


@router.post("/time-factor")
async def set_time_factor(payload: TimeFactorRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_realtime_factor(payload.value)
    return {"value": service.sim.realtime_factor}
