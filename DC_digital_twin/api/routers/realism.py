from fastapi import APIRouter, Depends

from api.deps import SimulatorService, get_simulator_service
from api.schemas import RealismModeRequest, RealismParamsRequest

router = APIRouter(prefix="/realism", tags=["realism"])


@router.get("")
async def get_realism(service: SimulatorService = Depends(get_simulator_service)):
    return service.sim.get_realism_state()


@router.post("/mode")
async def set_realism_mode(payload: RealismModeRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_realism_mode(payload.mode)
    return service.sim.get_realism_state()


@router.post("/params")
async def set_realism_params(payload: RealismParamsRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.update_realism_params(
        use_dynamic_crac_power=payload.use_dynamic_crac_power,
        room_temp_clip_min=payload.room_temp_clip_min,
        room_temp_clip_max=payload.room_temp_clip_max,
        chip_temp_clip_multiplier=payload.chip_temp_clip_multiplier,
    )
    return service.sim.get_realism_state()
