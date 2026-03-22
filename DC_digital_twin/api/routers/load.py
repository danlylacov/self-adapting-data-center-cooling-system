from fastapi import APIRouter, Depends

from api.deps import SimulatorService, get_simulator_service
from api.schemas import DatasetSelectRequest, LoadModeRequest, LoadParamsRequest

router = APIRouter(prefix="/load", tags=["load"])


@router.get("")
async def load_state(service: SimulatorService = Depends(get_simulator_service)):
    return {
        "mode": service.sim.load_generator.type,
        "config": service.sim.load_generator.config,
    }


@router.post("/mode")
async def set_mode(payload: LoadModeRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_load_mode(payload.mode)
    return {"mode": service.sim.load_generator.type}


@router.post("/params")
async def set_params(payload: LoadParamsRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_load_params(
        mean_load=payload.mean_load,
        std_load=payload.std_load,
        day_base=payload.day_base,
        night_base=payload.night_base,
        constant_load=payload.constant_load,
    )
    return {
        "mode": service.sim.load_generator.type,
        "config": service.sim.load_generator.config,
    }


@router.post("/dataset")
async def set_dataset(payload: DatasetSelectRequest, service: SimulatorService = Depends(get_simulator_service)):
    service.sim.set_load_dataset(payload.path)
    return {"mode": service.sim.load_generator.type, "dataset_path": payload.path}
