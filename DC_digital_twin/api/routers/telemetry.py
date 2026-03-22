import asyncio
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from api.deps import SimulatorService, get_simulator_service

router = APIRouter(tags=["telemetry"])


@router.get("/state")
async def state_alias(service: SimulatorService = Depends(get_simulator_service)):
    return service.sim.get_state()


@router.get("/telemetry")
async def telemetry_alias(service: SimulatorService = Depends(get_simulator_service)):
    return service.sim.get_telemetry()


@router.websocket("/ws/telemetry")
async def telemetry_ws(websocket: WebSocket):
    await websocket.accept()
    service = get_simulator_service()
    try:
        while True:
            await websocket.send_json(
                {
                    "status": service.status(),
                    "state": service.sim.get_state(),
                    "telemetry": service.sim.get_telemetry(),
                }
            )
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
