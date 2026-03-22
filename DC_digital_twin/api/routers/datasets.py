import os
from pathlib import Path
from typing import List

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from api.deps import SimulatorService, get_simulator_service
from api.schemas import DatasetSelectRequest

router = APIRouter(prefix="/datasets", tags=["datasets"])

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _list_csv_files() -> List[str]:
    return sorted([str(p) for p in UPLOAD_DIR.glob("*.csv")])


@router.get("")
async def list_datasets():
    return {"datasets": _list_csv_files()}


@router.post("/load/upload")
async def upload_load_dataset(file: UploadFile = File(...)):
    target = UPLOAD_DIR / file.filename
    content = await file.read()
    target.write_bytes(content)
    frame = pd.read_csv(target)
    if "load" not in frame.columns:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="load dataset must contain 'load' column")
    return {"path": str(target), "rows": len(frame)}


@router.post("/weather/upload")
async def upload_weather_dataset(file: UploadFile = File(...)):
    target = UPLOAD_DIR / file.filename
    content = await file.read()
    target.write_bytes(content)
    frame = pd.read_csv(target)
    if "outside_temperature" not in frame.columns:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="weather dataset must contain outside_temperature")
    return {"path": str(target), "rows": len(frame)}


@router.post("/load/select")
async def select_load_dataset(
    payload: DatasetSelectRequest, service: SimulatorService = Depends(get_simulator_service)
):
    if not os.path.exists(payload.path):
        raise HTTPException(status_code=404, detail="dataset path does not exist")
    service.sim.set_load_dataset(payload.path)
    return {"path": payload.path, "mode": service.sim.load_generator.type}


@router.post("/weather/select")
async def select_weather_dataset(
    payload: DatasetSelectRequest, service: SimulatorService = Depends(get_simulator_service)
):
    if not os.path.exists(payload.path):
        raise HTTPException(status_code=404, detail="dataset path does not exist")
    service.sim.set_weather_dataset(payload.path)
    return {"path": payload.path, "mode": service.sim.weather_mode}
