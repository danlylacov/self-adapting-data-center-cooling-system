import asyncio
import os
from typing import Any, Dict, Optional

from fastapi import HTTPException

from src.core.simulator import DataCenterSimulator
from src.default_config import get_default_config_copy


class SimulatorService:
    def __init__(self, config_path: Optional[str] = None):
        if config_path and os.path.isfile(config_path):
            self.sim = DataCenterSimulator(config_path=config_path)
        else:
            self.sim = DataCenterSimulator(config=get_default_config_copy())
        self._lock = asyncio.Lock()
        self._task: Optional[asyncio.Task] = None
        self._last_mode: Optional[str] = None
        self._last_error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def status(self) -> Dict[str, Any]:
        return {
            "running": self.running,
            "mode": self._last_mode,
            "step": self.sim.step_count,
            "time": self.sim.sim_time,
            "last_error": self._last_error,
        }

    async def start(self, mode: str, duration: Optional[float], steps: Optional[int]) -> Dict[str, Any]:
        async with self._lock:
            if self.running:
                raise HTTPException(status_code=409, detail="Simulation is already running")

            self._last_mode = mode
            self._last_error = None
            if mode == "realtime":
                if duration is None:
                    raise HTTPException(status_code=400, detail="duration is required for realtime mode")
                self._task = asyncio.create_task(self._run_realtime(duration))
            elif mode == "fast":
                if steps is None:
                    raise HTTPException(status_code=400, detail="steps is required for fast mode")
                self._task = asyncio.create_task(self._run_fast(steps))
            else:
                self._task = None
                self.sim.running = True
            return self.status()

    async def stop(self) -> Dict[str, Any]:
        async with self._lock:
            self.sim.stop()
            if self._task and not self._task.done():
                self._task.cancel()
                try:
                    await self._task
                except asyncio.CancelledError:
                    pass
            self._task = None
            return self.status()

    async def run_steps(self, steps: int, delta_time: Optional[float]) -> Dict[str, Any]:
        async with self._lock:
            if self.running:
                raise HTTPException(status_code=409, detail="Cannot step while realtime/fast run is active")
            for _ in range(steps):
                self.sim.step(delta_time=delta_time)
            return self.sim.get_state()

    async def reset(self, seed: Optional[int]) -> Dict[str, Any]:
        async with self._lock:
            if self.running:
                await self.stop()
            self.sim.reset(seed=seed)
            return self.status()

    async def _run_realtime(self, duration: float):
        try:
            await self.sim.run_realtime(duration=duration)
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
        finally:
            self._task = None

    async def _run_fast(self, steps: int):
        try:
            self.sim.run_fast(steps=steps)
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
        finally:
            self._task = None

    def close(self):
        self.sim.close()

    async def reconfigure(self, config: Dict[str, Any]) -> None:
        """Полная замена конфигурации и пересоздание симулятора."""
        async with self._lock:
            if self.running:
                raise HTTPException(
                    status_code=409,
                    detail="Stop simulation before applying new configuration",
                )
            self._last_error = None
            self._task = None
            self.sim.stop()
            self.sim.close()
            self.sim = DataCenterSimulator(config=config)


CONFIG_PATH = os.getenv("CONFIG_PATH", "") or None
simulator_service = SimulatorService(CONFIG_PATH)


def get_simulator_service() -> SimulatorService:
    return simulator_service
