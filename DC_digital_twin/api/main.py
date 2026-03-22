from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.deps import get_simulator_service
from api.routers import cooling, datasets, environment, full_config, load, realism, simulation, telemetry


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ = get_simulator_service()
    yield
    get_simulator_service().close()


app = FastAPI(title="DC Digital Twin API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(simulation.router)
app.include_router(cooling.router)
app.include_router(load.router)
app.include_router(environment.router)
app.include_router(datasets.router)
app.include_router(telemetry.router)
app.include_router(realism.router)
app.include_router(full_config.router)


@app.get("/health")
async def health():
    return {"ok": True}
