"""
Конфигурация по умолчанию (раньше — config.yaml).
Источник правды для API и CLI, если не задан CONFIG_PATH.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

DEFAULT_CONFIG: Dict[str, Any] = {
    "simulator": {
        "name": "Rack_001",
        "time_step": 1.0,
        "realtime_factor": 10.0,
    },
    "realism": {
        "mode": "realistic",
        "use_dynamic_crac_power": True,
        "room_temp_clip_min": 10.0,
        "room_temp_clip_max": 42.0,
        "chip_temp_clip_multiplier": 1.18,
    },
    "rack": {
        "height": 2.0,
        "width": 0.8,
        "depth": 1.2,
        "num_units": 1,
        "containment": "hot_aisle",
    },
    "servers": {
        "default_profile": "dl380",
        "count": 1,
        "arrangement": "uniform",
        "profiles": {
            "dl380": {
                "p_idle": 120,
                "p_max": 420,
                "t_max": 85,
                "c_thermal": 8000,
                "m_dot": 0.08,
            },
        },
    },
    "cooling": {
        "mode": "mixed",
        "crac": {
            "type": "chilled_water",
            "capacity": 28000,
            "cop_curve": [0.002, -0.15, 4.0],
            "default_setpoint": 22.0,
            "airflow_m_dot": 2.0,
            "supply_approach": 1.2,
        },
        "fans": {
            "max_power": 1600,
            "law": "cubic",
        },
    },
    "room": {
        "volume": 100,
        "wall_heat_transfer": 12,
        "supply_mixing_conductance": 14,
        "enthalpy_mixing_efficiency": 0.2,
        "initial_temperature": 22.0,
        "thermal_mass_factor": 10.0,
    },
    "load_generator": {
        "type": "random",
        "dataset_path": "data/sample_load.csv",
        "random_seed": 42,
        "mean_load": 0.55,
        "std_load": 0.12,
    },
    "mqtt": {
        "enabled": False,
        "broker": "localhost",
        "port": 1883,
        "topic_prefix": "dc/sim/rack_001",
        "publish_rate": 1.0,
        "qos": 1,
    },
    "output": {
        "enabled": True,
        "format": "csv",
        "path": "results",
        "save_interval": 100,
        "compression": True,
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "file": "logs/simulator.log",
    },
    "weather": {
        "enabled": False,
    },
}


def get_default_config_copy() -> Dict[str, Any]:
    return deepcopy(DEFAULT_CONFIG)
