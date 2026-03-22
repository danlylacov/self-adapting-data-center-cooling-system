"""
Главный класс симулятора
"""
import asyncio
import copy
import time
import math
import yaml
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import json
import urllib.parse
import urllib.request
import pandas as pd

from ..models.server import ServerConfig
from ..models.rack import Rack
from ..models.room import Room
from ..models.cooling import CoolingSystem
from .load_generator import LoadGenerator
from ..utils.logger import Logger
from ..mqtt.client import MQTTClient
from ..output.saver import ResultSaver


class DataCenterSimulator:
    """Главный класс симулятора ЦОД"""

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Инициализация симулятора

        Args:
            config_path: путь к YAML (опционально)
            config: словарь конфигурации (опционально; приоритет над файлом)
        """
        if config is not None:
            self.config = copy.deepcopy(config)
        elif config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            from ..default_config import get_default_config_copy

            self.config = get_default_config_copy()

        # Инициализируем логгер
        self.logger = Logger(self.config.get('logging', {}))

        # Параметры симуляции
        self.time_step = self.config['simulator']['time_step']
        self.realtime_factor = self.config['simulator'].get('realtime_factor', 1.0)
        self.sim_time = 0.0
        self.step_count = 0
        self.running = False
        self.realism_cfg: Dict[str, Any] = self.config.get('realism', {})
        # Always realistic mode.
        self.realism_mode: str = "realistic"
        self.use_dynamic_crac_power: bool = bool(
            self.realism_cfg.get("use_dynamic_crac_power", True)
        )

        # Инициализация компонентов
        self._init_components()

        # MQTT клиент
        self.mqtt_enabled = self.config.get('mqtt', {}).get('enabled', False)
        self.mqtt_client = None
        if self.mqtt_enabled:
            self.mqtt_client = MQTTClient(self.config['mqtt'])

        # История состояний
        self.state_history = []

        self.logger.info("Симулятор инициализирован",
                         name=self.config['simulator']['name'])

        self.result_saver = None
        if self.config.get('output', {}).get('enabled', False):
            self.result_saver = ResultSaver(self.config['output'])

        # Weather profile (optional): used to vary `room.outside_temperature` over time.
        self.weather_cfg: Dict[str, Any] = self.config.get('weather', {})
        self.weather_enabled: bool = bool(self.weather_cfg.get('enabled', False))
        self._weather_outside_temps: Optional[np.ndarray] = None
        self._weather_humidity: Optional[np.ndarray] = None
        self._weather_wind_speeds: Optional[np.ndarray] = None
        self._weather_total_hours: Optional[int] = None
        self.weather_mode: str = "openmeteo" if self.weather_enabled else "manual"
        self._weather_dataset: Optional[pd.DataFrame] = None
        self._last_return_temperature: float = 25.0

    def _init_components(self):
        """Инициализация всех компонентов"""

        # Создаем конфигурации серверов
        server_configs = []
        server_profile = self.config['servers']['profiles'][
            self.config['servers']['default_profile']
        ]

        chip_clip_multiplier = float(self.realism_cfg.get('chip_temp_clip_multiplier', 10.0))
        for i in range(self.config['servers']['count']):
            # Позиция в стойке (юнит)
            position = i + 1

            cfg = ServerConfig(
                server_id=i,
                p_idle=server_profile['p_idle'],
                p_max=server_profile['p_max'],
                t_max=server_profile['t_max'],
                c_thermal=server_profile['c_thermal'],
                m_dot=server_profile['m_dot'],
                position=position,
                chip_clip_multiplier=chip_clip_multiplier,
            )
            server_configs.append(cfg)

        # Создаем стойку
        self.rack = Rack(self.config['rack'])
        self.rack.add_servers(server_configs)

        # Создаем помещение
        room_cfg = dict(self.config['room'])
        room_cfg['temp_clip_min'] = float(self.realism_cfg.get('room_temp_clip_min', -273.15))
        room_cfg['temp_clip_max'] = float(self.realism_cfg.get('room_temp_clip_max', 1e6))
        self.room = Room(room_cfg)

        # Создаем систему охлаждения
        self.cooling = CoolingSystem(self.config['cooling'])

        # Создаем генератор нагрузки
        self.load_generator = LoadGenerator(
            self.config['load_generator'],
            len(self.rack.servers)
        )

    def _fetch_openmeteo_hourly_profile(self, total_hours: int) -> Dict[str, np.ndarray]:
        """
        Fetch outside-air profile from Open-Meteo archive (hourly grid).

        The simulator always needs outside temperature for the physics model.
        ML models additionally use humidity and wind speed if configured.
        Falls back to constants on any network/parse error.
        """
        base_date = str(self.weather_cfg.get('base_date', '2019-05-01'))
        lat = float(self.weather_cfg.get('lat', 55.7558))
        lon = float(self.weather_cfg.get('lon', 37.6173))

        try:
            base = datetime.strptime(base_date, "%Y-%m-%d")
        except ValueError:
            base = datetime(2019, 5, 1)

        days_needed = int(math.ceil(total_hours / 24.0))
        end_date = (base + timedelta(days=max(0, days_needed - 1))).strftime("%Y-%m-%d")

        variables_physics = self.weather_cfg.get('variables_physics', {}) or {}
        variables_ml = self.weather_cfg.get('variables_ml', {}) or {}

        temp_var = "temperature_2m"
        humidity_var = None
        wind_var = None

        if isinstance(variables_physics, dict) and variables_physics.get("temperature_2m") is True:
            temp_var = "temperature_2m"

        if isinstance(variables_ml, dict):
            humidity_var = variables_ml.get("humidity")  # e.g. "relative_humidity_2m"
            wind_var = variables_ml.get("wind_speed")  # e.g. "wind_speed_10m"

        hourly_vars: list[str] = [temp_var]
        if humidity_var:
            hourly_vars.append(str(humidity_var))
        if wind_var:
            hourly_vars.append(str(wind_var))

        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": base.strftime("%Y-%m-%d"),
            "end_date": end_date,
            "hourly": ",".join(hourly_vars),
            "timezone": str(self.weather_cfg.get('timezone', 'Europe/Moscow')),
        }

        query = urllib.parse.urlencode(params)
        request = urllib.request.Request(url=f"{url}?{query}", method="GET")
        try:
            with urllib.request.urlopen(request, timeout=60) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            self.logger.warning("Open-Meteo fetch failed; fallback to constants", error=str(e))
            outside_temp = np.full(total_hours, 20.0, dtype=np.float32)
            humidity = np.full(total_hours, 50.0, dtype=np.float32)
            wind_speed = np.full(total_hours, 0.0, dtype=np.float32)
            return {"outside_temperature": outside_temp, "humidity": humidity, "wind_speed": wind_speed}

        def _read_series(var_name: str, fallback_value: float) -> np.ndarray:
            series = payload.get("hourly", {}).get(var_name, [])
            if not isinstance(series, list) or len(series) < total_hours:
                if isinstance(series, list) and len(series) > 0:
                    padded = series + [series[-1]] * (total_hours - len(series))
                    return np.array(padded[:total_hours], dtype=np.float32)
                return np.full(total_hours, fallback_value, dtype=np.float32)
            return np.array(series[:total_hours], dtype=np.float32)

        outside_temp = _read_series(temp_var, 20.0)
        humidity = _read_series(humidity_var, 50.0) if humidity_var else np.full(total_hours, 50.0, dtype=np.float32)
        wind_speed = _read_series(wind_var, 0.0) if wind_var else np.full(total_hours, 0.0, dtype=np.float32)

        return {"outside_temperature": outside_temp, "humidity": humidity, "wind_speed": wind_speed}

    def _ensure_weather_loaded(self, total_steps: int) -> None:
        """Load a weather profile once for this run duration."""
        if not self.weather_enabled:
            return

        last_step = max(0, total_steps - 1)
        last_hour_idx = int((last_step * self.time_step) // 3600)
        total_hours_needed = last_hour_idx + 1

        if self._weather_outside_temps is not None and self._weather_total_hours == total_hours_needed:
            return

        profile = self._fetch_openmeteo_hourly_profile(total_hours_needed)
        self._weather_outside_temps = profile["outside_temperature"]
        self._weather_humidity = profile["humidity"]
        self._weather_wind_speeds = profile["wind_speed"]
        self._weather_total_hours = total_hours_needed

    def _apply_weather_for_current_step(self) -> None:
        """Set room outside temperature for current `step_count`."""
        if self.weather_mode == "manual":
            return
        if self.weather_mode == "dataset":
            if self._weather_dataset is None or self._weather_dataset.empty:
                return
            idx = self.step_count % len(self._weather_dataset)
            row = self._weather_dataset.iloc[idx]
            self.room.set_outside_temperature(float(row.get("outside_temperature", 20.0)))
            self.room.set_outside_humidity(float(row.get("humidity", 50.0)))
            self.room.set_wind_speed(float(row.get("wind_speed", 0.0)))
            return
        if not self.weather_enabled:
            return

        # In interactive mode we might not have preloaded weather yet.
        hour_idx = int((self.step_count * self.time_step) // 3600)

        if self._weather_outside_temps is None:
            # Bootstrap profile for the first hour that the simulation is about to enter.
            self._ensure_weather_loaded(total_steps=hour_idx + 2)

        if self._weather_outside_temps is None:
            return

        if hour_idx >= len(self._weather_outside_temps):
            # Interactive mode might run longer than the initial prefetch window;
            # extend the profile when needed.
            steps_needed = int(((hour_idx + 1) * 3600.0) / self.time_step) + 2
            self._ensure_weather_loaded(total_steps=steps_needed)

        if self._weather_outside_temps is None:
            return

        if 0 <= hour_idx < len(self._weather_outside_temps):
            self.room.set_outside_temperature(float(self._weather_outside_temps[hour_idx]))
            if self._weather_humidity is not None and 0 <= hour_idx < len(self._weather_humidity):
                self.room.set_outside_humidity(float(self._weather_humidity[hour_idx]))
            if self._weather_wind_speeds is not None and 0 <= hour_idx < len(self._weather_wind_speeds):
                self.room.set_wind_speed(float(self._weather_wind_speeds[hour_idx]))

    def set_realtime_factor(self, value: float):
        """Установить ускорение realtime-режима."""
        if value <= 0:
            raise ValueError("realtime_factor must be > 0")
        self.realtime_factor = float(value)

    def set_load_mode(self, mode: str):
        """Переключить источник нагрузки."""
        self.load_generator.set_mode(mode)

    def set_load_params(
        self,
        mean_load: Optional[float] = None,
        std_load: Optional[float] = None,
        day_base: Optional[float] = None,
        night_base: Optional[float] = None,
        constant_load: Optional[float] = None,
    ):
        """Обновить параметры random/periodic генератора."""
        if mean_load is not None or std_load is not None:
            mean_value = self.load_generator.config.get("mean_load", 0.6) if mean_load is None else mean_load
            std_value = self.load_generator.config.get("std_load", 0.2) if std_load is None else std_load
            self.load_generator.update_random_params(float(mean_value), float(std_value))
        if day_base is not None or night_base is not None:
            day_value = self.load_generator.config.get("day_base", 0.7) if day_base is None else day_base
            night_value = self.load_generator.config.get("night_base", 0.3) if night_base is None else night_base
            self.load_generator.update_periodic_params(float(day_value), float(night_value))
        if constant_load is not None:
            self.load_generator.config["constant_load"] = float(constant_load)

    def set_load_dataset(self, dataset_path: str):
        """Переключить генератор на dataset из указанного пути."""
        self.load_generator.load_dataset_path(dataset_path)

    def set_outside_environment(self, temperature: float, humidity: float = 50.0, wind_speed: float = 0.0):
        """Ручная установка внешней среды."""
        self.weather_mode = "manual"
        self.room.set_outside_temperature(float(temperature))
        self.room.set_outside_humidity(float(humidity))
        self.room.set_wind_speed(float(wind_speed))

    def set_weather_mode(self, mode: str):
        """Переключить режим внешней погоды: manual/openmeteo/dataset."""
        if mode not in {"manual", "openmeteo", "dataset"}:
            raise ValueError("weather mode must be manual/openmeteo/dataset")
        self.weather_mode = mode

    def set_weather_dataset(self, dataset_path: str):
        """Загрузить погодный датасет и активировать dataset-режим."""
        frame = pd.read_csv(dataset_path)
        required = {"outside_temperature"}
        if not required.issubset(set(frame.columns)):
            raise ValueError("weather dataset must contain outside_temperature column")
        if "humidity" not in frame.columns:
            frame["humidity"] = 50.0
        if "wind_speed" not in frame.columns:
            frame["wind_speed"] = 0.0
        self._weather_dataset = frame
        self.weather_mode = "dataset"

    def get_realism_state(self) -> Dict[str, Any]:
        """Текущие параметры режима реалистичности."""
        if self.rack.servers:
            chip_clip_multiplier = self.rack.servers[0].config.chip_clip_multiplier
        else:
            chip_clip_multiplier = float(self.realism_cfg.get("chip_temp_clip_multiplier", 1.2))
        return {
            "mode": self.realism_mode,
            "use_dynamic_crac_power": self.use_dynamic_crac_power,
            "room_temp_clip_min": float(self.room.temp_clip_min),
            "room_temp_clip_max": float(self.room.temp_clip_max),
            "chip_temp_clip_multiplier": float(chip_clip_multiplier),
        }

    def set_realism_mode(self, mode: str):
        """Переключить пресет реализма demo/realistic."""
        if mode != "realistic":
            raise ValueError("Only realistic mode is supported")
        self.realism_mode = "realistic"
        self.use_dynamic_crac_power = True
        self.room.temp_clip_min = -273.15
        self.room.temp_clip_max = 1e6
        chip_clip_multiplier = 10.0
        for server in self.rack.servers:
            server.config.chip_clip_multiplier = chip_clip_multiplier

    def update_realism_params(
        self,
        use_dynamic_crac_power: Optional[bool] = None,
        room_temp_clip_min: Optional[float] = None,
        room_temp_clip_max: Optional[float] = None,
        chip_temp_clip_multiplier: Optional[float] = None,
    ):
        """Точечное обновление параметров реалистичности."""
        if use_dynamic_crac_power is not None:
            self.use_dynamic_crac_power = bool(use_dynamic_crac_power)
        if room_temp_clip_min is not None:
            self.room.temp_clip_min = float(room_temp_clip_min)
        if room_temp_clip_max is not None:
            self.room.temp_clip_max = float(room_temp_clip_max)
        if (
            self.room.temp_clip_min is not None
            and self.room.temp_clip_max is not None
            and self.room.temp_clip_min >= self.room.temp_clip_max
        ):
            raise ValueError("room_temp_clip_min must be lower than room_temp_clip_max")
        if chip_temp_clip_multiplier is not None:
            value = float(chip_temp_clip_multiplier)
            for server in self.rack.servers:
                server.config.chip_clip_multiplier = value

    def set_cooling_setpoint(self, temperature: float):
        """Установка температуры подаваемого воздуха"""
        self.cooling.crac.set_setpoint(temperature)
        self.logger.debug("Setpoint изменен", setpoint=temperature)

    def set_fan_speed(self, speed: float):
        """Установка скорости вентиляторов CRAC (0-100%)"""
        self.cooling.crac.set_fan_speed(speed / 100.0)

    def set_cooling_mode(self, mode: str):
        """Выбор режима: 'free', 'chiller', 'mixed'"""
        self.cooling.set_mode(mode)

    def _effective_return_temperature(self) -> float:
        """
        Температура воздуха для расчёта Q_coil и подачи CRAC.

        В коде заданы два состояния: фильтрованный return_temperature (медленно тянется
        к смеси выхлопа и зала) и температура объёма зала. Пока return < setpoint,
        chiller даёт Q=0, хотя зал уже греется от IT — зал тогда только растёт.
        Берём max(...) как нижнюю оценку температуры воздуха, попадающего в контур CRAC.
        """
        return float(max(self.room.return_temperature, self.room.temperature))

    def step(self, delta_time: Optional[float] = None):
        """
        Выполнить один шаг симуляции

        Args:
            delta_time: шаг по времени (сек), если None - используем time_step
        """
        if delta_time is None:
            delta_time = self.time_step

        # Update outside air temperature for this simulated time index.
        self._apply_weather_for_current_step()

        # Генерируем нагрузку
        load_profile = self.load_generator.generate(self.step_count)

        # Обновляем стойку (см. _effective_return_temperature — согласование с теплом зала)
        thermal_state = self.cooling.compute_thermal_state(
            return_temperature=self._effective_return_temperature(),
            outside_temperature=self.room.outside_temperature,
        )
        supply_temperature = thermal_state["supply_temperature"]
        self.rack.update(
            supply_temperature=supply_temperature,
            load_profile=load_profile,
            delta_time=delta_time
        )

        heat_from_racks = self.rack.get_total_heat_generation()
        avg_exhaust = self.rack.get_avg_exhaust_temperature()

        # Вычисляем мощность охлаждения: return temp = средняя температура выхлопа стойки
        # (физически корректнее, чем room.temperature — воздух из горячего коридора)
        # Cooling coil return is room air, not raw rack exhaust.
        cooling_power = thermal_state["cooling_output_w"]
        self._last_return_temperature = self.room.return_temperature

        # Обновляем помещение
        self.room.update(
            heat_from_racks=heat_from_racks,
            cooling_power=cooling_power,
            delta_time=delta_time,
            avg_exhaust_temperature=avg_exhaust,
            airflow_m_dot=self.cooling.crac.airflow_m_dot,
            supply_temperature=float(thermal_state["supply_temperature"]),
        )

        # Обновляем время
        self.sim_time += delta_time
        self.step_count += 1

        # Сохраняем состояние
        state = self.get_state()
        self.state_history.append(state)

        # Сохраняем результаты если нужно
        if self.result_saver:
            telemetry = self.get_telemetry()
            self.result_saver.add_step(telemetry)

        # Публикуем в MQTT если нужно
        if self.mqtt_enabled and self.mqtt_client:
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self._publish_telemetry())
            except RuntimeError:
                # No running event loop (e.g. sync execution path). Skip MQTT publish.
                self.logger.debug("Skip MQTT publish: no running event loop")

    async def _publish_telemetry(self):
        """Публикация телеметрии в MQTT"""
        if not self.mqtt_client:
            return

        telemetry = self.get_telemetry()

        # Публикуем данные серверов
        for server in telemetry['servers']:
            server_id = server['server_id']
            await self.mqtt_client.publish(
                f"server/{server_id}/temperature/inlet",
                server['t_in']
            )
            await self.mqtt_client.publish(
                f"server/{server_id}/temperature/outlet",
                server['t_out']
            )
            await self.mqtt_client.publish(
                f"server/{server_id}/temperature/chip",
                server['t_chip']
            )
            await self.mqtt_client.publish(
                f"server/{server_id}/power",
                server['power']
            )
            await self.mqtt_client.publish(
                f"server/{server_id}/utilization",
                server['utilization']
            )

        # Публикуем данные охлаждения
        await self.mqtt_client.publish(
            "cooling/supply_temp",
            telemetry['cooling']['supply_temperature']
        )
        await self.mqtt_client.publish(
            "cooling/power",
            telemetry['cooling']['power_consumption']
        )

        # Публикуем PUE
        await self.mqtt_client.publish(
            "pue",
            telemetry['pue']
        )

        # Публикуем статус
        await self.mqtt_client.publish(
            "status",
            {
                'state': 'running',
                'step': self.step_count,
                'time': self.sim_time
            }
        )

    async def run_realtime(self, duration: float):
        """
        Запуск симуляции в реальном времени

        Args:
            duration: длительность симуляции (секунд симулированных)
        """
        self.running = True
        steps = int(duration / self.time_step)

        # One-time weather fetch for this run duration.
        self._ensure_weather_loaded(steps)

        # Подключаемся к MQTT если нужно
        if self.mqtt_enabled and self.mqtt_client:
            await self.mqtt_client.connect()

        self.logger.info("Запуск симуляции", duration=duration, steps=steps)

        for _ in range(steps):
            if not self.running:
                break

            start_time = time.time()

            # Шаг симуляции
            self.step()

            # Ждем для синхронизации с реальным временем
            elapsed = time.time() - start_time
            sleep_time = self.time_step / self.realtime_factor - elapsed

            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        self.logger.info("Симуляция завершена", steps=self.step_count)

        # Отключаем MQTT
        if self.mqtt_enabled and self.mqtt_client:
            await self.mqtt_client.disconnect()

    def run_fast(self, steps: int):
        """
        Запуск симуляции в ускоренном режиме (без реального времени)

        Args:
            steps: количество шагов
        """
        self.logger.info("Запуск ускоренной симуляции", steps=steps)

        # One-time weather fetch for this run duration.
        self._ensure_weather_loaded(steps)

        for _ in range(steps):
            self.step()

        self.logger.info("Ускоренная симуляция завершена")

    def get_state(self) -> Dict[str, Any]:
        """Получить текущее состояние всех компонентов"""
        rack_state = self.rack.get_state()
        room_state = self.room.get_state()
        cooling_state = self.cooling.get_state()

        # Enriched telemetry-like block for orchestration control-loop.
        servers = rack_state.get("servers", [])
        if servers:
            avg_chip_temp = float(np.mean([s.get("t_chip", 0.0) for s in servers]))
            avg_inlet_temp = float(np.mean([s.get("t_in", 0.0) for s in servers]))
            overheat_threshold_c = float(self.config.get("overheat_threshold_c", 75.0))
            overheat_risk = float(np.mean([s.get("t_chip", 0.0) > overheat_threshold_c for s in servers]))
        else:
            avg_chip_temp = 70.0
            avg_inlet_temp = 24.0
            overheat_risk = 0.0

        total_power_kw = float(rack_state.get("total_power", 0.0) / 1000.0)

        # Compute PUE to support PUE recommender integration.
        servers_power_w = float(rack_state.get("total_power", 0.0))
        outside_temperature = float(room_state.get("outside_temperature", 20.0))
        cooling_power_w = float(
            self.cooling.compute_total_power(
                return_temperature=self._effective_return_temperature(),
                outside_temperature=outside_temperature,
            )
        )
        pue_real = (servers_power_w + cooling_power_w) / servers_power_w if servers_power_w > 0 else 1.0

        # Shape cooling to what orchestrator expects.
        crac_state = cooling_state.get("crac", {})
        cooling_state["setpoint"] = float(crac_state.get("setpoint", cooling_state.get("setpoint", 22.0)))
        cooling_state["fan_speed_pct"] = float(crac_state.get("fan_speed", 0.0) * 100.0)

        return {
            'time': self.sim_time,
            'step': self.step_count,
            'rack': rack_state,
            'room': room_state,
            'cooling': cooling_state,
            'telemetry': {
                'timestamp': datetime.now().timestamp(),
                'step': self.step_count,
                'total_power_kw': total_power_kw,
                'avg_chip_temp': avg_chip_temp,
                'avg_inlet_temp': avg_inlet_temp,
                'overheat_risk': overheat_risk,
                'outside_temperature': float(room_state.get("outside_temperature", 20.0)),
                'humidity': float(room_state.get("humidity", 50.0)),
                'wind_speed': float(room_state.get("wind_speed", 0.0)),
                'avg_exhaust_temp': float(rack_state.get("avg_exhaust_temp", avg_inlet_temp)),
                'delta_time_sec': float(self.time_step),
                'pue_real': float(pue_real),
            },
            'control': {
                'realtime_factor': float(self.realtime_factor),
                'load_mode': self.load_generator.type,
                'weather_mode': self.weather_mode,
            },
        }

    def get_telemetry(self) -> Dict[str, Any]:
        """
        Получить телеметрию (только датчики) для публикации
        """
        rack_state = self.rack.get_state()

        # Вычисляем PUE
        servers_power = rack_state['total_power']
        cooling_power = self.cooling.compute_total_power(
            return_temperature=self._effective_return_temperature(),
            outside_temperature=self.room.outside_temperature,
        )
        pue = (servers_power + cooling_power) / servers_power if servers_power > 0 else 1.0

        return {
            'timestamp': datetime.now().timestamp(),
            'step': self.step_count,
            'servers': rack_state['servers'],
            'cooling': {
                'supply_temperature': rack_state['supply_temperature'],
                'power_consumption': cooling_power,
                'setpoint': self.cooling.crac.setpoint,
                'fan_speed': self.cooling.crac.fan_speed * 100
            },
            'room': {
                'temperature': self.room.temperature,
                'outside_temperature': self.room.outside_temperature
            },
            'pue': pue
        }

    def reset(self, seed: Optional[int] = None):
        """Сброс симулятора в начальное состояние"""
        self.sim_time = 0.0
        self.step_count = 0
        self.state_history = []

        self.rack.reset()
        self.room.reset()
        self._last_return_temperature = self.room.return_temperature

        # Переустанавливаем seed если нужно
        if seed is not None:
            np.random.seed(seed)

        self.logger.info("Симулятор сброшен")

    def stop(self):
        """Остановка симуляции"""
        self.running = False

    def close(self):
        """Закрытие симулятора и сохранение результатов"""
        if self.result_saver:
            self.result_saver.close()
        if self.mqtt_enabled and self.mqtt_client:
            try:
                asyncio.get_running_loop()
                asyncio.create_task(self.mqtt_client.disconnect())
            except RuntimeError:
                # Called outside of an event loop (e.g. some sync CLI paths).
                try:
                    asyncio.run(self.mqtt_client.disconnect())
                except Exception as exc:  # noqa: BLE001
                    self.logger.debug("MQTT disconnect failed in fallback", error=str(exc))