"""
Модель помещения
"""
import numpy as np
from typing import Dict, Any


class Room:
    """Модель помещения с тепловой инерцией и возвратным воздухом"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: конфигурация помещения
        """
        self.volume = config['volume']  # м³
        # Теплопотери через ограждение (Вт/К). Для изолированной серверной — порядка 5–20, не 50+.
        self.wall_heat_transfer = float(config.get('wall_heat_transfer', 12.0))
        # Слабый теплообмен объёма зала с подачей CRAC (Вт/К); не путать с полной Q_coil.
        self.supply_mixing_conductance = float(config.get('supply_mixing_conductance', 14.0))
        # Доля «полного» enthalpy-потока m_dot*cp*ΔT, идущего в баланс объёма зала (0..1).
        # Без этого слабый только G*ΔT не согласуется с реальной мощностью CRAC по воздуху.
        self.enthalpy_mixing_efficiency = float(config.get('enthalpy_mixing_efficiency', 0.2))
        self.initial_temperature = config.get('initial_temperature', 22.0)
        self.thermal_mass_factor = config.get('thermal_mass_factor', 20.0)
        self.temp_clip_min = config.get('temp_clip_min', 10.0)
        self.temp_clip_max = config.get('temp_clip_max', 40.0)

        # Текущая температура объема зала
        self.temperature = self.initial_temperature
        # Температура возвратного воздуха в CRAC (обычно выше средней по залу)
        self.return_temperature = self.initial_temperature

        # Внешняя температура (из API)
        self.outside_temperature = 20.0
        # Внешняя влажность (для ML)
        self.humidity = 50.0
        # Скорость ветра (для ML)
        self.wind_speed = 0.0

        # Теплоемкость воздуха в помещении
        # Cp_air ≈ 1005 Дж/(кг·К), плотность ≈ 1.2 кг/м³
        self.thermal_capacity = self.volume * 1.2 * 1005 * self.thermal_mass_factor  # Дж/К

        # История
        self.history = []

    def update(
        self,
        heat_from_racks: float,
        cooling_power: float,
        delta_time: float,
        avg_exhaust_temperature: float,
        airflow_m_dot: float,
        supply_temperature: float,
    ):
        """
        Обновление температуры в помещении

        Args:
            heat_from_racks: тепло от стоек (Вт)
            cooling_power: мощность катушки из контура охлаждения (Вт); в балансе объёма зала не вычитается —
                эффект уже в supply_temperature; поле сохраняется в историю для отладки
            delta_time: шаг по времени (сек)
            avg_exhaust_temperature: средняя температура выхлопа стойки (C)
            airflow_m_dot: массовый расход воздуха через CRAC (кг/с)
            supply_temperature: температура подачи в стойку (C)
        """
        # Теплообмен со стенами (снаружи)
        wall_heat = self.wall_heat_transfer * (self.outside_temperature - self.temperature)

        # Обмен с подачей CRAC: линейный член (Вт/К) + доля enthalpy-потока m_dot*cp*ΔT.
        # Нельзя дополнительно вычитать cooling_power (Q_coil): эта мощность уже «зашита» в
        # рассчитанную T_supply в cooling.compute_thermal_state; повторное −Q_coil в балансе
        # объёма зала даёт двойной учёт охлаждения и уводит T_room в отрицательные значения.
        cp_air = 1005.0
        d_supply = float(supply_temperature) - self.temperature
        mix_w_per_k = self.supply_mixing_conductance + self.enthalpy_mixing_efficiency * max(
            0.0, float(airflow_m_dot)
        ) * cp_air
        q_supply_mix = mix_w_per_k * d_supply

        net_heat = heat_from_racks + wall_heat + q_supply_mix
        dT = net_heat * delta_time / self.thermal_capacity
        self.temperature += dT
        self.temperature = float(np.clip(self.temperature, self.temp_clip_min, self.temp_clip_max))

        # Отдельно моделируем return air в CRAC как смесь hot aisle и общего зала.
        # Это убирает визуально «жесткую пропорциональность» room<->chip.
        cp_air = 1005.0
        room_air_mass = max(self.volume * 1.2, 1e-6)
        exchanged_mass = max(0.0, airflow_m_dot) * max(delta_time, 0.0)
        exchange_ratio = float(np.clip(exchanged_mass / room_air_mass, 0.0, 1.0))

        target_return = 0.65 * float(avg_exhaust_temperature) + 0.35 * float(self.temperature)
        self.return_temperature += exchange_ratio * (target_return - self.return_temperature)

        # Сохраняем в историю
        self.history.append({
            'time': len(self.history),
            'temperature': self.temperature,
            'return_temperature': self.return_temperature,
            'heat_from_racks': heat_from_racks,
            'cooling_power': cooling_power,
            'q_supply_mix_w': q_supply_mix,
            'net_heat_w': net_heat,
            'mix_w_per_k': mix_w_per_k,
        })

    def set_outside_temperature(self, temperature: float):
        """Установка внешней температуры"""
        self.outside_temperature = temperature

    def set_outside_humidity(self, humidity: float):
        """Установка внешней влажности (0-100, проценты)"""
        self.humidity = humidity

    def set_wind_speed(self, wind_speed: float):
        """Установка скорости ветра (м/с)"""
        self.wind_speed = wind_speed

    def get_state(self) -> Dict[str, Any]:
        """Получить состояние помещения"""
        return {
            'temperature': self.temperature,
            'outside_temperature': self.outside_temperature,
            'humidity': self.humidity,
            'wind_speed': self.wind_speed,
            'thermal_capacity': self.thermal_capacity,
            'return_temperature': self.return_temperature
        }

    def reset(self):
        """Сброс в начальное состояние"""
        self.temperature = self.initial_temperature
        self.return_temperature = self.initial_temperature
        self.outside_temperature = 20.0
        self.history = []