"""
Модель сервера
"""
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ServerConfig:
    """Конфигурация сервера"""
    server_id: int
    p_idle: float  # Вт
    p_max: float  # Вт
    t_max: float  # °C
    c_thermal: float  # Дж/К
    m_dot: float  # кг/с
    position: int  # юнит в стойке
    epsilon: float = 0.8  # эффективность теплообменника
    chip_clip_multiplier: float = 1.2  # верхний клип T_chip как доля от t_max


class Server:
    """Модель сервера"""

    def __init__(self, config: ServerConfig):
        self.config = config
        self.server_id = config.server_id

        # Динамические параметры
        self.utilization = 0.0
        self.t_in = 20.0  # температура на входе
        self.t_out = 25.0  # температура на выходе
        self.t_chip = 25.0  # температура чипа
        self.fan_speed = 0.5  # 0-1

        # Для логирования
        self.history = []

    @property
    def current_power(self) -> float:
        """Текущее потребление (тепловыделение)"""
        return self.config.p_idle + (self.config.p_max - self.config.p_idle) * self.utilization

    def update(self, utilization: float, t_in: float, delta_time: float):
        """
        Обновление состояния сервера

        Args:
            utilization: текущая загрузка (0-1)
            t_in: температура входящего воздуха
            delta_time: шаг по времени (сек)
        """
        self.utilization = np.clip(utilization, 0.0, 1.0)
        self.t_in = t_in

        # Тепловыделение сервера (Вт)
        q_server = self.current_power

        # Теплоемкость воздуха
        cp_air = 1005  # Дж/(кг·К)
        eps = self.config.epsilon

        # Корректная модель теплообменника с тепловой инерцией чипа:
        # T_out = T_in + epsilon * (T_chip - T_in)  — эффективность теплообмена
        # C * dT_chip/dt = P - m_dot * cp * (T_out - T_in)
        #   = P - m_dot * cp * epsilon * (T_chip - T_in)

        # Тепло, отводимое воздухом (Вт)
        heat_to_air = self.config.m_dot * cp_air * eps * (self.t_chip - self.t_in)
        heat_to_air = max(0.0, heat_to_air)  # не может быть отрицательным

        # Явный Эйлер: dT_chip/dt = (P - heat_to_air) / C
        dT_chip_dt = (q_server - heat_to_air) / self.config.c_thermal
        self.t_chip += delta_time * dT_chip_dt

        # Температура на выходе по соотношению теплообменника
        self.t_out = self.t_in + eps * (self.t_chip - self.t_in)

        # Без верхнего клипа: только физическое ограничение T_chip >= T_in.
        self.t_chip = np.maximum(self.t_chip, self.t_in)
        self.t_out = np.maximum(self.t_out, self.t_in)  # T_out >= T_in по 2-му закону

        # Скорость вентилятора зависит от температуры
        self._update_fan_speed()

        # Сохраняем в историю
        self.history.append({
            'time': len(self.history),
            'util': self.utilization,
            't_in': self.t_in,
            't_out': self.t_out,
            't_chip': self.t_chip,
            'power': self.current_power
        })

    def _update_fan_speed(self):
        """Обновление скорости вентилятора в зависимости от температуры"""
        # Простой П-регулятор: увеличиваем обороты при приближении к T_max
        temp_ratio = (self.t_chip - self.t_in) / (self.config.t_max - self.t_in)
        self.fan_speed = np.clip(temp_ratio * 1.2, 0.3, 1.0)

    def get_state(self) -> Dict[str, Any]:
        """Получить текущее состояние"""
        return {
            'server_id': self.server_id,
            'utilization': self.utilization,
            't_in': self.t_in,
            't_out': self.t_out,
            't_chip': self.t_chip,
            'power': self.current_power,
            'fan_speed': self.fan_speed
        }

    def reset(self):
        """Сброс в начальное состояние"""
        self.utilization = 0.0
        self.t_in = 20.0
        self.t_out = 25.0
        self.t_chip = 25.0
        self.fan_speed = 0.5
        self.history = []