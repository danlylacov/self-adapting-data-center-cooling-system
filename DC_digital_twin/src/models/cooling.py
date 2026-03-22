"""
Модели компонентов охлаждения
"""
import numpy as np
from typing import Dict, Any


class CRAC:
    """Computer Room Air Conditioner"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: конфигурация охлаждения
        """
        self.capacity = config['capacity']  # Вт
        self.cop_curve = config.get('cop_curve', [0.002, -0.15, 4.0])  # a, b, c для COP(T)
        self.default_setpoint = config.get('default_setpoint', 22.0)
        self.airflow_m_dot = float(config.get('airflow_m_dot', 1.5))  # кг/с
        self.supply_approach = float(config.get('supply_approach', 1.0))  # C for free-cooling approach

        # Текущее состояние
        self.setpoint = self.default_setpoint
        self.fan_speed = 0.5  # 0-1

        # Параметры вентиляторов
        self.fan_config = config.get('fans', {'max_power': 2000, 'law': 'cubic'})
        self.fan_max_power = self.fan_config['max_power']
        self.fan_law = self.fan_config.get('law', 'cubic')

    def set_setpoint(self, temperature: float):
        """Установка целевой температуры"""
        self.setpoint = np.clip(temperature, 18.0, 27.0)

    def set_fan_speed(self, speed: float):
        """Установка скорости вентилятора (0-1)"""
        self.fan_speed = np.clip(speed, 0.0, 1.0)

    def compute_cop(self, outside_temperature: float) -> float:
        """
        Расчет COP в зависимости от внешней температуры
        COP(T) = a*T² + b*T + c
        """
        a, b, c = self.cop_curve
        cop = a * outside_temperature**2 + b * outside_temperature + c
        return max(cop, 1.0)  # COP не может быть меньше 1

    def compute_cooling_power(self, return_temperature: float) -> float:
        """
        Расчет фактической мощности охлаждения

        Args:
            return_temperature: температура возвращаемого воздуха (из помещения)

        Returns:
            Фактическая мощность охлаждения (Вт)
        """
        # Разница температур
        delta_t = return_temperature - self.setpoint

        if delta_t <= 0:
            # Уже холоднее, чем нужно
            return 0.0

        # Мощность пропорциональна дельте (упрощенно)
        # и ограничена максимальной capacity
        power = min(self.capacity * (delta_t / 5.0), self.capacity)

        # Учитываем скорость вентилятора
        power *= self.fan_speed

        return max(power, 0.0)

    def get_power_consumption(self, outside_temperature: float, return_temperature: float = 25.0) -> float:
        """
        Потребление электроэнергии CRAC

        Args:
            outside_temperature: внешняя температура (влияет на COP)

        Returns:
            Потребляемая мощность (Вт)
        """
        cop = self.compute_cop(outside_temperature)
        cooling_power = self.compute_cooling_power(return_temperature)

        # Мощность компрессора
        compressor_power = cooling_power / cop if cop > 0 else 0

        # Мощность вентилятора (кубическая зависимость)
        if self.fan_law == 'cubic':
            fan_power = self.fan_max_power * (self.fan_speed ** 3)
        else:
            fan_power = self.fan_max_power * self.fan_speed

        return compressor_power + fan_power

    def get_state(self) -> Dict[str, Any]:
        """Получить состояние CRAC"""
        return {
            'setpoint': self.setpoint,
            'fan_speed': self.fan_speed,
            'capacity': self.capacity
        }


class Chiller:
    """Модель чиллера"""

    def __init__(self, config: Dict[str, Any]):
        self.nominal_capacity = config.get('capacity', 50000)  # Вт
        self.cop_curve = config.get('cop_curve', [0.001, -0.1, 3.5])
        self.type = config.get('type', 'air_cooled')

    def compute_cop(self, outside_temperature: float) -> float:
        """Расчет COP"""
        a, b, c = self.cop_curve
        cop = a * outside_temperature**2 + b * outside_temperature + c
        return max(cop, 1.0)

    def get_power_consumption(self, cooling_load: float, outside_temperature: float) -> float:
        """Потребление при заданной нагрузке"""
        cop = self.compute_cop(outside_temperature)
        return cooling_load / cop if cop > 0 else 0


class CoolingSystem:
    """Комплексная система охлаждения"""

    def __init__(self, config: Dict[str, Any]):
        self.crac = CRAC(config.get('crac', {}))
        self.chiller = Chiller(config.get('chiller', {'capacity': 50000}))
        self.mode = str(config.get('mode', 'mixed'))  # free / chiller / mixed
        self.last_cooling_output_w = 0.0
        self.last_supply_temperature = self.crac.setpoint

    def set_mode(self, mode: str):
        """Установка режима работы"""
        if mode in ['free', 'chiller', 'mixed']:
            self.mode = mode

    def _air_side_required_load(self, return_temperature: float, target_supply_temperature: float) -> float:
        cp_air = 1005.0
        delta_t = max(0.0, return_temperature - target_supply_temperature)
        return self.crac.airflow_m_dot * cp_air * delta_t

    def compute_thermal_state(self, return_temperature: float, outside_temperature: float) -> Dict[str, float]:
        """
        Unified thermo model for cooling:
        - computes actual cooling output (W)
        - computes resulting supply temperature (C)
        """
        fan_factor = np.clip(self.crac.fan_speed, 0.0, 1.0)
        available_capacity = self.crac.capacity * fan_factor

        if self.mode == 'free':
            # Free cooling when outside air is colder than return (экономайзер).
            if outside_temperature >= return_temperature:
                # Раньше здесь было Q=0: при жаркой наружке (35 °C) условие T_out >= T_ret
                # остаётся истинным при любой реалистичной T_ret зала — охлаждения нет,
                # тепло от стоек только копится. Нужен чиллер к уставке.
                target_supply = self.crac.setpoint
                q_required = self._air_side_required_load(return_temperature, target_supply)
                q_actual = min(q_required, available_capacity)
                cp_air = 1005.0
                supply_temp = return_temperature - q_actual / max(self.crac.airflow_m_dot * cp_air, 1e-6)
            else:
                target_supply = max(outside_temperature + self.crac.supply_approach, 0.0)
                q_required = self._air_side_required_load(return_temperature, target_supply)
                q_actual = min(q_required, available_capacity)
                cp_air = 1005.0
                supply_temp = return_temperature - q_actual / max(self.crac.airflow_m_dot * cp_air, 1e-6)
        elif self.mode == 'chiller':
            target_supply = self.crac.setpoint
            q_required = self._air_side_required_load(return_temperature, target_supply)
            q_actual = min(q_required, available_capacity)
            cp_air = 1005.0
            supply_temp = return_temperature - q_actual / max(self.crac.airflow_m_dot * cp_air, 1e-6)
        else:
            # mixed: try free-cooling first, then top-up by chiller to setpoint
            target_supply = self.crac.setpoint
            q_required_total = self._air_side_required_load(return_temperature, target_supply)
            free_potential = 0.0
            if outside_temperature < return_temperature:
                free_target_supply = max(outside_temperature + self.crac.supply_approach, 0.0)
                free_potential = self._air_side_required_load(return_temperature, free_target_supply)
            q_actual = min(q_required_total, available_capacity, free_potential + available_capacity)
            cp_air = 1005.0
            supply_temp = return_temperature - q_actual / max(self.crac.airflow_m_dot * cp_air, 1e-6)

        self.last_cooling_output_w = float(max(q_actual, 0.0))
        self.last_supply_temperature = float(supply_temp)
        return {
            "cooling_output_w": self.last_cooling_output_w,
            "supply_temperature": self.last_supply_temperature,
        }

    def compute_total_power(self, return_temperature: float, outside_temperature: float) -> float:
        """Суммарное потребление системы охлаждения"""
        state = self.compute_thermal_state(return_temperature, outside_temperature)
        cooling_load = state["cooling_output_w"]
        if self.mode == 'free':
            fan_power = self.crac.fan_max_power * (self.crac.fan_speed ** 3)
            # Холодная наружка: в основном вентиляторы экономайзера.
            # Жаркая наружка: compute_thermal_state уже считает механику к уставке — учитываем компрессор.
            if outside_temperature >= return_temperature:
                if cooling_load <= 0:
                    return fan_power
                cop = self.crac.compute_cop(outside_temperature)
                return cooling_load / max(cop, 1e-6) + fan_power
            return fan_power
        elif self.mode == 'chiller':
            # Только чиллер
            fan_power = self.crac.fan_max_power * (self.crac.fan_speed ** 3)
            return self.chiller.get_power_consumption(cooling_load, outside_temperature) + fan_power
        else:
            # Смешанный режим
            cop = self.crac.compute_cop(outside_temperature)
            compressor_power = cooling_load / max(cop, 1e-6)
            fan_power = self.crac.fan_max_power * (self.crac.fan_speed ** 3)
            return compressor_power + fan_power

    def get_state(self) -> Dict[str, Any]:
        """Получить состояние системы охлаждения"""
        return {
            'mode': self.mode,
            'crac': self.crac.get_state()
        }