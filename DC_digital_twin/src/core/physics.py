"""
Физические константы и утилиты
"""
import numpy as np


# Константы
CP_AIR = 1005  # Дж/(кг·К) - теплоемкость воздуха
RHO_AIR = 1.2  # кг/м³ - плотность воздуха при 20°C


def compute_air_flow_pressure(flow_rate: float, resistance: float) -> float:
    """
    Расчет давления воздушного потока

    Args:
        flow_rate: расход воздуха (м³/с)
        resistance: аэродинамическое сопротивление

    Returns:
        Давление (Па)
    """
    return resistance * flow_rate ** 2


def compute_fan_power(flow_rate: float, pressure: float, efficiency: float = 0.7) -> float:
    """
    Расчет мощности вентилятора

    Args:
        flow_rate: расход воздуха (м³/с)
        pressure: давление (Па)
        efficiency: КПД вентилятора

    Returns:
        Мощность (Вт)
    """
    return (flow_rate * pressure) / efficiency


def verify_energy_balance(servers_power: float, cooling_power: float,
                          delta_temperature: float, thermal_mass: float,
                          delta_time: float) -> float:
    """
    Проверка закона сохранения энергии

    Returns:
        Относительная погрешность (%)
    """
    # Энергия, выделенная серверами
    energy_in = servers_power * delta_time

    # Энергия, отведенная охлаждением
    energy_out = cooling_power * delta_time

    # Энергия, накопленная в тепловой массе
    energy_stored = thermal_mass * delta_temperature

    # Баланс
    error = abs(energy_in - energy_out - energy_stored) / max(energy_in, 1e-6) * 100

    return error


def compute_thermal_comfort(pmv: float) -> str:
    """
    Оценка теплового комфорта по PMV

    Args:
        pmv: Predicted Mean Vote (-3 до +3)

    Returns:
        Оценка комфорта
    """
    if pmv < -2:
        return "Холодно"
    elif pmv < -1:
        return "Прохладно"
    elif pmv < 1:
        return "Комфортно"
    elif pmv < 2:
        return "Тепло"
    else:
        return "Жарко"