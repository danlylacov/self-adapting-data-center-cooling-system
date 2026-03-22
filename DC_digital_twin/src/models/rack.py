"""
Модель стойки с серверами
"""
import numpy as np
from typing import List, Dict, Any
from .server import Server, ServerConfig


class Rack:
    """Модель стойки"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: конфигурация стойки из YAML
        """
        self.height = config['height']
        self.width = config['width']
        self.depth = config['depth']
        self.num_units = config['num_units']
        self.containment = config.get('containment', 'none')

        # Серверы в стойке
        self.servers: List[Server] = []

        # Матрица рециркуляции (упрощенная)
        # R[i][j] - какая доля тепла от сервера j попадает на вход сервера i
        self.recirculation_matrix = None

        # Температура подаваемого воздуха
        self.supply_temperature = 20.0

    def add_servers(self, server_configs: List[ServerConfig]):
        """Добавление серверов в стойку"""
        for cfg in server_configs:
            server = Server(cfg)
            self.servers.append(server)

        # Сортируем по позиции
        self.servers.sort(key=lambda s: s.config.position)

        # Инициализация матрицы рециркуляции
        self._init_recirculation_matrix()

    def _init_recirculation_matrix(self):
        """Инициализация упрощенной матрицы рециркуляции"""
        n = len(self.servers)
        self.recirculation_matrix = np.zeros((n, n))

        if self.containment == 'hot_aisle':
            # Горячий коридор: теплый воздух поднимается вверх
            for i in range(n):
                for j in range(i+1, n):  # Серверы выше влияют на те, что ниже?
                    # На самом деле в hot aisle теплый воздух уходит вверх,
                    # поэтому рециркуляция минимальна
                    self.recirculation_matrix[i, j] = 0.01 * (j - i) / n
        elif self.containment == 'cold_aisle':
            # Холодный коридор: холодный воздух внизу, теплый вверху
            for i in range(n):
                for j in range(i):
                    self.recirculation_matrix[i, j] = 0.05 * (i - j) / n
        else:
            # Без изоляции: равномерное перемешивание
            self.recirculation_matrix = 0.02 * np.ones((n, n))
            np.fill_diagonal(self.recirculation_matrix, 0)

    def compute_inlet_temperatures(self) -> np.ndarray:
        """
        Расчет температуры на входе каждого сервера
        с учетом рециркуляции
        """
        n = len(self.servers)
        t_out = np.array([s.t_out for s in self.servers])
        t_supply = self.supply_temperature

        # T_in,i = T_supply + sum_j R_ij * (T_out,j - T_supply)
        t_in = t_supply + np.dot(self.recirculation_matrix, t_out - t_supply)

        # Ограничиваем физически
        t_in = np.maximum(t_in, t_supply - 2)  # Не может быть сильно холоднее подачи
        t_in = np.minimum(t_in, t_supply + 15)  # Не может быть сильно горячее

        return t_in

    def update(self, supply_temperature: float, load_profile: np.ndarray, delta_time: float):
        """
        Обновление состояния всех серверов в стойке

        Args:
            supply_temperature: температура подаваемого воздуха
            load_profile: массив загрузки для каждого сервера
            delta_time: шаг по времени
        """
        self.supply_temperature = supply_temperature

        # Вычисляем температуры на входе с учетом рециркуляции
        inlet_temperatures = self.compute_inlet_temperatures()

        # Обновляем каждый сервер
        for i, server in enumerate(self.servers):
            server.update(
                utilization=load_profile[i] if i < len(load_profile) else 0.5,
                t_in=inlet_temperatures[i],
                delta_time=delta_time
            )

    def get_total_power(self) -> float:
        """Суммарное потребление всех серверов"""
        return sum(s.current_power for s in self.servers)

    def get_total_heat_generation(self) -> float:
        """Суммарное тепловыделение"""
        return self.get_total_power()  # Вся мощность уходит в тепло

    def get_avg_exhaust_temperature(self) -> float:
        """Средняя температура выхлопа"""
        if not self.servers:
            return self.supply_temperature
        return np.mean([s.t_out for s in self.servers])

    def get_state(self) -> Dict[str, Any]:
        """Получить состояние стойки"""
        return {
            'supply_temperature': self.supply_temperature,
            'total_power': self.get_total_power(),
            'avg_exhaust_temp': self.get_avg_exhaust_temperature(),
            'servers': [s.get_state() for s in self.servers]
        }

    def reset(self):
        """Сброс состояния"""
        for server in self.servers:
            server.reset()
        self.supply_temperature = 20.0