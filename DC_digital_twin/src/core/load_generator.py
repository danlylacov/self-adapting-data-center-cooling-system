"""
Генератор нагрузки для серверов
"""
import numpy as np
import pandas as pd
from typing import Optional, List
import random


class LoadGenerator:
    """Генератор нагрузки на серверы"""

    def __init__(self, config: dict, num_servers: int):
        """
        Args:
            config: конфигурация генератора нагрузки
            num_servers: количество серверов
        """
        self.config = config
        self.num_servers = num_servers
        self.type = config.get('type', 'random')
        self.seed = config.get('random_seed', 42)

        # Устанавливаем seed для воспроизводимости
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Для периодического режима
        self.time = 0

        # Загружаем датасет если нужно
        self.dataset = None
        if self.type == 'dataset':
            self._load_dataset()

    def _load_dataset(self):
        """Загрузка датасета с реальной нагрузкой"""
        dataset_path = self.config.get('dataset_path', 'data/sample_load.csv')
        try:
            self.dataset = pd.read_csv(dataset_path)
            print(f"Датасет загружен: {len(self.dataset)} записей")
        except FileNotFoundError:
            print(f"Датасет не найден, использую случайную нагрузку")
            self.type = 'random'

    def set_mode(self, mode: str):
        """Сменить режим генерации нагрузки."""
        if mode not in ['random', 'periodic', 'dataset', 'constant']:
            raise ValueError("mode must be one of: random, periodic, dataset, constant")
        self.type = mode
        if mode == 'dataset' and self.dataset is None:
            self._load_dataset()

    def update_random_params(self, mean_load: float, std_load: float):
        """Обновить параметры random-режима."""
        self.config['mean_load'] = float(mean_load)
        self.config['std_load'] = float(std_load)

    def update_periodic_params(self, day_base: float = 0.7, night_base: float = 0.3):
        """Обновить параметры periodic-режима."""
        self.config['day_base'] = float(day_base)
        self.config['night_base'] = float(night_base)

    def load_dataset_path(self, dataset_path: str):
        """Подменить путь к датасету и перезагрузить его."""
        self.config['dataset_path'] = dataset_path
        self.dataset = None
        self._load_dataset()
        if self.type != 'dataset':
            self.type = 'dataset'

    def generate(self, step: int) -> np.ndarray:
        """
        Генерация нагрузки для всех серверов на текущем шаге

        Args:
            step: номер шага симуляции

        Returns:
            Массив загрузки для каждого сервера (0-1)
        """
        if self.type == 'dataset' and self.dataset is not None:
            return self._generate_from_dataset(step)
        elif self.type == 'constant':
            return self._generate_constant(step)
        elif self.type == 'periodic':
            return self._generate_periodic(step)
        else:
            return self._generate_random(step)

    def _generate_constant(self, step: int) -> np.ndarray:
        """Постоянная нагрузка без тренда и шума."""
        value = float(self.config.get('constant_load', self.config.get('mean_load', 0.6)))
        value = np.clip(value, 0.0, 1.0)
        return np.full(self.num_servers, value, dtype=float)

    def _generate_random(self, step: int) -> np.ndarray:
        """Случайная нагрузка"""
        mean = self.config.get('mean_load', 0.6)
        std = self.config.get('std_load', 0.2)

        # Случайная нагрузка с нормальным распределением
        load = np.random.normal(mean, std, self.num_servers)

        # Ограничиваем
        load = np.clip(load, 0.0, 1.0)

        # Добавляем корреляцию между серверами (общий тренд)
        trend = np.sin(step / 100) * 0.2 + 0.5
        load = load * 0.7 + trend * 0.3

        return load

    def _generate_periodic(self, step: int) -> np.ndarray:
        """Периодическая нагрузка (имитация дня/ночи)"""
        # 24-часовой цикл
        hour = (step / 3600) % 24  # предполагаем шаг 1 сек

        # Дневная активность
        day_base = self.config.get('day_base', 0.7)
        night_base = self.config.get('night_base', 0.3)
        if 8 <= hour <= 20:
            base_load = day_base + 0.2 * np.sin((hour - 8) * np.pi / 12)
        else:
            base_load = night_base + 0.1 * np.sin((hour - 20) * np.pi / 8)

        # Добавляем случайные всплески
        spikes = np.random.random(self.num_servers) < 0.01
        spike_load = np.random.uniform(0.8, 1.0, self.num_servers) * spikes

        # Базовая нагрузка + шум
        load = base_load + np.random.normal(0, 0.05, self.num_servers)
        load = np.clip(load + spike_load, 0.0, 1.0)

        return load

    def _generate_from_dataset(self, step: int) -> np.ndarray:
        """Нагрузка из датасета"""
        idx = step % len(self.dataset)

        # Предполагаем, что в датасете есть колонка 'load'
        if 'load' in self.dataset.columns:
            base_load = self.dataset.iloc[idx]['load']
        else:
            base_load = 0.5

        # Добавляем вариацию между серверами
        load = base_load + np.random.normal(0, 0.1, self.num_servers)
        load = np.clip(load, 0.0, 1.0)

        return load