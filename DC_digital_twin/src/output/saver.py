"""
Модуль для сохранения результатов симуляции
"""
import os
import csv
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
from ..utils.logger import Logger


class ResultSaver:
    """Сохранение результатов симуляции"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: конфигурация вывода
        """
        self.config = config
        self.enabled = config.get('enabled', True)
        self.format = config.get('format', 'csv')
        self.base_path = config.get('path', 'results')
        self.save_interval = config.get('save_interval', 100)
        self.compression = config.get('compression', False)

        # Создаем директорию если нужно
        if self.enabled:
            os.makedirs(self.base_path, exist_ok=True)

        # Буфер для накопления данных
        self.buffer = []
        self.buffer_size = 0

        # Имя файла с timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_name = f"sim_{timestamp}"

        # Файлы для разных типов данных
        self.server_file = os.path.join(self.base_path, f"{self.run_name}_servers.{self.format}")
        self.summary_file = os.path.join(self.base_path, f"{self.run_name}_summary.{self.format}")

        self.logger = Logger()

    def set_base_path(self, base_path: str) -> None:
        """Переопределить каталог вывода (CLI --output); обновляет пути к CSV."""
        self.base_path = base_path
        if self.enabled:
            os.makedirs(self.base_path, exist_ok=True)
        self.server_file = os.path.join(self.base_path, f"{self.run_name}_servers.{self.format}")
        self.summary_file = os.path.join(self.base_path, f"{self.run_name}_summary.{self.format}")

    def add_step(self, step_data: Dict[str, Any]):
        """
        Добавить данные шага в буфер

        Args:
            step_data: данные с get_telemetry()
        """
        if not self.enabled:
            return

        self.buffer.append(step_data)
        self.buffer_size += 1

        # Сохраняем если накопилось достаточно
        if self.buffer_size >= self.save_interval:
            self.flush()

    def flush(self):
        """Сброс буфера в файл"""
        if not self.enabled or not self.buffer:
            return

        try:
            if self.format == 'csv':
                self._save_csv()
            elif self.format == 'json':
                self._save_json()
            elif self.format == 'parquet':
                self._save_parquet()

            self.logger.info(f"Сохранено {self.buffer_size} шагов в {self.base_path}")
            self.buffer = []
            self.buffer_size = 0

        except Exception as e:
            self.logger.error(f"Ошибка сохранения: {e}")

    def _save_csv(self):
        """Сохранение в CSV"""
        # Подготовка данных для серверов (денормализованный формат)
        server_rows = []
        summary_rows = []

        for step in self.buffer:
            timestamp = step['timestamp']
            step_num = step['step']

            # Сводные данные
            summary_rows.append({
                'step': step_num,
                'timestamp': timestamp,
                'room_temperature': step['room']['temperature'],
                'outside_temperature': step['room'].get('outside_temperature', np.nan),
                'pue': step['pue'],
                'cooling_setpoint': step['cooling']['setpoint'],
                'cooling_power': step['cooling']['power_consumption'],
                'cooling_fan_speed': step['cooling']['fan_speed']
            })

            # Данные по каждому серверу
            for server in step['servers']:
                server_rows.append({
                    'step': step_num,
                    'timestamp': timestamp,
                    'server_id': server['server_id'],
                    'utilization': server['utilization'],
                    't_in': server['t_in'],
                    't_out': server['t_out'],
                    't_chip': server['t_chip'],
                    'power': server['power'],
                    'fan_speed': server['fan_speed']
                })

        # Сохраняем сводку
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            # Проверяем, существует ли файл
            if os.path.exists(self.summary_file):
                # Добавляем без заголовка
                df_summary.to_csv(self.summary_file, mode='a', header=False, index=False)
            else:
                df_summary.to_csv(self.summary_file, index=False)

        # Сохраняем данные серверов
        if server_rows:
            df_servers = pd.DataFrame(server_rows)
            if os.path.exists(self.server_file):
                df_servers.to_csv(self.server_file, mode='a', header=False, index=False)
            else:
                df_servers.to_csv(self.server_file, index=False)

    def _save_json(self):
        """Сохранение в JSON"""
        # Для JSON сохраняем каждый шаг отдельным файлом в директории
        step_dir = os.path.join(self.base_path, self.run_name)
        os.makedirs(step_dir, exist_ok=True)

        for step in self.buffer:
            step_file = os.path.join(step_dir, f"step_{step['step']:06d}.json")
            with open(step_file, 'w') as f:
                json.dump(step, f, indent=2, default=self._json_serializer)

    def _save_parquet(self):
        """Сохранение в Parquet (эффективный формат для больших данных)"""
        # Подготовка данных как для CSV
        server_rows = []
        summary_rows = []

        for step in self.buffer:
            timestamp = step['timestamp']
            step_num = step['step']

            summary_rows.append({
                'step': step_num,
                'timestamp': timestamp,
                'room_temperature': step['room']['temperature'],
                'outside_temperature': step['room'].get('outside_temperature', np.nan),
                'pue': step['pue'],
                'cooling_setpoint': step['cooling']['setpoint'],
                'cooling_power': step['cooling']['power_consumption'],
                'cooling_fan_speed': step['cooling']['fan_speed']
            })

            for server in step['servers']:
                server_rows.append({
                    'step': step_num,
                    'timestamp': timestamp,
                    'server_id': server['server_id'],
                    'utilization': server['utilization'],
                    't_in': server['t_in'],
                    't_out': server['t_out'],
                    't_chip': server['t_chip'],
                    'power': server['power'],
                    'fan_speed': server['fan_speed']
                })

        # Сохраняем
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            df_summary.to_parquet(self.summary_file.replace('.parquet', '_summary.parquet'),
                                  compression='snappy' if self.compression else None)

        if server_rows:
            df_servers = pd.DataFrame(server_rows)
            df_servers.to_parquet(self.summary_file.replace('.parquet', '_servers.parquet'),
                                  compression='snappy' if self.compression else None)

    def _json_serializer(self, obj):
        """Сериализатор для JSON (обработка numpy типов)"""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    def close(self):
        """Закрытие и сохранение остатков"""
        self.flush()
        self.logger.info(f"Результаты сохранены в {self.base_path}")