"""
Структурированное логирование
"""
import json
import logging
import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional


class JsonFormatter(logging.Formatter):
    """Форматтер для JSON-логов"""

    def format(self, record):
        log_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'message': record.getMessage()
        }

        # Добавляем дополнительные поля если есть
        if hasattr(record, 'data'):
            log_record.update(record.data)

        return json.dumps(log_record, ensure_ascii=False)


class Logger:
    """Обертка для структурированного логирования"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config is None:
            config = {}

        self.level = config.get('level', 'INFO')
        self.format = config.get('format', 'json')
        self.file = config.get('file')

        # Создаем логгер
        self.logger = logging.getLogger('dc_simulator')
        self.logger.setLevel(getattr(logging, self.level))

        # Очищаем существующие handler-ы
        self.logger.handlers = []

        # Добавляем handler для stdout
        console_handler = logging.StreamHandler(sys.stdout)
        if self.format == 'json':
            console_handler.setFormatter(JsonFormatter())
        else:
            console_handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
        self.logger.addHandler(console_handler)

        # Добавляем файловый handler если нужно
        if self.file:
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
            file_handler = logging.FileHandler(self.file)
            if self.format == 'json':
                file_handler.setFormatter(JsonFormatter())
            else:
                file_handler.setFormatter(logging.Formatter(
                    '%(asctime)s - %(levelname)s - %(message)s'
                ))
            self.logger.addHandler(file_handler)

    def _log(self, level: str, message: str, **kwargs):
        """Внутренний метод логирования"""
        level_num = getattr(logging, level, logging.INFO)
        if not self.logger.isEnabledFor(level_num):
            return

        log_record = logging.LogRecord(
            name='dc_simulator',
            level=level_num,
            pathname='',
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )

        # Добавляем дополнительные данные
        if kwargs:
            log_record.data = kwargs

        self.logger.handle(log_record)

    def debug(self, message: str, **kwargs):
        self._log('DEBUG', message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log('INFO', message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log('WARNING', message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log('ERROR', message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log('CRITICAL', message, **kwargs)