"""
MQTT клиент для публикации телеметрии
"""
import asyncio
import json
import aiomqtt
from typing import Any, Dict, Optional
from ..utils.logger import Logger


class MQTTClient:
    """Асинхронный MQTT клиент"""

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: конфигурация MQTT
        """
        self.broker = config['broker']
        self.port = config.get('port', 1883)
        self.topic_prefix = config.get('topic_prefix', 'dc/sim')
        self.qos = config.get('qos', 1)
        self.publish_rate = config.get('publish_rate', 1.0)

        self.client = None
        self.logger = Logger()
        self.connected = False

    async def connect(self):
        """Подключение к MQTT брокеру"""
        try:
            self.client = aiomqtt.Client(self.broker, self.port)
            await self.client.connect()
            self.connected = True
            self.logger.info("MQTT подключен", broker=self.broker, port=self.port)
        except Exception as e:
            self.logger.error("Ошибка подключения MQTT", error=str(e))
            self.connected = False

    async def disconnect(self):
        """Отключение от MQTT брокера"""
        if self.client and self.connected:
            await self.client.disconnect()
            self.connected = False
            self.logger.info("MQTT отключен")

    async def publish(self, topic: str, payload: Any):
        """
        Публикация сообщения

        Args:
            topic: топик (относительный)
            payload: данные
        """
        if not self.connected:
            return

        full_topic = f"{self.topic_prefix}/{topic}"

        # Конвертируем в JSON если нужно
        if isinstance(payload, (dict, list)):
            payload = json.dumps(payload)
        else:
            payload = str(payload)

        try:
            await self.client.publish(
                full_topic,
                payload=payload,
                qos=self.qos
            )
        except Exception as e:
            self.logger.error("Ошибка публикации MQTT",
                              topic=full_topic, error=str(e))

    async def subscribe(self, topic: str, callback):
        """
        Подписка на топик

        Args:
            topic: топик (относительный)
            callback: функция обратного вызова
        """
        if not self.connected:
            return

        full_topic = f"{self.topic_prefix}/{topic}"

        try:
            await self.client.subscribe(full_topic, qos=self.qos)

            # Запускаем прослушивание
            asyncio.create_task(self._listen(callback))

            self.logger.info("Подписка на топик", topic=full_topic)
        except Exception as e:
            self.logger.error("Ошибка подписки MQTT",
                              topic=full_topic, error=str(e))

    async def _listen(self, callback):
        """Прослушивание входящих сообщений"""
        try:
            async for message in self.client.messages:
                topic = message.topic.value
                payload = message.payload.decode()

                # Пытаемся распарсить JSON
                try:
                    data = json.loads(payload)
                except:
                    data = payload

                # Вызываем callback
                await callback(topic, data)
        except Exception as e:
            self.logger.error("Ошибка в MQTT listener", error=str(e))