#!/usr/bin/env python3
"""
Точка входа в симулятор ЦОД
"""
import asyncio
import argparse
import os
import signal
import sys
from src.core.simulator import DataCenterSimulator


async def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(description='Симулятор ЦОД')
    parser.add_argument(
        '--config',
        '-c',
        type=str,
        default=None,
        help='Путь к YAML (опционально; иначе встроенный default_config)',
    )
    parser.add_argument('--mode', '-m', type=str, default='realtime',
                        choices=['realtime', 'fast', 'interactive'],
                        help='Режим работы')
    parser.add_argument('--duration', '-d', type=int, default=3600,
                        help='Длительность симуляции (сек)')
    parser.add_argument('--steps', '-s', type=int, default=1000,
                        help='Количество шагов (для fast режима)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Директория для сохранения результатов (переопределяет конфиг)')

    args = parser.parse_args()

    # Создаем симулятор
    sim = DataCenterSimulator(config_path=args.config) if args.config else DataCenterSimulator()

    # Переопределяем output если указан
    if args.output and sim.result_saver:
        sim.result_saver.set_base_path(args.output)

    # Обработка Ctrl+C
    def signal_handler(sig, frame):
        print("\nОстановка симуляции...")
        sim.stop()
        sim.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Запуск в выбранном режиме
        if args.mode == 'realtime':
            print(f"Запуск в реальном времени на {args.duration} секунд...")
            await sim.run_realtime(args.duration)
        elif args.mode == 'fast':
            print(f"Запуск в ускоренном режиме на {args.steps} шагов...")
            sim.run_fast(args.steps)
        elif args.mode == 'interactive':
            print("Интерактивный режим. Введите 'help' для справки.")
            await interactive_mode(sim)
    finally:
        # Обязательно сохраняем результаты
        sim.close()
        print(f"\nРезультаты сохранены")


async def interactive_mode(sim):
    """Интерактивный режим для отладки"""
    sim.running = True

    while sim.running:
        try:
            cmd = input("sim> ").strip().lower()

            if cmd == 'help':
                print("""
Команды:
  step [n]    - выполнить n шагов (по умолчанию 1)
  setpoint T  - установить температуру подачи
  fanspeed S  - установить скорость вентиляторов (0-100)
  mode M      - установить режим (free/chiller/mixed)
  state       - показать текущее состояние
  pue         - показать текущий PUE
  reset       - сбросить симулятор
  save        - принудительно сохранить результаты
  exit        - выход
                """)
            elif cmd.startswith('step'):
                parts = cmd.split()
                n = int(parts[1]) if len(parts) > 1 else 1
                for _ in range(n):
                    sim.step()
                print(f"Выполнено {n} шагов")
            elif cmd.startswith('setpoint'):
                temp = float(cmd.split()[1])
                sim.set_cooling_setpoint(temp)
                print(f"Setpoint установлен: {temp}°C")
            elif cmd.startswith('fanspeed'):
                speed = float(cmd.split()[1])
                sim.set_fan_speed(speed)
                print(f"Скорость вентиляторов: {speed}%")
            elif cmd.startswith('mode'):
                mode = cmd.split()[1]
                sim.set_cooling_mode(mode)
                print(f"Режим: {mode}")
            elif cmd == 'state':
                state = sim.get_state()
                print(f"Время: {state['time']:.1f} сек")
                print(f"Температура помещения: {state['room']['temperature']:.1f}°C")
                print(f"Мощность серверов: {state['rack']['total_power']:.0f} Вт")
                print(f"Setpoint: {state['cooling']['crac']['setpoint']}°C")
            elif cmd == 'pue':
                telemetry = sim.get_telemetry()
                print(f"PUE: {telemetry['pue']:.3f}")
            elif cmd == 'reset':
                sim.reset()
                print("Симулятор сброшен")
            elif cmd == 'save':
                if sim.result_saver:
                    sim.result_saver.flush()
                    print("Результаты сохранены")
            elif cmd == 'exit':
                sim.running = False
                break
            else:
                print(f"Неизвестная команда: {cmd}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ошибка: {e}")


if __name__ == '__main__':
    asyncio.run(main())