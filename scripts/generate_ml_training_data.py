#!/usr/bin/env python3
"""
Запуск ускоренной симуляции DC_digital_twin для сбора *_servers.csv / *_summary.csv
(те же файлы, что ожидают prepare_temp_dataset / prepare_pue_dataset / prepare_load_dataset).

Запуск из корня репозитория v2:
  python scripts/generate_ml_training_data.py --steps 3000

Требуется: каталог DC_digital_twin с main.py; конфиг по умолчанию config_google_300s.yaml.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dc = root / "DC_digital_twin"
    main_py = dc / "main.py"
    if not main_py.is_file():
        print("DC_digital_twin/main.py not found", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(dc / "config" / "config_google_300s.yaml"),
        help="YAML конфиг симулятора (должен совпадать с вашим двойником для согласованности)",
    )
    p.add_argument("--steps", type=int, default=3000, help="Число шагов fast-режима")
    p.add_argument(
        "--output",
        type=str,
        default=str(root / "models" / "_training_data" / "sim_csv"),
        help="Каталог для CSV (output.path)",
    )
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(main_py),
        "--mode",
        "fast",
        "--steps",
        str(args.steps),
        "--config",
        str(Path(args.config).resolve()),
        "--output",
        str(out.resolve()),
    ]
    print("Running:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(dc))
    if r.returncode != 0:
        sys.exit(r.returncode)
    print(f"Done. CSV should be under: {out.resolve()}")


if __name__ == "__main__":
    main()
