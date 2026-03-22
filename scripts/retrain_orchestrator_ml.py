#!/usr/bin/env python3
"""
Сквозной конвейер: снимок конфига (опц.), генерация CSV, prepare+train для
predict_server_temp, predict_load (Prophet), predict_pue.

Запуск из корня v2:
  PYTHONPATH=. python3 scripts/retrain_orchestrator_ml.py
  PYTHONPATH=. python3 scripts/retrain_orchestrator_ml.py --skip-sim   # если CSV уже есть
  PYTHONPATH=. python3 scripts/retrain_orchestrator_ml.py --sim-steps 1500
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd or ROOT))
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-sim", action="store_true", help="Не запускать generate_ml_training_data")
    p.add_argument("--skip-snapshot", action="store_true", help="Не вызывать snapshot_twin_config")
    p.add_argument("--sim-steps", type=int, default=2500)
    p.add_argument(
        "--results-dir",
        type=str,
        default=str(ROOT / "models" / "_training_data" / "sim_csv"),
    )
    p.add_argument("--dc-config", type=str, default=str(ROOT / "DC_digital_twin" / "config" / "config_google_300s.yaml"))
    p.add_argument(
        "--time-step-seconds",
        type=int,
        default=300,
        help="Должен совпадать с simulator.time_step в конфиге прогона (300 для config_google_300s, 1 для default_config)",
    )
    p.add_argument("--temp-epochs", type=int, default=25)
    p.add_argument("--pue-epochs", type=int, default=35)
    args = p.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    if not args.skip_snapshot:
        run([sys.executable, str(ROOT / "scripts" / "snapshot_twin_config.py")], cwd=ROOT)

    results = Path(args.results_dir)
    if not results.is_absolute():
        results = ROOT / results

    if not args.skip_sim:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_ml_training_data.py"),
                "--steps",
                str(args.sim_steps),
                "--output",
                str(results),
                "--config",
                str(Path(args.dc_config).resolve()),
            ],
            cwd=ROOT,
        )

    srv = list(results.glob("*_servers.csv"))
    if not srv:
        print(f"No *_servers.csv in {results}; run without --skip-sim or point --results-dir", file=sys.stderr)
        sys.exit(1)

    # --- predict_server_temp ---
    temp_npz = ROOT / "models" / "_training_data" / "temp_dataset.npz"
    run(
        [
            sys.executable,
            str(ROOT / "models" / "predict_server_temp" / "prepare_temp_dataset.py"),
            "--results_dir",
            str(results),
            "--output_npz",
            str(temp_npz),
            "--time_step_seconds",
            str(args.time_step_seconds),
        ],
        cwd=ROOT,
    )
    temp_pt = ROOT / "models" / "predict_server_temp" / "temp_predictor_retrained.pt"
    temp_meta = ROOT / "models" / "predict_server_temp" / "temp_predictor_retrained_meta.json"
    run(
        [
            sys.executable,
            str(ROOT / "models" / "predict_server_temp" / "train_temp_predictor.py"),
            "--dataset_npz",
            str(temp_npz),
            "--model_out",
            str(temp_pt),
            "--meta_out",
            str(temp_meta),
            "--epochs",
            str(args.temp_epochs),
            "--device",
            "cpu",
        ],
        cwd=ROOT,
    )
    # стандартные имена для api_fastapi
    import shutil

    shutil.copy2(temp_pt, ROOT / "models" / "predict_server_temp" / "temp_predictor_mdot002_75_300s.pt")
    shutil.copy2(temp_meta, ROOT / "models" / "predict_server_temp" / "temp_predictor_mdot002_75_300s_meta.json")

    # --- predict_load ---
    run(
        [
            sys.executable,
            str(ROOT / "models" / "predict_load" / "prepare_load_dataset.py"),
            "--results_dir",
            str(results),
            "--time_step_seconds",
            str(args.time_step_seconds),
        ],
        cwd=ROOT,
    )
    run([sys.executable, str(ROOT / "models" / "predict_load" / "train_load_models.py")], cwd=ROOT)

    # --- predict_pue ---
    pue_npz = ROOT / "models" / "predict_pue" / "pue_dataset.npz"
    run(
        [
            sys.executable,
            str(ROOT / "models" / "predict_pue" / "prepare_pue_dataset.py"),
            "--results_dir",
            str(results),
            "--out",
            str(pue_npz),
            "--dc_config",
            str(Path(args.dc_config).resolve()),
            "--time_step_seconds",
            str(args.time_step_seconds),
        ],
        cwd=ROOT,
    )
    run(
        [
            sys.executable,
            str(ROOT / "models" / "predict_pue" / "train_pue_residual.py"),
            "--dataset",
            str(pue_npz),
            "--epochs",
            str(args.pue_epochs),
        ],
        cwd=ROOT,
    )

    print("Done. Artifacts updated under models/predict_server_temp, predict_load/artifacts, predict_pue.")


if __name__ == "__main__":
    main()
