#!/usr/bin/env python3
"""Обучение LSTM residual PUE; сохраняет pue_residual_predictor.pt и pue_residual_meta.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from models.predict_pue.lstm_pue_residual import ModelConfig, PueResidualPredictorLSTM


def gaussian_nll(mean: torch.Tensor, std: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    var = std**2
    return torch.mean(torch.log(std.clamp(min=1e-4)) + 0.5 * ((y - mean) ** 2) / var.clamp(min=1e-8))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="models/predict_pue/pue_dataset.npz")
    ap.add_argument("--model_out", type=str, default="models/predict_pue/pue_residual_predictor.pt")
    ap.add_argument("--meta_out", type=str, default="models/predict_pue/pue_residual_meta.json")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    ds_path = Path(args.dataset)
    if not ds_path.is_absolute():
        ds_path = _REPO_ROOT / ds_path
    raw = np.load(ds_path, allow_pickle=True)
    X = raw["X"].astype(np.float32)
    y = raw["y"].astype(np.float32)
    input_hours = int(raw["input_hours"])
    horizon_hours = int(raw["horizon_hours"])
    _F = X.shape[2]

    order = np.arange(len(X))
    rng = np.random.default_rng(42)
    rng.shuffle(order)
    X = X[order]
    y = y[order]

    n = len(X)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    tr = np.arange(0, n_train)
    va = np.arange(n_train, n_train + n_val)
    te = np.arange(n_train + n_val, n)

    X_mean = X[tr].mean(axis=(0, 1), keepdims=True)
    X_std = X[tr].std(axis=(0, 1), keepdims=True) + 1e-6

    X_train = (X[tr] - X_mean) / X_std
    X_val = (X[va] - X_mean) / X_std
    X_test = (X[te] - X_mean) / X_std

    device = torch.device(args.device)
    cfg = ModelConfig(input_size=_F, hidden=128, num_layers=2, horizon_hours=horizon_hours)
    model = PueResidualPredictorLSTM(cfg).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y[tr], dtype=torch.float32)
    Xv = torch.tensor(X_val, dtype=torch.float32)
    yv = torch.tensor(y[va], dtype=torch.float32)

    best_val = float("inf")
    model_out = Path(args.model_out)
    if not model_out.is_absolute():
        model_out = _REPO_ROOT / model_out
    meta_out = Path(args.meta_out)
    if not meta_out.is_absolute():
        meta_out = _REPO_ROOT / meta_out
    model_out.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "room_temperature",
        "cooling_setpoint",
        "cooling_fan_speed",
        "outside_temperature",
        "humidity",
        "wind_speed",
        "avg_exhaust_temp",
        "servers_power_total",
        "pue_physics",
        "residual",
    ]

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i : i + args.batch_size]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            mean_p, std_p = model(xb)
            loss = gaussian_nll(mean_p, std_p, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())

        model.eval()
        with torch.no_grad():
            mv, sv = model(Xv.to(device))
            val_loss = float(gaussian_nll(mv, sv, yv.to(device)).cpu())

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), model_out)

        print(f"epoch {epoch:03d} train_nll={total/ max(1,(len(perm)+args.batch_size-1)//args.batch_size):.5f} val_nll={val_loss:.5f}")

    meta = {
        "input_hours": input_hours,
        "horizon_hours": horizon_hours,
        "feature_count": _F,
        "feature_cols": feature_cols,
        "X_mean": X_mean.reshape(-1).tolist(),
        "X_std": X_std.reshape(-1).tolist(),
        "model_config": {
            "input_size": _F,
            "hidden": 128,
            "num_layers": 2,
            "horizon_hours": horizon_hours,
        },
        "best_val_nll": best_val,
        "dataset_npz": str(ds_path),
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {model_out} and {meta_out}")


if __name__ == "__main__":
    main()
