#!/usr/bin/env python3
"""
Обучение упрощённой LSTM-модели прогнозирования температуры серверов.

Данные берутся из NPZ, который строится скриптом:
  models/predict_server_temp/prepare_temp_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lstm_temp_predictor import ModelConfig, TempPredictorLSTM


def gaussian_nll(mean: torch.Tensor, std: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    mean/std/y shape: [B, H]
    std must be positive.
    """
    var = std ** 2
    # log(std) терм уже включён через var
    return torch.mean(torch.log(std) + 0.5 * ((y - mean) ** 2) / var)


def main():
    parser = argparse.ArgumentParser(description="Train temperature predictor (LSTM).")
    parser.add_argument("--dataset_npz", type=str, default="temp_dataset.npz")
    parser.add_argument("--model_out", type=str, default="temp_predictor.pt")
    parser.add_argument("--meta_out", type=str, default="temp_predictor_meta.json")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--threshold_c", type=float, default=85.0)
    args = parser.parse_args()

    dataset = np.load(args.dataset_npz, allow_pickle=True)
    X = dataset["X"]  # [N, input_hours, F]
    y_mean = dataset["y_mean"]  # [N, horizon]
    overheat = dataset["overheat"]  # [N, horizon]
    window_start_hour = dataset["window_start_hour"]  # [N]
    server_id_arr = dataset["server_id_arr"]  # [N]

    input_hours = int(dataset["input_hours"])
    horizon_hours = int(dataset["horizon_hours"])
    time_step_seconds = int(dataset["time_step_seconds"]) if "time_step_seconds" in dataset else None
    base_date = str(dataset["base_date"]) if "base_date" in dataset else None
    feature_cols = dataset["feature_cols"] if "feature_cols" in dataset else None
    if feature_cols is not None:
        if isinstance(feature_cols, np.ndarray):
            feature_cols = str(feature_cols.item())
        if isinstance(feature_cols, (bytes, bytearray)):
            feature_cols = feature_cols.decode("utf-8")
        if isinstance(feature_cols, str):
            feature_cols = [x.strip() for x in feature_cols.split(",") if x.strip()]

    N, T, F = X.shape
    assert T == input_hours

    # Хронологическое разбиение: train / val / test по window_start_hour
    order = np.argsort(window_start_hour, kind="stable")
    X = X[order]
    y_mean = y_mean[order]
    overheat = overheat[order]
    window_start_hour = window_start_hour[order]
    server_id_arr = server_id_arr[order]

    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val
    train_idx = np.arange(0, n_train)
    val_idx = np.arange(n_train, n_train + n_val)
    test_idx = np.arange(n_train + n_val, n_train + n_val + n_test)

    X_train_raw = X[train_idx]
    X_val_raw = X[val_idx]
    X_test_raw = X[test_idx]

    # Нормализация только по train (чтобы не было leakage)
    X_mean = X_train_raw.mean(axis=(0, 1), keepdims=True)
    X_std = X_train_raw.std(axis=(0, 1), keepdims=True) + 1e-6

    X_train = torch.tensor((X_train_raw - X_mean) / X_std, dtype=torch.float32)
    y_train = torch.tensor(y_mean[train_idx], dtype=torch.float32)
    over_train = torch.tensor(overheat[train_idx], dtype=torch.float32)

    X_val = torch.tensor((X_val_raw - X_mean) / X_std, dtype=torch.float32)
    y_val = torch.tensor(y_mean[val_idx], dtype=torch.float32)
    over_val = torch.tensor(overheat[val_idx], dtype=torch.float32)

    X_test = torch.tensor((X_test_raw - X_mean) / X_std, dtype=torch.float32)
    y_test = torch.tensor(y_mean[test_idx], dtype=torch.float32)
    over_test = torch.tensor(overheat[test_idx], dtype=torch.float32)

    device = torch.device(args.device)
    cfg = ModelConfig(input_size=F, num_layers=2, horizon_hours=horizon_hours)
    model = TempPredictorLSTM(cfg).to(device)

    opt = optim.Adam(model.parameters(), lr=args.lr)
    bce = nn.BCELoss()

    best_val = float("inf")
    model_out = Path(args.model_out)
    meta_out = Path(args.meta_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(model: TempPredictorLSTM, X_t: torch.Tensor, y_t: torch.Tensor, over_t: torch.Tensor) -> Dict[str, float]:
        model.eval()
        with torch.no_grad():
            mean_pred, std_pred, p_over_pred, _ = model(X_t)
            mae = torch.mean(torch.abs(mean_pred - y_t)).item()
            rmse = torch.sqrt(torch.mean((mean_pred - y_t) ** 2)).item()
            nll = gaussian_nll(mean_pred, std_pred, y_t).item()
            brier = torch.mean((p_over_pred - over_t) ** 2).item()
            acc = torch.mean(((p_over_pred > 0.5).float() == (over_t > 0.5).float()).float()).item()
            return {"mae": mae, "rmse": rmse, "nll": nll, "brier": brier, "acc": acc}

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(X_train.shape[0])

        total_loss = 0.0
        for start in range(0, len(perm), args.batch_size):
            batch_idx = perm[start : start + args.batch_size]

            xb = X_train[batch_idx].to(device)
            yb = y_train[batch_idx].to(device)
            ob = over_train[batch_idx].to(device)

            mean_pred, std_pred, p_over_pred, _log_std = model(xb)

            loss_mean = gaussian_nll(mean_pred, std_pred, yb)
            loss_over = bce(p_over_pred, ob)
            loss = loss_mean + 0.5 * loss_over

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())

        # Валид
        model.eval()
        with torch.no_grad():
            mean_pred, std_pred, p_over_pred, _ = model(X_val.to(device))
            val_loss_mean = gaussian_nll(mean_pred, std_pred, y_val.to(device))
            val_loss_over = bce(p_over_pred, over_val.to(device))
            val_loss = val_loss_mean + 0.5 * val_loss_over

        metrics_val = evaluate(model, X_val.to(device), y_val.to(device), over_val.to(device))
        print(
            f"epoch={epoch:03d} train_loss={total_loss:.4f} "
            f"val_loss={float(val_loss.item()):.4f} "
            f"val_mae={metrics_val['mae']:.4f} val_rmse={metrics_val['rmse']:.4f} val_nll={metrics_val['nll']:.4f} "
            f"val_brier={metrics_val['brier']:.6f} val_acc={metrics_val['acc']:.3f}"
        )

        if float(val_loss.item()) < best_val:
            best_val = float(val_loss.item())
            torch.save(model.state_dict(), model_out)

    meta = {
        "input_hours": input_hours,
        "horizon_hours": horizon_hours,
        "feature_count": int(F),
        "threshold_c": args.threshold_c,
        "time_step_seconds": time_step_seconds,
        "base_date": base_date,
        "feature_cols": feature_cols,
        "X_mean": X_mean.reshape(-1).tolist(),
        "X_std": X_std.reshape(-1).tolist(),
        "model_config": cfg.__dict__,
    }
    meta_out.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved model: {model_out}")
    print(f"Saved meta: {meta_out}")

    # Финальная проверка на тесте
    metrics_test = evaluate(model, X_test.to(device), y_test.to(device), over_test.to(device))
    print(
        f"TEST: mae={metrics_test['mae']:.4f} rmse={metrics_test['rmse']:.4f} "
        f"nll={metrics_test['nll']:.4f} brier={metrics_test['brier']:.6f} acc={metrics_test['acc']:.3f}"
    )


if __name__ == "__main__":
    main()

