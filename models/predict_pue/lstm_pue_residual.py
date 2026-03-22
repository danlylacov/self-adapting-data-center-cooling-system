#!/usr/bin/env python3
"""
LSTM-модель для предсказания residual PUE на горизонте:
  residual(t) = pue_real(t) - pue_physics(t)

Выход:
- mean: предсказанный residual на горизонте (регрессия)
- std: неопределенность (через exp(log_std))
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ModelConfig:
    input_size: int
    hidden: int = 128
    num_layers: int = 2
    horizon_hours: int = 6


class PueResidualPredictorLSTM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.input_size,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=0.2 if cfg.num_layers > 1 else 0.0,
        )

        # Для каждого часа горизонта нужно 2 величины: mean и log_std.
        self.fc = nn.Linear(cfg.hidden, cfg.horizon_hours * 2)

    def forward(self, x: torch.Tensor):
        # x: [B, input_hours, F]
        out, _ = self.lstm(x)  # [B, input_hours, hidden]
        h = out[:, -1, :]  # last step => [B, hidden]
        pred = self.fc(h)  # [B, horizon*2]

        h = self.cfg.horizon_hours
        mean = pred[:, :h]
        log_std = pred[:, h : 2 * h]

        std = torch.exp(log_std).clamp(min=1e-4, max=1e3)
        return mean, std

