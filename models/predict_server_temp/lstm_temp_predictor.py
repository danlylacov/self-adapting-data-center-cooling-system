#!/usr/bin/env python3
"""
Упрощённая LSTM-модель прогнозирования температуры серверов:
- mean: прогноз t_chip (регрессия) на horizon_hours вперёд
- log_std: предсказание неопределённости (для доверительного интервала)
- p_overheat: вероятность превышения threshold_c

Вход:
  X: [B, input_hours, F]
Выход:
  mean: [B, horizon]
  std:  [B, horizon] via exp(log_std)
  p_overheat: [B, horizon]
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


class TempPredictorLSTM(nn.Module):
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

        # Для каждого часа horizon нужны:
        # mean и log_std (2) + logits вероятности overheat (1) => всего 3
        self.fc = nn.Linear(cfg.hidden, cfg.horizon_hours * 3)

    def forward(self, x: torch.Tensor):
        # x: [B, input_hours, F]
        out, _ = self.lstm(x)          # [B, input_hours, hidden]
        h = out[:, -1, :]             # last step => [B, hidden]
        pred = self.fc(h)             # [B, horizon*3]

        h = self.cfg.horizon_hours
        mean = pred[:, :h]
        log_std = pred[:, h : 2 * h]
        overheat_logits = pred[:, 2 * h : 3 * h]

        std = torch.exp(log_std).clamp(min=1e-4, max=1e3)
        p_overheat = torch.sigmoid(overheat_logits)
        return mean, std, p_overheat, log_std

