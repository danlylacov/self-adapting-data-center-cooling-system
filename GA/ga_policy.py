"""
Готовая политика охлаждения по средней температуре чипа (параметры из train_ga / JSON).

Встраивание: на каждом тике управления прочитать avg_chip_temp и текущую уставку CRAC,
вызвать recommend() и применить к API двойника setpoint + fanspeed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

from GA.ga_core import GENE_NAMES, SP_MAX_C, SP_MIN_C, clip_chromosome


@dataclass
class GaCoolingPolicy:
    """Пропорциональная политика: сдвиг уставки и вентилятор по ошибке (T_chip - target)."""

    target_chip_c: float
    kp: float
    fan_base: float
    k_fan: float
    max_delta_sp: float
    sp_min_c: float = SP_MIN_C
    sp_max_c: float = SP_MAX_C

    @classmethod
    def from_genes_array(cls, chrom: np.ndarray) -> GaCoolingPolicy:
        g = clip_chromosome(chrom)
        return cls(
            target_chip_c=float(g[0]),
            kp=float(g[1]),
            fan_base=float(g[2]),
            k_fan=float(g[3]),
            max_delta_sp=float(g[4]),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> GaCoolingPolicy:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        genes = data.get("genes") or {}
        vec = np.array([float(genes[name]) for name in GENE_NAMES])
        return cls.from_genes_array(vec)

    @classmethod
    def from_dict(cls, genes: dict[str, Any]) -> GaCoolingPolicy:
        vec = np.array([float(genes[name]) for name in GENE_NAMES])
        return cls.from_genes_array(vec)

    def recommend(self, avg_chip_temp_c: float, current_setpoint_c: float) -> Tuple[float, float]:
        """
        Возвращает (новая_уставка_°C, скорость_вентилятора_%).
        """
        err = float(avg_chip_temp_c) - self.target_chip_c
        delta_sp = -self.kp * err
        delta_sp = float(np.clip(delta_sp, -self.max_delta_sp, self.max_delta_sp))
        new_sp = float(np.clip(current_setpoint_c + delta_sp, self.sp_min_c, self.sp_max_c))
        fan = self.fan_base + self.k_fan * max(0.0, err)
        fan = float(np.clip(fan, 0.0, 100.0))
        return new_sp, fan

    def to_dict(self) -> dict[str, float]:
        return {
            "target_chip_c": self.target_chip_c,
            "kp": self.kp,
            "fan_base": self.fan_base,
            "k_fan": self.k_fan,
            "max_delta_sp": self.max_delta_sp,
            "sp_min_c": self.sp_min_c,
            "sp_max_c": self.sp_max_c,
        }
