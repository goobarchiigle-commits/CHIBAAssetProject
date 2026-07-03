"""Capacity proxy — simple liquidity-based capacity estimation.

Formula: participation_rate = position_size / ADV

Typical industry rule: > 10% of ADV is problematic for liquidity.
This module provides a deterministic scoring penalty for ranking.

No portfolio optimization. No allocation decisions.
Implementation is intentionally simple and auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CapacityProxy:
    avg_participation_rate: float    # mean(position_size / ADV)
    max_participation_rate: float    # max(position_size / ADV)
    liquidity_stress_proxy: float    # 95th percentile participation rate
    ranking_penalty: float           # 0-1 additive penalty for excessive participation

    def to_dict(self) -> dict:
        return {
            "avg_participation_rate": self.avg_participation_rate,
            "max_participation_rate": self.max_participation_rate,
            "liquidity_stress_proxy": self.liquidity_stress_proxy,
            "ranking_penalty": self.ranking_penalty,
        }

    @classmethod
    def from_dict(cls, d: dict) -> CapacityProxy:
        return cls(**d)


def compute_capacity_proxy(
    position_sizes: pd.Series,
    adv: pd.Series,
    max_acceptable_participation: float = 0.10,
) -> CapacityProxy:
    """Compute capacity proxy for a single strategy.

    Args:
        position_sizes: daily position size in currency units (JPY)
        adv: daily average dollar volume (ADV) in same currency
        max_acceptable_participation: threshold above which penalty applies

    Returns:
        CapacityProxy with deterministic scores.
    """
    if len(position_sizes) == 0 or len(adv) == 0:
        return CapacityProxy(
            avg_participation_rate=0.0,
            max_participation_rate=0.0,
            liquidity_stress_proxy=0.0,
            ranking_penalty=0.0,
        )

    pos, adv_aligned = position_sizes.align(adv, join="inner")
    adv_safe = adv_aligned.clip(lower=1.0)           # avoid division by zero
    participation = pos.abs() / adv_safe

    avg_rate  = float(participation.mean())
    max_rate  = float(participation.max())
    stress    = float(participation.quantile(0.95))  # 95th pct = liquidity stress

    # Penalty: 0 when avg_rate < half limit; ramps to 1 at 2× limit
    # penalty = clamp(0, 1, (avg_rate / limit - 0.5) / 1.5)
    t = (avg_rate / max_acceptable_participation - 0.5) / 1.5
    penalty = float(min(1.0, max(0.0, t)))

    return CapacityProxy(
        avg_participation_rate=avg_rate,
        max_participation_rate=max_rate,
        liquidity_stress_proxy=stress,
        ranking_penalty=penalty,
    )


def batch_capacity_proxies(
    position_map: Dict[str, pd.Series],
    adv_map: Dict[str, pd.Series],
    max_acceptable_participation: float = 0.10,
) -> Dict[str, CapacityProxy]:
    """Compute capacity proxies for multiple strategies.

    Returns dict keyed by experiment_id, sorted for determinism.
    """
    result: Dict[str, CapacityProxy] = {}
    for exp_id in sorted(position_map.keys()):
        pos = position_map[exp_id]
        adv = adv_map.get(exp_id, pd.Series(dtype=float))
        result[exp_id] = compute_capacity_proxy(pos, adv, max_acceptable_participation)
    return result
