"""ParameterSweep — deterministic cartesian product parameter grid.

Supports all parameter types that are JSON-serializable.
Combinations are always sorted by parameter-hash, independent of input order.

Typical usage::

    sweep = ParameterSweep(
        strategy_id="turtle_v1",
        grid={
            "holding_period":    [3, 5, 10, 20],
            "entry_threshold":   [0.5, 1.0, 1.5, 2.0],
            "exit_threshold":    [0.3, 0.5, 0.7],
            "vol_filter":        [True, False],
            "stop_atr_mult":     [2.0, 3.0],
        },
        seed=42,
    )
    combos = sweep.combinations()   # 4*4*3*2*2 = 192 deterministic dicts
"""
from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ParameterSweep:
    strategy_id: str
    grid: Dict[str, List[Any]]
    seed: int = 0

    # ── public API ─────────────────────────────────────────────────────────────

    def combinations(self) -> List[Dict[str, Any]]:
        """Full cartesian product, sorted deterministically by parameter hash.

        Same grid → same ordering, regardless of dict insertion order or platform.
        """
        if not self.grid:
            return [{}]

        keys = sorted(self.grid.keys())               # deterministic key order
        value_lists = [self.grid[k] for k in keys]

        combos: List[Dict[str, Any]] = [
            dict(zip(keys, combo)) for combo in itertools.product(*value_lists)
        ]
        combos.sort(key=lambda p: _param_hash(p))
        return combos

    def parameter_space_hash(self) -> str:
        """Stable 16-char hash of the entire grid definition."""
        payload = json.dumps(
            {k: self.grid[k] for k in sorted(self.grid)},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def count(self) -> int:
        """Number of parameter combinations."""
        if not self.grid:
            return 1
        n = 1
        for v in self.grid.values():
            n *= len(v)
        return n

    def subset(self, keys: List[str]) -> ParameterSweep:
        """Return a new sweep over a subset of parameters."""
        return ParameterSweep(
            strategy_id=self.strategy_id,
            grid={k: self.grid[k] for k in keys if k in self.grid},
            seed=self.seed,
        )


# ── Predefined dimension helpers ───────────────────────────────────────────────

def holding_period_dim(periods: List[int] = None) -> List[int]:
    return periods or [3, 5, 7, 10, 14, 20, 30]


def entry_threshold_dim(thresholds: List[float] = None) -> List[float]:
    return thresholds or [0.5, 1.0, 1.5, 2.0, 2.5]


def exit_threshold_dim(thresholds: List[float] = None) -> List[float]:
    return thresholds or [0.3, 0.5, 0.7, 1.0]


def vol_filter_dim() -> List[bool]:
    return [True, False]


def liquidity_filter_dim(thresholds: List[float] = None) -> List[float]:
    return thresholds or [0.2, 0.5, 1.0]


def stop_atr_mult_dim(mults: List[float] = None) -> List[float]:
    return mults or [1.5, 2.0, 3.0]


def universe_dim(variants: List[str] = None) -> List[str]:
    return variants or ["rsr42"]


def rebalance_freq_dim(freqs: List[str] = None) -> List[str]:
    return freqs or ["daily", "weekly"]


# ── Private ────────────────────────────────────────────────────────────────────

def _param_hash(params: Dict[str, Any]) -> str:
    payload = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def param_hash(params: Dict[str, Any]) -> str:
    """Public: stable 16-char hash for a single parameter dict."""
    return _param_hash(params)[:16]
