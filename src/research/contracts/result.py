"""ResultContract — immutable experiment result for a single fold."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RegimeMetrics:
    by_regime: Dict[str, Dict[str, float]]  # regime → {sharpe, max_drawdown, hit_rate, trade_count}

    def to_dict(self) -> dict:
        return {"by_regime": {k: dict(v) for k, v in self.by_regime.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> RegimeMetrics:
        return cls(by_regime={k: dict(v) for k, v in d["by_regime"].items()})

    @staticmethod
    def empty() -> RegimeMetrics:
        return RegimeMetrics(by_regime={})


@dataclass(frozen=True)
class StabilityMetrics:
    parameter_stability: float  # 0-1, 1=fully stable across folds
    oos_robustness: float       # 0-1, ratio of passing OOS folds
    sharpe_decay: float         # IS_sharpe - OOS_sharpe (positive = decay)

    def to_dict(self) -> dict:
        return {
            "parameter_stability": self.parameter_stability,
            "oos_robustness": self.oos_robustness,
            "sharpe_decay": self.sharpe_decay,
        }

    @classmethod
    def from_dict(cls, d: dict) -> StabilityMetrics:
        return cls(
            parameter_stability=float(d["parameter_stability"]),
            oos_robustness=float(d["oos_robustness"]),
            sharpe_decay=float(d["sharpe_decay"]),
        )

    @staticmethod
    def null() -> StabilityMetrics:
        return StabilityMetrics(parameter_stability=0.0, oos_robustness=0.0, sharpe_decay=0.0)


@dataclass(frozen=True)
class ResultContract:
    experiment_id: str
    fold_id: int
    sharpe: float
    sortino: float
    turnover: float
    max_drawdown: float  # negative value (e.g. -0.12 = -12%)
    exposure: float      # avg fraction of capital deployed
    pnl: float           # total PnL
    hit_rate: float      # fraction of winning trades
    tail_risk: float     # CVaR at 5%
    regime_metrics: RegimeMetrics
    stability_metrics: StabilityMetrics
    trade_count: int

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "fold_id": self.fold_id,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "turnover": self.turnover,
            "max_drawdown": self.max_drawdown,
            "exposure": self.exposure,
            "pnl": self.pnl,
            "hit_rate": self.hit_rate,
            "tail_risk": self.tail_risk,
            "regime_metrics": self.regime_metrics.to_dict(),
            "stability_metrics": self.stability_metrics.to_dict(),
            "trade_count": self.trade_count,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResultContract:
        return cls(
            experiment_id=d["experiment_id"],
            fold_id=int(d["fold_id"]),
            sharpe=float(d["sharpe"]),
            sortino=float(d["sortino"]),
            turnover=float(d["turnover"]),
            max_drawdown=float(d["max_drawdown"]),
            exposure=float(d["exposure"]),
            pnl=float(d["pnl"]),
            hit_rate=float(d["hit_rate"]),
            tail_risk=float(d["tail_risk"]),
            regime_metrics=RegimeMetrics.from_dict(d["regime_metrics"]),
            stability_metrics=StabilityMetrics.from_dict(d["stability_metrics"]),
            trade_count=int(d["trade_count"]),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)
