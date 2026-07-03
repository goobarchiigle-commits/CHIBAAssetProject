"""Shared fixtures for survivorship analysis tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.analysis.contract import SurvivorRecord


def make_record(
    exp_id: str = "exp_a",
    strategy_id: str = "strat_v1",
    parameter_hash: str = "abc123",
    feature_names=("f_mom", "f_vol"),
    regime_scores=None,
    score: float = 0.6,
    survived: bool = True,
    rejection_reasons=(),
    fold_ids=(0,),
) -> SurvivorRecord:
    if regime_scores is None:
        regime_scores = {"trend_up": 1.2, "range": 0.3, "high_vol": -0.2}
    return SurvivorRecord.build(
        experiment_id=exp_id,
        strategy_id=strategy_id,
        parameter_hash=parameter_hash,
        feature_names=list(feature_names),
        regime_scores=dict(regime_scores),
        score=score,
        survived=survived,
        rejection_reasons=list(rejection_reasons),
        fold_ids=list(fold_ids),
    )
