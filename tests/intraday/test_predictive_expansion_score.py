"""
test_predictive_expansion_score.py

Coverage targets (≥90%):
  - PredictiveExpansionScorer クラス
  - score_expansion() 関数
  - component 変換関数
  - spike 抑制 / hard cap
  - deterministic replay
  - boundary conditions
"""
from __future__ import annotations

import pytest

from src.intraday.expansion_state_engine import ExpansionState, MIN_BARS_FOR_VERDICT
from src.intraday.predictive_expansion_score import (
    PredictiveExpansionScorer,
    ExpansionScoreResult,
    score_expansion,
    SCORE_WEIGHTS,
    SCORE_NO_VERDICT,
    MAX_SCORE_WHEN_SPIKE,
    LABEL_STRONG,
    LABEL_MODERATE,
    LABEL_WEAK,
    _score_drive,
    _score_persistence,
    _score_rvol_accel,
    _score_rs,
    _score_gap,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def make_state(
    expansion_type: str = "continuation",
    drive_efficiency: float = 0.80,
    persistence_ratio: float = 0.75,
    rvol_acceleration: float = 1.20,
    rs_percentile: float = 70.0,
    gap_acceptance: float = 0.90,
    spike_confidence: float = 0.0,
    minutes_elapsed: int = 20,
    expansion_probability: float = 0.75,
    new_high_confirmed: bool = False,
    spike_reasons: list[str] | None = None,
    continuation_reasons: list[str] | None = None,
) -> ExpansionState:
    return ExpansionState(
        symbol="TEST",
        eval_ts="09:20",
        minutes_elapsed=minutes_elapsed,
        drive_efficiency=drive_efficiency,
        rvol_ratio=1.5,
        rvol_acceleration=rvol_acceleration,
        persistence_ratio=persistence_ratio,
        rs_percentile=rs_percentile,
        gap_acceptance=gap_acceptance,
        expansion_probability=expansion_probability,
        spike_confidence=spike_confidence,
        expansion_type=expansion_type,
        is_spike_pattern=(expansion_type == "spike"),
        new_high_confirmed=new_high_confirmed,
        early_peak=1050.0,
        early_reversal_pct=0.1 if expansion_type == "continuation" else 0.8,
        or_high=1060.0,
        or_low=1000.0,
        spike_reasons=spike_reasons or [],
        continuation_reasons=continuation_reasons or ["drive", "persist"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# SCORE_WEIGHTS sanity
# ─────────────────────────────────────────────────────────────────────────────

def test_score_weights_sum_to_1():
    total = sum(SCORE_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-9, f"SCORE_WEIGHTS sum={total}"


def test_score_weights_has_all_components():
    expected = {"drive_efficiency", "persistence_ratio", "rvol_acceleration",
                "rs_percentile", "gap_acceptance"}
    assert set(SCORE_WEIGHTS.keys()) == expected


# ─────────────────────────────────────────────────────────────────────────────
# Component score functions
# ─────────────────────────────────────────────────────────────────────────────

def test_score_drive_full():
    assert _score_drive(1.0) == 100.0

def test_score_drive_zero():
    assert _score_drive(0.0) == 0.0

def test_score_drive_half():
    assert abs(_score_drive(0.5) - 50.0) < 0.01

def test_score_persistence_full():
    assert _score_persistence(1.0) == 100.0

def test_score_rvol_accel_neutral():
    # rvol_accel = 1.0 → score = 50
    assert abs(_score_rvol_accel(1.0) - 50.0) < 0.01

def test_score_rvol_accel_zero():
    assert _score_rvol_accel(0.0) == 0.0

def test_score_rvol_accel_double():
    # rvol_accel = 2.0 → score = 100
    assert abs(_score_rvol_accel(2.0) - 100.0) < 0.01

def test_score_rs_direct():
    assert _score_rs(75.0) == 75.0
    assert _score_rs(0.0) == 0.0
    assert _score_rs(100.0) == 100.0

def test_score_gap_active_full():
    # gap_acceptance = 1.0 → score = 100
    assert abs(_score_gap(1.0, True) - 100.0) < 0.01

def test_score_gap_active_filled():
    # gap_acceptance = 0.0 (gap filled to prev_close) → score = 50
    assert abs(_score_gap(0.0, True) - 50.0) < 0.01

def test_score_gap_inactive_neutral():
    assert _score_gap(0.3, False) == 50.0

def test_score_gap_negative():
    # gap_acceptance = -1.0 → score = 0
    assert abs(_score_gap(-1.0, True) - 0.0) < 0.01


# ─────────────────────────────────────────────────────────────────────────────
# score_expansion function
# ─────────────────────────────────────────────────────────────────────────────

def test_strong_continuation_high_score():
    state = make_state(
        expansion_type="continuation",
        drive_efficiency=0.90,
        persistence_ratio=0.85,
        rvol_acceleration=1.50,
        rs_percentile=80.0,
        gap_acceptance=0.95,
        spike_confidence=0.0,
        minutes_elapsed=25,
    )
    result = score_expansion(state)
    assert result.score >= LABEL_STRONG, f"Expected STRONG, got {result.score}"
    assert result.score_label == "STRONG"


def test_spike_hard_cap():
    state = make_state(
        expansion_type="spike",
        spike_confidence=0.80,
        drive_efficiency=0.20,
        persistence_ratio=0.25,
        rvol_acceleration=0.30,
        minutes_elapsed=20,
    )
    result = score_expansion(state)
    assert result.score <= MAX_SCORE_WHEN_SPIKE, (
        f"Spike score must be ≤ {MAX_SCORE_WHEN_SPIKE}, got {result.score}"
    )
    assert result.score_label == "BLOCKED"


def test_spike_confidence_suppression():
    base = make_state(spike_confidence=0.0, minutes_elapsed=20)
    spike = make_state(spike_confidence=0.80, minutes_elapsed=20)
    r_base  = score_expansion(base)
    r_spike = score_expansion(spike)
    assert r_spike.score < r_base.score, (
        f"Spike confidence should suppress score: {r_spike.score} vs {r_base.score}"
    )


def test_no_verdict_when_bars_insufficient():
    state = make_state(minutes_elapsed=MIN_BARS_FOR_VERDICT - 1)
    result = score_expansion(state)
    assert result.score == SCORE_NO_VERDICT
    assert result.score_label == "NO_VERDICT"


def test_score_bounded_0_to_100():
    for expansion_type in ("continuation", "spike", "undetermined"):
        state = make_state(expansion_type=expansion_type, minutes_elapsed=20)
        result = score_expansion(state)
        if result.score != SCORE_NO_VERDICT:
            assert 0 <= result.score <= 100


def test_component_scores_present():
    state = make_state(minutes_elapsed=20)
    result = score_expansion(state)
    assert set(result.component_scores.keys()) == set(SCORE_WEIGHTS.keys())


def test_component_weights_match():
    state = make_state(minutes_elapsed=20)
    result = score_expansion(state)
    assert result.component_weights == SCORE_WEIGHTS


def test_explanation_populated():
    state = make_state(
        continuation_reasons=["drive_ok", "persist_ok"],
        minutes_elapsed=20,
    )
    result = score_expansion(state)
    assert len(result.explanation) > 0


def test_spike_explanation_contains_penalty():
    state = make_state(spike_confidence=0.7, minutes_elapsed=20)
    result = score_expansion(state)
    penalty_mentions = [e for e in result.explanation if "spike_penalty" in e]
    assert len(penalty_mentions) > 0


def test_label_moderate():
    state = make_state(
        expansion_type="continuation",
        drive_efficiency=0.55,
        persistence_ratio=0.55,
        rvol_acceleration=0.90,
        rs_percentile=50.0,
        gap_acceptance=0.60,
        spike_confidence=0.0,
        minutes_elapsed=20,
    )
    result = score_expansion(state)
    assert result.score_label in ("MODERATE", "STRONG", "WEAK", "BLOCKED")  # boundary-safe


def test_label_strong_threshold():
    state = make_state(
        drive_efficiency=0.90,
        persistence_ratio=0.90,
        rvol_acceleration=1.8,
        rs_percentile=90.0,
        gap_acceptance=1.0,
        spike_confidence=0.0,
        expansion_type="continuation",
        minutes_elapsed=25,
    )
    result = score_expansion(state)
    assert result.score >= LABEL_STRONG


def test_score_result_type():
    state = make_state(minutes_elapsed=20)
    result = score_expansion(state)
    assert isinstance(result, ExpansionScoreResult)


def test_spike_penalty_applied_field():
    state = make_state(spike_confidence=0.6, minutes_elapsed=20)
    result = score_expansion(state)
    assert result.spike_penalty_applied > 0.0


def test_no_spike_penalty_when_confidence_zero():
    state = make_state(spike_confidence=0.0, minutes_elapsed=20)
    result = score_expansion(state)
    assert result.spike_penalty_applied == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# PredictiveExpansionScorer class
# ─────────────────────────────────────────────────────────────────────────────

def test_scorer_default_weights():
    scorer = PredictiveExpansionScorer()
    assert scorer.weights == SCORE_WEIGHTS


def test_scorer_custom_weights():
    w = {k: 0.2 for k in SCORE_WEIGHTS}
    scorer = PredictiveExpansionScorer(weights=w)
    assert scorer.weights == w


def test_scorer_invalid_weights_raises():
    with pytest.raises(ValueError):
        PredictiveExpansionScorer(weights={"drive_efficiency": 0.5})  # sum ≠ 1.0


def test_scorer_score_method_returns_result():
    scorer = PredictiveExpansionScorer()
    state = make_state(minutes_elapsed=20)
    result = scorer.score(state)
    assert isinstance(result, ExpansionScoreResult)


def test_scorer_deterministic():
    scorer = PredictiveExpansionScorer()
    state = make_state(minutes_elapsed=20)
    r1 = scorer.score(state)
    r2 = scorer.score(state)
    assert r1.score == r2.score
    assert r1.score_label == r2.score_label


def test_scorer_batch():
    scorer = PredictiveExpansionScorer()
    states = [make_state(minutes_elapsed=20), make_state(expansion_type="spike", minutes_elapsed=20)]
    results = scorer.score_batch(states)
    assert len(results) == 2
    assert all(isinstance(r, ExpansionScoreResult) for _, r in results)


def test_scorer_top_k():
    scorer = PredictiveExpansionScorer()
    states = [
        make_state(minutes_elapsed=20),
        make_state(expansion_type="spike", minutes_elapsed=20),
        make_state(minutes_elapsed=MIN_BARS_FOR_VERDICT - 1),  # NO_VERDICT
    ]
    # spikes と NO_VERDICT は除外
    top = scorer.top_k(states, k=5)
    for sym, sc in top:
        assert sc >= 0
        assert sc <= 100


def test_scorer_top_k_deterministic_tiebreak():
    scorer = PredictiveExpansionScorer()
    # 同一スコアの場合はアルファベット順
    s1 = make_state(minutes_elapsed=20)
    s1.symbol = "Z"
    s2 = make_state(minutes_elapsed=20)
    s2.symbol = "A"
    top = scorer.top_k([s1, s2], k=2)
    assert len(top) == 2


def test_scorer_top_k_excludes_spike():
    scorer = PredictiveExpansionScorer()
    spike_state = make_state(expansion_type="spike", minutes_elapsed=20)
    top = scorer.top_k([spike_state], k=5)
    assert len(top) == 0


# ─────────────────────────────────────────────────────────────────────────────
# to_dict
# ─────────────────────────────────────────────────────────────────────────────

def test_score_result_to_dict():
    state = make_state(minutes_elapsed=20)
    result = score_expansion(state)
    d = result.to_dict()
    assert "score" in d
    assert "score_label" in d
    assert "component_scores" in d
    assert "explanation" in d
