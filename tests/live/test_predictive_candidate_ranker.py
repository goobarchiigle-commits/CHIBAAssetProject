"""tests/live/test_predictive_candidate_ranker.py

predictive_candidate_ranker — unit tests
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.live.predictive_candidate_ranker import (
    CandidateRankingResult,
    PredictiveTag,
    RankedCandidate,
    TAG_FALSE_EXP,
    TAG_PRED_HIGH,
    TAG_SPIKE_RISK,
    TAG_STRONG_CONT,
    W_CONT,
    W_EFF,
    W_PRED,
    _build_tags,
    _compute_spike_risk,
    _heuristic_continuation_probability,
    _normalize_scores,
    _safe_float,
    append_ranking_record,
    format_predictive_display,
    load_ranking_records,
    rank_with_predictive,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _buy_sig(symbol: str, **kw) -> dict:
    return {"symbol": symbol, "action": "BUY", **kw}


def _pred_result(
    scores: dict[str, float] | None = None,
    igniting: dict[str, bool] | None = None,
    leaders: dict[str, bool] | None = None,
    followers: dict[str, bool] | None = None,
    bo_exp: dict[str, float] | None = None,
    vol_r5: dict[str, float] | None = None,
) -> dict:
    sc: dict = {}
    if scores:
        sc["predictive_alpha_score"] = scores
    if igniting:
        sc["sector_ignition_active"] = igniting
    if leaders:
        sc["is_leader"] = leaders
    if followers:
        sc["is_follower"] = followers
    if bo_exp:
        sc["breakout_expansion"] = bo_exp
    if vol_r5:
        sc["volume_ratio_5d"] = vol_r5
    return {"scores": sc}


# ─────────────────────────────────────────────────────────────────────────────
# Weight assertions
# ─────────────────────────────────────────────────────────────────────────────

def test_weights_sum_to_one():
    assert abs(W_EFF + W_PRED + W_CONT - 1.0) < 1e-9


# ─────────────────────────────────────────────────────────────────────────────
# _safe_float
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_float_normal():
    assert _safe_float(1.5) == 1.5


def test_safe_float_nan_returns_default():
    import math
    assert _safe_float(math.nan, 0.0) == 0.0


def test_safe_float_inf_returns_default():
    import math
    assert _safe_float(math.inf, -1.0) == -1.0


def test_safe_float_non_numeric():
    assert _safe_float("bad", 99.0) == 99.0


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_scores
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_single_symbol_returns_fallback():
    result = _normalize_scores({"A": 50.0}, fallback=0.5)
    assert result["A"] == 0.5


def test_normalize_two_symbols_extremes():
    result = _normalize_scores({"A": 10.0, "B": 90.0})
    assert result["A"] < result["B"]


def test_normalize_preserves_order():
    raw = {"A": 10.0, "B": 30.0, "C": 80.0}
    norm = _normalize_scores(raw)
    assert norm["A"] < norm["B"] < norm["C"]


def test_normalize_all_same_values():
    raw = {"A": 50.0, "B": 50.0, "C": 50.0}
    norm = _normalize_scores(raw, fallback=0.5)
    # all identical → fallback for all
    assert all(v == 0.5 for v in norm.values())


def test_normalize_empty():
    assert _normalize_scores({}) == {}


# ─────────────────────────────────────────────────────────────────────────────
# _compute_spike_risk
# ─────────────────────────────────────────────────────────────────────────────

def test_spike_risk_high_breakout_low_volume():
    scores = {
        "breakout_expansion": {"X": 2.0},
        "volume_ratio_5d":    {"X": 0.5},
    }
    risk = _compute_spike_risk("X", scores)
    assert risk > 0.0


def test_spike_risk_no_breakout():
    scores = {
        "breakout_expansion": {"X": 1.0},  # below 1.5 threshold
        "volume_ratio_5d":    {"X": 0.5},
    }
    risk = _compute_spike_risk("X", scores)
    assert risk == 0.0


def test_spike_risk_high_breakout_high_volume():
    scores = {
        "breakout_expansion": {"X": 2.0},
        "volume_ratio_5d":    {"X": 1.5},   # above 0.90 threshold
    }
    risk = _compute_spike_risk("X", scores)
    assert risk == 0.0


def test_spike_risk_missing_symbol():
    scores = {"breakout_expansion": {}, "volume_ratio_5d": {}}
    risk = _compute_spike_risk("NOTEXIST", scores)
    assert risk == 0.0


def test_spike_risk_bounded_to_one():
    scores = {
        "breakout_expansion": {"X": 100.0},   # extreme breakout
        "volume_ratio_5d":    {"X": 0.0},     # zero volume
    }
    risk = _compute_spike_risk("X", scores)
    assert 0.0 <= risk <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# _build_tags
# ─────────────────────────────────────────────────────────────────────────────

def test_build_tags_pred_high():
    tags, expl = _build_tags("A", 80.0, False, False, 0.0, 0.0)
    tag_names = [t.name for t in tags]
    assert TAG_PRED_HIGH in tag_names
    assert any(TAG_PRED_HIGH in e for e in expl)


def test_build_tags_strong_continuation():
    tags, expl = _build_tags("A", 75.0, True, False, 0.0, 0.0)
    tag_names = [t.name for t in tags]
    assert TAG_STRONG_CONT in tag_names


def test_build_tags_follower_strong_continuation():
    tags, _ = _build_tags("A", 65.0, False, True, 0.0, 0.0)
    tag_names = [t.name for t in tags]
    assert TAG_STRONG_CONT in tag_names


def test_build_tags_no_strong_cont_without_role():
    tags, _ = _build_tags("A", 75.0, False, False, 0.0, 0.0)
    tag_names = [t.name for t in tags]
    assert TAG_STRONG_CONT not in tag_names


def test_build_tags_spike_risk():
    tags, _ = _build_tags("A", 40.0, False, False, 0.5, 0.0)
    tag_names = [t.name for t in tags]
    assert TAG_SPIKE_RISK in tag_names


def test_build_tags_spike_risk_below_threshold():
    tags, _ = _build_tags("A", 40.0, False, False, 0.05, 0.0)
    tag_names = [t.name for t in tags]
    assert TAG_SPIKE_RISK not in tag_names


def test_build_tags_false_expansion():
    tags, _ = _build_tags("A", 40.0, False, False, 0.0, 0.5)
    tag_names = [t.name for t in tags]
    assert TAG_FALSE_EXP in tag_names


def test_build_tags_no_false_expansion_below_threshold():
    tags, _ = _build_tags("A", 40.0, False, False, 0.0, 0.2)
    tag_names = [t.name for t in tags]
    assert TAG_FALSE_EXP not in tag_names


def test_build_tags_no_tags_neutral():
    tags, expl = _build_tags("A", 30.0, False, False, 0.0, 0.0)
    assert tags == []
    assert expl == []


# ─────────────────────────────────────────────────────────────────────────────
# _heuristic_continuation_probability
# ─────────────────────────────────────────────────────────────────────────────

def test_heuristic_cont_prob_high_score():
    p = _heuristic_continuation_probability(90.0, False, False, 0.0)
    assert p > 0.8


def test_heuristic_cont_prob_sector_igniting_bonus():
    p_base = _heuristic_continuation_probability(60.0, False, False, 0.0)
    p_ign  = _heuristic_continuation_probability(60.0, True, False, 0.0)
    assert p_ign > p_base


def test_heuristic_cont_prob_follower_bonus():
    p_base = _heuristic_continuation_probability(60.0, False, False, 0.0)
    p_fol  = _heuristic_continuation_probability(60.0, False, True, 0.0)
    assert p_fol > p_base


def test_heuristic_cont_prob_spike_penalty():
    p_no   = _heuristic_continuation_probability(60.0, False, False, 0.0)
    p_risk = _heuristic_continuation_probability(60.0, False, False, 1.0)
    assert p_risk < p_no


def test_heuristic_cont_prob_bounded():
    p = _heuristic_continuation_probability(100.0, True, True, 0.0)
    assert 0.0 <= p <= 1.0
    p2 = _heuristic_continuation_probability(0.0, False, False, 1.0)
    assert 0.0 <= p2 <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# rank_with_predictive — core ranking
# ─────────────────────────────────────────────────────────────────────────────

def test_rank_empty_buy_signals():
    result = rank_with_predictive([], {}, {})
    assert result.n_candidates == 0
    assert result.ranked == []


def test_rank_single_candidate():
    signals = [_buy_sig("A")]
    result = rank_with_predictive(
        signals,
        _pred_result(scores={"A": 75.0}),
        {"A": 0.8},
    )
    assert result.n_candidates == 1
    assert result.ranked[0].symbol == "A"
    assert result.ranked[0].combined_rank == 1


def test_rank_deterministic_same_input():
    signals = [_buy_sig("A"), _buy_sig("B"), _buy_sig("C")]
    pred = _pred_result(scores={"A": 80.0, "B": 60.0, "C": 50.0})
    eff  = {"A": 0.9, "B": 0.7, "C": 0.5}
    r1 = rank_with_predictive(signals, pred, eff)
    r2 = rank_with_predictive(signals, pred, eff)
    assert [c.symbol for c in r1.ranked] == [c.symbol for c in r2.ranked]


def test_rank_alphabetical_tiebreak():
    # All pred scores identical + all eff scores identical
    # → _normalize_scores returns fallback (0.5) for both
    # → no sector bonus → combined scores truly equal → alphabetical tiebreak
    signals = [_buy_sig("Z"), _buy_sig("A"), _buy_sig("M")]
    pred = _pred_result(scores={"Z": 50.0, "A": 50.0, "M": 50.0})
    eff  = {"Z": 0.5, "A": 0.5, "M": 0.5}
    result = rank_with_predictive(signals, pred, eff)
    syms = [c.symbol for c in result.ranked]
    assert syms == sorted(syms)


def test_rank_high_pred_score_outranks_low():
    signals = [_buy_sig("LOW"), _buy_sig("HIGH")]
    pred = _pred_result(scores={"LOW": 10.0, "HIGH": 95.0})
    eff  = {"LOW": 0.5, "HIGH": 0.5}
    result = rank_with_predictive(signals, pred, eff)
    assert result.ranked[0].symbol == "HIGH"


def test_rank_spike_risk_penalty_applied():
    # HIGH_PRED has high pred score but high spike risk → should be penalized
    signals = [_buy_sig("A"), _buy_sig("B")]
    pred = _pred_result(
        scores={"A": 90.0, "B": 80.0},
        bo_exp={"A": 3.0},
        vol_r5={"A": 0.3},
    )
    eff = {"A": 0.8, "B": 0.8}
    result = rank_with_predictive(signals, pred, eff)
    cand_a = next(c for c in result.ranked if c.symbol == "A")
    assert cand_a.spike_risk > 0.0
    # combined_score should be reduced by the penalty
    assert cand_a.combined_score < 1.0


def test_rank_sector_ignition_bonus():
    signals = [_buy_sig("IGN"), _buy_sig("PLAIN")]
    pred = _pred_result(
        scores={"IGN": 60.0, "PLAIN": 60.0},
        igniting={"IGN": True},
    )
    eff = {"IGN": 0.5, "PLAIN": 0.5}
    result = rank_with_predictive(signals, pred, eff)
    ign_cand = next(c for c in result.ranked if c.symbol == "IGN")
    plain_cand = next(c for c in result.ranked if c.symbol == "PLAIN")
    assert ign_cand.combined_score >= plain_cand.combined_score


def test_rank_follower_bonus():
    signals = [_buy_sig("FOL"), _buy_sig("PLAIN")]
    pred = _pred_result(
        scores={"FOL": 60.0, "PLAIN": 60.0},
        followers={"FOL": True},
    )
    eff = {"FOL": 0.5, "PLAIN": 0.5}
    result = rank_with_predictive(signals, pred, eff)
    fol_cand = next(c for c in result.ranked if c.symbol == "FOL")
    plain_cand = next(c for c in result.ranked if c.symbol == "PLAIN")
    assert fol_cand.combined_score >= plain_cand.combined_score


def test_rank_combined_score_bounded():
    signals = [_buy_sig("A"), _buy_sig("B"), _buy_sig("C")]
    pred = _pred_result(scores={"A": 100.0, "B": 50.0, "C": 0.0})
    eff  = {"A": 1.0, "B": 0.5, "C": 0.0}
    result = rank_with_predictive(signals, pred, eff)
    for c in result.ranked:
        assert 0.0 <= c.combined_score <= 1.0


def test_rank_combined_rank_is_sequential():
    signals = [_buy_sig("A"), _buy_sig("B"), _buy_sig("C")]
    pred = _pred_result(scores={"A": 80.0, "B": 60.0, "C": 40.0})
    eff  = {"A": 0.8, "B": 0.6, "C": 0.4}
    result = rank_with_predictive(signals, pred, eff)
    ranks = [c.combined_rank for c in result.ranked]
    assert ranks == list(range(1, len(ranks) + 1))


def test_rank_no_predictive_fallback_to_efficiency():
    signals = [_buy_sig("A"), _buy_sig("B")]
    result = rank_with_predictive(
        signals,
        {},           # no predictive
        {"A": 0.9, "B": 0.2},
    )
    assert result.predictive_available is False
    assert result.ranked[0].symbol == "A"
    assert any("predictive_result empty" in w for w in result.warnings)


def test_rank_no_efficiency_fallback_to_predictive():
    signals = [_buy_sig("A"), _buy_sig("B")]
    result = rank_with_predictive(
        signals,
        _pred_result(scores={"A": 90.0, "B": 20.0}),
        {},
    )
    assert result.efficiency_available is False
    assert result.ranked[0].symbol == "A"


def test_rank_shadow_continuation_prob_used():
    signals = [_buy_sig("A")]
    result = rank_with_predictive(
        signals,
        _pred_result(scores={"A": 60.0}),
        {"A": 0.5},
        shadow_continuation_probs={"A": 0.99},
    )
    assert result.ranked[0].continuation_probability == 0.99


def test_rank_custom_weights_valid():
    signals = [_buy_sig("A"), _buy_sig("B")]
    pred = _pred_result(scores={"A": 80.0, "B": 40.0})
    eff  = {"A": 0.5, "B": 0.9}
    # Pure efficiency weight
    result = rank_with_predictive(
        signals, pred, eff,
        weights={"eff": 1.0, "pred": 0.0, "cont": 0.0},
    )
    assert result.ranked[0].symbol == "B"


def test_rank_custom_weights_invalid_sum():
    signals = [_buy_sig("A")]
    result = rank_with_predictive(
        signals,
        _pred_result(scores={"A": 80.0}),
        {"A": 0.5},
        weights={"eff": 0.9, "pred": 0.9, "cont": 0.0},  # sum>1
    )
    assert any("weights_sum" in w for w in result.warnings)


def test_rank_false_expansion_rates_affect_tags():
    signals = [_buy_sig("A")]
    result = rank_with_predictive(
        signals,
        _pred_result(scores={"A": 60.0}),
        {"A": 0.5},
        false_expansion_rates={"A": 0.5},  # 50% > 30% threshold
    )
    tag_names = [t.name for t in result.ranked[0].tags]
    assert TAG_FALSE_EXP in tag_names


def test_rank_symbol_with_missing_signal_key():
    signals = [{"action": "BUY"}]  # no symbol
    result = rank_with_predictive(signals, {}, {})
    assert result.n_candidates == 0


def test_rank_run_ts_and_run_id_stored():
    signals = [_buy_sig("A")]
    result = rank_with_predictive(
        signals, {}, {}, run_ts="2026-05-23T09:00:00+0900", run_id="TEST_RUN"
    )
    assert result.run_ts == "2026-05-23T09:00:00+0900"
    assert result.run_id == "TEST_RUN"


# ─────────────────────────────────────────────────────────────────────────────
# RankedCandidate.to_dict / CandidateRankingResult.to_dict
# ─────────────────────────────────────────────────────────────────────────────

def test_ranked_candidate_to_dict_serializable():
    signals = [_buy_sig("A")]
    result = rank_with_predictive(
        signals, _pred_result(scores={"A": 75.0}), {"A": 0.6},
    )
    d = result.ranked[0].to_dict()
    assert isinstance(d, dict)
    assert d["symbol"] == "A"
    # tags are dicts, not PredictiveTag objects
    for t in d["tags"]:
        assert isinstance(t, dict)
        assert "name" in t


def test_candidate_ranking_result_to_dict():
    signals = [_buy_sig("A"), _buy_sig("B")]
    result = rank_with_predictive(
        signals,
        _pred_result(scores={"A": 70.0, "B": 50.0}),
        {"A": 0.7, "B": 0.5},
        run_id="R1",
    )
    d = result.to_dict()
    assert d["run_id"] == "R1"
    assert d["n_candidates"] == 2
    assert len(d["ranked"]) == 2
    # JSON serializable
    json.dumps(d)


# ─────────────────────────────────────────────────────────────────────────────
# append_ranking_record / load_ranking_records
# ─────────────────────────────────────────────────────────────────────────────

def test_append_and_load_roundtrip(tmp_path):
    path = tmp_path / "rankings.jsonl"
    signals = [_buy_sig("A"), _buy_sig("B")]
    result = rank_with_predictive(
        signals,
        _pred_result(scores={"A": 80.0, "B": 50.0}),
        {"A": 0.7, "B": 0.4},
        run_id="R1",
    )
    append_ranking_record(result, path)
    records = load_ranking_records(path)
    assert len(records) == 1
    assert records[0]["run_id"] == "R1"
    assert records[0]["n_candidates"] == 2


def test_append_multiple_records(tmp_path):
    path = tmp_path / "rankings.jsonl"
    for i in range(3):
        signals = [_buy_sig(f"S{i}")]
        result = rank_with_predictive(signals, {}, {}, run_id=f"R{i}")
        append_ranking_record(result, path)
    records = load_ranking_records(path)
    assert len(records) == 3
    assert {r["run_id"] for r in records} == {"R0", "R1", "R2"}


def test_load_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    assert load_ranking_records(path) == []


def test_load_nonexistent_file(tmp_path):
    assert load_ranking_records(tmp_path / "nofile.jsonl") == []


def test_load_malformed_last_line(tmp_path):
    path = tmp_path / "rankings.jsonl"
    good = json.dumps({"run_id": "R1", "n_candidates": 0, "ranked": []})
    path.write_text(good + "\n{broken json\n", encoding="utf-8")
    records = load_ranking_records(path)
    assert len(records) == 1
    assert records[0]["run_id"] == "R1"


def test_append_creates_parent_dir(tmp_path):
    path = tmp_path / "sub" / "deep" / "rankings.jsonl"
    result = rank_with_predictive([_buy_sig("A")], {}, {})
    append_ranking_record(result, path)
    assert path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# format_predictive_display
# ─────────────────────────────────────────────────────────────────────────────

def test_format_display_empty():
    result = rank_with_predictive([], {}, {})
    output = format_predictive_display(result)
    assert "候補なし" in output


def test_format_display_contains_symbols():
    signals = [_buy_sig("TOYOTA"), _buy_sig("SONY")]
    pred = _pred_result(scores={"TOYOTA": 80.0, "SONY": 60.0})
    result = rank_with_predictive(signals, pred, {"TOYOTA": 0.8, "SONY": 0.6})
    output = format_predictive_display(result)
    assert "TOYOTA" in output
    assert "SONY" in output


def test_format_display_contains_header():
    signals = [_buy_sig("A")]
    result = rank_with_predictive(signals, {}, {})
    output = format_predictive_display(result)
    assert "PREDICTIVE_RANKING" in output


def test_format_display_shows_at_most_10():
    signals = [_buy_sig(f"SYM{i:02d}") for i in range(20)]
    pred_scores = {f"SYM{i:02d}": float(i) for i in range(20)}
    result = rank_with_predictive(
        signals, _pred_result(scores=pred_scores), {}
    )
    output = format_predictive_display(result)
    count = sum(1 for line in output.split("\n") if "SYM" in line)
    assert count <= 10


def test_format_display_shows_warnings():
    signals = [_buy_sig("A")]
    result = rank_with_predictive(signals, {}, {})  # no pred → warning
    output = format_predictive_display(result)
    assert "WARN" in output


def test_format_display_tags_shown():
    signals = [_buy_sig("A")]
    pred = _pred_result(scores={"A": 90.0})  # → PRED_HIGH tag
    result = rank_with_predictive(signals, pred, {"A": 0.8})
    output = format_predictive_display(result)
    # tag shortened in display
    assert "PRED" in output or "EXPANSION" in output
