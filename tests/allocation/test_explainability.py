"""Tests for src/allocation/explainability.py"""
import json
import pytest
from src.allocation.efficiency_ranker import CandidateSignal, EfficiencyScore
from src.allocation.explainability import (
    AllocationDecision,
    record_to_dict,
    append_allocation_decision,
    load_allocation_decisions,
    load_recent_decisions,
    build_allocation_decision,
    SCHEMA_VERSION,
)


def _make_score(symbol="8035", score=0.74):
    return EfficiencyScore(
        symbol=symbol,
        capital_efficiency_score=score,
        expected_geometric_return=0.20,
        liquidity_penalty=1.0,
        correlation_penalty=1.3,
        drawdown_penalty=1.2,
        correlation_with_portfolio=0.30,
        rank=1,
    )


def _make_decision(symbol="8035", decision="SELECTED"):
    score = _make_score(symbol)
    return build_allocation_decision(
        timestamp="2026-05-16T09:00:00",
        trading_day="2026-05-16",
        symbol=symbol,
        decision=decision,
        rank=1,
        efficiency_score=score,
        edge_persistence=0.81,
        regime_confidence=0.68,
        post_cost_alpha_bps=19.0,
        aggressiveness_state="OPPORTUNISTIC",
        concentration_state="LOW",
        effective_independent_n=2.8,
        rejection_reason="" if decision == "SELECTED" else "low score",
    )


# ── record_to_dict ─────────────────────────────────────────────────────────────

def test_record_to_dict_keys():
    d = _make_decision()
    result = record_to_dict(d)
    assert isinstance(result, dict)
    assert "symbol" in result
    assert "capital_efficiency_score" in result
    assert "schema_version" in result


def test_record_to_dict_deterministic():
    d = _make_decision()
    r1 = record_to_dict(d)
    r2 = record_to_dict(d)
    assert r1 == r2


# ── build_allocation_decision ──────────────────────────────────────────────────

def test_build_decision_bounds_edge_persistence():
    score = _make_score()
    dec = build_allocation_decision(
        timestamp="T", trading_day="D", symbol="X",
        decision="SELECTED", rank=1, efficiency_score=score,
        edge_persistence=1.5,  # out of bounds
        regime_confidence=0.68,
        post_cost_alpha_bps=19.0,
        aggressiveness_state="AGGRESSIVE",
        concentration_state="LOW",
        effective_independent_n=3.0,
    )
    assert dec.edge_persistence == pytest.approx(1.0)


def test_build_decision_bounds_regime_confidence():
    score = _make_score()
    dec = build_allocation_decision(
        timestamp="T", trading_day="D", symbol="X",
        decision="SELECTED", rank=1, efficiency_score=score,
        edge_persistence=0.5,
        regime_confidence=-0.5,  # out of bounds
        post_cost_alpha_bps=19.0,
        aggressiveness_state="DEFENSIVE",
        concentration_state="HIGH",
        effective_independent_n=1.0,
    )
    assert dec.regime_confidence == pytest.approx(0.0)


def test_build_decision_rejected_has_reason():
    dec = _make_decision(decision="REJECTED")
    assert dec.rejection_reason != ""
    assert dec.decision == "REJECTED"


# ── append / load ──────────────────────────────────────────────────────────────

def test_append_and_load(tmp_path):
    p = tmp_path / "decisions.jsonl"
    d1 = _make_decision("8035", "SELECTED")
    d2 = _make_decision("7203", "REJECTED")
    append_allocation_decision(d1, p)
    append_allocation_decision(d2, p)
    loaded = load_allocation_decisions(p)
    assert len(loaded) == 2
    assert loaded[0].symbol == "8035"
    assert loaded[1].symbol == "7203"


def test_load_missing_file(tmp_path):
    p = tmp_path / "nonexistent.jsonl"
    loaded = load_allocation_decisions(p)
    assert loaded == []


def test_append_is_append_only(tmp_path):
    p = tmp_path / "decisions.jsonl"
    d1 = _make_decision("A", "SELECTED")
    append_allocation_decision(d1, p)
    d2 = _make_decision("B", "CANDIDATE")
    append_allocation_decision(d2, p)
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2


def test_load_recent_decisions(tmp_path):
    p = tmp_path / "decisions.jsonl"
    for i in range(10):
        d = _make_decision(str(i), "SELECTED")
        append_allocation_decision(d, p)
    recent = load_recent_decisions(p, n=3)
    assert len(recent) == 3
    # Recent = last 3
    assert recent[-1].symbol == "9"


def test_jsonl_lines_valid_json(tmp_path):
    p = tmp_path / "decisions.jsonl"
    for sym in ["A", "B", "C"]:
        append_allocation_decision(_make_decision(sym), p)
    for line in p.read_text(encoding="utf-8").splitlines():
        data = json.loads(line)
        assert "symbol" in data


def test_schema_version_preserved(tmp_path):
    p = tmp_path / "decisions.jsonl"
    d = _make_decision()
    append_allocation_decision(d, p)
    loaded = load_allocation_decisions(p)
    assert loaded[0].schema_version == SCHEMA_VERSION
