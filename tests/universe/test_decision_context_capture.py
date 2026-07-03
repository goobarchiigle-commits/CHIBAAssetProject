"""tests/universe/test_decision_context_capture.py — Phase 4C-lite tests (8)"""
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.universe.decision_context_capture import (
    ALL_DECISION_TYPES,
    DECISION_AUTO_FREEZE,
    DECISION_DEMOTION,
    DECISION_GOVERNANCE,
    DECISION_PROMOTION,
    DECISION_ROTATION,
    SCHEMA_VERSION,
    DailySummary,
    DecisionContext,
    DecisionContextCapture,
    GovernanceContext,
    MarketContext,
    PortfolioContext,
    UniverseContext,
    _safe_serialize,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tmp_file() -> Path:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    os.unlink(path)   # start absent so capture creates it fresh
    return Path(path)


def _make_dcc(path: Path | None = None) -> DecisionContextCapture:
    if path is None:
        path = _tmp_file()
    return DecisionContextCapture(history_file=path)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_capture_promotion_context():
    """Captures a PROMOTION context and persists it to the JSONL file."""
    p = _tmp_file()
    dcc = _make_dcc(p)
    record = dcc.capture(
        DECISION_PROMOTION,
        universe_context=UniverseContext(live_universe_size=40, shadow_universe_size=8, pending_exit_count=0),
        decision_inputs={"mode": "AUTO_PROMOTE_SAFE", "candidate_count": 8},
        decision_result={"promoted_symbols": ["1234"], "universe_size_after": 41},
    )
    assert record is not None
    assert record.decision_type == DECISION_PROMOTION
    assert record.schema_version == SCHEMA_VERSION
    assert p.exists()
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["decision_type"] == DECISION_PROMOTION
    assert d["universe_context"]["live_universe_size"] == 40
    assert d["decision_result"]["promoted_symbols"] == ["1234"]


def test_capture_rotation_context():
    """Captures a ROTATION context including plan metadata."""
    p = _tmp_file()
    dcc = _make_dcc(p)
    record = dcc.capture(
        DECISION_ROTATION,
        universe_context={"live_universe_size": 38, "shadow_universe_size": 10, "pending_exit_count": 2},
        decision_inputs={"deficit": 4, "candidate_count": 10},
        decision_result={"plan_id": "abc123", "symbols_to_add": ["5678", "9012"], "rotation_score": 0.85},
    )
    assert record is not None
    assert record.decision_type == DECISION_ROTATION
    d = json.loads(p.read_text(encoding="utf-8").strip())
    assert d["decision_result"]["symbols_to_add"] == ["5678", "9012"]
    assert d["decision_inputs"]["deficit"] == 4


def test_capture_governance_context():
    """Captures a GOVERNANCE context including health_score and effective_mode."""
    p = _tmp_file()
    dcc = _make_dcc(p)
    gctx = GovernanceContext(
        health_score=85.0,
        governance_mode="NORMAL",
        auto_freeze_mode="NORMAL",
        effective_mode="NORMAL",
    )
    record = dcc.capture(
        DECISION_GOVERNANCE,
        governance_context=gctx,
        decision_inputs={"checks_run": ["universe_size", "demotion_pipeline"]},
        decision_result={"governance_mode": "NORMAL", "effective_mode": "NORMAL", "mode_confirmed": True},
    )
    assert record is not None
    d = json.loads(p.read_text(encoding="utf-8").strip())
    assert d["governance_context"]["health_score"] == 85.0
    assert d["governance_context"]["effective_mode"] == "NORMAL"
    assert d["decision_result"]["mode_confirmed"] is True


def test_append_only_history():
    """Multiple captures append sequential lines; no line is overwritten."""
    p = _tmp_file()
    dcc = _make_dcc(p)
    dcc.capture(DECISION_PROMOTION, decision_result={"promoted_symbols": ["A"]})
    dcc.capture(DECISION_DEMOTION,  decision_result={"demotion_candidates": ["B"]})
    dcc.capture(DECISION_ROTATION,  decision_result={"symbols_to_add": ["C"]})

    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 3
    types = [json.loads(l)["decision_type"] for l in lines]
    assert types == [DECISION_PROMOTION, DECISION_DEMOTION, DECISION_ROTATION]


def test_fail_open_on_write_failure():
    """capture() returns None and does not raise when the write path is unwritable."""
    # Use a path inside a non-existent sub-sub-directory that can't be created
    # (simulate by pointing to a file path under a file, not a dir)
    p = _tmp_file()
    p.write_text("existing_content", encoding="utf-8")

    # Overwrite the _history_file with a Path that's a file masquerading as dir
    bad_path = p / "subdir" / "file.jsonl"   # p is a file; can't mkdir inside it
    dcc = DecisionContextCapture(history_file=bad_path)

    # Should not raise; returns None on failure
    result = dcc.capture(DECISION_GOVERNANCE, decision_result={"test": True})
    assert result is None


def test_replay_restores_context():
    """replay_decision_context() returns all records for the requested date."""
    p = _tmp_file()
    dcc = _make_dcc(p)

    r1 = dcc.capture(DECISION_PROMOTION, decision_result={"promoted_symbols": ["X"]})
    r2 = dcc.capture(DECISION_GOVERNANCE, decision_result={"governance_mode": "NORMAL"})

    assert r1 is not None and r2 is not None
    date_str = r1.timestamp[:10]   # YYYY-MM-DD from ISO timestamp

    replayed = dcc.replay_decision_context(date_str)
    assert len(replayed) == 2
    assert all(isinstance(r, DecisionContext) for r in replayed)
    types = {r.decision_type for r in replayed}
    assert types == {DECISION_PROMOTION, DECISION_GOVERNANCE}

    # decision_result preserved
    promo = next(r for r in replayed if r.decision_type == DECISION_PROMOTION)
    assert promo.decision_result["promoted_symbols"] == ["X"]


def test_multiple_decisions_same_day():
    """All 5 decision types in one session → daily_summary counts all correctly."""
    p = _tmp_file()
    dcc = _make_dcc(p)

    dcc.capture(DECISION_PROMOTION)
    dcc.capture(DECISION_DEMOTION)
    dcc.capture(DECISION_ROTATION)
    dcc.capture(DECISION_GOVERNANCE)
    dcc.capture(DECISION_AUTO_FREEZE)

    # All were just captured so any record's date works
    lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    date_str = json.loads(lines[0])["timestamp"][:10]

    summary = dcc.daily_summary(date_str)
    assert isinstance(summary, DailySummary)
    assert summary.decision_records == 5
    assert summary.promotion_context_records == 1
    assert summary.demotion_context_records == 1
    assert summary.rotation_context_records == 1
    assert summary.governance_context_records == 1
    assert summary.auto_freeze_context_records == 1


def test_empty_context_safe():
    """capture() with all-None / missing context fields succeeds without error."""
    p = _tmp_file()
    dcc = _make_dcc(p)

    record = dcc.capture(DECISION_AUTO_FREEZE)
    assert record is not None
    assert record.market_context == {}
    assert record.portfolio_context == {}
    assert record.universe_context == {}
    assert record.governance_context == {}
    assert record.decision_inputs == {}
    assert record.decision_result == {}
    assert record.schema_version == SCHEMA_VERSION

    # Replay on non-existent date returns empty list
    empty = dcc.replay_decision_context("1970-01-01")
    assert empty == []
