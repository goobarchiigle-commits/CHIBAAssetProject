"""
tests/universe/test_rsr_fail_closed.py

RSR FAIL_CLOSED hardening tests.
Verifies that RSR未取得時に昇格ゲートが無効化されない。

Required scenarios:
  1. snapshot exists → rsr_source="snapshot", promotions allowed
  2. snapshot missing + mtf exists → rsr_source="mtf", promotions allowed
  3. snapshot missing + mtf missing → rsr_source="missing", promotions BLOCKED
  4. rsr=59.9 (below MIN_RSR_FOR_SHADOW=60.0) → BLOCKED
  5. rsr=60.0 (exact threshold) → PASSES rsr gate
  6. rsr=60.1 (above threshold) → PASSES rsr gate
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from universe.deployable_universe_governance import (
    DeployabilityScorer,
    DeployableUniverseGovernor,
    run_universe_governance,
    ACTION_PROMOTE,
    ACTION_REJECT,
    AUTO_PROMOTE_SAFE,
    MAX_UNIT_PRICE_FRACTION,
    LOT_SIZE,
    MIN_RSR_FOR_SHADOW,
)

CAPITAL = 3_000_000
MAX_ALLOC = CAPITAL * MAX_UNIT_PRICE_FRACTION


def _affordable_price(capital: int = CAPITAL) -> float:
    return (capital * MAX_UNIT_PRICE_FRACTION * 0.5) / LOT_SIZE


def _price_lookup(symbol: str) -> float:
    return _affordable_price()


LIVE = {"8035.T": "電機精密", "8306.T": "銀行"}
SHADOW = {"3402.T": "繊維"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. snapshot exists → promotions allowed when RSR passes
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotPath:

    def test_snapshot_rsr_source_set(self):
        scorer = DeployabilityScorer(CAPITAL, LIVE)
        scores = scorer.score_all(SHADOW, _price_lookup, {"3402.T": 70.0}, rsr_source="snapshot")
        s = next(x for x in scores if x.symbol == "3402.T")
        assert s.rsr_source == "snapshot"

    def test_snapshot_high_rsr_passes_gate(self):
        scorer = DeployabilityScorer(CAPITAL, LIVE)
        s = scorer.score_candidate("3402.T", "繊維", _price_lookup, {"3402.T": 70.0})
        assert s.rsr_passes is True
        assert s.rsr_source != "missing"

    def test_snapshot_high_rsr_can_promote(self, tmp_path):
        d = tmp_path / "uni"
        d.mkdir()
        result = run_universe_governance(
            live_universe=LIVE,
            shadow_universe=SHADOW,
            capital=CAPITAL,
            run_id="snap_test",
            universe_dir=d,
            price_lookup=_price_lookup,
            rsr_scores={"3402.T": 70.0},
            rsr_source="snapshot",
        )
        # RSR gate passes; promotion depends on deployability_score too
        # Verify no rsr_below_threshold rejection
        rsr_rejects = [d for d in result.decisions if "rsr_below_threshold" in d.reason]
        assert rsr_rejects == []
        # Verify rsr_source recorded on decisions
        for dec in result.decisions:
            if dec.symbol == "3402.T":
                assert dec.rsr_source == "snapshot"


# ─────────────────────────────────────────────────────────────────────────────
# 2. snapshot missing + mtf exists → rsr_source="mtf", promotions allowed
# ─────────────────────────────────────────────────────────────────────────────

class TestMtfFallback:

    def test_mtf_rsr_source_set(self):
        scorer = DeployabilityScorer(CAPITAL, LIVE)
        scores = scorer.score_all(SHADOW, _price_lookup, {"3402.T": 70.0}, rsr_source="mtf")
        s = next(x for x in scores if x.symbol == "3402.T")
        assert s.rsr_source == "mtf"

    def test_mtf_high_rsr_passes_gate(self):
        scorer = DeployabilityScorer(CAPITAL, LIVE)
        s = scorer.score_candidate("3402.T", "繊維", _price_lookup, {"3402.T": 70.0})
        s.rsr_source = "mtf"
        assert s.rsr_passes is True

    def test_mtf_high_rsr_can_promote(self, tmp_path):
        d = tmp_path / "uni"
        d.mkdir()
        result = run_universe_governance(
            live_universe=LIVE,
            shadow_universe=SHADOW,
            capital=CAPITAL,
            run_id="mtf_test",
            universe_dir=d,
            price_lookup=_price_lookup,
            rsr_scores={"3402.T": 70.0},
            rsr_source="mtf",
        )
        rsr_rejects = [d for d in result.decisions if "rsr_below_threshold" in d.reason]
        assert rsr_rejects == []
        for dec in result.decisions:
            if dec.symbol == "3402.T":
                assert dec.rsr_source == "mtf"


# ─────────────────────────────────────────────────────────────────────────────
# 3. snapshot missing + mtf missing → rsr_source="missing", promotions BLOCKED
# ─────────────────────────────────────────────────────────────────────────────

class TestMissingRsr:

    def test_missing_rsr_fails_gate_at_scorer_level(self):
        scorer = DeployabilityScorer(CAPITAL, LIVE)
        s = scorer.score_candidate("3402.T", "繊維", _price_lookup, None)
        assert s.rsr_passes is False
        assert s.rsr_source == "missing"
        assert s.passes_gate is False

    def test_missing_rsr_source_propagates_in_score_all(self):
        scorer = DeployabilityScorer(CAPITAL, LIVE)
        scores = scorer.score_all(SHADOW, _price_lookup, None)
        for s in scores:
            assert s.rsr_source == "missing"
            assert s.rsr_passes is False

    def test_missing_rsr_zero_promotions(self, tmp_path):
        d = tmp_path / "uni"
        d.mkdir()
        result = run_universe_governance(
            live_universe=LIVE,
            shadow_universe=SHADOW,
            capital=CAPITAL,
            run_id="missing_rsr_test",
            universe_dir=d,
            price_lookup=_price_lookup,
            rsr_scores=None,
        )
        assert result.promoted_symbols == [], (
            f"RSR未取得時に昇格が発生: {result.promoted_symbols}"
        )

    def test_missing_rsr_decision_records_missing_source(self, tmp_path):
        d = tmp_path / "uni"
        d.mkdir()
        result = run_universe_governance(
            live_universe=LIVE,
            shadow_universe=SHADOW,
            capital=CAPITAL,
            run_id="missing_src_test",
            universe_dir=d,
            price_lookup=_price_lookup,
            rsr_scores=None,
        )
        for dec in result.decisions:
            if dec.symbol == "3402.T":
                assert dec.rsr_source == "missing"

    def test_missing_rsr_with_multiple_candidates(self, tmp_path):
        d = tmp_path / "uni"
        d.mkdir()
        shadow_multi = {"3402.T": "繊維", "2802.T": "食品", "9020.T": "陸運"}
        result = run_universe_governance(
            live_universe=LIVE,
            shadow_universe=shadow_multi,
            capital=CAPITAL,
            run_id="multi_missing_test",
            universe_dir=d,
            price_lookup=_price_lookup,
            rsr_scores=None,
        )
        assert result.promoted_symbols == [], (
            f"RSR未取得時に昇格が発生: {result.promoted_symbols}"
        )
        for dec in result.decisions:
            assert dec.rsr_source == "missing"


# ─────────────────────────────────────────────────────────────────────────────
# 4. rsr=59.9 → BLOCKED (below MIN_RSR_FOR_SHADOW=60.0)
# ─────────────────────────────────────────────────────────────────────────────

class TestRsrThresholdBoundary:

    def test_rsr_59_9_blocked(self):
        scorer = DeployabilityScorer(CAPITAL, LIVE)
        s = scorer.score_candidate("3402.T", "繊維", _price_lookup, {"3402.T": 59.9})
        assert s.rsr_passes is False
        assert s.passes_gate is False

    def test_rsr_59_9_not_promoted(self, tmp_path):
        d = tmp_path / "uni"
        d.mkdir()
        result = run_universe_governance(
            live_universe=LIVE,
            shadow_universe=SHADOW,
            capital=CAPITAL,
            run_id="rsr599_test",
            universe_dir=d,
            price_lookup=_price_lookup,
            rsr_scores={"3402.T": 59.9},
        )
        assert "3402.T" not in result.promoted_symbols

    # ── 5. rsr=60.0 (exact threshold) → PASSES rsr gate ──────────────────────

    def test_rsr_60_0_passes_gate(self):
        scorer = DeployabilityScorer(CAPITAL, LIVE)
        s = scorer.score_candidate("3402.T", "繊維", _price_lookup, {"3402.T": 60.0})
        assert s.rsr_passes is True

    def test_rsr_60_0_no_rsr_rejection(self, tmp_path):
        d = tmp_path / "uni"
        d.mkdir()
        result = run_universe_governance(
            live_universe=LIVE,
            shadow_universe=SHADOW,
            capital=CAPITAL,
            run_id="rsr600_test",
            universe_dir=d,
            price_lookup=_price_lookup,
            rsr_scores={"3402.T": 60.0},
        )
        rsr_rejects = [dec for dec in result.decisions if "rsr_below_threshold" in dec.reason]
        assert rsr_rejects == []

    # ── 6. rsr=60.1 (above threshold) → PASSES rsr gate ─────────────────────

    def test_rsr_60_1_passes_gate(self):
        scorer = DeployabilityScorer(CAPITAL, LIVE)
        s = scorer.score_candidate("3402.T", "繊維", _price_lookup, {"3402.T": 60.1})
        assert s.rsr_passes is True

    def test_rsr_60_1_no_rsr_rejection(self, tmp_path):
        d = tmp_path / "uni"
        d.mkdir()
        result = run_universe_governance(
            live_universe=LIVE,
            shadow_universe=SHADOW,
            capital=CAPITAL,
            run_id="rsr601_test",
            universe_dir=d,
            price_lookup=_price_lookup,
            rsr_scores={"3402.T": 60.1},
        )
        rsr_rejects = [dec for dec in result.decisions if "rsr_below_threshold" in dec.reason]
        assert rsr_rejects == []
