"""
tests/analytics/test_early_promotion_telemetry.py
Tests for early_promotion_telemetry module.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analytics.early_promotion_telemetry import (
    EarlyPromotionSignal,
    evaluate_early_promotion,
    append_early_promotion,
    load_early_promotion,
    run_early_promotion_telemetry_hook,
    _EARLY_ACCUM_MIN,
    _EARLY_DRIFT_MIN,
    _EARLY_CONSEC_MIN,
    _EARLY_ZONE,
    _EARLY_SCORE_MIN,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _passing_kwargs(**overrides) -> dict:
    base = dict(
        symbol="7203.T",
        signal_date="2026-05-25",
        future_leader_score=75.0,
        rsr_zone="weak",
        accum_score=0.25,
        drift_5d=0.25,
        consecutive_days=4,
    )
    base.update(overrides)
    return base


# ── Tests: evaluate_early_promotion ───────────────────────────────────────────

class TestEvaluateEarlyPromotion:
    def test_all_conditions_met(self):
        sig = evaluate_early_promotion(**_passing_kwargs())
        assert sig.condition_met is True
        assert sig.observation_only is True

    def test_observation_only_always_true(self):
        sig = evaluate_early_promotion(**_passing_kwargs())
        assert sig.observation_only is True

    def test_low_accum_fails(self):
        sig = evaluate_early_promotion(**_passing_kwargs(accum_score=_EARLY_ACCUM_MIN - 0.01))
        assert sig.condition_met is False
        assert sig.conditions_detail["accum_score"]["met"] is False

    def test_low_drift_fails(self):
        sig = evaluate_early_promotion(**_passing_kwargs(drift_5d=_EARLY_DRIFT_MIN - 0.01))
        assert sig.condition_met is False
        assert sig.conditions_detail["drift_5d"]["met"] is False

    def test_low_consecutive_fails(self):
        sig = evaluate_early_promotion(**_passing_kwargs(consecutive_days=_EARLY_CONSEC_MIN - 1))
        assert sig.condition_met is False
        assert sig.conditions_detail["consecutive_days"]["met"] is False

    def test_wrong_zone_fails(self):
        sig = evaluate_early_promotion(**_passing_kwargs(rsr_zone="emerging"))
        assert sig.condition_met is False
        assert sig.conditions_detail["rsr_zone"]["met"] is False

    def test_mature_zone_fails(self):
        sig = evaluate_early_promotion(**_passing_kwargs(rsr_zone="mature"))
        assert sig.condition_met is False

    def test_low_score_fails(self):
        sig = evaluate_early_promotion(**_passing_kwargs(future_leader_score=_EARLY_SCORE_MIN - 1))
        assert sig.condition_met is False
        assert sig.conditions_detail["future_leader_score"]["met"] is False

    def test_exact_thresholds_pass(self):
        sig = evaluate_early_promotion(
            symbol="7203.T",
            signal_date="2026-05-25",
            future_leader_score=_EARLY_SCORE_MIN,
            rsr_zone=_EARLY_ZONE,
            accum_score=_EARLY_ACCUM_MIN,
            drift_5d=_EARLY_DRIFT_MIN,
            consecutive_days=_EARLY_CONSEC_MIN,
        )
        assert sig.condition_met is True

    def test_conditions_detail_structure(self):
        sig = evaluate_early_promotion(**_passing_kwargs())
        assert "accum_score" in sig.conditions_detail
        assert "drift_5d" in sig.conditions_detail
        assert "consecutive_days" in sig.conditions_detail
        assert "rsr_zone" in sig.conditions_detail
        assert "future_leader_score" in sig.conditions_detail

    def test_observation_only_invariant_enforced(self):
        sig = evaluate_early_promotion(**_passing_kwargs())
        with pytest.raises(ValueError):
            # Manually bypass — should raise
            object.__setattr__(sig, "observation_only", False)
            sig.__post_init__()


class TestEarlyPromotionJSONL:
    def test_append_and_load(self, tmp_path):
        sig = evaluate_early_promotion(**_passing_kwargs())
        append_early_promotion(sig, tmp_path)
        loaded = load_early_promotion(tmp_path, date_str="20260525")
        assert len(loaded) == 1
        assert loaded[0].symbol == "7203.T"
        assert loaded[0].condition_met is True
        assert loaded[0].observation_only is True

    def test_multiple_appends(self, tmp_path):
        for sym in ["7203.T", "9984.T"]:
            sig = evaluate_early_promotion(**_passing_kwargs(symbol=sym))
            append_early_promotion(sig, tmp_path)
        loaded = load_early_promotion(tmp_path, date_str="20260525")
        assert len(loaded) == 2

    def test_load_missing_dir_returns_empty(self, tmp_path):
        result = load_early_promotion(tmp_path / "nonexistent")
        assert result == []

    def test_observation_only_enforced_on_load(self, tmp_path):
        sig = evaluate_early_promotion(**_passing_kwargs())
        append_early_promotion(sig, tmp_path)
        # Tamper with file: set observation_only=False
        path = tmp_path / "early_promotion_20260525.jsonl"
        lines = path.read_text(encoding="utf-8").splitlines()
        d = json.loads(lines[0])
        d["observation_only"] = False
        path.write_text(json.dumps(d) + "\n", encoding="utf-8")
        # Load should enforce observation_only=True regardless
        loaded = load_early_promotion(tmp_path, date_str="20260525")
        assert loaded[0].observation_only is True


class TestRunEarlyPromotionTelemetryHook:
    def test_hook_detects_candidates(self, tmp_path):
        candidates = [
            _passing_kwargs(symbol="7203.T"),
            _passing_kwargs(symbol="9984.T"),
            _passing_kwargs(symbol="6758.T", future_leader_score=40.0),  # fails score
        ]
        fired = run_early_promotion_telemetry_hook(candidates, tmp_path, "2026-05-25")
        assert len(fired) == 2
        syms = {s.symbol for s in fired}
        assert "7203.T" in syms
        assert "9984.T" in syms
        assert "6758.T" not in syms

    def test_hook_writes_jsonl(self, tmp_path):
        candidates = [_passing_kwargs(symbol="7203.T")]
        run_early_promotion_telemetry_hook(candidates, tmp_path, "2026-05-25")
        assert (tmp_path / "early_promotion_20260525.jsonl").exists()

    def test_hook_no_candidates(self, tmp_path):
        fired = run_early_promotion_telemetry_hook([], tmp_path, "2026-05-25")
        assert fired == []

    def test_hook_all_fail_conditions(self, tmp_path):
        candidates = [{"symbol": "7203.T", "future_leader_score": 30.0,
                       "rsr_zone": "emerging", "accum_score": 0.05,
                       "drift_5d": 0.05, "consecutive_days": 1}]
        fired = run_early_promotion_telemetry_hook(candidates, tmp_path, "2026-05-25")
        assert fired == []

    def test_no_live_universe_mutation(self, tmp_path):
        """Telemetry hook must not write to LIVE_UNIVERSE — check it does not touch any universe files."""
        import os
        before_files = set(str(f) for f in tmp_path.rglob("*"))
        candidates = [_passing_kwargs()]
        run_early_promotion_telemetry_hook(candidates, tmp_path, "2026-05-25")
        after_files = set(str(f) for f in tmp_path.rglob("*"))
        new_files = after_files - before_files
        # Only allowed: early_promotion_*.jsonl in log_dir; no universe/yaml files
        for f in new_files:
            fname = Path(f).name
            assert fname.startswith("early_promotion"), f"unexpected file written: {fname}"
            assert not fname.endswith((".yaml", ".json")) or "universe" not in fname
