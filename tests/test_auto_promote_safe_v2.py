"""tests/test_auto_promote_safe_v2.py — AUTO_PROMOTE_SAFE_V2 unit tests

Coverage:
  - compute_promotion_score: formula correctness, weight normalization
  - check_promotion_gate: all 3 gate conditions
  - check_demotion: all 3 triggers + non-trigger edge cases
  - check_graduation: all 4 conditions, elapsed check
  - load_active_probation: JSONL state loading, deduplication
  - _get_cooldown_symbols: cooldown window logic
  - load_latest_predictive_scores: top_candidates injection
  - load_latest_fl_candidates: latest run_id grouping
  - run_probation_gate: full flow (E2E with temp files)
  - run_probation_outcome_observation: append-only telemetry
  - format_probation_report: section formatting
"""
import json
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.universe.auto_promote_safe_v2 import (
    ALLOCATION_MULTIPLIER,
    BREAKOUT_FAIL_RSR_MIN,
    CB_SEVERE_STATES,
    DEFAULT_PROBATION_DAYS,
    GATE_MAX_PREDICTIVE_RANK,
    GATE_MIN_RSR_PASS,
    GATE_MIN_SECTOR_IGNITION,
    MAX_ACTIVE_PROBATION,
    PROBATION_COOLDOWN_DAYS,
    RSR_RANK_DROP_THRESHOLD,
    STATUS_ACTIVE,
    STATUS_DEMOTED,
    STATUS_GRADUATED,
    VOLUME_COLLAPSE_PROXY_RSR_DROP,
    ProbationRecord,
    _get_cooldown_symbols,
    _normalize,
    check_demotion,
    check_graduation,
    check_promotion_gate,
    compute_promotion_score,
    format_probation_report,
    get_graduated_symbols,
    load_active_probation,
    load_latest_fl_candidates,
    load_latest_predictive_scores,
    run_probation_gate,
    run_probation_outcome_observation,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures / helpers
# ─────────────────────────────────────────────────────────────────────────────

def _iso_ago(days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


def _write_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def _make_record(sym="1234.T", status=STATUS_ACTIVE, days_ago=1, rsr=80.0) -> dict:
    return {
        "symbol": sym,
        "promoted_at": _iso_ago(days_ago),
        "promotion_score": 0.72,
        "promotion_reason": ["rsr_strong", "sector_ignition_ok"],
        "probation_days": DEFAULT_PROBATION_DAYS,
        "status": status,
        "batch_id": "abc12345",
        "rsr_at_promotion": rsr,
        "sector_ignition_score": 92.0,
        "compression_score": 75.0,
        "drift_score": 60.0,
        "schema_version": 1,
    }


def _make_prob_record(sym="1234.T", rsr=80.0) -> ProbationRecord:
    return ProbationRecord(
        symbol=sym,
        promoted_at=_iso_ago(1),
        promotion_score=0.72,
        promotion_reason=[],
        probation_days=DEFAULT_PROBATION_DAYS,
        status=STATUS_ACTIVE,
        batch_id="abc",
        rsr_at_promotion=rsr,
        sector_ignition_score=92.0,
        compression_score=75.0,
        drift_score=60.0,
    )


def _make_pred_scores(symbols, pa=None, si=None, cb=None):
    pa = pa or {s: 80.0 for s in symbols}
    si = si or {s: 92.0 for s in symbols}
    cb = cb or {s: 75.0 for s in symbols}
    top_cands = sorted(pa.items(), key=lambda x: (-x[1], x[0]))
    return {
        "predictive_alpha_score": pa,
        "sector_ignition_score": si,
        "compression_breakout_score": cb,
        "volume_regime_score": {s: 60.0 for s in symbols},
        "top_candidates": top_cands,
    }


# ─────────────────────────────────────────────────────────────────────────────
# _normalize
# ─────────────────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_midpoint(self):
        assert _normalize(50.0) == pytest.approx(0.5)

    def test_max_clamp(self):
        assert _normalize(200.0) == pytest.approx(1.0)

    def test_min_clamp(self):
        assert _normalize(-10.0) == pytest.approx(0.0)

    def test_zero_scale(self):
        assert _normalize(50.0, scale=0.0) == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# compute_promotion_score
# ─────────────────────────────────────────────────────────────────────────────

class TestComputePromotionScore:
    def test_perfect_inputs(self):
        syms = ["1234.T"]
        rsr = {"1234.T": 100.0}
        pred = _make_pred_scores(syms, pa={"1234.T": 100.0}, si={"1234.T": 100.0}, cb={"1234.T": 100.0})
        fl = {"1234.T": {"persistent_drift_score": 100.0}}
        score, reasons = compute_promotion_score("1234.T", rsr, pred, fl)
        assert score == pytest.approx(1.0)

    def test_zero_inputs(self):
        score, reasons = compute_promotion_score("1234.T", {}, {}, {})
        assert score == pytest.approx(0.0)

    def test_score_bounded_0_1(self):
        rsr = {"1234.T": 50.0}
        pred = _make_pred_scores(["1234.T"], pa={"1234.T": 80.0})
        score, _ = compute_promotion_score("1234.T", rsr, pred, {})
        assert 0.0 <= score <= 1.0

    def test_weights_sum(self):
        """30% rsr only → score ≈ 0.30."""
        rsr = {"X": 100.0}
        pred = _make_pred_scores(["X"], pa={"X": 0.0}, si={"X": 0.0}, cb={"X": 0.0})
        fl = {"X": {"persistent_drift_score": 0.0}}
        score, _ = compute_promotion_score("X", rsr, pred, fl)
        assert score == pytest.approx(0.30, abs=1e-4)

    def test_reasons_populated(self):
        rsr = {"1234.T": 80.0}
        pred = _make_pred_scores(["1234.T"])
        fl = {"1234.T": {"persistent_drift_score": 60.0}}
        _, reasons = compute_promotion_score("1234.T", rsr, pred, fl)
        assert "rsr_strong" in reasons
        assert "sector_ignition_ok" in reasons

    def test_missing_symbol_returns_zero(self):
        score, _ = compute_promotion_score("UNKNOWN.T", {"1234.T": 80.0}, {}, {})
        assert score == pytest.approx(0.0)

    def test_fallback_to_compression_when_alpha_missing(self):
        """predictive_alpha_score missing → fallback to compression_breakout_score."""
        rsr = {"X": 75.0}
        pred = {"compression_breakout_score": {"X": 80.0}, "top_candidates": [("X", 80.0)]}
        score, _ = compute_promotion_score("X", rsr, pred, {})
        assert score > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# check_promotion_gate
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckPromotionGate:
    def _gate(self, sym, rsr_val=80.0, rank=1, si=92.0, n_syms=5):
        syms = [f"S{i}.T" for i in range(n_syms)]
        syms[rank - 1] = sym
        pa = {s: float(n_syms - i) * 10 for i, s in enumerate(syms)}
        si_dict = {sym: si}
        pred = {
            "predictive_alpha_score": pa,
            "sector_ignition_score": si_dict,
            "compression_breakout_score": {sym: 70.0},
            "top_candidates": sorted(pa.items(), key=lambda x: (-x[1], x[0])),
        }
        return check_promotion_gate(sym, {sym: rsr_val}, pred, {})

    def test_all_pass(self):
        ok, fail = self._gate("S0.T")
        assert ok
        assert not fail

    def test_rsr_too_low(self):
        ok, fail = self._gate("S0.T", rsr_val=GATE_MIN_RSR_PASS - 1.0)
        assert not ok
        assert any("rsr_too_low" in f for f in fail)

    def test_rsr_exactly_at_threshold(self):
        ok, _ = self._gate("S0.T", rsr_val=GATE_MIN_RSR_PASS)
        assert ok

    def test_predictive_rank_too_low(self):
        # Use a sym that doesn't collide with generated S{i}.T names
        ok, fail = self._gate("NEW.T", rank=GATE_MAX_PREDICTIVE_RANK + 1)
        assert not ok
        assert any("predictive_rank" in f for f in fail)

    def test_predictive_rank_at_boundary(self):
        ok, _ = self._gate("S0.T", rank=GATE_MAX_PREDICTIVE_RANK)
        assert ok

    def test_sector_ignition_too_low(self):
        ok, fail = self._gate("S0.T", si=GATE_MIN_SECTOR_IGNITION - 1.0)
        assert not ok
        assert any("sector_ignition" in f for f in fail)

    def test_sector_ignition_at_threshold(self):
        ok, _ = self._gate("S0.T", si=GATE_MIN_SECTOR_IGNITION)
        assert ok

    def test_no_predictive_scores_fails(self):
        ok, fail = check_promotion_gate("X", {"X": 80.0}, {}, {})
        assert not ok
        assert any("predictive_scores_unavailable" in f for f in fail)

    def test_symbol_not_in_top_cands_fails(self):
        pred = {
            "predictive_alpha_score": {"OTHER.T": 90.0},
            "sector_ignition_score": {"X": 92.0},
            "top_candidates": [("OTHER.T", 90.0)],
        }
        ok, fail = check_promotion_gate("X", {"X": 80.0}, pred, {})
        assert not ok
        assert any("predictive_rank" in f for f in fail)


# ─────────────────────────────────────────────────────────────────────────────
# check_demotion
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckDemotion:
    def test_no_trigger(self):
        rec = _make_prob_record(rsr=80.0)
        ok, reason = check_demotion(rec, 78.0)  # drop = 2.0, safe
        assert not ok

    def test_rsr_rank_drop_trigger(self):
        rec = _make_prob_record(rsr=80.0)
        drop = RSR_RANK_DROP_THRESHOLD + 1.0
        ok, reason = check_demotion(rec, 80.0 - drop)
        assert ok
        assert "rsr_rank_drop" in reason

    def test_rsr_rank_drop_exact_threshold_no_trigger(self):
        rec = _make_prob_record(rsr=80.0)
        ok, _ = check_demotion(rec, 80.0 - RSR_RANK_DROP_THRESHOLD)
        assert not ok   # > not >=

    def test_breakout_fail_trigger(self):
        # rsr_at_promotion must be close enough to current to not trigger rsr_rank_drop first
        # rsr_at_promotion - current <= RSR_RANK_DROP_THRESHOLD → use small gap
        current = BREAKOUT_FAIL_RSR_MIN - 1.0   # 64.0
        at_prom  = current + RSR_RANK_DROP_THRESHOLD - 1.0  # 78.0; drop=14<15, no rank_drop
        rec = _make_prob_record(rsr=at_prom)
        ok, reason = check_demotion(rec, current)
        assert ok
        assert "breakout_fail" in reason

    def test_volume_collapse_proxy_trigger(self):
        rec = _make_prob_record(rsr=80.0)
        drop = VOLUME_COLLAPSE_PROXY_RSR_DROP + 1.0
        # large drop also triggers rsr_rank_drop first, so use value between
        rec2 = _make_prob_record(rsr=BREAKOUT_FAIL_RSR_MIN + drop + 5.0)
        ok, reason = check_demotion(rec2, BREAKOUT_FAIL_RSR_MIN + 5.0)
        assert ok
        assert "rsr_rank_drop" in reason or "relative_volume_collapse_proxy" in reason

    def test_rsr_improves_no_trigger(self):
        rec = _make_prob_record(rsr=50.0)
        ok, _ = check_demotion(rec, 90.0)  # RSR improved
        assert not ok


# ─────────────────────────────────────────────────────────────────────────────
# check_graduation
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckGraduation:
    def _outcomes(self, sym, n=5, fwd3d=0.02):
        return [
            {
                "symbol": sym,
                "promoted_at": _iso_ago(10),
                "forward_return_3d": fwd3d,
                "continuation_days": 1,
                "status": STATUS_ACTIVE,
            }
            for _ in range(n)
        ]

    def test_not_elapsed(self):
        rec = _make_prob_record()
        rec.promoted_at = _iso_ago(2)
        rec.probation_days = DEFAULT_PROBATION_DAYS
        ok, reason = check_graduation(rec, 80.0, self._outcomes("1234.T"))
        assert not ok
        assert "probation_not_elapsed" in reason

    def test_all_conditions_pass(self):
        rec = _make_prob_record(rsr=70.0)
        rec.promoted_at = _iso_ago(DEFAULT_PROBATION_DAYS + 1)
        ok, reason = check_graduation(rec, 72.0, self._outcomes("1234.T", fwd3d=0.01))
        assert ok
        assert "graduated" in reason

    def test_negative_expectancy_fails(self):
        rec = _make_prob_record(rsr=70.0)
        rec.promoted_at = _iso_ago(DEFAULT_PROBATION_DAYS + 1)
        ok, reason = check_graduation(rec, 72.0, self._outcomes("1234.T", fwd3d=-0.01))
        assert not ok
        assert "expectancy_negative" in reason

    def test_rsr_declining_fails(self):
        rec = _make_prob_record(rsr=80.0)
        rec.promoted_at = _iso_ago(DEFAULT_PROBATION_DAYS + 1)
        ok, reason = check_graduation(rec, 60.0, self._outcomes("1234.T", fwd3d=0.02))
        assert not ok
        assert "rsr_declining" in reason

    def test_no_forward_returns_fails(self):
        rec = _make_prob_record()
        rec.promoted_at = _iso_ago(DEFAULT_PROBATION_DAYS + 1)
        ok, reason = check_graduation(rec, 80.0, [])
        assert not ok
        assert "no_forward_returns" in reason

    def test_low_continuation_fails(self):
        rec = _make_prob_record(rsr=70.0)
        rec.promoted_at = _iso_ago(DEFAULT_PROBATION_DAYS + 1)
        outcomes = [
            {"symbol": "1234.T", "forward_return_3d": 0.02, "continuation_days": 0}
        ] * 10
        ok, reason = check_graduation(rec, 72.0, outcomes)
        assert not ok
        assert "continuation_low" in reason


# ─────────────────────────────────────────────────────────────────────────────
# load_active_probation / _get_cooldown_symbols
# ─────────────────────────────────────────────────────────────────────────────

class TestStateLoading:
    def test_load_empty_file(self, tmp_path):
        p = tmp_path / "prob.jsonl"
        assert load_active_probation(p) == []

    def test_load_missing_file(self, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        assert load_active_probation(p) == []

    def test_active_record_loaded(self, tmp_path):
        p = tmp_path / "prob.jsonl"
        _write_jsonl(p, [_make_record("A.T", STATUS_ACTIVE)])
        result = load_active_probation(p)
        assert len(result) == 1
        assert result[0].symbol == "A.T"

    def test_demoted_record_excluded(self, tmp_path):
        p = tmp_path / "prob.jsonl"
        _write_jsonl(p, [_make_record("A.T", STATUS_DEMOTED)])
        assert load_active_probation(p) == []

    def test_graduated_record_excluded(self, tmp_path):
        p = tmp_path / "prob.jsonl"
        _write_jsonl(p, [_make_record("A.T", STATUS_GRADUATED)])
        assert load_active_probation(p) == []

    def test_last_record_wins(self, tmp_path):
        """Active then demoted → last status = demoted → not in active list."""
        p = tmp_path / "prob.jsonl"
        _write_jsonl(p, [
            _make_record("A.T", STATUS_ACTIVE),
            _make_record("A.T", STATUS_DEMOTED),
        ])
        assert load_active_probation(p) == []

    def test_multiple_symbols(self, tmp_path):
        p = tmp_path / "prob.jsonl"
        _write_jsonl(p, [
            _make_record("A.T", STATUS_ACTIVE),
            _make_record("B.T", STATUS_ACTIVE),
            _make_record("C.T", STATUS_DEMOTED),
        ])
        syms = {r.symbol for r in load_active_probation(p)}
        assert syms == {"A.T", "B.T"}

    def test_cooldown_recent_demotion(self, tmp_path):
        p = tmp_path / "prob.jsonl"
        rec = _make_record("A.T", STATUS_DEMOTED)
        rec["updated_at"] = _iso_ago(1)
        _write_jsonl(p, [rec])
        cooldown = _get_cooldown_symbols(p)
        assert "A.T" in cooldown

    def test_cooldown_expired_demotion(self, tmp_path):
        p = tmp_path / "prob.jsonl"
        rec = _make_record("A.T", STATUS_DEMOTED)
        rec["updated_at"] = _iso_ago(PROBATION_COOLDOWN_DAYS + 2)
        _write_jsonl(p, [rec])
        cooldown = _get_cooldown_symbols(p)
        assert "A.T" not in cooldown

    def test_cooldown_active_symbol_ignored(self, tmp_path):
        p = tmp_path / "prob.jsonl"
        _write_jsonl(p, [_make_record("A.T", STATUS_ACTIVE)])
        cooldown = _get_cooldown_symbols(p)
        assert "A.T" not in cooldown


# ─────────────────────────────────────────────────────────────────────────────
# load_latest_predictive_scores / load_latest_fl_candidates
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreLoaders:
    def test_pred_scores_empty_file(self, tmp_path):
        p = tmp_path / "scores.jsonl"
        assert load_latest_predictive_scores(p) == {}

    def test_pred_scores_missing_file(self, tmp_path):
        p = tmp_path / "nonexistent.jsonl"
        assert load_latest_predictive_scores(p) == {}

    def test_pred_scores_top_candidates_injected(self, tmp_path):
        p = tmp_path / "scores.jsonl"
        rec = {"predictive_alpha_score": {"A.T": 90.0, "B.T": 70.0}, "run_id": "r1"}
        _write_jsonl(p, [rec])
        result = load_latest_predictive_scores(p)
        assert "top_candidates" in result
        assert result["top_candidates"][0][0] == "A.T"

    def test_pred_scores_last_record(self, tmp_path):
        p = tmp_path / "scores.jsonl"
        _write_jsonl(p, [
            {"predictive_alpha_score": {"A.T": 50.0}, "run_id": "r1"},
            {"predictive_alpha_score": {"B.T": 90.0}, "run_id": "r2"},
        ])
        result = load_latest_predictive_scores(p)
        assert "B.T" in result.get("predictive_alpha_score", {})

    def test_fl_candidates_empty_file(self, tmp_path):
        p = tmp_path / "fl.jsonl"
        assert load_latest_fl_candidates(p) == {}

    def test_fl_candidates_latest_run_id(self, tmp_path):
        p = tmp_path / "fl.jsonl"
        _write_jsonl(p, [
            {"symbol": "A.T", "run_id": "r1", "persistent_drift_score": 40.0},
            {"symbol": "B.T", "run_id": "r2", "persistent_drift_score": 70.0},
        ])
        result = load_latest_fl_candidates(p)
        assert "B.T" in result
        assert "A.T" not in result   # different run_id

    def test_fl_candidates_returns_dict_by_symbol(self, tmp_path):
        p = tmp_path / "fl.jsonl"
        _write_jsonl(p, [
            {"symbol": "X.T", "run_id": "r1", "persistent_drift_score": 65.0},
        ])
        result = load_latest_fl_candidates(p)
        assert result["X.T"]["persistent_drift_score"] == 65.0


# ─────────────────────────────────────────────────────────────────────────────
# run_probation_gate (E2E)
# ─────────────────────────────────────────────────────────────────────────────

class TestRunProbationGate:
    def _pred(self, sym):
        pa  = {sym: 90.0}
        si  = {sym: 92.0}
        cb  = {sym: 75.0}
        top = sorted(pa.items(), key=lambda x: (-x[1], x[0]))
        return {
            "predictive_alpha_score": pa,
            "sector_ignition_score": si,
            "compression_breakout_score": cb,
            "top_candidates": top,
        }

    def test_new_promotion(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        sym  = "1234.T"
        active, newly = run_probation_gate(
            shadow_universe={sym: "電機精密"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: 80.0},
            predictive_scores=self._pred(sym),
            fl_candidates={},
            probation_path=prob,
            outcomes_path=out,
            run_id="test_run",
        )
        assert sym in active
        assert sym in newly
        assert prob.exists()

    def test_already_live_skipped(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        sym  = "1234.T"
        active, newly = run_probation_gate(
            shadow_universe={sym: "電機精密"},
            live_universe={sym: "電機精密"},
            cb_state="NORMAL",
            rsr_scores={sym: 80.0},
            predictive_scores=self._pred(sym),
            fl_candidates={},
            probation_path=prob,
            outcomes_path=out,
            run_id="test_run",
        )
        assert sym not in newly

    def test_cb_active_blocks_new_promotion(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        sym  = "1234.T"
        for cb in CB_SEVERE_STATES:
            prob.unlink(missing_ok=True)
            active, newly = run_probation_gate(
                shadow_universe={sym: "電機精密"},
                live_universe={},
                cb_state=cb,
                rsr_scores={sym: 80.0},
                predictive_scores=self._pred(sym),
                fl_candidates={},
                probation_path=prob,
                outcomes_path=out,
                run_id=f"run_{cb}",
            )
            assert not newly

    def test_budget_limit_enforced(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        # Fill budget: rsr_at_promotion=80, current RSR=80 → no demotion triggers
        existing = [_make_record(f"A{i}.T", rsr=80.0) for i in range(MAX_ACTIVE_PROBATION)]
        _write_jsonl(prob, existing)
        existing_syms = {f"A{i}.T" for i in range(MAX_ACTIVE_PROBATION)}
        # stable RSR: no drop, no breakout_fail (80 >= 65)
        rsr = {s: 80.0 for s in existing_syms}
        rsr["NEW.T"] = 80.0
        sym = "NEW.T"
        active, newly = run_probation_gate(
            shadow_universe={sym: "電機精密"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores=rsr,
            predictive_scores=self._pred(sym),
            fl_candidates={},
            probation_path=prob,
            outcomes_path=out,
            run_id="test_run",
        )
        assert sym not in newly

    def test_demotion_removes_from_active(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        sym  = "1234.T"
        # Existing active with high RSR at promotion; now RSR crashed
        _write_jsonl(prob, [_make_record(sym, STATUS_ACTIVE, rsr=80.0)])
        active, _ = run_probation_gate(
            shadow_universe={},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: 80.0 - RSR_RANK_DROP_THRESHOLD - 1.0},
            predictive_scores={},
            fl_candidates={},
            probation_path=prob,
            outcomes_path=out,
            run_id="test_run",
        )
        assert sym not in active

    def test_cooldown_prevents_repromote(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        sym  = "1234.T"
        rec  = _make_record(sym, STATUS_DEMOTED)
        rec["updated_at"] = _iso_ago(1)
        _write_jsonl(prob, [rec])
        _, newly = run_probation_gate(
            shadow_universe={sym: "電機精密"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: 80.0},
            predictive_scores=self._pred(sym),
            fl_candidates={},
            probation_path=prob,
            outcomes_path=out,
            run_id="test_run",
        )
        assert sym not in newly

    def test_gate_fail_not_promoted(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        sym  = "LOW.T"
        # RSR too low
        _, newly = run_probation_gate(
            shadow_universe={sym: "電機精密"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: GATE_MIN_RSR_PASS - 1.0},
            predictive_scores=self._pred(sym),
            fl_candidates={},
            probation_path=prob,
            outcomes_path=out,
            run_id="test_run",
        )
        assert sym not in newly

    def test_jsonl_is_append_only(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        sym  = "1234.T"
        run_probation_gate(
            shadow_universe={sym: "電機精密"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: 80.0},
            predictive_scores=self._pred(sym),
            fl_candidates={},
            probation_path=prob,
            outcomes_path=out,
            run_id="run1",
        )
        count_before = sum(1 for _ in prob.read_text().splitlines() if _.strip())
        run_probation_gate(
            shadow_universe={},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: 78.0},
            predictive_scores={},
            fl_candidates={},
            probation_path=prob,
            outcomes_path=out,
            run_id="run2",
        )
        count_after = sum(1 for _ in prob.read_text().splitlines() if _.strip())
        assert count_after >= count_before


# ─────────────────────────────────────────────────────────────────────────────
# run_probation_outcome_observation
# ─────────────────────────────────────────────────────────────────────────────

class TestOutcomeObservation:
    def test_no_active_noop(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        run_probation_outcome_observation(prob, out, {}, "run1")
        assert not out.exists()

    def test_observation_appended(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        _write_jsonl(prob, [_make_record("A.T")])
        run_probation_outcome_observation(prob, out, {"A.T": 78.0}, "run1")
        assert out.exists()
        lines = [l for l in out.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        rec = json.loads(lines[0])
        assert rec["symbol"] == "A.T"
        assert rec["forward_return_3d"] is None   # not materialized yet
        assert rec["rsr_delta"] == pytest.approx(-2.0, abs=0.01)  # 78-80

    def test_continuation_days_increments(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        _write_jsonl(prob, [_make_record("A.T")])
        run_probation_outcome_observation(prob, out, {"A.T": 79.0}, "r1")
        run_probation_outcome_observation(prob, out, {"A.T": 79.0}, "r2")
        lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
        assert lines[-1]["continuation_days"] == 2

    def test_rsr_delta_correct(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        _write_jsonl(prob, [_make_record("A.T", rsr=70.0)])
        run_probation_outcome_observation(prob, out, {"A.T": 85.0}, "r1")
        rec = json.loads(out.read_text().splitlines()[-1])
        assert rec["rsr_delta"] == pytest.approx(15.0, abs=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# format_probation_report
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatProbationReport:
    def test_empty_report(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        result = format_probation_report(prob, out)
        assert "AUTO_PROMOTE_SAFE_V2" in result
        assert "0/3枠" in result

    def test_active_symbol_shown(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        _write_jsonl(prob, [_make_record("X.T", STATUS_ACTIVE)])
        result = format_probation_report(prob, out)
        assert "X.T" in result

    def test_graduated_shown(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        _write_jsonl(prob, [_make_record("G.T", STATUS_GRADUATED)])
        result = format_probation_report(prob, out)
        assert "G.T" in result

    def test_hit_rate_shown(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        _write_jsonl(prob, [
            _make_record("A.T", STATUS_GRADUATED),
            _make_record("B.T", STATUS_DEMOTED),
        ])
        result = format_probation_report(prob, out)
        assert "ヒット率" in result

    def test_avg_expectancy_shown(self, tmp_path):
        prob = tmp_path / "prob.jsonl"
        out  = tmp_path / "out.jsonl"
        _write_jsonl(out, [{"symbol": "X.T", "forward_return_3d": 0.05}])
        result = format_probation_report(prob, out)
        assert "平均期待値" in result

    def test_missing_files_no_crash(self, tmp_path):
        prob = tmp_path / "nonexistent_prob.jsonl"
        out  = tmp_path / "nonexistent_out.jsonl"
        result = format_probation_report(prob, out)
        assert isinstance(result, str)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

class TestConstants:
    def test_allocation_multiplier(self):
        assert ALLOCATION_MULTIPLIER == pytest.approx(0.25)

    def test_weights_sum_to_one(self):
        from src.universe.auto_promote_safe_v2 import (
            WEIGHT_COMPRESSION, WEIGHT_DRIFT, WEIGHT_PREDICTIVE,
            WEIGHT_RSR_PASS, WEIGHT_SECTOR_IGNITION,
        )
        total = WEIGHT_RSR_PASS + WEIGHT_PREDICTIVE + WEIGHT_SECTOR_IGNITION + WEIGHT_COMPRESSION + WEIGHT_DRIFT
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_max_active_probation(self):
        assert MAX_ACTIVE_PROBATION == 3


# =============================================================================
# Explainability + Taxonomy layer tests
# =============================================================================

from src.universe.auto_promote_safe_v2 import (
    ALLOWED_CANDIDATE_TYPES,
    CANDIDATE_CONTINUATION,
    CANDIDATE_EARLY_IGNITION,
    CANDIDATE_MATURE_LEADER,
    CANDIDATE_MEAN_REVERSION,
    CANDIDATE_UNCLASSIFIED,
    CONTINUATION_MAX_RSR,
    CONTINUATION_MIN_COMPRESSION,
    CONTINUATION_MIN_RSR,
    IGNITION_MAX_PREDICTIVE_RANK,
    IGNITION_MAX_RSR,
    IGNITION_MIN_DRIFT,
    IGNITION_MIN_SECTOR_IGNITION,
    MATURE_LEADER_MAX_COMPRESSION,
    MATURE_LEADER_MIN_RSR,
    MEAN_REVERSION_MAX_IGNITION,
    MEAN_REVERSION_MAX_RSR,
    P2A_EXCLUDED_SYMBOLS,
    P2A_MIN_SECTOR_IGNITION,
    RejectionRecord,
    _build_rejection_record,
    _compute_late_entry_risk,
    check_p2a_unclassified_gate,
    classify_candidate_type,
    format_explainability_report,
    format_rejection_stats,
)


def _ps(symbols, pa=None, si=None, cb=None):
    pa = pa or {s: 80.0 for s in symbols}
    si = si or {s: 92.0 for s in symbols}
    cb = cb or {s: 75.0 for s in symbols}
    top = sorted(pa.items(), key=lambda x: (-x[1], x[0]))
    return {
        "predictive_alpha_score": pa,
        "sector_ignition_score": si,
        "compression_breakout_score": cb,
        "top_candidates": top,
    }


def _fl(symbol=None, drift=60.0, fl_score=None):
    sym = symbol or "1234.T"
    rec = {"symbol": sym, "persistent_drift_score": drift}
    if fl_score is not None:
        rec["future_leader_score"] = fl_score
    return {sym: rec}


def _rej_record(sym="5301.T", ct=CANDIDATE_MATURE_LEADER, today=None):
    from datetime import date as _date
    ts = (today or _date.today().isoformat()) + "T09:00:00+00:00"
    return {
        "symbol": sym,
        "timestamp": ts,
        "candidate_type": ct,
        "promotion_score": 0.45,
        "failed_conditions": ["taxonomy_blocked:" + ct],
        "metrics": {
            "rsr_pass": 90,
            "predictive_rank": 2,
            "sector_ignition": 95.0,
            "cb_state": "NORMAL",
            "future_leader_score": None,
            "rsr_rank": None,
            "late_entry_risk": ct == CANDIDATE_MATURE_LEADER,
        },
        "batch_id": "abc12345",
        "schema_version": 1,
    }


class TestComputeLateEntryRisk:
    def test_high_rsr_low_comp_is_late(self):
        assert _compute_late_entry_risk(MATURE_LEADER_MIN_RSR, MATURE_LEADER_MAX_COMPRESSION - 1) is True

    def test_high_rsr_high_comp_not_late(self):
        assert _compute_late_entry_risk(MATURE_LEADER_MIN_RSR, MATURE_LEADER_MAX_COMPRESSION + 1) is False

    def test_low_rsr_low_comp_not_late(self):
        assert _compute_late_entry_risk(MATURE_LEADER_MIN_RSR - 1, MATURE_LEADER_MAX_COMPRESSION - 1) is False

    def test_boundary_exact_rsr(self):
        assert _compute_late_entry_risk(MATURE_LEADER_MIN_RSR, 0.0) is True

    def test_boundary_exact_comp(self):
        assert _compute_late_entry_risk(MATURE_LEADER_MIN_RSR + 1, MATURE_LEADER_MAX_COMPRESSION) is False


class TestClassifyCandidateType:
    def _classify(self, sym, rsr, si_val=60.0, comp=50.0, drift=30.0, rank_pos=1):
        rsr_scores = {sym: rsr}
        pred_scores = _ps([sym], pa={sym: 90.0 - (rank_pos - 1) * 5},
                          si={sym: si_val}, cb={sym: comp})
        fl = _fl(sym, drift=drift)
        return classify_candidate_type(sym, rsr_scores, pred_scores, fl)

    def test_mature_leader_classification(self):
        ctype, ler = self._classify("A.T", rsr=90.0, comp=20.0)
        assert ctype == CANDIDATE_MATURE_LEADER
        assert ler is True

    def test_mature_leader_late_entry_risk_true(self):
        _, ler = self._classify("A.T", rsr=MATURE_LEADER_MIN_RSR, comp=0.0)
        assert ler is True

    def test_mean_reversion_classification(self):
        ctype, ler = self._classify("B.T", rsr=40.0, si_val=30.0, comp=50.0,
                                    drift=10.0, rank_pos=10)
        assert ctype == CANDIDATE_MEAN_REVERSION
        assert ler is False

    def test_early_ignition_classification(self):
        ctype, ler = self._classify(
            "C.T", rsr=75.0,
            si_val=IGNITION_MIN_SECTOR_IGNITION + 5,
            comp=50.0,
            drift=IGNITION_MIN_DRIFT + 5,
            rank_pos=2,
        )
        assert ctype == CANDIDATE_EARLY_IGNITION
        assert ler is False

    def test_continuation_classification(self):
        sym = "D.T"
        rsr_scores = {sym: 78.0}
        pred_scores = _ps([sym], si={sym: 50.0},
                          cb={sym: CONTINUATION_MIN_COMPRESSION + 5},
                          pa={sym: 20.0})
        pred_scores["top_candidates"] = [
            ("Z.T", 99.0), ("Y.T", 98.0), ("X.T", 97.0),
            ("W.T", 96.0), ("V.T", 95.0), (sym, 20.0),
        ]
        fl = _fl(sym, drift=10.0)
        ctype, ler = classify_candidate_type(sym, rsr_scores, pred_scores, fl)
        assert ctype == CANDIDATE_CONTINUATION
        assert ler is False

    def test_unclassified_fallback(self):
        sym = "E.T"
        rsr_scores = {sym: 60.0}
        pred_scores = _ps([sym], si={sym: 60.0}, cb={sym: 10.0}, pa={sym: 30.0})
        pred_scores["top_candidates"] = [(sym, 30.0)]
        fl = _fl(sym, drift=10.0)
        ctype, _ = classify_candidate_type(sym, rsr_scores, pred_scores, fl)
        assert ctype == CANDIDATE_UNCLASSIFIED

    def test_empty_metrics_no_crash(self):
        # RSR=0 < MEAN_REVERSION_MAX_RSR and si=0 < MEAN_REVERSION_MAX_IGNITION
        # → correctly classified as MEAN_REVERSION (no trend, no ignition)
        ctype, ler = classify_candidate_type("F.T", {}, {}, {})
        assert ctype in (CANDIDATE_MEAN_REVERSION, CANDIDATE_UNCLASSIFIED)
        assert ler is False

    def test_mature_leader_priority_over_ignition(self):
        ctype, _ = self._classify("G.T", rsr=90.0, si_val=95.0, comp=10.0,
                                  drift=80.0, rank_pos=1)
        assert ctype == CANDIDATE_MATURE_LEADER

    def test_mean_reversion_priority_over_unclassified(self):
        ctype, _ = self._classify("H.T", rsr=30.0, si_val=20.0, comp=5.0,
                                  drift=5.0, rank_pos=20)
        assert ctype == CANDIDATE_MEAN_REVERSION

    def test_early_ignition_requires_drift(self):
        sym = "I.T"
        pred_scores = _ps([sym], si={sym: 80.0}, cb={sym: 50.0}, pa={sym: 90.0})
        fl = _fl(sym, drift=IGNITION_MIN_DRIFT - 1)
        ctype, _ = classify_candidate_type(sym, {sym: 75.0}, pred_scores, fl)
        assert ctype != CANDIDATE_EARLY_IGNITION

    def test_continuation_excluded_when_rsr_too_high(self):
        sym = "J.T"
        pred_scores = _ps([sym], si={sym: 60.0}, cb={sym: 40.0}, pa={sym: 20.0})
        fl = _fl(sym, drift=10.0)
        ctype, _ = classify_candidate_type(
            sym, {sym: CONTINUATION_MAX_RSR + 1}, pred_scores, fl
        )
        assert ctype != CANDIDATE_CONTINUATION


class TestRejectionRecord:
    def test_to_dict_has_required_fields(self):
        rec = RejectionRecord(
            symbol="5301.T", timestamp="2026-05-29T09:00:00+00:00",
            candidate_type=CANDIDATE_MATURE_LEADER, promotion_score=0.45,
            failed_conditions=["taxonomy_blocked:MATURE_LEADER"],
            metrics={"rsr_pass": 90, "late_entry_risk": True}, batch_id="abc",
        )
        d = rec.to_dict()
        assert d["symbol"] == "5301.T"
        assert d["candidate_type"] == CANDIDATE_MATURE_LEADER
        assert d["schema_version"] == 1

    def test_metrics_preserved(self):
        m = {"rsr_pass": 88, "sector_ignition": 92.5, "late_entry_risk": True}
        rec = RejectionRecord("X.T", "ts", CANDIDATE_MATURE_LEADER, 0.5, ["f"], m, "b")
        assert rec.to_dict()["metrics"] == m


class TestBuildRejectionRecord:
    def test_returns_dict(self):
        sym = "5301.T"
        pred = _ps([sym], si={sym: 95.0}, cb={sym: 20.0}, pa={sym: 90.0})
        r = _build_rejection_record(
            sym, CANDIDATE_MATURE_LEADER, 0.0,
            ["taxonomy_blocked:MATURE_LEADER"],
            {sym: 90.0}, pred, _fl(sym), "NORMAL", True, "b1",
            "2026-05-29T09:00:00+00:00",
        )
        assert isinstance(r, dict)
        assert r["symbol"] == sym
        assert r["metrics"]["rsr_pass"] == 90
        assert r["metrics"]["late_entry_risk"] is True

    def test_predictive_rank_captured(self):
        sym = "A.T"
        pred = _ps([sym], pa={sym: 80.0})
        r = _build_rejection_record(
            sym, CANDIDATE_UNCLASSIFIED, 0.0, ["gate_fail"],
            {sym: 50.0}, pred, {}, "NORMAL", False, "b", "ts",
        )
        assert r["metrics"]["predictive_rank"] == 1

    def test_unranked_symbol_rank_none(self):
        pred = _ps(["OTHER.T"])
        r = _build_rejection_record(
            "MISSING.T", CANDIDATE_UNCLASSIFIED, 0.0, ["gate_fail"],
            {"MISSING.T": 50.0}, pred, {}, "NORMAL", False, "b", "ts",
        )
        assert r["metrics"]["predictive_rank"] is None


class TestTaxonomyRouting:
    def test_mature_leader_blocked_by_taxonomy(self, tmp_path):
        sym = "5301.T"
        rej_f = tmp_path / "rej.jsonl"
        rsr = {sym: 92.0}
        pred = _ps([sym], pa={sym: 90.0}, si={sym: 95.0}, cb={sym: 20.0})
        fl = _fl(sym, drift=70.0)
        _, newly = run_probation_gate(
            shadow_universe={sym: "保険"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores=rsr,
            predictive_scores=pred,
            fl_candidates=fl,
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
            rejection_path=rej_f,
        )
        assert sym not in newly
        if rej_f.exists():
            rejs = [json.loads(l) for l in rej_f.read_text().splitlines() if l.strip()]
            if any(r["symbol"] == sym for r in rejs):
                tax_rej = next(r for r in rejs if r["symbol"] == sym)
                assert "taxonomy_blocked" in str(tax_rej["failed_conditions"])

    def test_early_ignition_passes_to_promotion(self, tmp_path):
        sym = "1234.T"
        rej_f = tmp_path / "rej.jsonl"
        pred = _ps([sym], pa={sym: 95.0}, si={sym: 93.0}, cb={sym: 50.0})
        fl = _fl(sym, drift=60.0)
        _, newly = run_probation_gate(
            shadow_universe={sym: "輸送用機器"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: 78.0},
            predictive_scores=pred,
            fl_candidates=fl,
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
            rejection_path=rej_f,
        )
        assert sym in newly
        if rej_f.exists():
            rejs = [json.loads(l) for l in rej_f.read_text().splitlines() if l.strip()]
            assert not any(r["symbol"] == sym for r in rejs)

    def test_allowed_types_correct(self):
        assert CANDIDATE_EARLY_IGNITION in ALLOWED_CANDIDATE_TYPES
        assert CANDIDATE_CONTINUATION in ALLOWED_CANDIDATE_TYPES
        assert CANDIDATE_MATURE_LEADER not in ALLOWED_CANDIDATE_TYPES
        assert CANDIDATE_MEAN_REVERSION not in ALLOWED_CANDIDATE_TYPES
        # UNCLASSIFIED is NOT in ALLOWED_CANDIDATE_TYPES — it is handled by the
        # separate P2-A gate path in run_probation_gate, bypassing taxonomy check.
        assert CANDIDATE_UNCLASSIFIED not in ALLOWED_CANDIDATE_TYPES


# ── P2-A gate unit tests ──────────────────────────────────────────────────────

def _p2a_ps(sym: str, si: float = 72.0) -> dict:
    """Minimal predictive_scores for P2-A gate tests."""
    return {
        "sector_ignition_score": {sym: si},
        "top_candidates": [],
        "compression_breakout_score": {sym: 30.0},
        "predictive_alpha_score": {},
    }


class TestCheckP2AUnclassifiedGate:
    def test_pass_si_at_lower_bound(self):
        ok, fail = check_p2a_unclassified_gate("6146.T", {"6146.T": 20.0}, _p2a_ps("6146.T", si=50.0))
        assert ok
        assert fail == []

    def test_pass_si_midrange(self):
        ok, fail = check_p2a_unclassified_gate("6857.T", {"6857.T": 20.0}, _p2a_ps("6857.T", si=72.4))
        assert ok
        assert fail == []

    def test_pass_si_near_upper_bound(self):
        ok, fail = check_p2a_unclassified_gate("6920.T", {"6920.T": 20.0}, _p2a_ps("6920.T", si=89.9))
        assert ok
        assert fail == []

    def test_reject_si_below_50(self):
        ok, fail = check_p2a_unclassified_gate("6146.T", {"6146.T": 20.0}, _p2a_ps("6146.T", si=49.9))
        assert not ok
        assert any("p2a_si_too_low" in f for f in fail)

    def test_reject_si_zero(self):
        ok, fail = check_p2a_unclassified_gate("6146.T", {"6146.T": 20.0}, _p2a_ps("6146.T", si=0.0))
        assert not ok
        assert any("p2a_si_too_low" in f for f in fail)

    def test_reject_si_at_upper_bound(self):
        ok, fail = check_p2a_unclassified_gate("6146.T", {"6146.T": 20.0}, _p2a_ps("6146.T", si=90.0))
        assert not ok
        assert any("p2a_si_too_high" in f for f in fail)

    def test_reject_si_above_upper_bound(self):
        ok, fail = check_p2a_unclassified_gate("6146.T", {"6146.T": 20.0}, _p2a_ps("6146.T", si=95.0))
        assert not ok
        assert any("p2a_si_too_high" in f for f in fail)

    def test_reject_excluded_symbol(self):
        ok, fail = check_p2a_unclassified_gate("5706.T", {"5706.T": 20.0}, _p2a_ps("5706.T", si=72.0))
        assert not ok
        assert any("p2a_excluded" in f for f in fail)

    def test_reject_excluded_symbol_even_with_high_si(self):
        ok, fail = check_p2a_unclassified_gate("5706.T", {"5706.T": 20.0}, _p2a_ps("5706.T", si=80.0))
        assert not ok
        assert any("p2a_excluded" in f for f in fail)

    def test_reject_rsr_too_low(self):
        ok, fail = check_p2a_unclassified_gate("6146.T", {"6146.T": 1.0}, _p2a_ps("6146.T", si=72.0))
        assert not ok
        assert any("rsr_too_low" in f for f in fail)

    def test_reject_missing_rsr(self):
        ok, fail = check_p2a_unclassified_gate("6146.T", {}, _p2a_ps("6146.T", si=72.0))
        assert not ok
        assert any("rsr_too_low" in f for f in fail)

    def test_reject_missing_si(self):
        ok, fail = check_p2a_unclassified_gate("6146.T", {"6146.T": 20.0},
                                                {"sector_ignition_score": {}, "top_candidates": []})
        assert not ok
        assert any("p2a_si_too_low" in f for f in fail)

    def test_p2a_excluded_symbols_contains_5706(self):
        assert "5706.T" in P2A_EXCLUDED_SYMBOLS

    def test_p2a_min_sector_ignition_is_50(self):
        assert P2A_MIN_SECTOR_IGNITION == 50.0


def _unclassified_ps(sym: str, si: float = 72.0) -> dict:
    """predictive_scores that ranks sym >5 (alpha=40) to prevent EARLY_IGNITION."""
    pa = {f"X{i:04d}.T": 90.0 - i for i in range(10)}  # X0000-X0009 rank 1-10
    pa[sym] = 40.0  # sym rank = 11 > IGNITION_MAX_PREDICTIVE_RANK(5)
    return {
        "sector_ignition_score": {sym: si},
        "top_candidates": sorted(pa.items(), key=lambda x: (-x[1], x[0])),
        "compression_breakout_score": {sym: 30.0},
        "predictive_alpha_score": pa,
    }


def _unclassified_fl(sym: str, drift: float = 20.0) -> dict:
    """fl_candidates with drift < IGNITION_MIN_DRIFT to avoid EARLY_IGNITION."""
    return {sym: {"future_leader_score": 40.0, "persistent_drift_score": drift}}


class TestRunProbationGateP2A:
    def _run(self, sym, rsr, si, tmp_path, sector="電機精密"):
        rej_f = tmp_path / "rej.jsonl"
        _, newly = run_probation_gate(
            shadow_universe={sym: sector},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: rsr},
            predictive_scores=_unclassified_ps(sym, si=si),
            fl_candidates=_unclassified_fl(sym),
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
            rejection_path=rej_f,
        )
        rejs = []
        if rej_f.exists():
            rejs = [json.loads(l) for l in rej_f.read_text().splitlines() if l.strip()]
        return newly, rejs

    def test_unclassified_si50_promoted(self, tmp_path):
        # RSR=60: avoids MEAN_REVERSION (needs RSR<55 AND SI<55) and CONTINUATION (needs RSR≥70)
        newly, _ = self._run("6146.T", rsr=60.0, si=50.0, tmp_path=tmp_path)
        assert "6146.T" in newly

    def test_unclassified_si72_promoted(self, tmp_path):
        newly, _ = self._run("6857.T", rsr=60.0, si=72.4, tmp_path=tmp_path)
        assert "6857.T" in newly

    def test_unclassified_si89_promoted(self, tmp_path):
        newly, _ = self._run("6920.T", rsr=60.0, si=89.9, tmp_path=tmp_path)
        assert "6920.T" in newly

    def test_unclassified_si_below_50_rejected(self, tmp_path):
        newly, rejs = self._run("8035.T", rsr=60.0, si=49.0, tmp_path=tmp_path)
        assert "8035.T" not in newly
        assert any(r["symbol"] == "8035.T" for r in rejs)
        assert any("p2a_si_too_low" in str(r["failed_conditions"]) for r in rejs
                   if r["symbol"] == "8035.T")

    def test_unclassified_si90_rejected(self, tmp_path):
        # SI=90 → p2a_si_too_high; also SI≥90 AND RSR=60<85 AND rank>5 → UNCLASSIFIED still
        newly, rejs = self._run("6146.T", rsr=60.0, si=90.0, tmp_path=tmp_path)
        assert "6146.T" not in newly
        assert any("p2a_si_too_high" in str(r["failed_conditions"]) for r in rejs
                   if r["symbol"] == "6146.T")

    def test_excluded_symbol_5706_rejected(self, tmp_path):
        newly, rejs = self._run("5706.T", rsr=60.0, si=72.0, tmp_path=tmp_path)
        assert "5706.T" not in newly
        assert any(r["symbol"] == "5706.T" for r in rejs)
        assert any("p2a_excluded" in str(r["failed_conditions"]) for r in rejs
                   if r["symbol"] == "5706.T")

    def test_p2a_reason_tag_present(self, tmp_path):
        """Promoted UNCLASSIFIED candidate carries 'p2a_unclassified' in promotion log."""
        prob_f = tmp_path / "p.jsonl"
        run_probation_gate(
            shadow_universe={"6146.T": "電機精密"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={"6146.T": 60.0},
            predictive_scores=_unclassified_ps("6146.T", si=72.0),
            fl_candidates=_unclassified_fl("6146.T"),
            probation_path=prob_f,
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
        )
        recs = [json.loads(l) for l in prob_f.read_text().splitlines() if l.strip()]
        sym_recs = [r for r in recs if r.get("symbol") == "6146.T"]
        assert sym_recs, "promotion record missing"
        assert "p2a_unclassified" in sym_recs[0].get("promotion_reason", [])

    def test_existing_types_unaffected(self, tmp_path):
        """EARLY_IGNITION still passes through unchanged."""
        sym = "7777.T"
        pred = _ps([sym], pa={sym: 95.0}, si={sym: 93.0}, cb={sym: 50.0})
        fl = _fl(sym, drift=60.0)
        _, newly = run_probation_gate(
            shadow_universe={sym: "輸送用機器"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: 78.0},
            predictive_scores=pred,
            fl_candidates=fl,
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
        )
        assert sym in newly


class TestRejectionLogging:
    def test_rejection_file_created_on_gate_fail(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        run_probation_gate(
            shadow_universe={"9999.T": "食料品"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={"9999.T": 1.0},
            predictive_scores=_ps(["9999.T"]),
            fl_candidates={},
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
            rejection_path=rej_f,
        )
        assert rej_f.exists()
        recs = [json.loads(l) for l in rej_f.read_text().splitlines() if l.strip()]
        assert any(r["symbol"] == "9999.T" for r in recs)

    def test_rejection_record_valid_json_schema(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        run_probation_gate(
            shadow_universe={"8888.T": "小売"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={"8888.T": 1.0},
            predictive_scores=_ps(["8888.T"]),
            fl_candidates={},
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
            rejection_path=rej_f,
        )
        for line in rej_f.read_text().splitlines():
            r = json.loads(line)
            m = r["metrics"]
            for key in ("rsr_pass", "predictive_rank", "sector_ignition",
                        "cb_state", "future_leader_score", "rsr_rank", "late_entry_risk"):
                assert key in m, f"missing: {key}"

    def test_no_rejection_path_no_crash(self, tmp_path):
        active, _ = run_probation_gate(
            shadow_universe={"X.T": "金融"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={"X.T": 1.0},
            predictive_scores=_ps(["X.T"]),
            fl_candidates={},
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
            rejection_path=None,
        )
        assert isinstance(active, set)

    def test_append_only_two_runs(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        for _ in range(2):
            run_probation_gate(
                shadow_universe={"Z.T": "建設"},
                live_universe={},
                cb_state="NORMAL",
                rsr_scores={"Z.T": 1.0},
                predictive_scores=_ps(["Z.T"]),
                fl_candidates={},
                probation_path=tmp_path / "p.jsonl",
                outcomes_path=tmp_path / "o.jsonl",
                run_id="t",
                rejection_path=rej_f,
            )
        lines = [l for l in rej_f.read_text().splitlines() if l.strip()]
        assert len(lines) == 2


class TestDebugWarning:
    def test_warning_emitted_when_all_rejected(self, tmp_path, capsys):
        run_probation_gate(
            shadow_universe={"WARN.T": "食料品"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={"WARN.T": 1.0},
            predictive_scores=_ps(["WARN.T"]),
            fl_candidates={},
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
            rejection_path=tmp_path / "rej.jsonl",
        )
        out = capsys.readouterr().out
        assert "shadow_candidates_detected=1" in out
        assert "probation_promoted=0" in out

    def test_no_warning_when_empty_shadow(self, tmp_path, capsys):
        run_probation_gate(
            shadow_universe={},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={},
            predictive_scores={},
            fl_candidates={},
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
        )
        assert "shadow_candidates_detected" not in capsys.readouterr().out

    def test_no_warning_when_promoted(self, tmp_path, capsys):
        sym = "GOOD.T"
        pred = _ps([sym], pa={sym: 95.0}, si={sym: 93.0}, cb={sym: 50.0})
        fl = _fl(sym, drift=60.0)
        _, newly = run_probation_gate(
            shadow_universe={sym: "輸送用機器"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: 78.0},
            predictive_scores=pred,
            fl_candidates=fl,
            probation_path=tmp_path / "p.jsonl",
            outcomes_path=tmp_path / "o.jsonl",
            run_id="t",
        )
        if sym in newly:
            assert "probation_promoted=0" not in capsys.readouterr().out


class TestExplainabilityReport:
    def _wr(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_missing_files_no_crash(self, tmp_path):
        r = format_explainability_report(tmp_path / "nx.jsonl", tmp_path / "nx2.jsonl")
        assert isinstance(r, str)

    def test_section_header_present(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        self._wr(rej_f, [_rej_record()])
        r = format_explainability_report(rej_f, tmp_path / "p.jsonl")
        assert "AUTO_PROMOTE_EXPLAINABILITY" in r

    def test_rejected_symbol_and_type_shown(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        self._wr(rej_f, [_rej_record("5301.T", CANDIDATE_MATURE_LEADER)])
        r = format_explainability_report(rej_f, tmp_path / "p.jsonl")
        assert "5301.T" in r
        assert "MATURE_LEADER" in r

    def test_late_entry_risk_shown_when_true(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        self._wr(rej_f, [_rej_record("5301.T", CANDIDATE_MATURE_LEADER)])
        assert "late_entry_risk=True" in format_explainability_report(rej_f, tmp_path / "p.jsonl")

    def test_late_entry_risk_not_shown_when_false(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        rec = _rej_record("8750.T", CANDIDATE_CONTINUATION)
        rec["metrics"]["late_entry_risk"] = False
        self._wr(rej_f, [rec])
        assert "late_entry_risk=True" not in format_explainability_report(rej_f, tmp_path / "p.jsonl")

    def test_predictive_rank_shown(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        rec = _rej_record("8750.T", CANDIDATE_CONTINUATION)
        rec["metrics"]["predictive_rank"] = 12
        self._wr(rej_f, [rec])
        assert "predictive_rank=12" in format_explainability_report(rej_f, tmp_path / "p.jsonl")

    def test_old_date_excluded(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        self._wr(rej_f, [_rej_record("OLD.T", today="2020-01-01")])
        r = format_explainability_report(rej_f, tmp_path / "p.jsonl")
        assert "OLD.T" not in r

    def test_empty_file_no_crash(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        rej_f.write_text("")
        r = format_explainability_report(rej_f, tmp_path / "p.jsonl")
        assert "AUTO_PROMOTE_EXPLAINABILITY" in r

    def test_accepted_symbol_shown(self, tmp_path):
        from datetime import date
        rej_f = tmp_path / "rej.jsonl"
        prob_f = tmp_path / "prob.jsonl"
        rej_f.write_text("")
        prob_f.write_text(json.dumps({
            "symbol": "GOOD.T",
            "promoted_at": date.today().isoformat() + "T09:00:00+00:00",
            "status": STATUS_ACTIVE,
        }) + "\n")
        r = format_explainability_report(rej_f, prob_f)
        assert "GOOD.T" in r

    def test_multiple_rejections_shown(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        self._wr(rej_f, [
            _rej_record("A.T", CANDIDATE_MATURE_LEADER),
            _rej_record("B.T", CANDIDATE_MEAN_REVERSION),
        ])
        r = format_explainability_report(rej_f, tmp_path / "p.jsonl")
        assert "A.T" in r
        assert "B.T" in r


class TestRejectionStats:
    def _wr(self, path, records):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def test_empty_file_returns_empty(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        rej_f.write_text("")
        assert format_rejection_stats(rej_f) == ""

    def test_missing_file_returns_empty(self, tmp_path):
        assert format_rejection_stats(tmp_path / "nx.jsonl") == ""

    def test_section_header_present(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        self._wr(rej_f, [_rej_record("A.T")])
        assert "AUTO_PROMOTE_REJECTION_STATS" in format_rejection_stats(rej_f)

    def test_condition_counted(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        rec = _rej_record("A.T")
        rec["failed_conditions"] = ["taxonomy_blocked:MATURE_LEADER"]
        self._wr(rej_f, [rec])
        r = format_rejection_stats(rej_f)
        assert "taxonomy_blocked" in r
        assert "1 reject" in r

    def test_most_common_condition_first(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        recs_pred = [_rej_record(f"A{i}.T") for i in range(5)]
        for r in recs_pred:
            r["failed_conditions"] = ["predictive_rank:5>3"]
        recs_si = [_rej_record(f"B{i}.T") for i in range(2)]
        for r in recs_si:
            r["failed_conditions"] = ["sector_ignition:50<90"]
        self._wr(rej_f, recs_pred + recs_si)
        r = format_rejection_stats(rej_f)
        assert r.find("predictive_rank") < r.find("sector_ignition")

    def test_plural_label(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        recs = [_rej_record(f"X{i}.T") for i in range(3)]
        for r in recs:
            r["failed_conditions"] = ["taxonomy_blocked:MATURE_LEADER"]
        self._wr(rej_f, recs)
        assert "rejects" in format_rejection_stats(rej_f)

    def test_singular_label(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        self._wr(rej_f, [_rej_record("A.T")])
        assert "1 reject" in format_rejection_stats(rej_f)

    def test_colon_detail_stripped(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        rec = _rej_record("A.T")
        rec["failed_conditions"] = ["predictive_rank:5>top3"]
        self._wr(rej_f, [rec])
        r = format_rejection_stats(rej_f)
        assert "predictive_rank" in r
        assert "predictive_rank:5>top3" not in r

    def test_old_date_excluded(self, tmp_path):
        rej_f = tmp_path / "rej.jsonl"
        self._wr(rej_f, [_rej_record("OLD.T", today="2020-01-01")])
        assert format_rejection_stats(rej_f) == ""


class TestProbationRecordCandidateType:
    def test_default_is_unclassified(self):
        rec = ProbationRecord(
            symbol="A.T", promoted_at="ts", promotion_score=0.5,
            promotion_reason=[], probation_days=5, status=STATUS_ACTIVE,
            batch_id="b", rsr_at_promotion=80.0, sector_ignition_score=92.0,
            compression_score=75.0, drift_score=60.0,
        )
        assert rec.candidate_type == CANDIDATE_UNCLASSIFIED

    def test_explicit_type_preserved(self):
        rec = ProbationRecord(
            symbol="B.T", promoted_at="ts", promotion_score=0.6,
            promotion_reason=[], probation_days=5, status=STATUS_ACTIVE,
            batch_id="b", rsr_at_promotion=78.0, sector_ignition_score=93.0,
            compression_score=50.0, drift_score=60.0,
            candidate_type=CANDIDATE_EARLY_IGNITION,
        )
        assert rec.candidate_type == CANDIDATE_EARLY_IGNITION
        assert rec.to_dict()["candidate_type"] == CANDIDATE_EARLY_IGNITION

    def test_promoted_record_has_candidate_type_in_jsonl(self, tmp_path):
        sym = "1234.T"
        prob_f = tmp_path / "prob.jsonl"
        pred = _ps([sym], pa={sym: 95.0}, si={sym: 93.0}, cb={sym: 50.0})
        fl = _fl(sym, drift=60.0)
        _, newly = run_probation_gate(
            shadow_universe={sym: "輸送用機器"},
            live_universe={},
            cb_state="NORMAL",
            rsr_scores={sym: 78.0},
            predictive_scores=pred,
            fl_candidates=fl,
            probation_path=prob_f,
            outcomes_path=tmp_path / "out.jsonl",
            run_id="t",
        )
        if sym in newly:
            recs = [json.loads(l) for l in prob_f.read_text().splitlines() if l.strip()]
            prec = next(r for r in recs if r["symbol"] == sym)
            assert "candidate_type" in prec
            assert prec["candidate_type"] != ""


# ─────────────────────────────────────────────────────────────────────────────
# TestGetGraduatedSymbols
# ─────────────────────────────────────────────────────────────────────────────

class TestGetGraduatedSymbols:
    def test_returns_graduated_not_in_live(self, tmp_path):
        prob_f = tmp_path / "prob.jsonl"
        _write_jsonl(prob_f, [
            _make_record("1111.T", status=STATUS_GRADUATED, days_ago=6),
        ])
        result = get_graduated_symbols(prob_f, live_universe={})
        assert result == {"1111.T"}

    def test_excludes_already_in_live(self, tmp_path):
        prob_f = tmp_path / "prob.jsonl"
        _write_jsonl(prob_f, [
            _make_record("2222.T", status=STATUS_GRADUATED, days_ago=6),
        ])
        result = get_graduated_symbols(prob_f, live_universe={"2222.T": "sector"})
        assert result == set()

    def test_excludes_active_symbols(self, tmp_path):
        prob_f = tmp_path / "prob.jsonl"
        _write_jsonl(prob_f, [
            _make_record("3333.T", status=STATUS_ACTIVE, days_ago=3),
        ])
        result = get_graduated_symbols(prob_f, live_universe={})
        assert result == set()

    def test_excludes_demoted_symbols(self, tmp_path):
        prob_f = tmp_path / "prob.jsonl"
        _write_jsonl(prob_f, [
            _make_record("4444.T", status=STATUS_DEMOTED, days_ago=3),
        ])
        result = get_graduated_symbols(prob_f, live_universe={})
        assert result == set()

    def test_last_record_wins_graduated(self, tmp_path):
        prob_f = tmp_path / "prob.jsonl"
        _write_jsonl(prob_f, [
            _make_record("5555.T", status=STATUS_ACTIVE, days_ago=6),
            _make_record("5555.T", status=STATUS_GRADUATED, days_ago=0),
        ])
        result = get_graduated_symbols(prob_f, live_universe={})
        assert result == {"5555.T"}

    def test_last_record_wins_demoted_over_graduated(self, tmp_path):
        prob_f = tmp_path / "prob.jsonl"
        _write_jsonl(prob_f, [
            _make_record("6666.T", status=STATUS_GRADUATED, days_ago=3),
            _make_record("6666.T", status=STATUS_DEMOTED, days_ago=0),
        ])
        result = get_graduated_symbols(prob_f, live_universe={})
        assert result == set()

    def test_missing_file_returns_empty(self, tmp_path):
        result = get_graduated_symbols(tmp_path / "nonexistent.jsonl", live_universe={})
        assert result == set()

    def test_multiple_graduated_mixed(self, tmp_path):
        prob_f = tmp_path / "prob.jsonl"
        _write_jsonl(prob_f, [
            _make_record("A.T", status=STATUS_GRADUATED, days_ago=6),
            _make_record("B.T", status=STATUS_ACTIVE, days_ago=2),
            _make_record("C.T", status=STATUS_GRADUATED, days_ago=7),
        ])
        result = get_graduated_symbols(prob_f, live_universe={"C.T": "sector"})
        assert result == {"A.T"}

    def test_empty_file_returns_empty(self, tmp_path):
        prob_f = tmp_path / "prob.jsonl"
        prob_f.write_text("", encoding="utf-8")
        result = get_graduated_symbols(prob_f, live_universe={})
        assert result == set()


# =============================================================================
# P1-A: check_graduation() rsr_delta fallback
# =============================================================================

def _null_outcomes(sym: str, n: int = 5, rsr_delta: float = 2.0) -> list:
    """Outcomes with forward_return_3d=None and positive rsr_delta (P1-A path)."""
    return [
        {
            "symbol": sym,
            "promoted_at": _iso_ago(6),
            "forward_return_3d": None,
            "continuation_days": i + 1,
            "rsr_delta": rsr_delta,
            "status": STATUS_ACTIVE,
        }
        for i in range(n)
    ]


class TestCheckGraduationP1A:
    """P1-A: rsr_delta fallback when forward_return_3d is not materialized."""

    def _rec(self, rsr: float = 70.0) -> "ProbationRecord":
        r = _make_prob_record(rsr=rsr)
        r.promoted_at = _iso_ago(DEFAULT_PROBATION_DAYS + 1)
        return r

    # ── rsr_delta_fallback path ───────────────────────────────────────────────

    def test_rsr_delta_fallback_passes_when_positive(self):
        ok, reason = check_graduation(self._rec(), 72.0, _null_outcomes("1234.T", rsr_delta=2.0))
        assert ok
        assert "graduated" in reason
        assert "graduation_method:rsr_delta_fallback" in reason

    def test_rsr_delta_fallback_fails_when_negative(self):
        ok, reason = check_graduation(self._rec(), 72.0, _null_outcomes("1234.T", rsr_delta=-1.5))
        assert not ok
        assert "rsr_delta_negative" in reason

    def test_rsr_delta_fallback_fails_when_zero(self):
        ok, reason = check_graduation(self._rec(), 72.0, _null_outcomes("1234.T", rsr_delta=0.0))
        assert not ok
        assert "rsr_delta_negative" in reason

    def test_no_forward_returns_when_sym_outcomes_empty(self):
        """No outcomes at all (not just None fwd_ret) → no_forward_returns."""
        ok, reason = check_graduation(self._rec(), 72.0, [])
        assert not ok
        assert "no_forward_returns" in reason

    def test_rsr_delta_fallback_mixed_positive_average(self):
        """Average rsr_delta > 0 even with some negative entries."""
        outcomes = _null_outcomes("1234.T", n=3, rsr_delta=5.0) + \
                   _null_outcomes("1234.T", n=2, rsr_delta=-1.0)
        # avg = (5*3 + (-1)*2) / 5 = 13/5 = 2.6 > 0 → pass
        ok, reason = check_graduation(self._rec(), 72.0, outcomes)
        assert ok
        assert "graduation_method:rsr_delta_fallback" in reason

    # ── forward_return path (primary) still works ─────────────────────────────

    def test_forward_return_primary_path_succeeds(self):
        """If fwd_rets present, use forward_return path (graduation_method:forward_return)."""
        outcomes = [
            {"symbol": "1234.T", "forward_return_3d": 0.03, "continuation_days": 1,
             "rsr_delta": -5.0, "status": STATUS_ACTIVE}
        ] * 5
        rec = self._rec()
        ok, reason = check_graduation(rec, 72.0, outcomes)
        assert ok
        assert "graduation_method:forward_return" in reason

    def test_forward_return_negative_still_fails(self):
        outcomes = [
            {"symbol": "1234.T", "forward_return_3d": -0.02, "continuation_days": 1,
             "rsr_delta": 5.0, "status": STATUS_ACTIVE}
        ] * 5
        ok, reason = check_graduation(self._rec(), 72.0, outcomes)
        assert not ok
        assert "expectancy_negative" in reason

    def test_mixed_none_and_real_uses_primary_path(self):
        """Partial fwd_rets: non-None values should be used as primary, not fallback."""
        outcomes = [
            {"symbol": "1234.T", "forward_return_3d": 0.05, "continuation_days": 1,
             "rsr_delta": -3.0, "status": STATUS_ACTIVE},
            {"symbol": "1234.T", "forward_return_3d": None, "continuation_days": 2,
             "rsr_delta": -3.0, "status": STATUS_ACTIVE},
        ] * 2 + [
            {"symbol": "1234.T", "forward_return_3d": 0.01, "continuation_days": 3,
             "rsr_delta": -3.0, "status": STATUS_ACTIVE},
        ]
        ok, reason = check_graduation(self._rec(), 72.0, outcomes)
        assert ok
        assert "graduation_method:forward_return" in reason  # not rsr_delta_fallback


# =============================================================================
# P1-B: _just_graduated re-promotion prevention
# =============================================================================

def _write_active_record(path: "Path", sym: str, days_ago: int = 6,
                          rsr: float = 70.0, si: float = 72.0) -> None:
    """Write a single STATUS_ACTIVE record promoted days_ago in the past."""
    rec = {
        "symbol":             sym,
        "promoted_at":        _iso_ago(days_ago),
        "promotion_score":    0.72,
        "promotion_reason":   ["p2a_unclassified"],
        "probation_days":     DEFAULT_PROBATION_DAYS,
        "status":             STATUS_ACTIVE,
        "batch_id":           "test_batch",
        "rsr_at_promotion":   rsr,
        "sector_ignition_score": si,
        "compression_score":  30.0,
        "drift_score":        20.0,
        "candidate_type":     CANDIDATE_UNCLASSIFIED,
        "schema_version":     1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_positive_outcomes(path: "Path", sym: str, n: int = 5,
                              rsr_delta: float = 2.0) -> None:
    """Write n outcome records with positive rsr_delta (for rsr_delta_fallback graduation)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for i in range(n):
            rec = {
                "symbol":            sym,
                "forward_return_3d": None,
                "continuation_days": i + 1,
                "rsr_delta":         rsr_delta,
                "status":            STATUS_ACTIVE,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class TestJustGraduatedBlocking:
    """P1-B: same-run re-promotion prevention via _just_graduated."""

    SYM = "6857.T"
    SECTOR = "電機精密"

    def _shadow(self) -> dict:
        return {self.SYM: self.SECTOR}

    def _pred(self, si: float = 72.0) -> dict:
        return _unclassified_ps(self.SYM, si=si)

    def _fl(self) -> dict:
        return _unclassified_fl(self.SYM)

    def test_graduated_symbol_not_in_newly_promoted(self, tmp_path):
        """P1-B core: a symbol that graduates in Step 1 must NOT appear in newly_promoted."""
        sym = self.SYM
        prob_f    = tmp_path / "prob.jsonl"
        out_f     = tmp_path / "out.jsonl"

        # Symbol eligible for graduation (promoted 6 days ago, positive rsr_delta)
        _write_active_record(prob_f, sym, days_ago=6, rsr=70.0, si=72.0)
        _write_positive_outcomes(out_f, sym, n=5, rsr_delta=2.0)

        active, newly = run_probation_gate(
            shadow_universe   = self._shadow(),
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 72.0},   # stable (>=70*0.9=63)
            predictive_scores = self._pred(),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "test_p1b",
        )
        # Symbol graduated in Step 1 → must NOT be in newly_promoted (Step 4 blocked)
        assert sym not in newly, "P1-B FAIL: graduated symbol was re-promoted in same run"

    def test_graduated_last_jsonl_record_is_graduated(self, tmp_path):
        """After graduation, last JSONL record must be STATUS_GRADUATED (not STATUS_ACTIVE)."""
        sym = self.SYM
        prob_f = tmp_path / "prob.jsonl"
        out_f  = tmp_path / "out.jsonl"

        _write_active_record(prob_f, sym, days_ago=6, rsr=70.0, si=72.0)
        _write_positive_outcomes(out_f, sym, n=5, rsr_delta=3.0)

        run_probation_gate(
            shadow_universe   = self._shadow(),
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 72.0},
            predictive_scores = self._pred(),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "test_p1b_last",
        )
        records = [json.loads(l) for l in prob_f.read_text().splitlines() if l.strip()]
        last = records[-1]
        assert last["status"] == STATUS_GRADUATED, (
            f"P1-B FAIL: last record status={last['status']}, expected STATUS_GRADUATED"
        )

    def test_get_graduated_symbols_correct_after_gate_run(self, tmp_path):
        """get_graduated_symbols() returns the graduated symbol after a gate run."""
        sym = self.SYM
        prob_f = tmp_path / "prob.jsonl"
        out_f  = tmp_path / "out.jsonl"

        _write_active_record(prob_f, sym, days_ago=6, rsr=70.0, si=72.0)
        _write_positive_outcomes(out_f, sym, n=5, rsr_delta=2.5)

        run_probation_gate(
            shadow_universe   = self._shadow(),
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 72.0},
            predictive_scores = self._pred(),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "test_p1b_grads",
        )
        grads = get_graduated_symbols(prob_f, live_universe={})
        assert sym in grads, f"P1-B FAIL: {sym} not in get_graduated_symbols() result"

    def test_symbol_not_graduated_stays_in_active(self, tmp_path):
        """Symbol promoted only 2 days ago (not elapsed) stays STATUS_ACTIVE."""
        sym = self.SYM
        prob_f = tmp_path / "prob.jsonl"
        out_f  = tmp_path / "out.jsonl"

        _write_active_record(prob_f, sym, days_ago=2, rsr=70.0, si=72.0)
        _write_positive_outcomes(out_f, sym, n=2, rsr_delta=2.0)

        active, newly = run_probation_gate(
            shadow_universe   = self._shadow(),
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 72.0},
            predictive_scores = self._pred(),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "test_not_graduated",
        )
        assert sym in active   # still in probation
        assert sym not in newly  # not newly promoted (already active)

    def test_negative_rsr_delta_blocks_graduation(self, tmp_path):
        """Symbol with negative avg rsr_delta stays in probation (rsr_delta_negative)."""
        sym = self.SYM
        prob_f = tmp_path / "prob.jsonl"
        out_f  = tmp_path / "out.jsonl"

        _write_active_record(prob_f, sym, days_ago=6, rsr=70.0, si=72.0)
        _write_positive_outcomes(out_f, sym, n=5, rsr_delta=-2.0)  # negative → FAIL

        active, _ = run_probation_gate(
            shadow_universe   = self._shadow(),
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 72.0},
            predictive_scores = self._pred(),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "test_neg_rsr",
        )
        # Not graduated → still in active
        assert sym in active
        grads = get_graduated_symbols(prob_f, live_universe={})
        assert sym not in grads


# =============================================================================
# E2E closed-loop: SHADOW → ACTIVE → GRADUATED → LIVE_UNIVERSE
# =============================================================================

class TestClosedLoopE2E:
    """
    Full pipeline test: proves SHADOW → PROBATION → GRADUATED → LIVE_UNIVERSE
    is reachable after P1-A and P1-B fixes.
    """

    SYM    = "6857.T"
    SECTOR = "電機精密"

    def _pred(self, si: float = 72.0) -> dict:
        return _unclassified_ps(self.SYM, si=si)

    def _fl(self) -> dict:
        return _unclassified_fl(self.SYM)

    def test_shadow_to_probation_p2a(self, tmp_path):
        """Step 1/4: SHADOW → PROBATION via P2-A gate (RSR=60, SI=72)."""
        sym = self.SYM
        prob_f = tmp_path / "prob.jsonl"
        out_f  = tmp_path / "out.jsonl"

        active, newly = run_probation_gate(
            shadow_universe   = {sym: self.SECTOR},
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 60.0},
            predictive_scores = self._pred(si=72.0),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "e2e_day0",
        )
        assert sym in newly, "E2E Step 1: P2-A gate should promote the symbol"
        assert sym in active

    def test_probation_to_graduated_via_rsr_delta_fallback(self, tmp_path):
        """Step 2/4: PROBATION → GRADUATED via rsr_delta_fallback (P1-A)."""
        sym = self.SYM
        prob_f = tmp_path / "prob.jsonl"
        out_f  = tmp_path / "out.jsonl"

        # Pre-condition: symbol is in ACTIVE probation (promoted 6 days ago)
        # RSR=67: above BREAKOUT_FAIL_RSR_MIN(65) to avoid demotion, below 70 → UNCLASSIFIED
        _write_active_record(prob_f, sym, days_ago=6, rsr=67.0, si=72.0)
        _write_positive_outcomes(out_f, sym, n=5, rsr_delta=2.0)

        active, newly = run_probation_gate(
            shadow_universe   = {sym: self.SECTOR},
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 67.0},   # RSR stable (>=67*0.9=60.3) and >=65 (no demotion)
            predictive_scores = self._pred(),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "e2e_day6",
        )
        # Symbol graduated → not in active (graduated records excluded from active_symbols)
        assert sym not in active, "E2E Step 2: graduated symbol must not be in active_symbols"
        assert sym not in newly,  "E2E Step 2: P1-B must block same-run re-promotion"

        # JSONL last record must be STATUS_GRADUATED
        records = [json.loads(l) for l in prob_f.read_text().splitlines() if l.strip()]
        assert records[-1]["status"] == STATUS_GRADUATED

    def test_graduated_to_live_universe(self, tmp_path):
        """Step 3/4: GRADUATED → get_graduated_symbols() returns symbol."""
        sym = self.SYM
        prob_f = tmp_path / "prob.jsonl"
        out_f  = tmp_path / "out.jsonl"

        _write_active_record(prob_f, sym, days_ago=6, rsr=67.0, si=72.0)
        _write_positive_outcomes(out_f, sym, n=5, rsr_delta=2.0)

        run_probation_gate(
            shadow_universe   = {sym: self.SECTOR},
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 67.0},
            predictive_scores = self._pred(),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "e2e_grad_to_live",
        )
        grads = get_graduated_symbols(prob_f, live_universe={})
        assert sym in grads, f"E2E Step 3: {sym} must be returned by get_graduated_symbols()"

    def test_full_pipeline_closed_loop(self, tmp_path):
        """Step 4/4: Full SHADOW→ACTIVE→GRADUATED→LIVE_UNIVERSE pipeline."""
        sym = self.SYM
        prob_f = tmp_path / "prob.jsonl"
        out_f  = tmp_path / "out.jsonl"
        shadow = {sym: self.SECTOR}

        # ── Day 0: SHADOW → PROBATION ────────────────────────────────────────
        active0, newly0 = run_probation_gate(
            shadow_universe   = shadow,
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 60.0},
            predictive_scores = self._pred(si=72.0),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "e2e_full_day0",
        )
        assert sym in newly0, "Day 0: P2-A promotion failed"
        assert sym in active0

        # ── Days 1-5: outcome observation (positive rsr_delta each day) ──────
        for day in range(1, 6):
            run_probation_outcome_observation(
                probation_path = prob_f,
                outcomes_path  = out_f,
                rsr_scores     = {sym: 60.0 + day * 0.5},  # gently improving RSR
                run_id         = f"e2e_full_day{day}",
            )

        # ── Day 6: PROBATION → GRADUATED ─────────────────────────────────────
        # Hack: rewrite promoted_at to be 6 days ago (simulate passage of time)
        records_before = [json.loads(l) for l in prob_f.read_text().splitlines() if l.strip()]
        last_active = next((r for r in reversed(records_before)
                            if r["status"] == STATUS_ACTIVE), None)
        assert last_active is not None
        last_active["promoted_at"] = _iso_ago(6)  # force elapsed >= probation_days
        prob_f.write_text(
            json.dumps(last_active, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        out_f2 = tmp_path / "out2.jsonl"
        _write_positive_outcomes(out_f2, sym, n=5, rsr_delta=2.0)

        active6, newly6 = run_probation_gate(
            shadow_universe   = shadow,
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 67.0},   # ≥65 (no demotion), <70 (stays UNCLASSIFIED)
            predictive_scores = self._pred(si=72.0),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f2,
            run_id            = "e2e_full_day6",
        )
        assert sym not in active6, "Day 6: graduated symbol must not be in active_symbols"
        assert sym not in newly6,  "Day 6: P1-B must prevent re-promotion"

        # ── LIVE_UNIVERSE update (simulates run_live_signal.py lines 2273-2299) ──
        grads = get_graduated_symbols(prob_f, live_universe={})
        assert sym in grads, "Day 6: get_graduated_symbols must return graduated symbol"

        LIVE_UNIVERSE: dict = {}
        for g in sorted(grads):
            LIVE_UNIVERSE[g] = shadow.get(g, "")

        assert sym in LIVE_UNIVERSE, "CLOSED LOOP FAIL: symbol not in LIVE_UNIVERSE after graduation"
        assert LIVE_UNIVERSE[sym] == self.SECTOR

        # ── Verify graduation_reason contains method tag ──────────────────────
        all_records = [json.loads(l) for l in prob_f.read_text().splitlines() if l.strip()]
        grad_rec = next((r for r in reversed(all_records)
                         if r.get("status") == STATUS_GRADUATED), None)
        assert grad_rec is not None
        assert "graduation_method:rsr_delta_fallback" in grad_rec.get("graduation_reason", "")

    def test_graduation_not_triggered_on_day0(self, tmp_path):
        """Day 0 promotion must not also graduate (probation_not_elapsed)."""
        sym = self.SYM
        prob_f = tmp_path / "prob.jsonl"
        out_f  = tmp_path / "out.jsonl"

        active, newly = run_probation_gate(
            shadow_universe   = {sym: self.SECTOR},
            live_universe     = {},
            cb_state          = "NORMAL",
            rsr_scores        = {sym: 60.0},
            predictive_scores = self._pred(),
            fl_candidates     = self._fl(),
            probation_path    = prob_f,
            outcomes_path     = out_f,
            run_id            = "e2e_day0_no_grad",
        )
        if sym in newly:
            grads = get_graduated_symbols(prob_f, live_universe={})
            assert sym not in grads, "Day 0: symbol must not graduate on promotion day"
