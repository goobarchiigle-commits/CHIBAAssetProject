"""
tests/analytics/test_future_leader_transition.py — Future Leader Transition Analytics tests

Covers:
  - transition correctness (pair detection, from/to state resolution)
  - duplicate prevention (append-only dedup)
  - CLEAN-only filtering
  - insufficient_forward_data exclusion
  - append-only behavior
  - no lookahead contamination (pure function property)
  - no execution imports
  - observation_only invariant
  - delta_days window correctness (0 < d <= 10)
"""
import inspect
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

from src.analytics.future_leader_transition import (
    TransitionRecord,
    TransitionSummary,
    _build_merged_lookup,
    _filter_clean,
    _load_jsonl,
    _load_transition_keys,
    _resolve_state,
    _write_transition_log,
    compute_transition_summary,
    detect_transitions,
    format_transition_section,
    materialize_transitions,
    run_future_leader_transition_hook,
)

TODAY = date(2026, 5, 25)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs(tmp_path):
    dirs = {
        "failure":       tmp_path / "failure",
        "archetype":     tmp_path / "archetype",
        "survivability": tmp_path / "survivability",
        "regime":        tmp_path / "regime",
        "transition":    tmp_path / "transition",
        "reports":       tmp_path / "reports",
        "tmp":           tmp_path,
    }
    for d in dirs.values():
        if d != tmp_path:
            d.mkdir(parents=True)
    return dirs


def _write_jsonl(path: Path, records: list) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_failure(
    symbol="1234.T",
    obs_date="2026-01-10",
    primary_failure_reason="market_selloff",
    failure_detected=True,
    failure_day_offset=2,
    data_quality_weight=1.0,
    integrity_state="CLEAN",
) -> dict:
    return {
        "symbol":                 symbol,
        "observation_date":       obs_date,
        "primary_failure_reason": primary_failure_reason,
        "failure_detected":       failure_detected,
        "failure_day_offset":     failure_day_offset,
        "data_quality_weight":    data_quality_weight,
        "integrity_state":        integrity_state,
        "observation_only":       True,
    }


def _make_archetype(
    symbol="1234.T",
    obs_date="2026-01-10",
    failure_archetype="regime_mismatch",
    data_quality_weight=1.0,
) -> dict:
    return {
        "symbol":             symbol,
        "observation_date":   obs_date,
        "failure_archetype":  failure_archetype,
        "data_quality_weight": data_quality_weight,
        "observation_only":   True,
    }


def _make_survivability(
    symbol="1234.T",
    obs_date="2026-01-10",
    insufficient_forward_data=False,
    data_quality_weight=1.0,
) -> dict:
    return {
        "symbol":                    symbol,
        "observation_date":          obs_date,
        "insufficient_forward_data": insufficient_forward_data,
        "data_quality_weight":       data_quality_weight,
        "observation_only":          True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. State resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveState:
    def test_primary_failure_reason_priority(self):
        rec = {
            "failure_detected": True,
            "primary_failure_reason": "market_selloff",
            "failure_archetype": "regime_mismatch",
        }
        assert _resolve_state(rec) == "market_selloff"

    def test_falls_back_to_archetype_when_no_detection(self):
        rec = {
            "failure_detected": False,
            "primary_failure_reason": "market_selloff",
            "failure_archetype": "regime_mismatch",
        }
        assert _resolve_state(rec) == "regime_mismatch"

    def test_falls_back_to_archetype_when_reason_is_none(self):
        rec = {
            "failure_detected": True,
            "primary_failure_reason": None,
            "failure_archetype": "institutional_exit",
        }
        assert _resolve_state(rec) == "institutional_exit"

    def test_none_when_both_are_unresolvable(self):
        rec = {
            "failure_detected": False,
            "primary_failure_reason": None,
            "failure_archetype": "unresolved",
        }
        assert _resolve_state(rec) is None

    def test_none_when_reason_is_empty_string(self):
        rec = {
            "failure_detected": True,
            "primary_failure_reason": "",
            "failure_archetype": "unresolved",
        }
        assert _resolve_state(rec) is None

    def test_skip_none_archetype(self):
        rec = {
            "failure_detected": False,
            "primary_failure_reason": None,
            "failure_archetype": None,
        }
        assert _resolve_state(rec) is None

    def test_archetype_only_record(self):
        rec = {
            "failure_detected": False,
            "primary_failure_reason": None,
            "failure_archetype": "liquidity_fade",
        }
        assert _resolve_state(rec) == "liquidity_fade"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Merged lookup construction
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildMergedLookup:
    def test_failure_record_merged(self):
        fail = [_make_failure("A.T", "2026-01-10", "rsr_reversal")]
        arch = []
        surv = []
        merged = _build_merged_lookup(fail, arch, surv)
        assert ("A.T", "2026-01-10") in merged
        assert merged[("A.T", "2026-01-10")]["primary_failure_reason"] == "rsr_reversal"

    def test_archetype_overlaid(self):
        fail = [_make_failure("A.T", "2026-01-10", "market_selloff")]
        arch = [_make_archetype("A.T", "2026-01-10", "regime_mismatch")]
        surv = []
        merged = _build_merged_lookup(fail, arch, surv)
        rec = merged[("A.T", "2026-01-10")]
        assert rec["failure_archetype"] == "regime_mismatch"
        assert rec["primary_failure_reason"] == "market_selloff"

    def test_insufficient_forward_data_overlaid(self):
        fail = [_make_failure("B.T", "2026-01-11")]
        arch = []
        surv = [_make_survivability("B.T", "2026-01-11", insufficient_forward_data=True)]
        merged = _build_merged_lookup(fail, arch, surv)
        assert merged[("B.T", "2026-01-11")]["insufficient_forward_data"] is True

    def test_archetype_only_creates_entry(self):
        fail = []
        arch = [_make_archetype("C.T", "2026-01-12", "slow_bleed")]
        surv = []
        merged = _build_merged_lookup(fail, arch, surv)
        assert ("C.T", "2026-01-12") in merged
        assert merged[("C.T", "2026-01-12")]["failure_archetype"] == "slow_bleed"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Transition detection correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestDetectTransitions:
    def _make_lookup(self, entries: list) -> Dict[Tuple[str, str], dict]:
        return {(e["symbol"], e["observation_date"]): e for e in entries}

    def _base_rec(self, sym, obs_date, reason="market_selloff", detected=True,
                  arch="regime_mismatch", dq=1.0, insuf=False) -> dict:
        return {
            "symbol": sym,
            "observation_date": obs_date,
            "failure_detected": detected,
            "primary_failure_reason": reason,
            "failure_archetype": arch,
            "failure_day_offset": 2,
            "data_quality_weight": dq,
            "integrity_state": "CLEAN",
            "insufficient_forward_data": insuf,
        }

    def test_basic_transition_detected(self):
        lookup = self._make_lookup([
            self._base_rec("X.T", "2026-01-10", "market_selloff"),
            self._base_rec("X.T", "2026-01-15", "rsr_reversal"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert len(transitions) == 1
        assert transitions[0].transition_pair == "market_selloff->rsr_reversal"
        assert transitions[0].transition_days == 5

    def test_transition_within_10d_window(self):
        lookup = self._make_lookup([
            self._base_rec("Y.T", "2026-01-01", "volume_faded"),
            self._base_rec("Y.T", "2026-01-10", "rsr_reversal"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert len(transitions) == 1
        assert transitions[0].transition_days == 9

    def test_transition_exactly_10d(self):
        lookup = self._make_lookup([
            self._base_rec("Z.T", "2026-01-01", "volume_faded"),
            self._base_rec("Z.T", "2026-01-11", "rsr_reversal"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert len(transitions) == 1

    def test_transition_over_10d_excluded(self):
        lookup = self._make_lookup([
            self._base_rec("A.T", "2026-01-01", "volume_faded"),
            self._base_rec("A.T", "2026-01-12", "rsr_reversal"),   # 11 days
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert len(transitions) == 0

    def test_same_day_excluded(self):
        lookup = self._make_lookup([
            self._base_rec("B.T", "2026-01-05", "market_selloff"),
        ])
        # Only one date for symbol → no pairs
        transitions = detect_transitions(lookup, set(), {})
        assert len(transitions) == 0

    def test_unresolvable_from_state_skipped(self):
        lookup = self._make_lookup([
            self._base_rec("C.T", "2026-01-01", reason=None, detected=False,
                           arch="unresolved"),
            self._base_rec("C.T", "2026-01-05", "rsr_reversal"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert len(transitions) == 0

    def test_unresolvable_to_state_skipped(self):
        lookup = self._make_lookup([
            self._base_rec("D.T", "2026-01-01", "market_selloff"),
            self._base_rec("D.T", "2026-01-05", reason=None, detected=False,
                           arch="unresolved"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert len(transitions) == 0

    def test_transition_pair_format(self):
        lookup = self._make_lookup([
            self._base_rec("E.T", "2026-01-01", "weak_followthrough"),
            self._base_rec("E.T", "2026-01-04", "rsr_reversal"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert transitions[0].transition_pair == "weak_followthrough->rsr_reversal"

    def test_consecutive_pairs_only(self):
        """Three observations: A→B and B→C detected; A→C only if within 10d."""
        lookup = self._make_lookup([
            self._base_rec("F.T", "2026-01-01", "market_selloff"),
            self._base_rec("F.T", "2026-01-04", "rsr_reversal"),
            self._base_rec("F.T", "2026-01-07", "volume_faded"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        pairs = [t.transition_pair for t in transitions]
        assert "market_selloff->rsr_reversal" in pairs
        assert "rsr_reversal->volume_faded" in pairs
        assert len(transitions) == 2

    def test_multiple_symbols_independent(self):
        lookup = self._make_lookup([
            self._base_rec("G.T", "2026-01-01", "market_selloff"),
            self._base_rec("G.T", "2026-01-05", "rsr_reversal"),
            self._base_rec("H.T", "2026-01-02", "volume_faded"),
            self._base_rec("H.T", "2026-01-06", "gap_fill"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert len(transitions) == 2
        syms = {t.symbol for t in transitions}
        assert "G.T" in syms and "H.T" in syms

    def test_regime_from_regime_lookup(self):
        lookup = self._make_lookup([
            self._base_rec("I.T", "2026-01-01", "market_selloff"),
            self._base_rec("I.T", "2026-01-05", "rsr_reversal"),
        ])
        regime_lookup = {"2026-01-01": "risk_off"}
        transitions = detect_transitions(lookup, set(), regime_lookup)
        assert transitions[0].regime == "risk_off"

    def test_regime_fallback_unclassified(self):
        lookup = self._make_lookup([
            self._base_rec("J.T", "2026-01-01", "market_selloff"),
            self._base_rec("J.T", "2026-01-05", "rsr_reversal"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert transitions[0].regime == "unclassified"

    def test_observation_only_always_true(self):
        lookup = self._make_lookup([
            self._base_rec("K.T", "2026-01-01", "market_selloff"),
            self._base_rec("K.T", "2026-01-05", "rsr_reversal"),
        ])
        transitions = detect_transitions(lookup, set(), {})
        assert all(t.observation_only is True for t in transitions)

    def test_archetype_fallback_used_when_no_failure_detected(self):
        lookup = {
            ("L.T", "2026-01-01"): {
                "symbol": "L.T", "observation_date": "2026-01-01",
                "failure_detected": False, "primary_failure_reason": None,
                "failure_archetype": "liquidity_fade",
                "failure_day_offset": None, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
            ("L.T", "2026-01-05"): {
                "symbol": "L.T", "observation_date": "2026-01-05",
                "failure_detected": True, "primary_failure_reason": "rsr_reversal",
                "failure_archetype": "institutional_exit",
                "failure_day_offset": 4, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
        }
        transitions = detect_transitions(lookup, set(), {})
        assert len(transitions) == 1
        assert transitions[0].from_state == "liquidity_fade"
        assert transitions[0].to_state == "rsr_reversal"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Delta_days window correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaDaysWindow:
    def _lookup(self, d1, d2, sym="X.T"):
        return {
            (sym, d1): {
                "symbol": sym, "observation_date": d1,
                "failure_detected": True, "primary_failure_reason": "market_selloff",
                "failure_archetype": "regime_mismatch",
                "failure_day_offset": 2, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
            (sym, d2): {
                "symbol": sym, "observation_date": d2,
                "failure_detected": True, "primary_failure_reason": "rsr_reversal",
                "failure_archetype": "institutional_exit",
                "failure_day_offset": 3, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
        }

    @pytest.mark.parametrize("d1,d2,expected", [
        ("2026-01-01", "2026-01-02", 1),   # 1d — within
        ("2026-01-01", "2026-01-10", 1),   # 9d — within
        ("2026-01-01", "2026-01-11", 1),   # 10d — boundary (included)
        ("2026-01-01", "2026-01-12", 0),   # 11d — excluded
        ("2026-01-01", "2026-02-01", 0),   # 31d — excluded
    ])
    def test_delta_window(self, d1, d2, expected):
        lookup = self._lookup(d1, d2)
        result = detect_transitions(lookup, set(), {})
        assert len(result) == expected


# ─────────────────────────────────────────────────────────────────────────────
# 5. Duplicate prevention
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicatePrevention:
    def test_existing_key_skipped(self):
        lookup = {
            ("X.T", "2026-01-01"): {
                "symbol": "X.T", "observation_date": "2026-01-01",
                "failure_detected": True, "primary_failure_reason": "market_selloff",
                "failure_archetype": "regime_mismatch",
                "failure_day_offset": 2, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
            ("X.T", "2026-01-05"): {
                "symbol": "X.T", "observation_date": "2026-01-05",
                "failure_detected": True, "primary_failure_reason": "rsr_reversal",
                "failure_archetype": "institutional_exit",
                "failure_day_offset": 3, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
        }
        existing = {("X.T", "2026-01-01", "2026-01-05")}
        result = detect_transitions(lookup, existing, {})
        assert len(result) == 0

    def test_new_pair_processed_when_other_already_exists(self):
        lookup = {
            ("Y.T", "2026-01-01"): {
                "symbol": "Y.T", "observation_date": "2026-01-01",
                "failure_detected": True, "primary_failure_reason": "market_selloff",
                "failure_archetype": "regime_mismatch",
                "failure_day_offset": 2, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
            ("Y.T", "2026-01-05"): {
                "symbol": "Y.T", "observation_date": "2026-01-05",
                "failure_detected": True, "primary_failure_reason": "rsr_reversal",
                "failure_archetype": "institutional_exit",
                "failure_day_offset": 3, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
            ("Z.T", "2026-01-02"): {
                "symbol": "Z.T", "observation_date": "2026-01-02",
                "failure_detected": True, "primary_failure_reason": "volume_faded",
                "failure_archetype": "liquidity_fade",
                "failure_day_offset": 6, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
            ("Z.T", "2026-01-06"): {
                "symbol": "Z.T", "observation_date": "2026-01-06",
                "failure_detected": True, "primary_failure_reason": "rsr_reversal",
                "failure_archetype": "institutional_exit",
                "failure_day_offset": 3, "data_quality_weight": 1.0,
                "integrity_state": "CLEAN", "insufficient_forward_data": False,
            },
        }
        existing = {("Y.T", "2026-01-01", "2026-01-05")}
        result = detect_transitions(lookup, existing, {})
        assert len(result) == 1
        assert result[0].symbol == "Z.T"

    def test_transition_key_roundtrip(self, tmp_dirs):
        trans_dir = tmp_dirs["transition"]
        rec = TransitionRecord(
            symbol="A.T",
            observation_date="2026-01-01",
            to_observation_date="2026-01-05",
            transition_pair="market_selloff->rsr_reversal",
            from_state="market_selloff",
            to_state="rsr_reversal",
            transition_days=4,
            regime="risk_off",
            survival_days=2,
            half_life_day=None,
            data_quality_weight=1.0,
            integrity_state="CLEAN",
            insufficient_forward_data=False,
        )
        _write_transition_log([rec], trans_dir, today=TODAY)
        keys = _load_transition_keys(trans_dir)
        assert ("A.T", "2026-01-01", "2026-01-05") in keys


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLEAN-only filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanFilter:
    def test_clean_record_kept(self):
        recs = [{"data_quality_weight": 1.0, "insufficient_forward_data": False}]
        assert len(_filter_clean(recs)) == 1

    def test_low_dq_weight_excluded(self):
        recs = [{"data_quality_weight": 0.5, "insufficient_forward_data": False}]
        assert len(_filter_clean(recs)) == 0

    def test_zero_dq_excluded(self):
        recs = [{"data_quality_weight": 0.0, "insufficient_forward_data": False}]
        assert len(_filter_clean(recs)) == 0

    def test_insufficient_fwd_data_excluded(self):
        recs = [{"data_quality_weight": 1.0, "insufficient_forward_data": True}]
        assert len(_filter_clean(recs)) == 0

    def test_missing_dq_treated_as_zero(self):
        recs = [{"insufficient_forward_data": False}]
        assert len(_filter_clean(recs)) == 0

    def test_mixed_records(self):
        recs = [
            {"data_quality_weight": 1.0, "insufficient_forward_data": False},  # CLEAN
            {"data_quality_weight": 0.5, "insufficient_forward_data": False},  # PARTIAL
            {"data_quality_weight": 1.0, "insufficient_forward_data": True},   # insuf
        ]
        assert len(_filter_clean(recs)) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 7. Insufficient_forward_data exclusion
# ─────────────────────────────────────────────────────────────────────────────

class TestInsufficientForwardDataExclusion:
    def test_insuf_fwd_data_persisted_but_excluded_from_aggregation(self, tmp_dirs):
        failure_dir   = tmp_dirs["failure"]
        archetype_dir = tmp_dirs["archetype"]
        surv_dir      = tmp_dirs["survivability"]
        regime_dir    = tmp_dirs["regime"]
        trans_dir     = tmp_dirs["transition"]

        _write_jsonl(failure_dir / "future_leader_failure_20260525.jsonl", [
            _make_failure("A.T", "2026-01-01", "market_selloff"),
            _make_failure("A.T", "2026-01-05", "rsr_reversal"),
        ])
        # Mark first record as insufficient_forward_data
        _write_jsonl(surv_dir / "future_leader_survivability_20260525.jsonl", [
            _make_survivability("A.T", "2026-01-01", insufficient_forward_data=True),
        ])

        count = materialize_transitions(
            failure_dir, archetype_dir, surv_dir, regime_dir, trans_dir, today=TODAY
        )
        assert count == 1  # transition persisted

        # But excluded from CLEAN aggregation
        all_records = _load_jsonl(trans_dir, "future_leader_transition_*.jsonl")
        assert all_records[0]["insufficient_forward_data"] is True

        clean = _filter_clean(all_records)
        assert len(clean) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 8. Append-only behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendOnly:
    def test_write_appends_to_existing(self, tmp_dirs):
        trans_dir = tmp_dirs["transition"]
        p = trans_dir / f"future_leader_transition_{TODAY.strftime('%Y%m%d')}.jsonl"

        def _rec(sym, from_d, to_d, pair):
            return TransitionRecord(
                symbol=sym, observation_date=from_d, to_observation_date=to_d,
                transition_pair=pair, from_state=pair.split("->")[0],
                to_state=pair.split("->")[1], transition_days=4,
                regime="range", survival_days=2, half_life_day=None,
                data_quality_weight=1.0, integrity_state="CLEAN",
                insufficient_forward_data=False,
            )

        _write_transition_log([_rec("A.T", "2026-01-01", "2026-01-05", "a->b")], trans_dir, today=TODAY)
        _write_transition_log([_rec("B.T", "2026-01-02", "2026-01-06", "c->d")], trans_dir, today=TODAY)

        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_empty_write_no_file(self, tmp_dirs):
        trans_dir = tmp_dirs["transition"]
        _write_transition_log([], trans_dir, today=TODAY)
        assert not list(trans_dir.glob("*.jsonl"))


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full materialize pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestMaterializePipeline:
    def test_basic_pipeline(self, tmp_dirs):
        fail_dir  = tmp_dirs["failure"]
        arch_dir  = tmp_dirs["archetype"]
        surv_dir  = tmp_dirs["survivability"]
        reg_dir   = tmp_dirs["regime"]
        trans_dir = tmp_dirs["transition"]

        _write_jsonl(fail_dir / "future_leader_failure_20260525.jsonl", [
            _make_failure("X.T", "2026-01-01", "market_selloff"),
            _make_failure("X.T", "2026-01-06", "rsr_reversal"),
        ])

        count = materialize_transitions(fail_dir, arch_dir, surv_dir, reg_dir, trans_dir, today=TODAY)
        assert count == 1

        records = _load_jsonl(trans_dir, "future_leader_transition_*.jsonl")
        assert len(records) == 1
        assert records[0]["transition_pair"] == "market_selloff->rsr_reversal"
        assert records[0]["transition_days"] == 5

    def test_no_records_returns_zero(self, tmp_dirs):
        count = materialize_transitions(
            tmp_dirs["failure"], tmp_dirs["archetype"],
            tmp_dirs["survivability"], tmp_dirs["regime"],
            tmp_dirs["transition"], today=TODAY,
        )
        assert count == 0

    def test_idempotent(self, tmp_dirs):
        fail_dir  = tmp_dirs["failure"]
        arch_dir  = tmp_dirs["archetype"]
        surv_dir  = tmp_dirs["survivability"]
        reg_dir   = tmp_dirs["regime"]
        trans_dir = tmp_dirs["transition"]

        _write_jsonl(fail_dir / "future_leader_failure_20260525.jsonl", [
            _make_failure("Y.T", "2026-01-01", "volume_faded"),
            _make_failure("Y.T", "2026-01-04", "rsr_reversal"),
        ])

        c1 = materialize_transitions(fail_dir, arch_dir, surv_dir, reg_dir, trans_dir, today=TODAY)
        c2 = materialize_transitions(fail_dir, arch_dir, surv_dir, reg_dir, trans_dir, today=TODAY)

        assert c1 == 1
        assert c2 == 0  # already persisted

        records = _load_jsonl(trans_dir, "future_leader_transition_*.jsonl")
        assert len(records) == 1

    def test_archetype_used_as_state_source(self, tmp_dirs):
        fail_dir  = tmp_dirs["failure"]
        arch_dir  = tmp_dirs["archetype"]
        surv_dir  = tmp_dirs["survivability"]
        reg_dir   = tmp_dirs["regime"]
        trans_dir = tmp_dirs["transition"]

        # No failure_detected, use archetype
        _write_jsonl(fail_dir / "future_leader_failure_20260525.jsonl", [
            {**_make_failure("Z.T", "2026-01-01"), "failure_detected": False,
             "primary_failure_reason": None},
            {**_make_failure("Z.T", "2026-01-05"), "primary_failure_reason": "rsr_reversal"},
        ])
        _write_jsonl(arch_dir / "future_leader_archetype_20260525.jsonl", [
            _make_archetype("Z.T", "2026-01-01", "liquidity_fade"),
        ])

        count = materialize_transitions(fail_dir, arch_dir, surv_dir, reg_dir, trans_dir, today=TODAY)
        assert count == 1

        records = _load_jsonl(trans_dir, "future_leader_transition_*.jsonl")
        assert records[0]["from_state"] == "liquidity_fade"
        assert records[0]["to_state"] == "rsr_reversal"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Transition summary analytics
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeTransitionSummary:
    def _populate_transition_jsonl(self, trans_dir: Path, records: list) -> None:
        p = trans_dir / f"future_leader_transition_{TODAY.strftime('%Y%m%d')}.jsonl"
        _write_jsonl(p, records)

    def _clean_trans(self, pair, trans_days=4, surv_days=2, dq=1.0, insuf=False):
        return {
            "transition_pair": pair,
            "transition_days": trans_days,
            "survival_days": surv_days,
            "half_life_day": None,
            "data_quality_weight": dq,
            "insufficient_forward_data": insuf,
            "observation_only": True,
        }

    def test_count_correct(self, tmp_dirs):
        trans_dir = tmp_dirs["transition"]
        recs = [self._clean_trans("a->b")] * 3 + [self._clean_trans("c->d")] * 2
        self._populate_transition_jsonl(trans_dir, recs)
        summaries = compute_transition_summary(trans_dir)
        counts = {s.transition_pair: s.count for s in summaries}
        assert counts["a->b"] == 3
        assert counts["c->d"] == 2

    def test_frequency_ranking(self, tmp_dirs):
        trans_dir = tmp_dirs["transition"]
        recs = [self._clean_trans("rare->x")] + [self._clean_trans("common->y")] * 5
        self._populate_transition_jsonl(trans_dir, recs)
        summaries = compute_transition_summary(trans_dir)
        assert summaries[0].transition_pair == "common->y"
        assert summaries[1].transition_pair == "rare->x"

    def test_partial_excluded_from_aggregation(self, tmp_dirs):
        trans_dir = tmp_dirs["transition"]
        recs = [
            self._clean_trans("a->b", dq=1.0),
            self._clean_trans("a->b", dq=0.5),  # PARTIAL
        ]
        self._populate_transition_jsonl(trans_dir, recs)
        summaries = compute_transition_summary(trans_dir)
        s = next(s for s in summaries if s.transition_pair == "a->b")
        assert s.count == 1

    def test_median_transition_days(self, tmp_dirs):
        trans_dir = tmp_dirs["transition"]
        recs = [
            self._clean_trans("p->q", trans_days=2),
            self._clean_trans("p->q", trans_days=4),
            self._clean_trans("p->q", trans_days=6),
        ]
        self._populate_transition_jsonl(trans_dir, recs)
        summaries = compute_transition_summary(trans_dir)
        s = next(s for s in summaries if s.transition_pair == "p->q")
        assert s.median_transition_days == pytest.approx(4.0)

    def test_none_when_insufficient_sample(self, tmp_dirs):
        trans_dir = tmp_dirs["transition"]
        recs = [self._clean_trans("a->b")] * 2  # < _MIN_SAMPLE=3
        self._populate_transition_jsonl(trans_dir, recs)
        summaries = compute_transition_summary(trans_dir)
        s = next(s for s in summaries if s.transition_pair == "a->b")
        assert s.median_transition_days is None

    def test_empty_dir_returns_empty(self, tmp_dirs):
        summaries = compute_transition_summary(tmp_dirs["transition"])
        assert summaries == []


# ─────────────────────────────────────────────────────────────────────────────
# 11. Report section format
# ─────────────────────────────────────────────────────────────────────────────

class TestFormatTransitionSection:
    def test_header_present(self):
        out = format_transition_section([])
        assert "=== FAILURE TRANSITIONS ===" in out

    def test_no_data_message(self):
        out = format_transition_section([])
        assert "(no transition records)" in out

    def test_pair_and_count_shown(self):
        summaries = [TransitionSummary("a->b", 5, 3.0, 2.0, None)]
        out = format_transition_section(summaries)
        assert "a->b:" in out
        assert "count: 5" in out

    def test_metrics_shown_when_available(self):
        summaries = [TransitionSummary("x->y", 4, 4.0, 6.0, 3.0)]
        out = format_transition_section(summaries)
        assert "median transition" in out
        assert "median survival" in out
        assert "median half-life" in out

    def test_none_metrics_omitted(self):
        summaries = [TransitionSummary("a->b", 4, None, None, None)]
        out = format_transition_section(summaries)
        assert "median transition" not in out


# ─────────────────────────────────────────────────────────────────────────────
# 12. No lookahead contamination
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLookahead:
    def test_detect_transitions_is_pure(self):
        src = inspect.getsource(detect_transitions)
        # Check for actual code calls (not docstring mentions)
        forbidden = ["pd.read", "open(", "requests", "yfinance",
                     "datetime.now(", "date.today(", "load_ohlcv", "read_parquet"]
        for token in forbidden:
            assert token not in src, f"lookahead risk: '{token}' in detect_transitions"

    def test_resolve_state_is_pure(self):
        src = inspect.getsource(_resolve_state)
        forbidden = ["pd.read", "open(", "requests", "datetime.now", "date.today"]
        for token in forbidden:
            assert token not in src, f"lookahead risk: '{token}' in _resolve_state"

    def test_no_ohlcv_in_module(self):
        import src.analytics.future_leader_transition as mod
        src_code = inspect.getsource(mod)
        forbidden_imports = ["yfinance", "parquet", "ohlcv_loader", "_load_ohlcv"]
        for token in forbidden_imports:
            assert token not in src_code, f"forbidden: '{token}' in transition module"


# ─────────────────────────────────────────────────────────────────────────────
# 13. No execution imports
# ─────────────────────────────────────────────────────────────────────────────

class TestNoExecutionImports:
    _FORBIDDEN = ["allocat", "broker", "signal_bridge",
                  "deployable_universe", "order", "execution", "routing"]

    def test_no_forbidden_imports(self):
        import src.analytics.future_leader_transition as mod
        src_code = inspect.getsource(mod)
        for line in src_code.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                for forbidden in self._FORBIDDEN:
                    assert forbidden not in stripped.lower(), (
                        f"forbidden import '{forbidden}': {stripped}"
                    )

    def test_module_importable_standalone(self):
        import src.analytics.future_leader_transition as mod
        assert hasattr(mod, "run_future_leader_transition_hook")
        assert hasattr(mod, "detect_transitions")
        assert hasattr(mod, "compute_transition_summary")


# ─────────────────────────────────────────────────────────────────────────────
# 14. Observation_only invariant
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationOnlyInvariant:
    def test_record_default(self):
        rec = TransitionRecord(
            symbol="X.T", observation_date="2026-01-01",
            to_observation_date="2026-01-05",
            transition_pair="a->b", from_state="a", to_state="b",
            transition_days=4, regime="range",
            survival_days=None, half_life_day=None,
            data_quality_weight=1.0, integrity_state="CLEAN",
            insufficient_forward_data=False,
        )
        assert rec.observation_only is True

    def test_observation_only_in_jsonl(self, tmp_dirs):
        trans_dir = tmp_dirs["transition"]
        rec = TransitionRecord(
            symbol="Y.T", observation_date="2026-01-01",
            to_observation_date="2026-01-05",
            transition_pair="x->y", from_state="x", to_state="y",
            transition_days=4, regime="range",
            survival_days=2, half_life_day=None,
            data_quality_weight=1.0, integrity_state="CLEAN",
            insufficient_forward_data=False,
        )
        _write_transition_log([rec], trans_dir, today=TODAY)
        records = _load_jsonl(trans_dir, "future_leader_transition_*.jsonl")
        assert records[0]["observation_only"] is True

    def test_materialize_sets_observation_only(self, tmp_dirs):
        fail_dir  = tmp_dirs["failure"]
        trans_dir = tmp_dirs["transition"]

        _write_jsonl(fail_dir / "future_leader_failure_20260525.jsonl", [
            _make_failure("OBS.T", "2026-01-01", "market_selloff"),
            _make_failure("OBS.T", "2026-01-05", "rsr_reversal"),
        ])
        materialize_transitions(
            fail_dir, tmp_dirs["archetype"], tmp_dirs["survivability"],
            tmp_dirs["regime"], trans_dir, today=TODAY,
        )
        records = _load_jsonl(trans_dir, "future_leader_transition_*.jsonl")
        assert records[0]["observation_only"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 15. Hook integration
# ─────────────────────────────────────────────────────────────────────────────

class TestHookIntegration:
    def test_hook_creates_report_section(self, tmp_dirs):
        fail_dir  = tmp_dirs["failure"]
        arch_dir  = tmp_dirs["archetype"]
        surv_dir  = tmp_dirs["survivability"]
        reg_dir   = tmp_dirs["regime"]
        trans_dir = tmp_dirs["transition"]
        rep_dir   = tmp_dirs["reports"]

        _write_jsonl(fail_dir / "future_leader_failure_20260525.jsonl", [
            _make_failure("H.T", "2026-01-01", "market_selloff"),
            _make_failure("H.T", "2026-01-05", "rsr_reversal"),
        ])

        run_future_leader_transition_hook(
            failure_log_dir=fail_dir,
            archetype_log_dir=arch_dir,
            survivability_log_dir=surv_dir,
            regime_log_dir=reg_dir,
            transition_log_dir=trans_dir,
            reports_dir=rep_dir,
            today=TODAY,
        )

        report_path = rep_dir / f"future_leader_report_{TODAY.strftime('%Y%m%d')}.md"
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "FAILURE TRANSITIONS" in content

    def test_hook_fail_open_on_bad_dirs(self, tmp_dirs):
        bad = tmp_dirs["tmp"] / "nonexistent"
        run_future_leader_transition_hook(
            failure_log_dir=bad / "f",
            archetype_log_dir=bad / "a",
            survivability_log_dir=bad / "s",
            regime_log_dir=bad / "r",
            transition_log_dir=bad / "t",
            reports_dir=bad / "rep",
            today=TODAY,
        )  # must not raise

    def test_hook_idempotent_no_duplicate_transitions(self, tmp_dirs):
        fail_dir  = tmp_dirs["failure"]
        arch_dir  = tmp_dirs["archetype"]
        surv_dir  = tmp_dirs["survivability"]
        reg_dir   = tmp_dirs["regime"]
        trans_dir = tmp_dirs["transition"]
        rep_dir   = tmp_dirs["reports"]

        _write_jsonl(fail_dir / "future_leader_failure_20260525.jsonl", [
            _make_failure("I.T", "2026-01-01", "volume_faded"),
            _make_failure("I.T", "2026-01-04", "rsr_reversal"),
        ])

        run_future_leader_transition_hook(
            fail_dir, arch_dir, surv_dir, reg_dir, trans_dir, rep_dir, today=TODAY
        )
        run_future_leader_transition_hook(
            fail_dir, arch_dir, surv_dir, reg_dir, trans_dir, rep_dir, today=TODAY
        )

        records = _load_jsonl(trans_dir, "future_leader_transition_*.jsonl")
        assert len(records) == 1
