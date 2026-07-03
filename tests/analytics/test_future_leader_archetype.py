"""
tests/analytics/test_future_leader_archetype.py — Future Leader Archetype Layer tests

Covers:
  - archetype classification correctness (all rules)
  - transition telemetry correctness
  - CLEAN-only filtering
  - append-only behavior
  - duplicate prevention
  - no lookahead contamination (pure function property)
  - observation_only invariant
  - no execution imports
"""
import importlib
import inspect
import json
import sys
import types
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import pytest

from src.analytics.future_leader_archetype import (
    ArchetypeRecord,
    ArchetypeSummary,
    FailureArchetype,
    _filter_clean,
    _load_archetype_keys,
    _load_failure_records,
    _load_jsonl,
    _load_regime_lookup,
    _write_archetype_log,
    build_transition_pair,
    classify_failure_archetype,
    compute_archetype_summary,
    format_archetype_section,
    materialize_archetypes,
    run_future_leader_archetype_hook,
)

JST = timezone(timedelta(hours=9))
TODAY = date(2026, 5, 24)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dirs(tmp_path):
    failure_dir   = tmp_path / "failure"
    regime_dir    = tmp_path / "regime"
    archetype_dir = tmp_path / "archetype"
    reports_dir   = tmp_path / "reports"
    for d in (failure_dir, regime_dir, archetype_dir, reports_dir):
        d.mkdir(parents=True)
    return {
        "failure":   failure_dir,
        "regime":    regime_dir,
        "archetype": archetype_dir,
        "reports":   reports_dir,
        "tmp":       tmp_path,
    }


def _write_failure_jsonl(failure_dir: Path, records: list, filename: str = "future_leader_failure_20260524.jsonl") -> None:
    p = failure_dir / filename
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_regime_jsonl(regime_dir: Path, entries: list, filename: str = "future_leader_regime_20260524.jsonl") -> None:
    p = regime_dir / filename
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_failure_rec(
    symbol="1234.T",
    obs_date="2026-01-10",
    primary_failure_reason="market_selloff",
    failure_day_offset=2,
    mfe_10d=0.02,
    mae_10d=-0.03,
    data_quality_weight=1.0,
    integrity_state="CLEAN",
    failure_detected=True,
) -> dict:
    return {
        "symbol":                 symbol,
        "observation_date":       obs_date,
        "primary_failure_reason": primary_failure_reason,
        "failure_day_offset":     failure_day_offset,
        "mfe_10d":                mfe_10d,
        "mae_10d":                mae_10d,
        "data_quality_weight":    data_quality_weight,
        "integrity_state":        integrity_state,
        "failure_detected":       failure_detected,
        "observation_only":       True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 1. Archetype classification correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestClassifyFailureArchetype:
    def test_market_selloff_regime_mismatch(self):
        result = classify_failure_archetype(
            primary_failure_reason="market_selloff",
            survival_days=3,
            mfe_10d=0.01,
            mae_10d=-0.02,
            regime_tag="risk_off",
            rsr_delta=None,
        )
        assert result == FailureArchetype.regime_mismatch

    def test_rsr_reversal_institutional_exit(self):
        result = classify_failure_archetype(
            primary_failure_reason="rsr_reversal",
            survival_days=4,
            mfe_10d=0.03,
            mae_10d=-0.04,
            regime_tag="range",
            rsr_delta=-2.5,
        )
        assert result == FailureArchetype.institutional_exit

    def test_volume_faded_liquidity_fade_survival_ge5(self):
        result = classify_failure_archetype(
            primary_failure_reason="volume_faded",
            survival_days=5,
            mfe_10d=0.01,
            mae_10d=-0.01,
            regime_tag="range",
            rsr_delta=None,
        )
        assert result == FailureArchetype.liquidity_fade

    def test_volume_faded_not_liquidity_fade_survival_lt5(self):
        result = classify_failure_archetype(
            primary_failure_reason="volume_faded",
            survival_days=4,
            mfe_10d=0.01,
            mae_10d=-0.01,
            regime_tag="range",
            rsr_delta=None,
        )
        assert result == FailureArchetype.unresolved

    def test_gap_fill_volatility_trap(self):
        result = classify_failure_archetype(
            primary_failure_reason="gap_fill",
            survival_days=3,
            mfe_10d=0.08,
            mae_10d=-0.07,
            regime_tag="risk_on",
            rsr_delta=None,
        )
        assert result == FailureArchetype.volatility_trap

    def test_gap_fill_not_volatility_trap_mfe_too_low(self):
        result = classify_failure_archetype(
            primary_failure_reason="gap_fill",
            survival_days=3,
            mfe_10d=0.03,   # < 0.05
            mae_10d=-0.07,
            regime_tag="risk_on",
            rsr_delta=None,
        )
        assert result == FailureArchetype.unresolved

    def test_gap_fill_not_volatility_trap_mae_too_small(self):
        result = classify_failure_archetype(
            primary_failure_reason="gap_fill",
            survival_days=3,
            mfe_10d=0.08,
            mae_10d=-0.03,  # > -0.05
            regime_tag="risk_on",
            rsr_delta=None,
        )
        assert result == FailureArchetype.unresolved

    def test_weak_followthrough_instant_rejection_survival_le3(self):
        result = classify_failure_archetype(
            primary_failure_reason="weak_followthrough",
            survival_days=3,
            mfe_10d=0.01,
            mae_10d=-0.01,
            regime_tag="range",
            rsr_delta=None,
        )
        assert result == FailureArchetype.instant_rejection

    def test_weak_followthrough_survival_1(self):
        result = classify_failure_archetype(
            primary_failure_reason="weak_followthrough",
            survival_days=1,
            mfe_10d=0.0,
            mae_10d=-0.02,
            regime_tag="range",
            rsr_delta=None,
        )
        assert result == FailureArchetype.instant_rejection

    def test_weak_followthrough_not_instant_rejection_survival_gt3(self):
        result = classify_failure_archetype(
            primary_failure_reason="weak_followthrough",
            survival_days=5,
            mfe_10d=0.01,
            mae_10d=-0.01,
            regime_tag="range",
            rsr_delta=None,
        )
        assert result == FailureArchetype.unresolved

    def test_breakout_failed_3d_slow_bleed(self):
        result = classify_failure_archetype(
            primary_failure_reason="breakout_failed_3d",
            survival_days=2,
            mfe_10d=0.01,
            mae_10d=-0.02,
            regime_tag="range",
            rsr_delta=None,
        )
        assert result == FailureArchetype.slow_bleed

    def test_none_reason_unresolved(self):
        result = classify_failure_archetype(
            primary_failure_reason=None,
            survival_days=None,
            mfe_10d=None,
            mae_10d=None,
            regime_tag="range",
            rsr_delta=None,
        )
        assert result == FailureArchetype.unresolved

    def test_unknown_reason_unresolved(self):
        result = classify_failure_archetype(
            primary_failure_reason="some_future_reason",
            survival_days=3,
            mfe_10d=0.02,
            mae_10d=-0.02,
            regime_tag="range",
            rsr_delta=None,
        )
        assert result == FailureArchetype.unresolved

    def test_volume_faded_none_survival_unresolved(self):
        result = classify_failure_archetype(
            primary_failure_reason="volume_faded",
            survival_days=None,
            mfe_10d=0.01,
            mae_10d=-0.01,
            regime_tag="range",
            rsr_delta=None,
        )
        assert result == FailureArchetype.unresolved

    def test_market_selloff_takes_priority_over_other_conditions(self):
        # market_selloff with characteristics of volatility_trap — market_selloff wins
        result = classify_failure_archetype(
            primary_failure_reason="market_selloff",
            survival_days=3,
            mfe_10d=0.10,
            mae_10d=-0.10,
            regime_tag="risk_off",
            rsr_delta=-3.0,
        )
        assert result == FailureArchetype.regime_mismatch


# ─────────────────────────────────────────────────────────────────────────────
# 2. Transition telemetry correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestTransitionPair:
    def test_format(self):
        pair = build_transition_pair("market_selloff", FailureArchetype.regime_mismatch)
        assert pair == "market_selloff→regime_mismatch"

    def test_breakout_slow_bleed(self):
        pair = build_transition_pair("breakout_failed_3d", FailureArchetype.slow_bleed)
        assert pair == "breakout_failed_3d→slow_bleed"

    def test_none_reason_returns_none(self):
        pair = build_transition_pair(None, FailureArchetype.unresolved)
        assert pair is None

    def test_empty_reason_returns_none(self):
        pair = build_transition_pair("", FailureArchetype.unresolved)
        assert pair is None

    def test_separator_arrow(self):
        pair = build_transition_pair("rsr_reversal", FailureArchetype.institutional_exit)
        assert "→" in pair
        parts = pair.split("→")
        assert len(parts) == 2
        assert parts[0] == "rsr_reversal"
        assert parts[1] == "institutional_exit"


# ─────────────────────────────────────────────────────────────────────────────
# 3. CLEAN-only filtering
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanFilter:
    def test_clean_kept(self):
        recs = [{"data_quality_weight": 1.0, "symbol": "A"}, {"data_quality_weight": 1.5, "symbol": "B"}]
        out = _filter_clean(recs)
        assert len(out) == 2

    def test_partial_excluded(self):
        recs = [{"data_quality_weight": 0.5, "symbol": "A"}, {"data_quality_weight": 1.0, "symbol": "B"}]
        out = _filter_clean(recs)
        assert len(out) == 1
        assert out[0]["symbol"] == "B"

    def test_zero_weight_excluded(self):
        recs = [{"data_quality_weight": 0.0, "symbol": "MAJOR"}]
        out = _filter_clean(recs)
        assert len(out) == 0

    def test_missing_weight_treated_as_zero(self):
        recs = [{"symbol": "X"}]
        out = _filter_clean(recs)
        assert len(out) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 4. Append-only behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendOnly:
    def test_new_records_appended(self, tmp_dirs):
        arch_dir = tmp_dirs["archetype"]
        p = arch_dir / "future_leader_archetype_20260524.jsonl"

        rec = ArchetypeRecord(
            symbol="1111.T",
            observation_date="2026-01-10",
            primary_failure_reason="market_selloff",
            failure_archetype="regime_mismatch",
            transition_pair="market_selloff→regime_mismatch",
            survival_days=2,
            mfe_10d=0.02,
            mae_10d=-0.03,
            regime_tag="risk_off",
            integrity_state="CLEAN",
            data_quality_weight=1.0,
            observation_only=True,
        )
        _write_archetype_log([rec], arch_dir, today=TODAY)
        assert p.exists()
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

        # Second write appends
        rec2 = ArchetypeRecord(
            symbol="2222.T",
            observation_date="2026-01-11",
            primary_failure_reason="rsr_reversal",
            failure_archetype="institutional_exit",
            transition_pair="rsr_reversal→institutional_exit",
            survival_days=4,
            mfe_10d=0.03,
            mae_10d=-0.04,
            regime_tag="range",
            integrity_state="CLEAN",
            data_quality_weight=1.0,
            observation_only=True,
        )
        _write_archetype_log([rec2], arch_dir, today=TODAY)
        lines2 = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines2) == 2

    def test_empty_write_no_file_created(self, tmp_dirs):
        arch_dir = tmp_dirs["archetype"]
        _write_archetype_log([], arch_dir, today=TODAY)
        files = list(arch_dir.glob("*.jsonl"))
        assert len(files) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 5. Duplicate prevention
# ─────────────────────────────────────────────────────────────────────────────

class TestDuplicatePrevention:
    def test_already_classified_not_reprocessed(self, tmp_dirs):
        failure_dir   = tmp_dirs["failure"]
        regime_dir    = tmp_dirs["regime"]
        arch_dir      = tmp_dirs["archetype"]

        # Write a failure record
        fail_rec = _make_failure_rec(symbol="1234.T", obs_date="2026-01-10",
                                     primary_failure_reason="market_selloff")
        _write_failure_jsonl(failure_dir, [fail_rec])

        # Pre-populate archetype log with same key
        existing = {
            "symbol": "1234.T",
            "observation_date": "2026-01-10",
            "failure_archetype": "regime_mismatch",
            "data_quality_weight": 1.0,
            "observation_only": True,
        }
        arch_file = arch_dir / "future_leader_archetype_20260523.jsonl"
        arch_file.write_text(json.dumps(existing) + "\n", encoding="utf-8")

        count = materialize_archetypes(failure_dir, regime_dir, arch_dir, today=TODAY)
        assert count == 0  # already classified

    def test_new_symbol_date_processed(self, tmp_dirs):
        failure_dir   = tmp_dirs["failure"]
        regime_dir    = tmp_dirs["regime"]
        arch_dir      = tmp_dirs["archetype"]

        fail_recs = [
            _make_failure_rec(symbol="1111.T", obs_date="2026-01-10",
                              primary_failure_reason="market_selloff"),
            _make_failure_rec(symbol="2222.T", obs_date="2026-01-11",
                              primary_failure_reason="rsr_reversal"),
        ]
        _write_failure_jsonl(failure_dir, fail_recs)

        # Only 1111.T pre-classified
        existing = {"symbol": "1111.T", "observation_date": "2026-01-10",
                    "failure_archetype": "regime_mismatch", "data_quality_weight": 1.0,
                    "observation_only": True}
        (arch_dir / "future_leader_archetype_20260523.jsonl").write_text(
            json.dumps(existing) + "\n", encoding="utf-8"
        )

        count = materialize_archetypes(failure_dir, regime_dir, arch_dir, today=TODAY)
        assert count == 1  # only 2222.T new

    def test_dedup_duplicate_failure_records(self, tmp_dirs):
        failure_dir = tmp_dirs["failure"]
        regime_dir  = tmp_dirs["regime"]
        arch_dir    = tmp_dirs["archetype"]

        # Same key twice in failure JSONL
        rec = _make_failure_rec(symbol="1234.T", obs_date="2026-01-10",
                                primary_failure_reason="market_selloff")
        _write_failure_jsonl(failure_dir, [rec, rec])

        count = materialize_archetypes(failure_dir, regime_dir, arch_dir, today=TODAY)
        assert count == 1  # deduped to one


# ─────────────────────────────────────────────────────────────────────────────
# 6. No lookahead contamination
# ─────────────────────────────────────────────────────────────────────────────

class TestNoLookahead:
    def test_classify_is_pure_function(self):
        """classify_failure_archetype makes no external calls — pure function."""
        import inspect
        src = inspect.getsource(classify_failure_archetype)
        # Must not load OHLCV, broker data, or external state
        forbidden = ["pd.read", "open(", "requests", "yfinance", "load_ohlcv",
                     "datetime.now", "date.today"]
        for token in forbidden:
            assert token not in src, f"lookahead risk: '{token}' in classify_failure_archetype"

    def test_archetype_record_observation_only_always_true(self):
        """ArchetypeRecord.observation_only is always True — lookahead invariant."""
        rec = ArchetypeRecord(
            symbol="1234.T",
            observation_date="2026-01-10",
            primary_failure_reason="market_selloff",
            failure_archetype="regime_mismatch",
            transition_pair="market_selloff→regime_mismatch",
            survival_days=2,
            mfe_10d=0.01,
            mae_10d=-0.01,
            regime_tag="risk_off",
            integrity_state="CLEAN",
            data_quality_weight=1.0,
        )
        assert rec.observation_only is True

    def test_materialize_uses_only_confirmed_failure_records(self, tmp_dirs):
        """Archetype materialization reads from failure_log_dir (already lookahead-safe)."""
        failure_dir = tmp_dirs["failure"]
        regime_dir  = tmp_dirs["regime"]
        arch_dir    = tmp_dirs["archetype"]

        # Failure dir empty → nothing classified
        count = materialize_archetypes(failure_dir, regime_dir, arch_dir, today=TODAY)
        assert count == 0


# ─────────────────────────────────────────────────────────────────────────────
# 7. Observation-only invariant
# ─────────────────────────────────────────────────────────────────────────────

class TestObservationOnlyInvariant:
    def test_record_observation_only_default(self):
        rec = ArchetypeRecord(
            symbol="X",
            observation_date="2026-01-01",
            primary_failure_reason=None,
            failure_archetype="unresolved",
            transition_pair=None,
            survival_days=None,
            mfe_10d=None,
            mae_10d=None,
            regime_tag="range",
            integrity_state="CLEAN",
            data_quality_weight=1.0,
        )
        assert rec.observation_only is True

    def test_observation_only_in_jsonl_output(self, tmp_dirs):
        arch_dir = tmp_dirs["archetype"]
        rec = ArchetypeRecord(
            symbol="Y",
            observation_date="2026-01-02",
            primary_failure_reason="rsr_reversal",
            failure_archetype="institutional_exit",
            transition_pair="rsr_reversal→institutional_exit",
            survival_days=4,
            mfe_10d=0.03,
            mae_10d=-0.04,
            regime_tag="range",
            integrity_state="CLEAN",
            data_quality_weight=1.0,
        )
        _write_archetype_log([rec], arch_dir, today=TODAY)
        lines = [l for l in (arch_dir / "future_leader_archetype_20260524.jsonl")
                 .read_text(encoding="utf-8").splitlines() if l.strip()]
        data = json.loads(lines[0])
        assert data["observation_only"] is True

    def test_materialize_sets_observation_only_true(self, tmp_dirs):
        failure_dir = tmp_dirs["failure"]
        regime_dir  = tmp_dirs["regime"]
        arch_dir    = tmp_dirs["archetype"]

        _write_failure_jsonl(failure_dir, [
            _make_failure_rec(symbol="OBS.T", obs_date="2026-01-10",
                              primary_failure_reason="rsr_reversal")
        ])
        materialize_archetypes(failure_dir, regime_dir, arch_dir, today=TODAY)

        records = _load_jsonl(arch_dir, "future_leader_archetype_*.jsonl")
        assert len(records) == 1
        assert records[0]["observation_only"] is True


# ─────────────────────────────────────────────────────────────────────────────
# 8. No execution imports
# ─────────────────────────────────────────────────────────────────────────────

class TestNoExecutionImports:
    _FORBIDDEN_MODULES = [
        "allocat",
        "broker",
        "order",
        "signal_bridge",
        "deployable_universe",
    ]

    def _get_module_imports(self, module_name: str) -> list:
        import importlib
        mod = importlib.import_module(module_name)
        src = inspect.getsource(mod)
        imports = []
        for line in src.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped)
        return imports

    def test_no_forbidden_imports_in_archetype_module(self):
        imports = self._get_module_imports("src.analytics.future_leader_archetype")
        for imp in imports:
            for forbidden in self._FORBIDDEN_MODULES:
                assert forbidden not in imp.lower(), (
                    f"forbidden import '{forbidden}' found in archetype module: {imp}"
                )

    def test_module_importable_without_execution_deps(self):
        """Module must import cleanly without any execution-path side effects."""
        import src.analytics.future_leader_archetype as mod
        assert hasattr(mod, "classify_failure_archetype")
        assert hasattr(mod, "FailureArchetype")
        assert hasattr(mod, "run_future_leader_archetype_hook")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Full materialize pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestMaterializePipeline:
    def test_all_archetypes_classified(self, tmp_dirs):
        failure_dir = tmp_dirs["failure"]
        regime_dir  = tmp_dirs["regime"]
        arch_dir    = tmp_dirs["archetype"]

        fail_recs = [
            _make_failure_rec("A.T", "2026-01-01", "market_selloff",    failure_day_offset=2),
            _make_failure_rec("B.T", "2026-01-02", "rsr_reversal",      failure_day_offset=4),
            _make_failure_rec("C.T", "2026-01-03", "volume_faded",      failure_day_offset=6),
            _make_failure_rec("D.T", "2026-01-04", "gap_fill",
                              mfe_10d=0.08, mae_10d=-0.07, failure_day_offset=3),
            _make_failure_rec("E.T", "2026-01-05", "weak_followthrough", failure_day_offset=2),
            _make_failure_rec("F.T", "2026-01-06", "breakout_failed_3d", failure_day_offset=2),
        ]
        _write_failure_jsonl(failure_dir, fail_recs)

        count = materialize_archetypes(failure_dir, regime_dir, arch_dir, today=TODAY)
        assert count == 6

        written = _load_jsonl(arch_dir, "future_leader_archetype_*.jsonl")
        archetype_map = {r["symbol"]: r["failure_archetype"] for r in written}

        assert archetype_map["A.T"] == "regime_mismatch"
        assert archetype_map["B.T"] == "institutional_exit"
        assert archetype_map["C.T"] == "liquidity_fade"
        assert archetype_map["D.T"] == "volatility_trap"
        assert archetype_map["E.T"] == "instant_rejection"
        assert archetype_map["F.T"] == "slow_bleed"

    def test_regime_tag_used_from_regime_lookup(self, tmp_dirs):
        failure_dir = tmp_dirs["failure"]
        regime_dir  = tmp_dirs["regime"]
        arch_dir    = tmp_dirs["archetype"]

        _write_failure_jsonl(failure_dir, [
            _make_failure_rec("R.T", "2026-01-10", "market_selloff")
        ])
        _write_regime_jsonl(regime_dir, [
            {"observation_date": "2026-01-10", "regime_tag": "risk_off"}
        ])

        materialize_archetypes(failure_dir, regime_dir, arch_dir, today=TODAY)
        written = _load_jsonl(arch_dir, "future_leader_archetype_*.jsonl")
        assert written[0]["regime_tag"] == "risk_off"

    def test_regime_tag_fallback_unclassified(self, tmp_dirs):
        failure_dir = tmp_dirs["failure"]
        regime_dir  = tmp_dirs["regime"]
        arch_dir    = tmp_dirs["archetype"]

        _write_failure_jsonl(failure_dir, [
            _make_failure_rec("X.T", "2026-01-10", "rsr_reversal")
        ])
        # No regime JSONL written

        materialize_archetypes(failure_dir, regime_dir, arch_dir, today=TODAY)
        written = _load_jsonl(arch_dir, "future_leader_archetype_*.jsonl")
        assert written[0]["regime_tag"] == "unclassified"

    def test_partial_failure_persisted_but_count_included(self, tmp_dirs):
        failure_dir = tmp_dirs["failure"]
        regime_dir  = tmp_dirs["regime"]
        arch_dir    = tmp_dirs["archetype"]

        # PARTIAL: data_quality_weight = 0.5
        _write_failure_jsonl(failure_dir, [
            _make_failure_rec("P.T", "2026-01-10", "market_selloff",
                              data_quality_weight=0.5)
        ])
        count = materialize_archetypes(failure_dir, regime_dir, arch_dir, today=TODAY)
        assert count == 1  # persisted

        written = _load_jsonl(arch_dir, "future_leader_archetype_*.jsonl")
        assert len(written) == 1
        assert written[0]["data_quality_weight"] == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 10. Archetype summary — CLEAN-only aggregation
# ─────────────────────────────────────────────────────────────────────────────

class TestArchetypeSummary:
    def _populate_archetype_jsonl(self, arch_dir: Path, records: list) -> None:
        p = arch_dir / "future_leader_archetype_20260524.jsonl"
        lines = [json.dumps(r) for r in records]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_summary_count_clean_only(self, tmp_dirs):
        arch_dir = tmp_dirs["archetype"]
        recs = [
            {"failure_archetype": "regime_mismatch", "data_quality_weight": 1.0,
             "survival_days": 2, "mfe_10d": 0.01, "mae_10d": -0.02, "observation_only": True},
            {"failure_archetype": "regime_mismatch", "data_quality_weight": 0.5,  # PARTIAL
             "survival_days": 3, "mfe_10d": 0.02, "mae_10d": -0.03, "observation_only": True},
        ]
        self._populate_archetype_jsonl(arch_dir, recs)
        summaries = compute_archetype_summary(arch_dir)
        rm = next(s for s in summaries if s.archetype == "regime_mismatch")
        assert rm.count == 1  # partial excluded

    def test_summary_median_survival(self, tmp_dirs):
        arch_dir = tmp_dirs["archetype"]
        recs = [
            {"failure_archetype": "institutional_exit", "data_quality_weight": 1.0,
             "survival_days": 2, "mfe_10d": 0.01, "mae_10d": -0.01, "observation_only": True},
            {"failure_archetype": "institutional_exit", "data_quality_weight": 1.0,
             "survival_days": 4, "mfe_10d": 0.02, "mae_10d": -0.02, "observation_only": True},
            {"failure_archetype": "institutional_exit", "data_quality_weight": 1.0,
             "survival_days": 6, "mfe_10d": 0.03, "mae_10d": -0.03, "observation_only": True},
        ]
        self._populate_archetype_jsonl(arch_dir, recs)
        summaries = compute_archetype_summary(arch_dir)
        ie = next(s for s in summaries if s.archetype == "institutional_exit")
        assert ie.count == 3
        assert ie.median_survival == pytest.approx(4.0)

    def test_summary_none_when_insufficient_sample(self, tmp_dirs):
        arch_dir = tmp_dirs["archetype"]
        # Only 2 records — below _MIN_SAMPLE=3
        recs = [
            {"failure_archetype": "liquidity_fade", "data_quality_weight": 1.0,
             "survival_days": 5, "mfe_10d": 0.01, "mae_10d": -0.01, "observation_only": True},
            {"failure_archetype": "liquidity_fade", "data_quality_weight": 1.0,
             "survival_days": 7, "mfe_10d": 0.02, "mae_10d": -0.02, "observation_only": True},
        ]
        self._populate_archetype_jsonl(arch_dir, recs)
        summaries = compute_archetype_summary(arch_dir)
        lf = next(s for s in summaries if s.archetype == "liquidity_fade")
        assert lf.median_survival is None

    def test_empty_dir_returns_zero_counts(self, tmp_dirs):
        arch_dir = tmp_dirs["archetype"]
        summaries = compute_archetype_summary(arch_dir)
        assert all(s.count == 0 for s in summaries)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Report section format
# ─────────────────────────────────────────────────────────────────────────────

class TestReportFormat:
    def test_section_header(self):
        summaries = [ArchetypeSummary("unresolved", 0, None, None, None)]
        out = format_archetype_section(summaries)
        assert "=== FAILURE ARCHETYPES ===" in out

    def test_no_data_message(self):
        summaries = [s for a in FailureArchetype
                     for s in [ArchetypeSummary(a.value, 0, None, None, None)]]
        out = format_archetype_section(summaries)
        assert "(no archetype records)" in out

    def test_count_shown(self):
        summaries = [
            ArchetypeSummary("regime_mismatch", 5, 2.0, 0.01, -0.02),
            ArchetypeSummary("unresolved", 0, None, None, None),
        ]
        out = format_archetype_section(summaries)
        assert "regime_mismatch:" in out
        assert "count: 5" in out

    def test_metrics_shown_when_available(self):
        summaries = [
            ArchetypeSummary("volatility_trap", 4, 3.0, 0.072, -0.081),
        ]
        out = format_archetype_section(summaries)
        assert "median survival" in out
        assert "median MFE" in out
        assert "median MAE" in out

    def test_zero_count_archetypes_omitted(self):
        summaries = [
            ArchetypeSummary("regime_mismatch", 3, 2.0, None, None),
            ArchetypeSummary("slow_bleed",      0, None, None, None),
        ]
        out = format_archetype_section(summaries)
        assert "slow_bleed" not in out


# ─────────────────────────────────────────────────────────────────────────────
# 12. Hook integration
# ─────────────────────────────────────────────────────────────────────────────

class TestHookIntegration:
    def test_hook_creates_report_section(self, tmp_dirs):
        failure_dir   = tmp_dirs["failure"]
        regime_dir    = tmp_dirs["regime"]
        arch_dir      = tmp_dirs["archetype"]
        reports_dir   = tmp_dirs["reports"]

        _write_failure_jsonl(failure_dir, [
            _make_failure_rec("H.T", "2026-01-10", "market_selloff")
        ])

        run_future_leader_archetype_hook(
            failure_log_dir=failure_dir,
            regime_log_dir=regime_dir,
            archetype_log_dir=arch_dir,
            reports_dir=reports_dir,
            today=TODAY,
        )

        report_path = reports_dir / f"future_leader_report_{TODAY.strftime('%Y%m%d')}.md"
        assert report_path.exists()
        content = report_path.read_text(encoding="utf-8")
        assert "FAILURE ARCHETYPES" in content

    def test_hook_fail_open_on_bad_dirs(self, tmp_dirs):
        """Hook must not raise — fail-open."""
        nonexistent = tmp_paths = tmp_dirs["tmp"] / "nonexistent"
        run_future_leader_archetype_hook(
            failure_log_dir=nonexistent / "fail",
            regime_log_dir=nonexistent / "regime",
            archetype_log_dir=nonexistent / "arch",
            reports_dir=nonexistent / "reports",
            today=TODAY,
        )  # must not raise

    def test_hook_idempotent(self, tmp_dirs):
        failure_dir   = tmp_dirs["failure"]
        regime_dir    = tmp_dirs["regime"]
        arch_dir      = tmp_dirs["archetype"]
        reports_dir   = tmp_dirs["reports"]

        _write_failure_jsonl(failure_dir, [
            _make_failure_rec("I.T", "2026-01-10", "rsr_reversal")
        ])

        # Run twice
        run_future_leader_archetype_hook(failure_dir, regime_dir, arch_dir, reports_dir, today=TODAY)
        run_future_leader_archetype_hook(failure_dir, regime_dir, arch_dir, reports_dir, today=TODAY)

        records = _load_jsonl(arch_dir, "future_leader_archetype_*.jsonl")
        assert len(records) == 1  # not duplicated
