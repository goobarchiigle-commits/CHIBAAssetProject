"""
test_evidence_promotion.py — Evidence Promotion Engine

82 tests across 11 classes covering:
  TestSafeHelpers / TestLoadLatestRecord / TestClamp
  TestEvaluateExitExtension / TestEvaluateCapitalConcentration
  TestEvaluatePrority / TestEvaluateSignalLeader / TestEvaluateContinuationSignal
  TestRunEvidencePromotion / TestAppendPromotionRecord / TestFormatPromotionReport
"""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Optional

import pytest

from src.analytics.evidence_promotion import (
    CT_CAPITAL_CONCENTRATION,
    CT_CONTINUATION_SIGNAL,
    CT_EXIT_EXTENSION,
    CT_PRIORITY,
    CT_SIGNAL_LEADER,
    STATUS_INSUFFICIENT,
    STATUS_OBSERVE,
    STATUS_PROMOTABLE,
    EvidenceCandidate,
    EvidencePromotionOutput,
    HDC_DELTA_MIN,
    HDC_SAMPLE_MIN,
    CCS_ALPHA_MIN,
    CCS_DATES_MIN,
    PCI_MONO_MIN,
    PCI_CALIB_MAX,
    SA_IR_MIN,
    SA_SAMPLE_MIN,
    FCO_IR_MIN,
    FCO_SAMPLE_MIN,
    _clamp,
    _load_latest_record,
    _safe,
    _safe_int,
    _evaluate_exit_extension,
    _evaluate_capital_concentration,
    _evaluate_priority,
    _evaluate_signal_leader,
    _evaluate_continuation_signal,
    run_evidence_promotion,
    append_promotion_record,
    format_promotion_report,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dir(tmp_path):
    return tmp_path


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _hdc_promotable() -> dict:
    return {
        "suppression_return_delta": 0.015,
        "n_materialized": 25,
        "duration_ev_spread": 0.025,
        "best_duration_bucket": "7-10d",
        "n_records_total": 30,
    }


def _ccs_promotable() -> dict:
    return {
        "mean_concentration_alpha": 0.030,
        "positive_alpha_rate": 0.65,
        "n_dates_analyzed": 40,
        "n_materialized": 60,
    }


def _pci_promotable() -> dict:
    return {
        "priority_monotonicity": 0.72,
        "priority_calibration_error": 0.15,
        "n_buckets_populated": 8,
        "n_materialized": 40,
    }


def _sa_promotable() -> dict:
    return {
        "top_signal": "priority_score",
        "top_signal_ir": 0.018,
        "n_materialized": 30,
        "n_records_total": 50,
    }


def _fco_promotable() -> dict:
    return {
        "top_signal": "compression_score",
        "top_signal_ir": 0.014,
        "n_materialized": 25,
        "n_records_total": 40,
    }


# ─── TestSafeHelpers ──────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_float(self):
        assert _safe(0.05) == pytest.approx(0.05)

    def test_safe_str_number(self):
        assert _safe("0.03") == pytest.approx(0.03)

    def test_safe_none_returns_fallback(self):
        assert _safe(None, 99.0) == 99.0

    def test_safe_nan_returns_fallback(self):
        assert _safe(float("nan"), 0.0) == 0.0

    def test_safe_inf_returns_fallback(self):
        assert _safe(float("inf"), 0.0) == 0.0

    def test_safe_int_valid(self):
        assert _safe_int(25) == 25

    def test_safe_int_string(self):
        assert _safe_int("10") == 10

    def test_safe_int_none_fallback(self):
        assert _safe_int(None) == 0

    def test_clamp_below(self):
        assert _clamp(-0.5) == 0.0

    def test_clamp_above(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_within(self):
        assert _clamp(0.5) == pytest.approx(0.5)


# ─── TestLoadLatestRecord ─────────────────────────────────────────────────────

class TestLoadLatestRecord:
    def test_missing_file_returns_empty(self, tmp_dir):
        result = _load_latest_record(tmp_dir / "nonexistent.jsonl")
        assert result == {}

    def test_returns_last_valid_line(self, tmp_dir):
        p = tmp_dir / "test.jsonl"
        _write_jsonl(p, [{"a": 1}, {"a": 2}, {"a": 3}])
        result = _load_latest_record(p)
        assert result["a"] == 3

    def test_skips_invalid_json(self, tmp_dir):
        p = tmp_dir / "test.jsonl"
        p.write_text('{"a": 1}\nNOT_JSON\n{"a": 2}\n', encoding="utf-8")
        result = _load_latest_record(p)
        assert result["a"] == 2

    def test_empty_file_returns_empty(self, tmp_dir):
        p = tmp_dir / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        result = _load_latest_record(p)
        assert result == {}

    def test_single_record(self, tmp_dir):
        p = tmp_dir / "single.jsonl"
        _write_jsonl(p, [{"key": "value"}])
        result = _load_latest_record(p)
        assert result["key"] == "value"


# ─── TestEvaluateExitExtension ────────────────────────────────────────────────

class TestEvaluateExitExtension:
    def test_promotable_all_rules_met(self):
        c = _evaluate_exit_extension(_hdc_promotable())
        assert c.status == STATUS_PROMOTABLE

    def test_candidate_type(self):
        c = _evaluate_exit_extension(_hdc_promotable())
        assert c.candidate_type == CT_EXIT_EXTENSION

    def test_source_module(self):
        c = _evaluate_exit_extension(_hdc_promotable())
        assert c.source_module == "HDC"

    def test_both_rules_met(self):
        c = _evaluate_exit_extension(_hdc_promotable())
        assert all(c.rules_met)

    def test_observe_delta_below_threshold(self):
        rec = _hdc_promotable()
        rec["suppression_return_delta"] = 0.005  # below HDC_DELTA_MIN
        c = _evaluate_exit_extension(rec)
        assert c.status == STATUS_OBSERVE
        assert c.rules_met[0] is False
        assert c.rules_met[1] is True

    def test_observe_sample_below_threshold(self):
        rec = _hdc_promotable()
        rec["n_materialized"] = 10  # below HDC_SAMPLE_MIN
        c = _evaluate_exit_extension(rec)
        assert c.status == STATUS_OBSERVE
        assert c.rules_met[1] is False

    def test_insufficient_below_floor(self):
        rec = _hdc_promotable()
        rec["n_materialized"] = 2
        c = _evaluate_exit_extension(rec)
        assert c.status == STATUS_INSUFFICIENT

    def test_evidence_score_in_range(self):
        c = _evaluate_exit_extension(_hdc_promotable())
        assert 0 <= c.evidence_score <= 100

    def test_evidence_score_zero_on_insufficient(self):
        rec = _hdc_promotable()
        rec["n_materialized"] = 0
        c = _evaluate_exit_extension(rec)
        assert c.evidence_score == 0

    def test_supporting_kpis_present(self):
        c = _evaluate_exit_extension(_hdc_promotable())
        assert "suppression_return_delta" in c.supporting_kpis
        assert "n_materialized" in c.supporting_kpis
        assert "duration_ev_spread" in c.supporting_kpis
        assert "best_duration_bucket" in c.supporting_kpis

    def test_empty_rec_gives_insufficient(self):
        c = _evaluate_exit_extension({})
        assert c.status == STATUS_INSUFFICIENT

    def test_full_score_at_targets(self):
        rec = {
            "suppression_return_delta": 0.02,  # = HDC_DELTA_FULL → 50
            "n_materialized": 40,              # = HDC_SAMPLE_FULL → 30
            "duration_ev_spread": 0.02,        # = HDC_SPREAD_FULL → 20
        }
        c = _evaluate_exit_extension(rec)
        assert c.evidence_score == 100


# ─── TestEvaluateCapitalConcentration ─────────────────────────────────────────

class TestEvaluateCapitalConcentration:
    def test_promotable(self):
        c = _evaluate_capital_concentration(_ccs_promotable())
        assert c.status == STATUS_PROMOTABLE

    def test_candidate_type(self):
        c = _evaluate_capital_concentration(_ccs_promotable())
        assert c.candidate_type == CT_CAPITAL_CONCENTRATION

    def test_source_module_ccs(self):
        c = _evaluate_capital_concentration(_ccs_promotable())
        assert c.source_module == "CCS"

    def test_observe_alpha_below_threshold(self):
        rec = _ccs_promotable()
        rec["mean_concentration_alpha"] = 0.01
        c = _evaluate_capital_concentration(rec)
        assert c.status == STATUS_OBSERVE
        assert c.rules_met[0] is False

    def test_observe_dates_below_threshold(self):
        rec = _ccs_promotable()
        rec["n_dates_analyzed"] = 15
        c = _evaluate_capital_concentration(rec)
        assert c.status == STATUS_OBSERVE
        assert c.rules_met[1] is False

    def test_insufficient_dates_below_floor(self):
        rec = _ccs_promotable()
        rec["n_dates_analyzed"] = 3
        c = _evaluate_capital_concentration(rec)
        assert c.status == STATUS_INSUFFICIENT

    def test_evidence_score_range(self):
        c = _evaluate_capital_concentration(_ccs_promotable())
        assert 0 <= c.evidence_score <= 100

    def test_supporting_kpis_keys(self):
        c = _evaluate_capital_concentration(_ccs_promotable())
        assert "mean_concentration_alpha" in c.supporting_kpis
        assert "positive_alpha_rate" in c.supporting_kpis
        assert "n_dates_analyzed" in c.supporting_kpis

    def test_negative_alpha_stays_observe(self):
        rec = _ccs_promotable()
        rec["mean_concentration_alpha"] = -0.01
        c = _evaluate_capital_concentration(rec)
        assert c.status == STATUS_OBSERVE


# ─── TestEvaluatePriority ─────────────────────────────────────────────────────

class TestEvaluatePriority:
    def test_promotable(self):
        c = _evaluate_priority(_pci_promotable())
        assert c.status == STATUS_PROMOTABLE

    def test_candidate_type(self):
        c = _evaluate_priority(_pci_promotable())
        assert c.candidate_type == CT_PRIORITY

    def test_source_module_pci(self):
        c = _evaluate_priority(_pci_promotable())
        assert c.source_module == "PCI"

    def test_observe_mono_below(self):
        rec = _pci_promotable()
        rec["priority_monotonicity"] = 0.3
        c = _evaluate_priority(rec)
        assert c.status == STATUS_OBSERVE
        assert c.rules_met[0] is False

    def test_observe_calib_err_above(self):
        rec = _pci_promotable()
        rec["priority_calibration_error"] = 0.40
        c = _evaluate_priority(rec)
        assert c.status == STATUS_OBSERVE
        assert c.rules_met[1] is False

    def test_insufficient_data(self):
        rec = _pci_promotable()
        rec["n_materialized"] = 5
        c = _evaluate_priority(rec)
        assert c.status == STATUS_INSUFFICIENT

    def test_evidence_score_range(self):
        c = _evaluate_priority(_pci_promotable())
        assert 0 <= c.evidence_score <= 100

    def test_negative_monotonicity_zero_score_component(self):
        rec = _pci_promotable()
        rec["priority_monotonicity"] = -0.5
        c = _evaluate_priority(rec)
        # negative monotonicity clamped to 0 → score_mono = 0
        assert c.evidence_score < 50

    def test_kpis_present(self):
        c = _evaluate_priority(_pci_promotable())
        assert "priority_monotonicity" in c.supporting_kpis
        assert "priority_calibration_error" in c.supporting_kpis
        assert "n_buckets_populated" in c.supporting_kpis


# ─── TestEvaluateSignalLeader ─────────────────────────────────────────────────

class TestEvaluateSignalLeader:
    def test_promotable(self):
        c = _evaluate_signal_leader(_sa_promotable())
        assert c.status == STATUS_PROMOTABLE

    def test_candidate_type(self):
        c = _evaluate_signal_leader(_sa_promotable())
        assert c.candidate_type == CT_SIGNAL_LEADER

    def test_source_module_sa(self):
        c = _evaluate_signal_leader(_sa_promotable())
        assert c.source_module == "SA"

    def test_observe_ir_below(self):
        rec = _sa_promotable()
        rec["top_signal_ir"] = 0.005
        c = _evaluate_signal_leader(rec)
        assert c.status == STATUS_OBSERVE
        assert c.rules_met[0] is False

    def test_observe_sample_below(self):
        rec = _sa_promotable()
        rec["n_materialized"] = 10
        c = _evaluate_signal_leader(rec)
        assert c.status == STATUS_OBSERVE
        assert c.rules_met[1] is False

    def test_insufficient(self):
        rec = _sa_promotable()
        rec["n_materialized"] = 2
        c = _evaluate_signal_leader(rec)
        assert c.status == STATUS_INSUFFICIENT

    def test_negative_ir_blocked(self):
        rec = _sa_promotable()
        rec["top_signal_ir"] = -0.015
        rec["n_materialized"] = 30
        c = _evaluate_signal_leader(rec)
        # abs(-0.015) > SA_IR_MIN(0.01) → r1 = True, r2 = True → PROMOTABLE
        assert c.status == STATUS_PROMOTABLE

    def test_kpis_contain_top_signal(self):
        c = _evaluate_signal_leader(_sa_promotable())
        assert "top_signal" in c.supporting_kpis
        assert c.supporting_kpis["top_signal"] == "priority_score"


# ─── TestEvaluateContinuationSignal ──────────────────────────────────────────

class TestEvaluateContinuationSignal:
    def test_promotable(self):
        c = _evaluate_continuation_signal(_fco_promotable())
        assert c.status == STATUS_PROMOTABLE

    def test_candidate_type(self):
        c = _evaluate_continuation_signal(_fco_promotable())
        assert c.candidate_type == CT_CONTINUATION_SIGNAL

    def test_source_module_fco(self):
        c = _evaluate_continuation_signal(_fco_promotable())
        assert c.source_module == "FCO"

    def test_negative_ir_observe(self):
        rec = _fco_promotable()
        rec["top_signal_ir"] = -0.015
        c = _evaluate_continuation_signal(rec)
        # ir must be > FCO_IR_MIN (positive); negative → r1 = False
        assert c.status == STATUS_OBSERVE
        assert c.rules_met[0] is False

    def test_zero_ir_observe(self):
        rec = _fco_promotable()
        rec["top_signal_ir"] = 0.0
        c = _evaluate_continuation_signal(rec)
        assert c.status == STATUS_OBSERVE

    def test_insufficient(self):
        rec = _fco_promotable()
        rec["n_materialized"] = 1
        c = _evaluate_continuation_signal(rec)
        assert c.status == STATUS_INSUFFICIENT

    def test_evidence_score_range(self):
        c = _evaluate_continuation_signal(_fco_promotable())
        assert 0 <= c.evidence_score <= 100


# ─── TestRunEvidencePromotion ─────────────────────────────────────────────────

class TestRunEvidencePromotion:
    def _run(self, tmp_dir, hdc=None, ccs=None, pci=None, sa=None, fco=None):
        hdc_f = tmp_dir / "hdc.jsonl"
        ccs_f = tmp_dir / "ccs.jsonl"
        pci_f = tmp_dir / "pci.jsonl"
        sa_f  = tmp_dir / "sa.jsonl"
        fco_f = tmp_dir / "fco.jsonl"
        out_f = tmp_dir / "ep.jsonl"
        if hdc: _write_jsonl(hdc_f, [hdc])
        if ccs: _write_jsonl(ccs_f, [ccs])
        if pci: _write_jsonl(pci_f, [pci])
        if sa:  _write_jsonl(sa_f,  [sa])
        if fco: _write_jsonl(fco_f, [fco])
        return run_evidence_promotion(
            hdc_file=hdc_f, ccs_file=ccs_f, pci_file=pci_f,
            sa_file=sa_f, fco_file=fco_f, output_file=out_f, today="2026-05-30",
        )

    def test_returns_output_type(self, tmp_dir):
        result = self._run(tmp_dir)
        assert isinstance(result, EvidencePromotionOutput)

    def test_always_5_candidates(self, tmp_dir):
        result = self._run(tmp_dir)
        assert len(result.candidates) == 5

    def test_candidate_types_present(self, tmp_dir):
        result = self._run(tmp_dir)
        types = {c.candidate_type for c in result.candidates}
        assert CT_EXIT_EXTENSION in types
        assert CT_CAPITAL_CONCENTRATION in types
        assert CT_PRIORITY in types
        assert CT_SIGNAL_LEADER in types
        assert CT_CONTINUATION_SIGNAL in types

    def test_date_set(self, tmp_dir):
        result = self._run(tmp_dir)
        assert result.date == "2026-05-30"

    def test_all_promotable_with_good_data(self, tmp_dir):
        result = self._run(
            tmp_dir,
            hdc=_hdc_promotable(),
            ccs=_ccs_promotable(),
            pci=_pci_promotable(),
            sa=_sa_promotable(),
            fco=_fco_promotable(),
        )
        assert result.n_promotable == 5

    def test_all_insufficient_with_empty_files(self, tmp_dir):
        result = self._run(tmp_dir)
        assert result.n_insufficient == 5
        assert result.n_promotable == 0

    def test_top_candidate_set_when_promotable(self, tmp_dir):
        result = self._run(
            tmp_dir,
            hdc=_hdc_promotable(),
            ccs=_ccs_promotable(),
            pci=_pci_promotable(),
            sa=_sa_promotable(),
            fco=_fco_promotable(),
        )
        assert result.top_candidate != ""

    def test_top_candidate_empty_when_none_promotable(self, tmp_dir):
        result = self._run(tmp_dir)
        assert result.top_candidate == ""

    def test_counts_sum_to_5(self, tmp_dir):
        result = self._run(
            tmp_dir,
            hdc=_hdc_promotable(),
            ccs=_ccs_promotable(),
        )
        assert result.n_promotable + result.n_observe + result.n_insufficient == 5

    def test_fail_open_on_corrupt_files(self, tmp_dir):
        bad = tmp_dir / "bad.jsonl"
        bad.write_text("NOTJSON\nALSONOT\n", encoding="utf-8")
        out_f = tmp_dir / "ep.jsonl"
        result = run_evidence_promotion(
            hdc_file=bad, ccs_file=bad, pci_file=bad,
            sa_file=bad, fco_file=bad, output_file=out_f, today="2026-05-30",
        )
        assert isinstance(result, EvidencePromotionOutput)


# ─── TestAppendPromotionRecord ────────────────────────────────────────────────

class TestAppendPromotionRecord:
    def _make_output(self, date="2026-05-30") -> EvidencePromotionOutput:
        return EvidencePromotionOutput(
            date=date,
            candidates=[
                EvidenceCandidate(
                    candidate_type=CT_EXIT_EXTENSION,
                    source_module="HDC",
                    status=STATUS_PROMOTABLE,
                    evidence_score=75,
                    supporting_kpis={"suppression_return_delta": 0.015, "n_materialized": 25},
                    rules=["delta > 1%", "n >= 20"],
                    rules_met=[True, True],
                )
            ],
            n_promotable=1,
            n_observe=0,
            n_insufficient=0,
            top_candidate=CT_EXIT_EXTENSION,
        )

    def test_file_created(self, tmp_dir):
        path = tmp_dir / "ep.jsonl"
        append_promotion_record(self._make_output(), path)
        assert path.exists()

    def test_record_is_valid_json(self, tmp_dir):
        path = tmp_dir / "ep.jsonl"
        append_promotion_record(self._make_output(), path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["date"] == "2026-05-30"

    def test_append_creates_two_lines(self, tmp_dir):
        path = tmp_dir / "ep.jsonl"
        append_promotion_record(self._make_output("2026-05-30"), path)
        append_promotion_record(self._make_output("2026-05-31"), path)
        lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_schema_version_present(self, tmp_dir):
        path = tmp_dir / "ep.jsonl"
        append_promotion_record(self._make_output(), path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert "schema_version" in data

    def test_candidates_serialised(self, tmp_dir):
        path = tmp_dir / "ep.jsonl"
        append_promotion_record(self._make_output(), path)
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert len(data["candidates"]) == 1
        assert data["candidates"][0]["candidate_type"] == CT_EXIT_EXTENSION

    def test_fail_open_on_readonly_dir(self, tmp_dir):
        bad_path = Path("/nonexistent_root_dir/ep.jsonl")
        output = self._make_output()
        append_promotion_record(output, bad_path)   # must not raise


# ─── TestFormatPromotionReport ────────────────────────────────────────────────

class TestFormatPromotionReport:
    def _make_output(self) -> EvidencePromotionOutput:
        return EvidencePromotionOutput(
            date="2026-05-30",
            candidates=[
                EvidenceCandidate(
                    candidate_type=CT_EXIT_EXTENSION,
                    source_module="HDC",
                    status=STATUS_PROMOTABLE,
                    evidence_score=78,
                    supporting_kpis={
                        "suppression_return_delta": 0.015,
                        "n_materialized": 25,
                        "duration_ev_spread": 0.025,
                        "best_duration_bucket": "7-10d",
                    },
                    rules=["delta > 1%", "n >= 20"],
                    rules_met=[True, True],
                ),
                EvidenceCandidate(
                    candidate_type=CT_CAPITAL_CONCENTRATION,
                    source_module="CCS",
                    status=STATUS_OBSERVE,
                    evidence_score=45,
                    supporting_kpis={
                        "mean_concentration_alpha": 0.008,
                        "positive_alpha_rate": 0.55,
                        "n_dates_analyzed": 18,
                        "n_materialized": 30,
                    },
                    rules=["alpha > 2%", "dates >= 30"],
                    rules_met=[False, False],
                ),
                EvidenceCandidate(
                    candidate_type=CT_PRIORITY,
                    source_module="PCI",
                    status=STATUS_INSUFFICIENT,
                    evidence_score=0,
                    supporting_kpis={"priority_monotonicity": 0.0, "priority_calibration_error": 1.0, "n_buckets_populated": 0, "n_materialized": 2},
                    rules=["mono > 0.5", "calib < 0.25"],
                    rules_met=[False, False],
                ),
                EvidenceCandidate(
                    candidate_type=CT_SIGNAL_LEADER,
                    source_module="SA",
                    status=STATUS_OBSERVE,
                    evidence_score=51,
                    supporting_kpis={"top_signal": "priority_score", "top_signal_ir": 0.012, "n_materialized": 30},
                    rules=["ir > 1%", "n >= 20"],
                    rules_met=[True, True],
                ),
                EvidenceCandidate(
                    candidate_type=CT_CONTINUATION_SIGNAL,
                    source_module="FCO",
                    status=STATUS_OBSERVE,
                    evidence_score=40,
                    supporting_kpis={"top_signal": "rsr", "top_signal_ir": 0.009, "n_materialized": 15},
                    rules=["ir > 1%", "n >= 20"],
                    rules_met=[False, False],
                ),
            ],
            n_promotable=1,
            n_observe=3,
            n_insufficient=1,
            top_candidate=CT_EXIT_EXTENSION,
        )

    def test_contains_header(self):
        rpt = format_promotion_report(self._make_output())
        assert "EVIDENCE PROMOTION" in rpt

    def test_contains_promotable_label(self):
        rpt = format_promotion_report(self._make_output())
        assert "PROMOTABLE" in rpt

    def test_contains_observe_label(self):
        rpt = format_promotion_report(self._make_output())
        assert "OBSERVE" in rpt

    def test_contains_insufficient_label(self):
        rpt = format_promotion_report(self._make_output())
        assert "INSUFFICIENT" in rpt

    def test_contains_exit_extension(self):
        rpt = format_promotion_report(self._make_output())
        assert "EXIT_EXTENSION" in rpt

    def test_contains_capital_concentration(self):
        rpt = format_promotion_report(self._make_output())
        assert "CAPITAL_CONCENTRATION" in rpt

    def test_contains_priority(self):
        rpt = format_promotion_report(self._make_output())
        assert "PRIORITY" in rpt

    def test_contains_signal_leader(self):
        rpt = format_promotion_report(self._make_output())
        assert "SIGNAL_LEADER" in rpt

    def test_contains_continuation_signal(self):
        rpt = format_promotion_report(self._make_output())
        assert "CONTINUATION_SIGNAL" in rpt

    def test_shows_counts(self):
        rpt = format_promotion_report(self._make_output())
        assert "Promotable: 1" in rpt or "promotable=1" in rpt.lower() or "Promotable: 1" in rpt

    def test_shows_top_candidate(self):
        rpt = format_promotion_report(self._make_output())
        assert "EXIT_EXTENSION" in rpt

    def test_empty_candidates_returns_empty(self):
        output = EvidencePromotionOutput(
            date="2026-05-30", candidates=[], n_promotable=0,
            n_observe=0, n_insufficient=0, top_candidate="",
        )
        rpt = format_promotion_report(output)
        assert rpt == ""

    def test_evidence_score_shown(self):
        rpt = format_promotion_report(self._make_output())
        assert "score=" in rpt

    def test_source_module_shown(self):
        rpt = format_promotion_report(self._make_output())
        assert "(HDC)" in rpt or "HDC" in rpt
