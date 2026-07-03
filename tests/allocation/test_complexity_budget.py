"""Tests for src/allocation/complexity_budget.py"""
import pytest
from src.allocation.complexity_budget import (
    AllocatorFactor,
    ComplexityBudget,
    ComplexityAuditRecord,
    can_admit_factor,
    admit_factor,
    deactivate_factor,
    append_audit_record,
    load_complexity_budget,
    save_complexity_budget,
    load_audit_records,
    MAX_FACTORS,
    MIN_MARGINAL_CAGR_BPS,
)


def _valid_factor(factor_id="f_corr", marginal_cagr_bps=20.0):
    return AllocatorFactor(
        factor_id=factor_id,
        description="Test factor",
        admitted_date="2026-05-16",
        out_of_sample_confirmed=True,
        post_cost_confirmed=True,
        marginal_cagr_bps=marginal_cagr_bps,
        replay_deterministic=True,
        explainability_preserved=True,
        active=True,
    )


def _budget_with_n_active(n):
    factors = [
        {
            "factor_id": f"f{i}",
            "description": "x",
            "admitted_date": "2026-05-16",
            "out_of_sample_confirmed": True,
            "post_cost_confirmed": True,
            "marginal_cagr_bps": 20.0,
            "replay_deterministic": True,
            "explainability_preserved": True,
            "active": True,
        }
        for i in range(n)
    ]
    return ComplexityBudget(active_factor_count=n, factors=factors)


# ── can_admit_factor ───────────────────────────────────────────────────────────

def test_can_admit_valid():
    budget = ComplexityBudget()
    factor = _valid_factor()
    ok, reason = can_admit_factor(budget, factor)
    assert ok is True
    assert "criteria met" in reason


def test_reject_oos_not_confirmed():
    budget = ComplexityBudget()
    f = _valid_factor()
    f.out_of_sample_confirmed = False
    ok, reason = can_admit_factor(budget, f)
    assert ok is False
    assert "out-of-sample" in reason


def test_reject_post_cost_not_confirmed():
    budget = ComplexityBudget()
    f = _valid_factor()
    f.post_cost_confirmed = False
    ok, reason = can_admit_factor(budget, f)
    assert ok is False
    assert "post-cost" in reason


def test_reject_insufficient_marginal_cagr():
    budget = ComplexityBudget()
    f = _valid_factor(marginal_cagr_bps=MIN_MARGINAL_CAGR_BPS - 1.0)
    ok, reason = can_admit_factor(budget, f)
    assert ok is False
    assert "marginal CAGR" in reason


def test_reject_not_replay_deterministic():
    budget = ComplexityBudget()
    f = _valid_factor()
    f.replay_deterministic = False
    ok, reason = can_admit_factor(budget, f)
    assert ok is False
    assert "replay" in reason


def test_reject_explainability_not_preserved():
    budget = ComplexityBudget()
    f = _valid_factor()
    f.explainability_preserved = False
    ok, reason = can_admit_factor(budget, f)
    assert ok is False
    assert "explainability" in reason


def test_reject_at_capacity():
    budget = _budget_with_n_active(MAX_FACTORS)
    f = _valid_factor()
    ok, reason = can_admit_factor(budget, f)
    assert ok is False
    assert "cap" in reason


def test_admit_just_below_capacity():
    budget = _budget_with_n_active(MAX_FACTORS - 1)
    f = _valid_factor(factor_id="new_factor")
    ok, _ = can_admit_factor(budget, f)
    assert ok is True


# ── admit_factor ───────────────────────────────────────────────────────────────

def test_admit_factor_updates_budget(tmp_path):
    p = tmp_path / "audit.jsonl"
    budget = ComplexityBudget()
    f = _valid_factor()
    new_budget, ok, reason = admit_factor(budget, f, "T", p)
    assert ok is True
    assert new_budget.active_factor_count == 1
    assert len(new_budget.factors) == 1


def test_admit_factor_appends_audit(tmp_path):
    p = tmp_path / "audit.jsonl"
    budget = ComplexityBudget()
    f = _valid_factor()
    admit_factor(budget, f, "T", p)
    records = load_audit_records(p)
    assert len(records) == 1
    assert records[0].action == "ADMITTED"


def test_reject_factor_appends_audit(tmp_path):
    p = tmp_path / "audit.jsonl"
    budget = ComplexityBudget()
    f = _valid_factor()
    f.out_of_sample_confirmed = False
    _, ok, _ = admit_factor(budget, f, "T", p)
    assert ok is False
    records = load_audit_records(p)
    assert records[0].action == "REJECTED"


def test_admit_factor_does_not_modify_original(tmp_path):
    p = tmp_path / "audit.jsonl"
    budget = ComplexityBudget()
    f = _valid_factor()
    new_budget, _, _ = admit_factor(budget, f, "T", p)
    assert budget.active_factor_count == 0  # original unchanged
    assert new_budget.active_factor_count == 1


# ── deactivate_factor ──────────────────────────────────────────────────────────

def test_deactivate_reduces_count(tmp_path):
    p = tmp_path / "audit.jsonl"
    budget = _budget_with_n_active(3)
    new_budget = deactivate_factor(budget, "f0", "T", "stale", p)
    assert new_budget.active_factor_count == 2


def test_deactivate_appends_audit(tmp_path):
    p = tmp_path / "audit.jsonl"
    budget = _budget_with_n_active(2)
    deactivate_factor(budget, "f0", "T", "removed", p)
    records = load_audit_records(p)
    assert records[-1].action == "DEACTIVATED"


def test_deactivate_unknown_factor_no_crash(tmp_path):
    p = tmp_path / "audit.jsonl"
    budget = ComplexityBudget()
    new_budget = deactivate_factor(budget, "nonexistent", "T", "gone", p)
    assert new_budget.active_factor_count == 0


# ── persistence ───────────────────────────────────────────────────────────────

def test_save_load_budget(tmp_path):
    p = tmp_path / "budget.json"
    budget = _budget_with_n_active(3)
    save_complexity_budget(budget, p)
    loaded = load_complexity_budget(p)
    assert loaded.active_factor_count == 3
    assert loaded.max_factors == MAX_FACTORS


def test_load_missing_budget(tmp_path):
    p = tmp_path / "nonexistent.json"
    budget = load_complexity_budget(p)
    assert budget.active_factor_count == 0


def test_audit_append_only(tmp_path):
    p = tmp_path / "audit.jsonl"
    for i in range(5):
        rec = ComplexityAuditRecord(
            timestamp="T", action="ADMITTED", factor_id=f"f{i}",
            reason="ok", active_count_before=i, active_count_after=i + 1,
        )
        append_audit_record(rec, p)
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5


def test_audit_replay(tmp_path):
    p = tmp_path / "audit.jsonl"
    budget = ComplexityBudget()
    f1 = _valid_factor("f1")
    f2 = _valid_factor("f2")
    budget, _, _ = admit_factor(budget, f1, "T1", p)
    budget, _, _ = admit_factor(budget, f2, "T2", p)
    records = load_audit_records(p)
    assert len(records) == 2
    assert records[0].factor_id == "f1"
    assert records[1].factor_id == "f2"
