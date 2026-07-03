"""
tests/deployment/test_human_confirmation_telemetry.py
Phase 4B.1 — HumanConfirmationOutcome telemetry tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.deployment.human_confirmation_telemetry import (
    HumanConfirmationOutcome,
    append_human_confirmation,
    load_human_confirmations,
    make_approved_outcome,
    make_rejected_outcome,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _approved(**kwargs) -> HumanConfirmationOutcome:
    defaults = dict(
        ts="2026-05-26T09:00:00+09:00",
        execution_mode="human_confirmed",
        symbol="7203.T",
        intended_action="BUY",
        governance_hash="abc123",
        approved=True,
        rejected=False,
        modified=False,
    )
    defaults.update(kwargs)
    return HumanConfirmationOutcome(**defaults)


def _rejected(**kwargs) -> HumanConfirmationOutcome:
    defaults = dict(
        ts="2026-05-26T09:00:00+09:00",
        execution_mode="human_confirmed",
        symbol="7203.T",
        intended_action="BUY",
        governance_hash="abc123",
        approved=False,
        rejected=True,
        modified=False,
        rejection_reason="governance_frozen",
    )
    defaults.update(kwargs)
    return HumanConfirmationOutcome(**defaults)


# ── Tests: HumanConfirmationOutcome fields ────────────────────────────────────

class TestHumanConfirmationOutcome:
    def test_approved_outcome_fields(self):
        o = _approved()
        assert o.approved is True
        assert o.rejected is False
        assert o.modified is False
        assert o.rejection_reason is None
        assert o.symbol == "7203.T"
        assert o.governance_hash == "abc123"

    def test_rejected_outcome_fields(self):
        o = _rejected()
        assert o.approved is False
        assert o.rejected is True
        assert o.rejection_reason == "governance_frozen"

    def test_approved_and_rejected_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            HumanConfirmationOutcome(
                ts="t",
                execution_mode="human_confirmed",
                symbol="7203.T",
                intended_action="BUY",
                governance_hash="h",
                approved=True,
                rejected=True,
                modified=False,
            )

    def test_modified_coexists_with_approved(self):
        o = _approved(modified=True)
        assert o.approved is True
        assert o.modified is True

    def test_neither_approved_nor_rejected_is_valid(self):
        # Edge case: both False — e.g., pending state or observation-only
        o = _approved(approved=False, rejected=False)
        assert o.approved is False
        assert o.rejected is False

    def test_rejection_reason_none_when_approved(self):
        o = _approved()
        assert o.rejection_reason is None

    def test_rejection_reason_set_when_rejected(self):
        o = _rejected(rejection_reason="freeze_active")
        assert o.rejection_reason == "freeze_active"


# ── Tests: serialization ───────────────────────────────────────────────────────

class TestHumanConfirmationSerialization:
    def test_to_dict_keys(self):
        d = _approved().to_dict()
        assert set(d.keys()) == {
            "ts", "execution_mode", "symbol", "intended_action",
            "governance_hash", "approved", "rejected", "modified",
            "rejection_reason",
        }

    def test_approved_roundtrip(self):
        o = _approved(modified=True)
        restored = HumanConfirmationOutcome.from_dict(o.to_dict())
        assert restored.approved is True
        assert restored.modified is True
        assert restored.rejected is False
        assert restored.symbol == o.symbol
        assert restored.governance_hash == o.governance_hash

    def test_rejected_roundtrip(self):
        o = _rejected(rejection_reason="broker_timeout")
        restored = HumanConfirmationOutcome.from_dict(o.to_dict())
        assert restored.rejected is True
        assert restored.rejection_reason == "broker_timeout"
        assert restored.approved is False

    def test_from_dict_defaults(self):
        minimal = {"ts": "t", "execution_mode": "intent_only", "symbol": "X",
                   "intended_action": "SELL", "governance_hash": "h",
                   "approved": False, "rejected": False, "modified": False}
        o = HumanConfirmationOutcome.from_dict(minimal)
        assert o.rejection_reason is None
        assert o.symbol == "X"


# ── Tests: I/O helpers ─────────────────────────────────────────────────────────

class TestHumanConfirmationIO:
    def test_append_creates_file(self, tmp_path):
        append_human_confirmation(_approved(), tmp_path)
        files = list(tmp_path.glob("human_confirmation_*.jsonl"))
        assert len(files) == 1

    def test_load_empty_dir(self, tmp_path):
        assert load_human_confirmations(tmp_path) == []

    def test_load_nonexistent_dir(self, tmp_path):
        assert load_human_confirmations(tmp_path / "nodir") == []

    def test_append_and_load_approved(self, tmp_path):
        append_human_confirmation(_approved(), tmp_path)
        records = load_human_confirmations(tmp_path)
        assert len(records) == 1
        assert records[0]["approved"] is True
        assert records[0]["symbol"] == "7203.T"

    def test_append_and_load_rejected(self, tmp_path):
        append_human_confirmation(_rejected(), tmp_path)
        records = load_human_confirmations(tmp_path)
        assert len(records) == 1
        assert records[0]["rejected"] is True
        assert records[0]["rejection_reason"] == "governance_frozen"

    def test_append_only_multiple_records(self, tmp_path):
        for i in range(4):
            append_human_confirmation(_approved(symbol=f"sym_{i}.T"), tmp_path)
        records = load_human_confirmations(tmp_path)
        assert len(records) == 4

    def test_no_deletion_on_reopen(self, tmp_path):
        """Append 6× — file must contain 6 lines (no truncation)."""
        for _ in range(6):
            append_human_confirmation(_approved(), tmp_path)
        records = load_human_confirmations(tmp_path)
        assert len(records) == 6

    def test_load_by_date_str(self, tmp_path):
        append_human_confirmation(_approved(), tmp_path)
        files = list(tmp_path.glob("human_confirmation_*.jsonl"))
        date_str = files[0].stem.replace("human_confirmation_", "")
        loaded = load_human_confirmations(tmp_path, date_str=date_str)
        assert len(loaded) == 1

    def test_load_by_date_str_no_match(self, tmp_path):
        append_human_confirmation(_approved(), tmp_path)
        assert load_human_confirmations(tmp_path, date_str="20000101") == []

    def test_jsonl_one_object_per_line(self, tmp_path):
        """Each record is valid JSON on its own line."""
        for _ in range(3):
            append_human_confirmation(_approved(), tmp_path)
        files = list(tmp_path.glob("human_confirmation_*.jsonl"))
        lines = [l.strip() for l in files[0].read_text("utf-8").splitlines() if l.strip()]
        assert len(lines) == 3
        for line in lines:
            obj = json.loads(line)
            assert "ts" in obj


# ── Tests: convenience constructors ───────────────────────────────────────────

class TestConvenienceConstructors:
    def test_make_approved_outcome(self):
        o = make_approved_outcome(
            symbol="7203.T",
            intended_action="BUY",
            governance_hash="gh1",
        )
        assert o.approved is True
        assert o.rejected is False
        assert o.modified is False
        assert o.symbol == "7203.T"
        assert o.governance_hash == "gh1"
        assert o.execution_mode == "human_confirmed"

    def test_make_approved_outcome_modified(self):
        o = make_approved_outcome(
            symbol="7203.T",
            intended_action="BUY",
            governance_hash="gh1",
            modified=True,
        )
        assert o.approved is True
        assert o.modified is True

    def test_make_rejected_outcome(self):
        o = make_rejected_outcome(
            symbol="7203.T",
            intended_action="BUY",
            governance_hash="gh1",
            rejection_reason="governance_frozen",
        )
        assert o.rejected is True
        assert o.approved is False
        assert o.rejection_reason == "governance_frozen"

    def test_make_approved_custom_mode(self):
        o = make_approved_outcome(
            symbol="X",
            intended_action="SELL",
            governance_hash="h",
            execution_mode="intent_only",
        )
        assert o.execution_mode == "intent_only"

    def test_make_rejected_custom_mode(self):
        o = make_rejected_outcome(
            symbol="X",
            intended_action="BUY",
            governance_hash="h",
            rejection_reason="pressure_exceeded",
            execution_mode="fully_autonomous",
        )
        assert o.execution_mode == "fully_autonomous"
        assert o.rejection_reason == "pressure_exceeded"

    def test_constructors_produce_valid_serialization(self, tmp_path):
        for o in [
            make_approved_outcome("7203.T", "BUY", "h1"),
            make_rejected_outcome("6758.T", "BUY", "h2", "freeze"),
        ]:
            append_human_confirmation(o, tmp_path)
        records = load_human_confirmations(tmp_path)
        assert len(records) == 2
        assert records[0]["approved"] is True
        assert records[1]["rejected"] is True
