"""
tests/deployment/test_governance_diff.py
Phase 4B.1 — GovernanceDecisionDiff tests.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pytest

from src.deployment.governance_audit import (
    GovernanceAuditRecord,
    GovernanceDecisionDiff,
    append_governance_audit,
    append_governance_diff,
    compute_governance_diff,
    load_governance_diffs,
    load_last_governance_audit,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_audit(
    *,
    evaluation_date: str = "2026-05-26",
    governance_decisions: list[str] | None = None,
    rollback_decisions: list[str] | None = None,
    freeze_decisions: list[str] | None = None,
    deterministic_hash: str = "abc123",
    freeze_active: bool = False,
) -> GovernanceAuditRecord:
    return GovernanceAuditRecord(
        evaluation_date=evaluation_date,
        upstream_hashes={"run_id": "x"},
        eligibility_gate_passed=True,
        governance_decisions=governance_decisions or [],
        rollback_decisions=rollback_decisions or [],
        freeze_decisions=freeze_decisions or [],
        reconciliation_status="clean",
        deterministic_hash=deterministic_hash,
        freeze_active=freeze_active,
        timestamp="2026-05-26T09:00:00+09:00",
    )


# ── Tests: compute_governance_diff ─────────────────────────────────────────────

class TestComputeGovernanceDiff:
    def test_no_previous_first_run(self):
        current = _make_audit(governance_decisions=["ALLOW:7203.T"])
        diff = compute_governance_diff(None, current)
        # first run always changed=True (no previous hash to compare)
        assert diff.previous_hash is None
        assert diff.current_hash == "abc123"
        assert diff.changed is True
        assert "ALLOW:7203.T" in diff.added_decisions
        assert diff.removed_decisions == []
        assert diff.freeze_state_changed is False
        assert diff.rollback_state_changed is False

    def test_no_change_same_hash(self):
        rec = _make_audit(
            governance_decisions=["ALLOW:7203.T"],
            deterministic_hash="samehash",
        )
        diff = compute_governance_diff(rec, rec)
        assert diff.changed is False
        assert diff.added_decisions == []
        assert diff.removed_decisions == []
        assert diff.freeze_state_changed is False
        assert diff.rollback_state_changed is False
        assert diff.previous_hash == "samehash"
        assert diff.current_hash == "samehash"

    def test_added_governance_decision(self):
        prev = _make_audit(
            governance_decisions=["ALLOW:7203.T"],
            deterministic_hash="hash1",
        )
        curr = _make_audit(
            governance_decisions=["ALLOW:7203.T", "ALLOW:6758.T"],
            deterministic_hash="hash2",
        )
        diff = compute_governance_diff(prev, curr)
        assert diff.changed is True
        assert "ALLOW:6758.T" in diff.added_decisions
        assert diff.removed_decisions == []

    def test_removed_governance_decision(self):
        prev = _make_audit(
            governance_decisions=["ALLOW:7203.T", "ALLOW:6758.T"],
            deterministic_hash="hash1",
        )
        curr = _make_audit(
            governance_decisions=["ALLOW:7203.T"],
            deterministic_hash="hash2",
        )
        diff = compute_governance_diff(prev, curr)
        assert diff.changed is True
        assert "ALLOW:6758.T" in diff.removed_decisions
        assert diff.added_decisions == []

    def test_freeze_state_changed_unfrozen_to_frozen(self):
        prev = _make_audit(freeze_active=False, deterministic_hash="h1")
        curr = _make_audit(freeze_active=True, deterministic_hash="h2")
        diff = compute_governance_diff(prev, curr)
        assert diff.freeze_state_changed is True
        assert diff.changed is True

    def test_freeze_state_changed_frozen_to_unfrozen(self):
        prev = _make_audit(freeze_active=True, deterministic_hash="h1")
        curr = _make_audit(freeze_active=False, deterministic_hash="h2")
        diff = compute_governance_diff(prev, curr)
        assert diff.freeze_state_changed is True

    def test_freeze_state_unchanged(self):
        prev = _make_audit(freeze_active=False, deterministic_hash="h1")
        curr = _make_audit(freeze_active=False, deterministic_hash="h1")
        diff = compute_governance_diff(prev, curr)
        assert diff.freeze_state_changed is False

    def test_rollback_state_changed(self):
        prev = _make_audit(rollback_decisions=[], deterministic_hash="h1")
        curr = _make_audit(rollback_decisions=["ROLLBACK:7203.T"], deterministic_hash="h2")
        diff = compute_governance_diff(prev, curr)
        assert diff.rollback_state_changed is True
        assert diff.changed is True

    def test_rollback_state_unchanged(self):
        prev = _make_audit(rollback_decisions=["ROLLBACK:7203.T"], deterministic_hash="h1")
        curr = _make_audit(rollback_decisions=["ROLLBACK:7203.T"], deterministic_hash="h1")
        diff = compute_governance_diff(prev, curr)
        assert diff.rollback_state_changed is False

    def test_freeze_decision_in_added(self):
        prev = _make_audit(freeze_decisions=[], deterministic_hash="h1")
        curr = _make_audit(freeze_decisions=["FREEZE_TRIGGERED:broker_timeout"], deterministic_hash="h2")
        diff = compute_governance_diff(prev, curr)
        assert "FREEZE_TRIGGERED:broker_timeout" in diff.added_decisions

    def test_set_diff_order_invariant(self):
        """Decision lists may be in any order — set diff is order-agnostic."""
        prev = _make_audit(
            governance_decisions=["A", "B"],
            deterministic_hash="h1",
        )
        curr = _make_audit(
            governance_decisions=["B", "A"],
            deterministic_hash="h1",
        )
        diff = compute_governance_diff(prev, curr)
        assert diff.added_decisions == []
        assert diff.removed_decisions == []
        assert diff.changed is False

    def test_stability_same_inputs_same_result(self):
        prev = _make_audit(governance_decisions=["ALLOW:7203.T"], deterministic_hash="h1")
        curr = _make_audit(governance_decisions=["ALLOW:7203.T", "DENY:9432.T"], deterministic_hash="h2")
        diff1 = compute_governance_diff(prev, curr)
        diff2 = compute_governance_diff(prev, curr)
        assert diff1.changed == diff2.changed
        assert diff1.added_decisions == diff2.added_decisions
        assert diff1.removed_decisions == diff2.removed_decisions
        assert diff1.freeze_state_changed == diff2.freeze_state_changed
        assert diff1.rollback_state_changed == diff2.rollback_state_changed
        assert diff1.current_hash == diff2.current_hash

    def test_hash_only_diff_is_change(self):
        """Same decisions, different hash → changed=True."""
        prev = _make_audit(governance_decisions=["A"], deterministic_hash="hash_old")
        curr = _make_audit(governance_decisions=["A"], deterministic_hash="hash_new")
        diff = compute_governance_diff(prev, curr)
        assert diff.changed is True
        assert diff.added_decisions == []
        assert diff.removed_decisions == []


# ── Tests: to_dict / from_dict ────────────────────────────────────────────────

class TestGovernanceDecisionDiffSerialization:
    def _make_diff(self) -> GovernanceDecisionDiff:
        prev = _make_audit(governance_decisions=["ALLOW:7203.T"], deterministic_hash="h1")
        curr = _make_audit(governance_decisions=["ALLOW:7203.T", "DENY:x"], deterministic_hash="h2")
        return compute_governance_diff(prev, curr)

    def test_to_dict_keys(self):
        d = self._make_diff().to_dict()
        assert "ts" in d
        assert "previous_hash" in d
        assert "current_hash" in d
        assert "changed" in d
        assert "added_decisions" in d
        assert "removed_decisions" in d
        assert "freeze_state_changed" in d
        assert "rollback_state_changed" in d

    def test_from_dict_roundtrip(self):
        diff = self._make_diff()
        d = diff.to_dict()
        restored = GovernanceDecisionDiff.from_dict(d)
        assert restored.current_hash == diff.current_hash
        assert restored.previous_hash == diff.previous_hash
        assert restored.changed == diff.changed
        assert restored.added_decisions == diff.added_decisions
        assert restored.removed_decisions == diff.removed_decisions
        assert restored.freeze_state_changed == diff.freeze_state_changed
        assert restored.rollback_state_changed == diff.rollback_state_changed

    def test_from_dict_no_previous_hash(self):
        d = {"ts": "t", "previous_hash": None, "current_hash": "abc",
             "changed": True, "added_decisions": ["X"], "removed_decisions": [],
             "freeze_state_changed": False, "rollback_state_changed": False}
        diff = GovernanceDecisionDiff.from_dict(d)
        assert diff.previous_hash is None
        assert diff.added_decisions == ["X"]


# ── Tests: append + load ───────────────────────────────────────────────────────

class TestGovernanceDiffIO:
    def test_append_creates_file(self, tmp_path):
        prev = _make_audit(deterministic_hash="h1")
        curr = _make_audit(deterministic_hash="h2")
        diff = compute_governance_diff(prev, curr)
        append_governance_diff(diff, tmp_path)
        files = list(tmp_path.glob("governance_diff_*.jsonl"))
        assert len(files) == 1

    def test_load_roundtrip(self, tmp_path):
        prev = _make_audit(governance_decisions=["ALLOW:7203.T"], deterministic_hash="h1")
        curr = _make_audit(governance_decisions=["ALLOW:7203.T", "DENY:x"], deterministic_hash="h2")
        diff = compute_governance_diff(prev, curr)
        append_governance_diff(diff, tmp_path)
        loaded = load_governance_diffs(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["current_hash"] == "h2"
        assert "DENY:x" in loaded[0]["added_decisions"]

    def test_append_only_multiple_records(self, tmp_path):
        for i in range(3):
            prev = _make_audit(deterministic_hash=f"h{i}")
            curr = _make_audit(deterministic_hash=f"h{i+1}")
            diff = compute_governance_diff(prev, curr)
            append_governance_diff(diff, tmp_path)
        loaded = load_governance_diffs(tmp_path)
        assert len(loaded) == 3

    def test_no_deletion_on_reopen(self, tmp_path):
        """Simulate multiple writes — no truncation."""
        for _ in range(5):
            diff = compute_governance_diff(
                _make_audit(deterministic_hash="old"),
                _make_audit(deterministic_hash="new"),
            )
            append_governance_diff(diff, tmp_path)
        loaded = load_governance_diffs(tmp_path)
        assert len(loaded) == 5

    def test_load_empty_dir(self, tmp_path):
        assert load_governance_diffs(tmp_path) == []

    def test_load_nonexistent_dir(self, tmp_path):
        assert load_governance_diffs(tmp_path / "nonexistent") == []

    def test_load_by_date_str(self, tmp_path):
        diff = compute_governance_diff(
            _make_audit(deterministic_hash="h1"),
            _make_audit(deterministic_hash="h2"),
        )
        append_governance_diff(diff, tmp_path)
        files = list(tmp_path.glob("governance_diff_*.jsonl"))
        date_str = files[0].stem.replace("governance_diff_", "")
        loaded = load_governance_diffs(tmp_path, date_str=date_str)
        assert len(loaded) == 1


# ── Tests: load_last_governance_audit ─────────────────────────────────────────

class TestLoadLastGovernanceAudit:
    def test_empty_dir_returns_none(self, tmp_path):
        assert load_last_governance_audit(tmp_path) is None

    def test_nonexistent_dir_returns_none(self, tmp_path):
        assert load_last_governance_audit(tmp_path / "nodir") is None

    def test_single_record_returned(self, tmp_path):
        rec = _make_audit(deterministic_hash="first")
        append_governance_audit(rec, tmp_path)
        loaded = load_last_governance_audit(tmp_path)
        assert loaded is not None
        assert loaded.deterministic_hash == "first"

    def test_returns_last_of_multiple(self, tmp_path):
        for h in ["hash_a", "hash_b", "hash_c"]:
            rec = _make_audit(deterministic_hash=h)
            append_governance_audit(rec, tmp_path)
        loaded = load_last_governance_audit(tmp_path)
        assert loaded is not None
        assert loaded.deterministic_hash == "hash_c"

    def test_from_dict_reconstructs_correctly(self, tmp_path):
        rec = _make_audit(
            governance_decisions=["ALLOW:7203.T"],
            freeze_active=True,
            deterministic_hash="xyz",
        )
        append_governance_audit(rec, tmp_path)
        loaded = load_last_governance_audit(tmp_path)
        assert loaded is not None
        assert loaded.governance_decisions == ["ALLOW:7203.T"]
        assert loaded.freeze_active is True


# ── Tests: deterministic replay consistency ────────────────────────────────────

class TestDeterministicReplayConsistency:
    def test_diff_replay_same_decisions(self):
        """Same audit pair → same diff fields (excluding ts which uses now())."""
        prev = _make_audit(governance_decisions=["A", "B"], deterministic_hash="h1")
        curr = _make_audit(governance_decisions=["B", "C"], deterministic_hash="h2")
        d1 = compute_governance_diff(prev, curr)
        d2 = compute_governance_diff(prev, curr)
        assert d1.added_decisions == d2.added_decisions
        assert d1.removed_decisions == d2.removed_decisions
        assert d1.changed == d2.changed
        assert d1.freeze_state_changed == d2.freeze_state_changed
        assert d1.rollback_state_changed == d2.rollback_state_changed
        assert d1.previous_hash == d2.previous_hash
        assert d1.current_hash == d2.current_hash

    def test_audit_from_dict_roundtrip(self, tmp_path):
        rec = _make_audit(
            evaluation_date="2026-05-26",
            governance_decisions=["ALLOW:7203.T"],
            rollback_decisions=["ROLLBACK:6758.T"],
            freeze_decisions=["FREEZE_MAINTAINED: need 1d more"],
            deterministic_hash="fullhash",
            freeze_active=True,
        )
        append_governance_audit(rec, tmp_path)
        loaded = load_last_governance_audit(tmp_path)
        assert loaded is not None
        assert loaded.evaluation_date == "2026-05-26"
        assert loaded.rollback_decisions == ["ROLLBACK:6758.T"]
        assert loaded.freeze_decisions == ["FREEZE_MAINTAINED: need 1d more"]
        assert loaded.deterministic_hash == "fullhash"
        assert loaded.freeze_active is True
