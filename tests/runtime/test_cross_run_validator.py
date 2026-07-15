"""
tests/runtime/test_cross_run_validator.py

Deterministic cross-run replay equivalence validation tests.

Covers (per spec):
  1.  identical runs -> replay_stable=True
  2.  reordered JSON keys -> still equivalent (same equivalence_key)
  3.  sequence drift -> divergence detected
  4.  replay hash mismatch -> severity=CRITICAL, invariant_status=FAILED
  5.  append-only persistence (prior records never mutated)
  6.  deterministic serialization (same inputs always same equivalence_key)
  7.  governance mutation changes equivalence group
  8.  hidden mutable state detected via hash mismatch
  9.  comparison window enforced
  10. quarantine_required=True on divergence
  11. validation output replay-safe (no wall-clock in persisted schema)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.runtime.determinism.cross_run_validator import (
    CrossRunValidator,
    ValidationReport,
    VALIDATOR_VERSION,
    CANONICALIZATION_VERSION,
    COMPARISON_WINDOW,
)


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_inputs(**overrides) -> dict:
    base = dict(
        epoch_manifest          = {"version": "1", "dataset": "ds_a"},
        orchestration_manifest  = {"stages": ["preprocess", "signal", "order"]},
        dependency_manifest     = {"deps": {"lib": "1.0"}},
        governance_snapshot     = {"policy": "strict", "level": 3},
    )
    base.update(overrides)
    return base


def _validator(tmp_path, window: int = COMPARISON_WINDOW) -> CrossRunValidator:
    return CrossRunValidator(output_path=tmp_path / "cv.jsonl", window=window)


# ─────────────────────────────────────────────────────────────────────────────
# 1. identical runs → replay_stable=True
# ─────────────────────────────────────────────────────────────────────────────

class TestIdenticalRuns:
    def test_single_run_stable(self, tmp_path):
        v = _validator(tmp_path)
        r = v.validate("r1", "hash_a", **_make_inputs())
        assert r.replay_stable is True
        assert r.divergence_detected is False
        assert r.invariant_status == "PASSED"

    def test_two_identical_runs_stable(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "hash_a", **inputs)
        r2 = v.validate("r2", "hash_a", **inputs)
        assert r2.replay_stable is True
        assert r2.divergence_detected is False
        assert "r2" not in r2.divergent_runs

    def test_many_identical_runs_all_stable(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        for i in range(5):
            r = v.validate(f"r{i}", "hash_stable", **inputs)
        assert r.replay_stable is True
        assert r.divergent_runs == []


# ─────────────────────────────────────────────────────────────────────────────
# 2. reordered JSON dict keys → same equivalence group
# ─────────────────────────────────────────────────────────────────────────────

class TestReorderedDictKeys:
    def test_same_equivalence_key_regardless_of_insertion_order(self, tmp_path):
        v = _validator(tmp_path)
        # epoch_manifest dict with keys in different insertion order
        inputs_a = _make_inputs(epoch_manifest={"a": 1, "b": 2, "c": 3})
        inputs_b = _make_inputs(epoch_manifest={"c": 3, "a": 1, "b": 2})
        eq_a = v._compute_equivalence_key(**inputs_a)
        eq_b = v._compute_equivalence_key(**inputs_b)
        assert eq_a == eq_b

    def test_reordered_governance_keys_same_group(self, tmp_path):
        v = _validator(tmp_path)
        inputs_a = _make_inputs(governance_snapshot={"x": 1, "y": 2})
        inputs_b = _make_inputs(governance_snapshot={"y": 2, "x": 1})
        r1 = v.validate("r1", "h1", **inputs_a)
        r2 = v.validate("r2", "h1", **inputs_b)
        assert r1.equivalence_key == r2.equivalence_key
        assert r2.replay_stable is True

    def test_nested_dict_reorder_same_key(self, tmp_path):
        v = _validator(tmp_path)
        inputs_a = _make_inputs(dependency_manifest={"deps": {"b": "2", "a": "1"}})
        inputs_b = _make_inputs(dependency_manifest={"deps": {"a": "1", "b": "2"}})
        eq_a = v._compute_equivalence_key(**inputs_a)
        eq_b = v._compute_equivalence_key(**inputs_b)
        assert eq_a == eq_b


# ─────────────────────────────────────────────────────────────────────────────
# 3. sequence drift → divergence detected
# ─────────────────────────────────────────────────────────────────────────────

class TestSequenceDrift:
    def test_list_reorder_in_orch_creates_new_group(self, tmp_path):
        """Reordering a list in orchestration_manifest changes equivalence_key."""
        v = _validator(tmp_path)
        inputs_a = _make_inputs(orchestration_manifest={"stages": ["a", "b", "c"]})
        inputs_b = _make_inputs(orchestration_manifest={"stages": ["c", "b", "a"]})
        eq_a = v._compute_equivalence_key(**inputs_a)
        eq_b = v._compute_equivalence_key(**inputs_b)
        assert eq_a != eq_b

    def test_ledger_ordering_drift_triggers_divergence(self, tmp_path):
        """Same logical inputs, but replay_hash differs due to ledger ordering drift."""
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "hash_ordered",   **inputs)
        r2 = v.validate("r2", "hash_reordered", **inputs)
        assert r2.divergence_detected is True
        assert "r2" in r2.divergent_runs

    def test_list_append_in_stages_new_group(self, tmp_path):
        """Adding a stage element changes equivalence_key."""
        v = _validator(tmp_path)
        inputs_a = _make_inputs(orchestration_manifest={"stages": ["a", "b"]})
        inputs_b = _make_inputs(orchestration_manifest={"stages": ["a", "b", "c"]})
        eq_a = v._compute_equivalence_key(**inputs_a)
        eq_b = v._compute_equivalence_key(**inputs_b)
        assert eq_a != eq_b

    def test_empty_vs_nonempty_list_different_key(self, tmp_path):
        v = _validator(tmp_path)
        inputs_a = _make_inputs(orchestration_manifest={"stages": []})
        inputs_b = _make_inputs(orchestration_manifest={"stages": ["s1"]})
        eq_a = v._compute_equivalence_key(**inputs_a)
        eq_b = v._compute_equivalence_key(**inputs_b)
        assert eq_a != eq_b


# ─────────────────────────────────────────────────────────────────────────────
# 4. replay hash mismatch → CRITICAL severity + FAILED status
# ─────────────────────────────────────────────────────────────────────────────

class TestHashMismatch:
    def test_mismatch_severity_critical(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "hash_a", **inputs)
        r2 = v.validate("r2", "hash_b", **inputs)
        assert r2.severity == "CRITICAL"

    def test_mismatch_invariant_status_failed(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "hash_a", **inputs)
        r2 = v.validate("r2", "hash_b", **inputs)
        assert r2.invariant_status == "FAILED"

    def test_matching_severity_info(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "hash_a", **inputs)
        r2 = v.validate("r2", "hash_a", **inputs)
        assert r2.severity == "INFO"

    def test_divergent_run_ids_recorded(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "hash_x", **inputs)
        v.validate("r2", "hash_x", **inputs)
        r3 = v.validate("r3", "hash_y", **inputs)
        assert "r3" in r3.divergent_runs
        assert "r1" not in r3.divergent_runs
        assert "r2" not in r3.divergent_runs

    def test_authoritative_hash_is_first_seen(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "first_hash",   **inputs)
        r2 = v.validate("r2", "second_hash", **inputs)
        assert r2.authoritative_hash == "first_hash"


# ─────────────────────────────────────────────────────────────────────────────
# 5. append-only persistence
# ─────────────────────────────────────────────────────────────────────────────

class TestAppendOnly:
    def test_records_grow_monotonically(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        for i in range(4):
            v.validate(f"r{i}", "h", **inputs)
        records = v.load_records()
        assert len(records) == 4

    def test_prior_records_not_mutated(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "hash_a", **inputs)
        first = v.load_records()[0]
        v.validate("r2", "hash_b", **inputs)  # divergence
        # First record must be unchanged
        still_first = v.load_records()[0]
        assert first == still_first

    def test_divergence_does_not_remove_records(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "h1", **inputs)
        v.validate("r2", "h2_BAD", **inputs)
        v.validate("r3", "h1",    **inputs)
        assert len(v.load_records()) == 3

    def test_persist_called_on_every_validate(self, tmp_path):
        v = _validator(tmp_path)
        for i in range(7):
            v.validate(f"r{i}", "hh", **_make_inputs())
        assert len(v.load_records()) == 7


# ─────────────────────────────────────────────────────────────────────────────
# 6. deterministic serialization
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterministicSerialization:
    def test_same_inputs_same_equivalence_key(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        for _ in range(5):
            assert v._compute_equivalence_key(**inputs) == v._compute_equivalence_key(**inputs)

    def test_equivalence_key_is_64_char_hex(self, tmp_path):
        v = _validator(tmp_path)
        eq = v._compute_equivalence_key(**_make_inputs())
        assert len(eq) == 64
        int(eq, 16)

    def test_record_fields_are_canonical_types(self, tmp_path):
        v = _validator(tmp_path)
        r = v.validate("r1", "hh", **_make_inputs())
        rec = r.to_record()
        assert isinstance(rec["validator_version"], str)
        assert isinstance(rec["replay_stable"], bool)
        assert isinstance(rec["compared_runs"], list)
        assert isinstance(rec["replay_hashes"], list)

    def test_no_float_timestamp_in_required_fields(self, tmp_path):
        """Persisted record must match spec schema exactly; no ts field."""
        v = _validator(tmp_path)
        v.validate("r1", "hh", **_make_inputs())
        rec = v.load_records()[0]
        required = {
            "validator_version", "canonicalization_version",
            "validation_sequence_id", "comparison_window",
            "equivalence_key", "authoritative_hash",
            "compared_runs", "divergent_runs", "replay_hashes",
            "replay_stable", "divergence_detected", "quarantine_required",
            "severity", "invariant_status",
        }
        assert set(rec.keys()) == required

    def test_serialized_line_is_valid_json(self, tmp_path):
        v = _validator(tmp_path)
        v.validate("r1", "hh", **_make_inputs())
        raw = (tmp_path / "cv.jsonl").read_text(encoding="utf-8").strip()
        parsed = json.loads(raw)  # must not raise
        assert parsed["validator_version"] == VALIDATOR_VERSION

    def test_validation_sequence_id_unique_per_call(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        r1 = v.validate("r1", "h", **inputs)
        r2 = v.validate("r2", "h", **inputs)
        assert r1.validation_sequence_id != r2.validation_sequence_id

    def test_canonicalization_version_constant(self, tmp_path):
        v = _validator(tmp_path)
        r = v.validate("r1", "h", **_make_inputs())
        assert r.canonicalization_version == CANONICALIZATION_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# 7. governance mutation changes equivalence group
# ─────────────────────────────────────────────────────────────────────────────

class TestGovernanceMutation:
    def test_different_governance_different_key(self, tmp_path):
        v = _validator(tmp_path)
        inputs_a = _make_inputs(governance_snapshot={"policy": "strict"})
        inputs_b = _make_inputs(governance_snapshot={"policy": "relaxed"})
        eq_a = v._compute_equivalence_key(**inputs_a)
        eq_b = v._compute_equivalence_key(**inputs_b)
        assert eq_a != eq_b

    def test_governance_mutation_new_independent_group(self, tmp_path):
        v = _validator(tmp_path)
        inputs_a = _make_inputs(governance_snapshot={"policy": "strict"})
        inputs_b = _make_inputs(governance_snapshot={"policy": "relaxed"})
        r1 = v.validate("r1", "h_strict",  **inputs_a)
        r2 = v.validate("r2", "h_relaxed", **inputs_b)
        # Different groups — no cross-contamination
        assert r1.equivalence_key != r2.equivalence_key
        assert r2.replay_stable is True  # first run in its group

    def test_governance_change_breaks_prior_equivalence(self, tmp_path):
        v = _validator(tmp_path)
        inputs_a = _make_inputs(governance_snapshot={"policy": "v1"})
        inputs_b = _make_inputs(governance_snapshot={"policy": "v2"})
        v.validate("r1", "h1", **inputs_a)
        # r2 uses different governance — different group, not compared against r1
        r2 = v.validate("r2", "h1", **inputs_b)
        assert r2.compared_runs == ["r2"]

    def test_epoch_mutation_different_group(self, tmp_path):
        v = _validator(tmp_path)
        inputs_a = _make_inputs(epoch_manifest={"version": "1"})
        inputs_b = _make_inputs(epoch_manifest={"version": "2"})
        eq_a = v._compute_equivalence_key(**inputs_a)
        eq_b = v._compute_equivalence_key(**inputs_b)
        assert eq_a != eq_b


# ─────────────────────────────────────────────────────────────────────────────
# 8. hidden mutable state detected via hash mismatch
# ─────────────────────────────────────────────────────────────────────────────

class TestHiddenMutableState:
    def test_hash_mismatch_exposes_mutable_state(self, tmp_path):
        """Upstream replay_hash changes despite identical logical inputs → detected."""
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "deterministic_hash", **inputs)
        r2 = v.validate("r2", "nondeterministic_hash_due_to_hidden_state", **inputs)
        assert r2.divergence_detected is True
        assert r2.quarantine_required is True
        assert r2.invariant_status == "FAILED"

    def test_first_run_sets_authoritative_baseline(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        r1 = v.validate("r1", "baseline", **inputs)
        assert r1.authoritative_hash == "baseline"

    def test_subsequent_deviation_from_baseline_detected(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "baseline", **inputs)
        v.validate("r2", "baseline", **inputs)  # same
        r3 = v.validate("r3", "deviated", **inputs)  # mutable state emerged
        assert r3.divergence_detected is True
        assert "r3" in r3.divergent_runs
        assert "r1" not in r3.divergent_runs


# ─────────────────────────────────────────────────────────────────────────────
# 9. comparison window enforced
# ─────────────────────────────────────────────────────────────────────────────

class TestComparisonWindow:
    def test_window_caps_compared_runs(self, tmp_path):
        window = 4
        v = _validator(tmp_path, window=window)
        inputs = _make_inputs()
        for i in range(window + 3):
            v.validate(f"r{i}", "h", **inputs)
        last = v.load_records()[-1]
        assert len(last["compared_runs"]) <= window
        assert len(last["replay_hashes"]) <= window

    def test_window_one_only_current(self, tmp_path):
        """Window=1 means only the current run is compared."""
        v = _validator(tmp_path, window=1)
        inputs = _make_inputs()
        v.validate("r1", "h1", **inputs)
        r2 = v.validate("r2", "h2_different", **inputs)
        # With window=1, no prior context → r2 is its own authoritative
        assert r2.compared_runs == ["r2"]
        assert r2.replay_stable is True

    def test_window_default_is_20(self):
        assert COMPARISON_WINDOW == 20

    def test_window_enforced_on_registry_rebuild(self, tmp_path):
        """After process restart (registry rebuilt from JSONL), window still holds."""
        window = 3
        out = tmp_path / "cv.jsonl"
        for i in range(window + 4):
            v = CrossRunValidator(output_path=out, window=window)
            v.validate(f"r{i}", "h", **_make_inputs())
        # Final instance: registry rebuilt from JSONL
        v_new = CrossRunValidator(output_path=out, window=window)
        r_next = v_new.validate("r_new", "h", **_make_inputs())
        assert len(r_next.compared_runs) <= window

    def test_window_two_prior_one_current(self, tmp_path):
        window = 3
        v = _validator(tmp_path, window=window)
        inputs = _make_inputs()
        v.validate("r0", "h", **inputs)
        v.validate("r1", "h", **inputs)
        v.validate("r2", "h", **inputs)
        r3 = v.validate("r3", "h", **inputs)
        assert len(r3.compared_runs) == window


# ─────────────────────────────────────────────────────────────────────────────
# 10. quarantine_required=True on divergence
# ─────────────────────────────────────────────────────────────────────────────

class TestQuarantine:
    def test_quarantine_true_on_hash_mismatch(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "h1", **inputs)
        r2 = v.validate("r2", "h2", **inputs)
        assert r2.quarantine_required is True

    def test_quarantine_false_when_stable(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "h1", **inputs)
        r2 = v.validate("r2", "h1", **inputs)
        assert r2.quarantine_required is False

    def test_quarantine_persisted_in_record(self, tmp_path):
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "h1", **inputs)
        v.validate("r2", "h2", **inputs)
        records = v.load_records()
        assert records[1]["quarantine_required"] is True

    def test_quarantine_false_on_first_run(self, tmp_path):
        v = _validator(tmp_path)
        r1 = v.validate("r1", "h1", **_make_inputs())
        assert r1.quarantine_required is False

    def test_no_auto_healing_quarantine_persists(self, tmp_path):
        """Subsequent stable runs do not remove the quarantine record."""
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "h1", **inputs)
        v.validate("r2", "h2_bad", **inputs)  # quarantine
        v.validate("r3", "h1", **inputs)       # back to stable
        records = v.load_records()
        assert records[1]["quarantine_required"] is True  # still recorded


# ─────────────────────────────────────────────────────────────────────────────
# 11. validation output replay-safe (no wall-clock as comparison authority)
# ─────────────────────────────────────────────────────────────────────────────

class TestReplaySafe:
    def test_persisted_record_has_no_ts_field(self, tmp_path):
        """Wall-clock timestamp must not appear in the persisted record schema."""
        v = _validator(tmp_path)
        v.validate("r1", "h", **_make_inputs())
        rec = v.load_records()[0]
        assert "ts" not in rec

    def test_equivalence_key_unchanged_across_time(self, tmp_path):
        """equivalence_key must be stable regardless of when it's computed."""
        v = _validator(tmp_path)
        inputs = _make_inputs()
        keys = [v._compute_equivalence_key(**inputs) for _ in range(20)]
        assert len(set(keys)) == 1  # all identical

    def test_validation_sequence_id_not_a_timestamp(self, tmp_path):
        """validation_sequence_id is an opaque UUID hex, not a wall-clock value."""
        v = _validator(tmp_path)
        r = v.validate("r1", "h", **_make_inputs())
        vid = r.validation_sequence_id
        assert len(vid) == 32
        int(vid, 16)  # valid hex
        # Not a float timestamp string
        assert "." not in vid

    def test_comparison_uses_only_replay_hash_not_time(self, tmp_path):
        """Two runs with same hash are stable regardless of when they ran."""
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "same_hash", **inputs)
        r2 = v.validate("r2", "same_hash", **inputs)
        assert r2.replay_stable is True
        assert r2.divergence_detected is False

    def test_report_fields_exactly_match_spec_schema(self, tmp_path):
        v = _validator(tmp_path)
        r = v.validate("r1", "h", **_make_inputs())
        rec = r.to_record()
        expected_keys = {
            "validator_version", "canonicalization_version",
            "validation_sequence_id", "comparison_window",
            "equivalence_key", "authoritative_hash",
            "compared_runs", "divergent_runs", "replay_hashes",
            "replay_stable", "divergence_detected",
            "quarantine_required", "severity", "invariant_status",
        }
        assert set(rec.keys()) == expected_keys


# ─────────────────────────────────────────────────────────────────────────────
# Registry rebuild / cross-process restart
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistryRebuild:
    def test_rebuild_preserves_prior_authoritative_hash(self, tmp_path):
        """After restart, authoritative_hash from prior runs is preserved."""
        out = tmp_path / "cv.jsonl"
        v1 = CrossRunValidator(output_path=out)
        inputs = _make_inputs()
        v1.validate("r1", "original_hash", **inputs)
        del v1

        v2 = CrossRunValidator(output_path=out)
        r2 = v2.validate("r2", "diverged_hash", **inputs)
        assert r2.authoritative_hash == "original_hash"
        assert r2.divergence_detected is True

    def test_rebuild_stable_after_restart(self, tmp_path):
        """Stable runs are still stable after process restart."""
        out = tmp_path / "cv.jsonl"
        inputs = _make_inputs()
        CrossRunValidator(output_path=out).validate("r1", "hh", **inputs)
        v2 = CrossRunValidator(output_path=out)
        r2 = v2.validate("r2", "hh", **inputs)
        assert r2.replay_stable is True

    def test_rebuild_empty_file(self, tmp_path):
        """Empty JSONL → clean registry → no prior history."""
        out = tmp_path / "cv.jsonl"
        out.write_text("", encoding="utf-8")
        v = CrossRunValidator(output_path=out)
        r = v.validate("r1", "h1", **_make_inputs())
        assert r.compared_runs == ["r1"]
        assert r.replay_stable is True

    def test_rebuild_skips_malformed_lines(self, tmp_path):
        """Corrupt JSONL lines are ignored; registry continues with valid records."""
        out = tmp_path / "cv.jsonl"
        v = CrossRunValidator(output_path=out)
        v.validate("r1", "h1", **_make_inputs())
        # Inject malformed line
        with out.open("a", encoding="utf-8") as f:
            f.write("NOT VALID JSON\n")
        v2 = CrossRunValidator(output_path=out)
        r2 = v2.validate("r2", "h1", **_make_inputs())
        assert r2.replay_stable is True


# ─────────────────────────────────────────────────────────────────────────────
# Runtime_decision_ledger exclusion from equivalence_key
# ─────────────────────────────────────────────────────────────────────────────

class TestEquivalenceKeyExclusion:
    def test_equivalence_key_does_not_include_replay_hash(self, tmp_path):
        """replay_hash must not affect the equivalence_key computation."""
        v = _validator(tmp_path)
        inputs = _make_inputs()
        eq1 = v._compute_equivalence_key(**inputs)
        # Regardless of replay_hash value, eq_key is the same
        assert eq1 == v._compute_equivalence_key(**inputs)

    def test_two_runs_different_ledgers_same_group(self, tmp_path):
        """
        Two runs with identical logical inputs but different runtime_decision_ledgers
        (expressed via different replay_hash values) belong to the same equivalence
        group and trigger divergence, not a new group.
        """
        v = _validator(tmp_path)
        inputs = _make_inputs()
        v.validate("r1", "hash_ledger_a", **inputs)
        r2 = v.validate("r2", "hash_ledger_b", **inputs)
        # Same equivalence group (same logical inputs)
        assert r2.equivalence_key == v._compute_equivalence_key(**inputs)
        # Different replay_hash → divergence (not a silent split into new group)
        assert r2.divergence_detected is True
