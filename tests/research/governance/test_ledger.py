"""Tests: ledger.py — ExplorationLedger append-only tracking."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.governance.ledger import (
    ExplorationLedger, LedgerEntry,
    ENTRY_EXPLORED, ENTRY_REJECTED, ENTRY_UNSTABLE, ENTRY_COLLAPSED,
)
from tests.research.governance.conftest import (
    make_ledger, FAM_A, FAM_B, GH_A, GH_B, HYP_A, HYP_B, TMPL, PARAM_H, TAX, TS, TS2
)


class TestLedgerEntry:
    def test_frozen(self):
        ledger = make_ledger()
        entry = ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        with pytest.raises(Exception):
            entry.entry_type = "CHANGED"  # type: ignore

    def test_entry_id_prefix(self):
        ledger = make_ledger()
        entry = ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        assert entry.entry_id.startswith("led_")

    def test_to_dict_from_dict_roundtrip(self):
        ledger = make_ledger()
        entry = ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX,
                                        metadata={"k": "v"}, timestamp=TS)
        restored = LedgerEntry.from_dict(entry.to_dict())
        assert restored == entry

    def test_metadata_sorted(self):
        ledger = make_ledger()
        entry = ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX,
                                        metadata={"z": 9, "a": 1}, timestamp=TS)
        keys = [k for k, _ in entry.metadata]
        assert keys == sorted(keys)


class TestExplorationLedger:
    def test_record_explored(self):
        ledger = make_ledger()
        entry = ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        assert entry.entry_type == ENTRY_EXPLORED
        assert entry.family_id == FAM_A

    def test_record_rejected(self):
        ledger = make_ledger()
        entry = ledger.record_rejected(FAM_B, HYP_B, TMPL, PARAM_H, GH_B, TAX, timestamp=TS)
        assert entry.entry_type == ENTRY_REJECTED

    def test_record_unstable(self):
        ledger = make_ledger()
        entry = ledger.record_unstable(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        assert entry.entry_type == ENTRY_UNSTABLE

    def test_record_collapsed(self):
        ledger = make_ledger()
        entry = ledger.record_collapsed(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        assert entry.entry_type == ENTRY_COLLAPSED

    def test_invalid_entry_type_raises(self):
        ledger = make_ledger()
        with pytest.raises(ValueError):
            ledger._record("INVALID", FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, "", None, TS)

    def test_entries_for_family(self):
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        ledger.record_rejected(FAM_A, HYP_B, TMPL, PARAM_H, GH_A, TAX, timestamp=TS2)
        entries = ledger.entries_for_family(FAM_A)
        assert len(entries) == 2
        assert all(e.family_id == FAM_A for e in entries)

    def test_entries_for_family_sorted_by_timestamp(self):
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_B, TMPL, PARAM_H, GH_A, TAX, timestamp=TS2)
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        entries = ledger.entries_for_family(FAM_A)
        ts = [e.recorded_at for e in entries]
        assert ts == sorted(ts)

    def test_entries_for_grammar(self):
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        ledger.record_explored(FAM_B, HYP_B, TMPL, PARAM_H, GH_A, TAX, timestamp=TS2)
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_B, TAX, timestamp=TS)
        assert len(ledger.entries_for_grammar(GH_A)) == 2

    def test_explored_families(self):
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        ledger.record_rejected(FAM_B, HYP_B, TMPL, PARAM_H, GH_B, TAX, timestamp=TS)
        assert FAM_A in ledger.explored_families()
        assert FAM_B not in ledger.explored_families()

    def test_rejected_families(self):
        ledger = make_ledger()
        ledger.record_rejected(FAM_B, HYP_B, TMPL, PARAM_H, GH_B, TAX, timestamp=TS)
        assert FAM_B in ledger.rejected_families()

    def test_unstable_families(self):
        ledger = make_ledger()
        ledger.record_unstable(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        assert FAM_A in ledger.unstable_families()

    def test_is_explored_true(self):
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        assert ledger.is_explored(FAM_A)

    def test_is_explored_false(self):
        ledger = make_ledger()
        assert not ledger.is_explored(FAM_A)

    def test_is_rejected_true(self):
        ledger = make_ledger()
        ledger.record_rejected(FAM_B, HYP_B, TMPL, PARAM_H, GH_B, TAX, timestamp=TS)
        assert ledger.is_rejected(FAM_B)

    def test_exploration_density_zero_for_unexplored(self):
        ledger = make_ledger()
        assert ledger.exploration_density(FAM_A) == 0.0

    def test_exploration_density_nonzero_after_exploration(self):
        ledger = make_ledger()
        for _ in range(10):
            ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        density = ledger.exploration_density(FAM_A, max_entries=100)
        assert density == pytest.approx(0.1, abs=1e-9)

    def test_exploration_density_clamped_at_one(self):
        ledger = make_ledger()
        for _ in range(50):
            ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        assert ledger.exploration_density(FAM_A, max_entries=10) == 1.0

    def test_entry_count(self):
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        ledger.record_rejected(FAM_B, HYP_B, TMPL, PARAM_H, GH_B, TAX, timestamp=TS2)
        assert ledger.entry_count() == 2

    def test_family_count(self):
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        ledger.record_explored(FAM_B, HYP_B, TMPL, PARAM_H, GH_B, TAX, timestamp=TS2)
        assert ledger.family_count() == 2

    def test_all_entries_sorted_by_timestamp(self):
        ledger = make_ledger()
        ledger.record_explored(FAM_B, HYP_B, TMPL, PARAM_H, GH_B, TAX, timestamp=TS2)
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        all_e = ledger.all_entries()
        ts = [e.recorded_at for e in all_e]
        assert ts == sorted(ts)

    def test_deterministic_entry_id(self):
        l1 = make_ledger("test_l")
        l2 = make_ledger("test_l")
        e1 = l1.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        e2 = l2.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        assert e1.entry_id == e2.entry_id

    def test_to_manifest_keys(self):
        ledger = make_ledger()
        ledger.record_explored(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX, timestamp=TS)
        m = ledger.to_manifest()
        assert "ledger_id" in m
        assert "entry_count" in m
        assert "entries" in m

    def test_reason_preserved(self):
        ledger = make_ledger()
        entry = ledger.record_rejected(FAM_A, HYP_A, TMPL, PARAM_H, GH_A, TAX,
                                        reason="density_exceeded", timestamp=TS)
        assert entry.reason == "density_exceeded"
