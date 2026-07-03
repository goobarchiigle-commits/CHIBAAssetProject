"""Tests: memory.py — ResearchMemory."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.governance.memory import (
    ResearchMemory, MemoryEntry,
    MEMORY_EXPLORATION_PATH, MEMORY_REJECTION_HISTORY,
    MEMORY_SURVIVORSHIP_TRAJECTORY, MEMORY_TOPOLOGY_EVOLUTION,
    ALL_MEMORY_TYPES,
)
from tests.research.governance.conftest import (
    make_memory, FAM_A, FAM_B, GH_A, TS, TS2, TS3,
)


class TestMemoryEntry:
    def test_frozen(self):
        mem = make_memory()
        entry = mem.record_exploration_path(FAM_A, ["hyp_a"], timestamp=TS)
        with pytest.raises(Exception):
            entry.memory_type = "CHANGED"  # type: ignore

    def test_entry_id_prefix(self):
        mem = make_memory()
        entry = mem.record_exploration_path(FAM_A, ["hyp_a"], timestamp=TS)
        assert entry.entry_id.startswith("mem_")

    def test_to_dict_from_dict_roundtrip(self):
        mem = make_memory()
        entry = mem.record_exploration_path(FAM_A, ["hyp_a", "hyp_b"], timestamp=TS)
        restored = MemoryEntry.from_dict(entry.to_dict())
        assert restored == entry

    def test_content_hash_length(self):
        mem = make_memory()
        entry = mem.record_exploration_path(FAM_A, ["hyp_a"], timestamp=TS)
        assert len(entry.content_hash) == 16


class TestResearchMemory:
    def test_all_memory_types(self):
        assert len(ALL_MEMORY_TYPES) == 4

    def test_record_exploration_path(self):
        mem = make_memory()
        entry = mem.record_exploration_path(FAM_A, ["hyp_a", "hyp_b"], timestamp=TS)
        assert entry.memory_type == MEMORY_EXPLORATION_PATH
        assert entry.family_id == FAM_A

    def test_exploration_path_hypothesis_count_in_metadata(self):
        mem = make_memory()
        entry = mem.record_exploration_path(FAM_A, ["hyp_a", "hyp_b", "hyp_c"], timestamp=TS)
        metadata_dict = dict(entry.metadata)
        assert metadata_dict["hypothesis_count"] == "3"

    def test_record_rejection_history(self):
        mem = make_memory()
        entry = mem.record_rejection_history(FAM_A, 3, ["density", "topology"], timestamp=TS)
        assert entry.memory_type == MEMORY_REJECTION_HISTORY
        metadata = dict(entry.metadata)
        assert metadata["rejection_count"] == "3"

    def test_record_survivorship_trajectory(self):
        mem = make_memory()
        entry = mem.record_survivorship_trajectory(FAM_A, [0.8, 0.7, 0.6], timestamp=TS)
        assert entry.memory_type == MEMORY_SURVIVORSHIP_TRAJECTORY
        metadata = dict(entry.metadata)
        assert metadata["step_count"] == "3"

    def test_record_topology_evolution(self):
        mem = make_memory()
        entry = mem.record_topology_evolution(GH_A, [0.2, 0.5, 0.8], timestamp=TS)
        assert entry.memory_type == MEMORY_TOPOLOGY_EVOLUTION
        metadata = dict(entry.metadata)
        assert metadata["step_count"] == "3"

    def test_has_explored_true(self):
        mem = make_memory()
        mem.record_exploration_path(FAM_A, ["hyp_a"], timestamp=TS)
        assert mem.has_explored(FAM_A)

    def test_has_explored_false(self):
        mem = make_memory()
        assert not mem.has_explored(FAM_A)

    def test_is_known_failure_true(self):
        mem = make_memory()
        mem.record_rejection_history(FAM_A, 3, ["density"], timestamp=TS)
        assert mem.is_known_failure(FAM_A)

    def test_is_known_failure_false(self):
        mem = make_memory()
        assert not mem.is_known_failure(FAM_A)

    def test_known_failures_sorted(self):
        mem = make_memory()
        mem.record_rejection_history(FAM_B, 2, ["x"], timestamp=TS)
        mem.record_rejection_history(FAM_A, 1, ["y"], timestamp=TS2)
        failures = mem.known_failures()
        assert failures == sorted(failures)

    def test_exploration_path_sorted_by_timestamp(self):
        mem = make_memory()
        mem.record_exploration_path(FAM_A, ["hyp_b"], timestamp=TS2)
        mem.record_rejection_history(FAM_A, 1, ["x"], timestamp=TS)
        entries = mem.exploration_path(FAM_A)
        ts = [e.recorded_at for e in entries]
        assert ts == sorted(ts)

    def test_entries_of_type(self):
        mem = make_memory()
        mem.record_exploration_path(FAM_A, ["hyp_a"], timestamp=TS)
        mem.record_rejection_history(FAM_A, 1, ["x"], timestamp=TS2)
        mem.record_exploration_path(FAM_B, ["hyp_b"], timestamp=TS3)
        explored = mem.entries_of_type(MEMORY_EXPLORATION_PATH)
        assert len(explored) == 2
        assert all(e.memory_type == MEMORY_EXPLORATION_PATH for e in explored)

    def test_all_entries_sorted_by_timestamp(self):
        mem = make_memory()
        mem.record_exploration_path(FAM_A, ["hyp_b"], timestamp=TS2)
        mem.record_exploration_path(FAM_A, ["hyp_a"], timestamp=TS)
        entries = mem.all_entries()
        ts = [e.recorded_at for e in entries]
        assert ts == sorted(ts)

    def test_memory_hash_deterministic(self):
        m1 = make_memory("mem_x")
        m2 = make_memory("mem_x")
        m1.record_exploration_path(FAM_A, ["hyp_a"], timestamp=TS)
        m2.record_exploration_path(FAM_A, ["hyp_a"], timestamp=TS)
        assert m1.memory_hash() == m2.memory_hash()

    def test_memory_hash_changes_with_entries(self):
        mem = make_memory()
        h1 = mem.memory_hash()
        mem.record_exploration_path(FAM_A, ["hyp_a"], timestamp=TS)
        h2 = mem.memory_hash()
        assert h1 != h2

    def test_memory_hash_16_chars(self):
        mem = make_memory()
        assert len(mem.memory_hash()) == 16

    def test_entry_count(self):
        mem = make_memory()
        mem.record_exploration_path(FAM_A, ["h1"], timestamp=TS)
        mem.record_rejection_history(FAM_B, 2, ["r"], timestamp=TS2)
        assert mem.entry_count() == 2

    def test_content_hash_deterministic(self):
        m1 = make_memory()
        m2 = make_memory()
        e1 = m1.record_exploration_path(FAM_A, ["hyp_a", "hyp_b"], timestamp=TS)
        e2 = m2.record_exploration_path(FAM_A, ["hyp_b", "hyp_a"], timestamp=TS)  # different order
        # Content should be sorted before hashing
        assert e1.content_hash == e2.content_hash

    def test_to_manifest_keys(self):
        mem = make_memory()
        mem.record_exploration_path(FAM_A, ["h1"], timestamp=TS)
        m = mem.to_manifest()
        assert "memory_id" in m
        assert "entry_count" in m
        assert "known_failure_count" in m
        assert "memory_hash" in m
        assert "entries" in m

    def test_rejection_history_deduplicates_reasons(self):
        mem = make_memory()
        entry = mem.record_rejection_history(FAM_A, 5, ["x", "x", "y"], timestamp=TS)
        metadata = dict(entry.metadata)
        assert metadata["reason_count"] == "2"  # deduplicated: x, y

    def test_survivorship_trajectory_latest_rate_in_metadata(self):
        mem = make_memory()
        entry = mem.record_survivorship_trajectory(FAM_A, [0.9, 0.8, 0.75], timestamp=TS)
        metadata = dict(entry.metadata)
        assert metadata["latest_rate"] == str(round(0.75, 6))

    def test_topology_evolution_latest_fragility_in_metadata(self):
        mem = make_memory()
        entry = mem.record_topology_evolution(GH_A, [0.1, 0.3, 0.9], timestamp=TS)
        metadata = dict(entry.metadata)
        assert metadata["latest_fragility"] == str(round(0.9, 6))

    def test_multiple_families_indexed_separately(self):
        mem = make_memory()
        mem.record_exploration_path(FAM_A, ["h_a"], timestamp=TS)
        mem.record_exploration_path(FAM_B, ["h_b"], timestamp=TS2)
        assert mem.has_explored(FAM_A)
        assert mem.has_explored(FAM_B)
        assert len(mem.exploration_path(FAM_A)) == 1
        assert len(mem.exploration_path(FAM_B)) == 1
