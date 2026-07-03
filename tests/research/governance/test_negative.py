"""Tests: negative.py — NegativeKnowledgeRegistry."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.governance.negative import (
    NegativeKnowledgeRegistry, NegativeKnowledgeRecord,
    FAILURE_DEAD_TOPOLOGY, FAILURE_UNSTABLE_SURFACE, FAILURE_REGIME_FRAGILE,
    FAILURE_EXCESSIVE_TURNOVER, FAILURE_COLLAPSE_PRONE, FAILURE_PERSISTENT_REJECTION,
    ALL_FAILURE_TYPES,
)
from tests.research.governance.conftest import (
    make_negative, FAM_A, FAM_B, GH_A, GH_B, TS, TS2, TS3
)


class TestNegativeKnowledgeRecord:
    def test_frozen(self):
        neg = make_negative()
        rec = neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        with pytest.raises(Exception):
            rec.failure_count = 99  # type: ignore

    def test_record_id_prefix(self):
        neg = make_negative()
        rec = neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        assert rec.record_id.startswith("nkr_")

    def test_to_dict_from_dict_roundtrip(self):
        neg = make_negative()
        rec = neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY,
                                    metadata={"score": 0.9}, timestamp=TS)
        restored = NegativeKnowledgeRecord.from_dict(rec.to_dict())
        assert restored == rec

    def test_failure_signature_nonempty(self):
        neg = make_negative()
        rec = neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        assert len(rec.failure_signature) == 16


class TestNegativeKnowledgeRegistry:
    def test_all_failure_types_nonempty(self):
        assert len(ALL_FAILURE_TYPES) == 6

    def test_invalid_failure_type_raises(self):
        neg = make_negative()
        with pytest.raises(ValueError, match="Invalid failure_type"):
            neg.register_failure(FAM_A, GH_A, "TOTALLY_INVALID")

    def test_register_first_failure(self):
        neg = make_negative()
        rec = neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        assert rec.failure_count == 1
        assert rec.first_seen == TS
        assert rec.last_seen == TS

    def test_register_repeated_failure_increments_count(self):
        neg = make_negative()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS2)
        rec = neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS3)
        assert rec.failure_count == 3
        assert rec.first_seen == TS
        assert rec.last_seen == TS3

    def test_record_id_stable_on_update(self):
        neg = make_negative()
        r1 = neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        r2 = neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS2)
        assert r1.record_id == r2.record_id

    def test_different_failure_types_separate_records(self):
        neg = make_negative()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        neg.register_failure(FAM_A, GH_A, FAILURE_REGIME_FRAGILE, timestamp=TS)
        assert neg.record_count() == 2

    def test_has_failure_any_type(self):
        neg = make_negative()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        assert neg.has_failure(FAM_A)
        assert not neg.has_failure(FAM_B)

    def test_has_failure_specific_type(self):
        neg = make_negative()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        assert neg.has_failure(FAM_A, FAILURE_DEAD_TOPOLOGY)
        assert not neg.has_failure(FAM_A, FAILURE_REGIME_FRAGILE)

    def test_failure_count_aggregates_all_types(self):
        neg = make_negative()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS2)
        neg.register_failure(FAM_A, GH_A, FAILURE_REGIME_FRAGILE, timestamp=TS)
        # count=2 for DEAD_TOPOLOGY + count=1 for REGIME_FRAGILE = 3
        assert neg.failure_count(FAM_A) == 3

    def test_failure_count_zero_unknown_family(self):
        neg = make_negative()
        assert neg.failure_count("fam_unknown") == 0

    def test_failure_types_for_family_sorted(self):
        neg = make_negative()
        neg.register_failure(FAM_A, GH_A, FAILURE_REGIME_FRAGILE, timestamp=TS)
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        types = neg.failure_types_for_family(FAM_A)
        assert types == sorted(types)

    def test_failure_signature_empty_for_unknown(self):
        neg = make_negative()
        assert neg.failure_signature("unknown_family") == ""

    def test_failure_signature_nonempty_after_failure(self):
        neg = make_negative()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        assert len(neg.failure_signature(FAM_A)) == 16

    def test_failure_signature_deterministic(self):
        n1 = make_negative()
        n1.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        n2 = make_negative()
        n2.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        assert n1.failure_signature(FAM_A) == n2.failure_signature(FAM_A)

    def test_persistent_failures(self):
        neg = make_negative()
        for _ in range(3):
            neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        neg.register_failure(FAM_B, GH_B, FAILURE_REGIME_FRAGILE, timestamp=TS)
        persistent = neg.persistent_failures(min_count=3)
        assert len(persistent) == 1
        assert persistent[0].family_id == FAM_A

    def test_persistent_failures_sorted_by_count_desc(self):
        neg = make_negative()
        for _ in range(5):
            neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        for _ in range(3):
            neg.register_failure(FAM_B, GH_B, FAILURE_REGIME_FRAGILE, timestamp=TS)
        pf = neg.persistent_failures(min_count=2)
        counts = [r.failure_count for r in pf]
        assert counts == sorted(counts, reverse=True)

    def test_all_records_sorted_by_record_id(self):
        neg = make_negative()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        neg.register_failure(FAM_B, GH_B, FAILURE_REGIME_FRAGILE, timestamp=TS)
        records = neg.all_records()
        ids = [r.record_id for r in records]
        assert ids == sorted(ids)

    def test_affected_families_sorted(self):
        neg = make_negative()
        neg.register_failure(FAM_B, GH_B, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        neg.register_failure(FAM_A, GH_A, FAILURE_REGIME_FRAGILE, timestamp=TS)
        families = neg.affected_families()
        assert families == sorted(families)

    def test_to_manifest_keys(self):
        neg = make_negative()
        neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY, timestamp=TS)
        m = neg.to_manifest()
        assert "record_count" in m
        assert "records" in m
        assert "affected_family_count" in m

    def test_metadata_stored(self):
        neg = make_negative()
        rec = neg.register_failure(FAM_A, GH_A, FAILURE_DEAD_TOPOLOGY,
                                    metadata={"fragility": 0.9}, timestamp=TS)
        assert any(k == "fragility" for k, _ in rec.metadata)
