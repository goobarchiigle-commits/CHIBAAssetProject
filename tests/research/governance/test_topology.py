"""Tests: topology.py — TopologyFailureRegistry."""
from __future__ import annotations

import sys
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.governance.topology import (
    TopologyFailureRegistry, TopologyFailureRecord,
    TOPOLOGY_NARROW_PLATEAU, TOPOLOGY_CLIFF_EDGE, TOPOLOGY_UNSTABLE_NEIGHBORHOOD,
    TOPOLOGY_PATHOLOGICAL_TURNOVER, TOPOLOGY_UNSTABLE_REGIME_TRANSITION,
    ALL_TOPOLOGY_TYPES,
)
from tests.research.governance.conftest import (
    make_topology, FAM_A, FAM_B, GH_A, GH_B, TS, TS2, TS3
)


class TestTopologyFailureRecord:
    def test_frozen(self):
        topo = make_topology()
        rec = topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        with pytest.raises(Exception):
            rec.fragility_score = 0.0  # type: ignore

    def test_record_id_prefix(self):
        topo = make_topology()
        rec = topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        assert rec.record_id.startswith("topo_")

    def test_to_dict_from_dict_roundtrip(self):
        topo = make_topology()
        rec = topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.8,
                                              metadata={"reason": "sharp_drop"}, timestamp=TS)
        restored = TopologyFailureRecord.from_dict(rec.to_dict())
        assert restored == rec

    def test_collapse_signature_length(self):
        topo = make_topology()
        rec = topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        assert len(rec.collapse_propagation_signature) == 16


class TestTopologyFailureRegistry:
    def test_all_topology_types_nonempty(self):
        assert len(ALL_TOPOLOGY_TYPES) == 5

    def test_invalid_topology_type_raises(self):
        topo = make_topology()
        with pytest.raises(ValueError, match="Invalid topology_type"):
            topo.register_topology_failure(GH_A, FAM_A, "INVALID_TYPE", 0.5)

    def test_fragility_score_out_of_range_raises(self):
        topo = make_topology()
        with pytest.raises(ValueError):
            topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 1.5)
        with pytest.raises(ValueError):
            topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, -0.1)

    def test_register_first_failure(self):
        topo = make_topology()
        rec = topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        assert rec.recurrence_count == 1
        assert rec.fragility_score == pytest.approx(0.9, abs=1e-6)
        assert rec.first_seen == TS

    def test_register_repeated_increments_count(self):
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.8, timestamp=TS)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS2)
        rec = topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.7, timestamp=TS3)
        assert rec.recurrence_count == 3
        assert rec.first_seen == TS
        assert rec.last_seen == TS3

    def test_record_id_stable_on_update(self):
        topo = make_topology()
        r1 = topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.8, timestamp=TS)
        r2 = topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS2)
        assert r1.record_id == r2.record_id

    def test_mean_fragility_computed(self):
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.8, timestamp=TS)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.6, timestamp=TS2)
        # mean = (0.8 + 0.6) / 2 = 0.7
        rec = topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.6, timestamp=TS3)
        frag = topo.mean_fragility(GH_A)
        assert frag == pytest.approx((0.8 + 0.6 + 0.6) / 3, abs=1e-4)

    def test_mean_fragility_zero_unknown(self):
        topo = make_topology()
        assert topo.mean_fragility("unknown_gh") == 0.0

    def test_has_topology_failure_any(self):
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        assert topo.has_topology_failure(GH_A)
        assert not topo.has_topology_failure(GH_B)

    def test_has_topology_failure_specific_type(self):
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        assert topo.has_topology_failure(GH_A, TOPOLOGY_CLIFF_EDGE)
        assert not topo.has_topology_failure(GH_A, TOPOLOGY_NARROW_PLATEAU)

    def test_recurrence_count_sums_across_types(self):
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.8, timestamp=TS)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.8, timestamp=TS2)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_NARROW_PLATEAU, 0.5, timestamp=TS)
        assert topo.recurrence_count(GH_A) == 3

    def test_recurrence_count_zero_unknown(self):
        topo = make_topology()
        assert topo.recurrence_count("unknown") == 0

    def test_failure_types_for_grammar_sorted(self):
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_NARROW_PLATEAU, 0.5, timestamp=TS)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.8, timestamp=TS)
        types = topo.failure_types_for_grammar(GH_A)
        assert types == sorted(types)

    def test_persistent_failures(self):
        topo = make_topology()
        for _ in range(3):
            topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        topo.register_topology_failure(GH_B, FAM_B, TOPOLOGY_NARROW_PLATEAU, 0.5, timestamp=TS)
        pf = topo.persistent_failures(min_recurrence=2)
        assert len(pf) == 1
        assert pf[0].grammar_hash == GH_A

    def test_persistent_failures_sorted_by_recurrence_desc(self):
        topo = make_topology()
        for _ in range(5):
            topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        for _ in range(2):
            topo.register_topology_failure(GH_B, FAM_B, TOPOLOGY_NARROW_PLATEAU, 0.5, timestamp=TS)
        pf = topo.persistent_failures(min_recurrence=2)
        counts = [r.recurrence_count for r in pf]
        assert counts == sorted(counts, reverse=True)

    def test_all_records_sorted_by_record_id(self):
        topo = make_topology()
        topo.register_topology_failure(GH_B, FAM_B, TOPOLOGY_NARROW_PLATEAU, 0.5, timestamp=TS)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        records = topo.all_records()
        ids = [r.record_id for r in records]
        assert ids == sorted(ids)

    def test_affected_grammar_hashes_sorted(self):
        topo = make_topology()
        topo.register_topology_failure(GH_B, FAM_B, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_NARROW_PLATEAU, 0.5, timestamp=TS)
        hashes = topo.affected_grammar_hashes()
        assert hashes == sorted(hashes)

    def test_to_manifest_keys(self):
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.9, timestamp=TS)
        m = topo.to_manifest()
        assert "record_count" in m
        assert "records" in m
        assert "affected_grammar_count" in m

    def test_different_types_same_grammar_different_records(self):
        topo = make_topology()
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_CLIFF_EDGE, 0.8, timestamp=TS)
        topo.register_topology_failure(GH_A, FAM_A, TOPOLOGY_NARROW_PLATEAU, 0.6, timestamp=TS)
        assert topo.record_count() == 2
