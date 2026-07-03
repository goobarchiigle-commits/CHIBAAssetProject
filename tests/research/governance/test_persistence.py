"""Tests: persistence.py — FamilyPersistenceTracker."""
from __future__ import annotations

import sys
import math
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.governance.persistence import (
    FamilyPersistenceTracker, PersistenceRecord,
)
from tests.research.governance.conftest import (
    make_persistence, FAM_A, FAM_B, TS, TS2, TS3
)


class TestPersistenceRecord:
    def test_frozen(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        pt.record_observation(FAM_A, survived=True, timestamp=TS2)
        pt.record_observation(FAM_A, survived=True, timestamp=TS3)
        rec = pt.compute_persistence(FAM_A)
        with pytest.raises(Exception):
            rec.persistence_rate = 0.0  # type: ignore

    def test_record_id_prefix(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        rec = pt.compute_persistence(FAM_A)
        assert rec.record_id.startswith("pers_")

    def test_to_dict_from_dict_roundtrip(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        pt.record_observation(FAM_A, survived=False, timestamp=TS2)
        pt.record_observation(FAM_A, survived=True, timestamp=TS3)
        rec = pt.compute_persistence(FAM_A)
        restored = PersistenceRecord.from_dict(rec.to_dict())
        assert restored.persistence_rate == rec.persistence_rate
        assert restored.is_enduring_core == rec.is_enduring_core


class TestFamilyPersistenceTracker:
    def test_compute_persistence_no_observations_raises(self):
        pt = make_persistence()
        with pytest.raises(KeyError):
            pt.compute_persistence(FAM_A)

    def test_persistence_rate_all_survived(self):
        pt = make_persistence()
        for _ in range(5):
            pt.record_observation(FAM_A, survived=True, timestamp=TS)
        rec = pt.compute_persistence(FAM_A)
        assert rec.persistence_rate == pytest.approx(1.0, abs=1e-6)

    def test_persistence_rate_none_survived(self):
        pt = make_persistence()
        for _ in range(3):
            pt.record_observation(FAM_A, survived=False, timestamp=TS)
        rec = pt.compute_persistence(FAM_A)
        assert rec.persistence_rate == pytest.approx(0.0, abs=1e-6)

    def test_persistence_rate_mixed(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        pt.record_observation(FAM_A, survived=True, timestamp=TS2)
        pt.record_observation(FAM_A, survived=False, timestamp=TS3)
        rec = pt.compute_persistence(FAM_A)
        assert rec.persistence_rate == pytest.approx(2/3, abs=1e-5)

    def test_half_life_infinite_for_all_survived(self):
        pt = make_persistence()
        for _ in range(3):
            pt.record_observation(FAM_A, survived=True, timestamp=TS)
        rec = pt.compute_persistence(FAM_A)
        assert rec.half_life_estimate == -1.0  # sentinel for infinity

    def test_half_life_zero_for_none_survived(self):
        pt = make_persistence()
        for _ in range(3):
            pt.record_observation(FAM_A, survived=False, timestamp=TS)
        rec = pt.compute_persistence(FAM_A)
        assert rec.half_life_estimate == pytest.approx(0.0, abs=1e-9)

    def test_half_life_geometric_model(self):
        # persistence_rate = 0.5 → half_life = 1.0 (ln2 / -ln0.5 = 1.0)
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        pt.record_observation(FAM_A, survived=False, timestamp=TS2)
        rec = pt.compute_persistence(FAM_A)
        # rate = 0.5 → hl = ln(2)/(-ln(0.5)) = 1.0
        assert rec.half_life_estimate == pytest.approx(1.0, abs=0.01)

    def test_mutation_survivorship_no_mutations(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        rec = pt.compute_persistence(FAM_A)
        assert rec.mutation_survivorship == pytest.approx(1.0, abs=1e-9)

    def test_mutation_survivorship_with_mutations(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, mutation_id="mut_a", timestamp=TS)
        pt.record_observation(FAM_A, survived=False, mutation_id="mut_b", timestamp=TS2)
        rec = pt.compute_persistence(FAM_A)
        assert rec.mutation_survivorship == pytest.approx(0.5, abs=1e-6)

    def test_regime_resilience_no_regimes(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        rec = pt.compute_persistence(FAM_A)
        assert rec.regime_resilience == pytest.approx(1.0, abs=1e-9)

    def test_regime_resilience_two_regimes_both_survived(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, regime="trend", timestamp=TS)
        pt.record_observation(FAM_A, survived=True, regime="range", timestamp=TS2)
        rec = pt.compute_persistence(FAM_A)
        assert rec.regime_resilience == pytest.approx(1.0, abs=1e-9)

    def test_regime_resilience_partial(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, regime="trend", timestamp=TS)
        pt.record_observation(FAM_A, survived=False, regime="range", timestamp=TS2)
        rec = pt.compute_persistence(FAM_A)
        # survived in 1/2 regimes
        assert rec.regime_resilience == pytest.approx(0.5, abs=1e-6)

    def test_is_enduring_core_true(self):
        pt = make_persistence()
        # Need: rate >= 0.60, mutation_survivorship >= 0.50, count >= 3
        for _ in range(4):
            pt.record_observation(FAM_A, survived=True, mutation_id="m", timestamp=TS)
        rec = pt.compute_persistence(FAM_A)
        assert rec.is_enduring_core is True

    def test_is_enduring_core_false_low_count(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        pt.record_observation(FAM_A, survived=True, timestamp=TS2)  # only 2 obs < 3
        rec = pt.compute_persistence(FAM_A)
        assert rec.is_enduring_core is False

    def test_is_enduring_core_false_low_rate(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=False, timestamp=TS)
        pt.record_observation(FAM_A, survived=False, timestamp=TS2)
        pt.record_observation(FAM_A, survived=True, timestamp=TS3)
        rec = pt.compute_persistence(FAM_A)
        assert rec.is_enduring_core is False  # 1/3 < 0.60

    def test_enduring_cores_list(self):
        pt = make_persistence()
        for _ in range(4):
            pt.record_observation(FAM_A, survived=True, timestamp=TS)
        for _ in range(3):
            pt.record_observation(FAM_B, survived=False, timestamp=TS)
        cores = pt.enduring_cores()
        assert FAM_A in cores
        assert FAM_B not in cores

    def test_enduring_cores_sorted(self):
        pt = make_persistence()
        for fid in [FAM_B, FAM_A]:
            for _ in range(4):
                pt.record_observation(fid, survived=True, timestamp=TS)
        cores = pt.enduring_cores()
        assert cores == sorted(cores)

    def test_distinct_regimes(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, regime="trend", timestamp=TS)
        pt.record_observation(FAM_A, survived=False, regime="range", timestamp=TS2)
        pt.record_observation(FAM_A, survived=True, regime="trend", timestamp=TS3)  # duplicate
        regs = pt.distinct_regimes(FAM_A)
        assert regs == ["range", "trend"]

    def test_decay_trajectories_correct(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        pt.record_observation(FAM_A, survived=False, timestamp=TS2)
        trajs = pt.decay_trajectories()
        assert FAM_A in trajs
        # After step 1: rate=1.0, after step 2: rate=0.5
        assert trajs[FAM_A][0] == pytest.approx(1.0, abs=1e-6)
        assert trajs[FAM_A][1] == pytest.approx(0.5, abs=1e-6)

    def test_tracked_families_sorted(self):
        pt = make_persistence()
        pt.record_observation(FAM_B, survived=True, timestamp=TS)
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        families = pt.tracked_families()
        assert families == sorted(families)

    def test_persistence_rate_zero_unknown(self):
        pt = make_persistence()
        assert pt.persistence_rate("unknown_fam") == 0.0

    def test_first_seen_last_seen(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        pt.record_observation(FAM_A, survived=True, timestamp=TS3)
        rec = pt.compute_persistence(FAM_A)
        assert rec.first_seen == TS
        assert rec.last_seen == TS3

    def test_all_records(self):
        pt = make_persistence()
        pt.record_observation(FAM_A, survived=True, timestamp=TS)
        pt.record_observation(FAM_B, survived=False, timestamp=TS)
        records = pt.all_records()
        assert len(records) == 2
        family_ids = [r.family_id for r in records]
        assert family_ids == sorted(family_ids)

    def test_to_manifest_keys(self):
        pt = make_persistence()
        for _ in range(4):
            pt.record_observation(FAM_A, survived=True, timestamp=TS)
        m = pt.to_manifest()
        assert "family_count" in m
        assert "enduring_core_count" in m
        assert "enduring_cores" in m
        assert "records" in m
