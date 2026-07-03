"""Tests: survivorship.py — SurvivorshipLedger append-only, checksum, immutability."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.analysis.artifacts import ImmutabilityError
from src.research.analysis.contract import SCHEMA_VERSION
from src.research.analysis.survivorship import SurvivorshipLedger
from tests.research.analysis.conftest import make_record


def _build_records(n=3):
    return [make_record(f"e{i}", strategy_id="strat_x", survived=(i % 2 == 0)) for i in range(n)]


class TestSurvivorshipLedger:
    def test_record_creates_artifact(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        records = _build_records(3)
        entry = ledger.record(records=records, families=[])
        artifact_path = tmp_path / "surv" / f"{entry.ledger_id}.json"
        assert artifact_path.exists()

    def test_record_returns_valid_entry(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        records = _build_records(4)
        entry = ledger.record(records=records, families=[])
        assert entry.schema_version == SCHEMA_VERSION
        assert entry.experiment_count == 4
        assert entry.survivor_count == 2  # e0, e2 survived
        assert len(entry.checksum) == 64

    def test_checksum_validates(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        records = _build_records(2)
        entry = ledger.record(records=records, families=[])
        assert ledger.verify(entry.ledger_id) is True

    def test_missing_artifact_verify_false(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        assert ledger.verify("nonexistent_id") is False

    def test_manifest_appended(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        e1 = ledger.record(records=_build_records(1), families=[])
        e2 = ledger.record(records=_build_records(2), families=[])
        entries = ledger.list_entries()
        ids = [e["ledger_id"] for e in entries]
        assert e1.ledger_id in ids
        assert e2.ledger_id in ids

    def test_list_entries_sorted_by_time(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        e1 = ledger.record(records=_build_records(1), families=[])
        # Different record set so payload differs → different ledger_id
        e2 = ledger.record(records=_build_records(2), families=[])
        entries = ledger.list_entries()
        timestamps = [e["generated_at"] for e in entries]
        assert timestamps == sorted(timestamps)

    def test_load_roundtrip(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        records = _build_records(3)
        entry = ledger.record(records=records, families=[])
        loaded = ledger.load(entry.ledger_id)
        assert loaded.ledger_id == entry.ledger_id
        assert loaded.experiment_count == entry.experiment_count
        assert loaded.checksum == entry.checksum

    def test_load_validates_checksum(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        entry = ledger.record(records=_build_records(1), families=[])
        # Corrupt the artifact
        artifact = tmp_path / "surv" / f"{entry.ledger_id}.json"
        data = json.loads(artifact.read_text(encoding="utf-8"))
        data["checksum"] = "corrupted" * 7
        artifact.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(ValueError, match="Checksum mismatch"):
            ledger.load(entry.ledger_id)

    def test_artifact_immutable_on_collision(self, tmp_path, monkeypatch):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        records = _build_records(2)
        entry = ledger.record(records=records, families=[])
        # Force same ledger_id to test immutability guard
        from src.research.analysis import survivorship as smod
        monkeypatch.setattr(smod.SurvivorshipLedger, "_make_ledger_id", staticmethod(lambda *a: entry.ledger_id))
        with pytest.raises(ImmutabilityError):
            ledger.record(records=records, families=[])

    def test_verify_all(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        e1 = ledger.record(records=_build_records(1), families=[])
        e2 = ledger.record(records=_build_records(2), families=[])
        results = ledger.verify_all()
        assert results[e1.ledger_id] is True
        assert results[e2.ledger_id] is True

    def test_payload_records_sorted(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        records = [
            make_record("zzz", strategy_id="s"),
            make_record("aaa", strategy_id="s"),
        ]
        entry = ledger.record(records=records, families=[])
        record_ids = [r["experiment_id"] for r in entry.payload["records"]]
        assert record_ids == sorted(record_ids)

    def test_topology_included_in_payload(self, tmp_path):
        from src.research.analysis.contract import TopologyResult
        ledger = SurvivorshipLedger(tmp_path / "surv")
        records = _build_records(1)
        topo = {
            "e0": TopologyResult(
                experiment_id="e0", topology_fragility_score=0.3,
                plateau_width=0.7, neighbor_decay_rate=0.1,
                parameter_elasticity=0.2, surface_entropy=0.5,
                neighborhood_survivorship=1.0, neighbor_count=2,
            )
        }
        entry = ledger.record(records=records, families=[], topology=topo)
        assert "topology" in entry.payload
        assert "e0" in entry.payload["topology"]

    def test_entry_ledger_id_prefix(self, tmp_path):
        ledger = SurvivorshipLedger(tmp_path / "surv")
        entry = ledger.record(records=_build_records(1), families=[])
        assert entry.ledger_id.startswith("surv_")
