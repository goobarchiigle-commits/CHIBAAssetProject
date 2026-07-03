"""Tests: artifacts.py — ExecutionRealityArtifacts."""
from __future__ import annotations

import sys
import json
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.execution.reality.artifacts import ExecutionRealityArtifacts
from src.research.analysis.artifacts import ImmutabilityError


@pytest.fixture
def artifacts(tmp_path):
    return ExecutionRealityArtifacts(base_path=tmp_path / "execution_reality")


class TestExecutionRealityArtifacts:
    def test_write_snapshot(self, artifacts):
        artifacts.write_snapshot("exsnap_abc123", {"key": "val"})
        p = artifacts._base / "snapshots" / "exsnap_abc123.json"
        assert p.exists()

    def test_write_snapshot_immutable(self, artifacts):
        artifacts.write_snapshot("exsnap_abc123", {"key": "val"})
        with pytest.raises(ImmutabilityError):
            artifacts.write_snapshot("exsnap_abc123", {"key": "other"})

    def test_write_snapshot_index(self, artifacts):
        artifacts.write_snapshot_index("sess_001", {"count": 1})
        p = artifacts._base / "snapshots" / "index_sess_001.json"
        assert p.exists()

    def test_append_divergence_record(self, artifacts):
        artifacts.append_divergence_record("sess_001", {"type": "SLIPPAGE_DRIFT"})
        artifacts.append_divergence_record("sess_001", {"type": "LATENCY_DRIFT"})
        records = artifacts.read_divergence_records("sess_001")
        assert len(records) == 2

    def test_read_divergence_records_absent(self, artifacts):
        records = artifacts.read_divergence_records("no_session")
        assert records == []

    def test_write_divergence_summary(self, artifacts):
        artifacts.write_divergence_summary("sess_001", {"total": 5})
        p = artifacts._base / "divergence" / "summary_sess_001.json"
        assert p.exists()

    def test_write_regime_classification(self, artifacts):
        artifacts.write_regime_classification("rgm_xyz", {"regime": "NORMAL"})
        p = artifacts._base / "regime" / "rgm_xyz.json"
        assert p.exists()

    def test_write_slippage_summary(self, artifacts):
        artifacts.write_slippage_summary("sess_001", {"mean": 5.0})
        p = artifacts._base / "slippage" / "summary_sess_001.json"
        assert p.exists()

    def test_append_slippage_observation(self, artifacts):
        artifacts.append_slippage_observation("sess_001", {"bps": 3.0})
        artifacts.append_slippage_observation("sess_001", {"bps": 5.0})
        p = artifacts._base / "slippage" / "sess_001.jsonl"
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_write_fill_quality_summary(self, artifacts):
        artifacts.write_fill_quality_summary("sess_001", {"fill_ratio": 0.95})
        p = artifacts._base / "fill_quality" / "summary_sess_001.json"
        assert p.exists()

    def test_write_latency_summary(self, artifacts):
        artifacts.write_latency_summary("sess_001", {"mean_ns": 20_000_000})
        p = artifacts._base / "latency" / "summary_sess_001.json"
        assert p.exists()

    def test_write_liquidity_summary(self, artifacts):
        artifacts.write_liquidity_summary("sess_001", {"mean_spread": 10.0})
        p = artifacts._base / "liquidity" / "summary_sess_001.json"
        assert p.exists()

    def test_write_comparison_summary(self, artifacts):
        artifacts.write_comparison_summary("sess_001", {"divergence": 0.1})
        p = artifacts._base / "comparator" / "summary_sess_001.json"
        assert p.exists()

    def test_write_decay_summary(self, artifacts):
        artifacts.write_decay_summary("sess_001", {"decay": 20.0})
        p = artifacts._base / "decay" / "summary_sess_001.json"
        assert p.exists()

    def test_write_decision(self, artifacts):
        artifacts.write_decision("dec_abc123", {"decision": "ALLOW"})
        p = artifacts._base / "decisions" / "dec_abc123.json"
        assert p.exists()

    def test_append_and_read_decisions(self, artifacts):
        artifacts.append_decision("sess_001", {"decision": "ALLOW"})
        artifacts.append_decision("sess_001", {"decision": "REDUCE"})
        decisions = artifacts.read_decisions("sess_001")
        assert len(decisions) == 2

    def test_read_decisions_absent(self, artifacts):
        assert artifacts.read_decisions("no_session") == []

    def test_write_replay_result(self, artifacts):
        artifacts.write_replay_result("rply_xyz", {"scenario": "test"})
        p = artifacts._base / "replay" / "rply_xyz.json"
        assert p.exists()

    def test_write_replay_summary(self, artifacts):
        artifacts.write_replay_summary("sess_001", {"count": 3})
        p = artifacts._base / "replay" / "summary_sess_001.json"
        assert p.exists()

    def test_subdirs_created_on_demand(self, artifacts, tmp_path):
        artifacts.write_slippage_summary("sess_999", {"x": 1})
        assert (artifacts._base / "slippage").exists()
