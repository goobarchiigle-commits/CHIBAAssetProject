"""Tests: forensic_summary.py — ResearchForensicsSummary determinism and structure."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.research.analysis.forensic_summary import ResearchForensicsSummary, _percentiles
from src.research.analysis.family_graph import ExperimentFamilyGraph
from src.research.analysis.feature_survivorship import FeatureObservation, FeatureSurvivorshipAnalyzer
from src.research.analysis.regime_resilience import RegimeResilienceAnalyzer
from src.research.analysis.topology import ParameterTopologyAnalyzer
from tests.research.analysis.conftest import make_record


def _make_population(n=8):
    records = []
    for i in range(n):
        survived = i < (n // 2)
        records.append(make_record(
            f"e{i}",
            strategy_id="strat_a" if i % 2 == 0 else "strat_b",
            feature_names=["f_mom", "f_vol"] if i % 3 != 0 else ["f_mom"],
            regime_scores={"trend_up": 1.0 if survived else -0.5, "range": 0.3},
            score=0.7 if survived else -0.2,
            survived=survived,
            rejection_reasons=[] if survived else ["sharpe=-0.5 < 0"],
        ))
    return records


class TestResearchForensicsSummary:
    def test_empty_records_returns_report(self):
        summary = ResearchForensicsSummary()
        report = summary.generate(records=[], families=[], feature_stats={})
        assert report["payload"]["experiment_counts"]["total"] == 0

    def test_report_has_required_keys(self):
        records = _make_population()
        families = ExperimentFamilyGraph().build(records)
        obs = [
            FeatureObservation(r.experiment_id, list(r.feature_names), r.survived, "trend_up", "fam_x")
            for r in records
        ]
        feature_stats = FeatureSurvivorshipAnalyzer().analyze(obs)
        report = ResearchForensicsSummary().generate(records, families, feature_stats)
        for key in [
            "summary_id", "generated_at", "schema_version", "checksum", "payload"
        ]:
            assert key in report

        for section in [
            "experiment_counts", "rejection_rates", "survivorship_distributions",
            "dominant_failure_modes", "feature_instability_clusters",
            "family_survivorship_distribution",
        ]:
            assert section in report["payload"]

    def test_deterministic_across_record_order(self):
        records = _make_population(6)
        families = ExperimentFamilyGraph().build(records)
        feature_stats = {}
        summary = ResearchForensicsSummary()
        r1 = summary.generate(records, families, feature_stats)
        r2 = summary.generate(list(reversed(records)), families, feature_stats)
        # checksum covers payload; same data → same checksum
        assert r1["checksum"] == r2["checksum"]

    def test_checksum_validates_payload(self):
        from src.research.analysis.artifacts import verify_checksum
        records = _make_population(4)
        report = ResearchForensicsSummary().generate(records, [], {})
        assert verify_checksum(report["payload"], report["checksum"])

    def test_experiment_counts_accurate(self):
        records = _make_population(6)
        report = ResearchForensicsSummary().generate(records, [], {})
        counts = report["payload"]["experiment_counts"]
        assert counts["total"] == 6
        assert counts["survived"] + counts["rejected"] == 6

    def test_rejection_rates_sum(self):
        records = _make_population(6)
        report = ResearchForensicsSummary().generate(records, [], {})
        rates = report["payload"]["rejection_rates"]
        total = report["payload"]["experiment_counts"]["total"]
        rejected = report["payload"]["experiment_counts"]["rejected"]
        assert abs(rates["overall_rate"] - rejected / total) < 1e-9

    def test_report_json_serializable(self):
        records = _make_population(4)
        families = ExperimentFamilyGraph().build(records)
        report = ResearchForensicsSummary().generate(records, families, {})
        json.dumps(report)  # must not raise

    def test_summary_id_prefix(self):
        records = _make_population(2)
        report = ResearchForensicsSummary().generate(records, [], {})
        assert report["summary_id"].startswith("forensic_")

    def test_feature_instability_clusters_with_stats(self):
        records = _make_population(4)
        obs = [
            FeatureObservation(r.experiment_id, list(r.feature_names), r.survived, "trend_up", "fam_1")
            for r in records
        ]
        feature_stats = FeatureSurvivorshipAnalyzer().analyze(obs)
        report = ResearchForensicsSummary().generate(records, [], feature_stats)
        cluster = report["payload"]["feature_instability_clusters"]
        assert cluster["available"] is True
        assert cluster["total_features"] > 0

    def test_parameter_fragility_without_topology(self):
        records = _make_population(2)
        report = ResearchForensicsSummary().generate(records, [], {}, topology=None)
        sig = report["payload"]["parameter_fragility_signatures"]
        assert sig["available"] is False

    def test_parameter_fragility_with_topology(self):
        records = _make_population(4)
        exps = [(r.experiment_id, {"p": i, "q": i * 2}, r.score, r.survived) for i, r in enumerate(records)]
        topology = ParameterTopologyAnalyzer().analyze(exps)
        report = ResearchForensicsSummary().generate(records, [], {}, topology=topology)
        sig = report["payload"]["parameter_fragility_signatures"]
        assert sig["available"] is True

    def test_regime_collapse_without_resilience(self):
        records = _make_population(2)
        report = ResearchForensicsSummary().generate(records, [], {}, regime_resilience=None)
        assert report["payload"]["regime_collapse_patterns"]["available"] is False

    def test_regime_collapse_with_resilience(self):
        records = _make_population(4)
        resilience = RegimeResilienceAnalyzer().analyze(records)
        report = ResearchForensicsSummary().generate(records, [], {}, regime_resilience=resilience)
        patterns = report["payload"]["regime_collapse_patterns"]
        assert patterns["available"] is True
        assert "cross_regime_retention" in patterns

    def test_family_distribution_counts(self):
        records = _make_population(6)
        families = ExperimentFamilyGraph().build(records)
        report = ResearchForensicsSummary().generate(records, families, {})
        dist = report["payload"]["family_survivorship_distribution"]
        assert dist["total_families"] == len(families)


class TestPercentiles:
    def test_single_value(self):
        p = _percentiles([0.5])
        assert p["min"] == 0.5
        assert p["max"] == 0.5
        assert p["p50"] == 0.5

    def test_empty(self):
        assert _percentiles([]) == {}

    def test_sorted_order(self):
        p = _percentiles([0.1, 0.5, 0.9])
        assert p["min"] <= p["p25"] <= p["p50"] <= p["p75"] <= p["max"]

    def test_count_field(self):
        assert _percentiles([1, 2, 3])["count"] == 3
