"""ResearchForensicsSummary — deterministic batch-level research interpretability.

Aggregates topology, feature survivorship, regime resilience, and family data
into a single interpretable summary explaining WHY strategies survive or fail.

No ML. No dashboards. Pure deterministic aggregation.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Optional, Tuple

from src.research.analysis.artifacts import compute_checksum, utc_now_iso
from src.research.analysis.contract import (
    SCHEMA_VERSION,
    FeatureSurvivalStats,
    FamilyRecord,
    RegimeResilienceResult,
    SurvivorRecord,
    TopologyResult,
)

# Fragility threshold: experiments with fragility >= this are "fragile"
_FRAGILE_THRESH = 0.70
# Isolated spike: no neighbors
_ISOLATED_SPIKE_THRESH = 0
# Dominant failure mode: rejection reason count >= this fraction of total
_DOMINANT_FAILURE_RATE = 0.10


class ResearchForensicsSummary:
    """Generate deterministic forensic summaries from research populations.

    Usage:
        summary = ResearchForensicsSummary()
        report = summary.generate(records, families, feature_stats, topology, regime_resilience)
    """

    def generate(
        self,
        records: List[SurvivorRecord],
        families: List[FamilyRecord],
        feature_stats: Dict[str, FeatureSurvivalStats],
        topology: Optional[Dict[str, TopologyResult]] = None,
        regime_resilience: Optional[RegimeResilienceResult] = None,
    ) -> dict:
        """Generate a deterministic forensic summary report.

        Returns:
            Serializable dict with full summary. Includes checksum.
        """
        generated_at = utc_now_iso()

        if not records:
            empty = self._empty_report(generated_at)
            empty["checksum"] = compute_checksum(empty["payload"])
            return empty

        sorted_records = sorted(records, key=lambda r: r.experiment_id)
        sorted_families = sorted(families, key=lambda f: f.family_id)

        payload = {
            "experiment_counts": self._experiment_counts(sorted_records),
            "rejection_rates": self._rejection_rates(sorted_records),
            "survivorship_distributions": self._survivorship_distributions(sorted_records, sorted_families),
            "dominant_failure_modes": self._dominant_failure_modes(sorted_records),
            "parameter_fragility_signatures": self._fragility_signatures(topology),
            "regime_collapse_patterns": self._regime_collapse_patterns(regime_resilience, sorted_records),
            "feature_instability_clusters": self._feature_instability_clusters(feature_stats),
            "family_survivorship_distribution": self._family_distribution(sorted_families),
            "dominant_fragility_classes": self._dominant_fragility_classes(topology, sorted_records),
            "decay_archetypes": self._decay_archetypes(sorted_records, sorted_families),
            "robustness_distributions": self._robustness_distributions(sorted_records, topology),
            "failure_concentration_metrics": self._failure_concentration(sorted_records),
        }

        checksum = compute_checksum(payload)

        return {
            "summary_id": self._make_summary_id(checksum, generated_at),
            "generated_at": generated_at,
            "schema_version": SCHEMA_VERSION,
            "checksum": checksum,
            "payload": payload,
        }

    # ── Section builders ───────────────────────────────────────────────────────

    @staticmethod
    def _experiment_counts(records: List[SurvivorRecord]) -> dict:
        total = len(records)
        survived = sum(1 for r in records if r.survived)
        strategies = sorted({r.strategy_id for r in records})
        return {
            "total": total,
            "survived": survived,
            "rejected": total - survived,
            "unique_strategies": len(strategies),
            "strategy_ids": strategies,
        }

    @staticmethod
    def _rejection_rates(records: List[SurvivorRecord]) -> dict:
        total = len(records)
        rejected = [r for r in records if not r.survived]
        if total == 0:
            return {"overall_rate": 0.0, "by_reason": {}}

        reason_counts: Dict[str, int] = {}
        for r in rejected:
            for reason in r.rejection_reasons:
                # Normalize reason to metric key (strip numeric values)
                key = reason.split("=")[0].strip() if "=" in reason else reason[:50]
                reason_counts[key] = reason_counts.get(key, 0) + 1

        by_reason = {
            k: {"count": v, "rate": round(v / total, 4)}
            for k, v in sorted(reason_counts.items(), key=lambda x: (-x[1], x[0]))
        }
        return {
            "overall_rate": round(len(rejected) / total, 4),
            "by_reason": by_reason,
        }

    @staticmethod
    def _survivorship_distributions(
        records: List[SurvivorRecord],
        families: List[FamilyRecord],
    ) -> dict:
        scores = sorted(r.score for r in records if r.survived)
        family_survivorship = sorted(f.family_survivorship for f in families)
        return {
            "experiment_score_percentiles": _percentiles(scores),
            "family_survivorship_percentiles": _percentiles(family_survivorship),
        }

    @staticmethod
    def _dominant_failure_modes(records: List[SurvivorRecord]) -> List[dict]:
        """Top failure modes sorted by frequency desc."""
        reason_counts: Dict[str, int] = {}
        total_rejected = 0
        for r in records:
            if not r.survived:
                total_rejected += 1
                for reason in r.rejection_reasons:
                    key = reason.split("=")[0].strip() if "=" in reason else reason[:50]
                    reason_counts[key] = reason_counts.get(key, 0) + 1

        if total_rejected == 0:
            return []

        modes = sorted(
            [
                {
                    "reason_class": k,
                    "count": v,
                    "fraction_of_rejected": round(v / total_rejected, 4),
                }
                for k, v in reason_counts.items()
            ],
            key=lambda x: (-x["count"], x["reason_class"]),
        )
        return modes

    @staticmethod
    def _fragility_signatures(
        topology: Optional[Dict[str, TopologyResult]],
    ) -> dict:
        if not topology:
            return {"available": False}

        topos = list(topology.values())
        fragility_scores = sorted(t.topology_fragility_score for t in topos)
        isolated = [t for t in topos if t.neighbor_count == 0]
        fragile = [t for t in topos if t.topology_fragility_score >= _FRAGILE_THRESH]
        stable = [t for t in topos if t.plateau_width >= 0.5]

        return {
            "available": True,
            "total_experiments": len(topos),
            "isolated_count": len(isolated),
            "fragile_count": len(fragile),
            "stable_plateau_count": len(stable),
            "fragility_score_percentiles": _percentiles(fragility_scores),
            "mean_neighborhood_survivorship": round(
                sum(t.neighborhood_survivorship for t in topos) / len(topos), 4
            ),
        }

    @staticmethod
    def _regime_collapse_patterns(
        regime_resilience: Optional[RegimeResilienceResult],
        records: List[SurvivorRecord],
    ) -> dict:
        if regime_resilience is None:
            return {"available": False}

        # Most fragile transitions
        fragility_pairs = list(regime_resilience.transition_fragility)
        top_collapses = sorted(fragility_pairs, key=lambda x: (-x[1], x[0]))[:5]

        return {
            "available": True,
            "cross_regime_retention": regime_resilience.cross_regime_retention,
            "regime_resilience_score": regime_resilience.regime_resilience_score,
            "regime_concentration": regime_resilience.regime_concentration,
            "recovery_stability": regime_resilience.recovery_stability,
            "single_regime_dependent": regime_resilience.single_regime_dependent,
            "panic_only_alpha": regime_resilience.panic_only_alpha,
            "trend_only_alpha": regime_resilience.trend_only_alpha,
            "vol_sensitive_alpha": regime_resilience.vol_sensitive_alpha,
            "top_collapse_transitions": [
                {"transition": k, "fragility": v} for k, v in top_collapses
            ],
        }

    @staticmethod
    def _feature_instability_clusters(
        feature_stats: Dict[str, FeatureSurvivalStats],
    ) -> dict:
        if not feature_stats:
            return {"available": False, "features": []}

        all_stats = sorted(feature_stats.values(), key=lambda s: s.feature_name)
        survival_rates = sorted(s.feature_survival_rate for s in all_stats)

        # Unstable features: low decay resistance
        unstable = sorted(
            [s for s in all_stats if s.feature_decay_resistance < 0.50],
            key=lambda s: (s.feature_decay_resistance, s.feature_name),
        )
        # High-survival features
        persistent = sorted(
            [s for s in all_stats if s.feature_survival_rate >= 0.70],
            key=lambda s: (-s.feature_survival_rate, s.feature_name),
        )

        return {
            "available": True,
            "total_features": len(all_stats),
            "unstable_feature_count": len(unstable),
            "persistent_feature_count": len(persistent),
            "survival_rate_percentiles": _percentiles(survival_rates),
            "top_persistent_features": [s.feature_name for s in persistent[:10]],
            "top_unstable_features": [s.feature_name for s in unstable[:10]],
        }

    @staticmethod
    def _family_distribution(families: List[FamilyRecord]) -> dict:
        if not families:
            return {"total_families": 0}

        sizes = sorted(len(f.member_experiment_ids) for f in families)
        survivorship = sorted(f.family_survivorship for f in families)
        robust = [f for f in families if f.is_robust_core]

        return {
            "total_families": len(families),
            "robust_core_count": len(robust),
            "family_size_percentiles": _percentiles(sizes),
            "family_survivorship_percentiles": _percentiles(survivorship),
            "robust_core_family_ids": sorted(f.family_id for f in robust),
        }

    @staticmethod
    def _dominant_fragility_classes(
        topology: Optional[Dict[str, TopologyResult]],
        records: List[SurvivorRecord],
    ) -> List[str]:
        """Ordered list of dominant fragility archetypes."""
        classes = []
        if topology:
            topos = list(topology.values())
            isolated_frac = sum(1 for t in topos if t.neighbor_count == 0) / len(topos)
            fragile_frac = sum(1 for t in topos if t.topology_fragility_score >= _FRAGILE_THRESH) / len(topos)
            if isolated_frac >= 0.20:
                classes.append("isolated_spike")
            if fragile_frac >= 0.30:
                classes.append("parameter_cliff")

        survived = [r for r in records if r.survived]
        failed = [r for r in records if not r.survived]
        if len(records) > 0 and len(failed) / len(records) >= 0.50:
            classes.append("high_rejection_rate")
        if survived and failed:
            # Check if survivors have much higher scores
            mean_surv = sum(r.score for r in survived) / len(survived)
            mean_fail = sum(r.score for r in failed) / len(failed)
            if mean_surv - mean_fail > 0.3:
                classes.append("score_cliff")

        return sorted(set(classes))  # sorted for determinism

    @staticmethod
    def _decay_archetypes(
        records: List[SurvivorRecord],
        families: List[FamilyRecord],
    ) -> dict:
        """Summary of decay patterns."""
        total = len(records)
        if total == 0:
            return {}

        # Experiments with high rejection reason density
        multi_fail = [r for r in records if not r.survived and len(r.rejection_reasons) > 1]
        single_fail = [r for r in records if not r.survived and len(r.rejection_reasons) == 1]

        # Family decay: families where survivorship < 0.30
        decayed_families = [f for f in families if f.family_survivorship < 0.30]

        return {
            "multi_criterion_failures": len(multi_fail),
            "single_criterion_failures": len(single_fail),
            "decayed_family_count": len(decayed_families),
            "decayed_family_fraction": round(len(decayed_families) / max(len(families), 1), 4),
        }

    @staticmethod
    def _robustness_distributions(
        records: List[SurvivorRecord],
        topology: Optional[Dict[str, TopologyResult]],
    ) -> dict:
        scores = sorted(r.score for r in records)
        survived_scores = sorted(r.score for r in records if r.survived)
        result = {
            "score_percentiles_all": _percentiles(scores),
            "score_percentiles_survived": _percentiles(survived_scores),
        }
        if topology:
            plateau_widths = sorted(t.plateau_width for t in topology.values())
            result["plateau_width_percentiles"] = _percentiles(plateau_widths)
        return result

    @staticmethod
    def _failure_concentration(records: List[SurvivorRecord]) -> dict:
        """How concentrated are failures across strategies?"""
        if not records:
            return {}

        strategy_failures: Dict[str, int] = {}
        strategy_totals: Dict[str, int] = {}
        for r in records:
            strategy_totals[r.strategy_id] = strategy_totals.get(r.strategy_id, 0) + 1
            if not r.survived:
                strategy_failures[r.strategy_id] = strategy_failures.get(r.strategy_id, 0) + 1

        # Herfindahl of failure counts
        total_failures = sum(strategy_failures.values())
        h = 0.0
        if total_failures > 0:
            for count in strategy_failures.values():
                share = count / total_failures
                h += share ** 2

        return {
            "strategy_failure_counts": dict(sorted(strategy_failures.items())),
            "strategy_failure_rates": {
                sid: round(strategy_failures.get(sid, 0) / tot, 4)
                for sid, tot in sorted(strategy_totals.items())
            },
            "failure_herfindahl": round(h, 6),
        }

    # ── Private helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _empty_report(generated_at: str) -> dict:
        payload: dict = {
            "experiment_counts": {"total": 0, "survived": 0, "rejected": 0},
        }
        return {
            "summary_id": "surv_empty",
            "generated_at": generated_at,
            "schema_version": SCHEMA_VERSION,
            "payload": payload,
        }

    @staticmethod
    def _make_summary_id(checksum: str, generated_at: str) -> str:
        compact_ts = (
            generated_at
            .replace("-", "").replace(":", "").replace("T", "").replace("Z", "").replace(".", "")
        )
        return f"forensic_{compact_ts}_{checksum[:8]}"


# ── Module helpers ─────────────────────────────────────────────────────────────

def _percentiles(sorted_values: list) -> dict:
    """Deterministic percentile dict for a pre-sorted list."""
    if not sorted_values:
        return {}
    n = len(sorted_values)

    def pct(p: float) -> float:
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        frac = idx - lo
        return round(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac, 6)

    return {
        "min": sorted_values[0],
        "p10": pct(10),
        "p25": pct(25),
        "p50": pct(50),
        "p75": pct(75),
        "p90": pct(90),
        "max": sorted_values[-1],
        "mean": round(sum(sorted_values) / n, 6),
        "count": n,
    }
