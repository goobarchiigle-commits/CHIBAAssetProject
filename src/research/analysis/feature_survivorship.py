"""FeatureSurvivorshipAnalyzer — deterministic feature persistence analysis.

Identifies which feature structures persist across surviving alpha families.
No probabilistic attribution. No ML. Deterministic aggregation only.

Input unit: (experiment_id, feature_names, survived, regime, family_id)
Output: per-feature survival statistics, stable-sorted by feature_survival_rate desc.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, NamedTuple, Tuple

from src.research.analysis.contract import FeatureSurvivalStats


class FeatureObservation(NamedTuple):
    experiment_id: str
    feature_names: List[str]   # list of feature names for this experiment
    survived: bool
    regime: str                # dominant regime label (e.g. "trend_up")
    family_id: str


class FeatureSurvivorshipAnalyzer:
    """Compute feature-level survival statistics from experiment observations.

    All outputs are deterministic: same observations → same stats, regardless
    of observation order.
    """

    def analyze(
        self,
        observations: List[FeatureObservation],
    ) -> Dict[str, FeatureSurvivalStats]:
        """Return per-feature stats sorted by feature_name for stable access.

        Returns:
            Dict[feature_name, FeatureSurvivalStats]
        """
        if not observations:
            return {}

        # Validate no duplicate experiment_ids
        seen_ids = set()
        for obs in observations:
            if obs.experiment_id in seen_ids:
                raise ValueError(f"Duplicate experiment_id: {obs.experiment_id}")
            seen_ids.add(obs.experiment_id)

        # Accumulate per-feature data
        total: Dict[str, int] = defaultdict(int)
        survived_count: Dict[str, int] = defaultdict(int)
        regime_total: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        regime_survived: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        families: Dict[str, set] = defaultdict(set)  # feature -> set of family_ids

        # Split observations into two halves for decay resistance computation
        # Sort by experiment_id for deterministic split
        sorted_obs = sorted(observations, key=lambda o: o.experiment_id)
        mid = len(sorted_obs) // 2
        early_ids = frozenset(o.experiment_id for o in sorted_obs[:mid])
        late_ids = frozenset(o.experiment_id for o in sorted_obs[mid:])
        early_survived: Dict[str, int] = defaultdict(int)
        early_total: Dict[str, int] = defaultdict(int)
        late_survived: Dict[str, int] = defaultdict(int)
        late_total: Dict[str, int] = defaultdict(int)

        for obs in observations:
            for feat in obs.feature_names:
                total[feat] += 1
                if obs.survived:
                    survived_count[feat] += 1
                regime_total[feat][obs.regime] += 1
                if obs.survived:
                    regime_survived[feat][obs.regime] += 1
                families[feat].add(obs.family_id)
                if obs.experiment_id in early_ids:
                    early_total[feat] += 1
                    if obs.survived:
                        early_survived[feat] += 1
                else:
                    late_total[feat] += 1
                    if obs.survived:
                        late_survived[feat] += 1

        all_features = sorted(total.keys())
        result: Dict[str, FeatureSurvivalStats] = {}

        for feat in all_features:
            tot = total[feat]
            surv = survived_count[feat]
            survival_rate = surv / tot if tot > 0 else 0.0

            # Persistence: how often does the feature appear across families (normalized)
            n_families = len(families[feat])
            all_family_ids = set()
            for obs in observations:
                all_family_ids.add(obs.family_id)
            max_families = max(len(all_family_ids), 1)
            persistence = min(1.0, n_families / max_families)

            # Decay resistance: late survival rate / early survival rate
            e_tot = early_total[feat]
            l_tot = late_total[feat]
            e_rate = early_survived[feat] / e_tot if e_tot > 0 else 0.0
            l_rate = late_survived[feat] / l_tot if l_tot > 0 else 0.0
            if e_rate > 1e-9:
                decay_resistance = min(1.0, l_rate / e_rate)
            elif l_rate > 0:
                decay_resistance = 1.0  # survived late without early baseline
            else:
                decay_resistance = 0.0

            # Per-regime affinity: sorted tuple of (regime, survival_rate)
            regime_affinity_pairs = []
            for regime in sorted(regime_total[feat].keys()):
                r_tot = regime_total[feat][regime]
                r_surv = regime_survived[feat].get(regime, 0)
                r_rate = r_surv / r_tot if r_tot > 0 else 0.0
                regime_affinity_pairs.append((regime, round(r_rate, 6)))

            result[feat] = FeatureSurvivalStats(
                feature_name=feat,
                total_occurrences=tot,
                survived_occurrences=surv,
                feature_survival_rate=round(survival_rate, 6),
                feature_persistence_score=round(persistence, 6),
                feature_decay_resistance=round(decay_resistance, 6),
                feature_regime_affinity=tuple(regime_affinity_pairs),
                feature_family_frequency=n_families,
            )

        return result

    @staticmethod
    def rank_by_survival_rate(
        stats: Dict[str, FeatureSurvivalStats],
    ) -> List[FeatureSurvivalStats]:
        """Deterministic ranking: survival_rate desc, feature_name asc for ties."""
        return sorted(
            stats.values(),
            key=lambda s: (-s.feature_survival_rate, s.feature_name),
        )

    @staticmethod
    def rank_by_persistence(
        stats: Dict[str, FeatureSurvivalStats],
    ) -> List[FeatureSurvivalStats]:
        """Deterministic ranking: persistence_score desc, feature_name asc for ties."""
        return sorted(
            stats.values(),
            key=lambda s: (-s.feature_persistence_score, s.feature_name),
        )
