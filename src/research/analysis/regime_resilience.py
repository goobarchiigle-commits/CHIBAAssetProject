"""RegimeResilienceAnalyzer — deterministic regime transition survival analysis.

Measures how alpha families survive market regime transitions.
Detects single-regime dependency, panic-only alpha, trend-only alpha.

Input: SurvivorRecords with per-regime sharpe scores.
Output: RegimeResilienceResult — population-level resilience metrics.

All computations deterministic. No randomness. Canonical sorting everywhere.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Set, Tuple

from src.research.analysis.contract import RegimeResilienceResult, SurvivorRecord

# Regime label constants (mirrors src/research/regime/rules.py)
_TREND_REGIMES: frozenset = frozenset({"trend_up", "trend_down"})
_VOL_REGIMES: frozenset = frozenset({"high_vol", "low_vol"})
_PANIC_REGIME: str = "panic"

# Classification thresholds
_SINGLE_REGIME_CONCENTRATION_THRESH = 0.80   # Herfindahl
_PANIC_ONLY_MIN_RATIO = 0.70   # fraction of positive performance from panic
_TREND_ONLY_MIN_RATIO = 0.70   # fraction from trend regimes
_VOL_SENSITIVITY_THRESH = 0.50  # absolute gap between high_vol and low_vol sharpe


class RegimeResilienceAnalyzer:
    """Population-level regime resilience analysis.

    Args:
        min_positive_sharpe: sharpe threshold above which a regime is considered "positive".
        min_regime_experiments: minimum experiments to include a regime in transition analysis.
    """

    def __init__(
        self,
        min_positive_sharpe: float = 0.0,
        min_regime_experiments: int = 2,
    ) -> None:
        self.min_positive_sharpe = min_positive_sharpe
        self.min_regime_experiments = min_regime_experiments

    def analyze(self, records: List[SurvivorRecord]) -> RegimeResilienceResult:
        """Compute population-level regime resilience.

        Args:
            records: List of SurvivorRecords. May be in any order; internally sorted.

        Returns:
            RegimeResilienceResult with all metrics computed deterministically.
        """
        if not records:
            return _empty_result()

        sorted_records = sorted(records, key=lambda r: r.experiment_id)

        all_regimes = _collect_all_regimes(sorted_records)

        transition_pairs, transition_fragility = self._compute_transitions(sorted_records, all_regimes)
        cross_regime = self._cross_regime_retention(sorted_records, all_regimes)
        concentration = self._regime_concentration(sorted_records)
        recovery = self._recovery_stability(sorted_records)

        resilience_score = self._resilience_score(
            cross_regime, concentration, recovery, transition_pairs
        )

        single_dep = concentration >= _SINGLE_REGIME_CONCENTRATION_THRESH
        panic_only = self._is_panic_only_alpha(sorted_records)
        trend_only = self._is_trend_only_alpha(sorted_records)
        vol_sensitive = self._is_vol_sensitive_alpha(sorted_records)

        return RegimeResilienceResult(
            transition_pairs=tuple(transition_pairs),
            cross_regime_retention=round(cross_regime, 6),
            transition_fragility=tuple(transition_fragility),
            regime_concentration=round(concentration, 6),
            regime_resilience_score=round(resilience_score, 6),
            recovery_stability=round(recovery, 6),
            single_regime_dependent=single_dep,
            panic_only_alpha=panic_only,
            trend_only_alpha=trend_only,
            vol_sensitive_alpha=vol_sensitive,
        )

    # ── Transition matrix ──────────────────────────────────────────────────────

    def _compute_transitions(
        self,
        records: List[SurvivorRecord],
        all_regimes: List[str],
    ) -> Tuple[list, list]:
        """Compute cross-regime retention for all ordered pairs of regimes.

        transition(A→B): among experiments with data in A AND B,
        fraction where sharpe_A > threshold AND sharpe_B > threshold.
        """
        pairs = []
        fragility_items = []

        for regime_a in all_regimes:
            for regime_b in all_regimes:
                if regime_a == regime_b:
                    continue
                both = [
                    r for r in records
                    if regime_a in r.regime_scores() and regime_b in r.regime_scores()
                ]
                if len(both) < self.min_regime_experiments:
                    continue

                positive_a = [r for r in both if r.regime_scores()[regime_a] > self.min_positive_sharpe]
                if not positive_a:
                    retention = 0.0
                    fragility = 1.0
                else:
                    positive_both = [r for r in positive_a if r.regime_scores()[regime_b] > self.min_positive_sharpe]
                    retention = len(positive_both) / len(positive_a)
                    fragility = 1.0 - retention

                key = f"{regime_a}->{regime_b}"
                pairs.append(((regime_a, regime_b), round(retention, 6)))
                fragility_items.append((key, round(fragility, 6)))

        # Sorted for determinism
        pairs.sort(key=lambda x: (x[0][0], x[0][1]))
        fragility_items.sort(key=lambda x: x[0])
        return pairs, fragility_items

    # ── Cross-regime retention ─────────────────────────────────────────────────

    def _cross_regime_retention(
        self,
        records: List[SurvivorRecord],
        all_regimes: List[str],
    ) -> float:
        """Fraction of experiments with positive sharpe in ALL regimes they participate in."""
        if not records:
            return 0.0
        all_positive = sum(
            1 for r in records
            if all(
                v > self.min_positive_sharpe
                for v in r.regime_scores().values()
            )
            and len(r.regime_score_pairs) > 0
        )
        return all_positive / len(records)

    # ── Regime concentration (Herfindahl-like) ─────────────────────────────────

    def _regime_concentration(self, records: List[SurvivorRecord]) -> float:
        """Mean Herfindahl index of positive performance concentration per experiment.

        Herfindahl = sum(share²) where share = regime_sharpe / total_positive_sharpe.
        Returns 0 if no experiments have positive-regime data.
        """
        herfindahls = []
        for r in records:
            scores = r.regime_scores()
            positive_scores = {k: v for k, v in scores.items() if v > self.min_positive_sharpe}
            if not positive_scores:
                herfindahls.append(1.0)  # no positive regime = fully concentrated (worst)
                continue
            total = sum(positive_scores.values())
            if total < 1e-9:
                herfindahls.append(1.0)
                continue
            h = sum((v / total) ** 2 for v in positive_scores.values())
            herfindahls.append(h)
        return sum(herfindahls) / len(herfindahls) if herfindahls else 1.0

    # ── Recovery stability ─────────────────────────────────────────────────────

    def _recovery_stability(self, records: List[SurvivorRecord]) -> float:
        """Retention rate from panic regime to any non-panic regime.

        Defined as: among experiments with panic data, fraction where
        panic sharpe > threshold AND mean non-panic sharpe > threshold.
        """
        with_panic = [r for r in records if _PANIC_REGIME in r.regime_scores()]
        if not with_panic:
            return 0.0

        stable_recovery = 0
        for r in with_panic:
            scores = r.regime_scores()
            if scores[_PANIC_REGIME] <= self.min_positive_sharpe:
                continue
            non_panic_scores = [v for k, v in scores.items() if k != _PANIC_REGIME]
            if not non_panic_scores:
                continue
            mean_non_panic = sum(non_panic_scores) / len(non_panic_scores)
            if mean_non_panic > self.min_positive_sharpe:
                stable_recovery += 1

        return stable_recovery / len(with_panic)

    # ── Resilience score ───────────────────────────────────────────────────────

    @staticmethod
    def _resilience_score(
        cross_retention: float,
        concentration: float,
        recovery: float,
        transition_pairs: list,
    ) -> float:
        """Composite resilience score [0,1]. Higher = more resilient.

        Weights: 0.40 cross_regime_retention + 0.30 (1-concentration) + 0.30 recovery
        """
        mean_retention = (
            sum(r for _, r in transition_pairs) / len(transition_pairs)
            if transition_pairs else 0.0
        )
        diversification = 1.0 - concentration
        score = (
            0.35 * cross_retention
            + 0.25 * diversification
            + 0.25 * recovery
            + 0.15 * mean_retention
        )
        return max(0.0, min(1.0, score))

    # ── Classification helpers ─────────────────────────────────────────────────

    def _is_panic_only_alpha(self, records: List[SurvivorRecord]) -> bool:
        """True if majority of positive performance is concentrated in panic regime."""
        qualifying = [r for r in records if r.regime_score_pairs]
        if not qualifying:
            return False
        panic_concentrated = 0
        for r in qualifying:
            scores = r.regime_scores()
            panic_sharpe = scores.get(_PANIC_REGIME, 0.0)
            total_positive = sum(v for v in scores.values() if v > 0)
            if total_positive < 1e-9:
                continue
            if panic_sharpe > 0 and (panic_sharpe / total_positive) >= _PANIC_ONLY_MIN_RATIO:
                panic_concentrated += 1
        return (panic_concentrated / len(qualifying)) >= 0.5

    def _is_trend_only_alpha(self, records: List[SurvivorRecord]) -> bool:
        """True if majority of positive performance is concentrated in trend regimes."""
        qualifying = [r for r in records if r.regime_score_pairs]
        if not qualifying:
            return False
        trend_concentrated = 0
        for r in qualifying:
            scores = r.regime_scores()
            trend_total = sum(v for k, v in scores.items() if k in _TREND_REGIMES and v > 0)
            all_positive = sum(v for v in scores.values() if v > 0)
            if all_positive < 1e-9:
                continue
            if trend_total / all_positive >= _TREND_ONLY_MIN_RATIO:
                trend_concentrated += 1
        return (trend_concentrated / len(qualifying)) >= 0.5

    def _is_vol_sensitive_alpha(self, records: List[SurvivorRecord]) -> bool:
        """True if mean sharpe in high_vol vs low_vol differs by threshold."""
        high_vol_scores: List[float] = []
        low_vol_scores: List[float] = []
        for r in records:
            scores = r.regime_scores()
            if "high_vol" in scores:
                high_vol_scores.append(scores["high_vol"])
            if "low_vol" in scores:
                low_vol_scores.append(scores["low_vol"])
        if not high_vol_scores or not low_vol_scores:
            return False
        mean_high = sum(high_vol_scores) / len(high_vol_scores)
        mean_low = sum(low_vol_scores) / len(low_vol_scores)
        return abs(mean_high - mean_low) >= _VOL_SENSITIVITY_THRESH


# ── Helpers ────────────────────────────────────────────────────────────────────

def _collect_all_regimes(records: List[SurvivorRecord]) -> List[str]:
    """Sorted list of all regime names across all records."""
    regimes: Set[str] = set()
    for r in records:
        regimes.update(r.regime_scores().keys())
    return sorted(regimes)


def _empty_result() -> RegimeResilienceResult:
    return RegimeResilienceResult(
        transition_pairs=(),
        cross_regime_retention=0.0,
        transition_fragility=(),
        regime_concentration=1.0,
        regime_resilience_score=0.0,
        recovery_stability=0.0,
        single_regime_dependent=True,
        panic_only_alpha=False,
        trend_only_alpha=False,
        vol_sensitive_alpha=False,
    )
