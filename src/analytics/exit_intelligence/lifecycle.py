"""
src/analytics/exit_intelligence/lifecycle.py — Capital Lifecycle Layer

5 components:
  1. PartialExitAllocator
  2. CounterfactualExitAnalyzer
  3. CounterfactualPartialExitAnalytics
  4. LoserRetentionAnalysis
  5. ConditionalExpectancySurface

All pure analytics — no execution mutation.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import (
    ConditionalExpectancyPoint,
    CounterfactualExitOutcome,
    CounterfactualPartialExitRecord,
    ExitObservationRecord,
    ExitVelocityScore,
    LoserRetentionRecord,
    PartialExitDecision,
)

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))


def _now_jst() -> str:
    return datetime.now(JST).isoformat()


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _atomic_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if path.exists():
                fh.write(path.read_text(encoding="utf-8"))
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── 1. PartialExitAllocator ───────────────────────────────────────────────────

class PartialExitAllocator:
    """
    Deterministic staged-reduction schedule.
    Default: 33% → 33% → 34%
    Convexity preservation: if action_band < staged_reduction, returns no-reduce.
    """

    DEFAULT_STAGES = (0.33, 0.33, 0.34)

    def allocate(self, score: ExitVelocityScore) -> PartialExitDecision:
        band = score.action_band

        if band == "hold":
            first_reduction = 0.0
            rationale = "Convexity intact — no reduction warranted."
        elif band == "tighten_risk":
            first_reduction = 0.0
            rationale = "Tighten stop only — no size reduction yet."
        elif band == "staged_reduction":
            first_reduction = self.DEFAULT_STAGES[0]
            rationale = (
                f"Staged reduction: reduce {self.DEFAULT_STAGES[0]:.0%} now, "
                f"monitor for stages 2-3."
            )
        else:  # aggressive_exit
            first_reduction = 1.0
            rationale = "Aggressive exit: full position closure recommended."

        return PartialExitDecision(
            decision_id=_sha256(f"partial|{score.symbol}|{score.scored_date}"),
            symbol=score.symbol,
            decision_date=score.scored_date,
            velocity_score=score.total_score,
            action_band=band,
            stage_1_fraction=self.DEFAULT_STAGES[0],
            stage_2_fraction=self.DEFAULT_STAGES[1],
            stage_3_fraction=self.DEFAULT_STAGES[2],
            recommended_first_reduction=first_reduction,
            convexity_preservation_rationale=rationale,
            generated_at=_now_jst(),
        )


# ── 2. CounterfactualExitAnalyzer ─────────────────────────────────────────────

class CounterfactualExitAnalyzer:
    """
    Compute post-exit forward returns and excursion metrics.
    Input: exit observation + post-exit price path (list of closes after exit_date).
    """

    def analyze(
        self,
        obs: ExitObservationRecord,
        post_exit_prices: List[float],  # [close_d1, close_d2, ..., close_d20]
    ) -> CounterfactualExitOutcome:
        """
        post_exit_prices: daily closes AFTER exit date. Index 0 = 1d after exit.
        Must have at least 3 elements for meaningful analysis (uses 3/5/10/20d).
        """
        exit_p = obs.exit_price
        if exit_p <= 0:
            raise ValueError("exit_price must be positive")

        def _ret(idx: int) -> Optional[float]:
            if idx < len(post_exit_prices) and post_exit_prices[idx] > 0:
                return (post_exit_prices[idx] - exit_p) / exit_p
            return None

        r3 = _ret(2)   # d3 close
        r5 = _ret(4)
        r10 = _ret(9)
        r20 = _ret(19)

        # Max excursion (best price after exit)
        if post_exit_prices:
            max_p = max(post_exit_prices[:20]) if len(post_exit_prices) >= 20 else max(post_exit_prices)
            post_exit_max_excursion = (max_p - exit_p) / exit_p
        else:
            post_exit_max_excursion = None

        # Max drawdown after exit (worst from exit)
        if post_exit_prices:
            min_p = min(post_exit_prices[:20]) if len(post_exit_prices) >= 20 else min(post_exit_prices)
            post_exit_max_drawdown = (min_p - exit_p) / exit_p  # negative if drawdown
        else:
            post_exit_max_drawdown = None

        # Counterfactual convexity retention
        if r5 is not None and post_exit_max_excursion is not None and post_exit_max_excursion > 0:
            cf_convexity = _clamp(r5 / post_exit_max_excursion)
        else:
            cf_convexity = None

        # Premature / late exit classification
        exit_was_premature: Optional[bool] = None
        exit_was_late: Optional[bool] = None
        if r5 is not None:
            exit_was_premature = r5 > 0.03   # stock rose >3% after exit
        if obs.realized_return < -0.02 and obs.exit_velocity_score < 3.0:
            exit_was_late = True

        return CounterfactualExitOutcome(
            cf_id=_sha256(f"cf|{obs.obs_id}"),
            symbol=obs.symbol,
            actual_exit_date=obs.exit_date,
            actual_exit_price=exit_p,
            realized_return=obs.realized_return,
            post_exit_return_3d=r3,
            post_exit_return_5d=r5,
            post_exit_return_10d=r10,
            post_exit_return_20d=r20,
            post_exit_max_excursion=post_exit_max_excursion,
            post_exit_max_drawdown=post_exit_max_drawdown,
            counterfactual_convexity_retention=cf_convexity,
            exit_was_premature=exit_was_premature,
            exit_was_late=exit_was_late,
            generated_at=_now_jst(),
        )


# ── 3. CounterfactualPartialExitAnalytics ─────────────────────────────────────

class CounterfactualPartialExitAnalytics:
    """
    Compare full vs staged vs delayed vs accelerated exit scenarios.
    Input: actual exit return + post-exit prices.
    """

    def compare(
        self,
        obs: ExitObservationRecord,
        post_exit_prices: List[float],
        delay_days: int = 3,
    ) -> CounterfactualPartialExitRecord:
        exit_p = obs.exit_price
        actual_return = obs.realized_return

        def _price_at(d: int) -> Optional[float]:
            if d - 1 < len(post_exit_prices):
                return post_exit_prices[d - 1]
            return None

        # Staged exit: 33% now, 33% at d3, 34% at d5 (using post-exit continuation)
        staged_return: Optional[float] = None
        p3 = _price_at(3)
        p5 = _price_at(5)
        if p3 is not None and p5 is not None:
            r3 = (p3 - exit_p) / exit_p
            r5 = (p5 - exit_p) / exit_p
            staged_return = 0.33 * actual_return + 0.33 * r3 + 0.34 * r5

        # Delayed exit: hold delay_days more
        p_delay = _price_at(delay_days)
        delayed_return: Optional[float] = None
        if p_delay is not None:
            delayed_return = (p_delay - obs.entry_price) / obs.entry_price

        # Accelerated exit: would have exited 1d earlier (use entry_price proxy)
        # We don't have earlier prices in this interface, so skip if not available
        accelerated_return: Optional[float] = None

        # Best scenario
        candidates: Dict[str, float] = {"full": actual_return}
        if staged_return is not None:
            candidates["staged"] = staged_return
        if delayed_return is not None:
            candidates["delayed"] = delayed_return
        best_scenario = max(candidates, key=lambda k: candidates[k])

        # Convexity retained by staged exit
        cf_staged: Optional[float] = None
        if staged_return is not None and actual_return != 0:
            cf_staged = staged_return / actual_return if actual_return > 0 else None

        return CounterfactualPartialExitRecord(
            record_id=_sha256(f"cf_partial|{obs.obs_id}"),
            symbol=obs.symbol,
            exit_date=obs.exit_date,
            full_exit_return=actual_return,
            staged_exit_return=staged_return,
            delayed_3d_return=delayed_return,
            accelerated_exit_return=accelerated_return,
            best_scenario=best_scenario,
            convexity_retained_by_staged=cf_staged,
            generated_at=_now_jst(),
        )


# ── 4. LoserRetentionAnalysis ─────────────────────────────────────────────────

class LoserRetentionAnalysis:
    """
    Classify losing trades into: good_hold, bad_hold, good_cut, bad_cut.

    good_cut:  velocity score was high (≥5) → exit was correct
    bad_cut:   velocity score was low (<3) but stock recovered post-exit
    good_hold: stock eventually recovered (but we must have exit records)
    bad_hold:  velocity score was high but we held too long
    """

    VELOCITY_CUT_THRESHOLD = 5.0
    RECOVERY_THRESHOLD = 0.03  # 3% post-exit recovery = "would have recovered"

    def classify(
        self,
        obs: ExitObservationRecord,
        post_exit_return_5d: Optional[float] = None,
    ) -> LoserRetentionRecord:
        is_loser = obs.realized_return < 0
        vel_score = obs.exit_velocity_score
        post_ret = post_exit_return_5d

        if is_loser:
            if vel_score >= self.VELOCITY_CUT_THRESHOLD:
                classification = "good_cut"
                rationale = (
                    f"High velocity score ({vel_score:.1f}) justified exit. "
                    "Deterioration was real."
                )
                would_recover = post_ret > self.RECOVERY_THRESHOLD if post_ret is not None else None
            else:
                # Low velocity, but we lost money
                if post_ret is not None and post_ret > self.RECOVERY_THRESHOLD:
                    classification = "bad_cut"
                    rationale = (
                        f"Low velocity ({vel_score:.1f}) but stock recovered +{post_ret:.1%} post-exit. "
                        "Exit may have been premature."
                    )
                    would_recover = True
                elif post_ret is not None and post_ret <= 0:
                    classification = "good_cut"
                    rationale = (
                        f"Low velocity ({vel_score:.1f}) but stock continued to decline post-exit."
                    )
                    would_recover = False
                else:
                    classification = "bad_hold"
                    rationale = (
                        f"Velocity score {vel_score:.1f} — insufficient pressure signal. "
                        "Holding longer may not have helped."
                    )
                    would_recover = None
        else:
            # Winner — not a loser, classify as good_hold
            if vel_score >= self.VELOCITY_CUT_THRESHOLD:
                classification = "bad_hold"
                rationale = (
                    f"High velocity ({vel_score:.1f}) but position was profitable. "
                    "Could have exited earlier."
                )
            else:
                classification = "good_hold"
                rationale = (
                    f"Low velocity ({vel_score:.1f}) and position was profitable. Correct hold."
                )
            would_recover = None

        return LoserRetentionRecord(
            record_id=_sha256(f"loser|{obs.obs_id}"),
            symbol=obs.symbol,
            exit_date=obs.exit_date,
            holding_days=obs.holding_days,
            realized_return=obs.realized_return,
            exit_velocity_score_at_exit=vel_score,
            hold_classification=classification,
            classification_rationale=rationale,
            would_have_recovered=would_recover,
            generated_at=_now_jst(),
        )


# ── 5. ConditionalExpectancySurface ───────────────────────────────────────────

class ConditionalExpectancySurface:
    """
    Build conditional expectancy surface from historical observations.
    Conditions: regime × vol_state × rs_persistence × breadth_condition ×
                deterioration_velocity × convexity_quality × sector_leadership

    Returns a grid of ConditionalExpectancyPoint.
    Requires at least 5 observations per cell for statistical validity.
    """

    MIN_CELL_SAMPLES = 5

    def _discretize(self, obs: ExitObservationRecord) -> Dict[str, str]:
        """Map continuous metrics to discrete categories."""
        # RS persistence
        if obs.rs_rank_at_exit >= 80:
            rs_pers = "strong"
        elif obs.rs_rank_at_exit >= 60:
            rs_pers = "moderate"
        else:
            rs_pers = "weak"

        # Breadth condition
        if obs.breadth_score_at_exit >= 0.6:
            breadth = "expanding"
        elif obs.breadth_score_at_exit >= 0.4:
            breadth = "neutral"
        else:
            breadth = "contracting"

        # Deterioration velocity (use exit_velocity_score as proxy)
        if obs.exit_velocity_score <= 3.0:
            det_vel = "slow"
        elif obs.exit_velocity_score <= 6.0:
            det_vel = "moderate"
        else:
            det_vel = "fast"

        # Convexity quality (use distance_from_peak)
        if obs.distance_from_peak_pct <= 0.05:
            conv = "high"
        elif obs.distance_from_peak_pct <= 0.15:
            conv = "medium"
        else:
            conv = "low"

        # ATR-based volatility state
        if obs.atr_pct_at_exit <= 0.015:
            vol_state = "low"
        elif obs.atr_pct_at_exit <= 0.030:
            vol_state = "medium"
        else:
            vol_state = "high"

        # Sector leadership proxy (use rs_rank)
        if obs.rs_rank_at_exit >= 75:
            sect_lead = "leading"
        elif obs.rs_rank_at_exit >= 50:
            sect_lead = "neutral"
        else:
            sect_lead = "lagging"

        return {
            "regime": obs.regime_at_exit,
            "volatility_state": vol_state,
            "rs_persistence": rs_pers,
            "breadth_condition": breadth,
            "deterioration_velocity": det_vel,
            "convexity_quality": conv,
            "sector_leadership": sect_lead,
        }

    def build(
        self,
        observations: List[ExitObservationRecord],
        forward_returns_5d: Optional[Dict[str, float]] = None,  # obs_id → return
        forward_returns_10d: Optional[Dict[str, float]] = None,
    ) -> List[ConditionalExpectancyPoint]:
        """Build surface from completed observations."""
        if not observations:
            return []

        # Group by cell key
        cells: Dict[str, List] = {}
        for obs in observations:
            disc = self._discretize(obs)
            key = json.dumps(disc, sort_keys=True)
            if key not in cells:
                cells[key] = {"disc": disc, "obs": []}
            cells[key]["obs"].append(obs)

        results: List[ConditionalExpectancyPoint] = []
        for key, cell_data in cells.items():
            disc = cell_data["disc"]
            cell_obs = cell_data["obs"]
            n = len(cell_obs)

            f5: Optional[float] = None
            f10: Optional[float] = None
            if forward_returns_5d and n >= 1:
                vals5 = [
                    forward_returns_5d[o.obs_id]
                    for o in cell_obs
                    if o.obs_id in forward_returns_5d
                ]
                f5 = sum(vals5) / len(vals5) if vals5 else None
            if forward_returns_10d and n >= 1:
                vals10 = [
                    forward_returns_10d[o.obs_id]
                    for o in cell_obs
                    if o.obs_id in forward_returns_10d
                ]
                f10 = sum(vals10) / len(vals10) if vals10 else None

            cond_exp = f5 if n >= self.MIN_CELL_SAMPLES else None

            cell_id = _sha256(key)
            results.append(ConditionalExpectancyPoint(
                cell_id=cell_id,
                regime=disc["regime"],
                volatility_state=disc["volatility_state"],
                rs_persistence=disc["rs_persistence"],
                breadth_condition=disc["breadth_condition"],
                deterioration_velocity=disc["deterioration_velocity"],
                convexity_quality=disc["convexity_quality"],
                sector_leadership=disc["sector_leadership"],
                sample_count=n,
                mean_forward_return_5d=f5,
                mean_forward_return_10d=f10,
                conditional_expectancy=cond_exp,
                generated_at=_now_jst(),
            ))

        return sorted(results, key=lambda r: r.sample_count, reverse=True)


class LifecycleStore:
    """Append-only store for all lifecycle records."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def append(self, record: dict) -> None:
        _atomic_append(self._path, record)

    def load_all(self) -> List[dict]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception as exc:
                logger.warning("[LIFECYCLE] parse error: %s", exc)
        return records
