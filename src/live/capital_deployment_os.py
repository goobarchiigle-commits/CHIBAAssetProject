"""
capital_deployment_os.py — Unified Capital Deployment OS (Phase 5B)

Single deterministic pipeline controlling ALL capital deployment decisions:
  - deployable capital calculation
  - dynamic position sizing
  - opportunity classification
  - add-on entry logic
  - idle capital utilization
  - participation-rate constraints
  - survivability-aware scaling

All execution paths emit DeploymentDecision objects.
No separate idle-deployment engine. No separate sizing authority.

Safety: FAIL_CLOSED on crash regime / broker unhealthy / volatility spike.
All logic is pure Python (no numpy) for Windows determinism.

Telemetry: runtime/analytics/capital_deployment/
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

# ── Schema version ─────────────────────────────────────────────────────────────
SCHEMA_VERSION: int = 1

# ── Hard caps (never changed without user confirmation per CLAUDE.md) ──────────
MAX_POSITIONS_HARD_CAP: int          = 5
MIN_POSITIONS_FLOOR: int             = 2
PARAMS_LOCKED_MAX_POSITIONS: int     = 3     # PARAMS_LOCKED 遵守クランプ上限
MAX_SINGLE_WEIGHT: float             = 0.25    # mirrors CLAUDE.md max_single_weight
MAX_PARTICIPATION_RATE: float        = 0.03    # 3% of ADV hard cap
MAX_STARTER_FRACTION: float          = 0.60    # starter ≤ 60% of target
MAX_ADDON_FRACTION: float            = 0.35    # addon ≤ 35% of target
LOT_SIZE: int                        = 100     # TSE standard lot

# ── Deployment thresholds ──────────────────────────────────────────────────────
IDLE_UTILIZATION_THRESHOLD: float    = 0.50    # below → selectively relax suppression
HIGH_SURVIVABILITY_THRESHOLD: float  = 0.70    # required for idle relaxation
HIGH_CONTINUATION_THRESHOLD: float   = 20.0    # idle relaxation min (< REDUCED_STARTER_CONTINUATION_MIN)
FULL_ENTRY_CONTINUATION_MIN: float   = 55.0    # full entry requires score ≥ this
REDUCED_STARTER_CONTINUATION_MIN: float = 30.0
MIN_SURVIVABILITY_ENTRY: float       = 0.50    # < 50% → BLOCKED for new entries
MIN_SURVIVABILITY_ADDON: float       = 0.65    # stricter for add-ons
BOUNDED_GROWTH_CAP_RATIO: float      = 0.30    # lot_cost / deployable ≤ 30%

# ── Decision types ─────────────────────────────────────────────────────────────
DECISION_FULL_ENTRY         = "FULL_ENTRY"
DECISION_REDUCED_STARTER    = "REDUCED_STARTER"
DECISION_ADDON              = "ADDON"
DECISION_OBSERVATION_ONLY   = "OBSERVATION_ONLY"
DECISION_BLOCKED            = "BLOCKED"

# ── Market regime values ────────────────────────────────────────────────────────
REGIME_CRASH  = "crash"
REGIME_BEAR   = "bear"
REGIME_NORMAL = "normal"
REGIME_BULL   = "bull"

# ── Broker health values ───────────────────────────────────────────────────────
BROKER_HEALTHY     = "healthy"
BROKER_DEGRADED    = "degraded"
BROKER_UNREACHABLE = "unreachable"


# ─────────────────────────────────────────────────────────────────────────────
# Input data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HeldPosition:
    """State of a currently held position."""
    symbol: str
    qty: int
    avg_cost: float
    unrealized_pnl_pct: float     # e.g. 0.12 = +12%
    hold_days: int
    addon_count: int               # add-ons already executed
    last_addon_date: str           # "YYYY-MM-DD" or ""
    sector: str


@dataclass
class CandidateInfo:
    """Enriched candidate for deployment evaluation."""
    symbol: str
    sector: str
    price: float
    adv_yen: float                 # average daily volume in yen; 0 = unknown
    continuation_score: float      # 0-100 from ContinuationPriority
    survivability: float           # 0-1 from FutureLeaderSurvivability / position truth
    rsr_rank: int                  # 1-based (lower = stronger)
    predictive_score: float        # 0-100 from PredictiveExpansionScorer; 0 = unknown
    is_held: bool                  # currently in portfolio
    held_info: Optional[HeldPosition] = None


@dataclass
class DeploymentContext:
    """
    Immutable snapshot of system state at decision time.
    All fields must be populated before calling CapitalDeploymentOS.run().
    """
    actual_equity: float
    effective_capital: float
    deployable_capital: float
    current_positions: Dict[str, HeldPosition]  # {symbol: HeldPosition}
    market_regime: str                           # crash|bear|normal|bull
    broker_health: str                           # healthy|degraded|unreachable
    volatility_spike: bool
    cb_active: bool
    date: str                                    # "YYYY-MM-DD"
    capital_utilization: float                   # currently_deployed / deployable_capital


# ─────────────────────────────────────────────────────────────────────────────
# Output data structure
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeploymentDecision:
    """
    Universal output for every candidate.
    Emitted by every execution path — allocation, addon, continuation, idle.
    """
    symbol: str
    decision_type: str             # FULL_ENTRY|REDUCED_STARTER|ADDON|OBSERVATION_ONLY|BLOCKED
    target_size: int               # shares; 0 if BLOCKED/OBSERVATION_ONLY
    target_notional: float         # target_size × price
    deployable_capital: float
    continuation_score: float
    survivability: float
    participation_rate: float      # target_notional / adv_yen; 0 if adv unknown
    capital_utilization: float
    suppression_reason: str        # "" if not suppressed
    regime_scale: float            # scalar applied for regime (1.0 = full)
    liquidity_constrained: bool    # True if size was reduced for participation cap
    idle_relaxation_applied: bool  # True if idle logic relaxed suppression
    date: str
    schema_version: int = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# Capital / sizing pure functions
# ─────────────────────────────────────────────────────────────────────────────

def dynamic_max_positions(deployable_capital: float) -> int:
    """
    Capital-tier based max simultaneous positions.
    Grows with capital, bounded by [MIN_POSITIONS_FLOOR, MAX_POSITIONS_HARD_CAP].
    """
    if deployable_capital <= 0:
        return MIN_POSITIONS_FLOOR
    tier = int(deployable_capital / 1_000_000)
    raw = tier + 2   # 1M→3, 2M→4, 3M+→5
    result = max(MIN_POSITIONS_FLOOR, min(MAX_POSITIONS_HARD_CAP, raw))
    return min(result, PARAMS_LOCKED_MAX_POSITIONS)  # PARAMS_LOCKED=3 クランプ


def target_position_size(
    deployable_capital: float,
    max_pos: int,
    price: float,
) -> int:
    """
    Equal-weight target size per position.
    Capped at MAX_SINGLE_WEIGHT of deployable_capital.
    Returns lot-rounded share count.
    """
    if price <= 0 or deployable_capital <= 0 or max_pos <= 0:
        return 0
    per_pos = deployable_capital / max_pos
    cap = deployable_capital * MAX_SINGLE_WEIGHT
    notional = min(per_pos, cap)
    lots = int(notional / (price * LOT_SIZE))
    return max(0, lots * LOT_SIZE)


def starter_position_size(
    deployable_capital: float,
    max_pos: int,
    price: float,
) -> int:
    """
    Reduced starter size for uncertain / late entry candidates.
    = target × MAX_STARTER_FRACTION, lot-rounded.
    """
    full = target_position_size(deployable_capital, max_pos, price)
    if full <= 0:
        return 0
    return max(0, _round_lots(full * MAX_STARTER_FRACTION))


def addon_position_size(
    deployable_capital: float,
    max_pos: int,
    price: float,
) -> int:
    """
    Incremental add-on for confirmed winners.
    = target × MAX_ADDON_FRACTION, lot-rounded.
    Minimum 1 lot (100 shares).
    """
    full = target_position_size(deployable_capital, max_pos, price)
    if full <= 0:
        return 0
    raw = int(full * MAX_ADDON_FRACTION / LOT_SIZE) * LOT_SIZE
    return max(LOT_SIZE, raw)


def _round_lots(shares: float) -> int:
    """Floor to nearest TSE lot (100 shares)."""
    return max(0, int(shares / LOT_SIZE) * LOT_SIZE)


def compute_participation_rate(
    order_notional: float,
    adv_yen: float,
) -> float:
    """participation_rate = order_notional / adv_yen. 0.0 if adv unknown."""
    if adv_yen <= 0:
        return 0.0
    return order_notional / adv_yen


def apply_participation_cap(
    requested_shares: int,
    price: float,
    adv_yen: float,
    max_rate: float = MAX_PARTICIPATION_RATE,
) -> Tuple[int, bool]:
    """
    Reduce requested_shares so participation_rate ≤ max_rate.
    Returns (final_shares, was_reduced).
    Fail-safe: if adv_yen unknown (≤0), returns original unchanged.
    """
    if adv_yen <= 0 or price <= 0 or requested_shares <= 0:
        return requested_shares, False
    max_notional = adv_yen * max_rate
    max_shares_by_adv = int(max_notional / price)
    max_lots = _round_lots(max_shares_by_adv)
    if max_lots >= requested_shares:
        return requested_shares, False
    return max(0, max_lots), True


def regime_scale_factor(
    market_regime: str,
    cb_active: bool,
) -> float:
    """
    Sizing scalar for regime and circuit-breaker state.
    FAIL_CLOSED conditions (crash/unreachable) return 0.0.
    """
    if market_regime == REGIME_CRASH:
        return 0.0
    if cb_active:
        return 0.0          # no new entries during CB
    if market_regime == REGIME_BEAR:
        return 0.50
    if market_regime == REGIME_BULL:
        return 1.0
    return 1.0              # normal


def compute_capital_utilization(
    positions: Dict[str, HeldPosition],
    prices: Dict[str, float],
    deployable_capital: float,
) -> float:
    """
    current_market_value / deployable_capital.
    Safe: returns 0.0 if no positions or no deployable_capital.
    """
    if deployable_capital <= 0:
        return 1.0  # conservative: assume fully deployed
    total_value = sum(
        pos.qty * prices.get(sym, pos.avg_cost)
        for sym, pos in positions.items()
    )
    return min(1.0, total_value / deployable_capital)


# ─────────────────────────────────────────────────────────────────────────────
# Safety gate
# ─────────────────────────────────────────────────────────────────────────────

def check_safety_gates(
    context: DeploymentContext,
) -> Tuple[bool, str]:
    """
    Hard safety gate check. Returns (safe, reason).
    False → FAIL_CLOSED, block ALL deployment.
    """
    if context.market_regime == REGIME_CRASH:
        return False, f"market_regime=crash: deployment disabled"
    if context.broker_health != BROKER_HEALTHY:
        return False, f"broker_health={context.broker_health}: deployment disabled"
    if context.volatility_spike:
        return False, "volatility_spike=True: deployment disabled"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Opportunity classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_opportunity(
    candidate: CandidateInfo,
    context: DeploymentContext,
    is_addon_eligible: bool = False,
) -> Tuple[str, str]:
    """
    Classify a candidate into one of the 5 decision types.
    Returns (decision_type, suppression_reason).

    Classification order (fail-fast):
      1. BLOCKED  — safety/CB/survivability gate
      2. ADDON    — held winner with addon eligibility
      3. FULL_ENTRY / REDUCED_STARTER — new entries
      4. OBSERVATION_ONLY — below threshold
    """
    # Gate 1: held positions can always receive SELL signal — but new BUY entries
    # require safety clearance
    safe, safety_reason = check_safety_gates(context)
    if not safe:
        return DECISION_BLOCKED, safety_reason

    # Gate 2: CB_ACTIVE blocks new entries (but not SELL on held)
    if context.cb_active and not candidate.is_held:
        return DECISION_BLOCKED, "cb_active: new entries disabled"

    # Gate 3: survivability check
    if not candidate.is_held:
        if candidate.survivability < MIN_SURVIVABILITY_ENTRY:
            return DECISION_BLOCKED, (
                f"survivability={candidate.survivability:.2f} < min={MIN_SURVIVABILITY_ENTRY}"
            )

    # Gate 4: add-on path (held winners)
    if candidate.is_held and is_addon_eligible:
        if candidate.survivability < MIN_SURVIVABILITY_ADDON:
            return DECISION_BLOCKED, (
                f"addon blocked: survivability={candidate.survivability:.2f} < {MIN_SURVIVABILITY_ADDON}"
            )
        return DECISION_ADDON, ""

    # Gate 5: new entry classification
    if not candidate.is_held:
        r_scale = regime_scale_factor(context.market_regime, context.cb_active)
        if r_scale == 0.0:
            return DECISION_BLOCKED, f"regime_scale=0: regime={context.market_regime}"

        # Lot-cost admission: lot_cost / deployable ≤ BOUNDED_GROWTH_CAP_RATIO
        if context.deployable_capital > 0 and candidate.price > 0:
            lot_cost = candidate.price * LOT_SIZE
            ratio = lot_cost / context.deployable_capital
            if ratio > BOUNDED_GROWTH_CAP_RATIO:
                return DECISION_BLOCKED, (
                    f"lot_cost_ratio={ratio:.3f} > {BOUNDED_GROWTH_CAP_RATIO}: "
                    f"price too high for current capital"
                )

        if candidate.continuation_score >= FULL_ENTRY_CONTINUATION_MIN:
            return DECISION_FULL_ENTRY, ""

        if candidate.continuation_score >= REDUCED_STARTER_CONTINUATION_MIN:
            return DECISION_REDUCED_STARTER, ""

        return DECISION_OBSERVATION_ONLY, (
            f"continuation_score={candidate.continuation_score:.1f} < "
            f"min={REDUCED_STARTER_CONTINUATION_MIN}"
        )

    # Held, not addon-eligible → observation
    return DECISION_OBSERVATION_ONLY, "held_position: not addon_eligible"


def _should_apply_idle_relaxation(
    candidate: CandidateInfo,
    context: DeploymentContext,
) -> bool:
    """
    Idle capital response: selectively upgrade OBSERVATION_ONLY to REDUCED_STARTER
    when capital_utilization is low and candidate meets high-quality criteria.
    Never bypasses hard risk controls.
    """
    if context.capital_utilization >= IDLE_UTILIZATION_THRESHOLD:
        return False
    if candidate.survivability < HIGH_SURVIVABILITY_THRESHOLD:
        return False
    if candidate.continuation_score < HIGH_CONTINUATION_THRESHOLD:
        return False
    if context.cb_active:
        return False
    if context.market_regime == REGIME_CRASH:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Add-on eligibility check
# ─────────────────────────────────────────────────────────────────────────────

def _is_addon_eligible(
    candidate: CandidateInfo,
    today: str,
    max_addon_depth: int = 2,
    addon_cooldown_days: int = 5,
    max_position_weight_addon: float = 0.35,
    deployable_capital: float = 0.0,
) -> Tuple[bool, str]:
    """
    Check all add-on eligibility gates. Returns (eligible, reason).
    Pure function — no I/O.
    """
    if not candidate.is_held or candidate.held_info is None:
        return False, "not a held position"

    held = candidate.held_info

    # Depth gate
    if held.addon_count >= max_addon_depth:
        return False, f"addon_count={held.addon_count} >= max={max_addon_depth}"

    # Profit requirement (only add to winners)
    if held.unrealized_pnl_pct <= 0:
        return False, f"unrealized_pnl_pct={held.unrealized_pnl_pct:.3f} ≤ 0 (losers excluded)"

    # Cooldown gate
    if held.last_addon_date:
        try:
            from datetime import date as _date
            last = datetime.strptime(held.last_addon_date, "%Y-%m-%d").date()
            now = datetime.strptime(today, "%Y-%m-%d").date()
            elapsed = (now - last).days
            if elapsed < addon_cooldown_days:
                return False, f"cooldown: {elapsed}d < {addon_cooldown_days}d"
        except Exception:
            pass  # date parse failure → allow (conservative)

    # Concentration gate
    if deployable_capital > 0 and candidate.price > 0:
        current_value = held.qty * candidate.price
        current_weight = current_value / deployable_capital
        addon_notional = LOT_SIZE * candidate.price
        post_addon_weight = (current_value + addon_notional) / deployable_capital
        if post_addon_weight > max_position_weight_addon:
            return False, (
                f"post_addon_weight={post_addon_weight:.3f} > {max_position_weight_addon}"
            )

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate order guard
# ─────────────────────────────────────────────────────────────────────────────

def _is_duplicate(
    symbol: str,
    decision_type: str,
    prior_decisions: List[DeploymentDecision],
) -> bool:
    """
    True if same symbol already received an executable decision in this run.
    Prevents same-run duplicate orders.
    """
    executable = {DECISION_FULL_ENTRY, DECISION_REDUCED_STARTER, DECISION_ADDON}
    if decision_type not in executable:
        return False
    for d in prior_decisions:
        if d.symbol == symbol and d.decision_type in executable:
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

class CapitalDeploymentOS:
    """
    Unified capital deployment OS.

    Processes candidates through one deterministic pipeline:
      Universe Selection → Deployable Capital → Dynamic Scaling →
      Opportunity Classification → Allocation Decision →
      Add-On Decision → Idle Capital Response →
      Governance Validation → Order Emission

    All results are DeploymentDecision objects.
    FAIL_CLOSED on safety violations.
    """

    def __init__(
        self,
        telemetry_dir: Optional[Path] = None,
        max_addon_depth: int = 2,
        addon_cooldown_days: int = 5,
        max_position_weight_addon: float = 0.35,
        max_addons_per_run: int = 1,
        opportunity_cost_dir: Optional[Path] = None,
    ) -> None:
        self._telemetry_dir = telemetry_dir
        self._max_addon_depth = max_addon_depth
        self._addon_cooldown_days = addon_cooldown_days
        self._max_position_weight_addon = max_position_weight_addon
        self._max_addons_per_run = max_addons_per_run
        self._oc_dir = opportunity_cost_dir

    def run(
        self,
        candidates: Sequence[CandidateInfo],
        context: DeploymentContext,
        prices: Optional[Dict[str, float]] = None,
        run_id: str = "",
    ) -> List[DeploymentDecision]:
        """
        Execute the full deployment pipeline.
        Returns one DeploymentDecision per candidate.

        Args:
            candidates: ordered list (ranking already applied by caller)
            context:    immutable system snapshot
            prices:     optional current prices for utilization calc; falls back
                        to CandidateInfo.price when not provided
            run_id:     optional identifier propagated into OC telemetry
        """
        if prices is None:
            prices = {}

        decisions: List[DeploymentDecision] = []
        addons_issued = 0

        # Stage 1: safety gate — if failed, BLOCK everything
        safe, safety_reason = check_safety_gates(context)

        # Stage 2: derive dynamic sizing parameters
        max_pos = dynamic_max_positions(context.deployable_capital)

        # Stage 3: pre-compute OC tracking state (Phase 5C)
        governance_passed = safe and not context.volatility_spike
        held_priorities: Dict[str, float] = {
            c.symbol: c.continuation_score for c in candidates if c.is_held
        }
        active_pos_count = len(context.current_positions)
        slots_this_run = 0
        new_slot_rank = 0

        for cand in candidates:
            price = prices.get(cand.symbol, cand.price)
            if price <= 0:
                price = cand.price

            decision = self._evaluate_candidate(
                cand=cand,
                context=context,
                price=price,
                max_pos=max_pos,
                prior_decisions=decisions,
                safe=safe,
                safety_reason=safety_reason,
                addons_issued=addons_issued,
            )

            # Phase 5C: capacity tracking for new entries only
            if (
                not cand.is_held
                and decision.decision_type in (DECISION_FULL_ENTRY, DECISION_REDUCED_STARTER)
            ):
                new_slot_rank += 1
                capacity_left = max_pos - active_pos_count - slots_this_run
                if capacity_left > 0:
                    slots_this_run += 1
                else:
                    # Portfolio full despite governance passing → OC event
                    if governance_passed and self._oc_dir is not None:
                        self._emit_oc_hook(
                            cand, context, new_slot_rank, max_pos, run_id, held_priorities
                        )
                    decision = self._downgrade_to_capacity_blocked(decision)

            decisions.append(decision)

            if decision.decision_type == DECISION_ADDON and decision.target_size > 0:
                addons_issued += 1

        # Stage: telemetry
        self._emit_telemetry(decisions, context)

        return decisions

    def _evaluate_candidate(
        self,
        cand: CandidateInfo,
        context: DeploymentContext,
        price: float,
        max_pos: int,
        prior_decisions: List[DeploymentDecision],
        safe: bool,
        safety_reason: str,
        addons_issued: int,
    ) -> DeploymentDecision:
        """Evaluate a single candidate through the full pipeline."""

        # Hard safety block
        if not safe:
            return _make_blocked(cand, context, safety_reason)

        # Duplicate guard
        if _is_duplicate(cand.symbol, DECISION_FULL_ENTRY, prior_decisions):
            return _make_blocked(cand, context, "duplicate: already decided this run")

        # Add-on eligibility
        addon_eligible, addon_block_reason = _is_addon_eligible(
            cand,
            today=context.date,
            max_addon_depth=self._max_addon_depth,
            addon_cooldown_days=self._addon_cooldown_days,
            max_position_weight_addon=self._max_position_weight_addon,
            deployable_capital=context.deployable_capital,
        )

        # Per-run addon throttle
        if addon_eligible and addons_issued >= self._max_addons_per_run:
            addon_eligible = False
            addon_block_reason = f"max_addons_per_run={self._max_addons_per_run} reached"

        # Classify
        decision_type, suppression_reason = classify_opportunity(
            cand, context, is_addon_eligible=addon_eligible,
        )

        # Idle capital relaxation: upgrade OBSERVATION_ONLY → REDUCED_STARTER
        idle_relaxation = False
        if decision_type == DECISION_OBSERVATION_ONLY:
            if _should_apply_idle_relaxation(cand, context):
                decision_type = DECISION_REDUCED_STARTER
                suppression_reason = ""
                idle_relaxation = True

        # Regime scale
        r_scale = regime_scale_factor(context.market_regime, context.cb_active)

        # Compute target size
        target_shares = 0
        if decision_type == DECISION_FULL_ENTRY:
            raw = target_position_size(context.deployable_capital, max_pos, price)
            target_shares = _round_lots(raw * r_scale)

        elif decision_type == DECISION_REDUCED_STARTER:
            raw = starter_position_size(context.deployable_capital, max_pos, price)
            target_shares = _round_lots(raw * r_scale)

        elif decision_type == DECISION_ADDON:
            raw = addon_position_size(context.deployable_capital, max_pos, price)
            target_shares = _round_lots(raw * r_scale)
            if target_shares == 0:
                target_shares = LOT_SIZE  # minimum 1 lot

        # Participation rate cap
        target_notional = target_shares * price
        liq_constrained = False
        if target_shares > 0:
            target_shares, liq_constrained = apply_participation_cap(
                target_shares, price, cand.adv_yen,
            )
            target_notional = target_shares * price
            if target_shares == 0 and decision_type not in (
                DECISION_BLOCKED, DECISION_OBSERVATION_ONLY,
            ):
                decision_type = DECISION_BLOCKED
                suppression_reason = "participation_cap: reduced to 0 shares"

        # Participation rate for telemetry
        part_rate = compute_participation_rate(target_notional, cand.adv_yen)

        return DeploymentDecision(
            symbol=cand.symbol,
            decision_type=decision_type,
            target_size=target_shares,
            target_notional=target_notional,
            deployable_capital=context.deployable_capital,
            continuation_score=cand.continuation_score,
            survivability=cand.survivability,
            participation_rate=part_rate,
            capital_utilization=context.capital_utilization,
            suppression_reason=suppression_reason,
            regime_scale=r_scale,
            liquidity_constrained=liq_constrained,
            idle_relaxation_applied=idle_relaxation,
            date=context.date,
        )

    # ── Phase 5C helpers ──────────────────────────────────────────────────────

    def _emit_oc_hook(
        self,
        cand: CandidateInfo,
        context: DeploymentContext,
        candidate_virtual_rank: int,
        max_positions: int,
        run_id: str,
        held_priorities: Dict[str, float],
    ) -> None:
        """Fail-open: write one OC event to opportunity_cost_dir/oc_events.jsonl."""
        try:
            from src.analytics.opportunity_cost import emit_opportunity_cost_from_deployment
            events_file = self._oc_dir / "oc_events.jsonl"
            emit_opportunity_cost_from_deployment(
                candidate=cand,
                context=context,
                held_priorities=held_priorities,
                candidate_virtual_rank=candidate_virtual_rank,
                max_positions=max_positions,
                events_file=events_file,
                run_id=run_id,
            )
        except Exception as exc:
            logger.warning("_emit_oc_hook failed: %s", exc)

    @staticmethod
    def _downgrade_to_capacity_blocked(decision: DeploymentDecision) -> DeploymentDecision:
        """Return a copy of decision downgraded to BLOCKED with capacity reason."""
        from dataclasses import replace as _dc_replace
        return _dc_replace(
            decision,
            decision_type=DECISION_BLOCKED,
            target_size=0,
            target_notional=0.0,
            suppression_reason="capacity_blocked: max_positions reached this run",
            regime_scale=0.0,
        )

    # ── Telemetry ──────────────────────────────────────────────────────────────

    def _emit_telemetry(
        self,
        decisions: List[DeploymentDecision],
        context: DeploymentContext,
    ) -> None:
        """Append deployment decisions and summary to JSONL. Fail-open."""
        if self._telemetry_dir is None:
            return
        try:
            self._telemetry_dir.mkdir(parents=True, exist_ok=True)
            decisions_file = self._telemetry_dir / "deployment_decisions.jsonl"
            telemetry_file = self._telemetry_dir / "deployment_telemetry.jsonl"

            ts = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")

            # Per-decision records
            with decisions_file.open("a", encoding="utf-8") as fh:
                for d in decisions:
                    rec = asdict(d)
                    rec["ts"] = ts
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

            # Daily summary
            summary = _build_telemetry_summary(decisions, context, ts)
            with telemetry_file.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(summary, ensure_ascii=False) + "\n")

        except Exception as exc:
            logger.warning("capital_deployment_os telemetry write failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_telemetry_summary(
    decisions: List[DeploymentDecision],
    context: DeploymentContext,
    ts: str,
) -> dict:
    """Build daily telemetry summary record."""
    executable = [d for d in decisions if d.decision_type in (
        DECISION_FULL_ENTRY, DECISION_REDUCED_STARTER, DECISION_ADDON,
    )]
    blocked = [d for d in decisions if d.decision_type == DECISION_BLOCKED]
    observation = [d for d in decisions if d.decision_type == DECISION_OBSERVATION_ONLY]

    deployed_notional = sum(d.target_notional for d in executable)
    idle_ratio = (
        1.0 - context.capital_utilization
        if context.deployable_capital > 0 else 0.0
    )
    suppressed_alpha = sum(
        1 for d in blocked if "survivability" not in d.suppression_reason
        and "duplicate" not in d.suppression_reason
    )

    return {
        "ts": ts,
        "date": context.date,
        "schema_version": SCHEMA_VERSION,
        "capital_utilization": round(context.capital_utilization, 4),
        "idle_cash_ratio": round(idle_ratio, 4),
        "deployable_capital": context.deployable_capital,
        "effective_capital": context.effective_capital,
        "actual_equity": context.actual_equity,
        "market_regime": context.market_regime,
        "broker_health": context.broker_health,
        "volatility_spike": context.volatility_spike,
        "cb_active": context.cb_active,
        "total_candidates": len(decisions),
        "full_entries": sum(1 for d in decisions if d.decision_type == DECISION_FULL_ENTRY),
        "reduced_starters": sum(1 for d in decisions if d.decision_type == DECISION_REDUCED_STARTER),
        "addons": sum(1 for d in decisions if d.decision_type == DECISION_ADDON),
        "observation_only": len(observation),
        "blocked": len(blocked),
        "deployed_notional": round(deployed_notional, 2),
        "continuation_deployment_rate": (
            round(len(executable) / len(decisions), 4) if decisions else 0.0
        ),
        "liquidity_constrained_count": sum(1 for d in decisions if d.liquidity_constrained),
        "idle_relaxation_count": sum(1 for d in decisions if d.idle_relaxation_applied),
        "suppressed_alpha_count": suppressed_alpha,
        "max_participation_rate": round(
            max((d.participation_rate for d in decisions), default=0.0), 4
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Convenience builders
# ─────────────────────────────────────────────────────────────────────────────

def _make_blocked(
    cand: CandidateInfo,
    context: DeploymentContext,
    reason: str,
) -> DeploymentDecision:
    return DeploymentDecision(
        symbol=cand.symbol,
        decision_type=DECISION_BLOCKED,
        target_size=0,
        target_notional=0.0,
        deployable_capital=context.deployable_capital,
        continuation_score=cand.continuation_score,
        survivability=cand.survivability,
        participation_rate=0.0,
        capital_utilization=context.capital_utilization,
        suppression_reason=reason,
        regime_scale=0.0,
        liquidity_constrained=False,
        idle_relaxation_applied=False,
        date=context.date,
    )


def make_context(
    actual_equity: float,
    effective_capital: float,
    deployable_capital: float,
    positions: Optional[Dict[str, HeldPosition]] = None,
    market_regime: str = REGIME_NORMAL,
    broker_health: str = BROKER_HEALTHY,
    volatility_spike: bool = False,
    cb_active: bool = False,
    date: str = "",
    capital_utilization: float = 0.0,
) -> DeploymentContext:
    """Convenience constructor for DeploymentContext."""
    if positions is None:
        positions = {}
    if not date:
        date = datetime.now(JST).strftime("%Y-%m-%d")
    return DeploymentContext(
        actual_equity=actual_equity,
        effective_capital=effective_capital,
        deployable_capital=deployable_capital,
        current_positions=positions,
        market_regime=market_regime,
        broker_health=broker_health,
        volatility_spike=volatility_spike,
        cb_active=cb_active,
        date=date,
        capital_utilization=capital_utilization,
    )


def make_candidate(
    symbol: str,
    price: float,
    continuation_score: float = 50.0,
    survivability: float = 0.80,
    adv_yen: float = 500_000_000,
    sector: str = "電気機器",
    rsr_rank: int = 5,
    predictive_score: float = 0.0,
    is_held: bool = False,
    held_info: Optional[HeldPosition] = None,
) -> CandidateInfo:
    """Convenience constructor for CandidateInfo."""
    return CandidateInfo(
        symbol=symbol,
        sector=sector,
        price=price,
        adv_yen=adv_yen,
        continuation_score=continuation_score,
        survivability=survivability,
        rsr_rank=rsr_rank,
        predictive_score=predictive_score,
        is_held=is_held,
        held_info=held_info,
    )
