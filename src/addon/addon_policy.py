"""
addon_policy.py — AddOnExecutionPolicy

Generates discrete 100-share add-on BUY orders for confirmed winners.

Safety constraints (hard bounds, evaluated in order):
  1. Portfolio heat gate    — portfolio_pnl_pct < MIN_PORTFOLIO_PNL → block all
  2. Winner confirmation    — WinnerConfirmationEngine.confirmed == True required
  3. Cooldown gate          — >= ADDON_COOLDOWN_DAYS since last add-on
  4. Depth gate             — addon_count < MAX_ADDON_DEPTH
  5. Concentration gate     — post-addon position weight <= MAX_POSITION_WEIGHT_ADDON
  6. Liquidity gate         — addon_value <= ADV / MIN_ADV_MULTIPLE (fail-open when ADV=0)
  7. Run throttle           — at most MAX_ADDONS_PER_RUN per morning execution

No martingale. No unrestricted concentration. Deterministic replay.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .winner_confirmation import WinnerConfirmation
from .addon_state import (
    AddonSymbolState,
    AddonDecisionRecord,
    load_addon_state,
    save_addon_state,
    append_decision_record,
    purge_exited_symbols,
)

JST = timezone(timedelta(hours=9))
logger = logging.getLogger(__name__)

# ── Policy constants ──────────────────────────────────────────────────────────
UNIT_SHARES: int                     = 100      # Japanese unit constraint (1単元)
MAX_ADDON_DEPTH: int                 = 2        # max add-on units per position lifetime
ADDON_COOLDOWN_DAYS: int             = 5        # business-day cooldown between add-ons
MAX_POSITION_WEIGHT_ADDON: float     = 0.35     # max weight after add-on (>= max_single_weight=0.25)
MIN_PORTFOLIO_PNL_FOR_ADDON: float   = -0.05    # portfolio pnl_pct < -5% → block all adds
MIN_ADV_MULTIPLE: float              = 10.0     # addon_value must be < ADV / 10 (liquidity gate)
MAX_ADDONS_PER_RUN: int              = 1        # at most 1 add-on per morning execution


@dataclass
class AddOnOrder:
    """A single generated add-on BUY instruction."""
    symbol: str
    qty: int
    estimated_price: float
    estimated_cost: float
    reason: str
    confirmation_score: float
    addon_count_before: int


@dataclass
class AddOnResult:
    """Output of AddOnExecutionPolicy.run()."""
    addon_orders: list[AddOnOrder] = field(default_factory=list)
    blocked: list[dict] = field(default_factory=list)
    state_updated: bool = False


# ── Gate helpers (pure, no I/O) ───────────────────────────────────────────────

def _cooldown_elapsed(last_addon_date: str, today: str, cooldown_days: int) -> bool:
    """True when enough calendar days have passed since last add-on."""
    if not last_addon_date:
        return True
    try:
        last = datetime.strptime(last_addon_date, "%Y-%m-%d").date()
        now  = datetime.strptime(today, "%Y-%m-%d").date()
        return (now - last).days >= cooldown_days
    except Exception:
        return True


def _check_concentration(
    cur_position_value: float,
    addon_cost: float,
    portfolio_equity: float,
    max_weight: float,
) -> tuple[bool, str]:
    """Returns (allowed, block_reason)."""
    if portfolio_equity <= 0:
        return False, "portfolio_equity=0"
    post_weight = (cur_position_value + addon_cost) / portfolio_equity
    if post_weight > max_weight:
        return False, (
            f"post_addon_weight={post_weight:.1%} > max={max_weight:.1%}"
        )
    return True, ""


def _check_liquidity(
    addon_shares: int,
    estimated_price: float,
    avg_daily_volume_yen: float,
) -> tuple[bool, str]:
    """Returns (allowed, block_reason). Fail-open when ADV unknown (= 0)."""
    if avg_daily_volume_yen <= 0:
        return True, ""
    addon_value = addon_shares * estimated_price
    threshold = avg_daily_volume_yen / MIN_ADV_MULTIPLE
    if addon_value > threshold:
        return False, (
            f"liquidity: ¥{addon_value:,.0f} > ADV/{MIN_ADV_MULTIPLE:.0f}=¥{threshold:,.0f}"
        )
    return True, ""


# ── Policy class ──────────────────────────────────────────────────────────────

class AddOnExecutionPolicy:
    """
    Evaluates winner confirmations and generates bounded add-on BUY orders.

    State is loaded once per run() call and saved atomically on exit.
    Decision records are appended to the JSONL log immediately (fail-open).
    """

    def __init__(
        self,
        state_path: Path,
        decisions_path: Path,
        max_addon_depth: int             = MAX_ADDON_DEPTH,
        cooldown_days: int               = ADDON_COOLDOWN_DAYS,
        max_position_weight: float       = MAX_POSITION_WEIGHT_ADDON,
        min_portfolio_pnl: float         = MIN_PORTFOLIO_PNL_FOR_ADDON,
        unit_shares: int                 = UNIT_SHARES,
        max_addons_per_run: int          = MAX_ADDONS_PER_RUN,
    ) -> None:
        self.state_path          = state_path
        self.decisions_path      = decisions_path
        self.max_addon_depth     = max_addon_depth
        self.cooldown_days       = cooldown_days
        self.max_position_weight = max_position_weight
        self.min_portfolio_pnl   = min_portfolio_pnl
        self.unit_shares         = unit_shares
        self.max_addons_per_run  = max_addons_per_run

    def run(
        self,
        confirmations: list[WinnerConfirmation],
        portfolio_equity: float,
        portfolio_pnl_pct: float,
        held_positions: dict[str, dict],
        today: str,
        run_id: str = "",
    ) -> AddOnResult:
        """
        Evaluate winner confirmations and generate at most MAX_ADDONS_PER_RUN
        add-on orders.

        Args:
            confirmations:      list of WinnerConfirmation results (all held symbols)
            portfolio_equity:   effective capital / total equity (¥)
            portfolio_pnl_pct:  current portfolio return from peak (negative = drawdown)
            held_positions:     {symbol: {"qty": int, "current_price": float,
                                          "avg_daily_volume_yen": float}}
            today:              "YYYY-MM-DD"
            run_id:             for decision log correlation
        """
        held_syms = set(held_positions.keys())
        state = load_addon_state(self.state_path)
        state = purge_exited_symbols(state, held_syms)

        result = AddOnResult()
        addons_this_run = 0

        # Gate 1: portfolio heat (global block)
        if portfolio_pnl_pct < self.min_portfolio_pnl:
            reason = (
                f"portfolio_heat: pnl={portfolio_pnl_pct:.1%}"
                f" < {self.min_portfolio_pnl:.1%}"
            )
            for c in confirmations:
                result.blocked.append({"symbol": c.symbol, "reason": reason})
                self._log_blocked(c, state.get(c.symbol, AddonSymbolState()), reason, today, run_id)
            save_addon_state(state, self.state_path)
            return result

        # Gate 2: filter to confirmed winners only; best first
        confirmed = sorted(
            [c for c in confirmations if c.confirmed],
            key=lambda c: c.confidence_score,
            reverse=True,
        )

        for c in confirmed:
            if addons_this_run >= self.max_addons_per_run:
                result.blocked.append({
                    "symbol": c.symbol,
                    "reason": f"run_throttle: {addons_this_run}/{self.max_addons_per_run}",
                })
                break

            sym = c.symbol
            pos = held_positions.get(sym, {})
            if not pos:
                continue

            qty_held   = int(pos.get("qty", 0))
            cur_price  = float(pos.get("current_price", 0.0))
            adv_yen    = float(pos.get("avg_daily_volume_yen", 0.0))

            if cur_price <= 0 or qty_held <= 0:
                result.blocked.append({"symbol": sym, "reason": "invalid_position_data"})
                continue

            sym_state    = state.get(sym, AddonSymbolState())
            addon_shares = self.unit_shares
            addon_cost   = addon_shares * cur_price

            # Gate 3: cooldown
            if not _cooldown_elapsed(sym_state.last_addon_date, today, self.cooldown_days):
                bl = f"cooldown: last={sym_state.last_addon_date} require={self.cooldown_days}d"
                result.blocked.append({"symbol": sym, "reason": bl})
                self._log_blocked(c, sym_state, bl, today, run_id)
                continue

            # Gate 4: depth
            if sym_state.addon_count >= self.max_addon_depth:
                bl = f"max_depth: count={sym_state.addon_count} >= {self.max_addon_depth}"
                result.blocked.append({"symbol": sym, "reason": bl})
                self._log_blocked(c, sym_state, bl, today, run_id)
                continue

            # Gate 5: concentration
            cur_pos_value = qty_held * cur_price
            conc_ok, conc_reason = _check_concentration(
                cur_pos_value, addon_cost, portfolio_equity, self.max_position_weight,
            )
            if not conc_ok:
                result.blocked.append({"symbol": sym, "reason": conc_reason})
                self._log_blocked(c, sym_state, conc_reason, today, run_id)
                continue

            # Gate 6: liquidity
            liq_ok, liq_reason = _check_liquidity(addon_shares, cur_price, adv_yen)
            if not liq_ok:
                result.blocked.append({"symbol": sym, "reason": liq_reason})
                self._log_blocked(c, sym_state, liq_reason, today, run_id)
                continue

            # All gates passed → approve add-on
            reason = (
                f"ADDON[winner_add_on#{sym_state.addon_count + 1}]:"
                f" pnl={c.unrealized_pnl_pct:.1%}"
                f" RSR={c.rsr:.1f} rank={c.rsr_rank}"
                f" hold={c.hold_days}d"
                f" score={c.confidence_score:.3f}"
                f" depth={sym_state.addon_count + 1}/{self.max_addon_depth}"
            )
            result.addon_orders.append(AddOnOrder(
                symbol=sym,
                qty=addon_shares,
                estimated_price=cur_price,
                estimated_cost=addon_cost,
                reason=reason,
                confirmation_score=c.confidence_score,
                addon_count_before=sym_state.addon_count,
            ))

            # Update state
            count_before           = sym_state.addon_count
            sym_state.addon_count        += 1
            sym_state.last_addon_date     = today
            sym_state.total_addon_shares += addon_shares
            sym_state.last_confirmation_score = c.confidence_score
            state[sym] = sym_state
            addons_this_run += 1

            self._log_approved(c, sym_state, count_before, addon_shares, addon_cost, today, run_id)

        result.state_updated = addons_this_run > 0
        save_addon_state(state, self.state_path)
        return result

    # ── logging helpers ───────────────────────────────────────────────────────

    def _log_approved(
        self,
        c: WinnerConfirmation,
        state: AddonSymbolState,
        count_before: int,
        addon_shares: int,
        estimated_cost: float,
        today: str,
        run_id: str,
    ) -> None:
        append_decision_record(AddonDecisionRecord(
            ts=datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            run_id=run_id,
            symbol=c.symbol,
            decision="ADDON_APPROVED",
            block_reason="",
            addon_count_before=count_before,
            addon_count_after=state.addon_count,
            addon_shares=addon_shares,
            confirmation_score=c.confidence_score,
            unrealized_pnl_pct=c.unrealized_pnl_pct,
            hold_days=c.hold_days,
            rsr_rank=c.rsr_rank,
            rsr=c.rsr,
            estimated_cost=estimated_cost,
        ), self.decisions_path)

    def _log_blocked(
        self,
        c: WinnerConfirmation,
        state: AddonSymbolState,
        reason: str,
        today: str,
        run_id: str,
    ) -> None:
        append_decision_record(AddonDecisionRecord(
            ts=datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            run_id=run_id,
            symbol=c.symbol,
            decision="ADDON_BLOCKED",
            block_reason=reason,
            addon_count_before=state.addon_count,
            addon_count_after=state.addon_count,
            addon_shares=0,
            confirmation_score=c.confidence_score,
            unrealized_pnl_pct=c.unrealized_pnl_pct,
            hold_days=c.hold_days,
            rsr_rank=c.rsr_rank,
            rsr=c.rsr,
            estimated_cost=0.0,
        ), self.decisions_path)
