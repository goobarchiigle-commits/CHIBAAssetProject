"""
src/analytics/exits/trailing_replay.py — Counterfactual trailing-stop policy replayer

Policies (analytics only, no execution mutation):
  fixed_trailing         — fixed % trailing stop from peak
  atr_trailing           — ATR-multiple trailing stop
  vol_scaled             — volatility-scaled trailing stop
  time_decay             — max holding duration exit
  hybrid                 — min(fixed_trailing, time_decay)
  momentum_exhaustion    — N-day momentum sign-reversal exit

All replays are deterministic for fixed inputs.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.analytics.exits.models import (
    ALL_POLICIES,
    POLICY_ATR_TRAILING,
    POLICY_FIXED_TRAILING,
    POLICY_HYBRID,
    POLICY_MOMENTUM_EXHAUSTION,
    POLICY_TIME_DECAY,
    POLICY_VOL_SCALED,
    CompletedTrade,
    HoldingPath,
    PolicyReplayResult,
    SCHEMA_VERSION,
)

logger = logging.getLogger(__name__)

_EPS = 1e-10

# Default policy parameters
_DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    POLICY_FIXED_TRAILING:      {"stop_pct": 0.05},
    POLICY_ATR_TRAILING:        {"n_atr": 2.0, "atr_period": 5},
    POLICY_VOL_SCALED:          {"k": 2.0},
    POLICY_TIME_DECAY:          {"max_days": 20.0},
    POLICY_HYBRID:              {"stop_pct": 0.05, "max_days": 20.0},
    POLICY_MOMENTUM_EXHAUSTION: {"momentum_window": 3},
}


def _safe_div(num: float, denom: float, fallback: float = 0.0) -> float:
    return num / denom if abs(denom) > _EPS else fallback


def _clip(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _compute_atr(prices: List[float], period: int) -> float:
    """Simplified ATR using mean close-to-close absolute differences."""
    if len(prices) < 2:
        return 0.0
    diffs = [abs(prices[i] - prices[i - 1]) for i in range(1, len(prices))]
    window = diffs[-period:] if len(diffs) >= period else diffs
    return sum(window) / len(window) if window else 0.0


def _compute_realized_vol(prices: List[float]) -> float:
    """Daily close-to-close std dev (not annualized)."""
    if len(prices) < 2:
        return 0.0
    returns = [_safe_div(prices[i] - prices[i - 1], prices[i - 1], 0.0) for i in range(1, len(prices))]
    n = len(returns)
    mean = sum(returns) / n
    variance = sum((r - mean) ** 2 for r in returns) / n
    return math.sqrt(variance)


def _find_fixed_trailing_exit(
    prices: List[float],
    days: List[float],
    stop_pct: float,
) -> tuple[int, float]:
    """
    Find index and day where fixed trailing stop triggers.
    Returns (exit_idx, exit_day). Falls back to last index if never triggered.
    """
    if not prices:
        return 0, 0.0
    peak = prices[0]
    for i, (p, d) in enumerate(zip(prices, days)):
        if p > peak:
            peak = p
        stop_level = peak * (1.0 - stop_pct)
        if p <= stop_level:
            return i, d
    return len(prices) - 1, days[-1]


def _find_atr_trailing_exit(
    prices: List[float],
    days: List[float],
    n_atr: float,
    atr_period: int,
) -> tuple[int, float]:
    """ATR trailing stop. Computes ATR over rolling window."""
    if not prices:
        return 0, 0.0
    peak = prices[0]
    for i, (p, d) in enumerate(zip(prices, days)):
        if p > peak:
            peak = p
        atr = _compute_atr(prices[: i + 1], atr_period)
        stop_level = peak - n_atr * atr
        if p <= stop_level and atr > 0:
            return i, d
    return len(prices) - 1, days[-1]


def _find_vol_scaled_exit(
    prices: List[float],
    days: List[float],
    k: float,
) -> tuple[int, float]:
    """Vol-scaled trailing stop: stop_pct = k * realized_daily_vol."""
    if not prices:
        return 0, 0.0
    peak = prices[0]
    for i, (p, d) in enumerate(zip(prices, days)):
        if p > peak:
            peak = p
        vol = _compute_realized_vol(prices[: i + 1])
        stop_pct = k * vol if vol > 0 else 0.03  # fallback 3%
        stop_level = peak * (1.0 - stop_pct)
        if p <= stop_level:
            return i, d
    return len(prices) - 1, days[-1]


def _find_time_decay_exit(
    days: List[float],
    max_days: float,
) -> tuple[int, float]:
    """Exit after max_days holding."""
    for i, d in enumerate(days):
        if d >= max_days:
            return i, d
    return len(days) - 1, days[-1] if days else (0, 0.0)


def _find_momentum_exhaustion_exit(
    prices: List[float],
    days: List[float],
    momentum_window: int,
) -> tuple[int, float]:
    """
    Exit when N-day momentum crosses below zero (price lower than N days ago).
    Only fires after the initial momentum_window is established.
    """
    if len(prices) < momentum_window + 1:
        return len(prices) - 1, days[-1] if days else (0, 0.0)
    for i in range(momentum_window, len(prices)):
        if prices[i] < prices[i - momentum_window]:
            return i, days[i]
    return len(prices) - 1, days[-1]


class TrailingPolicyReplayer:
    """
    Replays counterfactual trailing-stop policies against completed trades.

    Each replay is:
      - Pure: same (trade, holding_path, policy) → same result
      - Analytics-only: does not modify trade records or execution state
    """

    def replay(
        self,
        trade: CompletedTrade,
        holding_path: HoldingPath,
        policy: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> PolicyReplayResult:
        """
        Replay one policy against one trade.

        Args:
            trade:        Completed trade (for actual return baseline).
            holding_path: Price path [{"day": float, "price": float}, ...].
            policy:       One of ALL_POLICIES.
            params:       Optional parameter overrides. Defaults from _DEFAULT_PARAMS.

        Returns:
            PolicyReplayResult (deterministic for fixed inputs).
        """
        if policy not in ALL_POLICIES:
            raise ValueError(f"Unknown policy: {policy!r}. Must be one of {ALL_POLICIES}")

        p = {**_DEFAULT_PARAMS.get(policy, {}), **(params or {})}

        prices = [pt["price"] for pt in holding_path.holding_price_path if "price" in pt]
        days = [pt["day"] for pt in holding_path.holding_price_path if "day" in pt]

        if not prices or not days:
            return self._no_data_result(trade, policy)

        # Find hypothetical exit
        if policy == POLICY_FIXED_TRAILING:
            exit_idx, exit_day = _find_fixed_trailing_exit(prices, days, p["stop_pct"])
        elif policy == POLICY_ATR_TRAILING:
            exit_idx, exit_day = _find_atr_trailing_exit(prices, days, p["n_atr"], p["atr_period"])
        elif policy == POLICY_VOL_SCALED:
            exit_idx, exit_day = _find_vol_scaled_exit(prices, days, p["k"])
        elif policy == POLICY_TIME_DECAY:
            exit_idx, exit_day = _find_time_decay_exit(days, p["max_days"])
        elif policy == POLICY_HYBRID:
            i_fixed, d_fixed = _find_fixed_trailing_exit(prices, days, p["stop_pct"])
            i_time, d_time = _find_time_decay_exit(days, p["max_days"])
            if d_fixed <= d_time:
                exit_idx, exit_day = i_fixed, d_fixed
            else:
                exit_idx, exit_day = i_time, d_time
        elif policy == POLICY_MOMENTUM_EXHAUSTION:
            exit_idx, exit_day = _find_momentum_exhaustion_exit(
                prices, days, int(p["momentum_window"])
            )
        else:
            exit_idx, exit_day = len(prices) - 1, days[-1]

        hyp_exit_price = prices[exit_idx]
        entry_price = trade.entry_price
        hyp_return = _safe_div(hyp_exit_price - entry_price, entry_price, 0.0)

        # Convexity retention: how much of MFE was captured
        peak_price = max(prices[:exit_idx + 1]) if prices[:exit_idx + 1] else entry_price
        mfe = _safe_div(peak_price - entry_price, entry_price, 0.0)
        convexity_retention = _clip(_safe_div(hyp_return, mfe + _EPS, 1.0))

        # PnL giveback at hypothetical exit
        peak_pnl = max(0.0, (peak_price - entry_price) * trade.shares)
        hyp_pnl = (hyp_exit_price - entry_price) * trade.shares
        pnl_giveback = _clip(_safe_div(peak_pnl - hyp_pnl, peak_pnl, 0.0)) if peak_pnl > 0 else 0.0

        # Runner retention vs actual
        runner_retention = _clip(
            _safe_div(hyp_return, trade.realized_return + _EPS, 1.0)
        ) if trade.realized_return > 0 else 1.0

        # Hypothetical exit timestamp: approximate by offsetting exit_day from entry
        hyp_exit_ts = f"hyp_day_{exit_day:.2f}_from_{trade.entry_timestamp}"

        generated_at = datetime.now(timezone.utc).isoformat()
        replay_id = PolicyReplayResult.make_replay_id(trade.record_id, policy)

        return PolicyReplayResult(
            replay_id=replay_id,
            policy_name=policy,
            trade_record_id=trade.record_id,
            symbol=trade.symbol,
            entry_timestamp=trade.entry_timestamp,
            actual_exit_timestamp=trade.exit_timestamp,
            hypothetical_exit_timestamp=hyp_exit_ts,
            actual_realized_return=trade.realized_return,
            hypothetical_realized_return=hyp_return,
            convexity_retention=convexity_retention,
            pnl_giveback=pnl_giveback,
            runner_retention=runner_retention,
            generated_at=generated_at,
            schema_version=SCHEMA_VERSION,
        )

    def replay_all_policies(
        self,
        trade: CompletedTrade,
        holding_path: HoldingPath,
        policy_params: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[PolicyReplayResult]:
        """
        Replay all policies for a single trade. Deterministic order = ALL_POLICIES order.
        """
        results = []
        for policy in ALL_POLICIES:
            params = (policy_params or {}).get(policy)
            try:
                result = self.replay(trade, holding_path, policy, params)
                results.append(result)
            except Exception as exc:
                logger.warning("[TRAILING_REPLAY] policy=%s trade=%s error=%s", policy, trade.record_id, exc)
        return results

    def _no_data_result(self, trade: CompletedTrade, policy: str) -> PolicyReplayResult:
        generated_at = datetime.now(timezone.utc).isoformat()
        return PolicyReplayResult(
            replay_id=PolicyReplayResult.make_replay_id(trade.record_id, policy),
            policy_name=policy,
            trade_record_id=trade.record_id,
            symbol=trade.symbol,
            entry_timestamp=trade.entry_timestamp,
            actual_exit_timestamp=trade.exit_timestamp,
            hypothetical_exit_timestamp=trade.exit_timestamp,
            actual_realized_return=trade.realized_return,
            hypothetical_realized_return=trade.realized_return,
            convexity_retention=1.0,
            pnl_giveback=0.0,
            runner_retention=1.0,
            generated_at=generated_at,
            schema_version=SCHEMA_VERSION,
        )
