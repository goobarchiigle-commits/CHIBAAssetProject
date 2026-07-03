"""
capital_state.py — Immutable capital state abstraction

Separates actual_equity (real account value) from effective_capital
(rate-limited deployment base) and risk_adjusted_capital.

Deposit or MTM spike cannot instantly increase effective_capital.
Drawdowns propagate immediately (fail-closed for losses).

effective_capital = min(actual_equity, day_open_effective_capital × (1 + daily_growth_limit))

The growth ceiling anchors to day_open_effective_capital (fixed at the first
update of each calendar day), not the previous call's effective_capital, so
multiple same-day invocations (DRY then LIVE) cannot each consume a fresh
daily_growth_limit allowance.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DAILY_GROWTH_LIMIT_DEFAULT: float = 0.015  # 1.5%/day — 1M injection ramps over ~3 weeks


@dataclass
class CapitalState:
    actual_equity: float            # broker-reported account equity
    effective_capital: float        # growth-rate-limited deployment base
    deployable_capital: float       # effective × (1 - cash_buffer)
    risk_adjusted_capital: float    # deployable × vol_scalar × liq_scalar × exec_scalar
    capital_growth_rate: float      # growth rate applied this step (0.0 = frozen/drawdown)
    last_updated: str               # ISO-8601 date (YYYY-MM-DD)
    day_open_effective_capital: float = 0.0  # effective_capital anchor at start of last_updated date
    schema_version: int = 1

    @staticmethod
    def initial(actual_equity: float, cash_buffer: float = 0.10) -> "CapitalState":
        """Bootstrap state for first run (no prior state)."""
        deployable = actual_equity * max(0.0, 1.0 - cash_buffer)
        return CapitalState(
            actual_equity=actual_equity,
            effective_capital=actual_equity,
            deployable_capital=deployable,
            risk_adjusted_capital=deployable,
            capital_growth_rate=0.0,
            last_updated=date.today().isoformat(),
            day_open_effective_capital=actual_equity,
        )


def compute_effective_capital(
    actual_equity: float,
    prev_effective: float,
    daily_growth_limit: float = DAILY_GROWTH_LIMIT_DEFAULT,
    day_anchor: Optional[float] = None,
) -> tuple[float, float]:
    """
    Returns (new_effective_capital, growth_rate_applied).

    Drawdowns propagate immediately (equity < prev → no rate limit).
    Growth (deposit / MTM gain) is rate-limited.
    daily_growth_limit=0.0 → freeze (no growth, drawdowns still propagate).

    day_anchor: effective_capital as of the start of the current calendar day.
    The growth ceiling is computed from day_anchor (not prev_effective) so that
    multiple same-day invocations (e.g. DRY then LIVE) cannot each consume a
    fresh daily_growth_limit allowance. Defaults to prev_effective when omitted
    (single-call-per-day callers see unchanged behavior).
    """
    if actual_equity <= 0.0:
        return 0.0, 0.0

    anchor = day_anchor if (day_anchor is not None and day_anchor > 0.0) else prev_effective

    if actual_equity <= prev_effective:
        # Drawdown: propagate immediately
        rate = (actual_equity - prev_effective) / prev_effective if prev_effective > 0 else 0.0
        return actual_equity, rate

    # Growth: rate-limited against the day's opening anchor (idempotent same-day)
    max_allowed = anchor * (1.0 + max(0.0, daily_growth_limit))
    new_effective = min(actual_equity, max(max_allowed, prev_effective))
    rate = (new_effective - prev_effective) / prev_effective if prev_effective > 0 else 0.0
    return new_effective, rate


def compute_deployable(
    effective_capital: float,
    cash_buffer: float = 0.10,
) -> float:
    """effective_capital minus mandatory cash reserve. buffer clamped to [0, 1]."""
    buffer = max(0.0, min(1.0, cash_buffer))
    return effective_capital * (1.0 - buffer)


def compute_risk_adjusted(
    deployable_capital: float,
    volatility_scalar: float = 1.0,
    liquidity_scalar: float = 1.0,
    execution_scalar: float = 1.0,
) -> float:
    """
    Risk-adjusted capital with multiplicative market-condition scalars.
    Each scalar is clamped to [0, 1] — cannot increase deployable_capital.
    """
    vol_s  = max(0.0, min(1.0, volatility_scalar))
    liq_s  = max(0.0, min(1.0, liquidity_scalar))
    exec_s = max(0.0, min(1.0, execution_scalar))
    return deployable_capital * vol_s * liq_s * exec_s


def update_capital_state(
    state: CapitalState,
    actual_equity: float,
    volatility_scalar: float = 1.0,
    liquidity_scalar: float = 1.0,
    execution_scalar: float = 1.0,
    cash_buffer: float = 0.10,
    daily_growth_limit: float = DAILY_GROWTH_LIMIT_DEFAULT,
    growth_frozen: bool = False,
) -> CapitalState:
    """
    Compute updated CapitalState from new actual_equity + market scalars.

    growth_frozen=True → effective_capital growth suspended.
    Drawdowns still propagate when frozen (fail-closed).

    day-gate: the daily_growth_limit ceiling anchors to day_open_effective_capital,
    which is fixed at the first update of each calendar day and carried unchanged
    through subsequent same-day calls. This prevents multiple same-day invocations
    (e.g. DRY run followed by LIVE run) from each consuming a fresh 1.5%/day
    allowance and compounding effective_capital growth beyond the intended rate.
    """
    today_str = date.today().isoformat()
    is_new_day = state.last_updated != today_str
    day_anchor = (
        state.effective_capital
        if is_new_day or state.day_open_effective_capital <= 0.0
        else state.day_open_effective_capital
    )

    eff_limit = 0.0 if growth_frozen else daily_growth_limit
    new_effective, growth_rate = compute_effective_capital(
        actual_equity, state.effective_capital, eff_limit, day_anchor=day_anchor,
    )
    deployable = compute_deployable(new_effective, cash_buffer)
    risk_adj = compute_risk_adjusted(deployable, volatility_scalar, liquidity_scalar, execution_scalar)

    return CapitalState(
        actual_equity=actual_equity,
        effective_capital=round(new_effective, 2),
        deployable_capital=round(deployable, 2),
        risk_adjusted_capital=round(risk_adj, 2),
        capital_growth_rate=round(growth_rate, 6),
        last_updated=today_str,
        day_open_effective_capital=round(day_anchor, 2),
        schema_version=state.schema_version,
    )


def load_capital_state(path: Path) -> Optional[CapitalState]:
    """Load from JSON. Returns None if missing or corrupt."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        fields = {k for k in CapitalState.__dataclass_fields__}
        return CapitalState(**{k: v for k, v in data.items() if k in fields})
    except Exception as exc:
        logger.warning("capital_state load failed (%s) — returning None", exc)
        return None


def save_capital_state(state: CapitalState, path: Path) -> None:
    """Atomic write (tmp → rename). Replay-safe."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)
