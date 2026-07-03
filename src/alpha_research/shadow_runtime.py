"""
shadow_runtime.py — Shadow allocation runtime

Executes virtual alpha candidate allocations without real capital.
Production state is NEVER modified by shadow operations.

Shadow fills are deterministic: given same signal and market snapshot,
fill simulation always produces the same result.

Tracks:
- expected vs realized slippage
- expected vs realized turnover
- signal persistence
- edge decay velocity
- post-cost alpha retention
- liquidity deterioration

Fail-closed: missing market data → no fill, record FAILED.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION: int = 1
DEFAULT_SLIPPAGE_BPS: float = 10.0   # conservative default fill estimate
MAX_SHADOW_POSITIONS: int = 5        # per-candidate position limit
MIN_ADV_FRACTION: float = 0.01       # max 1% of ADV per shadow fill


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class ShadowFill:
    """One shadow fill event. Deterministic given same inputs."""
    candidate_id: str
    trading_day: str
    symbol: str
    side: str                        # BUY | SELL
    shadow_qty: int
    shadow_price: float
    expected_slippage_bps: float
    realized_slippage_bps: float     # proxy: same as expected in shadow
    fill_quality: str                # GOOD | DEGRADED | FAILED
    adv_yen: float                   # average daily volume context
    schema_version: int = SCHEMA_VERSION


@dataclass
class ShadowPortfolioState:
    """Mutable shadow portfolio. One per candidate. Atomic save."""
    candidate_id: str
    positions: dict = field(default_factory=dict)   # {symbol: lots}
    entry_prices: dict = field(default_factory=dict)  # {symbol: avg_price}
    cash: float = 0.0
    realized_pnl_bps: float = 0.0
    unrealized_pnl_bps: float = 0.0
    total_trades: int = 0
    total_fills: int = 0
    failed_fills: int = 0
    peak_portfolio_value: float = 0.0
    max_drawdown: float = 0.0
    schema_version: int = SCHEMA_VERSION


@dataclass
class ShadowObservation:
    """Per-cycle shadow runtime observation. Append-only."""
    candidate_id: str
    trading_day: str
    timestamp: str
    n_signals: int
    n_fills: int
    n_failed: int
    realized_slippage_bps: float
    expected_slippage_bps: float
    slippage_ratio: float           # realized / expected; > 1 = worse than expected
    edge_proxy_bps: float          # rough post-cost alpha proxy
    post_cost_retention: float     # [0, 1]
    portfolio_value: float
    drawdown: float
    schema_version: int = SCHEMA_VERSION


# ── Deterministic fill simulation ──────────────────────────────────────────────

def simulate_shadow_fill(
    candidate_id: str,
    trading_day: str,
    symbol: str,
    side: str,
    requested_qty: int,
    last_price: float,
    adv_yen: float,
    expected_slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> ShadowFill:
    """
    Deterministic shadow fill simulation.
    Returns FAILED if price or ADV missing/invalid.
    Respects MIN_ADV_FRACTION cap.
    """
    if last_price <= 0.0 or adv_yen <= 0.0 or requested_qty <= 0:
        return ShadowFill(
            candidate_id=candidate_id,
            trading_day=trading_day,
            symbol=symbol,
            side=side,
            shadow_qty=0,
            shadow_price=last_price,
            expected_slippage_bps=expected_slippage_bps,
            realized_slippage_bps=0.0,
            fill_quality="FAILED",
            adv_yen=adv_yen,
        )

    # Cap qty by ADV fraction
    max_value = adv_yen * MIN_ADV_FRACTION
    max_qty = int(max_value / last_price)
    fill_qty = min(requested_qty, max(max_qty, 1))
    fill_quality = "GOOD" if fill_qty >= requested_qty else "DEGRADED"

    # Deterministic slippage: expected in shadow context (no realized uncertainty)
    return ShadowFill(
        candidate_id=candidate_id,
        trading_day=trading_day,
        symbol=symbol,
        side=side,
        shadow_qty=fill_qty,
        shadow_price=round(last_price, 2),
        expected_slippage_bps=round(expected_slippage_bps, 2),
        realized_slippage_bps=round(expected_slippage_bps, 2),  # deterministic proxy
        fill_quality=fill_quality,
        adv_yen=adv_yen,
    )


# ── Portfolio state update ──────────────────────────────────────────────────────

def apply_shadow_fill(
    state: ShadowPortfolioState,
    fill: ShadowFill,
) -> ShadowPortfolioState:
    """
    Apply one shadow fill to portfolio state.
    Returns updated state. Deterministic.
    Does NOT mutate input state.
    """
    if fill.fill_quality == "FAILED":
        return ShadowPortfolioState(
            candidate_id=state.candidate_id,
            positions=dict(state.positions),
            entry_prices=dict(state.entry_prices),
            cash=state.cash,
            realized_pnl_bps=state.realized_pnl_bps,
            unrealized_pnl_bps=state.unrealized_pnl_bps,
            total_trades=state.total_trades,
            total_fills=state.total_fills,
            failed_fills=state.failed_fills + 1,
            peak_portfolio_value=state.peak_portfolio_value,
            max_drawdown=state.max_drawdown,
        )

    positions = dict(state.positions)
    entry_prices = dict(state.entry_prices)
    cash = state.cash
    realized_pnl_bps = state.realized_pnl_bps
    trade_value = fill.shadow_qty * fill.shadow_price
    slippage_cost = trade_value * fill.realized_slippage_bps / 10_000.0

    if fill.side == "BUY":
        existing_qty = positions.get(fill.symbol, 0)
        existing_price = entry_prices.get(fill.symbol, fill.shadow_price)
        total_qty = existing_qty + fill.shadow_qty
        if total_qty > 0:
            entry_prices[fill.symbol] = (
                (existing_qty * existing_price + fill.shadow_qty * fill.shadow_price)
                / total_qty
            )
        positions[fill.symbol] = total_qty
        cash -= trade_value + slippage_cost
    elif fill.side == "SELL":
        existing_qty = positions.get(fill.symbol, 0)
        sell_qty = min(fill.shadow_qty, existing_qty)
        if sell_qty > 0:
            entry_price = entry_prices.get(fill.symbol, fill.shadow_price)
            pnl = (fill.shadow_price - entry_price) * sell_qty
            gross_bps = pnl / max(1.0, entry_price * sell_qty) * 10_000
            realized_pnl_bps += gross_bps - fill.realized_slippage_bps
            positions[fill.symbol] = existing_qty - sell_qty
            if positions[fill.symbol] == 0:
                entry_prices.pop(fill.symbol, None)
                del positions[fill.symbol]
        cash += sell_qty * fill.shadow_price - slippage_cost

    total_fills = state.total_fills + 1
    total_trades = state.total_trades + 1

    portfolio_value = cash + sum(
        qty * entry_prices.get(sym, 0.0)
        for sym, qty in positions.items()
    )
    peak = max(state.peak_portfolio_value, portfolio_value)
    dd = (peak - portfolio_value) / peak if peak > 0 else 0.0

    return ShadowPortfolioState(
        candidate_id=state.candidate_id,
        positions=positions,
        entry_prices=entry_prices,
        cash=round(cash, 2),
        realized_pnl_bps=round(realized_pnl_bps, 4),
        unrealized_pnl_bps=state.unrealized_pnl_bps,
        total_trades=total_trades,
        total_fills=total_fills,
        failed_fills=state.failed_fills,
        peak_portfolio_value=round(peak, 2),
        max_drawdown=round(dd, 4),
    )


def build_shadow_observation(
    state: ShadowPortfolioState,
    timestamp: str,
    fills: list[ShadowFill],
    expected_slippage_bps: float,
    edge_proxy_bps: float,
    portfolio_value: float,
) -> ShadowObservation:
    """Build append-only observation from current cycle."""
    n_fills = sum(1 for f in fills if f.fill_quality != "FAILED")
    n_failed = sum(1 for f in fills if f.fill_quality == "FAILED")
    realized_slippage = (
        sum(f.realized_slippage_bps for f in fills if f.fill_quality != "FAILED") / n_fills
        if n_fills else 0.0
    )
    slippage_ratio = realized_slippage / expected_slippage_bps if expected_slippage_bps > 0 else 1.0
    gross_edge = edge_proxy_bps + realized_slippage  # add back costs to get gross
    retention = max(0.0, edge_proxy_bps / gross_edge) if gross_edge > 0 else 0.0

    return ShadowObservation(
        candidate_id=state.candidate_id,
        trading_day=fills[0].trading_day if fills else "",
        timestamp=timestamp,
        n_signals=len(fills),
        n_fills=n_fills,
        n_failed=n_failed,
        realized_slippage_bps=round(realized_slippage, 2),
        expected_slippage_bps=round(expected_slippage_bps, 2),
        slippage_ratio=round(slippage_ratio, 4),
        edge_proxy_bps=round(edge_proxy_bps, 2),
        post_cost_retention=round(retention, 4),
        portfolio_value=round(portfolio_value, 2),
        drawdown=round(state.max_drawdown, 4),
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def save_shadow_portfolio(state: ShadowPortfolioState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(state), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_shadow_portfolio(path: Path, candidate_id: str) -> ShadowPortfolioState:
    if not path.exists():
        return ShadowPortfolioState(candidate_id=candidate_id, cash=0.0)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ShadowPortfolioState(
            candidate_id=data.get("candidate_id", candidate_id),
            positions=data.get("positions", {}),
            entry_prices=data.get("entry_prices", {}),
            cash=float(data.get("cash", 0.0)),
            realized_pnl_bps=float(data.get("realized_pnl_bps", 0.0)),
            unrealized_pnl_bps=float(data.get("unrealized_pnl_bps", 0.0)),
            total_trades=int(data.get("total_trades", 0)),
            total_fills=int(data.get("total_fills", 0)),
            failed_fills=int(data.get("failed_fills", 0)),
            peak_portfolio_value=float(data.get("peak_portfolio_value", 0.0)),
            max_drawdown=float(data.get("max_drawdown", 0.0)),
        )
    except Exception as exc:
        logger.warning("shadow_portfolio load failed (%s) — default", exc)
        return ShadowPortfolioState(candidate_id=candidate_id, cash=0.0)


def append_shadow_fill(fill: ShadowFill, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(fill), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def append_shadow_observation(obs: ShadowObservation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(obs), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_shadow_observations(path: Path) -> list[ShadowObservation]:
    if not path.exists():
        return []
    result: list[ShadowObservation] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            result.append(ShadowObservation(
                candidate_id=data.get("candidate_id", ""),
                trading_day=data.get("trading_day", ""),
                timestamp=data.get("timestamp", ""),
                n_signals=int(data.get("n_signals", 0)),
                n_fills=int(data.get("n_fills", 0)),
                n_failed=int(data.get("n_failed", 0)),
                realized_slippage_bps=float(data.get("realized_slippage_bps", 0.0)),
                expected_slippage_bps=float(data.get("expected_slippage_bps", 0.0)),
                slippage_ratio=float(data.get("slippage_ratio", 1.0)),
                edge_proxy_bps=float(data.get("edge_proxy_bps", 0.0)),
                post_cost_retention=float(data.get("post_cost_retention", 0.0)),
                portfolio_value=float(data.get("portfolio_value", 0.0)),
                drawdown=float(data.get("drawdown", 0.0)),
            ))
        except Exception as exc:
            logger.warning("shadow_obs parse error (%s) — skipped", exc)
    return result
