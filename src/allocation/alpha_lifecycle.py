"""
alpha_lifecycle.py — Lightweight per-symbol alpha lifecycle tracking

Purpose: prevent stale alpha from consuming capital indefinitely.

Tracks (per symbol):
- edge_persistence: rolling stability of win rate
- edge_decay: flag when win rate drops below threshold for recent trades
- post_cost_alpha_retention: fraction of gross alpha surviving costs
- regime_specific_stability: win rate by regime tag

Design constraints:
- No future leakage: only uses completed trade records
- Deterministic replay: same inputs → same outputs
- Append-only telemetry
- No large predictive framework: simple rolling statistics only
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
LIFECYCLE_WINDOW: int = 10        # rolling window for lifecycle metrics
DECAY_THRESHOLD: float = 0.35     # win_rate below this → decay flag
PERSISTENCE_WINDOW: int = 5       # recent trades for fast persistence check
MIN_TRADES_FOR_SIGNAL: int = 3    # minimum before lifecycle is considered valid
SCHEMA_VERSION: int = 1


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class AlphaRecord:
    """Single completed trade observation. Immutable after creation."""
    symbol: str
    trade_date: str          # YYYY-MM-DD (exit/completion date)
    return_bps: float        # actual realized return in bps (gross)
    post_cost_bps: float     # net of slippage + commission
    regime: str              # regime tag at entry ("BULL" | "BEAR" | "UNKNOWN")
    holding_days: int        # actual holding period


@dataclass
class AlphaLifecycleState:
    """Per-symbol lifecycle state. Rebuilt from trade records on replay."""
    symbol: str
    n_trades: int = 0
    n_wins: int = 0                          # trades with post_cost_bps > 0
    rolling_win_rate: float = 0.0            # over last LIFECYCLE_WINDOW trades
    rolling_avg_return_bps: float = 0.0      # over last LIFECYCLE_WINDOW trades
    rolling_avg_post_cost_bps: float = 0.0   # over last LIFECYCLE_WINDOW trades
    edge_persistence: float = 0.5            # [0, 1] stability score
    decay_flag: bool = False                  # True when win_rate < DECAY_THRESHOLD
    post_cost_retention: float = 0.0         # avg post_cost / avg gross [0, 1]
    regime_win_rates: dict = field(default_factory=dict)  # {regime: win_rate}
    last_trade_date: str = ""
    schema_version: int = SCHEMA_VERSION


# ── Pure helpers ───────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    n = len(values)
    mean = _mean(values)
    return (sum((x - mean) ** 2 for x in values) / (n - 1)) ** 0.5


def _compute_edge_persistence(
    returns: list[float],
    window: int = LIFECYCLE_WINDOW,
    fast_window: int = PERSISTENCE_WINDOW,
) -> float:
    """
    Edge persistence = stability × level of recent win rate.
    Low coefficient-of-variation + positive mean → high persistence.
    Uses only past observations. No leakage.
    """
    if len(returns) < MIN_TRADES_FOR_SIGNAL:
        return 0.5  # neutral / unknown

    recent = returns[-window:]
    wins = [1.0 if r > 0 else 0.0 for r in recent]
    win_rate = _mean(wins)

    # Stability: compare fast window vs full window consistency
    fast_wins = [1.0 if r > 0 else 0.0 for r in returns[-fast_window:]]
    fast_wr = _mean(fast_wins) if fast_wins else win_rate
    consistency = 1.0 - abs(fast_wr - win_rate)  # [0, 1]

    # Persistence = win_rate * consistency
    persistence = win_rate * consistency
    return round(max(0.0, min(1.0, persistence)), 4)


def _compute_regime_win_rates(records: list[AlphaRecord]) -> dict:
    """Win rate by regime from all records. No leakage (uses completed trades)."""
    regime_data: dict[str, list[float]] = {}
    for rec in records:
        regime_data.setdefault(rec.regime, []).append(
            1.0 if rec.post_cost_bps > 0 else 0.0
        )
    return {
        regime: round(_mean(wins), 4)
        for regime, wins in regime_data.items()
    }


# ── Core update ────────────────────────────────────────────────────────────────

def build_lifecycle_state(
    symbol: str,
    records: list[AlphaRecord],
) -> AlphaLifecycleState:
    """
    Rebuild lifecycle state from complete trade history.
    Deterministic replay: always produces the same result for same records.
    No future leakage: records must be sorted by trade_date ascending.
    """
    if not records:
        return AlphaLifecycleState(symbol=symbol)

    n = len(records)
    n_wins = sum(1 for r in records if r.post_cost_bps > 0)
    recent = records[-LIFECYCLE_WINDOW:]

    recent_returns = [r.post_cost_bps for r in recent]
    recent_gross = [r.return_bps for r in recent]
    recent_wins = sum(1 for r in recent if r.post_cost_bps > 0)
    recent_n = len(recent)

    rolling_win_rate = recent_wins / recent_n if recent_n > 0 else 0.0
    rolling_avg_post = _mean(recent_returns)
    rolling_avg_gross = _mean(recent_gross)

    # Post-cost retention: how much gross alpha survives costs
    if rolling_avg_gross > 0.0:
        retention = max(0.0, min(1.0, rolling_avg_post / rolling_avg_gross))
    else:
        retention = 0.0

    all_returns = [r.post_cost_bps for r in records]
    persistence = _compute_edge_persistence(all_returns)
    decay = rolling_win_rate < DECAY_THRESHOLD and recent_n >= MIN_TRADES_FOR_SIGNAL
    regime_rates = _compute_regime_win_rates(records)
    last_date = records[-1].trade_date if records else ""

    return AlphaLifecycleState(
        symbol=symbol,
        n_trades=n,
        n_wins=n_wins,
        rolling_win_rate=round(rolling_win_rate, 4),
        rolling_avg_return_bps=round(rolling_avg_gross, 2),
        rolling_avg_post_cost_bps=round(rolling_avg_post, 2),
        edge_persistence=persistence,
        decay_flag=decay,
        post_cost_retention=round(retention, 4),
        regime_win_rates=regime_rates,
        last_trade_date=last_date,
    )


def append_alpha_record(record: AlphaRecord, path: Path) -> None:
    """Append completed trade record to per-symbol JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_alpha_records(path: Path) -> list[AlphaRecord]:
    """Load all alpha records from JSONL. Returns empty list if missing."""
    if not path.exists():
        return []
    records: list[AlphaRecord] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(AlphaRecord(
                symbol=data.get("symbol", ""),
                trade_date=data.get("trade_date", ""),
                return_bps=float(data.get("return_bps", 0.0)),
                post_cost_bps=float(data.get("post_cost_bps", 0.0)),
                regime=data.get("regime", "UNKNOWN"),
                holding_days=int(data.get("holding_days", 0)),
            ))
    except Exception as exc:
        logger.warning("alpha_records load failed (%s)", exc)
    return records


def save_lifecycle_state(state: AlphaLifecycleState, path: Path) -> None:
    """Save lifecycle state as JSON (atomic write)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_lifecycle_state(path: Path, symbol: str) -> AlphaLifecycleState:
    if not path.exists():
        return AlphaLifecycleState(symbol=symbol)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AlphaLifecycleState(
            symbol=data.get("symbol", symbol),
            n_trades=int(data.get("n_trades", 0)),
            n_wins=int(data.get("n_wins", 0)),
            rolling_win_rate=float(data.get("rolling_win_rate", 0.0)),
            rolling_avg_return_bps=float(data.get("rolling_avg_return_bps", 0.0)),
            rolling_avg_post_cost_bps=float(data.get("rolling_avg_post_cost_bps", 0.0)),
            edge_persistence=float(data.get("edge_persistence", 0.5)),
            decay_flag=bool(data.get("decay_flag", False)),
            post_cost_retention=float(data.get("post_cost_retention", 0.0)),
            regime_win_rates=data.get("regime_win_rates", {}),
            last_trade_date=data.get("last_trade_date", ""),
        )
    except Exception as exc:
        logger.warning("lifecycle_state load failed (%s) — default", exc)
        return AlphaLifecycleState(symbol=symbol)


def build_lifecycle_map(
    lifecycle_dir: Path,
    symbols: list[str],
) -> dict:
    """
    Load or rebuild lifecycle states for a list of symbols.
    Returns {symbol: AlphaLifecycleState}.
    Each symbol's records file is at lifecycle_dir/{symbol}.jsonl.
    """
    result: dict[str, AlphaLifecycleState] = {}
    for sym in symbols:
        records_file = lifecycle_dir / f"{sym}.jsonl"
        records = load_alpha_records(records_file)
        result[sym] = build_lifecycle_state(sym, records)
    return result
