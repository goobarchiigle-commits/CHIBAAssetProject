"""
src/universe/universe_ev_engine.py — Universe Expected Value Engine (Phase 2)

Scores live and shadow symbols by expected CAGR contribution, not RSR rank alone.
Provides replacement_delta gate and exploration budget for promotion decisions.

Design:
  - observation_only for analytics sources (reads JSONL, never writes to analytics)
  - Bayesian expectancy: backtest prior + live update (weight grows with sample count)
  - RSR acceleration: 2nd derivative of RSR trend from daily snapshots
  - Rotation cost model: commission + slippage penalty in EV units
  - FAIL_OPEN: all computation errors return neutral defaults

Invariants:
  - No execution path connection
  - No signal mutation / universe write (governance handles that)
  - No ML model / no optimizer / no adaptive parameter tuning
  - Deterministic: same inputs → same output
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Scoring weights ───────────────────────────────────────────────────────────
EV_W_EXPECTANCY    = 0.35
EV_W_SURVIVABILITY = 0.25
EV_W_RSR_ACCEL     = 0.20
EV_W_LIQUIDITY     = 0.10
EV_W_EXEC_STAB     = 0.10

# ── Rotation cost in EV-score units ──────────────────────────────────────────
# Covers commission (0.055%×2) + slippage (0.1%×2) + spread (0.05%×2) ≈ 0.41%
# Mapped to EV scale: a swap that gains < 0.05 EV points is not worth the cost.
ROTATION_COST_EV: float = 0.05

# ── Promotion gate ────────────────────────────────────────────────────────────
PROMOTION_EV_THRESHOLD: float = 0.05  # net_advantage must clear this after cost

# ── Exploration budget ────────────────────────────────────────────────────────
EXPLORATION_SAMPLE_THRESHOLD: int = 3  # symbols with < N live trades → explorable

# ── Backtest priors (from dyn_rsr42_bear_rs0 IS 2018-2024) ────────────────────
_PRIOR_WIN_RATE: float = 0.616   # long 11d+: 61.6% (research_state.md)
_PRIOR_PAYOFF_R: float = 2.16    # R-ratio from IS (avg_win / avg_loss)
_PRIOR_EV_RAW   = _PRIOR_WIN_RATE - (1.0 - _PRIOR_WIN_RATE) / _PRIOR_PAYOFF_R  # ≈ 0.430
_PRIOR_EV_NORM  = min(1.0, max(0.0, _PRIOR_EV_RAW / 0.5))  # normalize; 0.5 = ceiling

# ── RSR acceleration normalization ───────────────────────────────────────────
_RSR_ACCEL_RANGE: float = 20.0   # ±20 RSR points per 5-day window = extreme


@dataclass
class EVScore:
    symbol: str
    expectancy_score: float      # 0–1: Bayesian win_rate × payoff quality
    survivability: float         # 0–1: RSR persistence proxy
    rsr_acceleration: float      # 0–1: RSR momentum 2nd derivative
    liquidity_quality: float     # 0–1: volume stability
    exec_stability: float        # 0–1: execution consistency (0.5 until data matures)
    composite: float             # weighted composite [0, 1]
    sample_n: int                # live SELL count (drives Bayesian weight)
    source: str                  # "live+prior" | "prior_only" | "default"

    def to_dict(self) -> dict:
        return asdict(self)


class UniverseEVEngine:
    """
    Computes EV scores for universe symbols.
    All I/O is read-only from existing log files and OHLCV cache.
    """

    def __init__(
        self,
        ohlcv_dir: Path,
        trades_jsonl: Path,
        rsr_snapshot_dir: Path,
        rsr_history_days: int = 15,
    ) -> None:
        self._ohlcv_dir        = ohlcv_dir
        self._trades_jsonl     = trades_jsonl
        self._rsr_snapshot_dir = rsr_snapshot_dir

        self._live_trades: Dict[str, List[dict]] = self._load_live_trades()
        self._rsr_history: Dict[str, List[float]] = self._load_rsr_history(rsr_history_days)

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_ev_score(self, symbol: str) -> EVScore:
        """Compute EV score for a single symbol. Returns neutral default on error."""
        try:
            expectancy   = self._compute_expectancy(symbol)
            survivability= self._compute_survivability(symbol)
            rsr_accel    = self._compute_rsr_acceleration(symbol)
            liquidity    = self._compute_liquidity(symbol)
            exec_stab    = 0.5  # neutral default; improves as execution data grows

            sample_n = len(self._live_trades.get(symbol, []))
            source   = "live+prior" if sample_n > 0 else "prior_only"

            composite = (
                EV_W_EXPECTANCY    * expectancy
                + EV_W_SURVIVABILITY * survivability
                + EV_W_RSR_ACCEL   * rsr_accel
                + EV_W_LIQUIDITY   * liquidity
                + EV_W_EXEC_STAB   * exec_stab
            )
            composite = round(max(0.0, min(1.0, composite)), 4)

            return EVScore(
                symbol=symbol,
                expectancy_score=round(expectancy, 4),
                survivability=round(survivability, 4),
                rsr_acceleration=round(rsr_accel, 4),
                liquidity_quality=round(liquidity, 4),
                exec_stability=round(exec_stab, 4),
                composite=composite,
                sample_n=sample_n,
                source=source,
            )
        except Exception as exc:
            logger.warning("[EV_ENGINE] compute_ev_score(%s) error: %s", symbol, exc)
            return self._neutral_score(symbol)

    def compute_all_scores(self, universe: Dict[str, str]) -> Dict[str, EVScore]:
        """Score all symbols in a universe dict {symbol: sector}."""
        return {sym: self.compute_ev_score(sym) for sym in universe}

    def compute_replacement_delta(
        self,
        candidate: EVScore,
        live_ev_scores: Dict[str, EVScore],
    ) -> Tuple[float, Optional[str]]:
        """
        Returns (net_advantage, weakest_live_symbol).

        net_advantage = candidate.composite - weakest_live.composite - ROTATION_COST_EV

        Positive net_advantage means the swap is expected to improve portfolio EV
        even after accounting for rotation costs.
        """
        if not live_ev_scores:
            net_adv = candidate.composite - ROTATION_COST_EV
            return round(net_adv, 4), None
        weakest_sym = min(live_ev_scores, key=lambda s: live_ev_scores[s].composite)
        weakest_ev  = live_ev_scores[weakest_sym].composite
        net_adv     = candidate.composite - weakest_ev - ROTATION_COST_EV
        return round(net_adv, 4), weakest_sym

    def is_exploration_candidate(self, score: EVScore) -> bool:
        """True if symbol has high uncertainty (few live samples → exploit exploration budget)."""
        return score.sample_n < EXPLORATION_SAMPLE_THRESHOLD

    def live_ev_floor(self, live_ev_scores: Dict[str, EVScore]) -> float:
        """Minimum EV composite across live universe (the symbol to beat)."""
        if not live_ev_scores:
            return 0.0
        return min(s.composite for s in live_ev_scores.values())

    # ── Expectancy ────────────────────────────────────────────────────────────

    def _compute_expectancy(self, symbol: str) -> float:
        trades = self._live_trades.get(symbol, [])
        if not trades:
            return _PRIOR_EV_NORM

        wins      = [t["pnl_pct"] for t in trades if t["pnl_pct"] > 0]
        losses    = [abs(t["pnl_pct"]) for t in trades if t["pnl_pct"] <= 0]
        win_rate  = len(wins) / len(trades)
        avg_win   = sum(wins)   / len(wins)   if wins   else 0.01
        avg_loss  = sum(losses) / len(losses) if losses else 0.01
        payoff    = avg_win / avg_loss if avg_loss > 0 else 1.0
        live_ev   = win_rate - (1.0 - win_rate) / payoff
        live_norm = max(0.0, min(1.0, live_ev / 0.5))

        # Bayesian: prior weight decays as live samples grow (full live at n=20)
        live_w   = min(1.0, len(trades) / 20.0)
        blended  = _PRIOR_EV_NORM * (1.0 - live_w) + live_norm * live_w
        return max(0.0, min(1.0, blended))

    # ── Survivability ─────────────────────────────────────────────────────────

    def _compute_survivability(self, symbol: str) -> float:
        """RSR persistence proxy: fraction of last N days with RSR ≥ 65."""
        history = self._rsr_history.get(symbol, [])
        if len(history) < 3:
            return 0.5
        above = sum(1 for r in history if r >= 65) / len(history)
        return above

    # ── RSR acceleration ──────────────────────────────────────────────────────

    def _compute_rsr_acceleration(self, symbol: str) -> float:
        """
        RSR 2nd derivative.  Positive = momentum accelerating → high score.
        Uses two 5-day velocity windows.
        """
        history = self._rsr_history.get(symbol, [])
        if len(history) < 6:
            return 0.5  # neutral

        vel_recent = history[-1] - history[-min(6, len(history))]  # last 5 days
        if len(history) >= 11:
            vel_older = history[-6] - history[-11]
        elif len(history) >= 6:
            vel_older = history[-3] - history[-6] if len(history) >= 6 else 0.0
        else:
            vel_older = 0.0

        accel = vel_recent - vel_older
        # Normalize: ±_RSR_ACCEL_RANGE maps to [0, 1]
        normalized = max(0.0, min(1.0, (accel + _RSR_ACCEL_RANGE) / (2.0 * _RSR_ACCEL_RANGE)))
        return normalized

    # ── Liquidity quality ─────────────────────────────────────────────────────

    def _compute_liquidity(self, symbol: str) -> float:
        """Volume stability: low coefficient of variation → high quality."""
        try:
            import pandas as pd
            csv_path = self._ohlcv_dir / f"{symbol}.csv"
            if not csv_path.exists():
                return 0.5
            df = pd.read_csv(csv_path)
            vol_col = next((c for c in ("Volume", "volume") if c in df.columns), None)
            if vol_col is None or len(df) < 20:
                return 0.5
            vol = df[vol_col].tail(20).astype(float)
            mean_vol = vol.mean()
            if mean_vol <= 0:
                return 0.0
            cv = vol.std() / mean_vol  # coefficient of variation (lower = better)
            return max(0.0, min(1.0, 1.0 - cv))
        except Exception:
            return 0.5

    # ── Data loaders ──────────────────────────────────────────────────────────

    def _load_live_trades(self) -> Dict[str, List[dict]]:
        """Parse SELL records from trades.jsonl → {symbol: [{pnl_pct, hold_days}, ...]}."""
        trades: Dict[str, List[dict]] = {}
        if not self._trades_jsonl.exists():
            return trades
        try:
            for line in self._trades_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("side") != "SELL":
                    continue
                sym = rec["symbol"]
                trades.setdefault(sym, []).append({
                    "pnl_pct":   float(rec.get("pnl_pct", 0.0)),
                    "hold_days": int(rec.get("hold_days", 0)),
                })
        except Exception as exc:
            logger.warning("[EV_ENGINE] trades.jsonl load error: %s", exc)
        return trades

    def _load_rsr_history(self, days: int) -> Dict[str, List[float]]:
        """Load last N daily RSR snapshots → {symbol: [score_oldest, ..., score_newest]}."""
        history: Dict[str, List[float]] = {}
        if not self._rsr_snapshot_dir.exists():
            return history
        try:
            snap_files = sorted(self._rsr_snapshot_dir.glob("*.json"))[-days:]
            for f in snap_files:
                snap = json.loads(f.read_text(encoding="utf-8"))
                for sym, rsr in snap.get("scores", {}).items():
                    history.setdefault(sym, []).append(float(rsr))
        except Exception as exc:
            logger.warning("[EV_ENGINE] RSR history load error: %s", exc)
        return history

    @staticmethod
    def _neutral_score(symbol: str) -> EVScore:
        return EVScore(
            symbol=symbol,
            expectancy_score=0.5,
            survivability=0.5,
            rsr_acceleration=0.5,
            liquidity_quality=0.5,
            exec_stability=0.5,
            composite=0.5,
            sample_n=0,
            source="default",
        )


# ── Module-level helper for run_live_signal.py ───────────────────────────────

def build_ev_engine(paths_module) -> "UniverseEVEngine":
    """
    Convenience factory using project paths.
    paths_module: the src.paths module (passed to avoid circular imports).
    """
    from src.paths import OHLCV_DIR, RSR_SNAPSHOT_DIR
    trades_path = paths_module.LOGS_DIR / "trades.jsonl"
    return UniverseEVEngine(
        ohlcv_dir        = OHLCV_DIR,
        trades_jsonl     = trades_path,
        rsr_snapshot_dir = RSR_SNAPSHOT_DIR,
    )
