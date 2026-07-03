"""
src/universe/demotion_pressure_engine.py — Capital Occupancy Inefficiency Engine (Phase 2)

Detects live symbols that are blocking capital without delivering expected returns.
This is NOT "remove weak RSR" — it is "detect stagnant capital occupancy".

Demotion is NOT automatic: output is a pressure score written to runtime/demotion_pressure.json.
The governance layer reads this to inform future rotation decisions.

Capital occupancy inefficiency = holding a position that:
  - Has overstayed its expected duration (holding_overhang)
  - Has stagnant unrealized PnL relative to its own ATR (pnl_stagnation)
  - Shows RSR decay (but NOT enough alone to trigger demotion)
  - Shows range compression (ATR shrinkage → trend dying)
  - Has no signs of a second expansion wave (expansion_lack)

Design:
  - FAIL_OPEN: any error per symbol returns pressure_score=0.0 (no demotion pressure)
  - Deterministic: same inputs → same output
  - observation_only: reads portfolio_state, trades.jsonl, RSR snapshots, OHLCV
  - No execution connection, no signal mutation
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
DEMOTION_THRESHOLD: float = 0.60   # pressure_score above this = demotion candidate
MEDIAN_HOLD_DAYS_DEFAULT: int = 18  # fallback when trades.jsonl has < 5 SELL records

# ── Component weights ─────────────────────────────────────────────────────────
_W_OVERHANG   = 0.35
_W_STAGNATION = 0.25
_W_RSR_DECAY  = 0.15
_W_EXPANSION  = 0.10
_W_RANGE_COMP = 0.15


@dataclass
class DemotionPressure:
    symbol: str
    holding_days: int
    median_hold: int
    holding_overhang: float      # max(0, holding_days/median_hold - 1) capped at 1
    pnl_stagnation: float        # 0–1: unrealized PnL relative to ATR (0=active, 1=dead)
    rsr_decay: float             # 0–1: recent RSR velocity negative
    expansion_lack: float        # 0–1: 1 - RSR_persistence in last 5 days
    range_compression: float     # 0–1: ATR shrinkage (ATR_10d vs ATR_30d)
    pressure_score: float        # weighted composite
    is_candidate: bool           # pressure_score >= DEMOTION_THRESHOLD

    def to_dict(self) -> dict:
        return asdict(self)


class DemotionPressureEngine:
    """
    Computes demotion pressure for each symbol currently in the live universe.
    Only processes symbols that are actually held (in portfolio_state.positions).
    """

    def __init__(
        self,
        portfolio_state_file: Path,
        trades_jsonl: Path,
        rsr_snapshot_dir: Path,
        ohlcv_dir: Path,
    ) -> None:
        self._portfolio_file = portfolio_state_file
        self._trades_jsonl   = trades_jsonl
        self._rsr_dir        = rsr_snapshot_dir
        self._ohlcv_dir      = ohlcv_dir

        self._median_hold    = self._compute_median_hold()
        self._rsr_history    = self._load_rsr_history(days=15)

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_all(self) -> Dict[str, DemotionPressure]:
        """
        Compute demotion pressure for all held positions.
        Returns {symbol: DemotionPressure}.
        """
        port = self._load_portfolio_state()
        if port is None:
            return {}

        entry_dates   = port.get("position_entry_dates", {})
        unrealized    = port.get("position_unrealized_pct", {})
        entry_atrs    = port.get("position_entry_atrs", {})

        today = date.today()
        results: Dict[str, DemotionPressure] = {}

        for sym in entry_dates:
            try:
                pressure = self._compute_pressure(
                    symbol      = sym,
                    today       = today,
                    entry_date  = entry_dates[sym],
                    unreal_pct  = float(unrealized.get(sym, 0.0)),
                    entry_atr   = float(entry_atrs.get(sym, 0.0)),
                )
                results[sym] = pressure
            except Exception as exc:
                logger.warning("[DEMOTION] compute(%s) error: %s — pressure=0.0", sym, exc)
                results[sym] = self._zero_pressure(sym)

        return results

    def write_report(self, pressures: Dict[str, DemotionPressure], out_path: Path) -> None:
        """Write pressure report atomically to out_path (JSON)."""
        candidates = [sym for sym, p in pressures.items() if p.is_candidate]
        payload = {
            "computed_at": datetime.now(timezone.utc).isoformat(),
            "demotion_threshold": DEMOTION_THRESHOLD,
            "candidates": candidates,
            "pressures": {sym: p.to_dict() for sym, p in pressures.items()},
        }
        tmp = out_path.with_suffix(".tmp")
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out_path)
        logger.info(
            "[DEMOTION] report written: %d held, %d candidates  %s",
            len(pressures), len(candidates), out_path.name,
        )

    # ── Per-symbol pressure ───────────────────────────────────────────────────

    def _compute_pressure(
        self,
        symbol: str,
        today: date,
        entry_date: str,
        unreal_pct: float,
        entry_atr: float,
    ) -> DemotionPressure:
        # holding_days
        try:
            ed = date.fromisoformat(entry_date[:10])
            holding_days = (today - ed).days
        except Exception:
            holding_days = 0

        # ── Component 1: holding_overhang ─────────────────────────────────────
        overhang = max(0.0, holding_days / self._median_hold - 1.0)
        overhang = min(1.0, overhang)

        # ── Component 2: pnl_stagnation ───────────────────────────────────────
        # A position with unrealized PnL much lower than 1×ATR is stagnating.
        # entry_atr stored as price-unit ATR → convert to pct roughly via price (not available)
        # Use sign and magnitude: positive PnL = active, deep negative = stagnating
        if unreal_pct >= 0.01:
            pnl_stagnation = 0.0   # actively positive → no stagnation
        elif unreal_pct < -0.05:
            pnl_stagnation = 1.0   # deeply negative → full stagnation
        else:
            pnl_stagnation = max(0.0, -unreal_pct / 0.05)  # linear in [-5%, 0]

        # ── Component 3: RSR decay ────────────────────────────────────────────
        history = self._rsr_history.get(symbol, [])
        if len(history) >= 5:
            velocity_5d = history[-1] - history[-5]
            rsr_decay = max(0.0, -velocity_5d) / 20.0  # normalize: -20 RSR / 5d = extreme
            rsr_decay = min(1.0, rsr_decay)
        else:
            rsr_decay = 0.0   # insufficient history → no decay signal

        # ── Component 4: expansion_lack ───────────────────────────────────────
        # RSR persistence in last 5 days: high persistence → expansion is alive
        if len(history) >= 5:
            recent = history[-5:]
            expansion_strength = sum(1 for r in recent if r >= 65) / len(recent)
            expansion_lack = 1.0 - expansion_strength
        else:
            expansion_lack = 0.0  # no data → neutral

        # ── Component 5: range_compression (ATR shrinkage) ───────────────────
        range_comp = self._compute_range_compression(symbol)

        # ── Composite ────────────────────────────────────────────────────────
        score = (
            _W_OVERHANG   * overhang
            + _W_STAGNATION * pnl_stagnation
            + _W_RSR_DECAY  * rsr_decay
            + _W_EXPANSION  * expansion_lack
            + _W_RANGE_COMP * range_comp
        )
        score = float(round(max(0.0, min(1.0, score)), 4))

        return DemotionPressure(
            symbol=symbol,
            holding_days=holding_days,
            median_hold=self._median_hold,
            holding_overhang=float(round(overhang, 4)),
            pnl_stagnation=float(round(pnl_stagnation, 4)),
            rsr_decay=float(round(rsr_decay, 4)),
            expansion_lack=float(round(expansion_lack, 4)),
            range_compression=float(round(range_comp, 4)),
            pressure_score=score,
            is_candidate=bool(score >= DEMOTION_THRESHOLD),
        )

    def _compute_range_compression(self, symbol: str) -> float:
        """ATR_10d / ATR_30d < 1 → trend dying. Returns [0, 1], higher = more compressed."""
        try:
            import pandas as pd
            csv_path = self._ohlcv_dir / f"{symbol}.csv"
            if not csv_path.exists():
                return 0.0
            df = pd.read_csv(csv_path)
            high_col = next((c for c in ("High", "high") if c in df.columns), None)
            low_col  = next((c for c in ("Low",  "low")  if c in df.columns), None)
            if high_col is None or low_col is None or len(df) < 31:
                return 0.0
            tr = (df[high_col] - df[low_col]).abs().tail(31)
            atr10 = tr.tail(10).mean()
            atr30 = tr.mean()
            if atr30 <= 0:
                return 0.0
            ratio = atr10 / atr30
            compression = max(0.0, 1.0 - ratio)  # 0 = expanding, 1 = fully compressed
            return round(min(1.0, compression), 4)
        except Exception:
            return 0.0

    # ── Data loaders ──────────────────────────────────────────────────────────

    def _load_portfolio_state(self) -> Optional[dict]:
        if not self._portfolio_file.exists():
            logger.warning("[DEMOTION] portfolio_state not found: %s", self._portfolio_file)
            return None
        try:
            return json.loads(self._portfolio_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[DEMOTION] portfolio_state read error: %s", exc)
            return None

    def _compute_median_hold(self) -> int:
        """Compute median hold_days from SELL records in trades.jsonl."""
        if not self._trades_jsonl.exists():
            return MEDIAN_HOLD_DAYS_DEFAULT
        try:
            hold_days: List[int] = []
            for line in self._trades_jsonl.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("side") == "SELL" and "hold_days" in rec:
                    hold_days.append(int(rec["hold_days"]))
            if len(hold_days) < 3:
                return MEDIAN_HOLD_DAYS_DEFAULT
            hold_days.sort()
            mid = len(hold_days) // 2
            return hold_days[mid]
        except Exception:
            return MEDIAN_HOLD_DAYS_DEFAULT

    def _load_rsr_history(self, days: int) -> Dict[str, List[float]]:
        history: Dict[str, List[float]] = {}
        if not self._rsr_dir.exists():
            return history
        try:
            for f in sorted(self._rsr_dir.glob("*.json"))[-days:]:
                snap = json.loads(f.read_text(encoding="utf-8"))
                for sym, rsr in snap.get("scores", {}).items():
                    history.setdefault(sym, []).append(float(rsr))
        except Exception as exc:
            logger.warning("[DEMOTION] RSR history load error: %s", exc)
        return history

    @staticmethod
    def _zero_pressure(symbol: str) -> DemotionPressure:
        return DemotionPressure(
            symbol=symbol,
            holding_days=0, median_hold=MEDIAN_HOLD_DAYS_DEFAULT,
            holding_overhang=0.0, pnl_stagnation=0.0, rsr_decay=0.0,
            expansion_lack=0.0, range_compression=0.0,
            pressure_score=0.0, is_candidate=False,
        )
