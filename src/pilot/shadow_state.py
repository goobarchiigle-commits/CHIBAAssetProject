"""
pilot/shadow_state.py — Study13 pilot state machine

PHASE1_SHADOW → PHASE2_LIVE → PHASE3_PROMOTE / STOPPED
"""
from __future__ import annotations

import hashlib, json, math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

PILOT_STATE_FILE = Path("runtime/study13_pilot_state.json")
PILOT_LOG_DIR    = Path("logs/pilot")
PILOT_REPORT_DIR = Path("reports/study13_pilot")

PHASE1 = "PHASE1_SHADOW"
PHASE2 = "PHASE2_LIVE"
PHASE3 = "PHASE3_PROMOTE"
STOPPED = "STOPPED"

# Promotion thresholds
PHASE1_DAYS_MIN     = 30
TRACKING_ERR_MAX    = 0.01    # 1.0%
REJECT_RATE_MAX     = 0.005   # 0.5%
LATENCY_P99_MAX_MS  = 5000.0
BACKTEST_CALMAR     = 1.439   # Study12 ideal avg calmar (Study9 Case B + annual)

# Phase 2 safety
KILL_SWITCH_DD_SLACK = 2.0    # pp above prod_DD
FREEZE_REJECT_CONSEC = 3
ROLLBACK_ALPHA_REAL  = 0.80   # rolling 30d
DEGRADE_FILL_RATE    = 0.95


@dataclass
class ShadowPosition:
    symbol:      str
    qty:         int
    entry_price: float
    entry_date:  str
    entry_idx:   int   # day index within pilot run
    rsr_entry:   float
    d90_entry:   int
    days_below:  int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ShadowPosition":
        return cls(**d)


@dataclass
class PilotState:
    phase:            str   = PHASE1
    mode:             str   = "backtest"   # "backtest" | "live_shadow" | "live"
    start_date:       str   = ""
    last_date:        str   = ""
    total_days:       int   = 0
    trading_days:     int   = 0           # days where market was open
    shadow_cash:      float = 750_000.0
    rebal_withdrawn:  float = 0.0
    rebal_injected:   float = 0.0
    rebal_events:     int   = 0
    position:         Optional[dict] = None   # ShadowPosition.to_dict()

    # ── Audit counters ────────────────────────────────────────────────
    n_orders:              int   = 0
    n_filled:              int   = 0
    n_rejected:            int   = 0
    n_position_diff_days:  int   = 0
    n_cash_drift_events:   int   = 0
    n_state_hash_break:    int   = 0
    n_duplicate_guard:     int   = 0
    n_missed_fill:         int   = 0
    n_order_retry:         int   = 0
    n_rollback:            int   = 0
    n_manual_intervention: int   = 0

    # Latency tracking (ms per order, shadow=0ms)
    latency_ms_samples: list[float] = field(default_factory=list)

    # Equity / tracking
    equity_history:    list[float] = field(default_factory=list)
    equity_dates:      list[str]   = field(default_factory=list)
    expected_equity:   list[float] = field(default_factory=list)   # from ideal backtest
    tracking_errors:   list[float] = field(default_factory=list)   # abs diff

    # State hashes (one per day) — detect corruption
    state_hashes:      list[str]   = field(default_factory=list)

    # Phase 2 live monitoring
    live_pnl_history:   list[float] = field(default_factory=list)
    consec_reject:      int   = 0   # consecutive broker rejects
    phase2_start_date:  str   = ""
    phase2_days:        int   = 0

    def shadow_equity(self, close_prices: dict[str, float]) -> float:
        pos_val = 0.0
        if self.position:
            p = ShadowPosition.from_dict(self.position)
            pos_val = p.qty * close_prices.get(p.symbol, p.entry_price)
        return self.shadow_cash + pos_val

    def state_hash(self, ds: str) -> str:
        payload = json.dumps({
            "date":     ds,
            "cash":     round(self.shadow_cash, 2),
            "position": self.position,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def latency_p99(self) -> float:
        if not self.latency_ms_samples: return 0.0
        s = sorted(self.latency_ms_samples)
        i = max(0, int(len(s) * 0.99) - 1)
        return s[i]

    def tracking_error_max(self) -> float:
        return max(self.tracking_errors, default=0.0)

    def broker_reject_rate(self) -> float:
        return self.n_rejected / max(1, self.n_orders)

    def save(self, path: Path = PILOT_STATE_FILE) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path = PILOT_STATE_FILE) -> "PilotState":
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
