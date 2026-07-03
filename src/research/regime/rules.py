"""Regime rule constants — all deterministic, no ML, no randomness."""

TREND_UP = "trend_up"
TREND_DOWN = "trend_down"
RANGE = "range"
HIGH_VOL = "high_vol"
LOW_VOL = "low_vol"
PANIC = "panic"
GAP_UP = "gap_up"
GAP_DOWN = "gap_down"

ALL_REGIMES: tuple = (
    TREND_UP, TREND_DOWN, RANGE,
    HIGH_VOL, LOW_VOL,
    PANIC, GAP_UP, GAP_DOWN,
)

# Priority order for label assignment (lower index = higher priority)
REGIME_PRIORITY: tuple = (GAP_UP, GAP_DOWN, PANIC, HIGH_VOL, LOW_VOL, TREND_UP, TREND_DOWN, RANGE)

# ── Gap detection ──────────────────────────────────────────────────────────────
GAP_UP_THRESH: float = 0.02    # open/prev_close - 1 > 2%
GAP_DOWN_THRESH: float = -0.02

# ── Panic detection ────────────────────────────────────────────────────────────
PANIC_RETURN_THRESH: float = -0.03           # single-day return < -3%
PANIC_VOL_PERCENTILE: float = 0.80           # ATR% must be in top 20%

# ── Trend detection ────────────────────────────────────────────────────────────
ADX_TREND_THRESH: float = 25.0
ATR_PERIOD: int = 14
MA_PERIOD: int = 20

# ── Volatility regime ─────────────────────────────────────────────────────────
VOL_LOOKBACK: int = 60
HIGH_VOL_PERCENTILE: float = 0.75
LOW_VOL_PERCENTILE: float = 0.25
