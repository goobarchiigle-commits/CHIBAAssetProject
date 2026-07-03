"""
compression_continuation.py — Healthy compression vs distribution for held positions.

Purpose:
  Distinguish VCP-style healthy compression (exit deferral candidate) from
  distribution (trend failure) for currently-held positions.

Design:
  Pure functions. No I/O. No market data loading. NaN-safe. Deterministic.
  Caller pre-computes all inputs from daily OHLCV and passes them in.

Phase classification:
  compression  — ATR縮小 + 出来高減少 + 高値維持 + RSR安定  → hold extension candidate
  distribution — 高値圏出来高増 + 安値切り下げ + RSR劣化  → accelerate exit
  breakout     — compression後のvolume expansion + 高値更新
  neutral      — none of the above

suppression_eligible=True:
  phase == "compression" AND score >= SUPPRESSION_SCORE_MIN
  AND rsr >= SUPPRESSION_RSR_MIN AND rsr_momentum >= SUPPRESSION_MOM_MIN
  AND no invalidation flags
  → runtime_exit_orchestrator may defer RSR_EXIT up to 3 business days.

Invalidation (override suppression_eligible → False regardless of score):
  compression_ratio > INVALID_RATIO_MAX  (ATR急拡大)
  rsr < INVALID_RSR_MIN                  (RSR閾値割れ)
  phase == "distribution"
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "v1"
_EPS = 1e-9

# ── Suppression eligibility thresholds ────────────────────────────────────────
SUPPRESSION_SCORE_MIN: float = 65.0   # minimum composite score
SUPPRESSION_RSR_MIN:   float = 75.0   # minimum RSR value
SUPPRESSION_MOM_MIN:   float = -1.5   # minimum rsr_momentum

# ── Phase gate thresholds: healthy compression ────────────────────────────────
COMPRESS_RATIO_MAX:     float = 0.75  # ATR5/ATR60 must be < this
COMPRESS_VOLUME_MAX:    float = 0.85  # vol5d/vol20d must be < this
COMPRESS_NEAR_HIGH_MIN: float = 0.70  # (close-low20)/(high20-low20) must be > this
COMPRESS_MOM_MIN:       float = -1.0  # rsr_momentum floor

# ── Phase gate thresholds: distribution ───────────────────────────────────────
DISTRIB_VOLUME_MIN: float = 1.20   # vol ratio above this
DISTRIB_HL_MAX:     int   = 0      # higher_low_streak must be <= this
DISTRIB_MOM_MAX:    float = -2.0   # rsr_momentum must be < this

# ── Invalidation thresholds ───────────────────────────────────────────────────
INVALID_RATIO_MAX: float = 0.90   # ATR expanding → override suppression
INVALID_RSR_MIN:   float = 72.0   # RSR floor → override suppression

# ── Score component weights (must sum to 1.0) ─────────────────────────────────
_W_COMPRESSION = 0.25
_W_TIGHTNESS   = 0.20
_W_VOLUME      = 0.20
_W_NEAR_HIGH   = 0.20
_W_HIGHER_LOW  = 0.10
_W_RSR         = 0.05


def _clip(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _safe(v: float, fallback: float = 0.0) -> float:
    """Return fallback if v is NaN, inf, or not a number."""
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Input / Output dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CompressionContinuationInput:
    """
    Per-symbol inputs for compression/distribution assessment.
    All numeric values pre-computed by caller from daily OHLCV.
    """
    symbol:             str
    hold_days:          int
    unrealized_pnl_pct: float   # fraction e.g. 0.05 = +5%

    # ── Reusable from predictive_expansion.py ─────────────────────────────────
    compression_ratio:  float   # ATR_5 / ATR_60;      < 0.75 = compressed
    price_tightness:    float   # std_5 / std_20;      < 0.60 = tight
    volume_ratio_5d:    float   # vol_5d / vol_20d avg; < 0.85 = drying up

    # ── Newly computed from OHLCV ─────────────────────────────────────────────
    higher_low_streak:  int     # consecutive days of higher lows (0 if broken)
    dip_recovery_pct:   float   # max pullback / ATR_20 (lower = shallower dip)
    near_high_pct:      float   # (close - low_20d) / (high_20d - low_20d)

    # ── From signal dict ──────────────────────────────────────────────────────
    rsr:          float   # RSR percentile [0, 100]
    rsr_momentum: float   # RSR change over recent period (signed)


@dataclass
class CompressionContinuationResult:
    """
    Phase classification and suppression eligibility for one held position.
    suppression_eligible=True → runtime_exit_orchestrator may defer RSR_EXIT.
    fail-safe default: score=0, suppression_eligible=False.
    """
    symbol:                 str
    score:                  float          # composite [0, 100]
    phase:                  str            # "compression" | "distribution" | "breakout" | "neutral"
    is_healthy_compression: bool
    is_distribution:        bool
    suppression_eligible:   bool
    invalidation_flags:     list[str]      # active invalidation conditions
    reasons:                list[str]      # score explanation tokens
    schema_version:         str = SCHEMA_VERSION

    # ── Component scores for telemetry (not used for decisions) ───────────────
    c_compression: float = 0.0
    c_tightness:   float = 0.0
    c_volume:      float = 0.0
    c_near_high:   float = 0.0
    c_higher_low:  float = 0.0
    c_rsr:         float = 0.0

    def to_dict(self) -> dict:
        return {
            "symbol":                 self.symbol,
            "score":                  round(self.score, 2),
            "phase":                  self.phase,
            "is_healthy_compression": self.is_healthy_compression,
            "is_distribution":        self.is_distribution,
            "suppression_eligible":   self.suppression_eligible,
            "invalidation_flags":     list(self.invalidation_flags),
            "reasons":                list(self.reasons),
            "schema_version":         self.schema_version,
            "components": {
                "compression":   round(self.c_compression, 2),
                "tightness":     round(self.c_tightness, 2),
                "volume":        round(self.c_volume, 2),
                "near_high":     round(self.c_near_high, 2),
                "higher_low":    round(self.c_higher_low, 2),
                "rsr_stability": round(self.c_rsr, 2),
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal computation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_components(inp: CompressionContinuationInput) -> tuple[float, float, float, float, float, float]:
    """
    Compute 6 component scores [0, 100] from sanitized inputs.
    Each dimension is independently clipped to avoid NaN/inf propagation.
    """
    cr  = _safe(inp.compression_ratio, 1.0)
    pt  = _safe(inp.price_tightness,   1.0)
    vr  = _safe(inp.volume_ratio_5d,   1.0)
    nh  = _safe(inp.near_high_pct,     0.5)
    hl  = max(0, int(inp.higher_low_streak))
    mom = _safe(inp.rsr_momentum,      0.0)

    # ATR縮小: ratio=0.50 → 100, ratio=0.85 → 0, ratio>0.85 → 0
    c_compress = _clip((0.85 - cr) / 0.35 * 100.0)
    # std縮小: tightness=0.45 → 100, tightness=0.80 → 0
    c_tight    = _clip((0.80 - pt) / 0.35 * 100.0)
    # 出来高減少: vol=0.45 → 100, vol=0.90 → 0, vol>0.90 → 0
    c_vol      = _clip((0.90 - vr) / 0.45 * 100.0)
    # 高値近辺維持: nh=1.0 → 100, nh=0.55 → 0, nh<0.55 → 0
    c_nh       = _clip((nh - 0.55) / 0.45 * 100.0)
    # 安値切り上げ: 4連続日 = 100 (25 points/day)
    c_hl       = _clip(hl * 25.0)
    # RSR安定: mom=+5 → 100, mom=0 → 50, mom=-5 → 0
    c_rsr      = _clip(50.0 + mom * 10.0)

    return c_compress, c_tight, c_vol, c_nh, c_hl, c_rsr


def _composite_score(
    c_compress: float, c_tight: float, c_vol: float,
    c_nh: float, c_hl: float, c_rsr: float,
) -> float:
    return round(
        _W_COMPRESSION * c_compress
        + _W_TIGHTNESS * c_tight
        + _W_VOLUME    * c_vol
        + _W_NEAR_HIGH * c_nh
        + _W_HIGHER_LOW * c_hl
        + _W_RSR       * c_rsr,
        2,
    )


def _classify_phase(inp: CompressionContinuationInput) -> str:
    """
    Classify holding phase. Distribution is checked first as negative signals
    dominate: a distribution reading overrides compression signals.
    """
    cr  = _safe(inp.compression_ratio, 1.0)
    vr  = _safe(inp.volume_ratio_5d,   1.0)
    nh  = _safe(inp.near_high_pct,     0.5)
    mom = _safe(inp.rsr_momentum,      0.0)
    hl  = max(0, int(inp.higher_low_streak))

    # Distribution: volume expanding at highs + higher lows breaking + RSR declining
    if vr > DISTRIB_VOLUME_MIN and hl <= DISTRIB_HL_MAX and mom < DISTRIB_MOM_MAX:
        return "distribution"

    # Breakout: volume expanding from compression with new highs and rising RSR
    if vr > 1.30 and nh > 0.85 and cr > 0.75 and mom > 0.0:
        return "breakout"

    # Healthy compression: all 4 conditions simultaneously
    if (cr < COMPRESS_RATIO_MAX
            and vr < COMPRESS_VOLUME_MAX
            and nh > COMPRESS_NEAR_HIGH_MIN
            and mom > COMPRESS_MOM_MIN):
        return "compression"

    return "neutral"


def _get_invalidations(inp: CompressionContinuationInput, phase: str) -> list[str]:
    """Return active invalidation flags. Any flag blocks suppression_eligible."""
    flags: list[str] = []
    if _safe(inp.compression_ratio, 1.0) > INVALID_RATIO_MAX:
        flags.append("atr_expanding")
    if _safe(inp.rsr, 100.0) < INVALID_RSR_MIN:
        flags.append("rsr_weak")
    if phase == "distribution":
        flags.append("distribution")
    return flags


def _score_reasons(
    inp: CompressionContinuationInput,
    c_compress: float, c_tight: float, c_vol: float,
    c_nh: float, c_hl: float, c_rsr: float,
    phase: str,
) -> list[str]:
    """Human-readable score decomposition tokens (for logging, not decisions)."""
    tokens: list[str] = [f"phase={phase}"]
    if c_compress >= 70:
        tokens.append("atr_compressed")
    elif c_compress <= 10:
        tokens.append("atr_wide")
    if c_vol >= 70:
        tokens.append("vol_drying")
    elif c_vol <= 10:
        tokens.append("vol_expanding")
    if c_nh >= 70:
        tokens.append("near_high")
    elif c_nh <= 10:
        tokens.append("far_from_high")
    hl = max(0, int(inp.higher_low_streak))
    if hl >= 3:
        tokens.append(f"hl_streak={hl}d")
    if c_rsr >= 70:
        tokens.append("rsr_stable")
    elif c_rsr <= 30:
        tokens.append("rsr_declining")
    return tokens


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_compression_continuation(
    inp: CompressionContinuationInput,
) -> CompressionContinuationResult:
    """
    Compute compression continuation assessment for a single held position.

    Fail-safe: on any exception returns a conservative result with
    suppression_eligible=False and score=0.0. Never raises.
    """
    try:
        c_compress, c_tight, c_vol, c_nh, c_hl, c_rsr = _compute_components(inp)
        score = _composite_score(c_compress, c_tight, c_vol, c_nh, c_hl, c_rsr)
        phase = _classify_phase(inp)
        invalidations = _get_invalidations(inp, phase)
        reasons = _score_reasons(inp, c_compress, c_tight, c_vol, c_nh, c_hl, c_rsr, phase)

        is_compression  = phase == "compression"
        is_distribution = phase == "distribution"

        suppression_eligible = (
            is_compression
            and score >= SUPPRESSION_SCORE_MIN
            and _safe(inp.rsr, 0.0) >= SUPPRESSION_RSR_MIN
            and _safe(inp.rsr_momentum, -99.0) >= SUPPRESSION_MOM_MIN
            and len(invalidations) == 0
        )

        return CompressionContinuationResult(
            symbol=inp.symbol,
            score=score,
            phase=phase,
            is_healthy_compression=is_compression,
            is_distribution=is_distribution,
            suppression_eligible=suppression_eligible,
            invalidation_flags=invalidations,
            reasons=reasons,
            c_compression=c_compress,
            c_tightness=c_tight,
            c_volume=c_vol,
            c_near_high=c_nh,
            c_higher_low=c_hl,
            c_rsr=c_rsr,
        )

    except Exception as exc:
        logger.warning(
            "[COMPRESS_CONT] compute failed for %s: %s",
            getattr(inp, "symbol", "?"), exc,
        )
        return CompressionContinuationResult(
            symbol=getattr(inp, "symbol", "?"),
            score=0.0,
            phase="neutral",
            is_healthy_compression=False,
            is_distribution=False,
            suppression_eligible=False,
            invalidation_flags=["compute_error"],
            reasons=[f"error:{exc}"],
        )
