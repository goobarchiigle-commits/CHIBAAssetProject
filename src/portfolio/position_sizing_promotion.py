"""
position_sizing_promotion.py — Position Sizing Promotion Engine

Primary objective: prove that the Virtual Conviction Weight portfolio
consistently outperforms the Equal Weight portfolio.

Tier lifecycle:
  Tier 0 (Observation):  equal weight only; conviction scores logged
  Tier 1 (Validated):    conviction monotonicity + expectancy confirmed
  Tier 2 (Strong):       virtual > equal portfolio return confirmed; hybrid blend
  Tier 3 (Live Ready):   Sharpe improvement confirmed; conviction_weight mode

Tier thresholds:
  0→1: sample_count >= 100, conviction_monotonicity >= 0.60,
       top_vs_bottom_expectancy_delta > 0
  1→2: 30d+ in Tier1,
       virtual_vs_equal_return_delta > 0,
       virtual_vs_equal_expectancy_delta > 0
  2→3: 60d+ in Tier2,
       virtual_vs_equal_sharpe_delta > 0,
       top_vs_bottom_expectancy_delta > 0 (continuing)

Safety demotion (any trigger → Tier 0, logged):
  - conviction_monotonicity < 0.45
  - top_vs_bottom_expectancy_delta <= 0
  - virtual_vs_equal_return_delta <= 0 when currently Tier >= 2
  - sample_count < 30

Outputs:
  position_sizing_promotion.json  ← current tier state (atomic write)
  position_sizing_history.csv     ← append-only daily snapshot

Forward returns: 5-trading-day window from data/ohlcv/{symbol}.csv
Feature flag: strategy.yaml → position_sizing.enabled (default: true)
FAIL_OPEN throughout.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))

# ── Tier mappings ──────────────────────────────────────────────────────────────
CONVICTION_BLEND_FACTORS: Dict[int, float] = {
    0: 0.0,   # pure equal weight
    1: 0.0,   # pure equal weight (evidence accumulating)
    2: 0.5,   # hybrid: 50% conviction, 50% equal
    3: 1.0,   # pure conviction weight
}

TIER_WEIGHT_MODES: Dict[int, str] = {
    0: "equal_weight",
    1: "equal_weight",
    2: "hybrid_weight",
    3: "conviction_weight",
}

TIER_NAMES: Dict[int, str] = {
    0: "Observation",
    1: "Validated",
    2: "Strong",
    3: "Live Ready",
}

# ── Promotion thresholds ───────────────────────────────────────────────────────
FORWARD_WINDOW:        int   = 5     # trading days for forward return
EVALUATION_WINDOWS:    tuple = (30, 60, 90)

TIER0_SAMPLE_MIN:          int   = 100
TIER0_MONOTONICITY_MIN:    float = 0.60
TIER0_EXPECTANCY_DELTA_MIN: float = 0.0   # strict >

TIER1_MIN_DAYS:            int   = 30
TIER1_RETURN_DELTA_MIN:    float = 0.0   # strict >
TIER1_EXPECTANCY_DELTA_MIN: float = 0.0  # strict >

TIER2_MIN_DAYS:            int   = 60
TIER2_SHARPE_DELTA_MIN:    float = 0.0   # strict >

# ── Safety demotion ────────────────────────────────────────────────────────────
SAFETY_MONOTONICITY_FLOOR:    float = 0.45
SAFETY_EXPECTANCY_DELTA_FLOOR: float = 0.0   # <= triggers demotion
SAFETY_SAMPLE_FLOOR:          int   = 30
# return_delta <=0 trigger only for Tier >= 2

SCHEMA_VERSION = "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuintileStats:
    quintile:       int    # 1=lowest conviction, 5=highest conviction
    sample_count:   int
    avg_conviction: float
    win_rate:       Optional[float]     # fraction return > 0; None if no returns
    expectancy:     Optional[float]     # mean forward return
    avg_return:     Optional[float]     # alias of expectancy
    sharpe_proxy:   Optional[float]     # mean/std; None if < 2 samples


@dataclass
class PortfolioComparison:
    window_days:          int
    sample_dates:         int          # dates with ≥2 symbols + forward returns
    equal_return:         Optional[float]   # mean per-date equal-weight return
    virtual_return:       Optional[float]   # mean per-date virtual-weight return
    return_delta:         Optional[float]   # virtual - equal
    equal_expectancy:     Optional[float]   # same as equal_return (alias)
    virtual_expectancy:   Optional[float]   # same as virtual_return (alias)
    expectancy_delta:     Optional[float]
    equal_sharpe:         Optional[float]
    virtual_sharpe:       Optional[float]
    sharpe_delta:         Optional[float]
    sufficient:           bool              # enough data for claims


@dataclass
class WindowAnalysis:
    window_days:              int
    sample_count:             int           # records (including those without returns)
    enriched_count:           int           # records with forward returns
    conviction_monotonicity:  float         # fraction of Q pairs non-decreasing
    top_quintile_expectancy:  Optional[float]
    bottom_quintile_expectancy: Optional[float]
    expectancy_delta:         Optional[float]   # top - bottom
    quintile_stats:           List[QuintileStats]
    portfolio_comparison:     PortfolioComparison
    sufficient:               bool


@dataclass
class PromotionState:
    current_tier:               int
    current_blend_factor:       float
    recommended_weight_mode:    str
    tier_start_date:            str       # ISO date YYYY-MM-DD
    last_evaluated:             str       # ISO date YYYY-MM-DD
    windows:                    Dict[str, Any]   # keyed by str(window_days)
    promotion_recommendation:   str       # PROMOTE / HOLD / DEMOTE / SAFETY_DEMOTE
    safety_triggered:           bool
    safety_reasons:             List[str]
    events:                     List[Dict[str, Any]]
    schema_version:             str = SCHEMA_VERSION


# ─────────────────────────────────────────────────────────────────────────────
# OHLCV helpers (inline; modelled after outcome_materializer.py)
# ─────────────────────────────────────────────────────────────────────────────

def _load_ohlcv(ohlcv_dir: Path, symbol: str) -> Optional[Dict[str, float]]:
    """Load {date_str: close} from data/ohlcv/{symbol}.csv. None if missing."""
    csv_path = ohlcv_dir / f"{symbol}.csv"
    if not csv_path.exists():
        return None
    ohlcv: Dict[str, float] = {}
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("date"):
            continue
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            ohlcv[parts[0].strip()] = float(parts[1].strip())
        except ValueError:
            continue
    return ohlcv or None


def _compute_fwd_return(
    ohlcv: Dict[str, float],
    base_date: str,
    window: int,
) -> Optional[float]:
    """5-day forward return from base_date. None if data insufficient."""
    dates = sorted(ohlcv.keys())
    base_idx: Optional[int] = None
    for i, d in enumerate(dates):
        if d >= base_date:
            base_idx = i
            break
    if base_idx is None:
        return None
    base_close = ohlcv[dates[base_idx]]
    if base_close <= 0:
        return None
    fwd_idx = base_idx + window
    if fwd_idx >= len(dates):
        return None
    fwd_close = ohlcv[dates[fwd_idx]]
    return round((fwd_close - base_close) / base_close, 6)


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry loading and enrichment
# ─────────────────────────────────────────────────────────────────────────────

def _load_telemetry(
    telemetry_path: Path,
    window_days:    int,
    today:          date,
) -> List[dict]:
    """Load records within window_days. FAIL_OPEN → []."""
    try:
        if not telemetry_path.exists():
            return []
        cutoff = today - timedelta(days=window_days)
        records: List[dict] = []
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                rec_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
                if rec_date >= cutoff:
                    records.append(r)
            except Exception:
                continue
        return records
    except Exception as exc:
        logger.warning("[PSP] telemetry load failed: %s", exc)
        return []


def _enrich_with_returns(
    records:  List[dict],
    ohlcv_dir: Path,
    today:    date,
) -> List[dict]:
    """
    Add forward_return_5d to records where OHLCV available and forward window
    has elapsed (record date + 7 calendar days <= today as proxy for 5 trading days).
    FAIL_OPEN: leaves forward_return_5d=None for missing data.
    """
    ohlcv_cache: Dict[str, Optional[Dict[str, float]]] = {}
    enriched: List[dict] = []
    for rec in records:
        r = dict(rec)
        r.setdefault("forward_return_5d", None)
        sym       = r.get("symbol", "")
        base_date = r.get("date", "")
        if not sym or not base_date:
            enriched.append(r)
            continue
        try:
            rec_date = datetime.strptime(base_date, "%Y-%m-%d").date()
            # Only materialize if at least 7 calendar days have passed (proxy for 5 trading days)
            if (today - rec_date).days < 7:
                enriched.append(r)
                continue
            if sym not in ohlcv_cache:
                try:
                    ohlcv_cache[sym] = _load_ohlcv(ohlcv_dir, sym)
                except Exception:
                    ohlcv_cache[sym] = None
            ohlcv = ohlcv_cache[sym]
            if ohlcv:
                fwd = _compute_fwd_return(ohlcv, base_date, FORWARD_WINDOW)
                r["forward_return_5d"] = fwd
        except Exception as exc:
            logger.debug("[PSP] enrich error %s/%s: %s", sym, base_date, exc)
        enriched.append(r)
    return enriched


# ─────────────────────────────────────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe_proxy(returns: List[float]) -> Optional[float]:
    """mean/std; None if < 2 samples or zero std."""
    if len(returns) < 2:
        return None
    mean_r = sum(returns) / len(returns)
    var    = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
    std    = math.sqrt(var) if var > 0 else 0.0
    if std <= 0:
        return None
    return round(mean_r / std, 4)


def _compute_quintile_stats(enriched_records: List[dict]) -> List[QuintileStats]:
    """
    Compute per-quintile stats.  Quintile boundaries are based on conviction_score
    (20th/40th/60th/80th percentiles of the sample).
    Returns list of 5 QuintileStats (Q1=lowest, Q5=highest conviction).
    """
    records_with_conviction = [
        r for r in enriched_records
        if r.get("conviction_score") is not None
    ]
    if not records_with_conviction:
        return []

    convictions = sorted(float(r["conviction_score"]) for r in records_with_conviction)
    n = len(convictions)
    boundaries = [
        convictions[int(n * pct) - 1] if int(n * pct) > 0 else convictions[0]
        for pct in (0.20, 0.40, 0.60, 0.80)
    ]

    def _quintile(score: float) -> int:
        for i, boundary in enumerate(boundaries):
            if score <= boundary:
                return i + 1
        return 5

    buckets: Dict[int, List[dict]] = {q: [] for q in range(1, 6)}
    for r in records_with_conviction:
        q = _quintile(float(r["conviction_score"]))
        buckets[q].append(r)

    stats: List[QuintileStats] = []
    for q in range(1, 6):
        recs  = buckets[q]
        n_q   = len(recs)
        convs = [float(r["conviction_score"]) for r in recs]
        avg_c = round(sum(convs) / n_q, 2) if convs else 0.0

        returns = [
            float(r["forward_return_5d"])
            for r in recs
            if r.get("forward_return_5d") is not None
        ]
        if returns:
            win_rate    = round(sum(1 for x in returns if x > 0) / len(returns), 4)
            expectancy  = round(sum(returns) / len(returns), 6)
            sharpe      = _sharpe_proxy(returns)
        else:
            win_rate = expectancy = sharpe = None

        stats.append(QuintileStats(
            quintile       = q,
            sample_count   = n_q,
            avg_conviction = avg_c,
            win_rate       = win_rate,
            expectancy     = expectancy,
            avg_return     = expectancy,
            sharpe_proxy   = sharpe,
        ))
    return stats


def _conviction_monotonicity(quintile_stats: List[QuintileStats]) -> float:
    """
    Fraction of adjacent (Q_i, Q_i+1) pairs where expectancy is non-decreasing.
    Returns 0.0 if insufficient data.
    """
    ordered  = sorted(quintile_stats, key=lambda s: s.quintile)
    exp_vals = [s.expectancy for s in ordered if s.expectancy is not None]
    if len(exp_vals) < 2:
        return 0.0
    non_decreasing = sum(1 for a, b in zip(exp_vals, exp_vals[1:]) if b >= a)
    return round(non_decreasing / (len(exp_vals) - 1), 4)


def _compute_portfolio_comparison(
    enriched_records: List[dict],
    window_days:      int,
) -> PortfolioComparison:
    """
    Compare equal-weight vs virtual-weight portfolio returns.
    Groups records by date; each date must have ≥2 records with forward returns.
    """
    empty = PortfolioComparison(
        window_days=window_days, sample_dates=0,
        equal_return=None, virtual_return=None, return_delta=None,
        equal_expectancy=None, virtual_expectancy=None, expectancy_delta=None,
        equal_sharpe=None, virtual_sharpe=None, sharpe_delta=None,
        sufficient=False,
    )
    try:
        # Group by date; only records with a valid forward return
        by_date: Dict[str, List[dict]] = {}
        for r in enriched_records:
            if r.get("forward_return_5d") is None:
                continue
            d = r.get("date", "")
            if d:
                by_date.setdefault(d, []).append(r)

        # Keep only dates with ≥2 symbols
        valid_dates = {d: recs for d, recs in by_date.items() if len(recs) >= 2}
        if not valid_dates:
            return empty

        equal_daily:   List[float] = []
        virtual_daily: List[float] = []

        for recs in valid_dates.values():
            fwd_returns = [float(r["forward_return_5d"]) for r in recs]  # type: ignore[arg-type]
            eq_ret = sum(fwd_returns) / len(fwd_returns)

            # Renormalize virtual weights in case some records are missing
            vw_raw = [float(r.get("virtual_weight") or 0.0) for r in recs]
            vw_sum = sum(vw_raw)
            if vw_sum <= 0:
                vw_norm = [1.0 / len(recs)] * len(recs)
            else:
                vw_norm = [w / vw_sum for w in vw_raw]

            virt_ret = sum(w * fr for w, fr in zip(vw_norm, fwd_returns))
            equal_daily.append(eq_ret)
            virtual_daily.append(virt_ret)

        n = len(equal_daily)
        if n == 0:
            return empty

        eq_mean    = round(sum(equal_daily)   / n, 6)
        virt_mean  = round(sum(virtual_daily) / n, 6)
        ret_delta  = round(virt_mean - eq_mean, 6)

        eq_sharpe   = _sharpe_proxy(equal_daily)
        virt_sharpe = _sharpe_proxy(virtual_daily)
        sharpe_delta = (
            round(virt_sharpe - eq_sharpe, 4)
            if (eq_sharpe is not None and virt_sharpe is not None)
            else None
        )

        return PortfolioComparison(
            window_days        = window_days,
            sample_dates       = n,
            equal_return       = eq_mean,
            virtual_return     = virt_mean,
            return_delta       = ret_delta,
            equal_expectancy   = eq_mean,
            virtual_expectancy = virt_mean,
            expectancy_delta   = ret_delta,
            equal_sharpe       = eq_sharpe,
            virtual_sharpe     = virt_sharpe,
            sharpe_delta       = sharpe_delta,
            sufficient         = n >= 5,
        )
    except Exception as exc:
        logger.warning("[PSP] portfolio comparison failed: %s", exc)
        return empty


def compute_window_analysis(
    telemetry_path: Path,
    ohlcv_dir:      Path,
    window_days:    int,
    today:          date,
) -> WindowAnalysis:
    """
    Full analysis for a single evaluation window. FAIL_OPEN.
    """
    try:
        records  = _load_telemetry(telemetry_path, window_days, today)
        enriched = _enrich_with_returns(records, ohlcv_dir, today)

        n_total    = len(records)
        n_enriched = sum(1 for r in enriched if r.get("forward_return_5d") is not None)

        quintile_stats = _compute_quintile_stats(enriched)
        monotonicity   = _conviction_monotonicity(quintile_stats)

        top_exp    = next((s.expectancy for s in quintile_stats if s.quintile == 5), None)
        bot_exp    = next((s.expectancy for s in quintile_stats if s.quintile == 1), None)
        exp_delta  = (
            round(top_exp - bot_exp, 6)
            if (top_exp is not None and bot_exp is not None)
            else None
        )

        pc = _compute_portfolio_comparison(enriched, window_days)

        sufficient = n_total >= 30 and (n_enriched >= 5 or n_total >= SAFETY_SAMPLE_FLOOR)

        return WindowAnalysis(
            window_days              = window_days,
            sample_count             = n_total,
            enriched_count           = n_enriched,
            conviction_monotonicity  = monotonicity,
            top_quintile_expectancy  = top_exp,
            bottom_quintile_expectancy = bot_exp,
            expectancy_delta         = exp_delta,
            quintile_stats           = quintile_stats,
            portfolio_comparison     = pc,
            sufficient               = sufficient,
        )
    except Exception as exc:
        logger.warning("[PSP] window analysis failed (w=%d): %s", window_days, exc)
        pc_empty = PortfolioComparison(
            window_days=window_days, sample_dates=0,
            equal_return=None, virtual_return=None, return_delta=None,
            equal_expectancy=None, virtual_expectancy=None, expectancy_delta=None,
            equal_sharpe=None, virtual_sharpe=None, sharpe_delta=None,
            sufficient=False,
        )
        return WindowAnalysis(
            window_days=window_days, sample_count=0, enriched_count=0,
            conviction_monotonicity=0.0, top_quintile_expectancy=None,
            bottom_quintile_expectancy=None, expectancy_delta=None,
            quintile_stats=[], portfolio_comparison=pc_empty,
            sufficient=False,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Safety + tier transition
# ─────────────────────────────────────────────────────────────────────────────

def check_safety_triggers(
    w30:          WindowAnalysis,
    current_tier: int,
) -> Tuple[bool, List[str]]:
    """
    Returns (triggered, reasons).
    Safety triggers cause immediate demotion to Tier 0.
    """
    reasons: List[str] = []

    if w30.sample_count < SAFETY_SAMPLE_FLOOR:
        reasons.append(f"sample_count={w30.sample_count} < {SAFETY_SAMPLE_FLOOR}")

    if w30.conviction_monotonicity < SAFETY_MONOTONICITY_FLOOR:
        reasons.append(
            f"monotonicity={w30.conviction_monotonicity:.2f} < {SAFETY_MONOTONICITY_FLOOR}"
        )

    if (
        w30.expectancy_delta is not None
        and w30.expectancy_delta <= SAFETY_EXPECTANCY_DELTA_FLOOR
        and w30.enriched_count >= 10   # only trigger if we have enough returns data
    ):
        reasons.append(
            f"expectancy_delta={w30.expectancy_delta:.6f} <= {SAFETY_EXPECTANCY_DELTA_FLOOR}"
        )

    if (
        current_tier >= 2
        and w30.portfolio_comparison.return_delta is not None
        and w30.portfolio_comparison.return_delta <= 0
        and w30.portfolio_comparison.sufficient
    ):
        reasons.append(
            f"return_delta={w30.portfolio_comparison.return_delta:.6f} <= 0 at Tier {current_tier}"
        )

    return len(reasons) > 0, reasons


def _days_in_tier(tier_start_date: str, today: date) -> int:
    """Calendar days since tier start."""
    try:
        start = datetime.strptime(tier_start_date, "%Y-%m-%d").date()
        return (today - start).days
    except Exception:
        return 0


def evaluate_tier_transition(
    current_tier:     int,
    tier_start_date:  str,
    w30:              WindowAnalysis,
    w60:              WindowAnalysis,
    w90:              WindowAnalysis,
    today:            date,
) -> Tuple[int, str, List[str]]:
    """
    Returns (new_tier, recommendation, reasons).
    recommendation: PROMOTE / HOLD / DEMOTE / SAFETY_DEMOTE
    """
    safety_triggered, safety_reasons = check_safety_triggers(w30, current_tier)
    if safety_triggered:
        return 0, "SAFETY_DEMOTE", safety_reasons

    days_in = _days_in_tier(tier_start_date, today)

    if current_tier == 0:
        # 0→1: sample ≥100, monotonicity ≥0.60, expectancy_delta > 0
        reasons: List[str] = []
        ok_sample = w30.sample_count >= TIER0_SAMPLE_MIN
        ok_mono   = w30.conviction_monotonicity >= TIER0_MONOTONICITY_MIN
        ok_exp    = (
            w30.expectancy_delta is not None
            and w30.expectancy_delta > TIER0_EXPECTANCY_DELTA_MIN
            and w30.enriched_count >= 20
        )
        if ok_sample and ok_mono and ok_exp:
            reasons.append(f"sample={w30.sample_count}, mono={w30.conviction_monotonicity:.2f}")
            reasons.append(f"exp_delta={w30.expectancy_delta:.6f}")
            return 1, "PROMOTE", reasons
        missing = []
        if not ok_sample:
            missing.append(f"sample={w30.sample_count}/{TIER0_SAMPLE_MIN}")
        if not ok_mono:
            missing.append(f"mono={w30.conviction_monotonicity:.2f}/{TIER0_MONOTONICITY_MIN}")
        if not ok_exp:
            ed = w30.expectancy_delta
            missing.append(f"exp_delta={ed}")
        return 0, "HOLD", missing

    if current_tier == 1:
        # 1→2: 30d+, return_delta > 0, expectancy_delta > 0
        reasons = []
        ok_days  = days_in >= TIER1_MIN_DAYS
        pc30     = w30.portfolio_comparison
        ok_ret   = (
            pc30.sufficient
            and pc30.return_delta is not None
            and pc30.return_delta > TIER1_RETURN_DELTA_MIN
        )
        ok_exp   = (
            w30.expectancy_delta is not None
            and w30.expectancy_delta > TIER1_EXPECTANCY_DELTA_MIN
            and w30.enriched_count >= 20
        )
        if ok_days and ok_ret and ok_exp:
            reasons.append(f"days_in_tier={days_in}")
            reasons.append(f"return_delta={pc30.return_delta:.6f}")
            reasons.append(f"exp_delta={w30.expectancy_delta:.6f}")
            return 2, "PROMOTE", reasons
        missing = []
        if not ok_days:
            missing.append(f"days={days_in}/{TIER1_MIN_DAYS}")
        if not ok_ret:
            rd = pc30.return_delta
            missing.append(f"return_delta={rd} sufficient={pc30.sufficient}")
        if not ok_exp:
            missing.append(f"exp_delta={w30.expectancy_delta}")
        return 1, "HOLD", missing

    if current_tier == 2:
        # 2→3: 60d+, sharpe_delta > 0, expectancy improving
        reasons = []
        ok_days  = days_in >= TIER2_MIN_DAYS
        pc60     = w60.portfolio_comparison
        ok_sharpe = (
            pc60.sufficient
            and pc60.sharpe_delta is not None
            and pc60.sharpe_delta > TIER2_SHARPE_DELTA_MIN
        )
        ok_exp   = (
            w60.expectancy_delta is not None
            and w60.expectancy_delta > 0
            and w60.enriched_count >= 30
        )
        if ok_days and ok_sharpe and ok_exp:
            reasons.append(f"days_in_tier={days_in}")
            reasons.append(f"sharpe_delta={pc60.sharpe_delta:.4f}")
            reasons.append(f"exp_delta_60d={w60.expectancy_delta:.6f}")
            return 3, "PROMOTE", reasons
        missing = []
        if not ok_days:
            missing.append(f"days={days_in}/{TIER2_MIN_DAYS}")
        if not ok_sharpe:
            sd = pc60.sharpe_delta
            missing.append(f"sharpe_delta={sd} sufficient={pc60.sufficient}")
        if not ok_exp:
            missing.append(f"exp_delta_60d={w60.expectancy_delta}")
        return 2, "HOLD", missing

    # current_tier == 3: maintain unless safety triggered (already checked)
    return 3, "HOLD", ["AT_MAX_TIER"]


# ─────────────────────────────────────────────────────────────────────────────
# Persistence helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_state(promotion_file: Path) -> Optional[PromotionState]:
    """Load persisted PromotionState from JSON. Returns None if missing/invalid."""
    try:
        if not promotion_file.exists():
            return None
        data = json.loads(promotion_file.read_text(encoding="utf-8"))
        return PromotionState(
            current_tier             = int(data.get("current_tier", 0)),
            current_blend_factor     = float(data.get("current_blend_factor", 0.0)),
            recommended_weight_mode  = str(data.get("recommended_weight_mode", "equal_weight")),
            tier_start_date          = str(data.get("tier_start_date", "")),
            last_evaluated           = str(data.get("last_evaluated", "")),
            windows                  = data.get("windows", {}),
            promotion_recommendation = str(data.get("promotion_recommendation", "HOLD")),
            safety_triggered         = bool(data.get("safety_triggered", False)),
            safety_reasons           = list(data.get("safety_reasons", [])),
            events                   = list(data.get("events", [])),
            schema_version           = str(data.get("schema_version", SCHEMA_VERSION)),
        )
    except Exception as exc:
        logger.warning("[PSP] state load failed: %s", exc)
        return None


def _save_state(state: PromotionState, promotion_file: Path) -> None:
    """Atomic write of PromotionState to JSON. FAIL_OPEN."""
    try:
        promotion_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_tier":             state.current_tier,
            "current_blend_factor":     state.current_blend_factor,
            "recommended_weight_mode":  state.recommended_weight_mode,
            "tier_start_date":          state.tier_start_date,
            "last_evaluated":           state.last_evaluated,
            "promotion_recommendation": state.promotion_recommendation,
            "safety_triggered":         state.safety_triggered,
            "safety_reasons":           state.safety_reasons,
            "events":                   state.events,
            "schema_version":           state.schema_version,
        }
        # Store window analysis (serialize, skip non-serializable objects)
        windows_out: Dict[str, Any] = {}
        for k, wa in state.windows.items():
            if isinstance(wa, WindowAnalysis):
                pc = wa.portfolio_comparison
                qs_list = [
                    {
                        "quintile": s.quintile,
                        "sample_count": s.sample_count,
                        "avg_conviction": s.avg_conviction,
                        "win_rate": s.win_rate,
                        "expectancy": s.expectancy,
                        "sharpe_proxy": s.sharpe_proxy,
                    }
                    for s in wa.quintile_stats
                ]
                windows_out[k] = {
                    "window_days":                wa.window_days,
                    "sample_count":               wa.sample_count,
                    "enriched_count":             wa.enriched_count,
                    "conviction_monotonicity":    wa.conviction_monotonicity,
                    "top_quintile_expectancy":    wa.top_quintile_expectancy,
                    "bottom_quintile_expectancy": wa.bottom_quintile_expectancy,
                    "expectancy_delta":           wa.expectancy_delta,
                    "quintile_stats":             qs_list,
                    "portfolio_comparison": {
                        "window_days":          pc.window_days,
                        "sample_dates":         pc.sample_dates,
                        "equal_return":         pc.equal_return,
                        "virtual_return":       pc.virtual_return,
                        "return_delta":         pc.return_delta,
                        "equal_sharpe":         pc.equal_sharpe,
                        "virtual_sharpe":       pc.virtual_sharpe,
                        "sharpe_delta":         pc.sharpe_delta,
                        "sufficient":           pc.sufficient,
                    },
                    "sufficient": wa.sufficient,
                }
            else:
                windows_out[k] = wa
        data["windows"] = windows_out

        # Atomic write via temp file
        tmp_path = promotion_file.with_suffix(".tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        tmp_path.replace(promotion_file)
    except Exception as exc:
        logger.warning("[PSP] state save failed: %s", exc)


def _append_history_csv(
    history_file: Path,
    state:        PromotionState,
    today_str:    str,
) -> None:
    """Append one row to history CSV. FAIL_OPEN."""
    try:
        history_file.parent.mkdir(parents=True, exist_ok=True)
        write_header = not history_file.exists()
        with history_file.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "date", "tier", "blend_factor", "weight_mode",
                "recommendation", "safety_triggered", "tier_start_date",
            ])
            if write_header:
                writer.writeheader()
            writer.writerow({
                "date":              today_str,
                "tier":              state.current_tier,
                "blend_factor":      state.current_blend_factor,
                "weight_mode":       state.recommended_weight_mode,
                "recommendation":    state.promotion_recommendation,
                "safety_triggered":  state.safety_triggered,
                "tier_start_date":   state.tier_start_date,
            })
    except Exception as exc:
        logger.warning("[PSP] history CSV append failed: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_position_sizing_promotion(
    telemetry_file:  Path,
    promotion_file:  Path,
    history_file:    Path,
    ohlcv_dir:       Path,
    today_str:       str,
) -> PromotionState:
    """
    Main entry point. Load telemetry, compute window analyses, evaluate tier
    transition, persist state, append history. FAIL_OPEN.

    Returns PromotionState.
    """
    try:
        today = datetime.strptime(today_str, "%Y-%m-%d").date()

        # Load previous state (to preserve tier history)
        prev = _load_state(promotion_file)
        is_fresh = prev is None
        if prev is None:
            prev = PromotionState(
                current_tier             = 0,
                current_blend_factor     = CONVICTION_BLEND_FACTORS[0],
                recommended_weight_mode  = TIER_WEIGHT_MODES[0],
                tier_start_date          = today_str,
                last_evaluated           = "",
                windows                  = {},
                promotion_recommendation = "HOLD",
                safety_triggered         = False,
                safety_reasons           = [],
                events                   = [],
            )

        # Already evaluated today → return existing state (idempotent)
        if not is_fresh and prev.last_evaluated == today_str:
            return prev

        # Compute window analyses
        w30 = compute_window_analysis(telemetry_file, ohlcv_dir, 30,  today)
        w60 = compute_window_analysis(telemetry_file, ohlcv_dir, 60,  today)
        w90 = compute_window_analysis(telemetry_file, ohlcv_dir, 90,  today)

        # Evaluate tier transition
        new_tier, recommendation, reasons = evaluate_tier_transition(
            current_tier    = prev.current_tier,
            tier_start_date = prev.tier_start_date,
            w30             = w30,
            w60             = w60,
            w90             = w90,
            today           = today,
        )

        # Update tier start date on promotion/demotion
        if new_tier != prev.current_tier:
            tier_start = today_str
            event: Dict[str, Any] = {
                "date":         today_str,
                "from_tier":    prev.current_tier,
                "to_tier":      new_tier,
                "recommendation": recommendation,
                "reasons":      reasons,
            }
            events = list(prev.events) + [event]
            logger.info("[PSP] tier %d→%d: %s — %s", prev.current_tier, new_tier, recommendation, reasons)
        else:
            tier_start = prev.tier_start_date
            events     = prev.events

        safety_triggered = recommendation == "SAFETY_DEMOTE"

        new_state = PromotionState(
            current_tier             = new_tier,
            current_blend_factor     = CONVICTION_BLEND_FACTORS[new_tier],
            recommended_weight_mode  = TIER_WEIGHT_MODES[new_tier],
            tier_start_date          = tier_start,
            last_evaluated           = today_str,
            windows                  = {"30": w30, "60": w60, "90": w90},
            promotion_recommendation = recommendation,
            safety_triggered         = safety_triggered,
            safety_reasons           = reasons if safety_triggered else [],
            events                   = events,
        )

        _save_state(new_state, promotion_file)
        _append_history_csv(history_file, new_state, today_str)

        return new_state

    except Exception as exc:
        logger.warning("[PSP] run_position_sizing_promotion FAIL_OPEN: %s", exc)
        return PromotionState(
            current_tier             = 0,
            current_blend_factor     = CONVICTION_BLEND_FACTORS[0],
            recommended_weight_mode  = TIER_WEIGHT_MODES[0],
            tier_start_date          = today_str,
            last_evaluated           = today_str,
            windows                  = {},
            promotion_recommendation = "HOLD",
            safety_triggered         = False,
            safety_reasons           = [str(exc)],
            events                   = [],
        )


def get_psp_blend_factor(
    promotion_file:    Path,
    fallback_factor:   float = 0.0,
    auto_apply_enabled: bool = False,
) -> float:
    """
    Returns the current blend factor for Position Sizing v2.
    Only applies conviction weight if auto_apply_enabled=True.
    FAIL_OPEN → fallback_factor.
    """
    if not auto_apply_enabled:
        return fallback_factor
    try:
        state = _load_state(promotion_file)
        if state is None:
            return fallback_factor
        return state.current_blend_factor
    except Exception as exc:
        logger.warning("[PSP] get_psp_blend_factor FAIL_OPEN: %s", exc)
        return fallback_factor


def format_psp_report_section(
    promotion_file: Path,
    today_str:      str,
) -> str:
    """
    Format [POSITION SIZING PROMOTION] section for weekly review report.
    FAIL_OPEN: returns "" on error.
    """
    try:
        state = _load_state(promotion_file)
        if state is None:
            return ""

        tier      = state.current_tier
        tier_name = TIER_NAMES.get(tier, f"Tier{tier}")
        blend     = state.current_blend_factor
        mode      = state.recommended_weight_mode
        rec       = state.promotion_recommendation
        days_in   = _days_in_tier(state.tier_start_date, datetime.strptime(today_str, "%Y-%m-%d").date())

        lines = [
            "",
            "━━ [POSITION SIZING PROMOTION — コンビクション有効性評価] ━━━━━",
            f"  現在Tier       : Tier {tier} ({tier_name})",
            f"  推奨配分モード : {mode}",
            f"  ブレンド係数   : {blend:.1f}  ({blend*100:.0f}% conviction / {(1-blend)*100:.0f}% equal)",
            f"  判定            : {rec}",
            f"  Tier継続日数   : {days_in}日 (開始: {state.tier_start_date})",
        ]

        # Most recent window (30d)
        w30_data = state.windows.get("30")
        if w30_data:
            if isinstance(w30_data, WindowAnalysis):
                mono = w30_data.conviction_monotonicity
                ed   = w30_data.expectancy_delta
                n    = w30_data.sample_count
                ne   = w30_data.enriched_count
                pc   = w30_data.portfolio_comparison
            else:
                mono = w30_data.get("conviction_monotonicity")
                ed   = w30_data.get("expectancy_delta")
                n    = w30_data.get("sample_count", 0)
                ne   = w30_data.get("enriched_count", 0)
                pc   = w30_data.get("portfolio_comparison", {})

            lines.append(f"  ── 直近30日分析 (n={n}, リターン有={ne}件) ─────────────────")
            if mono is not None:
                lines.append(f"  コンビクション単調性      : {mono:.2f}  (閾値: ≥{TIER0_MONOTONICITY_MIN})")
            if ed is not None:
                lines.append(f"  Q5-Q1 期待値差            : {ed:+.4%}")

            if isinstance(pc, PortfolioComparison):
                rd = pc.return_delta
                sd = pc.sharpe_delta
                nd = pc.sample_dates
            else:
                rd = pc.get("return_delta") if pc else None
                sd = pc.get("sharpe_delta") if pc else None
                nd = pc.get("sample_dates", 0) if pc else 0

            if rd is not None:
                lines.append(
                    f"  仮想ポートフォリオ優位    : {rd:+.4%}  (日数={nd}, "
                    + ("十分" if (isinstance(pc, PortfolioComparison) and pc.sufficient) or (isinstance(pc, dict) and pc.get("sufficient")) else "不十分") + ")"
                )
            if sd is not None:
                lines.append(f"  Sharpe差                  : {sd:+.4f}")

        if state.safety_triggered and state.safety_reasons:
            lines.append("  ⚠ SAFETY_DEMOTE 発動理由:")
            for r in state.safety_reasons:
                lines.append(f"    - {r}")

        if state.events:
            last_event = state.events[-1]
            lines.append(
                f"  最終Tier変化    : {last_event.get('from_tier')}→{last_event.get('to_tier')}"
                f" ({last_event.get('date')}) [{last_event.get('recommendation')}]"
            )

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[PSP] format_psp_report_section FAIL_OPEN: %s", exc)
        return ""
