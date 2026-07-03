"""
src/analytics/weekly_market_intelligence.py — Weekly market intelligence engine

Long-horizon statistical intelligence and alpha-survivability analysis.
Detects structural patterns invisible from daily dry-run logs.

Sections:
  1. Opportunity Capture Analysis
  2. Entry Archetype Expectancy (ATR-based overextension + realized PnL)
  3. Market Regime Transition
  4. Capital Efficiency
  5. Holding Period Decay
  6. Expectancy Drift Detection
  7. Research Priority Engine (autonomous, rank-ordered)

Outputs:
  runtime/reports/weekly_market_intelligence_YYYYMMDD.{md,html,json}
  runtime/reports/charts/weekly_YYYYMMDD/*.png

Requirements:
  - deterministic outputs (same inputs → byte-identical markdown)
  - no silent exceptions — every error logged + stored in quality_warnings
  - fail-closed on corrupted trade history (raises WeeklyIntelligenceError)
  - atomic report writes (temp + os.replace)
  - structured logging only
  - all timestamps JST-aware
  - append-only historical archive (weekly_intelligence_archive.jsonl)
  - Windows compatible
  - no external paid APIs
  - no notebook dependencies
"""
from __future__ import annotations

import hashlib
import html as html_mod
import json
import logging
import math
import os
import smtplib
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
SCHEMA_VERSION: int = 1

MAX_POSITIONS: int = 3
CAPITAL: float = 3_000_000.0
COMMISSION: float = 0.00055
SLIPPAGE: float = 0.001

EXPECTANCY_DRIFT_WARN_THRESHOLD: float = -0.05  # rolling expectancy < -5% → alert
BREADTH_TREND_PERSISTENT: float = 0.50
BREADTH_TREND_WEAK: float = 0.30
DRAWDOWN_DANGER: float = -0.10
SIGNAL_DROUGHT_DAYS: int = 5


# ─────────────────────────────────────────────────────────────────────────────
# Exceptions
# ─────────────────────────────────────────────────────────────────────────────

class WeeklyIntelligenceError(RuntimeError):
    """Fail-closed: raised only on corrupted trade history."""


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses — report sections
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpportunityCapture:
    available: bool
    opportunity_capture_ratio: Optional[float]
    false_negative_rate: Optional[float]
    missed_alpha_score: Optional[float]
    skip_reason_distribution: Dict[str, int]
    weekly_rsr_blocked: int
    weekly_breakout_blocked: int
    weekly_candidates_executed: int
    weekly_candidates_signaled: int
    insufficiency_markers: List[str] = field(default_factory=list)


@dataclass
class ArchetypeCluster:
    name: str
    trade_count: int
    avg_atr_pct: float
    avg_entry_price: float
    realized_expectancy: Optional[float]
    realized_win_rate: Optional[float]
    overextension_score: float


@dataclass
class ArchetypeExpectancy:
    available: bool
    total_buys: int
    total_sells: int
    matched_pairs: int
    avg_pnl_pct: Optional[float]
    win_rate: Optional[float]
    realized_expectancy: Optional[float]
    avg_hold_days: Optional[float]
    avg_atr_pct: float
    overextension_alert: bool
    clusters: List[ArchetypeCluster]
    sector_distribution: Dict[str, int]
    regime_distribution: Dict[str, int]
    insufficiency_markers: List[str] = field(default_factory=list)


@dataclass
class RegimeTransition:
    available: bool
    current_regime: str
    regime_confidence: float
    breadth_50_latest: Optional[float]
    breadth_75_latest: Optional[float]
    breadth_50_trend_7d: Optional[float]
    breadth_50_trend_30d: Optional[float]
    equity_drawdown_current: Optional[float]
    equity_drawdown_max_30d: Optional[float]
    signal_drought_consecutive: int
    signal_count_trend_7d: Optional[float]
    failed_breakout_proxy: Optional[float]
    regime_state: str
    instability_score: float
    weekly_regime_history: List[Dict]
    insufficiency_markers: List[str] = field(default_factory=list)


@dataclass
class CapitalEfficiency:
    available: bool
    avg_cash_ratio_7d: float
    avg_cash_ratio_30d: float
    avg_exposure_7d: float
    avg_exposure_30d: float
    slot_utilization_7d: float
    slot_utilization_30d: float
    idle_cash_ratio: float
    peak_positions_30d: int
    total_rsr_blocked_30d: int
    total_breakout_blocked_30d: int
    capital_fragmentation_score: float
    scenario_50pct_capture_gain: Optional[float]
    scenario_100pct_capture_gain: Optional[float]
    scenario_200pct_capture_gain: Optional[float]
    insufficiency_markers: List[str] = field(default_factory=list)


@dataclass
class HoldingPeriodBucket:
    label: str
    trade_count: int
    avg_pnl_pct: Optional[float]
    win_rate: Optional[float]


@dataclass
class HoldingPeriodDecay:
    available: bool
    completed_trades: int
    avg_hold_days: Optional[float]
    max_hold_days: Optional[int]
    min_hold_days: Optional[int]
    buckets: List[HoldingPeriodBucket]
    optimal_exit_bucket: Optional[str]
    insufficiency_markers: List[str] = field(default_factory=list)


@dataclass
class ExpectancyDrift:
    available: bool
    equity_trend_7d: Optional[float]
    equity_trend_30d: Optional[float]
    drawdown_acceleration: Optional[float]
    signal_count_trend_7d: Optional[float]
    edge_stability_score: Optional[float]
    alpha_decay_score: Optional[float]
    instability_alert_level: str
    rolling_win_rate_30d: Optional[float]
    warnings: List[str]
    insufficiency_markers: List[str] = field(default_factory=list)


@dataclass
class ResearchPriority:
    rank: int
    category: str
    title: str
    confidence: str
    evidence: str
    affected_symbols: List[str]
    statistical_justification: str
    suggested_direction: str


@dataclass
class WeeklyIntelligenceReport:
    run_id: str
    generated_at: str
    report_date: str
    week_start: str
    week_end: str
    schema_version: int
    opportunity_capture: OpportunityCapture
    archetype_expectancy: ArchetypeExpectancy
    regime_transition: RegimeTransition
    capital_efficiency: CapitalEfficiency
    holding_period: HoldingPeriodDecay
    expectancy_drift: ExpectancyDrift
    research_priorities: List[ResearchPriority]
    data_availability: Dict[str, bool]
    data_quality_warnings: List[str]
    chart_paths: List[str]
    report_md_path: Optional[str] = None
    report_html_path: Optional[str] = None
    report_json_path: Optional[str] = None
    forward_expectancy: Optional[object] = None  # ForwardExpectancyReport | None


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders — each returns (data, available_flag)
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> Tuple[List[dict], bool]:
    if not path.exists():
        return [], False
    try:
        lines = [l for l in path.read_text(encoding="utf-8", errors="replace").splitlines() if l.strip()]
        rows = []
        for line in lines:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt JSONL line in %s", path)
        return rows, True
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return [], False


def _load_trades(base: Path) -> Tuple[List[dict], bool]:
    rows, ok = _load_jsonl(base / "logs" / "trades.jsonl")
    if ok and not rows:
        return [], True
    if ok:
        buys = [r for r in rows if r.get("side") == "BUY"]
        sells = [r for r in rows if r.get("side") == "SELL"]
        # Fail-closed: raise on corrupted trade history (non-numeric price)
        for r in rows:
            try:
                float(r["price"])
                float(r["qty"])
            except (KeyError, TypeError, ValueError) as exc:
                raise WeeklyIntelligenceError(
                    f"Corrupted trade record — {exc}: {r}"
                ) from exc
        logger.info("Loaded %d trades (%d BUY / %d SELL)", len(rows), len(buys), len(sells))
    return rows, ok


def _load_metrics(base: Path) -> Tuple[List[dict], bool]:
    rows, ok = _load_jsonl(base / "logs" / "diagnostics" / "metrics.jsonl")
    if ok and rows:
        # Deduplicate: keep latest run_at per date
        by_date: Dict[str, dict] = {}
        for r in rows:
            d = r.get("date", "")
            if not d:
                continue
            existing = by_date.get(d)
            if existing is None or r.get("run_at", "") > existing.get("run_at", ""):
                by_date[d] = r
        rows = sorted(by_date.values(), key=lambda r: r.get("date", ""))
        logger.info("Loaded %d daily metrics rows", len(rows))
    return rows, ok


def _load_phase2(base: Path) -> Tuple[List[dict], bool]:
    rows, ok = _load_jsonl(base / "logs" / "phase2_live_metrics.jsonl")
    if ok and rows:
        rows = sorted(rows, key=lambda r: r.get("date", ""))
        logger.info("Loaded %d phase2 metrics rows", len(rows))
    return rows, ok


def _load_skipped_opportunities(base: Path) -> Tuple[List[dict], bool]:
    rows, ok = _load_jsonl(base / "runtime" / "analytics" / "skipped_opportunities.jsonl")
    return rows, ok


def _load_exit_records(base: Path) -> Tuple[List[dict], bool]:
    rows, ok = _load_jsonl(base / "runtime" / "exit_analytics" / "trades" / "exit_records.jsonl")
    return rows, ok


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _filter_to_window(rows: List[dict], start: date, end: date) -> List[dict]:
    result = []
    for r in rows:
        d_str = r.get("date", "")
        if not d_str:
            continue
        try:
            d = date.fromisoformat(d_str[:10])
        except ValueError:
            continue
        if start <= d <= end:
            result.append(r)
    return result


def _linear_slope(values: List[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(values) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, values))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0:
        return 0.0
    return num / den


def _safe_mean(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _win_rate(pnl_list: List[float]) -> Optional[float]:
    if not pnl_list:
        return None
    wins = sum(1 for p in pnl_list if p > 0)
    return wins / len(pnl_list)


def _expectancy(pnl_list: List[float]) -> Optional[float]:
    if len(pnl_list) < 2:
        return None
    return sum(pnl_list) / len(pnl_list)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1 — Opportunity Capture
# ─────────────────────────────────────────────────────────────────────────────

def analyze_opportunity_capture(
    metrics: List[dict],
    skipped: List[dict],
    week_start: date,
    week_end: date,
    window_days: int = 30,
) -> OpportunityCapture:
    window_end = week_end
    window_start = window_end - timedelta(days=window_days)
    recent = _filter_to_window(metrics, window_start, window_end)
    insufficiency: List[str] = []

    rsr_blocked = sum(r.get("signals_blocked_rsr", 0) for r in recent)
    bp_blocked = sum(r.get("signals_blocked_breakout", 0) for r in recent)
    candidates_exec = sum(r.get("candidate_count", 0) for r in recent)
    candidates_sig = sum(r.get("buy_candidates", 0) for r in recent)

    total_signaled = candidates_sig + rsr_blocked + bp_blocked
    capture_ratio: Optional[float] = None
    fnr: Optional[float] = None
    missed_alpha: Optional[float] = None

    if total_signaled > 0:
        capture_ratio = candidates_exec / total_signaled
        fnr = (rsr_blocked + bp_blocked) / total_signaled

    if not skipped:
        insufficiency.append(
            "skipped_opportunities.jsonl missing — forward return attribution unavailable; "
            "missed_alpha_score estimated from RSR/breakout block counts only"
        )
        if rsr_blocked + bp_blocked > 0 and candidates_exec >= 0:
            missed_alpha = min(1.0, (rsr_blocked + bp_blocked) / max(candidates_exec, 1) * 0.05)

    skip_dist: Dict[str, int] = {
        "rsr_filter": rsr_blocked,
        "breakout_filter": bp_blocked,
    }

    return OpportunityCapture(
        available=len(recent) > 0,
        opportunity_capture_ratio=capture_ratio,
        false_negative_rate=fnr,
        missed_alpha_score=missed_alpha,
        skip_reason_distribution=skip_dist,
        weekly_rsr_blocked=rsr_blocked,
        weekly_breakout_blocked=bp_blocked,
        weekly_candidates_executed=candidates_exec,
        weekly_candidates_signaled=candidates_sig,
        insufficiency_markers=insufficiency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 2 — Entry Archetype Expectancy
# ─────────────────────────────────────────────────────────────────────────────

def _atr_pct(trade: dict) -> float:
    try:
        return float(trade.get("atr20", 0)) / float(trade.get("price", 1))
    except (TypeError, ZeroDivisionError):
        return 0.0


def _classify_archetype(trade: dict) -> str:
    reason = str(trade.get("reason", "")).lower()
    atr_p = _atr_pct(trade)
    if "sepa" in reason or "trend" in reason:
        return "trend_continuation"
    if "turtle" in reason:
        return "breakout_extension" if atr_p > 0.02 else "trend_continuation"
    if "earnings" in reason or "gap" in reason:
        return "earnings_gap_momentum"
    if "rsr" in reason:
        return "sector_rotation_entry"
    return "breakout_extension"


def analyze_archetype_expectancy(
    trades: List[dict],
    week_end: date,
    window_days: int = 60,
) -> ArchetypeExpectancy:
    insufficiency: List[str] = []
    window_start = week_end - timedelta(days=window_days)

    buys = _filter_to_window([t for t in trades if t.get("side") == "BUY"], window_start, week_end)
    sells = _filter_to_window([t for t in trades if t.get("side") == "SELL"], window_start, week_end)

    if not buys:
        return ArchetypeExpectancy(
            available=False, total_buys=0, total_sells=0, matched_pairs=0,
            avg_pnl_pct=None, win_rate=None, realized_expectancy=None,
            avg_hold_days=None, avg_atr_pct=0.0, overextension_alert=False,
            clusters=[], sector_distribution={}, regime_distribution={},
            insufficiency_markers=["No BUY trades in window"],
        )

    # ATR-based overextension score (our original metric — no forward data needed)
    atr_pcts = [_atr_pct(t) for t in buys]
    avg_atr_pct = sum(atr_pcts) / len(atr_pcts) if atr_pcts else 0.0
    # High ATR% relative to price → entry on volatile/extended days
    overextension_alert = avg_atr_pct > 0.025

    # Realized PnL from matched sells
    pnl_pcts = [s.get("pnl_pct", 0.0) for s in sells if s.get("pnl_pct") is not None]
    hold_days_list = [s.get("hold_days", 0) for s in sells if s.get("hold_days") is not None]

    if len(sells) < 3:
        insufficiency.append(
            f"Only {len(sells)} completed sell(s) in window — expectancy and win_rate statistically unreliable; "
            "min 5 required"
        )

    # Sector and regime distributions
    sector_dist: Dict[str, int] = {}
    regime_dist: Dict[str, int] = {}
    for t in buys:
        sector_dist[t.get("sector", "unknown")] = sector_dist.get(t.get("sector", "unknown"), 0) + 1
        regime_dist[t.get("entry_regime", "unknown")] = regime_dist.get(t.get("entry_regime", "unknown"), 0) + 1

    # Cluster by archetype
    cluster_map: Dict[str, List[dict]] = {}
    for t in buys:
        arch = _classify_archetype(t)
        cluster_map.setdefault(arch, []).append(t)

    clusters: List[ArchetypeCluster] = []
    for arch_name, arch_trades in sorted(cluster_map.items()):
        arch_sells = [
            s for s in sells
            if _classify_archetype(s) == arch_name and s.get("pnl_pct") is not None
        ]
        arch_pnl = [s["pnl_pct"] for s in arch_sells]
        clusters.append(ArchetypeCluster(
            name=arch_name,
            trade_count=len(arch_trades),
            avg_atr_pct=sum(_atr_pct(t) for t in arch_trades) / len(arch_trades),
            avg_entry_price=sum(float(t.get("price", 0)) for t in arch_trades) / len(arch_trades),
            realized_expectancy=_expectancy(arch_pnl),
            realized_win_rate=_win_rate(arch_pnl),
            overextension_score=sum(_atr_pct(t) for t in arch_trades) / len(arch_trades) / 0.02,
        ))

    return ArchetypeExpectancy(
        available=True,
        total_buys=len(buys),
        total_sells=len(sells),
        matched_pairs=len(sells),
        avg_pnl_pct=_safe_mean(pnl_pcts),
        win_rate=_win_rate(pnl_pcts),
        realized_expectancy=_expectancy(pnl_pcts),
        avg_hold_days=_safe_mean([float(h) for h in hold_days_list]),
        avg_atr_pct=avg_atr_pct,
        overextension_alert=overextension_alert,
        clusters=clusters,
        sector_distribution=sector_dist,
        regime_distribution=regime_dist,
        insufficiency_markers=insufficiency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 3 — Market Regime Transition
# ─────────────────────────────────────────────────────────────────────────────

def _detect_regime_state(
    breadth_50: Optional[float],
    breadth_75: Optional[float],
    drawdown: Optional[float],
    signal_drought: int,
    signal_trend: Optional[float],
) -> Tuple[str, float]:
    if breadth_50 is None:
        return "unknown", 0.0

    confidence = 0.5
    if breadth_50 >= BREADTH_TREND_PERSISTENT and (breadth_75 or 0) >= 0.25:
        state = "trend_persistent"
        confidence = min(0.9, 0.5 + breadth_50 * 0.5)
    elif breadth_50 < BREADTH_TREND_WEAK:
        if (drawdown or 0) < DRAWDOWN_DANGER:
            state = "high_volatility"
            confidence = 0.75
        elif signal_drought >= SIGNAL_DROUGHT_DAYS:
            state = "momentum_exhaustion"
            confidence = 0.70
        else:
            state = "mean_reverting"
            confidence = 0.60
    elif signal_drought >= SIGNAL_DROUGHT_DAYS:
        state = "rotational"
        confidence = 0.65
    elif (signal_trend or 0) < -0.5:
        state = "rotational"
        confidence = 0.60
    else:
        state = "trend_persistent"
        confidence = 0.55

    return state, confidence


def analyze_regime_transition(
    metrics: List[dict],
    phase2: List[dict],
    week_start: date,
    week_end: date,
) -> RegimeTransition:
    insufficiency: List[str] = []
    if not metrics and not phase2:
        return RegimeTransition(
            available=False, current_regime="unknown", regime_confidence=0.0,
            breadth_50_latest=None, breadth_75_latest=None,
            breadth_50_trend_7d=None, breadth_50_trend_30d=None,
            equity_drawdown_current=None, equity_drawdown_max_30d=None,
            signal_drought_consecutive=0, signal_count_trend_7d=None,
            failed_breakout_proxy=None, regime_state="unknown",
            instability_score=0.0, weekly_regime_history=[],
            insufficiency_markers=["Neither metrics.jsonl nor phase2_live_metrics.jsonl available"],
        )

    window_30 = week_end - timedelta(days=30)
    window_7 = week_end - timedelta(days=7)

    recent_30 = _filter_to_window(phase2, window_30, week_end)
    recent_7 = _filter_to_window(phase2, window_7, week_end)

    # Breadth
    b50_series_30 = [r["breadth_50"] for r in recent_30 if "breadth_50" in r]
    b50_series_7 = [r["breadth_50"] for r in recent_7 if "breadth_50" in r]
    b75_latest = recent_30[-1].get("breadth_75") if recent_30 else None
    b50_latest = recent_30[-1].get("breadth_50") if recent_30 else None

    b50_trend_7 = _linear_slope(b50_series_7)
    b50_trend_30 = _linear_slope(b50_series_30)

    # Equity / drawdown
    dd_series = [r.get("drawdown", 0.0) for r in recent_30 if "drawdown" in r]
    dd_current = recent_30[-1].get("drawdown") if recent_30 else None
    dd_max_30 = min(dd_series) if dd_series else None

    # Signal drought (consecutive days with signal_count=0)
    sig_counts = [r.get("signal_count", 0) for r in recent_30]
    drought = 0
    for sc in reversed(sig_counts):
        if sc == 0:
            drought += 1
        else:
            break

    sig_7 = [r.get("signal_count", 0) for r in recent_7 if "signal_count" in r]
    sig_trend_7 = _linear_slope(sig_7)

    # Failed breakout proxy: high breadth_75 decline rate
    if len(b50_series_30) < 5:
        insufficiency.append("Fewer than 5 phase2 data points in 30d window — regime confidence limited")
        fbp = None
    else:
        if b50_trend_30 is not None and b50_latest is not None:
            fbp = max(0.0, -b50_trend_30 * 10)
        else:
            fbp = None

    # Regime state
    regime_state, confidence = _detect_regime_state(
        b50_latest, b75_latest, dd_current, drought, sig_trend_7
    )

    # Instability score: composite of drawdown severity + signal drought + breadth weakness
    instability = 0.0
    if dd_current is not None:
        instability += min(1.0, abs(dd_current) / 0.15) * 0.40
    if drought > 0:
        instability += min(1.0, drought / 10) * 0.30
    if b50_latest is not None:
        instability += max(0.0, 1.0 - b50_latest) * 0.30
    instability = round(instability, 4)

    # Weekly history (last 4 weeks summary)
    weekly_hist: List[Dict] = []
    for wk_offset in range(3, -1, -1):
        wk_end = week_end - timedelta(days=wk_offset * 7)
        wk_start = wk_end - timedelta(days=7)
        wk_rows = _filter_to_window(phase2, wk_start, wk_end)
        if wk_rows:
            weekly_hist.append({
                "week_end": wk_end.isoformat(),
                "avg_breadth_50": round(sum(r.get("breadth_50", 0) for r in wk_rows) / len(wk_rows), 4),
                "avg_signal_count": round(sum(r.get("signal_count", 0) for r in wk_rows) / len(wk_rows), 2),
                "min_drawdown": round(min(r.get("drawdown", 0) for r in wk_rows), 4),
            })

    return RegimeTransition(
        available=True,
        current_regime="bull" if (b50_latest or 0) >= 0.40 else "mixed",
        regime_confidence=round(confidence, 4),
        breadth_50_latest=b50_latest,
        breadth_75_latest=b75_latest,
        breadth_50_trend_7d=round(b50_trend_7, 6) if b50_trend_7 is not None else None,
        breadth_50_trend_30d=round(b50_trend_30, 6) if b50_trend_30 is not None else None,
        equity_drawdown_current=dd_current,
        equity_drawdown_max_30d=dd_max_30,
        signal_drought_consecutive=drought,
        signal_count_trend_7d=round(sig_trend_7, 4) if sig_trend_7 is not None else None,
        failed_breakout_proxy=round(fbp, 4) if fbp is not None else None,
        regime_state=regime_state,
        instability_score=instability,
        weekly_regime_history=weekly_hist,
        insufficiency_markers=insufficiency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 4 — Capital Efficiency
# ─────────────────────────────────────────────────────────────────────────────

def analyze_capital_efficiency(
    metrics: List[dict],
    week_end: date,
) -> CapitalEfficiency:
    insufficiency: List[str] = []
    window_30 = week_end - timedelta(days=30)
    window_7 = week_end - timedelta(days=7)

    recent_30 = _filter_to_window(metrics, window_30, week_end)
    recent_7 = _filter_to_window(metrics, window_7, week_end)

    if not recent_30:
        return CapitalEfficiency(
            available=False, avg_cash_ratio_7d=0.0, avg_cash_ratio_30d=0.0,
            avg_exposure_7d=0.0, avg_exposure_30d=0.0,
            slot_utilization_7d=0.0, slot_utilization_30d=0.0,
            idle_cash_ratio=0.0, peak_positions_30d=0,
            total_rsr_blocked_30d=0, total_breakout_blocked_30d=0,
            capital_fragmentation_score=0.0,
            scenario_50pct_capture_gain=None, scenario_100pct_capture_gain=None,
            scenario_200pct_capture_gain=None,
            insufficiency_markers=["No metrics data in 30d window"],
        )

    def _mean_field(rows: List[dict], key: str) -> float:
        vals = [r.get(key, 0.0) for r in rows if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    cash_7 = _mean_field(recent_7, "cash_ratio")
    cash_30 = _mean_field(recent_30, "cash_ratio")
    exp_7 = _mean_field(recent_7, "exposure")
    exp_30 = _mean_field(recent_30, "exposure")
    pos_30 = [r.get("positions", 0) for r in recent_30 if "positions" in r]
    pos_7 = [r.get("positions", 0) for r in recent_7 if "positions" in r]
    slot_7 = (_mean_field(recent_7, "positions") / MAX_POSITIONS) if recent_7 else 0.0
    slot_30 = (_mean_field(recent_30, "positions") / MAX_POSITIONS) if recent_30 else 0.0
    peak_pos = max(pos_30) if pos_30 else 0
    rsr_blocked = sum(r.get("signals_blocked_rsr", 0) for r in recent_30)
    bp_blocked = sum(r.get("signals_blocked_breakout", 0) for r in recent_30)

    # Capital fragmentation: high cash + many blocked signals = capital is the constraint
    idle_cash = cash_30
    frag_score = round(idle_cash * min(1.0, rsr_blocked / max(1, len(recent_30)) * 0.2), 4)

    # Scenario analysis: estimate additional capture with more capital
    # Assumption: each unit of additional capital proportionally reduces high-price exclusions
    # With 50% more capital, we might capture ~20% of currently blocked signals
    base_signal_loss = rsr_blocked + bp_blocked
    s50 = base_signal_loss * 0.20 * 0.03  # ~3% avg return per additional capture
    s100 = base_signal_loss * 0.35 * 0.03
    s200 = base_signal_loss * 0.55 * 0.03

    if rsr_blocked == 0 and bp_blocked == 0:
        insufficiency.append(
            "Zero signal blocks recorded — capital constraint scenario analysis based on zero blocked signals; "
            "scenarios are trivially zero"
        )

    return CapitalEfficiency(
        available=True,
        avg_cash_ratio_7d=round(cash_7, 4),
        avg_cash_ratio_30d=round(cash_30, 4),
        avg_exposure_7d=round(exp_7, 4),
        avg_exposure_30d=round(exp_30, 4),
        slot_utilization_7d=round(slot_7, 4),
        slot_utilization_30d=round(slot_30, 4),
        idle_cash_ratio=round(idle_cash, 4),
        peak_positions_30d=peak_pos,
        total_rsr_blocked_30d=rsr_blocked,
        total_breakout_blocked_30d=bp_blocked,
        capital_fragmentation_score=frag_score,
        scenario_50pct_capture_gain=round(s50, 4) if s50 is not None else None,
        scenario_100pct_capture_gain=round(s100, 4) if s100 is not None else None,
        scenario_200pct_capture_gain=round(s200, 4) if s200 is not None else None,
        insufficiency_markers=insufficiency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 5 — Holding Period Decay
# ─────────────────────────────────────────────────────────────────────────────

_HOLD_BUCKETS: List[Tuple[str, int, int]] = [
    ("1d", 0, 1), ("3d", 2, 3), ("5d", 4, 5),
    ("10d", 6, 10), ("20d", 11, 20), ("40d", 21, 40), ("60d+", 41, 9999),
]


def analyze_holding_period_decay(
    trades: List[dict],
    exit_records: List[dict],
    week_end: date,
    window_days: int = 90,
) -> HoldingPeriodDecay:
    insufficiency: List[str] = []
    window_start = week_end - timedelta(days=window_days)

    sells = _filter_to_window([t for t in trades if t.get("side") == "SELL"], window_start, week_end)
    ext_exits = _filter_to_window(exit_records, window_start, week_end)

    all_exits = sells + ext_exits
    if len(all_exits) < 5:
        insufficiency.append(
            f"Only {len(all_exits)} completed trades in {window_days}d window "
            f"(need ≥5 for statistical significance; currently {len(sells)} from trades.jsonl, "
            f"{len(ext_exits)} from exit_records)"
        )

    if not all_exits:
        return HoldingPeriodDecay(
            available=False, completed_trades=0,
            avg_hold_days=None, max_hold_days=None, min_hold_days=None,
            buckets=[], optimal_exit_bucket=None,
            insufficiency_markers=insufficiency,
        )

    hold_days_list = []
    pnl_pcts: Dict[str, List[float]] = {b[0]: [] for b in _HOLD_BUCKETS}

    for ex in all_exits:
        hd = ex.get("hold_days") or ex.get("holding_days")
        pnl = ex.get("pnl_pct") or ex.get("return_pct")
        if hd is None or pnl is None:
            continue
        try:
            hd = int(float(hd))
            pnl = float(pnl)
        except (TypeError, ValueError):
            continue
        hold_days_list.append(hd)
        for label, lo, hi in _HOLD_BUCKETS:
            if lo <= hd <= hi:
                pnl_pcts[label].append(pnl)
                break

    buckets: List[HoldingPeriodBucket] = []
    best_expectancy: Optional[float] = None
    optimal_bucket: Optional[str] = None
    for label, lo, hi in _HOLD_BUCKETS:
        pl = pnl_pcts[label]
        exp = _expectancy(pl)
        wrt = _win_rate(pl)
        buckets.append(HoldingPeriodBucket(
            label=label, trade_count=len(pl),
            avg_pnl_pct=round(exp, 4) if exp is not None else None,
            win_rate=round(wrt, 4) if wrt is not None else None,
        ))
        if exp is not None and (best_expectancy is None or exp > best_expectancy):
            best_expectancy = exp
            optimal_bucket = label

    return HoldingPeriodDecay(
        available=True,
        completed_trades=len(all_exits),
        avg_hold_days=round(_safe_mean([float(h) for h in hold_days_list]), 1) if hold_days_list else None,
        max_hold_days=max(hold_days_list) if hold_days_list else None,
        min_hold_days=min(hold_days_list) if hold_days_list else None,
        buckets=buckets,
        optimal_exit_bucket=optimal_bucket,
        insufficiency_markers=insufficiency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 6 — Expectancy Drift Detection
# ─────────────────────────────────────────────────────────────────────────────

def analyze_expectancy_drift(
    trades: List[dict],
    phase2: List[dict],
    week_end: date,
) -> ExpectancyDrift:
    insufficiency: List[str] = []
    warnings: List[str] = []

    window_7 = week_end - timedelta(days=7)
    window_30 = week_end - timedelta(days=30)

    p2_30 = _filter_to_window(phase2, window_30, week_end)
    p2_7 = _filter_to_window(phase2, window_7, week_end)

    # Equity trend
    eq_30 = [r.get("equity", 0.0) for r in p2_30 if "equity" in r]
    eq_7 = [r.get("equity", 0.0) for r in p2_7 if "equity" in r]
    eq_trend_7 = _linear_slope(eq_7)
    eq_trend_30 = _linear_slope(eq_30)

    # Drawdown acceleration
    dd_30 = [r.get("drawdown", 0.0) for r in p2_30 if "drawdown" in r]
    dd_accel: Optional[float] = None
    if len(dd_30) >= 7:
        early_dd = sum(dd_30[:7]) / 7
        late_dd = sum(dd_30[-7:]) / 7
        dd_accel = round(late_dd - early_dd, 4)
        if dd_accel < -0.03:
            warnings.append(f"Drawdown accelerating: early_avg={early_dd:.3f} → late_avg={late_dd:.3f}")

    # Signal count trend
    sig_7 = [r.get("signal_count", 0) for r in p2_7 if "signal_count" in r]
    sig_trend_7 = _linear_slope(sig_7)
    if sig_trend_7 is not None and sig_trend_7 < -0.5:
        warnings.append(f"Signal count declining: slope={sig_trend_7:.3f}/day over 7d")

    # Realized win rate from sells
    sells_30 = _filter_to_window([t for t in trades if t.get("side") == "SELL"], window_30, week_end)
    pnl_pcts = [s.get("pnl_pct", 0.0) for s in sells_30 if s.get("pnl_pct") is not None]
    rolling_wr_30 = _win_rate(pnl_pcts)
    rolling_exp_30 = _expectancy(pnl_pcts)

    if len(pnl_pcts) < 3:
        insufficiency.append(
            f"Only {len(pnl_pcts)} completed trade(s) in 30d window — "
            "rolling expectancy and win_rate statistically unreliable"
        )

    if rolling_exp_30 is not None and rolling_exp_30 < EXPECTANCY_DRIFT_WARN_THRESHOLD:
        warnings.append(
            f"Rolling expectancy below threshold: {rolling_exp_30:.3f} < {EXPECTANCY_DRIFT_WARN_THRESHOLD}"
        )

    # Edge stability score: composite
    edge_score: Optional[float] = None
    if p2_30:
        dd_component = 1.0 - min(1.0, abs(min(dd_30, default=0)) / 0.15) if dd_30 else 0.5
        sig_component = min(1.0, (sum(sig_7) / max(len(sig_7), 1)) / 3.0) if sig_7 else 0.5
        edge_score = round((dd_component * 0.6 + sig_component * 0.4), 4)

    # Alpha decay score: if signal trend is declining AND equity is flat/down
    alpha_decay: Optional[float] = None
    if eq_trend_30 is not None and sig_trend_7 is not None:
        alpha_decay = round(max(0.0, -sig_trend_7 * 0.5 + (-min(0, eq_trend_30) / 10000)), 4)

    # Alert level
    if warnings:
        if len(warnings) >= 2 or (dd_accel is not None and dd_accel < -0.05):
            alert = "HIGH"
        else:
            alert = "MEDIUM"
    else:
        alert = "LOW"

    return ExpectancyDrift(
        available=len(p2_30) > 0,
        equity_trend_7d=round(eq_trend_7, 2) if eq_trend_7 is not None else None,
        equity_trend_30d=round(eq_trend_30, 2) if eq_trend_30 is not None else None,
        drawdown_acceleration=dd_accel,
        signal_count_trend_7d=round(sig_trend_7, 4) if sig_trend_7 is not None else None,
        edge_stability_score=edge_score,
        alpha_decay_score=alpha_decay,
        instability_alert_level=alert,
        rolling_win_rate_30d=round(rolling_wr_30, 4) if rolling_wr_30 is not None else None,
        warnings=warnings,
        insufficiency_markers=insufficiency,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Section 7 — Research Priority Engine
# ─────────────────────────────────────────────────────────────────────────────

def generate_research_priorities(
    opp: OpportunityCapture,
    arch: ArchetypeExpectancy,
    regime: RegimeTransition,
    cap: CapitalEfficiency,
    hold: HoldingPeriodDecay,
    drift: ExpectancyDrift,
    trades: List[dict],
    week_end: date,
    ffe: Optional[object] = None,
) -> List[ResearchPriority]:
    items: List[Tuple[float, ResearchPriority]] = []
    rank_counter = [0]

    def add(score: float, category: str, title: str, confidence: str,
            evidence: str, symbols: List[str], justification: str, direction: str) -> None:
        rank_counter[0] += 1
        items.append((score, ResearchPriority(
            rank=rank_counter[0], category=category, title=title,
            confidence=confidence, evidence=evidence, affected_symbols=symbols,
            statistical_justification=justification, suggested_direction=direction,
        )))

    # P1: Drift / alert
    if drift.instability_alert_level == "HIGH":
        add(0.95, "expectancy_drift", "エッジ品質の高アラート検出",
            "HIGH",
            f"警告数={len(drift.warnings)}件: {'; '.join(drift.warnings[:2])}",
            [], f"edge_stability={drift.edge_stability_score}, DD加速={drift.drawdown_acceleration}",
            "ドライランを継続しシグナル選別基準を引き上げる")
    elif drift.instability_alert_level == "MEDIUM":
        add(0.70, "expectancy_drift", "エッジ品質の中程度の劣化傾向",
            "MEDIUM",
            f"警告: {'; '.join(drift.warnings[:2])}",
            [], f"edge_stability={drift.edge_stability_score}",
            "シグナル発生率とドローダウン進行を5日単位でモニタリング")

    # P2: Signal drought
    if regime.signal_drought_consecutive >= SIGNAL_DROUGHT_DAYS:
        add(0.85, "regime_transition", f"シグナル枯渇 {regime.signal_drought_consecutive}日連続",
            "HIGH",
            f"signal_drought_consecutive={regime.signal_drought_consecutive}日, "
            f"regime_state={regime.regime_state}",
            [], f"breadth_50={regime.breadth_50_latest}, instability={regime.instability_score}",
            "RSR閾値を一時的に引き下げるか、ユニバース拡張を検討")

    # P3: Capital constraint
    if cap.available and cap.avg_cash_ratio_30d > 0.70:
        symbols_affected = sorted({t.get("symbol","") for t in trades if t.get("side") == "BUY"})[:5]
        add(0.75, "capital_efficiency",
            f"資本効率低下: 平均遊休資金 {cap.avg_cash_ratio_30d*100:.0f}%",
            "HIGH" if cap.avg_cash_ratio_30d > 0.85 else "MEDIUM",
            f"30d平均現金比率={cap.avg_cash_ratio_30d:.3f}, "
            f"RSRブロック={cap.total_rsr_blocked_30d}件, スロット稼働={cap.slot_utilization_30d:.2f}",
            symbols_affected,
            f"slot_utilization_30d={cap.slot_utilization_30d}, peak_positions={cap.peak_positions_30d}",
            "ポジション数上限を4〜5に引き上げる検証 or 1枚当たり資本配分の削減")

    # P4: Overextension
    if arch.available and arch.overextension_alert:
        add(0.72, "entry_archetype",
            f"エントリー過熱度アラート: 平均ATR比={arch.avg_atr_pct*100:.2f}%",
            "MEDIUM",
            f"avg_atr_pct={arch.avg_atr_pct:.4f} > 閾値2.5%",
            [], f"total_buys={arch.total_buys}, overextension_score計算済み",
            "ATR比に上限フィルター(例: ATR/price < 2.0%)を追加してエントリー質を改善")

    # P5: Regime mismatch
    if regime.regime_state in ("mean_reverting", "high_volatility", "momentum_exhaustion"):
        add(0.68, "regime_transition",
            f"レジーム不整合: {regime.regime_state} (トレンドフォロー戦略に不利)",
            "MEDIUM" if regime.regime_confidence > 0.65 else "LOW",
            f"regime_state={regime.regime_state}, confidence={regime.regime_confidence}, "
            f"breadth_50={regime.breadth_50_latest}, instability={regime.instability_score}",
            [],
            f"breadth_50_trend_7d={regime.breadth_50_trend_7d}",
            "ポジションサイズを縮小し、レジーム転換（breadth_50>0.5）まで新規エントリーを制限")

    # P6: Holding period insufficient data but we have some sells
    if hold.completed_trades > 0 and hold.completed_trades < 5:
        add(0.55, "holding_period",
            "保有期間分析データ不足: 統計的意義のある分析には売却記録が5件必要",
            "LOW",
            f"completed_trades={hold.completed_trades}件, avg_hold={hold.avg_hold_days}d",
            [], "min_trade_required=5 (CLAUDE.md VALIDATION)",
            "売却後に本分析を再実行して保有期間ごとの期待値を確認する")

    # P7: RSR block dominance
    if opp.available and opp.weekly_rsr_blocked > 5 * max(opp.weekly_candidates_executed, 1):
        add(0.65, "opportunity_capture",
            f"RSRフィルターが機会損失の主因: {opp.weekly_rsr_blocked}件ブロック vs {opp.weekly_candidates_executed}件実行",
            "MEDIUM",
            f"rsr_blocked={opp.weekly_rsr_blocked}, executed={opp.weekly_candidates_executed}, "
            f"capture_ratio={opp.opportunity_capture_ratio}",
            [],
            f"false_negative_rate={opp.false_negative_rate}",
            "RSR閾値(現在75.0)の感度分析を実施し、65-80の範囲でウォークフォワードを再実行")

    # P_FFE1: ATR overheating confirmed by forward returns
    if ffe is not None:
        try:
            if getattr(ffe, "overheating_verdict", "") == "overheating_confirmed":
                _ffe_conf = getattr(ffe, "overheating_confidence", "MEDIUM")
                _ffe_evid = getattr(ffe, "overheating_evidence", "")
                add(0.78, "entry_archetype",
                    "フィーチャー分析: 高ATRエントリーの順伝播期待値が低ATRを下回る",
                    _ffe_conf, _ffe_evid, [],
                    f"enriched_snapshots={getattr(ffe, 'enriched_count', 0)} — ATR bucket attribution",
                    "ATR/price < 2.5%のエントリーに絞るフィルターを追加し、高ATRエントリー率を削減")
        except Exception:
            pass

    # P_FFE2: Sector monoculture
    if ffe is not None:
        try:
            _sc = getattr(ffe, "sector_concentration", None)
            if _sc is not None and getattr(_sc, "monoculture_alert", False):
                _dom = getattr(_sc, "dominant_sector", "unknown")
                _share = getattr(_sc, "dominant_sector_share", 0.0)
                _herf = getattr(_sc, "concentration_index", 0.0)
                add(0.72, "portfolio_construction",
                    f"セクター集中リスク: {_dom}が全エントリーの{_share*100:.0f}%を占める",
                    "HIGH" if _share > 0.75 else "MEDIUM",
                    f"dominant_sector={_dom} share={_share:.3f} Herfindahl={_herf:.3f}",
                    [],
                    f"effective_sector_count={getattr(_sc, 'effective_sector_count', 0):.1f}",
                    "セクター分散を強制する制約（同一セクター上限2銘柄）を追加")
        except Exception:
            pass

    # P_FFE3: Counterfactual winner alert
    if ffe is not None:
        try:
            _cf_list = getattr(ffe, "counterfactual", [])
            if _cf_list:
                _top_cf = _cf_list[0]
                _cf_exp5 = getattr(_top_cf, "expectancy_5d", None)
                _cf_cnt = getattr(_top_cf, "count", 0)
                _cf_reason = getattr(_top_cf, "skip_reason", "")
                if _cf_exp5 is not None and _cf_exp5 > 0.02 and _cf_cnt >= 2:
                    add(0.65, "opportunity_capture",
                        f"スキップ銘柄が実行銘柄より高い順伝播期待値: skip_reason={_cf_reason}",
                        "MEDIUM",
                        f"skip_reason='{_cf_reason}' expectancy_5d={_cf_exp5:.4f} count={_cf_cnt}",
                        [],
                        f"missed_alpha_score={getattr(_top_cf, 'missed_alpha_score', 0):.4f}",
                        f"'{_cf_reason}'でブロックされたシグナルの再評価を検討")
        except Exception:
            pass

    # Fallback: データ充足時にピンポイントで出さないと "none" になる → 必ず最低1件出力
    if not items:
        add(0.40, "data_collection",
            "週次分析データ蓄積フェーズ: 現時点では統計的に有意な問題を検出せず",
            "LOW",
            f"trades={len(trades)}件, phase2_rows={len([])}, metrics={opp.weekly_candidates_executed}件実行",
            [],
            "min_trade_required=5 (CLAUDE.md VALIDATION) — 現在データ蓄積中",
            "ドライランを継続し、売却記録を5件以上積み上げてから本分析を再実行")

    # Sort by score descending, re-assign ranks
    items.sort(key=lambda x: x[0], reverse=True)
    priorities: List[ResearchPriority] = []
    for i, (_, pr) in enumerate(items, start=1):
        pr.rank = i
        priorities.append(pr)
    return priorities


# ─────────────────────────────────────────────────────────────────────────────
# Chart generation (matplotlib — fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_charts(
    trades: List[dict],
    metrics: List[dict],
    phase2: List[dict],
    report: "WeeklyIntelligenceReport",
    chart_dir: Path,
    ffe_report: Optional[object] = None,
) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not available — charts skipped")
        return []

    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    def _save(fig: "plt.Figure", name: str) -> Optional[str]:
        p = chart_dir / name
        try:
            fig.savefig(str(p), dpi=100, bbox_inches="tight")
            plt.close(fig)
            return str(p)
        except Exception as exc:
            logger.warning("Chart save failed (%s): %s", name, exc)
            plt.close(fig)
            return None

    # 1. Equity curve
    if phase2:
        p2_sorted = sorted(phase2, key=lambda r: r.get("date", ""))
        dates = [datetime.strptime(r["date"][:10], "%Y-%m-%d") for r in p2_sorted if "date" in r]
        equities = [r.get("equity", 0) for r in p2_sorted if "date" in r]
        if len(dates) >= 2:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(dates, equities, color="#2196F3", linewidth=1.5)
            ax.fill_between(dates, min(equities), equities, alpha=0.15, color="#2196F3")
            ax.set_title("Equity Curve (JPY)", fontsize=12)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            p = _save(fig, "01_equity_curve.png")
            if p:
                paths.append(p)

    # 2. Breadth / regime
    if phase2:
        p2_sorted = sorted(phase2, key=lambda r: r.get("date", ""))
        dates = [datetime.strptime(r["date"][:10], "%Y-%m-%d") for r in p2_sorted if "date" in r]
        b50 = [r.get("breadth_50", 0) for r in p2_sorted if "date" in r]
        b75 = [r.get("breadth_75", 0) for r in p2_sorted if "date" in r]
        if len(dates) >= 2:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.plot(dates, b50, label="breadth_50", color="#4CAF50", linewidth=1.5)
            ax.plot(dates, b75, label="breadth_75", color="#FF9800", linewidth=1.0, linestyle="--")
            ax.axhline(BREADTH_TREND_PERSISTENT, color="green", alpha=0.4, linestyle=":")
            ax.axhline(BREADTH_TREND_WEAK, color="red", alpha=0.4, linestyle=":")
            ax.set_title("Market Breadth (RSR > 50/75 percentile)", fontsize=12)
            ax.legend(fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = _save(fig, "02_breadth_regime.png")
            if p:
                paths.append(p)

    # 3. Capital utilization (cash ratio + exposure)
    if metrics:
        m_sorted = sorted(metrics, key=lambda r: r.get("date", ""))
        dates = [datetime.strptime(r["date"][:10], "%Y-%m-%d") for r in m_sorted if "date" in r]
        cash = [r.get("cash_ratio", 0) for r in m_sorted if "date" in r]
        exp = [r.get("exposure", 0) for r in m_sorted if "date" in r]
        if len(dates) >= 2:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.stackplot(dates, [exp, cash], labels=["exposure", "idle_cash"],
                         colors=["#2196F3", "#BDBDBD"], alpha=0.7)
            ax.set_ylim(0, 1)
            ax.set_title("Capital Utilization (exposure vs idle cash)", fontsize=12)
            ax.legend(loc="upper left", fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(alpha=0.2)
            fig.tight_layout()
            p = _save(fig, "03_capital_utilization.png")
            if p:
                paths.append(p)

    # 4. Signal flow (RSR blocked vs executed)
    if metrics:
        m_sorted = sorted(metrics, key=lambda r: r.get("date", ""))
        dates = [datetime.strptime(r["date"][:10], "%Y-%m-%d") for r in m_sorted if "date" in r]
        rsr_bl = [r.get("signals_blocked_rsr", 0) for r in m_sorted if "date" in r]
        cands = [r.get("candidate_count", 0) for r in m_sorted if "date" in r]
        if len(dates) >= 2:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(dates, rsr_bl, label="blocked_by_rsr", color="#F44336", alpha=0.6, width=0.8)
            ax.bar(dates, cands, label="executed", color="#4CAF50", alpha=0.8, width=0.8, bottom=rsr_bl)
            ax.set_title("Signal Flow: RSR-blocked vs Executed Candidates", fontsize=12)
            ax.legend(fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            p = _save(fig, "04_signal_flow.png")
            if p:
                paths.append(p)

    # 5. Drawdown
    if phase2:
        p2_sorted = sorted(phase2, key=lambda r: r.get("date", ""))
        dates = [datetime.strptime(r["date"][:10], "%Y-%m-%d") for r in p2_sorted if "date" in r]
        dd = [r.get("drawdown", 0) * 100 for r in p2_sorted if "date" in r]
        if len(dates) >= 2:
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.fill_between(dates, dd, 0, where=[d < 0 for d in dd], alpha=0.6, color="#F44336")
            ax.plot(dates, dd, color="#B71C1C", linewidth=1.0)
            ax.axhline(-15, color="red", linestyle="--", alpha=0.5, label="DD limit -15%")
            ax.set_title("Portfolio Drawdown (%)", fontsize=12)
            ax.legend(fontsize=9)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))
            ax.grid(alpha=0.3)
            fig.tight_layout()
            p = _save(fig, "05_drawdown.png")
            if p:
                paths.append(p)

    # 6-10. Feature forward expectancy charts (fail-open)
    if ffe_report is not None:
        try:
            from src.analytics.feature_forward_expectancy import generate_ffe_charts as _gen_ffe
            ffe_paths = _gen_ffe(ffe_report, chart_dir)
            paths.extend(ffe_paths)
        except Exception as _ffe_chart_err:
            logger.warning("FFE chart generation failed: %s", _ffe_chart_err)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Atomic report writers
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_wmi_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _fmt_opt(v: Optional[float], fmt: str = ".4f", suffix: str = "") -> str:
    if v is None:
        return "N/A"
    return f"{v:{fmt}}{suffix}"


def generate_markdown(report: WeeklyIntelligenceReport) -> str:
    r = report
    lines: List[str] = []
    a = lines.append

    a(f"# 週次市場インテリジェンス レポート {r.report_date}")
    a(f"\n生成日時: {r.generated_at}  |  run_id: `{r.run_id}`")
    a(f"対象週: {r.week_start} 〜 {r.week_end}\n")
    a("---\n")

    # Data availability
    a("## データ可用性\n")
    for source, avail in sorted(r.data_availability.items()):
        icon = "✓" if avail else "✗"
        a(f"- `{source}`: {icon}")
    if r.data_quality_warnings:
        a("\n**品質警告:**")
        for w in r.data_quality_warnings:
            a(f"- {w}")
    a("")

    # Section 1
    opp = r.opportunity_capture
    a("## Section 1 — 機会捕捉分析\n")
    a(f"- 機会捕捉率 (capture_ratio): {_fmt_opt(opp.opportunity_capture_ratio, '.3f')}")
    a(f"- 偽陰性率 (false_negative_rate): {_fmt_opt(opp.false_negative_rate, '.3f')}")
    a(f"- ミスアルファスコア (missed_alpha_score): {_fmt_opt(opp.missed_alpha_score, '.4f')}")
    a(f"- 30日間 RSRフィルターブロック: {opp.weekly_rsr_blocked}件")
    a(f"- 30日間 ブレイクアウトフィルターブロック: {opp.weekly_breakout_blocked}件")
    a(f"- 30日間 実行シグナル数: {opp.weekly_candidates_executed}件")
    if opp.insufficiency_markers:
        a("\n**データ不足マーカー:**")
        for m in opp.insufficiency_markers:
            a(f"> {m}")
    a("")

    # Section 2
    arch = r.archetype_expectancy
    a("## Section 2 — エントリー原型期待値\n")
    a(f"- 対象期間BUY件数: {arch.total_buys}")
    a(f"- 対象期間SELL件数: {arch.total_sells}")
    a(f"- 平均ATR比 (avg_atr_pct): {_fmt_opt(arch.avg_atr_pct, '.4f')} "
      f"{'**[過熱アラート]**' if arch.overextension_alert else ''}")
    a(f"- 実現勝率: {_fmt_opt(arch.win_rate, '.3f')}")
    a(f"- 実現期待値: {_fmt_opt(arch.realized_expectancy, '.4f')}")
    a(f"- 平均保有日数: {_fmt_opt(arch.avg_hold_days, '.1f')}日")
    if arch.clusters:
        a("\n**アーキタイプ別集計:**\n")
        a("| アーキタイプ | BUY件数 | ATR比 | 実現期待値 | 勝率 | 過熱スコア |")
        a("|---|---|---|---|---|---|")
        for cl in arch.clusters:
            a(f"| {cl.name} | {cl.trade_count} | {cl.avg_atr_pct:.4f} | "
              f"{_fmt_opt(cl.realized_expectancy, '.4f')} | "
              f"{_fmt_opt(cl.realized_win_rate, '.3f')} | {cl.overextension_score:.2f} |")
    if arch.sector_distribution:
        a("\n**セクター分布:**")
        for sec, cnt in sorted(arch.sector_distribution.items(), key=lambda x: -x[1]):
            a(f"  - {sec}: {cnt}件")
    if arch.insufficiency_markers:
        a("\n**データ不足マーカー:**")
        for m in arch.insufficiency_markers:
            a(f"> {m}")
    a("")

    # Section 3
    reg = r.regime_transition
    a("## Section 3 — 市場レジーム遷移\n")
    a(f"- 現在レジーム: **{reg.current_regime}** (信頼度: {reg.regime_confidence:.2f})")
    a(f"- レジーム状態: **{reg.regime_state}**")
    a(f"- 不安定スコア: {reg.instability_score:.4f}")
    a(f"- breadth_50 (最新): {_fmt_opt(reg.breadth_50_latest, '.4f')}")
    a(f"- breadth_75 (最新): {_fmt_opt(reg.breadth_75_latest, '.4f')}")
    a(f"- breadth_50 トレンド 7d: {_fmt_opt(reg.breadth_50_trend_7d, '+.5f')}/日")
    a(f"- breadth_50 トレンド 30d: {_fmt_opt(reg.breadth_50_trend_30d, '+.5f')}/日")
    a(f"- 現在ドローダウン: {_fmt_opt(reg.equity_drawdown_current, '.3f')}")
    a(f"- 30日最大ドローダウン: {_fmt_opt(reg.equity_drawdown_max_30d, '.3f')}")
    a(f"- シグナル枯渇連続日数: {reg.signal_drought_consecutive}日")
    if reg.weekly_regime_history:
        a("\n**週次レジーム履歴:**\n")
        a("| 週末 | avg_breadth_50 | avg_signal | min_drawdown |")
        a("|---|---|---|---|")
        for wh in reg.weekly_regime_history:
            a(f"| {wh['week_end']} | {wh['avg_breadth_50']:.4f} | "
              f"{wh['avg_signal_count']:.1f} | {wh['min_drawdown']:.4f} |")
    if reg.insufficiency_markers:
        a("\n**データ不足マーカー:**")
        for m in reg.insufficiency_markers:
            a(f"> {m}")
    a("")

    # Section 4
    cap = r.capital_efficiency
    a("## Section 4 — 資本効率\n")
    a(f"- 7d 平均現金比率: {_fmt_opt(cap.avg_cash_ratio_7d, '.3f')}")
    a(f"- 30d 平均現金比率: {_fmt_opt(cap.avg_cash_ratio_30d, '.3f')}")
    a(f"- 7d 平均エクスポージャー: {_fmt_opt(cap.avg_exposure_7d, '.3f')}")
    a(f"- 30d 平均エクスポージャー: {_fmt_opt(cap.avg_exposure_30d, '.3f')}")
    a(f"- 7d スロット稼働率: {_fmt_opt(cap.slot_utilization_7d, '.3f')}")
    a(f"- 30d スロット稼働率: {_fmt_opt(cap.slot_utilization_30d, '.3f')}")
    a(f"- 30日間RSRブロック合計: {cap.total_rsr_blocked_30d}件")
    a(f"- 資本断片化スコア: {_fmt_opt(cap.capital_fragmentation_score, '.4f')}")
    a("\n**資本増額シナリオ (30d blocked signals ベース):**")
    a(f"- +50% 追加捕捉推定: {_fmt_opt(cap.scenario_50pct_capture_gain, '.4f')}")
    a(f"- +100% 追加捕捉推定: {_fmt_opt(cap.scenario_100pct_capture_gain, '.4f')}")
    a(f"- +200% 追加捕捉推定: {_fmt_opt(cap.scenario_200pct_capture_gain, '.4f')}")
    if cap.insufficiency_markers:
        a("\n**データ不足マーカー:**")
        for m in cap.insufficiency_markers:
            a(f"> {m}")
    a("")

    # Section 5
    hold = r.holding_period
    a("## Section 5 — 保有期間減衰\n")
    a(f"- 完了取引数: {hold.completed_trades}")
    a(f"- 平均保有日数: {_fmt_opt(hold.avg_hold_days, '.1f')}日")
    a(f"- 最短保有日数: {hold.min_hold_days}")
    a(f"- 最長保有日数: {hold.max_hold_days}")
    a(f"- 最適出口バケット: {hold.optimal_exit_bucket or 'N/A'}")
    if hold.buckets:
        a("\n**バケット別期待値:**\n")
        a("| バケット | 件数 | 平均損益% | 勝率 |")
        a("|---|---|---|---|")
        for b in hold.buckets:
            a(f"| {b.label} | {b.trade_count} | "
              f"{_fmt_opt(b.avg_pnl_pct, '.4f')} | {_fmt_opt(b.win_rate, '.3f')} |")
    if hold.insufficiency_markers:
        a("\n**データ不足マーカー:**")
        for m in hold.insufficiency_markers:
            a(f"> {m}")
    a("")

    # Section 6
    drft = r.expectancy_drift
    a("## Section 6 — 期待値ドリフト検出\n")
    a(f"- アラートレベル: **{drft.instability_alert_level}**")
    a(f"- エッジ安定スコア: {_fmt_opt(drft.edge_stability_score, '.4f')}")
    a(f"- アルファ減衰スコア: {_fmt_opt(drft.alpha_decay_score, '.4f')}")
    a(f"- 7d 株式資産トレンド: {_fmt_opt(drft.equity_trend_7d, '+.1f')}円/日")
    a(f"- 30d 株式資産トレンド: {_fmt_opt(drft.equity_trend_30d, '+.1f')}円/日")
    a(f"- ドローダウン加速: {_fmt_opt(drft.drawdown_acceleration, '+.4f')}")
    a(f"- 7d シグナル数トレンド: {_fmt_opt(drft.signal_count_trend_7d, '+.4f')}/日")
    a(f"- 30d ローリング勝率: {_fmt_opt(drft.rolling_win_rate_30d, '.3f')}")
    if drft.warnings:
        a("\n**警告:**")
        for w in drft.warnings:
            a(f"- {w}")
    if drft.insufficiency_markers:
        a("\n**データ不足マーカー:**")
        for m in drft.insufficiency_markers:
            a(f"> {m}")
    a("")

    # Section 7
    a("## Section 7 — 研究優先度エンジン\n")
    for pr in r.research_priorities:
        a(f"### [{pr.rank}] {pr.title}")
        a(f"- **カテゴリ**: {pr.category}")
        a(f"- **信頼度**: {pr.confidence}")
        a(f"- **エビデンス**: {pr.evidence}")
        if pr.affected_symbols:
            a(f"- **関連銘柄**: {', '.join(pr.affected_symbols)}")
        a(f"- **統計的根拠**: {pr.statistical_justification}")
        a(f"- **推奨方向**: {pr.suggested_direction}")
        a("")

    # Section 8 — Feature forward expectancy
    ffe = r.forward_expectancy
    a("## Section 8 — フィーチャー順伝播期待値分析\n")
    if ffe is None or not getattr(ffe, "available", False):
        _ffe_markers = getattr(ffe, "insufficiency_markers", ["entry_features.jsonl 未蓄積"])
        a("> データ未蓄積: 本セクションはエントリーフィーチャースナップショット蓄積後に有効化されます")
        for _m in (_ffe_markers or [])[:3]:
            a(f"> {_m}")
    else:
        a(f"- スナップショット数: {ffe.snapshot_count} (エンリッチ済み: {ffe.enriched_count})")  # type: ignore[attr-defined]
        _dr = ffe.eval_date_range  # type: ignore[attr-defined]
        a(f"- 評価日範囲: {_dr[0]} 〜 {_dr[1]}")
        a(f"- 過熱度判定: **{ffe.overheating_verdict}** (信頼度: {ffe.overheating_confidence})")  # type: ignore[attr-defined]
        a(f"- エビデンス: {ffe.overheating_evidence}")  # type: ignore[attr-defined]

        _atrs = getattr(ffe, "atr_buckets", [])
        if _atrs:
            a("\n**ATRバケット別順伝播期待値:**\n")
            a("| バケット | n | 実行 | スキップ | 5d期待値 | 10d期待値 | 勝率5d | BF率 |")
            a("|---|---|---|---|---|---|---|---|")
            for _b in _atrs:
                a(f"| {_b.label} | {_b.sample_count} | {_b.executed_count} | {_b.skipped_count} |"
                  f" {_fmt_opt(_b.expectancy_5d, '.4f')} |"
                  f" {_fmt_opt(_b.expectancy_10d, '.4f')} |"
                  f" {_fmt_opt(_b.win_rate_5d, '.3f')} |"
                  f" {_fmt_opt(_b.breakout_failure_rate, '.3f')} |")

        _sc = getattr(ffe, "sector_concentration", None)
        if _sc is not None:
            a("\n**セクター集中度:**")
            _alert = " **[MONOCULTURE ALERT]**" if _sc.monoculture_alert else ""
            a(f"- effective_sector_count: {_sc.effective_sector_count:.1f} / Herfindahl: {_sc.concentration_index:.3f}{_alert}")
            a(f"- 支配セクター: {_sc.dominant_sector} ({_sc.dominant_sector_share*100:.0f}%)")

        _cf = getattr(ffe, "counterfactual", [])
        if _cf:
            a("\n**カウンターファクチュアル（スキップシグナル分析）:**\n")
            a("| スキップ理由 | 件数 | 5d期待値 | 勝率5d | ミスアルファ |")
            a("|---|---|---|---|---|")
            for _c in _cf[:5]:
                a(f"| {_c.skip_reason} | {_c.count} |"
                  f" {_fmt_opt(_c.expectancy_5d, '.4f')} |"
                  f" {_fmt_opt(_c.win_rate_5d, '.3f')} |"
                  f" {_fmt_opt(_c.missed_alpha_score, '.4f')} |")

        _ffe_markers2 = getattr(ffe, "insufficiency_markers", [])
        if _ffe_markers2:
            a("\n**データ不足マーカー:**")
            for _m in _ffe_markers2:
                a(f"> {_m}")
    a("")

    if r.chart_paths:
        a("## チャート\n")
        for cp in r.chart_paths:
            name = Path(cp).name
            a(f"- `{name}`")
        a("")

    return "\n".join(lines)


def generate_html(md_content: str, report_date: str) -> str:
    escaped = html_mod.escape(md_content)
    # Minimal markdown-to-html conversion for sections
    import re
    text = md_content
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'`(.+?)`', r'<code>\1</code>', text)
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^> (.+)$', r'<blockquote>\1</blockquote>', text, flags=re.MULTILINE)
    text = re.sub(r'\n\n', r'<br><br>', text)
    return (
        '<!DOCTYPE html><html lang="ja"><head>'
        '<meta charset="UTF-8">'
        f'<title>週次市場インテリジェンス {report_date}</title>'
        '<style>body{font-family:sans-serif;max-width:900px;margin:2rem auto;padding:1rem}'
        'h1,h2{border-bottom:1px solid #ccc}table{border-collapse:collapse}'
        'td,th{border:1px solid #ddd;padding:4px 8px}code{background:#f4f4f4;padding:2px 4px}'
        'blockquote{color:#888;border-left:3px solid #ddd;margin:0;padding:0 1em}'
        '</style></head><body>'
        + text
        + '</body></html>'
    )


def _run_id() -> str:
    return hashlib.sha256(uuid.uuid4().bytes).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Email sender
# ─────────────────────────────────────────────────────────────────────────────

def send_email_report(
    report: WeeklyIntelligenceReport,
    md_path: Path,
    chart_paths: List[str],
) -> bool:
    smtp_user = os.environ.get("NOTIFY_SMTP_USER", "")
    smtp_pass = os.environ.get("NOTIFY_SMTP_PASSWORD", "")
    smtp_to = os.environ.get("NOTIFY_TO", smtp_user)

    if not smtp_user or not smtp_pass:
        logger.warning("NOTIFY_SMTP_USER / NOTIFY_SMTP_PASSWORD not set — email skipped")
        return False

    try:
        subject = (
            f"[CHIBAAsset] 週次市場インテリジェンス {report.report_date} "
            f"| レジーム={report.regime_transition.regime_state} "
            f"| アラート={report.expectancy_drift.instability_alert_level}"
        )

        msg = MIMEMultipart("mixed")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = smtp_to

        # Body: markdown as plain text
        body_text = md_path.read_text(encoding="utf-8") if md_path.exists() else "(本文なし)"
        # Priority summary
        prio_lines = [f"■ 研究優先度 Top {min(3, len(report.research_priorities))}:"]
        for pr in report.research_priorities[:3]:
            prio_lines.append(f"  [{pr.rank}] {pr.title} ({pr.confidence})")
        prio_summary = "\n".join(prio_lines)

        body = MIMEText(prio_summary + "\n\n---\n\n" + body_text, "plain", "utf-8")
        msg.attach(body)

        # Attach charts (up to 10 — includes FFE charts 06-10)
        for cp in chart_paths[:10]:
            cp_path = Path(cp)
            if not cp_path.exists():
                continue
            try:
                with cp_path.open("rb") as f:
                    att = MIMEBase("image", "png")
                    att.set_payload(f.read())
                encoders.encode_base64(att)
                att.add_header("Content-Disposition", "attachment", filename=cp_path.name)
                msg.attach(att)
            except Exception as exc:
                logger.warning("Failed to attach chart %s: %s", cp_path.name, exc)

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as s:
            s.login(smtp_user, smtp_pass)
            s.sendmail(smtp_user, [smtp_to], msg.as_bytes())

        logger.info("Email sent to %s", smtp_to)
        return True

    except Exception as exc:
        logger.error("Email send failed: %s", exc)
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Archive (append-only JSONL)
# ─────────────────────────────────────────────────────────────────────────────

def _report_to_dict(report: WeeklyIntelligenceReport) -> dict:
    def _dc(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {k: _dc(v) for k, v in asdict(obj).items()}
        if isinstance(obj, list):
            return [_dc(i) for i in obj]
        if isinstance(obj, dict):
            return {k: _dc(v) for k, v in obj.items()}
        return obj
    return _dc(report)


def append_to_archive(report: WeeklyIntelligenceReport, archive_path: Path) -> None:
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(_report_to_dict(report), ensure_ascii=False, sort_keys=True)
    with archive_path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(record + "\n")
    logger.info("Appended to archive: %s", archive_path)


# ─────────────────────────────────────────────────────────────────────────────
# WeeklyMarketIntelligenceEngine
# ─────────────────────────────────────────────────────────────────────────────

class WeeklyMarketIntelligenceEngine:
    """
    Autonomous weekly market intelligence runtime.

    Usage:
        engine = WeeklyMarketIntelligenceEngine(base_dir=Path("C:/ai-trading"))
        report = engine.run(send_email=True)
    """

    def __init__(
        self,
        base_dir: Path,
        report_dir: Optional[Path] = None,
        archive_file: Optional[Path] = None,
        feature_snapshots_path: Optional[Path] = None,
        position_lifecycle_path: Optional[Path] = None,
        forward_cache_dir: Optional[Path] = None,
    ) -> None:
        self.base_dir = base_dir
        self.report_dir = report_dir or (base_dir / "runtime" / "reports")
        self.archive_file = archive_file or (self.report_dir / "weekly_intelligence_archive.jsonl")
        self.feature_snapshots_path = feature_snapshots_path or (
            base_dir / "runtime" / "feature_snapshots" / "entry_features.jsonl"
        )
        self.position_lifecycle_path = position_lifecycle_path or (
            base_dir / "runtime" / "feature_snapshots" / "position_lifecycle.jsonl"
        )
        self.forward_cache_dir = forward_cache_dir or (
            base_dir / "runtime" / "feature_snapshots" / "forward_return_cache"
        )

    def _resolve_week(self, reference_date: Optional[date] = None) -> Tuple[date, date]:
        ref = reference_date or datetime.now(JST).date()
        # Week ending on Friday (weekday 4); if today is Friday use today
        days_since_friday = (ref.weekday() - 4) % 7
        week_end = ref - timedelta(days=days_since_friday)
        week_start = week_end - timedelta(days=6)
        return week_start, week_end

    def run(
        self,
        reference_date: Optional[date] = None,
        send_email: bool = True,
        dry_run: bool = False,
    ) -> WeeklyIntelligenceReport:
        now_jst = datetime.now(JST)
        week_start, week_end = self._resolve_week(reference_date)
        run_id = _run_id()
        quality_warnings: List[str] = []

        logger.info(
            "=== WeeklyMarketIntelligenceEngine start run_id=%s week=%s/%s ===",
            run_id, week_start, week_end,
        )

        # Load data
        try:
            trades, trades_ok = _load_trades(self.base_dir)
        except WeeklyIntelligenceError:
            raise  # fail-closed on corrupted trades

        metrics, metrics_ok = _load_metrics(self.base_dir)
        phase2, phase2_ok = _load_phase2(self.base_dir)
        skipped, skipped_ok = _load_skipped_opportunities(self.base_dir)
        exit_records, exits_ok = _load_exit_records(self.base_dir)

        data_avail = {
            "trades.jsonl": trades_ok,
            "diagnostics/metrics.jsonl": metrics_ok,
            "phase2_live_metrics.jsonl": phase2_ok,
            "skipped_opportunities.jsonl": skipped_ok,
            "exit_records.jsonl": exits_ok,
        }

        if not trades_ok:
            quality_warnings.append("trades.jsonl unavailable")
        if not metrics_ok:
            quality_warnings.append("diagnostics/metrics.jsonl unavailable")
        if not phase2_ok:
            quality_warnings.append("phase2_live_metrics.jsonl unavailable")

        # Run sections
        opp = analyze_opportunity_capture(metrics, skipped, week_start, week_end)
        arch = analyze_archetype_expectancy(trades, week_end)
        regime = analyze_regime_transition(metrics, phase2, week_start, week_end)
        cap = analyze_capital_efficiency(metrics, week_end)
        hold = analyze_holding_period_decay(trades, exit_records, week_end)
        drift = analyze_expectancy_drift(trades, phase2, week_end)

        # Feature forward expectancy (fail-open)
        ffe_report: Optional[object] = None
        try:
            from src.analytics.feature_forward_expectancy import FeatureForwardExpectancyEngine as _FFEEngine
            _ffe_engine = _FFEEngine(
                snapshots_path=self.feature_snapshots_path,
                lifecycle_path=self.position_lifecycle_path,
                cache_dir=self.forward_cache_dir,
            )
            ffe_report = _ffe_engine.run(week_start=week_start, week_end=week_end, trades=trades)
            data_avail["entry_features.jsonl"] = getattr(ffe_report, "available", False)
            _ffe_markers = getattr(ffe_report, "insufficiency_markers", [])
            for _m in _ffe_markers[:2]:
                quality_warnings.append(f"[FFE] {_m}")
        except Exception as _ffe_err:
            logger.warning("FeatureForwardExpectancy failed (ignored): %s", _ffe_err)
            data_avail["entry_features.jsonl"] = False

        priorities = generate_research_priorities(
            opp, arch, regime, cap, hold, drift, trades, week_end, ffe=ffe_report
        )

        # Charts
        date_tag = week_end.strftime("%Y%m%d")
        chart_dir = self.report_dir / "charts" / f"weekly_{date_tag}"
        chart_paths: List[str] = []
        if not dry_run:
            try:
                chart_paths = _generate_charts(trades, metrics, phase2,
                                                None, chart_dir,  # type: ignore[arg-type]
                                                ffe_report=ffe_report)
            except Exception as exc:
                logger.warning("Chart generation failed: %s", exc)
                quality_warnings.append(f"Chart generation error: {exc}")

        # Build report
        report = WeeklyIntelligenceReport(
            run_id=run_id,
            generated_at=now_jst.isoformat(),
            report_date=week_end.isoformat(),
            week_start=week_start.isoformat(),
            week_end=week_end.isoformat(),
            schema_version=SCHEMA_VERSION,
            opportunity_capture=opp,
            archetype_expectancy=arch,
            regime_transition=regime,
            capital_efficiency=cap,
            holding_period=hold,
            expectancy_drift=drift,
            research_priorities=priorities,
            data_availability=data_avail,
            data_quality_warnings=quality_warnings,
            chart_paths=chart_paths,
            forward_expectancy=ffe_report,
        )

        if dry_run:
            logger.info("[DRY RUN] Report generated (no files written)")
            return report

        # Write reports
        md_path = self.report_dir / f"weekly_market_intelligence_{date_tag}.md"
        html_path = self.report_dir / f"weekly_market_intelligence_{date_tag}.html"
        json_path = self.report_dir / f"weekly_market_intelligence_{date_tag}.json"

        md_content = generate_markdown(report)
        _atomic_write(md_path, md_content)
        _atomic_write(html_path, generate_html(md_content, week_end.isoformat()))
        _atomic_write(json_path, json.dumps(_report_to_dict(report), ensure_ascii=False,
                                            indent=2, sort_keys=True))

        report.report_md_path = str(md_path)
        report.report_html_path = str(html_path)
        report.report_json_path = str(json_path)

        logger.info("Reports written: %s / %s / %s", md_path, html_path, json_path)

        # Archive
        try:
            append_to_archive(report, self.archive_file)
        except Exception as exc:
            logger.error("Archive append failed: %s", exc)
            quality_warnings.append(f"Archive append error: {exc}")

        # Email
        if send_email:
            email_ok = send_email_report(report, md_path, chart_paths)
            if not email_ok:
                quality_warnings.append("Email send failed or skipped")

        logger.info(
            "=== WeeklyMarketIntelligenceEngine done priorities=%d warnings=%d ===",
            len(priorities), len(quality_warnings),
        )
        return report


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: load archive
# ─────────────────────────────────────────────────────────────────────────────

def load_weekly_archive(archive_path: Path) -> List[dict]:
    rows, _ = _load_jsonl(archive_path)
    return rows
