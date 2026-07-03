"""
src/analytics/executed_vs_skipped_expectancy.py — Executed-vs-skipped expectancy analytics

Observational analytics only.
No strategy modification. No parameter optimization. No adaptive trading changes.

Answers:
  - Do skipped opportunities outperform executed trades?
  - Which skip reasons cause the most alpha leakage?
  - Is capital allocation systematically distorted?
  - Are risk gates over-filtering by regime/sector/ATR?
  - Is universe selection degrading over time?

Design:
  - append-only JSONL (records never mutated)
  - opportunity_id = sha256(symbol + eval_date + executed + skip_reason) — dedup key
  - atomic append via temp-file rename
  - UTF-8, fsync durability
  - deterministic: same inputs → same output
  - Windows compatible
"""
from __future__ import annotations

import hashlib
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

SCHEMA_VERSION: int = 1

# ── ATR buckets ───────────────────────────────────────────────────────────────
ATR_BUCKETS: List[Tuple[str, float, float]] = [
    ("<1.5%",    0.000, 0.015),
    ("1.5-2.5%", 0.015, 0.025),
    ("2.5-4%",   0.025, 0.040),
    ("4%+",      0.040, 9.999),
]

# ── Slot utilization buckets ──────────────────────────────────────────────────
SLOT_BUCKETS: List[Tuple[str, float, float]] = [
    ("0-33%",  0.000, 0.333),
    ("33-67%", 0.333, 0.667),
    ("67-100%", 0.667, 1.001),
]

# ── Skip reason taxonomy ──────────────────────────────────────────────────────
SKIP_CAPITAL_CONSTRAINT   = "capital_constraint"
SKIP_PORTFOLIO_HEAT       = "portfolio_heat_limit"
SKIP_SECTOR_EXPOSURE      = "sector_exposure_limit"
SKIP_LIQUIDITY_FILTER     = "liquidity_filter"
SKIP_VOLATILITY_FILTER    = "volatility_filter"
SKIP_UNIVERSE_FILTER      = "universe_filter"
SKIP_DUPLICATE_SIGNAL     = "duplicate_signal"
SKIP_COOLDOWN             = "cooldown"
SKIP_EXECUTION_FAILURE    = "execution_failure"
SKIP_MANUAL_OVERRIDE      = "manual_override"
SKIP_UNKNOWN              = "unknown"

ALL_SKIP_REASONS = (
    SKIP_CAPITAL_CONSTRAINT,
    SKIP_PORTFOLIO_HEAT,
    SKIP_SECTOR_EXPOSURE,
    SKIP_LIQUIDITY_FILTER,
    SKIP_VOLATILITY_FILTER,
    SKIP_UNIVERSE_FILTER,
    SKIP_DUPLICATE_SIGNAL,
    SKIP_COOLDOWN,
    SKIP_EXECUTION_FAILURE,
    SKIP_MANUAL_OVERRIDE,
    SKIP_UNKNOWN,
)


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _opt_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _safe_mean(values: List[float]) -> Optional[float]:
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 6)


def _win_rate(values: List[float]) -> Optional[float]:
    valid = [v for v in values if v is not None and math.isfinite(v)]
    if not valid:
        return None
    return round(sum(1 for v in valid if v > 0) / len(valid), 4)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TradeOpportunityRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TradeOpportunityRecord:
    """
    Immutable record for a single evaluated trade opportunity (executed or skipped).
    Written once at evaluation time; forward returns populated by enrichment.
    """
    opportunity_id: str          # sha256 dedup key
    schema_version: int
    eval_date: str               # YYYY-MM-DD
    symbol: str
    executed: bool
    skip_reason: Optional[str]   # None if executed; one of SKIP_* if skipped
    capital_available_pct: float # fraction of capital available at eval time
    portfolio_heat: float        # portfolio heat proxy (0-1)
    slot_utilization: float      # fraction of max_positions occupied (0-1)
    sector: str
    market_regime: str
    atr_pct: float               # ATR20 / price
    rs_rank: int                 # relative strength rank (1=best)
    entry_score: float           # composite entry score (0-1)
    liquidity_score: float       # liquidity score (0-1)
    position_lifecycle_available: bool
    # Forward attribution — None until enriched
    forward_5d_return: Optional[float] = None
    forward_20d_return: Optional[float] = None
    enrichment_status: str = "pending"

    @staticmethod
    def create(
        eval_date: str,
        symbol: str,
        executed: bool,
        skip_reason: Optional[str],
        capital_available_pct: float,
        portfolio_heat: float,
        slot_utilization: float,
        sector: str,
        market_regime: str,
        atr_pct: float,
        rs_rank: int,
        entry_score: float,
        liquidity_score: float,
        position_lifecycle_available: bool,
        forward_5d_return: Optional[float] = None,
        forward_20d_return: Optional[float] = None,
        enrichment_status: str = "pending",
    ) -> "TradeOpportunityRecord":
        opp_id = _sha256(_canonical_json({
            "eval_date": eval_date,
            "executed": executed,
            "skip_reason": skip_reason,
            "symbol": symbol,
        }))
        return TradeOpportunityRecord(
            opportunity_id=opp_id,
            schema_version=SCHEMA_VERSION,
            eval_date=eval_date,
            symbol=symbol,
            executed=executed,
            skip_reason=skip_reason,
            capital_available_pct=capital_available_pct,
            portfolio_heat=portfolio_heat,
            slot_utilization=slot_utilization,
            sector=sector,
            market_regime=market_regime,
            atr_pct=atr_pct,
            rs_rank=rs_rank,
            entry_score=entry_score,
            liquidity_score=liquidity_score,
            position_lifecycle_available=position_lifecycle_available,
            forward_5d_return=forward_5d_return,
            forward_20d_return=forward_20d_return,
            enrichment_status=enrichment_status,
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TradeOpportunityRecord":
        return TradeOpportunityRecord(
            opportunity_id=d.get("opportunity_id", ""),
            schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
            eval_date=d.get("eval_date", ""),
            symbol=d.get("symbol", ""),
            executed=bool(d.get("executed", False)),
            skip_reason=d.get("skip_reason"),
            capital_available_pct=float(d.get("capital_available_pct", 0.0)),
            portfolio_heat=float(d.get("portfolio_heat", 0.0)),
            slot_utilization=float(d.get("slot_utilization", 0.0)),
            sector=d.get("sector", ""),
            market_regime=d.get("market_regime", ""),
            atr_pct=float(d.get("atr_pct", 0.0)),
            rs_rank=int(d.get("rs_rank", 0)),
            entry_score=float(d.get("entry_score", 0.0)),
            liquidity_score=float(d.get("liquidity_score", 0.0)),
            position_lifecycle_available=bool(d.get("position_lifecycle_available", False)),
            forward_5d_return=_opt_float(d.get("forward_5d_return")),
            forward_20d_return=_opt_float(d.get("forward_20d_return")),
            enrichment_status=d.get("enrichment_status", "pending"),
        )


# ─────────────────────────────────────────────────────────────────────────────
# 2. SkipReasonClassifier
# ─────────────────────────────────────────────────────────────────────────────

class SkipReasonClassifier:
    """
    Classifies raw skip reason strings into canonical SKIP_* taxonomy.
    Pure function — no side effects.
    """

    # Keyword → canonical reason mapping (order matters: first match wins)
    _RULES: List[Tuple[str, str]] = [
        ("capital",       SKIP_CAPITAL_CONSTRAINT),
        ("cash",          SKIP_CAPITAL_CONSTRAINT),
        ("alloc_cap",     SKIP_CAPITAL_CONSTRAINT),
        ("insufficient",  SKIP_CAPITAL_CONSTRAINT),
        ("heat",          SKIP_PORTFOLIO_HEAT),
        ("drawdown",      SKIP_PORTFOLIO_HEAT),
        ("dd_limit",      SKIP_PORTFOLIO_HEAT),
        ("sector",        SKIP_SECTOR_EXPOSURE),
        ("concentration", SKIP_SECTOR_EXPOSURE),
        ("liquidity",     SKIP_LIQUIDITY_FILTER),
        ("volume",        SKIP_LIQUIDITY_FILTER),
        ("spread",        SKIP_LIQUIDITY_FILTER),
        ("volatility",    SKIP_VOLATILITY_FILTER),
        ("atr",           SKIP_VOLATILITY_FILTER),
        ("vola",          SKIP_VOLATILITY_FILTER),
        ("universe",      SKIP_UNIVERSE_FILTER),
        ("epoch",         SKIP_UNIVERSE_FILTER),
        ("manifest",      SKIP_UNIVERSE_FILTER),
        ("duplicate",     SKIP_DUPLICATE_SIGNAL),
        ("dup",           SKIP_DUPLICATE_SIGNAL),
        ("already",       SKIP_DUPLICATE_SIGNAL),
        ("cooldown",      SKIP_COOLDOWN),
        ("rate_limit",    SKIP_COOLDOWN),
        ("throttle",      SKIP_COOLDOWN),
        ("execution",     SKIP_EXECUTION_FAILURE),
        ("broker",        SKIP_EXECUTION_FAILURE),
        ("order_fail",    SKIP_EXECUTION_FAILURE),
        ("manual",        SKIP_MANUAL_OVERRIDE),
        ("override",      SKIP_MANUAL_OVERRIDE),
    ]

    def classify(self, raw_reason: Optional[str]) -> str:
        if raw_reason is None:
            return SKIP_UNKNOWN
        lower = raw_reason.lower()
        for keyword, canonical in self._RULES:
            if keyword in lower:
                return canonical
        return SKIP_UNKNOWN

    def classify_record(self, record: TradeOpportunityRecord) -> str:
        if record.executed:
            return ""
        if record.skip_reason in ALL_SKIP_REASONS:
            return record.skip_reason
        return self.classify(record.skip_reason)


_classifier = SkipReasonClassifier()


# ─────────────────────────────────────────────────────────────────────────────
# 3. ExpectancyAnalyticsEngine
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExpectancySlice:
    """Summary statistics for a group of opportunities."""
    label: str
    n_executed: int
    n_skipped: int
    executed_expectancy_5d: Optional[float]
    skipped_expectancy_5d: Optional[float]
    expectancy_delta_5d: Optional[float]   # skipped − executed (positive = missed alpha)
    executed_expectancy_20d: Optional[float]
    skipped_expectancy_20d: Optional[float]
    expectancy_delta_20d: Optional[float]
    executed_win_rate: Optional[float]
    skipped_win_rate: Optional[float]
    opportunity_capture_ratio: Optional[float]  # n_executed / (n_executed + n_skipped)
    skipped_convexity_score: Optional[float]    # mean(max(0, skipped_5d)) / max(1e-9, abs(mean(skipped_5d)))


def _expectancy_slice(
    label: str,
    executed: List[TradeOpportunityRecord],
    skipped: List[TradeOpportunityRecord],
) -> ExpectancySlice:
    ex5  = [r.forward_5d_return  for r in executed  if r.forward_5d_return  is not None]
    ex20 = [r.forward_20d_return for r in executed  if r.forward_20d_return is not None]
    sk5  = [r.forward_5d_return  for r in skipped   if r.forward_5d_return  is not None]
    sk20 = [r.forward_20d_return for r in skipped   if r.forward_20d_return is not None]

    ex_exp5  = _safe_mean(ex5)
    sk_exp5  = _safe_mean(sk5)
    ex_exp20 = _safe_mean(ex20)
    sk_exp20 = _safe_mean(sk20)

    delta5  = round(sk_exp5  - ex_exp5,  6) if (sk_exp5  is not None and ex_exp5  is not None) else None
    delta20 = round(sk_exp20 - ex_exp20, 6) if (sk_exp20 is not None and ex_exp20 is not None) else None

    total = len(executed) + len(skipped)
    capture_ratio = round(len(executed) / total, 4) if total > 0 else None

    convexity: Optional[float] = None
    if sk5:
        upside = [max(0.0, v) for v in sk5]
        mean_up = sum(upside) / len(upside)
        mean_abs = abs(_safe_mean(sk5) or 0.0) or 1e-9
        convexity = round(mean_up / mean_abs, 4)

    return ExpectancySlice(
        label=label,
        n_executed=len(executed),
        n_skipped=len(skipped),
        executed_expectancy_5d=ex_exp5,
        skipped_expectancy_5d=sk_exp5,
        expectancy_delta_5d=delta5,
        executed_expectancy_20d=ex_exp20,
        skipped_expectancy_20d=sk_exp20,
        expectancy_delta_20d=delta20,
        executed_win_rate=_win_rate(ex5),
        skipped_win_rate=_win_rate(sk5),
        opportunity_capture_ratio=capture_ratio,
        skipped_convexity_score=convexity,
    )


@dataclass
class ExpectancyReport:
    """Full expectancy analytics output."""
    generated_at: str
    n_total: int
    overall: ExpectancySlice
    by_skip_reason: Dict[str, ExpectancySlice]
    by_regime: Dict[str, ExpectancySlice]
    by_sector: Dict[str, ExpectancySlice]
    by_atr_bucket: Dict[str, ExpectancySlice]
    by_slot_utilization: Dict[str, ExpectancySlice]


class ExpectancyAnalyticsEngine:
    """
    Pure analytics engine. Computes expectancy slices across dimensions.
    No side effects. No strategy changes.
    """

    def analyze(
        self,
        records: List[TradeOpportunityRecord],
    ) -> ExpectancyReport:
        enriched = [r for r in records if r.enrichment_status == "enriched"]
        executed = [r for r in enriched if r.executed]
        skipped  = [r for r in enriched if not r.executed]

        overall = _expectancy_slice("overall", executed, skipped)

        by_skip_reason = self._slice_by(
            executed, skipped,
            key=lambda r: (_classifier.classify_record(r) if not r.executed else "__executed__"),
            executed_label="__executed__",
        )

        by_regime = self._slice_dimension(executed, skipped, key=lambda r: r.market_regime)
        by_sector = self._slice_dimension(executed, skipped, key=lambda r: r.sector)
        by_atr    = self._slice_dimension(executed, skipped, key=lambda r: _atr_bucket_label(r.atr_pct))
        by_slot   = self._slice_dimension(executed, skipped, key=lambda r: _slot_bucket_label(r.slot_utilization))

        return ExpectancyReport(
            generated_at=datetime.now(timezone.utc).isoformat(),
            n_total=len(records),
            overall=overall,
            by_skip_reason=by_skip_reason,
            by_regime=by_regime,
            by_sector=by_sector,
            by_atr_bucket=by_atr,
            by_slot_utilization=by_slot,
        )

    def _slice_dimension(
        self,
        executed: List[TradeOpportunityRecord],
        skipped: List[TradeOpportunityRecord],
        key,
    ) -> Dict[str, ExpectancySlice]:
        labels: set = set()
        for r in executed + skipped:
            labels.add(key(r))
        result = {}
        for label in sorted(labels):
            ex_group = [r for r in executed if key(r) == label]
            sk_group = [r for r in skipped  if key(r) == label]
            result[label] = _expectancy_slice(label, ex_group, sk_group)
        return result

    def _slice_by(
        self,
        executed: List[TradeOpportunityRecord],
        skipped: List[TradeOpportunityRecord],
        key,
        executed_label: str,
    ) -> Dict[str, ExpectancySlice]:
        groups: Dict[str, Tuple[List, List]] = {}
        for r in executed:
            lbl = executed_label
            if lbl not in groups:
                groups[lbl] = ([], [])
            groups[lbl][0].append(r)
        for r in skipped:
            lbl = key(r)
            if lbl not in groups:
                groups[lbl] = ([], [])
            groups[lbl][1].append(r)
        result = {}
        for lbl, (ex, sk) in sorted(groups.items()):
            result[lbl] = _expectancy_slice(lbl, ex, sk)
        return result


def _atr_bucket_label(atr_pct: float) -> str:
    for label, lo, hi in ATR_BUCKETS:
        if lo <= atr_pct < hi:
            return label
    return "4%+"


def _slot_bucket_label(slot: float) -> str:
    for label, lo, hi in SLOT_BUCKETS:
        if lo <= slot < hi:
            return label
    return "67-100%"


# ─────────────────────────────────────────────────────────────────────────────
# 4. CounterfactualPortfolioAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CounterfactualSummary:
    """Counterfactual portfolio impact summary."""
    n_skipped_with_attribution: int
    mean_skipped_5d_return: Optional[float]
    mean_executed_5d_return: Optional[float]
    estimated_missed_alpha_5d: Optional[float]   # mean(skipped_5d) − mean(executed_5d)
    mean_skipped_20d_return: Optional[float]
    mean_executed_20d_return: Optional[float]
    estimated_missed_alpha_20d: Optional[float]
    persistent_skip_bias: Dict[str, float]       # skip_reason → mean_5d_return (positive = leak)
    systematic_under_allocation: List[str]       # labels where skipped ≫ executed expectancy
    top_missed_sectors: List[str]
    top_missed_regimes: List[str]


class CounterfactualPortfolioAnalyzer:
    """
    Estimates counterfactual portfolio impact of skipped opportunities.
    Observational only — no trading changes.
    """

    def analyze(
        self,
        records: List[TradeOpportunityRecord],
        engine: Optional[ExpectancyAnalyticsEngine] = None,
    ) -> CounterfactualSummary:
        if engine is None:
            engine = ExpectancyAnalyticsEngine()

        enriched = [r for r in records if r.enrichment_status == "enriched"]
        executed = [r for r in enriched if r.executed]
        skipped  = [r for r in enriched if not r.executed]

        sk5  = [r.forward_5d_return  for r in skipped if r.forward_5d_return  is not None]
        ex5  = [r.forward_5d_return  for r in executed if r.forward_5d_return  is not None]
        sk20 = [r.forward_20d_return for r in skipped if r.forward_20d_return is not None]
        ex20 = [r.forward_20d_return for r in executed if r.forward_20d_return is not None]

        m_sk5  = _safe_mean(sk5)
        m_ex5  = _safe_mean(ex5)
        m_sk20 = _safe_mean(sk20)
        m_ex20 = _safe_mean(ex20)

        missed5  = round(m_sk5  - m_ex5,  6) if (m_sk5  is not None and m_ex5  is not None) else None
        missed20 = round(m_sk20 - m_ex20, 6) if (m_sk20 is not None and m_ex20 is not None) else None

        # Persistent skip bias per reason
        bias: Dict[str, float] = {}
        for reason in ALL_SKIP_REASONS:
            group = [r for r in skipped if _classifier.classify_record(r) == reason]
            vals  = [r.forward_5d_return for r in group if r.forward_5d_return is not None]
            if vals:
                bias[reason] = round(sum(vals) / len(vals), 6)

        # Systematic under-allocation: skip_reason groups where missed5 > 0
        under_alloc = [r for r, v in bias.items() if v > 0]

        # Top missed sectors
        sector_returns: Dict[str, List[float]] = {}
        for r in skipped:
            if r.forward_5d_return is not None:
                sector_returns.setdefault(r.sector, []).append(r.forward_5d_return)
        top_sectors = sorted(
            sector_returns.keys(),
            key=lambda s: sum(sector_returns[s]) / len(sector_returns[s]),
            reverse=True,
        )[:5]

        # Top missed regimes
        regime_returns: Dict[str, List[float]] = {}
        for r in skipped:
            if r.forward_5d_return is not None:
                regime_returns.setdefault(r.market_regime, []).append(r.forward_5d_return)
        top_regimes = sorted(
            regime_returns.keys(),
            key=lambda rg: sum(regime_returns[rg]) / len(regime_returns[rg]),
            reverse=True,
        )[:5]

        return CounterfactualSummary(
            n_skipped_with_attribution=len(sk5),
            mean_skipped_5d_return=m_sk5,
            mean_executed_5d_return=m_ex5,
            estimated_missed_alpha_5d=missed5,
            mean_skipped_20d_return=m_sk20,
            mean_executed_20d_return=m_ex20,
            estimated_missed_alpha_20d=missed20,
            persistent_skip_bias=bias,
            systematic_under_allocation=sorted(under_alloc),
            top_missed_sectors=top_sectors,
            top_missed_regimes=top_regimes,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 5. IntegrityValidator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntegrityViolation:
    violation_type: str
    opportunity_id: str
    detail: str


@dataclass
class IntegrityReport:
    n_records: int
    n_violations: int
    violations: List[IntegrityViolation]
    passed: bool


class IntegrityValidator:
    """
    Validates a batch of TradeOpportunityRecord for structural integrity.
    Fail-closed: any CRITICAL violation sets passed=False.
    """

    CRITICAL_TYPES = {
        "duplicate_opportunity_id",
        "missing_executed_status",
        "future_leakage",
    }

    def validate(
        self,
        records: List[TradeOpportunityRecord],
        as_of_date: Optional[str] = None,
    ) -> IntegrityReport:
        violations: List[IntegrityViolation] = []
        today_str = as_of_date or date.today().isoformat()

        seen_ids: Dict[str, int] = {}
        for i, r in enumerate(records):
            # Duplicate opportunity_id
            if r.opportunity_id in seen_ids:
                violations.append(IntegrityViolation(
                    violation_type="duplicate_opportunity_id",
                    opportunity_id=r.opportunity_id,
                    detail=f"index={i} duplicates index={seen_ids[r.opportunity_id]}",
                ))
            else:
                seen_ids[r.opportunity_id] = i

            # Missing executed status — verify field is present and a bool
            if not isinstance(r.executed, bool):
                violations.append(IntegrityViolation(
                    violation_type="missing_executed_status",
                    opportunity_id=r.opportunity_id,
                    detail=f"executed={r.executed!r} is not bool",
                ))

            # Forward attribution existence — enriched records must have attribution
            if r.enrichment_status == "enriched":
                if r.forward_5d_return is None and r.forward_20d_return is None:
                    violations.append(IntegrityViolation(
                        violation_type="missing_forward_attribution",
                        opportunity_id=r.opportunity_id,
                        detail="enrichment_status=enriched but both forward returns are None",
                    ))

            # Future leakage: eval_date is after as_of_date
            if r.eval_date > today_str:
                violations.append(IntegrityViolation(
                    violation_type="future_leakage",
                    opportunity_id=r.opportunity_id,
                    detail=f"eval_date={r.eval_date} > as_of_date={today_str}",
                ))

            # Lifecycle linkage consistency
            if r.executed and not r.position_lifecycle_available:
                violations.append(IntegrityViolation(
                    violation_type="lifecycle_linkage_missing",
                    opportunity_id=r.opportunity_id,
                    detail="executed=True but position_lifecycle_available=False",
                ))

        has_critical = any(v.violation_type in self.CRITICAL_TYPES for v in violations)
        return IntegrityReport(
            n_records=len(records),
            n_violations=len(violations),
            violations=violations,
            passed=not has_critical,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_STORE_PATH = Path("runtime/analytics/executed_vs_skipped.jsonl")


def append_opportunity(record: TradeOpportunityRecord, path: Path = DEFAULT_STORE_PATH) -> None:
    """
    Atomic append to JSONL store. fsync durability. Never raises — analytics continues on error.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True)
        # Atomic write via temp file in same directory
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as tf:
                # Read existing content
                existing = path.read_text(encoding="utf-8") if path.exists() else ""
                tf.write(existing)
                tf.write(line + "\n")
                tf.flush()
                os.fsync(tf.fileno())
        except Exception:
            os.unlink(tmp_path)
            raise
        os.replace(tmp_path, str(path))
    except Exception as exc:
        logger.warning("[EVS] append failed (%s) — analytics continues", exc)


def load_opportunities(path: Path = DEFAULT_STORE_PATH) -> List[TradeOpportunityRecord]:
    """Load all records from JSONL. Corrupted lines are skipped with a warning."""
    if not path.exists():
        return []
    records: List[TradeOpportunityRecord] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(TradeOpportunityRecord.from_dict(d))
        except Exception as exc:
            logger.warning("[EVS] parse error at line %d: %s", i + 1, exc)
    return records


def deduplicate_opportunities(
    records: List[TradeOpportunityRecord],
) -> List[TradeOpportunityRecord]:
    """Deduplicate by opportunity_id. Prefer enriched over pending. Stable sort by eval_date."""
    seen: Dict[str, TradeOpportunityRecord] = {}
    for r in records:
        existing = seen.get(r.opportunity_id)
        if existing is None:
            seen[r.opportunity_id] = r
        elif r.enrichment_status == "enriched" and existing.enrichment_status != "enriched":
            seen[r.opportunity_id] = r
    return sorted(seen.values(), key=lambda r: (r.eval_date, r.symbol))


def enrich_forward_returns(
    records: List[TradeOpportunityRecord],
    forward_returns: Dict[str, Dict[str, Optional[float]]],
) -> List[TradeOpportunityRecord]:
    """
    Pure enrichment function. Fills forward returns from lookup table.

    Args:
        records:        List of TradeOpportunityRecord.
        forward_returns: {symbol: {"5d": float|None, "20d": float|None}}

    Returns:
        New list with pending records enriched where data available.
    """
    import dataclasses as _dc
    result = []
    for r in records:
        if r.enrichment_status != "pending":
            result.append(r)
            continue
        sym_data = forward_returns.get(r.symbol, {})
        f5  = _opt_float(sym_data.get("5d"))
        f20 = _opt_float(sym_data.get("20d"))
        if f5 is not None or f20 is not None:
            result.append(_dc.replace(
                r,
                forward_5d_return=f5,
                forward_20d_return=f20,
                enrichment_status="enriched",
            ))
        else:
            result.append(r)
    return result
