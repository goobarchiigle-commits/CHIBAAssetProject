"""
skipped_opportunity_analytics.py — Deployable-alpha leakage opportunity tracking

Captures every skipped or suppressed signal with enough context to:
  1. Attribute the skip reason deterministically (record_id dedup)
  2. Compute counterfactual returns via forward-return enrichment
  3. Estimate missed PnL

Rejection reasons tracked:
  REJECTION_ALLOC_CAP            — effective-N / concentration cap zeroed qty
  REJECTION_HIGH_PRICE           — single unit cost exceeds capital allocation limit
  REJECTION_SECTOR_SUPPRESSION   — sector filter blocked the signal
  REJECTION_CONCENTRATION_THROTTLE — portfolio concentration throttle
  REJECTION_INSUFFICIENT_CASH    — not enough available cash
  REJECTION_STALE_SIGNAL         — stale signal / data freshness gate
  REJECTION_TRUTH_CONFIDENCE     — truth-layer confidence gate (< 0.70)
  REJECTION_RATE_LIMIT           — daily order rate limit
  REJECTION_DUPLICATE            — duplicate order (already ordered today)
  REJECTION_EPOCH_BLOCKED        — epoch governance / authority manifest blocked

Enrichment lifecycle:
  ENRICHMENT_PENDING  — forward returns not yet computed
  ENRICHMENT_ENRICHED — forward returns filled in (pure function, deterministic)
  ENRICHMENT_EXPIRED  — enrichment window passed without price data

Design:
  - Append-only JSONL (original records never mutated)
  - record_id = sha256(run_id + symbol + rejection_reason + timestamp) — dedup key
  - Forward returns are None at write time; filled by enrich_forward_returns()
  - Analytics-only: no broker mutations, no execution-path writes
  - Windows compatible (no symlinks, no POSIX paths)
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 2

# ─────────────────────────────────────────────────────────────────────────────
# Rejection reasons
# ─────────────────────────────────────────────────────────────────────────────

REJECTION_ALLOC_CAP               = "alloc_cap"
REJECTION_HIGH_PRICE              = "high_price"
REJECTION_SECTOR_SUPPRESSION      = "sector_suppression"
REJECTION_CONCENTRATION_THROTTLE  = "concentration_throttle"
REJECTION_INSUFFICIENT_CASH       = "insufficient_cash"
REJECTION_STALE_SIGNAL            = "stale_signal"
REJECTION_TRUTH_CONFIDENCE        = "truth_confidence"
REJECTION_RATE_LIMIT              = "rate_limit"
REJECTION_DUPLICATE               = "duplicate"
REJECTION_EPOCH_BLOCKED           = "epoch_blocked"
# SCHEMA_VERSION 2 (2026-07-08 Opportunity Capture integrity fix): the only
# writer of this file was the Phase5A exposure-aware-sizing block in
# run_live_signal.py, reachable ONLY in LIVE mode AND only for candidates
# that already survived SignalBridge._build_orders() in full. The dominant
# real-world rejection causes (position capacity, allocation cap inside
# _build_orders(), sector concentration, pre_trade_risk_check, top_k ranking
# cutoff) never reached this writer at all, which is why
# runtime/analytics/skipped_opportunities.jsonl never got created. These
# reasons mirror src.analytics.executed_vs_skipped_expectancy.SKIP_* so both
# stores describe the same underlying stage_audit taxonomy.
REJECTION_POSITION_FULL           = "position_full"
REJECTION_RANKING_CUTOFF          = "ranking_cutoff"
REJECTION_SIZING_ZERO             = "sizing_zero_qty"
REJECTION_RISK_CHECK              = "risk_check_reject"
REJECTION_DAILY_LIMIT             = "daily_new_buy_limit"

# stage (src.analytics.live_stage_audit の stage名) → REJECTION_* taxonomy。
STAGE_TO_REJECTION_REASON: Dict[str, str] = {
    "RANKING":              REJECTION_RANKING_CUTOFF,
    "CAPACITY":             REJECTION_POSITION_FULL,
    "DAILY_LIMIT":          REJECTION_DAILY_LIMIT,
    "CAPITAL":              REJECTION_HIGH_PRICE,
    "SIZING":               REJECTION_SIZING_ZERO,
    "SECTOR_CONCENTRATION": REJECTION_SECTOR_SUPPRESSION,
    "RISK":                 REJECTION_RISK_CHECK,
    "ORDER_SEND_FAILED":    REJECTION_RATE_LIMIT,  # send-time failure; closest existing bucket
}

ENRICHMENT_PENDING  = "pending"
ENRICHMENT_ENRICHED = "enriched"
ENRICHMENT_EXPIRED  = "expired"

ENRICHMENT_LOOKBACK_DAYS = 7   # records older than this are marked expired if still pending

_FORWARD_RETURN_DAYS = (1, 3, 5)


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_json(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# SkippedOpportunityRecord
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SkippedOpportunityRecord:
    """
    Full attribution record for a single skipped signal.
    Written once at skip time; forward returns populated by enricher.
    """
    record_id: str               # sha256 dedup key
    run_id: str
    timestamp: str               # ISO-8601 UTC
    symbol: str
    strategy_id: str
    signal_strength: float       # RSR rank or equivalent (higher = stronger)
    predicted_rank: int          # rank in universe (1 = best)
    rejection_reason: str        # one of REJECTION_* constants
    available_cash: float        # cash available at skip time (JPY)
    alloc_cap: int               # effective allocation cap (0 = fully blocked)
    intended_position_size: int  # shares that would have been purchased
    sector_state: str            # sector name (e.g., "輸送用機器")
    concentration_state: float   # portfolio concentration proxy at skip time
    price_at_signal: float       # estimated entry price at signal time (JPY)
    # Forward return fields — None until enriched
    forward_return_1d: Optional[float] = None
    forward_return_3d: Optional[float] = None
    forward_return_5d: Optional[float] = None
    max_favorable_excursion: Optional[float] = None   # max return in [t+1, t+5]
    max_adverse_excursion: Optional[float] = None     # min return in [t+1, t+5]
    hypothetical_return: Optional[float] = None       # forward_return_5d or best available
    estimated_missed_pnl: Optional[float] = None      # intended_size × price × hypothetical_return
    enrichment_status: str = ENRICHMENT_PENDING
    schema_version: int = SCHEMA_VERSION
    # SCHEMA_VERSION 2 fields — shared run_id/mode/source_script with
    # executed_vs_skipped_expectancy.TradeOpportunityRecord and
    # live_stage_audit so all three stores line up on the same run.
    mode: str = "unknown"          # "DRY" | "LIVE"
    final_stage: str = ""          # stage name from live_stage_audit
    source_script: str = ""        # "run_live_signal.py" | "run_morning_signal.py"

    @staticmethod
    def create(
        run_id: str,
        timestamp: str,
        symbol: str,
        strategy_id: str,
        signal_strength: float,
        predicted_rank: int,
        rejection_reason: str,
        available_cash: float,
        alloc_cap: int,
        intended_position_size: int,
        sector_state: str,
        concentration_state: float,
        price_at_signal: float,
        mode: str = "unknown",
        final_stage: str = "",
        source_script: str = "",
    ) -> "SkippedOpportunityRecord":
        record_id = _sha256(_canonical_json({
            "rejection_reason": rejection_reason,
            "run_id": run_id,
            "symbol": symbol,
            "timestamp": timestamp,
        }))
        return SkippedOpportunityRecord(
            record_id=record_id,
            run_id=run_id,
            timestamp=timestamp,
            symbol=symbol,
            strategy_id=strategy_id,
            signal_strength=signal_strength,
            predicted_rank=predicted_rank,
            rejection_reason=rejection_reason,
            available_cash=available_cash,
            alloc_cap=alloc_cap,
            intended_position_size=intended_position_size,
            sector_state=sector_state,
            concentration_state=concentration_state,
            price_at_signal=price_at_signal,
            mode=mode,
            final_stage=final_stage,
            source_script=source_script,
        )


# ─────────────────────────────────────────────────────────────────────────────
# 1b. build_skipped_opportunity_records — the ONE correct way to build these
# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA_VERSION 2 (2026-07-08 Opportunity Capture integrity fix). Derives
# SkippedOpportunityRecord for EVERY skipped BUY candidate — not just the
# narrow Phase5A exposure-layer subset the old sole call site covered.
# Uses the SAME (signals, stage_audit, run_id, run_timestamp, mode,
# source_script) inputs as
# executed_vs_skipped_expectancy.build_opportunity_records(), so both stores
# — plus src.analytics.live_stage_audit — describe the identical run and can
# be joined on run_id (Phase3 requirement).

def build_skipped_opportunity_records(
    signals: List[dict],
    stage_audit: List[dict],
    send_results: List[dict],
    run_id: str,
    run_timestamp: str,
    mode: str,
    source_script: str,
    available_cash: float,
    strategy_id: str = "fujiko_v2",
) -> List["SkippedOpportunityRecord"]:
    from src.analytics.executed_vs_skipped_expectancy import build_opportunity_records

    opp_records = build_opportunity_records(
        signals=signals,
        stage_audit=stage_audit,
        send_results=send_results,
        run_id=run_id,
        run_timestamp=run_timestamp,
        mode=mode,
        source_script=source_script,
        capital_available_pct=0.0,
        portfolio_heat=0.0,
        market_regime="unknown",
        max_positions=3,
    )
    # StockSignal dicts don't carry price/qty directly; recover them from the
    # matching signal entry (entry_price is 0 for non-held candidates in this
    # codebase's schema, so use distance_25ma_pct-adjacent fields where
    # present; otherwise leave 0 — never fabricate a price).
    sig_by_symbol = {str(s.get("symbol", "")): s for s in signals}

    records: List[SkippedOpportunityRecord] = []
    for opp in opp_records:
        if opp.executed:
            continue
        sig = sig_by_symbol.get(opp.symbol, {})
        rejection_reason = STAGE_TO_REJECTION_REASON.get(opp.final_stage, REJECTION_STALE_SIGNAL)
        rec = SkippedOpportunityRecord.create(
            run_id=run_id,
            timestamp=run_timestamp,
            symbol=opp.symbol,
            strategy_id=strategy_id,
            signal_strength=opp.entry_score * 100.0,
            predicted_rank=opp.rs_rank,
            rejection_reason=rejection_reason,
            available_cash=available_cash,
            alloc_cap=0,
            intended_position_size=0,
            sector_state=opp.sector,
            concentration_state=0.0,
            price_at_signal=float(sig.get("entry_price") or 0.0),
            mode=mode,
            final_stage=opp.final_stage,
            source_script=source_script,
        )
        records.append(rec)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Forward return enrichment (pure function — deterministic, replay-safe)
# ─────────────────────────────────────────────────────────────────────────────

def enrich_forward_returns(
    records: List[SkippedOpportunityRecord],
    price_history: Dict[str, List[float]],
    now_ts: Optional[str] = None,
) -> List[SkippedOpportunityRecord]:
    """
    Fill in forward returns for pending records. Pure function — no side effects.

    Args:
        records:       List of SkippedOpportunityRecord (any enrichment_status).
        price_history: symbol → list of daily closing prices [t0, t+1, t+2, t+3, t+4, t+5]
                       where t0 is the price at signal. At least 6 values required for
                       full enrichment; partial enrichment uses available values.
        now_ts:        ISO-8601 timestamp for enriched_at (defaults to now).

    Returns:
        New list of records (pending ones are enriched; others are unchanged).
    """
    if not now_ts:
        now_ts = datetime.now(timezone.utc).isoformat()

    result: List[SkippedOpportunityRecord] = []
    for rec in records:
        if rec.enrichment_status != ENRICHMENT_PENDING:
            result.append(rec)
            continue

        prices = price_history.get(rec.symbol)
        if not prices or len(prices) < 2:
            # Check if expired
            enrichment_status = _check_expiry(rec, now_ts)
            result.append(_replace_enrichment_status(rec, enrichment_status))
            continue

        p0 = rec.price_at_signal if rec.price_at_signal > 0 else (prices[0] if prices[0] > 0 else None)
        if p0 is None or p0 <= 0:
            result.append(rec)
            continue

        def _ret(idx: int) -> Optional[float]:
            if idx < len(prices) and prices[idx] is not None and prices[idx] > 0:
                return round((prices[idx] - p0) / p0, 6)
            return None

        fr1 = _ret(1)
        fr3 = _ret(3)
        fr5 = _ret(5)

        window_returns = [_ret(i) for i in range(1, min(6, len(prices)))]
        valid = [r for r in window_returns if r is not None]
        mfe = max(valid) if valid else None
        mae = min(valid) if valid else None

        hyp = fr5 if fr5 is not None else fr3 if fr3 is not None else fr1
        missed_pnl: Optional[float] = None
        if hyp is not None and rec.price_at_signal > 0 and rec.intended_position_size > 0:
            missed_pnl = round(rec.intended_position_size * rec.price_at_signal * hyp, 2)

        import dataclasses
        enriched = dataclasses.replace(
            rec,
            forward_return_1d=fr1,
            forward_return_3d=fr3,
            forward_return_5d=fr5,
            max_favorable_excursion=mfe,
            max_adverse_excursion=mae,
            hypothetical_return=hyp,
            estimated_missed_pnl=missed_pnl,
            enrichment_status=ENRICHMENT_ENRICHED,
        )
        result.append(enriched)
    return result


def _check_expiry(rec: SkippedOpportunityRecord, now_ts: str) -> str:
    try:
        rec_dt = datetime.fromisoformat(rec.timestamp)
        now_dt = datetime.fromisoformat(now_ts)
        if rec_dt.tzinfo is None:
            rec_dt = rec_dt.replace(tzinfo=timezone.utc)
        if now_dt.tzinfo is None:
            now_dt = now_dt.replace(tzinfo=timezone.utc)
        age_days = (now_dt - rec_dt).days
        if age_days > ENRICHMENT_LOOKBACK_DAYS:
            return ENRICHMENT_EXPIRED
    except Exception:
        pass
    return ENRICHMENT_PENDING


def _replace_enrichment_status(rec: SkippedOpportunityRecord, status: str) -> SkippedOpportunityRecord:
    import dataclasses
    return dataclasses.replace(rec, enrichment_status=status)


# ─────────────────────────────────────────────────────────────────────────────
# Deduplication
# ─────────────────────────────────────────────────────────────────────────────

def deduplicate_records(
    records: List[SkippedOpportunityRecord],
) -> List[SkippedOpportunityRecord]:
    """
    Deduplicate by record_id. For duplicates, prefer enriched over pending.
    Deterministic: stable output order.
    """
    seen: Dict[str, SkippedOpportunityRecord] = {}
    for rec in records:
        existing = seen.get(rec.record_id)
        if existing is None:
            seen[rec.record_id] = rec
        elif rec.enrichment_status == ENRICHMENT_ENRICHED and existing.enrichment_status != ENRICHMENT_ENRICHED:
            seen[rec.record_id] = rec
    # Sort by timestamp for deterministic ordering
    return sorted(seen.values(), key=lambda r: r.timestamp)


# ─────────────────────────────────────────────────────────────────────────────
# Enrichment pipeline entry point (2026-07-08 Opportunity Capture fix)
# ─────────────────────────────────────────────────────────────────────────────
# enrich_forward_returns() above is a pure function and was never actually
# invoked anywhere in the live pipeline (confirmed by grep — same class of
# gap as executed_vs_skipped_expectancy's own enrichment, and the reason
# every record ever written stayed enrichment_status="pending" forever, and
# analyze_opportunity_capture() could never compute missed_alpha_score /
# forward_return from real data). This wraps load → enrich → atomic rewrite
# into one call so run_live_signal.py / run_morning_signal.py only need a
# price_fetcher callback (usually backed by the OHLCV parquet cache).

def enrich_and_rewrite_store(
    path: Path,
    price_fetcher,  # Callable[[str, str], Optional[List[float]]] — symbol, iso_date -> [t0..t+5] closes
    now_ts: Optional[str] = None,
) -> int:
    """
    Loads all records, enriches pending ones via price_fetcher, deduplicates
    (preferring enriched), and atomically rewrites the store.
    Returns the number of records newly enriched this call. Never raises.
    """
    try:
        records = load_skipped_opportunities(path)
        if not records:
            return 0
        price_history: Dict[str, List[float]] = {}
        for rec in records:
            if rec.enrichment_status != ENRICHMENT_PENDING or rec.symbol in price_history:
                continue
            prices = price_fetcher(rec.symbol, rec.timestamp[:10])
            if prices:
                price_history[rec.symbol] = prices

        enriched = enrich_forward_returns(records, price_history, now_ts=now_ts)
        final = deduplicate_records(enriched)

        tmp = path.with_suffix(".tmp")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("w", encoding="utf-8") as f:
            for rec in final:
                f.write(json.dumps(asdict(rec), ensure_ascii=False, sort_keys=True) + "\n")
        tmp.replace(path)

        before_ids = {r.record_id for r in records if r.enrichment_status == ENRICHMENT_ENRICHED}
        after_ids = {r.record_id for r in final if r.enrichment_status == ENRICHMENT_ENRICHED}
        return len(after_ids - before_ids)
    except Exception as exc:
        logger.warning("[SKIPPED_OPP] enrich_and_rewrite_store failed (%s) — analytics continues", exc)
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

def append_skipped_opportunity(record: SkippedOpportunityRecord, path: Path) -> None:
    """Fail-safe append. Never raises. Analytics-only: no execution-path side effects."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as exc:
        logger.warning("[SKIPPED_OPP] write failed (%s) — analytics continues", exc)


def load_skipped_opportunities(path: Path) -> List[SkippedOpportunityRecord]:
    """Load all records. Corrupted lines are skipped with a warning."""
    if not path.exists():
        return []
    result: List[SkippedOpportunityRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            result.append(SkippedOpportunityRecord(
                record_id=d.get("record_id", ""),
                run_id=d.get("run_id", ""),
                timestamp=d.get("timestamp", ""),
                symbol=d.get("symbol", ""),
                strategy_id=d.get("strategy_id", ""),
                signal_strength=float(d.get("signal_strength", 0.0)),
                predicted_rank=int(d.get("predicted_rank", 0)),
                rejection_reason=d.get("rejection_reason", ""),
                available_cash=float(d.get("available_cash", 0.0)),
                alloc_cap=int(d.get("alloc_cap", 0)),
                intended_position_size=int(d.get("intended_position_size", 0)),
                sector_state=d.get("sector_state", ""),
                concentration_state=float(d.get("concentration_state", 0.0)),
                price_at_signal=float(d.get("price_at_signal", 0.0)),
                forward_return_1d=_opt_float(d.get("forward_return_1d")),
                forward_return_3d=_opt_float(d.get("forward_return_3d")),
                forward_return_5d=_opt_float(d.get("forward_return_5d")),
                max_favorable_excursion=_opt_float(d.get("max_favorable_excursion")),
                max_adverse_excursion=_opt_float(d.get("max_adverse_excursion")),
                hypothetical_return=_opt_float(d.get("hypothetical_return")),
                estimated_missed_pnl=_opt_float(d.get("estimated_missed_pnl")),
                enrichment_status=d.get("enrichment_status", ENRICHMENT_PENDING),
                schema_version=int(d.get("schema_version", SCHEMA_VERSION)),
                mode=d.get("mode", "unknown"),
                final_stage=d.get("final_stage", ""),
                source_script=d.get("source_script", ""),
            ))
        except Exception as exc:
            logger.warning("[SKIPPED_OPP] parse error: %s", exc)
    return result


def load_skipped_opportunities_for_date(path: Path, date_str: str) -> List[SkippedOpportunityRecord]:
    """Load records for a specific date (YYYY-MM-DD prefix match on timestamp)."""
    return [r for r in load_skipped_opportunities(path) if r.timestamp.startswith(date_str)]


def _opt_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
