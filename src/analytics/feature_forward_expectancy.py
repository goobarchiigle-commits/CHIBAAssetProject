"""
src/analytics/feature_forward_expectancy.py — Feature-to-forward-return attribution

Answers:
  - Are high-ATR entries actually dangerous?
  - Which entry extensions still trend after entry?
  - Which regimes destroy breakout expectancy?
  - Are skipped trades outperforming executed trades?
  - Where does unrealized alpha decay?
  - Is sector concentration silently increasing?

Design:
  - deterministic statistical attribution only (no prediction, no auto-tuning)
  - append-only JSONL snapshot store
  - forward returns via yfinance (fail-open if unavailable)
  - results cached per-symbol per-date (avoid repeated downloads)
  - all timestamps JST-aware
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
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))
SCHEMA_VERSION: int = 1

# ── ATR bucket boundaries (atr_pct = atr20/price) ────────────────────────────
ATR_BUCKETS: List[Tuple[str, float, float]] = [
    ("<1.5%",   0.000, 0.015),
    ("1.5-2.5%", 0.015, 0.025),
    ("2.5-4%",  0.025, 0.040),
    ("4%+",     0.040, 9.999),
]

# ── Extension bucket definitions ──────────────────────────────────────────────
DIST_25MA_BUCKETS: List[Tuple[str, float, float]] = [
    ("<2%",   0.00, 0.02), ("2-5%",  0.02, 0.05),
    ("5-10%", 0.05, 0.10), ("10-15%", 0.10, 0.15), ("15%+", 0.15, 9.99),
]
RSI_BUCKETS: List[Tuple[str, float, float]] = [
    ("<40",  0, 40), ("40-50", 40, 50), ("50-60", 50, 60),
    ("60-70", 60, 70), ("70-80", 70, 80), ("80+", 80, 100),
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EntryFeatureSnapshot:
    """Immutable entry feature snapshot persisted at every BUY signal evaluation."""
    schema_version: int
    snapshot_id: str
    run_id: str
    timestamp_jst: str
    eval_date: str

    symbol: str
    executed: bool
    skip_reason: Optional[str]

    entry_price: float
    atr_pct: float

    distance_25ma_pct: Optional[float]
    distance_20d_high_pct: Optional[float]
    distance_52w_high_pct: Optional[float]
    atr_expansion_ratio: Optional[float]

    rsi14: Optional[float]
    gap_up_pct: Optional[float]
    volume_zscore: Optional[float]

    sector: Optional[str]
    sector_rank: Optional[int]
    rsr_rank: Optional[float]
    rsr_score: Optional[float]

    market_regime: Optional[str]
    breadth_50: Optional[float]
    breadth_75: Optional[float]

    breakout_strength: Optional[float]
    volatility_regime: Optional[str]

    cash_ratio: Optional[float]
    slot_utilization: Optional[float]

    executed_qty: Optional[int]
    capital_at_risk: Optional[float]

    # Forward returns — filled by enrich_forward_returns()
    forward_1d: Optional[float] = None
    forward_3d: Optional[float] = None
    forward_5d: Optional[float] = None
    forward_10d: Optional[float] = None
    forward_20d: Optional[float] = None
    forward_mae: Optional[float] = None
    forward_mfe: Optional[float] = None
    continuation_probability: Optional[float] = None
    breakout_failure_flag: Optional[bool] = None
    enrichment_status: str = "pending"


@dataclass
class PositionLifecycleRecord:
    """Daily snapshot for an open position."""
    schema_version: int
    snapshot_id: str
    timestamp_jst: str
    symbol: str
    days_held: int
    entry_price: float
    current_price: float
    unrealized_pnl_pct: float
    distance_from_peak_pct: Optional[float]
    atr_pct: Optional[float]
    rsr_rank: Optional[float]
    sector: Optional[str]
    market_regime: Optional[str]
    drawdown_from_peak: Optional[float]
    momentum_decay: Optional[float]


@dataclass
class ExpectancyBucket:
    label: str
    sample_count: int
    executed_count: int
    skipped_count: int
    expectancy_5d: Optional[float]
    expectancy_10d: Optional[float]
    expectancy_20d: Optional[float]
    mae: Optional[float]
    mfe: Optional[float]
    win_rate_5d: Optional[float]
    continuation_probability: Optional[float]
    breakout_failure_rate: Optional[float]


@dataclass
class RegimeExpectancyCell:
    regime: str
    atr_bucket: str
    sample_count: int
    expectancy_5d: Optional[float]
    expectancy_10d: Optional[float]
    win_rate_5d: Optional[float]


@dataclass
class CounterfactualResult:
    skip_reason: str
    count: int
    expectancy_5d: Optional[float]
    expectancy_10d: Optional[float]
    win_rate_5d: Optional[float]
    missed_alpha_score: Optional[float]


@dataclass
class SectorConcentrationResult:
    effective_sector_count: float
    concentration_index: float
    dominant_sector: Optional[str]
    dominant_sector_share: float
    sector_shares: Dict[str, float]
    monoculture_alert: bool
    monoculture_threshold: float = 0.60


@dataclass
class LifecycleDecay:
    """Unrealized PnL decay by holding duration bucket."""
    label: str
    avg_unrealized_pnl_pct: Optional[float]
    avg_drawdown_from_peak: Optional[float]
    sample_count: int


@dataclass
class ForwardExpectancyReport:
    available: bool
    snapshot_count: int
    enriched_count: int
    eval_date_range: Tuple[str, str]

    atr_buckets: List[ExpectancyBucket]
    extension_expectancy: List[ExpectancyBucket]
    regime_matrix: List[RegimeExpectancyCell]
    counterfactual: List[CounterfactualResult]
    sector_concentration: Optional[SectorConcentrationResult]
    lifecycle_decay: List[LifecycleDecay]

    overheating_verdict: str
    overheating_confidence: str
    overheating_evidence: str

    insufficiency_markers: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot ID
# ─────────────────────────────────────────────────────────────────────────────

def _make_snapshot_id(run_id: str, symbol: str, eval_date: str, executed: bool) -> str:
    key = f"{run_id}|{symbol}|{eval_date}|{executed}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ─────────────────────────────────────────────────────────────────────────────
# Persistence — append-only, immutable, atomic
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line)


def append_entry_snapshot(snapshot: EntryFeatureSnapshot, path: Path) -> None:
    d = asdict(snapshot)
    _atomic_append(path, d)
    logger.debug("Snapshot appended: %s %s executed=%s", snapshot.symbol, snapshot.eval_date, snapshot.executed)


def append_position_lifecycle(record: PositionLifecycleRecord, path: Path) -> None:
    _atomic_append(path, asdict(record))


def load_entry_snapshots(path: Path) -> List[EntryFeatureSnapshot]:
    if not path.exists():
        return []
    snapshots: List[EntryFeatureSnapshot] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            snapshots.append(EntryFeatureSnapshot(**{
                k: d.get(k) for k in EntryFeatureSnapshot.__dataclass_fields__
            }))
        except Exception as exc:
            logger.warning("Skipping corrupt snapshot: %s", exc)
    return snapshots


def load_position_lifecycle(path: Path) -> List[PositionLifecycleRecord]:
    if not path.exists():
        return []
    records: List[PositionLifecycleRecord] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            records.append(PositionLifecycleRecord(**{
                k: d.get(k) for k in PositionLifecycleRecord.__dataclass_fields__
            }))
        except Exception as exc:
            logger.warning("Skipping corrupt lifecycle record: %s", exc)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Snapshot writer — called from run_live_signal.py (fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def write_signal_snapshots(
    signals: list,
    run_id: str,
    cash_ratio: float,
    slot_utilization: float,
    breadth_50: float,
    breadth_75: float,
    market_regime: str,
    snapshots_path: Path,
) -> int:
    """
    Write EntryFeatureSnapshot for all BUY signals (executed and skipped).
    Fail-open: never raises.
    Returns number of snapshots written.
    """
    now_jst = datetime.now(JST)
    eval_date = now_jst.strftime("%Y-%m-%d")
    written = 0

    for sig in signals:
        action = sig.get("action", "")
        signal_val = sig.get("signal", 0)
        if action not in ("BUY", "SKIP_BUY") and signal_val != 1:
            continue

        symbol = sig.get("symbol", "")
        if not symbol:
            continue

        executed = action == "BUY"
        entry_price = float(sig.get("estimated_price") or sig.get("price") or 0.0)
        atr20 = float(sig.get("atr20") or 0.0)
        atr_pct = atr20 / entry_price if entry_price > 0 else 0.0

        _d25 = sig.get("distance_25ma_pct")
        if _d25 is None:
            _p = float(sig.get("price") or sig.get("close") or 0.0)
            _m = float(sig.get("sma_25") or sig.get("ma25") or 0.0)
            _d25 = round((_p - _m) / _m, 6) if _m > 0 and _p > 0 else None

        snap = EntryFeatureSnapshot(
            schema_version=SCHEMA_VERSION,
            snapshot_id=_make_snapshot_id(run_id, symbol, eval_date, executed),
            run_id=run_id,
            timestamp_jst=now_jst.isoformat(),
            eval_date=eval_date,
            symbol=symbol,
            executed=executed,
            skip_reason=sig.get("reason") if not executed else None,
            entry_price=entry_price,
            atr_pct=round(atr_pct, 6),
            distance_25ma_pct=_d25,
            distance_20d_high_pct=None,
            distance_52w_high_pct=None,
            atr_expansion_ratio=None,
            rsi14=None,
            gap_up_pct=None,
            volume_zscore=None,
            sector=sig.get("sector"),
            sector_rank=None,
            rsr_rank=float(sig.get("rsr_rank") or sig.get("rsr") or 0.0),
            rsr_score=float(sig.get("rsr") or 0.0),
            market_regime=market_regime,
            breadth_50=breadth_50,
            breadth_75=breadth_75,
            breakout_strength=None,
            volatility_regime="high" if atr_pct > 0.04 else ("low" if atr_pct < 0.015 else "medium"),
            cash_ratio=cash_ratio,
            slot_utilization=slot_utilization,
            executed_qty=int(sig.get("qty") or 0) if executed else None,
            capital_at_risk=float(sig.get("qty") or 0) * entry_price if executed else None,
            enrichment_status="pending",
        )

        try:
            append_entry_snapshot(snap, snapshots_path)
            written += 1
        except Exception as exc:
            logger.warning("Snapshot write failed for %s: %s", symbol, exc)

    return written


# ─────────────────────────────────────────────────────────────────────────────
# Backfill — create snapshots from historical trades.jsonl
# ─────────────────────────────────────────────────────────────────────────────

def backfill_from_trades(trades: List[dict], snapshots_path: Path) -> int:
    """
    Populate entry_features.jsonl from existing trades.jsonl BUY records.
    Only writes records not already in the snapshot store.
    Returns number of new snapshots written.
    """
    existing = load_entry_snapshots(snapshots_path)
    existing_ids = {s.snapshot_id for s in existing}

    buys = [t for t in trades if t.get("side") == "BUY"]
    written = 0

    for trade in buys:
        symbol = trade.get("symbol", "")
        eval_date = str(trade.get("date", ""))[:10]
        price = float(trade.get("price") or 0.0)
        atr20 = float(trade.get("atr20") or 0.0)
        atr_pct = atr20 / price if price > 0 else 0.0

        # Use a stable run_id for backfilled records
        snap_id = _make_snapshot_id("backfill", symbol, eval_date, True)
        if snap_id in existing_ids:
            continue

        snap = EntryFeatureSnapshot(
            schema_version=SCHEMA_VERSION,
            snapshot_id=snap_id,
            run_id="backfill",
            timestamp_jst=f"{eval_date}T09:00:00+09:00",
            eval_date=eval_date,
            symbol=symbol,
            executed=True,
            skip_reason=None,
            entry_price=price,
            atr_pct=round(atr_pct, 6),
            distance_25ma_pct=None,
            distance_20d_high_pct=None,
            distance_52w_high_pct=None,
            atr_expansion_ratio=None,
            rsi14=None,
            gap_up_pct=None,
            volume_zscore=None,
            sector=trade.get("sector"),
            sector_rank=None,
            rsr_rank=None,
            rsr_score=None,
            market_regime=trade.get("entry_regime"),
            breadth_50=None,
            breadth_75=None,
            breakout_strength=None,
            volatility_regime="high" if atr_pct > 0.04 else ("low" if atr_pct < 0.015 else "medium"),
            cash_ratio=None,
            slot_utilization=None,
            executed_qty=int(float(trade.get("qty") or 0)),
            capital_at_risk=float(trade.get("amount") or 0.0),
            enrichment_status="pending",
        )
        try:
            append_entry_snapshot(snap, snapshots_path)
            written += 1
            existing_ids.add(snap_id)
        except Exception as exc:
            logger.warning("Backfill write failed for %s: %s", symbol, exc)

    logger.info("Backfill: %d new snapshots written (total buys=%d)", written, len(buys))
    return written


# ─────────────────────────────────────────────────────────────────────────────
# Forward return computation (yfinance — fail-open, cached)
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(symbol: str, entry_date_str: str) -> str:
    return hashlib.md5(f"{symbol}:{entry_date_str}".encode()).hexdigest()


def _load_cache(cache_dir: Path, symbol: str, entry_date_str: str) -> Optional[dict]:
    key = _cache_key(symbol, entry_date_str)
    p = cache_dir / f"{key}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(cache_dir: Path, symbol: str, entry_date_str: str, data: dict) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(symbol, entry_date_str)
    p = cache_dir / f"{key}.json"
    try:
        p.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def fetch_forward_returns(
    symbol: str,
    entry_date_str: str,
    entry_price: float,
    cache_dir: Path,
) -> Optional[dict]:
    """
    Fetch forward OHLCV via yfinance and compute forward returns.
    Returns dict with forward_Nd keys, MAE, MFE, continuation_probability.
    Caches results to avoid re-downloading.
    Fail-open: returns None on any error.
    """
    cached = _load_cache(cache_dir, symbol, entry_date_str)
    if cached is not None:
        return cached

    try:
        import yfinance as yf
    except ImportError:
        logger.debug("yfinance not available — forward returns skipped")
        return None

    if not os.environ.get("ALLOW_YFINANCE_NETWORK", "false").lower() == "true":
        return None

    try:
        entry_date = date.fromisoformat(entry_date_str)
        fetch_start = entry_date
        fetch_end = entry_date + timedelta(days=32)  # cover 20 trading days + buffer

        # yfinance: require the ticker to end in .T for Tokyo stocks
        ticker = yf.Ticker(symbol)
        hist = ticker.history(
            start=fetch_start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            auto_adjust=True,
        )

        if hist.empty or len(hist) < 2:
            _save_cache(cache_dir, symbol, entry_date_str, {"status": "no_data"})
            return None

        closes = hist["Close"].values.tolist()
        lows = hist["Low"].values.tolist()
        highs = hist["High"].values.tolist()

        # Reference price: first available close (entry day or next trading day)
        ref_price = closes[0]
        if ref_price <= 0:
            return None

        def _fwd(n: int) -> Optional[float]:
            if len(closes) > n:
                return round((closes[n] - ref_price) / ref_price, 6)
            return None

        # MAE / MFE over 20 trading days
        window = closes[:21]
        lows_w = lows[:21]
        highs_w = highs[:21]
        mae = round((min(lows_w) - ref_price) / ref_price, 6) if lows_w else None
        mfe = round((max(highs_w) - ref_price) / ref_price, 6) if highs_w else None

        # Continuation: price at day5 > day0 (uptrend continues)
        fwd5 = _fwd(5)
        cont_prob = 1.0 if (fwd5 is not None and fwd5 > 0) else 0.0

        # Breakout failure: fwd5 < -5% or < entry after 3d
        fwd3 = _fwd(3)
        fail_flag = bool(fwd3 is not None and fwd3 < -0.05)

        result = {
            "status": "enriched",
            "forward_1d": _fwd(1),
            "forward_3d": fwd3,
            "forward_5d": fwd5,
            "forward_10d": _fwd(10),
            "forward_20d": _fwd(20),
            "forward_mae": mae,
            "forward_mfe": mfe,
            "continuation_probability": cont_prob,
            "breakout_failure_flag": fail_flag,
        }
        _save_cache(cache_dir, symbol, entry_date_str, result)
        return result

    except Exception as exc:
        logger.warning(
            "yfinance fetch failed symbol=%s start=%s end=%s exc_type=%s msg=%s",
            symbol,
            entry_date_str,
            (date.fromisoformat(entry_date_str[:10]) + timedelta(days=32)).isoformat()
            if entry_date_str else "?",
            type(exc).__name__,
            exc,
            exc_info=True,
        )
        return None


def enrich_snapshots_with_forward_returns(
    snapshots: List[EntryFeatureSnapshot],
    cache_dir: Path,
    max_enrichment_days: int = 25,
) -> List[EntryFeatureSnapshot]:
    """
    Fill forward_Nd fields for snapshots where eval_date is far enough in the past.
    Modifies snapshots in-place (returns same list).
    """
    today = datetime.now(JST).date()
    enriched = 0

    for snap in snapshots:
        if snap.enrichment_status == "enriched":
            continue
        try:
            eval_d = date.fromisoformat(snap.eval_date[:10])
        except ValueError:
            continue
        days_ago = (today - eval_d).days
        if days_ago < 2:
            continue  # too recent for meaningful forward returns

        fwd = fetch_forward_returns(snap.symbol, snap.eval_date, snap.entry_price, cache_dir)
        if fwd is None:
            if days_ago > max_enrichment_days:
                snap.enrichment_status = "expired"
            continue

        if fwd.get("status") == "no_data":
            snap.enrichment_status = "expired"
            continue

        snap.forward_1d = fwd.get("forward_1d")
        snap.forward_3d = fwd.get("forward_3d")
        snap.forward_5d = fwd.get("forward_5d")
        snap.forward_10d = fwd.get("forward_10d")
        snap.forward_20d = fwd.get("forward_20d")
        snap.forward_mae = fwd.get("forward_mae")
        snap.forward_mfe = fwd.get("forward_mfe")
        snap.continuation_probability = fwd.get("continuation_probability")
        snap.breakout_failure_flag = fwd.get("breakout_failure_flag")
        snap.enrichment_status = "enriched"
        enriched += 1

    logger.info("Enriched %d/%d snapshots with forward returns", enriched, len(snapshots))
    return snapshots


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def _win_rate(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return round(sum(1 for v in vals if v > 0) / len(vals), 4)


def _expectancy_stats(snaps: List[EntryFeatureSnapshot], fwd_key: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """Returns (expectancy, win_rate, mae, mfe, cont_prob) for a list of snapshots."""
    fwd_vals = [getattr(s, fwd_key) for s in snaps if getattr(s, fwd_key) is not None]
    mae_vals = [s.forward_mae for s in snaps if s.forward_mae is not None]
    mfe_vals = [s.forward_mfe for s in snaps if s.forward_mfe is not None]
    cont_vals = [s.continuation_probability for s in snaps if s.continuation_probability is not None]
    return (
        _safe_mean(fwd_vals),
        _win_rate(fwd_vals),
        _safe_mean(mae_vals),
        _safe_mean(mfe_vals),
        _safe_mean(cont_vals),
    )


def _breakout_failure_rate(snaps: List[EntryFeatureSnapshot]) -> Optional[float]:
    flags = [s.breakout_failure_flag for s in snaps if s.breakout_failure_flag is not None]
    if not flags:
        return None
    return round(sum(1 for f in flags if f) / len(flags), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────

def analyze_atr_buckets(snapshots: List[EntryFeatureSnapshot]) -> List[ExpectancyBucket]:
    enriched = [s for s in snapshots if s.enrichment_status == "enriched"]
    results: List[ExpectancyBucket] = []
    for label, lo, hi in ATR_BUCKETS:
        in_bucket = [s for s in snapshots if lo <= s.atr_pct < hi]
        executed_b = [s for s in in_bucket if s.executed]
        skipped_b = [s for s in in_bucket if not s.executed]
        enriched_b = [s for s in in_bucket if s.enrichment_status == "enriched"]
        exp5, wr5, mae, mfe, cont = _expectancy_stats(enriched_b, "forward_5d")
        exp10, _, _, _, _ = _expectancy_stats(enriched_b, "forward_10d")
        exp20, _, _, _, _ = _expectancy_stats(enriched_b, "forward_20d")
        results.append(ExpectancyBucket(
            label=label,
            sample_count=len(in_bucket),
            executed_count=len(executed_b),
            skipped_count=len(skipped_b),
            expectancy_5d=exp5,
            expectancy_10d=exp10,
            expectancy_20d=exp20,
            mae=mae,
            mfe=mfe,
            win_rate_5d=wr5,
            continuation_probability=cont,
            breakout_failure_rate=_breakout_failure_rate(enriched_b),
        ))
    return results


def analyze_extension_expectancy(
    snapshots: List[EntryFeatureSnapshot],
    feature_name: str,
    buckets: List[Tuple[str, float, float]],
) -> List[ExpectancyBucket]:
    results: List[ExpectancyBucket] = []
    for label, lo, hi in buckets:
        in_bucket = [
            s for s in snapshots
            if getattr(s, feature_name, None) is not None
            and lo <= getattr(s, feature_name) < hi
        ]
        executed_b = [s for s in in_bucket if s.executed]
        enriched_b = [s for s in in_bucket if s.enrichment_status == "enriched"]
        exp5, wr5, mae, mfe, cont = _expectancy_stats(enriched_b, "forward_5d")
        exp10, _, _, _, _ = _expectancy_stats(enriched_b, "forward_10d")
        exp20, _, _, _, _ = _expectancy_stats(enriched_b, "forward_20d")
        results.append(ExpectancyBucket(
            label=label,
            sample_count=len(in_bucket),
            executed_count=len(executed_b),
            skipped_count=len(in_bucket) - len(executed_b),
            expectancy_5d=exp5,
            expectancy_10d=exp10,
            expectancy_20d=exp20,
            mae=mae,
            mfe=mfe,
            win_rate_5d=wr5,
            continuation_probability=cont,
            breakout_failure_rate=_breakout_failure_rate(enriched_b),
        ))
    return results


def analyze_regime_conditioned(snapshots: List[EntryFeatureSnapshot]) -> List[RegimeExpectancyCell]:
    regimes = sorted({s.market_regime for s in snapshots if s.market_regime})
    results: List[RegimeExpectancyCell] = []
    for regime in regimes:
        regime_snaps = [s for s in snapshots if s.market_regime == regime]
        for atr_label, lo, hi in ATR_BUCKETS:
            bucket_snaps = [s for s in regime_snaps if lo <= s.atr_pct < hi]
            enriched_b = [s for s in bucket_snaps if s.enrichment_status == "enriched"]
            exp5, wr5, _, _, _ = _expectancy_stats(enriched_b, "forward_5d")
            exp10, _, _, _, _ = _expectancy_stats(enriched_b, "forward_10d")
            results.append(RegimeExpectancyCell(
                regime=regime,
                atr_bucket=atr_label,
                sample_count=len(bucket_snaps),
                expectancy_5d=exp5,
                expectancy_10d=exp10,
                win_rate_5d=wr5,
            ))
    return results


def analyze_counterfactual(snapshots: List[EntryFeatureSnapshot]) -> List[CounterfactualResult]:
    skipped = [s for s in snapshots if not s.executed and s.skip_reason]
    if not skipped:
        return []

    # Group by skip reason
    reason_map: Dict[str, List[EntryFeatureSnapshot]] = {}
    for s in skipped:
        reason = (s.skip_reason or "unknown")[:40]
        reason_map.setdefault(reason, []).append(s)

    results: List[CounterfactualResult] = []
    for reason, snaps in sorted(reason_map.items()):
        enriched_b = [s for s in snaps if s.enrichment_status == "enriched"]
        exp5, wr5, _, _, _ = _expectancy_stats(enriched_b, "forward_5d")
        exp10, _, _, _, _ = _expectancy_stats(enriched_b, "forward_10d")
        missed = None
        if exp5 is not None and len(enriched_b) > 0:
            missed = round(abs(exp5) * len(enriched_b), 4) if exp5 > 0 else 0.0
        results.append(CounterfactualResult(
            skip_reason=reason,
            count=len(snaps),
            expectancy_5d=exp5,
            expectancy_10d=exp10,
            win_rate_5d=wr5,
            missed_alpha_score=missed,
        ))
    # Sort by expected loss (non-executed winners are most important)
    results.sort(key=lambda r: -(r.expectancy_5d or 0))
    return results


def analyze_sector_concentration(snapshots: List[EntryFeatureSnapshot]) -> Optional[SectorConcentrationResult]:
    sector_counts: Dict[str, int] = {}
    for s in snapshots:
        sec = s.sector or "unknown"
        sector_counts[sec] = sector_counts.get(sec, 0) + 1

    total = sum(sector_counts.values())
    if total == 0:
        return None

    shares = {sec: cnt / total for sec, cnt in sector_counts.items()}
    dominant = max(shares, key=lambda k: shares[k])
    dominant_share = shares[dominant]

    # Shannon entropy-based effective sector count
    entropy = -sum(p * math.log(p) for p in shares.values() if p > 0)
    eff_count = round(math.exp(entropy), 2)

    # Herfindahl index
    herf = sum(p ** 2 for p in shares.values())
    conc_index = round(herf, 4)

    return SectorConcentrationResult(
        effective_sector_count=eff_count,
        concentration_index=conc_index,
        dominant_sector=dominant,
        dominant_sector_share=round(dominant_share, 4),
        sector_shares={k: round(v, 4) for k, v in sorted(shares.items(), key=lambda x: -x[1])},
        monoculture_alert=dominant_share > 0.60,
    )


def analyze_lifecycle_decay(
    lifecycle: List[PositionLifecycleRecord],
) -> List[LifecycleDecay]:
    """Compute unrealized PnL decay by holding duration bucket."""
    buckets = [("1-3d", 1, 3), ("4-7d", 4, 7), ("8-14d", 8, 14),
               ("15-21d", 15, 21), ("22-40d", 22, 40), ("40d+", 41, 9999)]
    results: List[LifecycleDecay] = []
    for label, lo, hi in buckets:
        in_b = [r for r in lifecycle if lo <= r.days_held <= hi]
        avg_pnl = None
        avg_dd = None
        if in_b:
            pnls = [r.unrealized_pnl_pct for r in in_b if r.unrealized_pnl_pct is not None]
            dds = [r.drawdown_from_peak for r in in_b if r.drawdown_from_peak is not None]
            avg_pnl = round(sum(pnls) / len(pnls), 4) if pnls else None
            avg_dd = round(sum(dds) / len(dds), 4) if dds else None
        results.append(LifecycleDecay(
            label=label, avg_unrealized_pnl_pct=avg_pnl,
            avg_drawdown_from_peak=avg_dd, sample_count=len(in_b),
        ))
    return results


def _overheating_verdict(
    atr_buckets: List[ExpectancyBucket],
    enriched_count: int,
) -> Tuple[str, str, str]:
    """Determine whether high-ATR entries show statistically weak expectancy."""
    if enriched_count < 5:
        return "insufficient_data", "LOW", f"Only {enriched_count} enriched snapshots (need ≥5)"

    # Compare <2.5% bucket vs 2.5%+ bucket
    low_atr = [b for b in atr_buckets if b.label in ("<1.5%", "1.5-2.5%")]
    high_atr = [b for b in atr_buckets if b.label in ("2.5-4%", "4%+")]

    low_exp = [b.expectancy_5d for b in low_atr if b.expectancy_5d is not None and b.sample_count >= 3]
    high_exp = [b.expectancy_5d for b in high_atr if b.expectancy_5d is not None and b.sample_count >= 3]

    if not low_exp or not high_exp:
        return "insufficient_data", "LOW", "Insufficient samples in ATR buckets for comparison"

    avg_low = sum(low_exp) / len(low_exp)
    avg_high = sum(high_exp) / len(high_exp)
    diff = avg_high - avg_low
    high_bucket_best = atr_buckets[-1] if atr_buckets else None

    if diff < -0.02:
        verdict = "overheating_confirmed"
        conf = "HIGH" if diff < -0.05 else "MEDIUM"
        evid = f"High-ATR 5d expectancy={avg_high:.3f} vs low-ATR={avg_low:.3f} (diff={diff:+.3f})"
    elif diff > 0.02:
        verdict = "momentum_capture"
        conf = "MEDIUM"
        evid = f"High-ATR shows superior 5d expectancy={avg_high:.3f} vs low-ATR={avg_low:.3f}"
    else:
        verdict = "neutral"
        conf = "LOW"
        evid = f"No significant ATR→expectancy relationship (diff={diff:+.3f})"

    return verdict, conf, evid


# ─────────────────────────────────────────────────────────────────────────────
# Chart generation (matplotlib, fail-open)
# ─────────────────────────────────────────────────────────────────────────────

def generate_ffe_charts(
    report: ForwardExpectancyReport,
    chart_dir: Path,
) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return []

    chart_dir.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []

    def _save(fig, name: str) -> Optional[str]:
        p = chart_dir / name
        try:
            fig.savefig(str(p), dpi=100, bbox_inches="tight")
            plt.close(fig)
            return str(p)
        except Exception as exc:
            logger.warning("Chart save failed %s: %s", name, exc)
            plt.close(fig)
            return None

    # 1. ATR bucket expectancy (bar chart)
    if report.atr_buckets:
        labels = [b.label for b in report.atr_buckets]
        exp5 = [b.expectancy_5d * 100 if b.expectancy_5d is not None else 0 for b in report.atr_buckets]
        exp10 = [b.expectancy_10d * 100 if b.expectancy_10d is not None else 0 for b in report.atr_buckets]
        counts = [b.sample_count for b in report.atr_buckets]
        x = range(len(labels))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar([i - 0.2 for i in x], exp5, width=0.35, label="5d expectancy%", color="#2196F3", alpha=0.8)
        ax.bar([i + 0.2 for i in x], exp10, width=0.35, label="10d expectancy%", color="#FF9800", alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.8)
        for i, (cnt, e) in enumerate(zip(counts, exp5)):
            ax.text(i - 0.2, max(0, e) + 0.1, f"n={cnt}", ha="center", fontsize=7)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels)
        ax.set_title("ATR Bucket Forward Expectancy (%)", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        p = _save(fig, "06_atr_bucket_expectancy.png")
        if p:
            paths.append(p)

    # 2. Regime-conditioned expectancy matrix (heatmap-style)
    if report.regime_matrix:
        regimes = sorted({c.regime for c in report.regime_matrix})
        atr_labels = [b[0] for b in ATR_BUCKETS]
        matrix = {}
        for cell in report.regime_matrix:
            matrix[(cell.regime, cell.atr_bucket)] = cell.expectancy_5d

        if regimes and atr_labels:
            data = [[matrix.get((r, a), 0) or 0 for a in atr_labels] for r in regimes]
            import numpy as np
            arr = np.array(data) * 100
            fig, ax = plt.subplots(figsize=(9, max(3, len(regimes) * 0.8 + 1)))
            im = ax.imshow(arr, cmap="RdYlGn", vmin=-10, vmax=10, aspect="auto")
            ax.set_xticks(range(len(atr_labels)))
            ax.set_xticklabels(atr_labels, fontsize=9)
            ax.set_yticks(range(len(regimes)))
            ax.set_yticklabels(regimes, fontsize=9)
            for i in range(len(regimes)):
                for j in range(len(atr_labels)):
                    v = arr[i, j]
                    cnt = next((c.sample_count for c in report.regime_matrix
                                if c.regime == regimes[i] and c.atr_bucket == atr_labels[j]), 0)
                    ax.text(j, i, f"{v:+.1f}%\nn={cnt}", ha="center", va="center", fontsize=7)
            plt.colorbar(im, ax=ax, label="5d expectancy%")
            ax.set_title("Regime × ATR Bucket 5d Expectancy", fontsize=11)
            fig.tight_layout()
            p = _save(fig, "07_regime_expectancy_matrix.png")
            if p:
                paths.append(p)

    # 3. Sector concentration pie
    if report.sector_concentration:
        sc = report.sector_concentration
        labels = list(sc.sector_shares.keys())[:8]
        sizes = [sc.sector_shares[l] * 100 for l in labels]
        fig, ax = plt.subplots(figsize=(7, 5))
        colors = plt.cm.Set3.colors[:len(labels)]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, autopct="%1.0f%%",
            colors=colors, startangle=90,
        )
        for t in autotexts:
            t.set_fontsize(9)
        alert = " [ALERT]" if sc.monoculture_alert else ""
        ax.set_title(f"Sector Concentration{alert}\nHerfindahl={sc.concentration_index:.3f}, eff_sectors={sc.effective_sector_count:.1f}", fontsize=11)
        fig.tight_layout()
        p = _save(fig, "08_sector_concentration.png")
        if p:
            paths.append(p)

    # 4. Skipped opportunity expectancy (if counterfactual data available)
    if report.counterfactual and any(c.expectancy_5d is not None for c in report.counterfactual):
        reasons = [c.skip_reason[:20] for c in report.counterfactual[:6]]
        exp5_cf = [c.expectancy_5d * 100 if c.expectancy_5d is not None else 0 for c in report.counterfactual[:6]]
        counts_cf = [c.count for c in report.counterfactual[:6]]
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = ["#F44336" if e > 0 else "#4CAF50" for e in exp5_cf]
        bars = ax.barh(reasons, exp5_cf, color=colors, alpha=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        for bar, cnt in zip(bars, counts_cf):
            ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    f"n={cnt}", va="center", fontsize=8)
        ax.set_title("Skipped Opportunity 5d Expectancy (red=missed alpha)", fontsize=11)
        ax.set_xlabel("5d forward return %")
        fig.tight_layout()
        p = _save(fig, "09_skipped_opportunity.png")
        if p:
            paths.append(p)

    # 5. Lifecycle decay curve
    if report.lifecycle_decay and any(ld.sample_count > 0 for ld in report.lifecycle_decay):
        ld_labels = [ld.label for ld in report.lifecycle_decay if ld.sample_count > 0]
        ld_pnl = [ld.avg_unrealized_pnl_pct * 100 if ld.avg_unrealized_pnl_pct is not None else 0 for ld in report.lifecycle_decay if ld.sample_count > 0]
        if ld_labels:
            fig, ax = plt.subplots(figsize=(8, 3))
            ax.plot(range(len(ld_labels)), ld_pnl, marker="o", color="#4CAF50", linewidth=1.5)
            ax.fill_between(range(len(ld_labels)), ld_pnl, 0,
                            where=[p >= 0 for p in ld_pnl], alpha=0.2, color="#4CAF50")
            ax.fill_between(range(len(ld_labels)), ld_pnl, 0,
                            where=[p < 0 for p in ld_pnl], alpha=0.2, color="#F44336")
            ax.axhline(0, color="gray", linewidth=0.8)
            ax.set_xticks(range(len(ld_labels)))
            ax.set_xticklabels(ld_labels, fontsize=9)
            ax.set_title("Position Lifecycle: Unrealized PnL by Hold Duration", fontsize=11)
            ax.set_ylabel("Avg unrealized PnL %")
            ax.grid(axis="y", alpha=0.3)
            fig.tight_layout()
            p = _save(fig, "10_lifecycle_decay.png")
            if p:
                paths.append(p)

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# FeatureForwardExpectancyEngine
# ─────────────────────────────────────────────────────────────────────────────

class FeatureForwardExpectancyEngine:
    """
    Deterministic feature → forward-return attribution engine.

    Usage:
        engine = FeatureForwardExpectancyEngine(
            snapshots_path=ENTRY_FEATURES_FILE,
            lifecycle_path=POSITION_LIFECYCLE_FILE,
            cache_dir=FORWARD_RETURN_CACHE_DIR,
        )
        report = engine.run()
    """

    def __init__(
        self,
        snapshots_path: Path,
        lifecycle_path: Path,
        cache_dir: Path,
    ) -> None:
        self.snapshots_path = snapshots_path
        self.lifecycle_path = lifecycle_path
        self.cache_dir = cache_dir

    def run(
        self,
        week_start: Optional[date] = None,
        week_end: Optional[date] = None,
        trades: Optional[List[dict]] = None,
        enrich: bool = True,
    ) -> ForwardExpectancyReport:
        insufficiency: List[str] = []

        # Backfill from trades if available
        if trades:
            try:
                backfill_from_trades(trades, self.snapshots_path)
            except Exception as exc:
                logger.warning("Backfill failed: %s", exc)
                insufficiency.append(f"Backfill error: {exc}")

        # Load snapshots
        snapshots = load_entry_snapshots(self.snapshots_path)
        lifecycle = load_position_lifecycle(self.lifecycle_path)

        # Filter to window if specified
        if week_start and week_end:
            window_start = week_start - timedelta(days=90)  # 90d lookback for ATR analysis
            snapshots = [
                s for s in snapshots
                if window_start.isoformat() <= s.eval_date[:10] <= week_end.isoformat()
            ]

        if not snapshots:
            insufficiency.append("No entry feature snapshots available")
            return ForwardExpectancyReport(
                available=False, snapshot_count=0, enriched_count=0,
                eval_date_range=("", ""),
                atr_buckets=[], extension_expectancy=[], regime_matrix=[],
                counterfactual=[], sector_concentration=None, lifecycle_decay=[],
                overheating_verdict="insufficient_data",
                overheating_confidence="LOW",
                overheating_evidence="No snapshots available",
                insufficiency_markers=insufficiency,
            )

        # Enrich with forward returns
        if enrich:
            try:
                snapshots = enrich_snapshots_with_forward_returns(snapshots, self.cache_dir)
            except Exception as exc:
                logger.warning("Enrichment failed: %s", exc)
                insufficiency.append(f"Forward return enrichment error: {exc}")

        enriched_count = sum(1 for s in snapshots if s.enrichment_status == "enriched")
        eval_dates = sorted({s.eval_date[:10] for s in snapshots})
        date_range = (eval_dates[0], eval_dates[-1]) if eval_dates else ("", "")

        if enriched_count < 3:
            insufficiency.append(
                f"Only {enriched_count} enriched snapshots — forward expectancy analysis unreliable; "
                "need ≥3 with completed forward windows"
            )

        # Run analyses
        atr_buckets = analyze_atr_buckets(snapshots)
        ext_expectancy = analyze_extension_expectancy(snapshots, "distance_25ma_pct", DIST_25MA_BUCKETS)
        regime_matrix = analyze_regime_conditioned(snapshots)
        counterfactual = analyze_counterfactual(snapshots)
        sector_conc = analyze_sector_concentration(snapshots)
        lifecycle_decay = analyze_lifecycle_decay(lifecycle)

        # Overheating verdict
        verdict, conf, evid = _overheating_verdict(atr_buckets, enriched_count)

        # Warn on sector monoculture
        if sector_conc and sector_conc.monoculture_alert:
            insufficiency.append(
                f"Sector monoculture: {sector_conc.dominant_sector} = "
                f"{sector_conc.dominant_sector_share*100:.0f}% of entries"
            )

        # Note on missing features
        has_dist25ma = any(s.distance_25ma_pct is not None for s in snapshots)
        if not has_dist25ma:
            insufficiency.append(
                "distance_25ma_pct not available in current snapshots — "
                "extension expectancy table uses ATR proxy only"
            )

        return ForwardExpectancyReport(
            available=True,
            snapshot_count=len(snapshots),
            enriched_count=enriched_count,
            eval_date_range=date_range,
            atr_buckets=atr_buckets,
            extension_expectancy=ext_expectancy,
            regime_matrix=regime_matrix,
            counterfactual=counterfactual,
            sector_concentration=sector_conc,
            lifecycle_decay=lifecycle_decay,
            overheating_verdict=verdict,
            overheating_confidence=conf,
            overheating_evidence=evid,
            insufficiency_markers=insufficiency,
        )
