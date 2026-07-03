"""
src/universe/outcome_materializer.py — Phase 2.5 Forward Return Materializer

Scans unmaterialized records in promotion_outcome.jsonl and fills in:
  forward_return_5d, forward_return_10d, forward_return_20d, forward_return_30d
  benchmark_return_5d, benchmark_return_20d
  realized_alpha_5d, realized_alpha_20d

Materialization writes to a SEPARATE append-only file (_materialized.jsonl).
Original promotion_outcome.jsonl is NEVER mutated.

Design:
  - No network: reads local OHLCV CSV cache only
  - FAIL_OPEN per symbol: one bad CSV does not block others
  - Idempotent: tracks materialized record_ids; replays are safe
  - Deterministic: same OHLCV → same returns
  - Replayable: materialized JSONL is append-only

OHLCV CSV format: date,close (date in YYYY-MM-DD, close in float)
Directory: data/ohlcv/{symbol}.csv  (symbol as-is, e.g. "1234.T")
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
TRADING_DAYS_WINDOWS = (5, 10, 20, 30)

# Benchmark symbol — TOPIX ETF or index CSV; None to disable benchmark.
# Override by setting BENCHMARK_SYMBOL to a symbol present in OHLCV_DIR.
BENCHMARK_SYMBOL: Optional[str] = "1306.T"  # TOPIX ETF (common proxy); set None to disable


# ── Public entry point ────────────────────────────────────────────────────────

@dataclass
class MaterializationSummary:
    total_unmaterialized: int
    newly_materialized: int
    skipped_no_data: int          # OHLCV not yet available (forward window too short)
    skipped_no_file: int          # CSV file missing entirely
    failed: int                   # CSV parse or other error
    benchmark_available: bool


def materialize_forward_returns(
    outcome_dir: Path,
    ohlcv_dir: Path,
) -> MaterializationSummary:
    """
    Main entry point. FAIL_OPEN wrapper around _run_materialization().

    Returns MaterializationSummary for logging.
    """
    try:
        return _run_materialization(outcome_dir, ohlcv_dir)
    except Exception as exc:
        logger.warning("[MATERIALIZER] top-level error (FAIL_OPEN): %s", exc)
        return MaterializationSummary(0, 0, 0, 0, 1, False)


# ── Internal implementation ───────────────────────────────────────────────────

def _run_materialization(
    outcome_dir: Path,
    ohlcv_dir: Path,
) -> MaterializationSummary:
    promo_file  = outcome_dir / "promotion_outcome.jsonl"
    mat_file    = outcome_dir / "promotion_outcome_materialized.jsonl"

    if not promo_file.exists():
        logger.debug("[MATERIALIZER] no promotion_outcome.jsonl yet — skipping")
        return MaterializationSummary(0, 0, 0, 0, 0, False)

    # Load OHLCV for benchmark once (shared across all symbol materializations).
    benchmark_ohlcv: Optional[Dict[str, float]] = None
    benchmark_available = False
    if BENCHMARK_SYMBOL:
        try:
            benchmark_ohlcv = _load_ohlcv(ohlcv_dir, BENCHMARK_SYMBOL)
            benchmark_available = benchmark_ohlcv is not None and len(benchmark_ohlcv) > 30
        except Exception as exc:
            logger.debug("[MATERIALIZER] benchmark load failed (%s): %s", BENCHMARK_SYMBOL, exc)

    # Load already-materialized record_ids to avoid duplicate writes.
    already_done = _load_materialized_ids(mat_file)

    # Load unmaterialized promotion records.
    records = _load_unmaterialized(promo_file, already_done)

    total       = len(records)
    newly_done  = 0
    no_data     = 0
    no_file     = 0
    failed      = 0

    for rec in records:
        sym       = rec.get("symbol", "")
        record_id = rec.get("record_id", "")
        promo_date = rec.get("promoted_at", "")

        if not sym or not record_id or not promo_date:
            failed += 1
            continue

        # Load symbol OHLCV.
        try:
            sym_ohlcv = _load_ohlcv(ohlcv_dir, sym)
        except Exception as exc:
            logger.debug("[MATERIALIZER] OHLCV load error %s: %s", sym, exc)
            failed += 1
            continue

        if sym_ohlcv is None:
            no_file += 1
            continue

        # Compute forward returns.
        result = _compute_returns(sym_ohlcv, promo_date, TRADING_DAYS_WINDOWS)
        if result is None:
            # Forward window not yet available (data too recent).
            no_data += 1
            continue

        # Compute benchmark returns.
        bm_5d: Optional[float] = None
        bm_20d: Optional[float] = None
        if benchmark_ohlcv is not None:
            bm_result = _compute_returns(benchmark_ohlcv, promo_date, (5, 20))
            if bm_result:
                bm_5d  = bm_result.get(5)
                bm_20d = bm_result.get(20)

        fwd_5d  = result.get(5)
        fwd_20d = result.get(20)
        alpha_5d  = _safe_sub(fwd_5d,  bm_5d)
        alpha_20d = _safe_sub(fwd_20d, bm_20d)

        mat_record: dict = {
            "record_id":           record_id,
            "schema_version":      SCHEMA_VERSION,
            "record_type":         "promotion_materialized",
            "symbol":              sym,
            "promoted_at":         promo_date,
            "materialized_at":     _now_utc(),
            "forward_return_5d":   result.get(5),
            "forward_return_10d":  result.get(10),
            "forward_return_20d":  result.get(20),
            "forward_return_30d":  result.get(30),
            "benchmark_return_5d": bm_5d,
            "benchmark_return_20d": bm_20d,
            "realized_alpha_5d":   alpha_5d,
            "realized_alpha_20d":  alpha_20d,
            "benchmark_symbol":    BENCHMARK_SYMBOL,
        }

        try:
            _append_jsonl(mat_file, mat_record)
            already_done.add(record_id)
            newly_done += 1
            logger.debug(
                "[MATERIALIZER] %s: fwd5d=%s fwd20d=%s alpha20d=%s",
                sym,
                _fmt(result.get(5)), _fmt(result.get(20)), _fmt(alpha_20d),
            )
        except Exception as exc:
            logger.warning("[MATERIALIZER] write error %s: %s", sym, exc)
            failed += 1

    if total > 0 or newly_done > 0:
        logger.info(
            "[MATERIALIZER] total=%d newly_materialized=%d no_data=%d no_file=%d failed=%d",
            total, newly_done, no_data, no_file, failed,
        )

    return MaterializationSummary(
        total_unmaterialized=total,
        newly_materialized=newly_done,
        skipped_no_data=no_data,
        skipped_no_file=no_file,
        failed=failed,
        benchmark_available=benchmark_available,
    )


# Also materialize rotation_outcome forward returns (20d for promoted + demoted).
def materialize_rotation_returns(
    outcome_dir: Path,
    ohlcv_dir: Path,
) -> int:
    """
    Fills promoted_forward_20d / demoted_forward_20d / net_rotation_alpha
    in rotation_outcome_materialized.jsonl.
    Returns count of newly materialized records.
    """
    try:
        return _run_rotation_materialization(outcome_dir, ohlcv_dir)
    except Exception as exc:
        logger.warning("[MATERIALIZER] rotation materialization error (FAIL_OPEN): %s", exc)
        return 0


def _run_rotation_materialization(outcome_dir: Path, ohlcv_dir: Path) -> int:
    rot_file = outcome_dir / "rotation_outcome.jsonl"
    mat_file = outcome_dir / "rotation_outcome_materialized.jsonl"

    if not rot_file.exists():
        return 0

    already_done = _load_materialized_ids(mat_file)
    records      = _load_unmaterialized(rot_file, already_done)
    newly_done   = 0

    for rec in records:
        record_id     = rec.get("record_id", "")
        rot_date      = rec.get("rotation_at", "")
        promoted_sym  = rec.get("promoted_symbol", "")
        demoted_sym   = rec.get("demoted_symbol")   # may be null

        if not record_id or not rot_date or not promoted_sym:
            continue

        try:
            prom_ohlcv = _load_ohlcv(ohlcv_dir, promoted_sym)
            prom_fwd20 = None
            if prom_ohlcv:
                r = _compute_returns(prom_ohlcv, rot_date, (20,))
                prom_fwd20 = r.get(20) if r else None

            dem_fwd20 = None
            if demoted_sym:
                dem_ohlcv = _load_ohlcv(ohlcv_dir, demoted_sym)
                if dem_ohlcv:
                    r2 = _compute_returns(dem_ohlcv, rot_date, (20,))
                    dem_fwd20 = r2.get(20) if r2 else None

            if prom_fwd20 is None:
                continue   # promoted symbol data not yet available

            net_alpha = _safe_sub(prom_fwd20, dem_fwd20)

            mat_record: dict = {
                "record_id":           record_id,
                "schema_version":      SCHEMA_VERSION,
                "record_type":         "rotation_materialized",
                "promoted_symbol":     promoted_sym,
                "demoted_symbol":      demoted_sym,
                "rotation_at":         rot_date,
                "materialized_at":     _now_utc(),
                "promoted_forward_20d": prom_fwd20,
                "demoted_forward_20d":  dem_fwd20,
                "net_rotation_alpha":   net_alpha,
            }
            _append_jsonl(mat_file, mat_record)
            already_done.add(record_id)
            newly_done += 1
        except Exception as exc:
            logger.debug("[MATERIALIZER] rotation record error %s: %s", promoted_sym, exc)

    if newly_done:
        logger.info("[MATERIALIZER] rotation: newly_materialized=%d", newly_done)
    return newly_done


# ── OHLCV helpers ─────────────────────────────────────────────────────────────

def _load_ohlcv(ohlcv_dir: Path, symbol: str) -> Optional[Dict[str, float]]:
    """
    Load OHLCV CSV as {date_str: close_price} ordered dict.
    Returns None if file missing. Raises on parse error (caller handles).
    CSV format: date,close (header on row 0).
    """
    csv_path = ohlcv_dir / f"{symbol}.csv"
    if not csv_path.exists():
        return None

    ohlcv: Dict[str, float] = {}
    for line in csv_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("date"):
            continue  # header
        parts = line.split(",")
        if len(parts) < 2:
            continue
        date_str = parts[0].strip()
        try:
            close = float(parts[1].strip())
            ohlcv[date_str] = close
        except ValueError:
            continue
    return ohlcv if ohlcv else None


def _compute_returns(
    ohlcv: Dict[str, float],
    base_date: str,
    windows: tuple[int, ...],
) -> Optional[Dict[int, Optional[float]]]:
    """
    Compute forward returns for each window (in trading days).
    base_date: YYYY-MM-DD string (may be a weekend/holiday — finds next available).
    Returns None if base_date not yet in OHLCV (data insufficient).
    Returns {window: return_pct} where return_pct may be None if not enough data.
    """
    dates = sorted(ohlcv.keys())  # ascending date strings
    if not dates:
        return None

    # Find base index: first trading date >= base_date.
    base_idx = None
    for i, d in enumerate(dates):
        if d >= base_date:
            base_idx = i
            break

    if base_idx is None:
        return None  # base_date after all available data

    base_close = ohlcv[dates[base_idx]]
    if base_close <= 0:
        return None

    result: Dict[int, Optional[float]] = {}
    for window in windows:
        target_idx = base_idx + window
        if target_idx >= len(dates):
            result[window] = None  # not enough forward data yet
        else:
            fwd_close = ohlcv[dates[target_idx]]
            result[window] = round((fwd_close - base_close) / base_close, 6) if base_close > 0 else None

    return result


# ── Utilities ─────────────────────────────────────────────────────────────────

def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 6)


def _fmt(v: Optional[float]) -> str:
    return f"{v:+.3%}" if v is not None else "null"


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _load_materialized_ids(mat_file: Path) -> set[str]:
    ids: set[str] = set()
    if not mat_file.exists():
        return ids
    try:
        for line in mat_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("record_id")
                if rid:
                    ids.add(rid)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("[MATERIALIZER] materialized_ids load error: %s", exc)
    return ids


def _load_unmaterialized(source_file: Path, already_done: set[str]) -> List[dict]:
    """Return records from source_file not yet in already_done."""
    out: List[dict] = []
    try:
        for line in source_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                rid = rec.get("record_id", "")
                if rid and rid not in already_done:
                    out.append(rec)
            except Exception:
                pass
    except Exception as exc:
        logger.warning("[MATERIALIZER] source load error %s: %s", source_file.name, exc)
    return out
