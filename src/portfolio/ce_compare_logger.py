"""
portfolio/ce_compare_logger.py

CE vs Baseline comparison logging.  No side effects on live execution.
All file I/O is append-only or idempotent.  No lookahead.

Storage layout
--------------
  runtime/ce_compare_pending.json    open entries awaiting exit prices
  runtime/ce_compare_trades.jsonl    completed trade records  (CE_COMPARE tag)
  logs/ce_compare_daily.csv          daily aggregated metrics

Timing model
------------
  Day T  (signal)   record_order()             → pending
  Day T  (exec)     update_fill()              → pending[fill_price] updated
  Day T+1 (startup) try_fill_forward_returns() → pending → trades.jsonl
                    flush_daily_csv()          → CSV appended

PnL model
---------
  fill_price_base == fill_price_ce (same market-open execution).
  PnL delta arises purely from quantity difference.

  pnl_base = qty_base * fill_price * forward_return_1d
  pnl_ce   = qty_ce   * fill_price * forward_return_1d
  delta_pnl = pnl_ce - pnl_base
"""

from __future__ import annotations

import csv
import json
import logging
import warnings
from dataclasses import dataclass, asdict, field
from datetime import date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CSV_DAILY_COLS = [
    "date", "pnl_base", "pnl_ce", "delta_pnl", "cum_delta",
    "fill_rate_base", "fill_rate_ce", "fill_rate_diff",
    "n_trades", "sign_match_rate", "mean_qty_ratio",
]


@dataclass
class TradeRecord:
    tag:             str            = "CE_COMPARE"
    date:            str            = ""
    symbol:          str            = ""
    side:            str            = ""
    qty_base:        int            = 0
    qty_ce:          int            = 0
    fill_price_base: float          = 0.0
    fill_price_ce:   float          = 0.0
    filled_base:     bool           = False
    filled_ce:       bool           = False
    ea:              float          = 0.0
    confidence:      float          = 0.0
    sample_size:     int            = 0
    forward_return:  Optional[float] = None
    sign_match:      Optional[bool]  = None
    qty_ratio:       Optional[float] = None


class CECompareLogger:
    """CE vs baseline comparison logger.  Thread-unsafe; single-process only."""

    def __init__(self, runtime_dir: Path, logs_dir: Path) -> None:
        self._pending_file = runtime_dir / "ce_compare_pending.json"
        self._trades_file  = runtime_dir / "ce_compare_trades.jsonl"
        self._daily_csv    = logs_dir    / "ce_compare_daily.csv"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_header()

    # ------------------------------------------------------------------ #
    # pending state I/O
    # ------------------------------------------------------------------ #

    def _load_pending(self) -> dict:
        if not self._pending_file.exists():
            return {}
        try:
            return json.loads(self._pending_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[CE_COMPARE] pending load error: %s", exc)
            return {}

    def _save_pending(self, pending: dict) -> None:
        try:
            self._pending_file.write_text(
                json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.warning("[CE_COMPARE] pending save error: %s", exc)

    # ------------------------------------------------------------------ #
    # trade log I/O
    # ------------------------------------------------------------------ #

    def _append_trade(self, record: TradeRecord) -> None:
        try:
            with open(self._trades_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.warning("[CE_COMPARE] trade append error: %s", exc)

    def _load_trades(self) -> list[TradeRecord]:
        if not self._trades_file.exists():
            return []
        records: list[TradeRecord] = []
        try:
            for raw in self._trades_file.read_text(encoding="utf-8").splitlines():
                raw = raw.strip()
                if raw:
                    d = json.loads(raw)
                    records.append(TradeRecord(**{k: d.get(k) for k in TradeRecord.__dataclass_fields__}))
        except Exception as exc:
            logger.warning("[CE_COMPARE] trade load error: %s", exc)
        return records

    # ------------------------------------------------------------------ #
    # CSV
    # ------------------------------------------------------------------ #

    def _ensure_csv_header(self) -> None:
        if not self._daily_csv.exists():
            with open(self._daily_csv, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=_CSV_DAILY_COLS).writeheader()

    def _existing_csv_dates(self) -> set[str]:
        dates: set[str] = set()
        if not self._daily_csv.exists():
            return dates
        try:
            with open(self._daily_csv, "r", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    dates.add(row["date"])
        except Exception:
            pass
        return dates

    def _last_cum_delta(self) -> float:
        if not self._daily_csv.exists():
            return 0.0
        try:
            with open(self._daily_csv, "r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                return float(rows[-1].get("cum_delta") or 0.0)
        except Exception:
            pass
        return 0.0

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def record_order(
        self,
        date_str:    str,
        symbol:      str,
        side:        str,
        qty_base:    int,
        qty_ce:      int,
        fill_price:  float,
        ea:          float,
        confidence:  float,
        sample_size: int,
    ) -> None:
        """
        Record a comparison entry at signal time.

        fill_price is the estimated execution price (last close) — identical
        for base and CE since both route through the same market-open order.
        """
        pending = self._load_pending()
        pending.setdefault(date_str, {})[symbol] = {
            "side":           side,
            "qty_base":       qty_base,
            "qty_ce":         qty_ce,
            "fill_price":     fill_price,
            "filled_base":    qty_base > 0,
            "filled_ce":      qty_ce > 0,
            "ea":             ea,
            "confidence":     confidence,
            "sample_size":    sample_size,
            "forward_return": None,
        }
        self._save_pending(pending)
        logger.info(
            "[CE_COMPARE] recorded %s %s | qty_base=%d qty_ce=%d ea=%.4f conf=%.2f n=%d",
            date_str, symbol, qty_base, qty_ce, ea, confidence, sample_size,
        )

    def update_fill(self, date_str: str, symbol: str, actual_fill_price: float) -> None:
        """Update fill price after live execution confirmation."""
        pending = self._load_pending()
        if date_str in pending and symbol in pending[date_str]:
            pending[date_str][symbol]["fill_price"] = actual_fill_price
            self._save_pending(pending)
            logger.info("[CE_COMPARE] fill_price updated %s %s → %.1f", date_str, symbol, actual_fill_price)

    def try_fill_forward_returns(self, today: date) -> int:
        """
        For pending entries from days strictly before today, compute:
            forward_return = close_T / fill_price - 1

        No lookahead: today's entries are never touched.
        Completed entries are moved from pending to trades.jsonl.
        Returns number of records filled.
        """
        pending = self._load_pending()
        today_str = today.isoformat()
        prior_dates = [d for d in pending if d < today_str]
        if not prior_dates:
            return 0

        all_syms = list({sym for d in prior_dates for sym in pending[d]})
        prices   = self._fetch_close_prices_before(all_syms, today)
        filled   = 0

        for date_str in prior_dates:
            for symbol, entry in pending[date_str].items():
                close = prices.get(symbol)
                fp    = entry["fill_price"]
                if close and close > 0 and fp > 0:
                    fwd = close / fp - 1.0
                    ea  = entry["ea"]
                    sign_match = (
                        (ea > 0 and fwd > 0)
                        or (ea < 0 and fwd < 0)
                        or abs(ea) < 1e-8
                    )
                    qty_b = entry["qty_base"]
                    qty_c = entry["qty_ce"]
                    qty_ratio = round(qty_c / qty_b, 4) if qty_b > 0 else None
                    rec = TradeRecord(
                        date            = date_str,
                        symbol          = symbol,
                        side            = entry["side"],
                        qty_base        = qty_b,
                        qty_ce          = qty_c,
                        fill_price_base = fp,
                        fill_price_ce   = fp,
                        filled_base     = entry["filled_base"],
                        filled_ce       = entry["filled_ce"],
                        ea              = ea,
                        confidence      = entry["confidence"],
                        sample_size     = entry["sample_size"],
                        forward_return  = round(fwd, 6),
                        sign_match      = sign_match,
                        qty_ratio       = qty_ratio,
                    )
                    self._append_trade(rec)
                    filled += 1
                    logger.info(
                        "[CE_COMPARE] fwd filled %s %s | fwd=%.4f sign_match=%s ea=%.4f",
                        date_str, symbol, fwd, sign_match, ea,
                    )
                    print(
                        f"  [CE_COMPARE] {date_str} {symbol}: fwd={fwd:+.4f}"
                        f" sign_match={sign_match} ea={ea:.4f}"
                    )
                else:
                    logger.warning(
                        "[CE_COMPARE] no price for %s %s (close=%s fp=%.1f) — skipping",
                        date_str, symbol, close, fp,
                    )
            del pending[date_str]

        self._save_pending(pending)
        return filled

    def flush_daily_csv(self, today: date) -> None:
        """
        Compute daily metrics from completed trades (forward_return filled) and
        append one CSV row per date.  Skips dates already in the CSV (idempotent).
        Only processes dates strictly before today.
        """
        trades = self._load_trades()
        if not trades:
            return

        today_str    = today.isoformat()
        done_dates   = self._existing_csv_dates()
        cum_delta    = self._last_cum_delta()

        # Group completed trades by date
        by_date: dict[str, list[TradeRecord]] = {}
        for t in trades:
            if t.forward_return is not None and t.date < today_str:
                by_date.setdefault(t.date, []).append(t)

        rows = []
        for date_str in sorted(by_date):
            if date_str in done_dates:
                continue
            day = by_date[date_str]

            pnl_base = sum(
                t.qty_base * t.fill_price_base * (t.forward_return or 0.0)
                for t in day if t.filled_base
            )
            pnl_ce = sum(
                t.qty_ce * t.fill_price_ce * (t.forward_return or 0.0)
                for t in day if t.filled_ce
            )
            delta_pnl  = pnl_ce - pnl_base
            cum_delta += delta_pnl

            n              = len(day)
            fill_base      = sum(1 for t in day if t.filled_base) / max(1, n)
            fill_ce        = sum(1 for t in day if t.filled_ce)   / max(1, n)
            fill_diff      = fill_ce - fill_base

            sm_vals        = [t.sign_match for t in day if t.sign_match is not None]
            sign_match_rate = round(sum(sm_vals) / len(sm_vals), 4) if sm_vals else ""

            qr_vals        = [t.qty_ratio for t in day if t.qty_ratio is not None]
            mean_qty_ratio = round(sum(qr_vals) / len(qr_vals), 4) if qr_vals else ""

            rows.append({
                "date":           date_str,
                "pnl_base":       round(pnl_base, 0),
                "pnl_ce":         round(pnl_ce, 0),
                "delta_pnl":      round(delta_pnl, 0),
                "cum_delta":      round(cum_delta, 0),
                "fill_rate_base": round(fill_base, 4),
                "fill_rate_ce":   round(fill_ce, 4),
                "fill_rate_diff": round(fill_diff, 4),
                "n_trades":       n,
                "sign_match_rate": sign_match_rate,
                "mean_qty_ratio":  mean_qty_ratio,
            })

        if not rows:
            return

        with open(self._daily_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_DAILY_COLS)
            for row in rows:
                writer.writerow(row)

        logger.info("[CE_COMPARE] daily CSV: %d row(s) appended → %s", len(rows), self._daily_csv)
        print(f"  [CE_COMPARE] daily CSV: {len(rows)} row(s) → {self._daily_csv.name}")
        for row in rows:
            print(
                f"    {row['date']}: pnl_base={row['pnl_base']:>8}  pnl_ce={row['pnl_ce']:>8}"
                f"  delta={row['delta_pnl']:>8}  cum_delta={row['cum_delta']:>10}"
                f"  sign_match={row['sign_match_rate']}  qty_ratio={row['mean_qty_ratio']}"
            )

    # ------------------------------------------------------------------ #
    # price fetch helper
    # ------------------------------------------------------------------ #

    def _fetch_close_prices_before(
        self, symbols: list[str], reference_date: date
    ) -> dict[str, float]:
        """
        Fetch the most recent close price for each symbol strictly before reference_date.
        Uses yfinance with a 5-day lookback — same dependency as the signal bridge.
        """
        if not symbols:
            return {}
        try:
            import yfinance as yf
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    symbols, period="5d", progress=False, auto_adjust=True, group_by="ticker"
                )
            ref_str = reference_date.isoformat()
            result: dict[str, float] = {}
            for sym in symbols:
                try:
                    # yfinance group_by="ticker" → MultiIndex (ticker, field)
                    if hasattr(raw.columns, "levels") and sym in raw.columns.get_level_values(0):
                        col = raw[sym]["Close"].dropna()
                    elif sym in raw.columns:
                        col = raw[sym].dropna()
                    else:
                        col = raw["Close"].dropna()  # single-symbol flat
                    col = col[col.index.strftime("%Y-%m-%d") < ref_str]
                    if not col.empty:
                        result[sym] = float(col.iloc[-1])
                except Exception as exc:
                    logger.debug("[CE_COMPARE] price lookup %s: %s", sym, exc)
            logger.info("[CE_COMPARE] fetched %d/%d close prices", len(result), len(symbols))
            return result
        except Exception as exc:
            logger.warning("[CE_COMPARE] yfinance batch fetch failed: %s", exc)
            return {}
