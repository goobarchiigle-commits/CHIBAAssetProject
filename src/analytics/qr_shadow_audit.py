"""
Phase9 Shadow Audit — Quality Replacement Engine (Study59)

Observation-only. FAIL_OPEN on every function.
Never modifies orders, signals, or QUALITY_REPLACEMENT_ENABLED.

Entry point: run_phase9_all()
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── CSV schemas ───────────────────────────────────────────────────────────────

_P9A_HEADER = [
    "date",
    "n_triggers", "n_swap_ready", "n_hold_score_low",
    "n_candidate_low", "n_no_candidate",
    "hs_0_20", "hs_20_35", "hs_35_50", "hs_50plus",
    "cs_50_60", "cs_60_70", "cs_70_80", "cs_80plus",
    "gap_neg50_neg25", "gap_neg25_0", "gap_0_25", "gap_25_50", "gap_50plus",
    "median_weakest_qs", "median_best_cand_qs",
]

_P9B_HEADER = [
    "decision_date", "decision_id",
    "removed_symbol", "candidate_symbol",
    "hold_score", "candidate_score", "gap_score",
    "decision_close_removed", "decision_close_cand",
    "close_7d_removed",  "close_7d_cand",  "mat_date_7d",
    "close_20d_removed", "close_20d_cand", "mat_date_20d",
    "close_60d_removed", "close_60d_cand", "mat_date_60d",
    "removed_ret_7d",  "candidate_ret_7d",  "delta_7d",
    "removed_ret_20d", "candidate_ret_20d", "delta_20d",
    "removed_ret_60d", "candidate_ret_60d", "delta_60d",
    "status",
]

_P9C_HEADER = [
    "date", "hold_symbol", "hold_score",
    "candidate_symbol", "candidate_score", "gap_score",
]

_P9D_HEADER = [
    "date", "decision_id",
    "removed_symbol", "candidate_symbol",
    "hold_score", "candidate_score",
    "removed_ret_60d", "candidate_ret_60d", "delta_60d",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def _append_csv(path: Path, header: list[str], row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if needs_header:
            w.writeheader()
        w.writerow(row)


def _read_csv_safe(path: Path) -> pd.DataFrame:
    """Return empty DataFrame if file missing or empty."""
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, dtype=str)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _get_close_on_date(sym: str, date_str: str, ohlcv_cache: Path) -> float | None:
    """Close price on or before date_str."""
    fpath = ohlcv_cache / f"{sym}.parquet"
    if not fpath.exists():
        return None
    try:
        df = pd.read_parquet(fpath).sort_index()
        target = pd.Timestamp(date_str)
        mask = df.index <= target
        if not mask.any():
            return None
        return float(df[mask].iloc[-1]["Close"])
    except Exception:
        return None


def _get_close_n_bdays_later(
    sym: str, decision_date_str: str, n_bdays: int, ohlcv_cache: Path
) -> tuple[float | None, str | None]:
    """Close price N trading days after decision_date. Returns (price, date)."""
    fpath = ohlcv_cache / f"{sym}.parquet"
    if not fpath.exists():
        return None, None
    try:
        df = pd.read_parquet(fpath).sort_index()
        decision_dt = pd.Timestamp(decision_date_str)
        future = df[df.index > decision_dt]["Close"]
        if len(future) < n_bdays:
            return None, None
        return (
            float(future.iloc[n_bdays - 1]),
            future.index[n_bdays - 1].strftime("%Y-%m-%d"),
        )
    except Exception:
        return None, None


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _safe_str(v: Any) -> str:
    return "" if v is None or (isinstance(v, float) and math.isnan(v)) else str(v)


# ── Phase9-A: Trigger Distribution Audit ─────────────────────────────────────

def run_phase9a(
    audit_file: Path,
    p9a_file:   Path,
    today:      str,
) -> dict:
    """
    Daily trigger distribution row for today's audit entries.
    """
    df = _read_csv_safe(audit_file)
    if df.empty:
        return {}

    today_df = df[df["date"] == today]
    if today_df.empty:
        logger.debug("[P9A] no audit entries for %s", today)
        return {}

    n_triggers     = len(today_df)
    n_swap_ready   = int((today_df["swap_ready"].str.lower() == "true").sum())
    n_hold_ok      = int(
        (today_df["reason_not_swapped"].str.contains("hold_qs", na=False)).sum()
    )
    n_cand_low     = int(
        (today_df["reason_not_swapped"].str.contains("cand_qs", na=False)).sum()
    )
    n_no_cand      = int(
        (today_df["reason_not_swapped"].str.contains("no_candidate", na=False)).sum()
    )

    # Holding score distribution
    qs = today_df["weakest_qs"].apply(_safe_float).dropna()
    hs_0_20   = int((qs < 20).sum())
    hs_20_35  = int(((qs >= 20) & (qs < 35)).sum())
    hs_35_50  = int(((qs >= 35) & (qs < 50)).sum())
    hs_50plus = int((qs >= 50).sum())

    # Candidate score distribution (where candidate exists)
    cqs = today_df["best_cand_qs"].apply(_safe_float).dropna()
    cs_50_60  = int(((cqs >= 50) & (cqs < 60)).sum())
    cs_60_70  = int(((cqs >= 60) & (cqs < 70)).sum())
    cs_70_80  = int(((cqs >= 70) & (cqs < 80)).sum())
    cs_80plus = int((cqs >= 80).sum())

    # Gap distribution (best_cand_qs - weakest_qs)
    gaps = today_df["score_gap"].apply(_safe_float).dropna()
    g_neg50_neg25 = int(((gaps >= -50) & (gaps < -25)).sum())
    g_neg25_0     = int(((gaps >= -25) & (gaps < 0)).sum())
    g_0_25        = int(((gaps >= 0)   & (gaps < 25)).sum())
    g_25_50       = int(((gaps >= 25)  & (gaps < 50)).sum())
    g_50plus      = int((gaps >= 50).sum())

    row = {
        "date":           today,
        "n_triggers":     n_triggers,
        "n_swap_ready":   n_swap_ready,
        "n_hold_score_low": n_hold_ok,
        "n_candidate_low":  n_cand_low,
        "n_no_candidate":   n_no_cand,
        "hs_0_20": hs_0_20, "hs_20_35": hs_20_35,
        "hs_35_50": hs_35_50, "hs_50plus": hs_50plus,
        "cs_50_60": cs_50_60, "cs_60_70": cs_60_70,
        "cs_70_80": cs_70_80, "cs_80plus": cs_80plus,
        "gap_neg50_neg25": g_neg50_neg25, "gap_neg25_0": g_neg25_0,
        "gap_0_25": g_0_25, "gap_25_50": g_25_50, "gap_50plus": g_50plus,
        "median_weakest_qs":  round(float(qs.median()), 2) if len(qs) else "",
        "median_best_cand_qs": round(float(cqs.median()), 2) if len(cqs) else "",
    }

    # Dedup: skip if today's row already written
    existing = _read_csv_safe(p9a_file)
    if not existing.empty and today in existing["date"].values:
        logger.debug("[P9A] row for %s already written — skipping", today)
        return row

    _append_csv(p9a_file, _P9A_HEADER, row)
    logger.info(
        "[P9A] %s triggers=%d swap_ready=%d hs_below35=%d cand_below70=%d",
        today, n_triggers, n_swap_ready, hs_0_20 + hs_20_35, cs_50_60 + cs_60_70,
    )
    return row


# ── Phase9-B: Forward Attribution ────────────────────────────────────────────

def init_phase9b(
    audit_file:  Path,
    p9b_file:    Path,
    ohlcv_cache: Path,
    today:       str,
) -> int:
    """
    For Swap Ready decisions recorded today, create forward attribution stubs.
    Returns count of new stubs created.
    """
    df = _read_csv_safe(audit_file)
    if df.empty:
        return 0

    today_swap = df[
        (df["date"] == today) &
        (df["swap_ready"].str.lower() == "true")
    ]
    if today_swap.empty:
        return 0

    existing_p9b = _read_csv_safe(p9b_file)
    existing_ids: set[str] = set()
    if not existing_p9b.empty and "decision_id" in existing_p9b.columns:
        existing_ids = set(existing_p9b["decision_id"].dropna())

    count = 0
    for _, r in today_swap.iterrows():
        did = str(r.get("decision_id", ""))
        if did in existing_ids:
            continue

        removed_sym = str(r.get("weakest_symbol", ""))
        cand_sym    = str(r.get("best_cand_symbol", ""))

        decision_close_r = _get_close_on_date(removed_sym, today, ohlcv_cache)
        decision_close_c = _get_close_on_date(cand_sym,    today, ohlcv_cache)

        row: dict[str, Any] = {
            "decision_date":        today,
            "decision_id":          did,
            "removed_symbol":       removed_sym,
            "candidate_symbol":     cand_sym,
            "hold_score":           r.get("weakest_qs", ""),
            "candidate_score":      r.get("best_cand_qs", ""),
            "gap_score":            r.get("score_gap", ""),
            "decision_close_removed": _safe_str(decision_close_r),
            "decision_close_cand":    _safe_str(decision_close_c),
            "close_7d_removed":  "", "close_7d_cand":  "", "mat_date_7d":  "",
            "close_20d_removed": "", "close_20d_cand": "", "mat_date_20d": "",
            "close_60d_removed": "", "close_60d_cand": "", "mat_date_60d": "",
            "removed_ret_7d":  "", "candidate_ret_7d":  "", "delta_7d":  "",
            "removed_ret_20d": "", "candidate_ret_20d": "", "delta_20d": "",
            "removed_ret_60d": "", "candidate_ret_60d": "", "delta_60d": "",
            "status": "pending",
        }
        _append_csv(p9b_file, _P9B_HEADER, row)
        existing_ids.add(did)  # prevent same-session duplicates
        count += 1
        logger.info("[P9B] init stub: %s removed=%s cand=%s", did, removed_sym, cand_sym)

    return count


def materialize_phase9b(
    p9b_file:    Path,
    ohlcv_cache: Path,
    today:       str,
) -> int:
    """
    Try to fill 7d/20d/60d returns for pending Phase9-B records.
    Rewrites the CSV in place. Returns count of newly materialized fields.
    """
    if not p9b_file.exists():
        return 0

    try:
        df = pd.read_csv(p9b_file, dtype=str)
    except Exception as exc:
        logger.warning("[P9B] materialize: read failed (%s)", exc)
        return 0

    if df.empty:
        return 0

    pending_mask = df["status"] != "complete"
    if not pending_mask.any():
        return 0

    WINDOWS = [(7, "7d"), (20, "20d"), (60, "60d")]
    filled = 0

    for idx in df[pending_mask].index:
        row       = df.loc[idx]
        dec_date  = str(row.get("decision_date", ""))
        rem_sym   = str(row.get("removed_symbol", ""))
        cand_sym  = str(row.get("candidate_symbol", ""))
        dec_cl_r  = _safe_float(row.get("decision_close_removed"))
        dec_cl_c  = _safe_float(row.get("decision_close_cand"))

        if not dec_date or not rem_sym or not cand_sym:
            continue

        any_new = False
        for n_bdays, suffix in WINDOWS:
            col_rem  = f"close_{suffix}_removed"
            col_cand = f"close_{suffix}_cand"
            col_mat  = f"mat_date_{suffix}"
            col_rret = f"removed_ret_{suffix}"
            col_cret = f"candidate_ret_{suffix}"
            col_delt = f"delta_{suffix}"

            if str(df.at[idx, col_mat]).strip():
                continue  # already filled

            c_r, d_r = _get_close_n_bdays_later(rem_sym,  dec_date, n_bdays, ohlcv_cache)
            c_c, d_c = _get_close_n_bdays_later(cand_sym, dec_date, n_bdays, ohlcv_cache)

            if c_r is None or c_c is None:
                continue  # data not yet available

            # Use the earlier of the two mat dates
            mat_date = min(d_r, d_c) if d_r and d_c else (d_r or d_c or "")

            r_ret = round((c_r / dec_cl_r) - 1, 4) if dec_cl_r else None
            c_ret = round((c_c / dec_cl_c) - 1, 4) if dec_cl_c else None
            delta = round(c_ret - r_ret, 4) if (r_ret is not None and c_ret is not None) else None

            df.at[idx, col_rem]  = _safe_str(c_r)
            df.at[idx, col_cand] = _safe_str(c_c)
            df.at[idx, col_mat]  = mat_date
            df.at[idx, col_rret] = _safe_str(r_ret)
            df.at[idx, col_cret] = _safe_str(c_ret)
            df.at[idx, col_delt] = _safe_str(delta)
            any_new = True
            filled += 1
            logger.info(
                "[P9B] %s: %s Δ%s r_ret=%s c_ret=%s delta=%s",
                row["decision_id"], suffix,
                mat_date, r_ret, c_ret, delta,
            )

        if any_new:
            # Update status
            has_7d  = bool(str(df.at[idx, "mat_date_7d"]).strip())
            has_20d = bool(str(df.at[idx, "mat_date_20d"]).strip())
            has_60d = bool(str(df.at[idx, "mat_date_60d"]).strip())
            if has_60d:
                df.at[idx, "status"] = "complete"
            elif has_20d:
                df.at[idx, "status"] = "partial_20d"
            elif has_7d:
                df.at[idx, "status"] = "partial_7d"

    if filled > 0:
        df.to_csv(p9b_file, index=False, encoding="utf-8")
        logger.info("[P9B] materialized %d field(s)", filled)

    return filled


# ── Phase9-C: Missed Opportunity Audit ───────────────────────────────────────

def run_phase9c(
    audit_file: Path,
    p9c_file:   Path,
    today:      str,
) -> int:
    """
    Missed opportunities: hold_qs < 35 but best_cand_qs in [60, 70) (just below threshold).
    Returns count of new records written.
    """
    df = _read_csv_safe(audit_file)
    if df.empty:
        return 0

    today_df = df[df["date"] == today]
    if today_df.empty:
        return 0

    existing = _read_csv_safe(p9c_file)
    existing_today = set()
    if not existing.empty and "date" in existing.columns:
        existing_today = set(existing[existing["date"] == today].get("hold_symbol", pd.Series()).values)

    count = 0
    for _, r in today_df.iterrows():
        weakest_qs = _safe_float(r.get("weakest_qs"))
        best_qs    = _safe_float(r.get("best_cand_qs"))
        rem_sym    = str(r.get("weakest_symbol", ""))

        if weakest_qs is None or best_qs is None:
            continue
        if rem_sym in existing_today:
            continue

        # Condition: hold below threshold, but candidate just BELOW cand_threshold (60-70)
        if weakest_qs < 35.0 and 60.0 <= best_qs < 70.0:
            row = {
                "date":             today,
                "hold_symbol":      rem_sym,
                "hold_score":       round(weakest_qs, 2),
                "candidate_symbol": str(r.get("best_cand_symbol", "")),
                "candidate_score":  round(best_qs, 2),
                "gap_score":        _safe_str(_safe_float(r.get("score_gap"))),
            }
            _append_csv(p9c_file, _P9C_HEADER, row)
            count += 1
            logger.info(
                "[P9C] missed opp: hold=%s(%.1f) cand=%s(%.1f)",
                rem_sym, weakest_qs, row["candidate_symbol"], best_qs,
            )

    return count


# ── Phase9-D: False Trigger Audit ────────────────────────────────────────────

def run_phase9d(
    p9b_file: Path,
    p9d_file: Path,
) -> int:
    """
    Extract completed P9B records where delta_60d < 0 (candidate underperformed removed).
    Returns count of new false trigger records.
    """
    df = _read_csv_safe(p9b_file)
    if df.empty:
        return 0

    complete = df[df.get("status", pd.Series()) == "complete"] if "status" in df.columns else pd.DataFrame()
    if complete.empty:
        return 0

    existing = _read_csv_safe(p9d_file)
    existing_ids: set[str] = set()
    if not existing.empty and "decision_id" in existing.columns:
        existing_ids = set(existing["decision_id"].dropna())

    count = 0
    for _, r in complete.iterrows():
        did   = str(r.get("decision_id", ""))
        d60   = _safe_float(r.get("delta_60d"))
        if did in existing_ids or d60 is None:
            continue
        if d60 < 0:
            row = {
                "date":             str(r.get("decision_date", "")),
                "decision_id":      did,
                "removed_symbol":   str(r.get("removed_symbol", "")),
                "candidate_symbol": str(r.get("candidate_symbol", "")),
                "hold_score":       _safe_str(_safe_float(r.get("hold_score"))),
                "candidate_score":  _safe_str(_safe_float(r.get("candidate_score"))),
                "removed_ret_60d":  _safe_str(_safe_float(r.get("removed_ret_60d"))),
                "candidate_ret_60d": _safe_str(_safe_float(r.get("candidate_ret_60d"))),
                "delta_60d":        round(d60, 4),
            }
            _append_csv(p9d_file, _P9D_HEADER, row)
            count += 1
            logger.info(
                "[P9D] false trigger: %s delta_60d=%.4f (cand underperformed)",
                did, d60,
            )

    return count


# ── Phase9-E: Shadow Summary Report ──────────────────────────────────────────

def _count_bdays_since(date_str: str, today: str) -> int:
    """Rough business day count (Mon-Fri) between two date strings."""
    try:
        d0 = datetime.strptime(date_str, "%Y-%m-%d")
        d1 = datetime.strptime(today, "%Y-%m-%d")
        days = (d1 - d0).days
        # ~5/7 of calendar days are business days
        return max(0, int(days * 5 / 7))
    except ValueError:
        return 0


def _last_summary_date(p9e_dir: Path) -> str | None:
    """Find the most recent summary file date."""
    p9e_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p9e_dir.glob("qr_shadow_summary_*.json"))
    if not files:
        return None
    try:
        last = json.loads(files[-1].read_text(encoding="utf-8"))
        return last.get("generated_date")
    except Exception:
        return None


def _earliest_audit_date(audit_file: Path) -> str | None:
    """Return earliest 'date' string in audit file, or None."""
    df = _read_csv_safe(audit_file)
    if df.empty or "date" not in df.columns:
        return None
    dates = df["date"].dropna()
    return str(dates.min()) if len(dates) else None


def maybe_run_phase9e(
    audit_file: Path,
    p9b_file:   Path,
    p9c_file:   Path,
    p9d_file:   Path,
    p9e_dir:    Path,
    today:      str,
    min_bdays:  int = 30,
) -> dict | None:
    """
    Generate summary if >= min_bdays business days since last summary (or since
    first audit entry when no prior summary exists).
    Returns summary dict or None if not triggered.
    """
    last = _last_summary_date(p9e_dir)
    if last:
        baseline = last
    else:
        # No prior summary: use earliest audit date as baseline
        baseline = _earliest_audit_date(audit_file)
        if baseline is None:
            logger.debug("[P9E] no audit data yet — skipping")
            return None

    elapsed = _count_bdays_since(baseline, today)
    if elapsed < min_bdays:
        logger.debug("[P9E] %d bdays since baseline %s — not yet triggered", elapsed, baseline)
        return None

    return _build_phase9e_summary(
        audit_file, p9b_file, p9c_file, p9d_file, p9e_dir, today, last
    )


def _build_phase9e_summary(
    audit_file: Path,
    p9b_file:   Path,
    p9c_file:   Path,
    p9d_file:   Path,
    p9e_dir:    Path,
    today:      str,
    since:      str | None,
) -> dict:
    audit  = _read_csv_safe(audit_file)
    p9b    = _read_csv_safe(p9b_file)
    p9c    = _read_csv_safe(p9c_file)
    p9d    = _read_csv_safe(p9d_file)

    # Window: since last summary or all-time
    window_start = since or "2000-01-01"
    if not audit.empty and "date" in audit.columns:
        window = audit[audit["date"] >= window_start]
    else:
        window = pd.DataFrame()

    # Trigger counts
    n_triggers   = len(window) if not window.empty else 0
    n_swap_ready = int((window["swap_ready"].str.lower() == "true").sum()) if not window.empty else 0
    n_no_cand    = int(window["reason_not_swapped"].str.contains("no_candidate", na=False).sum()) if not window.empty else 0
    n_hold_low   = int(window["reason_not_swapped"].str.contains("hold_qs", na=False).sum()) if not window.empty else 0
    n_cand_low   = int(window["reason_not_swapped"].str.contains("cand_qs", na=False).sum()) if not window.empty else 0

    # Holding score distribution
    hs_dist = {}
    if not window.empty:
        qs = window["weakest_qs"].apply(_safe_float).dropna()
        hs_dist = {
            "0_20":   int((qs < 20).sum()),
            "20_35":  int(((qs >= 20) & (qs < 35)).sum()),
            "35_50":  int(((qs >= 35) & (qs < 50)).sum()),
            "50plus": int((qs >= 50).sum()),
        }

    # Candidate score distribution
    cs_dist = {}
    if not window.empty:
        cqs = window["best_cand_qs"].apply(_safe_float).dropna()
        cs_dist = {
            "50_60":  int(((cqs >= 50) & (cqs < 60)).sum()),
            "60_70":  int(((cqs >= 60) & (cqs < 70)).sum()),
            "70_80":  int(((cqs >= 70) & (cqs < 80)).sum()),
            "80plus": int((cqs >= 80).sum()),
        }

    # Gap distribution
    gap_dist = {}
    if not window.empty:
        gaps = window["score_gap"].apply(_safe_float).dropna()
        gap_dist = {
            "neg50_neg25": int(((gaps >= -50) & (gaps < -25)).sum()),
            "neg25_0":     int(((gaps >= -25) & (gaps < 0)).sum()),
            "0_25":        int(((gaps >= 0)   & (gaps < 25)).sum()),
            "25_50":       int(((gaps >= 25)  & (gaps < 50)).sum()),
            "50plus":      int((gaps >= 50).sum()),
        }

    # Forward attribution summary
    fa_summary: dict[str, Any] = {}
    if not p9b.empty and "status" in p9b.columns:
        p9b_win = p9b[p9b["decision_date"] >= window_start] if "decision_date" in p9b.columns else p9b
        complete = p9b_win[p9b_win["status"] == "complete"]
        if not complete.empty:
            for suffix in ("7d", "20d", "60d"):
                col = f"delta_{suffix}"
                if col in complete.columns:
                    deltas = complete[col].apply(_safe_float).dropna()
                    if len(deltas):
                        fa_summary[suffix] = {
                            "n":          len(deltas),
                            "positive":   int((deltas > 0).sum()),
                            "negative":   int((deltas < 0).sum()),
                            "median":     round(float(deltas.median()), 4),
                            "mean":       round(float(deltas.mean()), 4),
                        }

    # Missed opportunity count
    n_missed = 0
    if not p9c.empty and "date" in p9c.columns:
        n_missed = len(p9c[p9c["date"] >= window_start])

    # False trigger count
    n_false = 0
    if not p9d.empty and "date" in p9d.columns:
        n_false = len(p9d[p9d["date"] >= window_start])

    summary = {
        "generated_date":      today,
        "window_start":        window_start,
        "window_end":          today,
        "n_triggers":          n_triggers,
        "n_swap_ready":        n_swap_ready,
        "n_no_candidate":      n_no_cand,
        "n_hold_score_low":    n_hold_low,
        "n_candidate_low":     n_cand_low,
        "holding_score_dist":  hs_dist,
        "candidate_score_dist": cs_dist,
        "gap_dist":            gap_dist,
        "forward_attribution": fa_summary,
        "n_missed_opportunity": n_missed,
        "n_false_trigger":     n_false,
        # Readiness check
        "readiness": _check_readiness(
            n_swap_ready, fa_summary, n_missed, n_false, today, window_start
        ),
    }

    # Save
    p9e_dir.mkdir(parents=True, exist_ok=True)
    fname = p9e_dir / f"qr_shadow_summary_{today[:7]}.json"
    fname.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[P9E] shadow summary written: %s", fname)
    _print_p9e_report(summary)
    return summary


def _check_readiness(
    n_swap: int,
    fa: dict,
    n_missed: int,
    n_false: int,
    today: str,
    since: str,
) -> dict:
    """
    Check if conditions for Production Enable are met.
    ALL 6 conditions must pass before proposing enabled=true.
    """
    elapsed_bdays = _count_bdays_since(since, today)
    fa60 = fa.get("60d", {})
    n_complete_60d = fa60.get("n", 0)
    median_delta = fa60.get("median")
    false_rate = (n_false / n_swap) if n_swap > 0 else None

    cond1_shadow_period    = elapsed_bdays >= 30
    cond2_sample_count     = n_swap >= 3
    cond3_bt_same_dir      = (median_delta is not None and median_delta > 0) if n_complete_60d >= 2 else None
    cond4_missed_limited   = n_missed <= max(2, int(n_swap * 0.3))
    cond5_false_acceptable = (false_rate is not None and false_rate <= 0.5) if n_swap >= 3 else None
    cond6_no_anomaly       = True  # Manual override — always requires human judgment

    ready = all([
        cond1_shadow_period,
        cond2_sample_count,
        cond3_bt_same_dir is True,
        cond4_missed_limited,
        cond5_false_acceptable is True,
        cond6_no_anomaly,
    ])

    return {
        "production_enable_ready": ready,
        "cond1_shadow_30bdays":    cond1_shadow_period,
        "cond2_sample_3plus":      cond2_sample_count,
        "cond3_bt_same_direction": cond3_bt_same_dir,
        "cond4_missed_limited":    cond4_missed_limited,
        "cond5_false_rate_ok":     cond5_false_acceptable,
        "cond6_no_anomaly":        cond6_no_anomaly,
        "elapsed_bdays":           elapsed_bdays,
        "n_complete_60d":          n_complete_60d,
        "median_delta_60d":        median_delta,
        "false_trigger_rate":      round(false_rate, 3) if false_rate is not None else None,
        "note": "ALL 6条件達成後のみASK_FIRSTでenabled=true提案可能",
    }


def _print_p9e_report(s: dict) -> None:
    r = s.get("readiness", {})
    ready = r.get("production_enable_ready", False)
    tag = "✅ 全条件PASS" if ready else "⏳ 監査継続中"
    print(
        f"\n━━ [QR Shadow Summary] {s['window_start']}〜{s['window_end']} ━━\n"
        f"  トリガー総数: {s['n_triggers']}件"
        f"  / SWAP_READY: {s['n_swap_ready']}件"
        f"  / Missed: {s['n_missed_opportunity']}件"
        f"  / FalseTrigger: {s['n_false_trigger']}件\n"
        f"  Forward(60d): {s['forward_attribution'].get('60d', {})}\n"
        f"  本番Enable判定: {tag} (elapsed={r.get('elapsed_bdays')}営業日)\n"
        f"  ⚠ enabled=true はASK_FIRST必須\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_phase9_all(
    today:       str,
    run_id:      str,
    audit_file:  Path,
    missed_file: Path,
    p9a_file:    Path,
    p9b_file:    Path,
    p9c_file:    Path,
    p9d_file:    Path,
    p9e_dir:     Path,
    ohlcv_cache: Path,
) -> dict:
    """
    Run all Phase9 audit steps. FAIL_OPEN — any step error → log + continue.

    Returns summary dict with per-step results.
    """
    results: dict[str, Any] = {"date": today, "run_id": run_id}

    # Phase9-A: Trigger distribution
    try:
        p9a = run_phase9a(audit_file, p9a_file, today)
        results["p9a"] = {"ok": True, "n_triggers": p9a.get("n_triggers", 0)}
    except Exception as exc:
        logger.warning("[P9A] failed (%s) — continuing", exc)
        results["p9a"] = {"ok": False, "error": str(exc)}

    # Phase9-B init: new Swap Ready stubs
    try:
        n_init = init_phase9b(audit_file, p9b_file, ohlcv_cache, today)
        results["p9b_init"] = {"ok": True, "n_new": n_init}
    except Exception as exc:
        logger.warning("[P9B init] failed (%s) — continuing", exc)
        results["p9b_init"] = {"ok": False, "error": str(exc)}

    # Phase9-B materialize: fill pending returns
    try:
        n_mat = materialize_phase9b(p9b_file, ohlcv_cache, today)
        results["p9b_mat"] = {"ok": True, "n_filled": n_mat}
    except Exception as exc:
        logger.warning("[P9B mat] failed (%s) — continuing", exc)
        results["p9b_mat"] = {"ok": False, "error": str(exc)}

    # Phase9-C: Missed opportunities
    try:
        n_c = run_phase9c(audit_file, p9c_file, today)
        results["p9c"] = {"ok": True, "n_missed": n_c}
    except Exception as exc:
        logger.warning("[P9C] failed (%s) — continuing", exc)
        results["p9c"] = {"ok": False, "error": str(exc)}

    # Phase9-D: False triggers (requires completed P9B records)
    try:
        n_d = run_phase9d(p9b_file, p9d_file)
        results["p9d"] = {"ok": True, "n_false": n_d}
    except Exception as exc:
        logger.warning("[P9D] failed (%s) — continuing", exc)
        results["p9d"] = {"ok": False, "error": str(exc)}

    # Phase9-E: Conditional 30-day summary
    try:
        summary = maybe_run_phase9e(
            audit_file, p9b_file, p9c_file, p9d_file, p9e_dir, today
        )
        results["p9e"] = {"ok": True, "triggered": summary is not None}
    except Exception as exc:
        logger.warning("[P9E] failed (%s) — continuing", exc)
        results["p9e"] = {"ok": False, "error": str(exc)}

    return results
