"""
src/analytics/future_leader_regime.py — Future Leader Regime Segmentation

Purpose:
  Attach market regime tags to Future Leader observations and compute
  regime-conditioned alpha persistence metrics. Answers: "in which market
  environment does institutional drift survivability collapse?"

INVARIANTS (変更禁止):
  - observation_only=True: no execution path connection
  - TOPIX OHLCV only: no VIX, macro data, or external signals
  - lookahead prevention: TOPIX bars strictly <= observation_date for regime classification
  - CLEAN-only aggregation: data_quality_weight >= 1.0, insufficient_forward_data=False
  - PARTIAL_FAILURE: persisted by prior modules; excluded from aggregation here
  - append-only regime JSONL: observation_date once tagged, never re-classified

禁止:
  - order generation / signal mutation / allocator linkage / broker linkage
  - deployable universe mutation / predictive_entry_enabled mutation
  - VIX / macro data / ML clustering
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

# ── Regime classification parameters ─────────────────────────────────────────
TOPIX_RETURN_WINDOW: int   = 10     # business days for TOPIX trend return
TOPIX_TREND_THRESH:  float = 0.03   # +/- 3% boundary: risk_on / risk_off
ATR_WINDOW:          int   = 14     # bars for ATR smoothing
ATR_HIST_WINDOW:     int   = 252    # historical ATR reference window (~1 year)
ATR_VOLATILE_PCTILE: float = 0.75   # top-quartile volatility → volatile regime

_CLEAN_QUALITY_MIN: float = 1.0
_MIN_SAMPLE:        int   = 5
_REGIME_ORDER: List[str] = ["risk_on", "risk_off", "range", "volatile"]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RegimeExpectancySummary:
    """
    Regime-conditioned alpha persistence metrics. observation_only=True.
    """
    regime:           str
    sample_count:     int
    day5_expectancy:  Optional[float]
    day10_expectancy: Optional[float]
    median_survival:  Optional[float]
    half_life_day:    Optional[int]
    median_mfe:       Optional[float]
    median_mae:       Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# ATR percentile  (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_atr_percentile(df: pd.DataFrame) -> Optional[float]:
    """
    Rolling ATR_WINDOW-day ATR percentile vs ATR_HIST_WINDOW-day reference.

    Computes ATR = mean(H-L) for each of (ATR_HIST_WINDOW + 1) overlapping windows.
    Percentile = fraction(historical ATRs <= current ATR).
    Returns None if insufficient bars (< ATR_WINDOW + ATR_HIST_WINDOW).
    """
    needed = ATR_WINDOW + ATR_HIST_WINDOW
    if len(df) < needed:
        return None
    try:
        hl = (df["High"] - df["Low"]).values[-needed:].astype(float)
        # ATR_HIST_WINDOW + 1 rolling windows: index 0..ATR_HIST_WINDOW
        # last window = most recent ATR
        n_windows = ATR_HIST_WINDOW + 1
        atr_vals = np.array([
            np.nanmean(hl[i: i + ATR_WINDOW])
            for i in range(n_windows)
        ])
        current_atr = atr_vals[-1]
        historical  = atr_vals[:-1]
        valid = historical[np.isfinite(historical)]
        if len(valid) == 0:
            return None
        return round(float(np.mean(valid <= current_atr)), 4)
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Regime classification  (pure)
# ─────────────────────────────────────────────────────────────────────────────

def classify_regime(
    topix_df: pd.DataFrame,
    obs_date_str: str,
) -> Tuple[str, Optional[float], Optional[float]]:
    """
    Classify market regime at observation_date from TOPIX OHLCV.

    Lookahead prevention: uses only TOPIX bars with index <= obs_date_ts.
    Falls back to ('range', None, None) on insufficient data or any error.

    Priority:
      1. risk_on  — TOPIX 10d return >  +3%
      2. risk_off — TOPIX 10d return <  -3%
      3. volatile — flat trend AND ATR percentile >= 0.75
      4. range    — flat trend AND ATR percentile <  0.75 (or insufficient ATR history)

    Returns:
        (regime_tag, topix_10d_return, atr_percentile)
    """
    ret_10d:    Optional[float] = None
    atr_pctile: Optional[float] = None

    try:
        obs_ts = pd.Timestamp(obs_date_str).normalize()
        df     = topix_df[topix_df.index <= obs_ts]

        if len(df) < TOPIX_RETURN_WINDOW + 1:
            return "range", ret_10d, atr_pctile

        c_now  = float(df["Close"].iloc[-1])
        c_past = float(df["Close"].iloc[-(TOPIX_RETURN_WINDOW + 1)])

        if not (math.isfinite(c_now) and math.isfinite(c_past) and c_past > 0):
            return "range", ret_10d, atr_pctile

        ret_10d = round(c_now / c_past - 1.0, 6)

        if ret_10d > TOPIX_TREND_THRESH:
            return "risk_on", ret_10d, atr_pctile
        if ret_10d < -TOPIX_TREND_THRESH:
            return "risk_off", ret_10d, atr_pctile

        atr_pctile = _compute_atr_percentile(df)
        if atr_pctile is not None and atr_pctile >= ATR_VOLATILE_PCTILE:
            return "volatile", ret_10d, atr_pctile

        return "range", ret_10d, atr_pctile

    except Exception:
        return "range", ret_10d, atr_pctile


def build_regime_lookup(
    obs_dates: List[str],
    topix_df:  pd.DataFrame,
) -> Tuple[Dict[str, str], Dict[str, dict]]:
    """
    Compute regime tag for each unique observation_date.
    Each date uses TOPIX bars <= that date (lookahead-safe).

    Returns:
        regime_map: {observation_date: regime_tag}
        meta_map:   {observation_date: {"topix_10d_return": ..., "atr_percentile": ...}}
    """
    regime_map: Dict[str, str]  = {}
    meta_map:   Dict[str, dict] = {}
    for d in sorted(set(obs_dates)):
        tag, ret, atr = classify_regime(topix_df, d)
        regime_map[d] = tag
        meta_map[d]   = {"topix_10d_return": ret, "atr_percentile": atr}
    return regime_map, meta_map


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers  (separated from computation)
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(log_dir: Path, pattern: str) -> List[dict]:
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob(pattern)):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
    return records


def _load_survivability_records(log_dir: Path) -> List[dict]:
    return _load_jsonl(log_dir, "future_leader_survivability_*.jsonl")


def _load_failure_records(log_dir: Path) -> List[dict]:
    return _load_jsonl(log_dir, "future_leader_failure_*.jsonl")


def _load_regime_tags(regime_log_dir: Path) -> Dict[str, str]:
    """
    Load persisted {observation_date: regime_tag}.
    Last occurrence per date wins (append-only JSONL may accumulate re-runs).
    """
    result: Dict[str, str] = {}
    for rec in _load_jsonl(regime_log_dir, "future_leader_regime_*.jsonl"):
        dt  = rec.get("observation_date", "")
        tag = rec.get("regime_tag", "")
        if dt and tag:
            result[dt] = tag
    return result


def _write_regime_tags(
    new_dates:      List[str],
    regime_map:     Dict[str, str],
    meta_map:       Dict[str, dict],
    regime_log_dir: Path,
    today:          Optional[date] = None,
) -> None:
    """Append newly classified regime tags to today's JSONL. Fail-open."""
    if not new_dates:
        return
    if today is None:
        today = datetime.now(JST).date()
    today_str = today.strftime("%Y%m%d")
    log_path  = regime_log_dir / f"future_leader_regime_{today_str}.jsonl"
    try:
        regime_log_dir.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now(JST).isoformat()
        lines = []
        for obs_date in sorted(new_dates):
            meta = meta_map.get(obs_date, {})
            d = {
                "observation_date":   obs_date,
                "regime_tag":         regime_map.get(obs_date, "range"),
                "materialization_ts": ts,
                **meta,
            }
            lines.append(json.dumps(d, ensure_ascii=False, default=str))
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("[FL_REGIME] wrote %d tags → %s", len(new_dates), log_path)
    except Exception as e:
        logger.warning("[FL_REGIME] tag write failed (fail-open): %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# CLEAN filters  (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_clean_survivability(records: List[dict]) -> List[dict]:
    """CLEAN survivability: dq >= 1.0, insufficient_forward_data=False. Dedup first."""
    seen: Dict[Tuple[str, str], bool] = {}
    out: List[dict] = []
    for r in records:
        key = (r.get("symbol", ""), r.get("observation_date", ""))
        if not (key[0] and key[1]) or key in seen:
            continue
        seen[key] = True
        if (float(r.get("data_quality_weight", 0.0)) >= _CLEAN_QUALITY_MIN
                and not r.get("insufficient_forward_data", True)):
            out.append(r)
    return out


def _filter_clean_failure(records: List[dict]) -> List[dict]:
    """CLEAN failure: dq >= 1.0. Dedup first."""
    seen: Dict[Tuple[str, str], bool] = {}
    out: List[dict] = []
    for r in records:
        key = (r.get("symbol", ""), r.get("observation_date", ""))
        if not (key[0] and key[1]) or key in seen:
            continue
        seen[key] = True
        if float(r.get("data_quality_weight", 0.0)) >= _CLEAN_QUALITY_MIN:
            out.append(r)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Expectancy computation  (pure)
# ─────────────────────────────────────────────────────────────────────────────

def _opt_mean(vals: List[float]) -> Optional[float]:
    return round(float(np.mean(vals)), 6) if len(vals) >= _MIN_SAMPLE else None


def _opt_median(vals: List[float]) -> Optional[float]:
    return round(float(np.median(vals)), 6) if len(vals) >= _MIN_SAMPLE else None


def _half_life_from_expectancy(e5: Optional[float], e10: Optional[float]) -> Optional[int]:
    """
    Half-life day from two expectancy data points.
    First day where expectancy ≤ 50% of peak.
    Returns None when alpha persists through all available days.
    """
    available = [(d, v) for d, v in [(5, e5), (10, e10)] if v is not None]
    if not available:
        return None
    peak = max(v for _, v in available)
    if peak <= 0:
        return None
    threshold = peak * 0.5
    for day, val in available:
        if val <= threshold:
            return day
    return None


def compute_regime_expectancy(
    regime:      str,
    surv_group:  List[dict],
    fail_lookup: Dict[Tuple[str, str], dict],
) -> RegimeExpectancySummary:
    """
    Compute expectancy metrics for one regime group.
    surv_group:  CLEAN complete survivability records for this regime.
    fail_lookup: CLEAN failure records keyed by (symbol, observation_date).
    observation_only=True: pure analytics.
    """
    from src.analytics.future_leader_half_life import HALF_LIFE_WINDOW

    fwd5  = [float(r["forward_return_5d"])  for r in surv_group if r.get("forward_return_5d")  is not None]
    fwd10 = [float(r["forward_return_10d"]) for r in surv_group if r.get("forward_return_10d") is not None]
    mfe   = [float(r["mfe_10d"])  for r in surv_group if r.get("mfe_10d")  is not None]
    mae   = [float(r["mae_10d"])  for r in surv_group if r.get("mae_10d")  is not None]

    e5  = _opt_mean(fwd5)
    e10 = _opt_mean(fwd10)

    survival_days: List[int] = []
    for r in surv_group:
        key  = (r.get("symbol", ""), r.get("observation_date", ""))
        fail = fail_lookup.get(key)
        if fail is None:
            continue
        if fail.get("failure_detected", False):
            offset = fail.get("failure_day_offset")
            if offset is not None:
                survival_days.append(int(offset))
        else:
            survival_days.append(HALF_LIFE_WINDOW)

    return RegimeExpectancySummary(
        regime=regime,
        sample_count=len(surv_group),
        day5_expectancy=e5,
        day10_expectancy=e10,
        median_survival=_opt_median(survival_days),
        half_life_day=_half_life_from_expectancy(e5, e10),
        median_mfe=_opt_median(mfe),
        median_mae=_opt_median(mae),
    )


def compute_all_regime_expectancies(
    clean_surv:    List[dict],
    clean_fail:    List[dict],
    regime_lookup: Dict[str, str],
) -> List[RegimeExpectancySummary]:
    """
    Group CLEAN survivability records by regime and compute per-regime expectancy.
    Returns summaries in _REGIME_ORDER.
    observation_only=True: no execution path connection.
    """
    fail_lookup: Dict[Tuple[str, str], dict] = {
        (r.get("symbol", ""), r.get("observation_date", "")): r
        for r in clean_fail
        if r.get("symbol") and r.get("observation_date")
    }

    groups: Dict[str, List[dict]] = {r: [] for r in _REGIME_ORDER}
    for r in clean_surv:
        obs_date = r.get("observation_date", "")
        regime   = regime_lookup.get(obs_date, "range")
        groups[regime].append(r)

    return [
        compute_regime_expectancy(regime, groups[regime], fail_lookup)
        for regime in _REGIME_ORDER
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Report formatter  (pure — no I/O)
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v: Optional[float]) -> str:
    return f"{v * 100:+.1f}%" if v is not None else "N/A"


def _day(v: Optional[int]) -> str:
    return f"{v}d" if v is not None else "N/A"


def _float1(v: Optional[float]) -> str:
    return f"{v:.1f}" if v is not None else "N/A"


def format_regime_section(summaries: List[RegimeExpectancySummary]) -> str:
    """Render REGIME EXPECTANCY section. Pure function: no I/O."""
    lines = ["=== REGIME EXPECTANCY ==="]
    any_data = any(s.sample_count >= _MIN_SAMPLE for s in summaries)
    if not any_data:
        lines.append("(insufficient CLEAN records for regime analysis)")
        return "\n".join(lines)

    for s in summaries:
        if s.sample_count == 0:
            continue
        lines.append("")
        lines.append(f"{s.regime}:")
        lines.append(f"  sample: {s.sample_count}")
        if s.sample_count < _MIN_SAMPLE:
            lines.append("  (insufficient sample)")
            continue
        if s.day10_expectancy is not None:
            lines.append(f"  day10 expectancy: {_pct(s.day10_expectancy)}")
        if s.day5_expectancy is not None:
            lines.append(f"  day5  expectancy: {_pct(s.day5_expectancy)}")
        if s.median_survival is not None:
            lines.append(f"  median survival:  {_float1(s.median_survival)}d")
        if s.half_life_day is not None:
            lines.append(f"  half-life:        {_day(s.half_life_day)}")
        if s.median_mfe is not None:
            lines.append(f"  median MFE:       {_pct(s.median_mfe)}")
        if s.median_mae is not None:
            lines.append(f"  median MAE:       {_pct(s.median_mae)}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_regime_hook(
    survivability_log_dir: Path,
    failure_log_dir:       Path,
    regime_log_dir:        Path,
    reports_dir:           Path,
    backtest_dataset_dir:  Path,
    default_data_version:  str,
    cache_dir:             Path,
    today:                 Optional[date] = None,
) -> None:
    """
    Integration hook for run_live_signal.py. Fail-open.

    1. Identifies observation_dates not yet regime-tagged.
    2. Loads TOPIX OHLCV (confirmed bars only: index < today).
    3. Classifies regime for each new date and persists to regime JSONL.
    4. Computes regime-conditioned expectancy from CLEAN survivability records.
    5. Appends REGIME EXPECTANCY section to today's future_leader_report_YYYYMMDD.md.

    observation_only=True: no execution path connection.
    Positioned after future leader half-life hook in run_live_signal.py.
    """
    if today is None:
        today = datetime.now(JST).date()

    today_str   = today.strftime("%Y%m%d")
    report_path = reports_dir / f"future_leader_report_{today_str}.md"

    try:
        all_surv = _load_survivability_records(survivability_log_dir)
        all_fail = _load_failure_records(failure_log_dir)

        if not all_surv:
            logger.info("[FL_REGIME] no survivability records — section skipped")
            return

        clean_surv = _filter_clean_survivability(all_surv)
        clean_fail = _filter_clean_failure(all_fail)

        # Find observation_dates needing regime classification
        all_obs_dates = list({r.get("observation_date", "") for r in all_surv if r.get("observation_date")})
        existing_tags = _load_regime_tags(regime_log_dir)
        new_dates     = [d for d in all_obs_dates if d not in existing_tags]

        regime_lookup: Dict[str, str] = dict(existing_tags)

        if new_dates:
            # Load TOPIX; filter to confirmed bars only (index < today)
            try:
                from src.live.future_leader_screener import _load_ohlcv_for_screener
                ohlcv_map = _load_ohlcv_for_screener(
                    symbols=["1306.T"],
                    backtest_dataset_dir=backtest_dataset_dir,
                    default_data_version=default_data_version,
                    cache_dir=cache_dir,
                )
                topix_raw = ohlcv_map.get("1306.T")
                if topix_raw is not None and "Close" in topix_raw.columns:
                    today_ts  = pd.Timestamp(today).normalize()
                    topix_df  = topix_raw.copy()
                    topix_df.index = pd.DatetimeIndex(topix_df.index).normalize()
                    topix_df  = topix_df[topix_df.index < today_ts]  # exclude today's incomplete bar
                    new_regime_map, new_meta_map = build_regime_lookup(new_dates, topix_df)
                    _write_regime_tags(new_dates, new_regime_map, new_meta_map, regime_log_dir, today=today)
                    regime_lookup.update(new_regime_map)
                    logger.info("[FL_REGIME] classified %d new dates", len(new_dates))
                else:
                    logger.warning("[FL_REGIME] TOPIX data unavailable — regime tags not updated")
            except Exception as topix_err:
                logger.warning("[FL_REGIME] TOPIX load failed (fail-open): %s", topix_err)

        if not regime_lookup:
            logger.info("[FL_REGIME] no regime tags available — section skipped")
            return

        summaries = compute_all_regime_expectancies(clean_surv, clean_fail, regime_lookup)
        section   = format_regime_section(summaries)

        reports_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if report_path.exists() else "w"
        with report_path.open(mode, encoding="utf-8") as f:
            f.write("\n\n" + section + "\n")

        total_tagged = sum(s.sample_count for s in summaries)
        logger.info(
            "[FL_REGIME] section written → %s (tagged=%d, clean_surv=%d)",
            report_path, total_tagged, len(clean_surv),
        )

    except Exception as e:
        logger.warning("[FL_REGIME] hook failed (fail-open): %s", e)
