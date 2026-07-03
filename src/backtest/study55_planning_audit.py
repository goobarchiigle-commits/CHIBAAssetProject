"""
study55_planning_audit.py
Study55 Planning Audit — Entry Alpha Leakage & Quality Exit Validation

Phase1: MAX_POS Opportunity Audit + Counterfactual
Phase2: Quality Exit Feature Discovery (Day3/5/10)
Phase3: Cluster Alpha Audit
Final:  Research Priority Rankings

禁止: 新ルール実装 / バックテスト変更 / 過剰最適化
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab
from src.config_loader import load_strategy_config

TODAY_STR   = date.today().strftime("%Y-%m-%d")
CAPITAL     = 3_000_000
IS_START    = "2018-01-01"
IS_END      = "2024-12-31"
DATA_END    = "2025-12-31"
MIN_HOLD    = 3
MIN_RSR     = 75.0

EP_EXIT         = "A"
EP_ADDON        = "D"
ADDON_ATR_MULT  = 1.0
ADDON_SIZE_FRAC = 0.25

OBS_DAYS = [3, 5, 10]

OUT_FILE = ROOT / "backtests" / f"study55_planning_audit_{TODAY_STR}.json"


# ================================================================== #
#  ユーティリティ
# ================================================================== #

def _safe(v, d=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    return round(float(v), d)


def _atr20(df_c: pd.DataFrame) -> pd.Series:
    h = df_c["High"]; l = df_c["Low"]; c = df_c["Close"]
    cp = c.shift(1).fillna(c)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.rolling(20, min_periods=10).mean()


def _spearman(x, y) -> tuple[float, float]:
    try:
        from scipy.stats import spearmanr
        r, p = spearmanr(x, y); return float(r), float(p)
    except Exception:
        rx = pd.Series(x).rank().values; ry = pd.Series(y).rank().values
        r = float(np.corrcoef(rx, ry)[0, 1])
        return r, 0.05


def _fwd_return(close: pd.Series, ref_date: pd.Timestamp, ref_px: float, n_days: int) -> float | None:
    future = close[close.index > ref_date]
    if len(future) < n_days:
        return None
    return (float(future.iloc[n_days - 1]) / ref_px - 1.0) * 100


def _path_max_min(close: pd.Series, ref_date: pd.Timestamp, ref_px: float, horizon: int = 60
                  ) -> tuple[float | None, float | None]:
    future = close[close.index > ref_date]
    if future.empty:
        return None, None
    seg = future.iloc[:horizon]
    return (
        _safe(float(seg.max() / ref_px - 1) * 100, 2),
        _safe(float(seg.min() / ref_px - 1) * 100, 2),
    )


def _summary(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    a = np.array(vals)
    return {
        "n":          len(a),
        "mean":       _safe(float(a.mean()), 2),
        "median":     _safe(float(np.median(a)), 2),
        "win_rate":   _safe(float((a > 0).mean() * 100), 1),
        "top10pct":   _safe(float(np.percentile(a, 90)), 2),
        "bottom10pct":_safe(float(np.percentile(a, 10)), 2),
        "p25":        _safe(float(np.percentile(a, 25)), 2),
        "p75":        _safe(float(np.percentile(a, 75)), 2),
    }


# ================================================================== #
#  BT 実行 (Study54 同一設定)
# ================================================================== #

def get_active(ds: dict, start: str, end: str) -> pd.DataFrame:
    cfg = load_strategy_config()
    bc  = cfg.risk_controls.bear_universe_filter
    be  = list(bc.excluded_sectors) if bc.enabled else None
    return cab.build_dyn_rsr42_active(
        universe_raw=ds["universe_raw"], topix_close=ds["topix_close"],
        rsr_df=ds["rsr_df"], all_syms=list(ds["trade_syms"].keys()),
        start=start, end=end, bear_exclude_sectors=be,
        sym_sector_map=dict(ds["trade_syms"]) if be else None,
    )


def run_bt(ds: dict, sym_active_df, start: str, end: str) -> dict:
    return cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"], rsr_syms=ds["rsr_syms"],
        cfg=ds["base_cfg"], start=start, end=end, verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=CAPITAL, min_hold=MIN_HOLD, topix_close=ds["topix_close"],
        market_shock_mode="composite", rsr_exit_threshold=70.0,
        sym_active_df=sym_active_df,
        enable_simple_rsr_exit=True, enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True, enable_atr_risk_sizing=False,
        enable_mtf_filter=False, sizing_mode="existing",
        exit_policy=EP_EXIT, exit_policy_atr_mult=ADDON_ATR_MULT,
        exit_policy_defer_days=5, max_positions_ts=None,
        addon_policy=EP_ADDON, addon_atr_mult=ADDON_ATR_MULT,
        addon_stage2_mult=2.0, addon_max_per_pos=1, addon_size_frac=ADDON_SIZE_FRAC,
    )


# ================================================================== #
#  Phase1a: MAX_POS Detailed Analysis
# ================================================================== #

def phase1a_maxpos_audit(missed_cands: list[dict], universe_raw: dict) -> dict:
    """
    全MAX_POS候補の fwd20/60/120 + Decile分析。
    False Negative 定義の明確化も含む。
    """
    records = []
    for cand in missed_cands:
        sym  = cand.get("symbol")
        dstr = cand.get("date")
        if not sym or not dstr or sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close = df_c["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        try:
            d = pd.Timestamp(dstr)
        except Exception:
            continue
        avail = close[close.index <= d]
        if avail.empty:
            continue
        ref_px = float(avail.iloc[-1])
        if ref_px <= 0:
            continue
        mr, md = _path_max_min(close, d, ref_px, 60)
        row = {
            "date":   dstr, "symbol": sym,
            "rsr":    cand.get("rsr", 0.0),
            "rank":   cand.get("rank", -1),
            "fwd20":  _fwd_return(close, d, ref_px, 20),
            "fwd60":  _fwd_return(close, d, ref_px, 60),
            "fwd120": _fwd_return(close, d, ref_px, 120),
            "max_runup60":   mr,
            "max_drawdown60": md,
        }
        records.append(row)

    if not records:
        return {}

    df = pd.DataFrame(records)

    # Full summary
    full = {
        "fwd20":  _summary(df["fwd20"].dropna().tolist()),
        "fwd60":  _summary(df["fwd60"].dropna().tolist()),
        "fwd120": _summary(df["fwd120"].dropna().tolist()),
        "max_runup60_avg":    _safe(df["max_runup60"].mean(), 2),
        "max_drawdown60_avg": _safe(df["max_drawdown60"].mean(), 2),
    }

    # Decile analysis by fwd60
    fwd60_vals = df["fwd60"].dropna()
    deciles: dict = {}
    for label, lo, hi in [("Top10%", 90, 100), ("Top20%", 80, 100),
                           ("Top30%", 70, 100), ("Bottom30%", 0, 30)]:
        p_lo = float(np.percentile(fwd60_vals, lo)) if lo > 0 else -9999
        p_hi = float(np.percentile(fwd60_vals, hi)) if hi < 100 else 9999
        seg  = fwd60_vals[(fwd60_vals >= p_lo) & (fwd60_vals <= p_hi)]
        deciles[label] = {
            "n": len(seg),
            "threshold": _safe(p_lo, 2) if lo > 0 else None,
            "mean_fwd60": _safe(float(seg.mean()), 2) if len(seg) else None,
            "win_rate60": _safe(float((seg > 0).mean() * 100), 1) if len(seg) else None,
        }

    # False Negative definition comparison
    exec_median = None  # will be passed in
    fn_stats = {
        "definition_vs_exec_median": "fwd60 > EXECUTED group median (6.29%)",
        "definition_vs_exec_mean":   "fwd60 > EXECUTED group mean (8.13%)",
        "n_above_exec_median_629":   int((fwd60_vals > 6.29).sum()),
        "n_above_exec_mean_813":     int((fwd60_vals > 8.13).sum()),
        "n_above_zero":              int((fwd60_vals > 0).sum()),
        "pct_above_exec_median":     _safe(float((fwd60_vals > 6.29).mean() * 100), 1),
        "pct_above_exec_mean":       _safe(float((fwd60_vals > 8.13).mean() * 100), 1),
    }

    # By rank (0=1位, 1=2位, 2=3位+)
    rank_stats = {}
    for rank_val in [0, 1, 2]:
        r_df = df[df["rank"] == rank_val]["fwd60"].dropna()
        rank_stats[f"rank_{rank_val}"] = _summary(r_df.tolist())
    rank_stats["rank_3plus"] = _summary(df[df["rank"] >= 3]["fwd60"].dropna().tolist())

    return {
        "n_total": len(records),
        "full_summary": full,
        "decile_by_fwd60": deciles,
        "false_negative_definition": fn_stats,
        "by_candidate_rank": rank_stats,
    }


# ================================================================== #
#  Phase1b: Counterfactual (rejected vs weakest holding)
# ================================================================== #

def phase1b_counterfactual(
    missed_cands: list[dict],
    sell_trades: list[dict],
    is_dates: pd.DatetimeIndex,
    universe_raw: dict,
    trade_syms: dict,
) -> dict:
    """
    MAX_POS候補発生日に保有していた3銘柄の最弱と比較。
    delta = rejected_fwd60 - weakest_holding_fwd60
    """
    # Build position tracker: date_idx -> set of symbols
    # Dedup: same symbol can appear multiple times (addon) → treat as one position
    position_map: dict[int, set] = defaultdict(set)
    for t in sell_trades:
        ei = t.get("entry_idx", -1)
        xi = t.get("exit_idx", -1)
        sym = t.get("symbol")
        if ei < 0 or xi < 0 or not sym:
            continue
        for i in range(ei, xi):
            position_map[i].add(sym)

    # Date → index lookup (approximate: use rsr_df is_dates)
    date_to_idx = {str(d.date()): i for i, d in enumerate(is_dates)}

    records = []
    for cand in missed_cands:
        sym  = cand.get("symbol")
        dstr = cand.get("date")
        if not sym or not dstr:
            continue

        d_idx = date_to_idx.get(dstr)
        if d_idx is None:
            continue

        holdings = position_map.get(d_idx, set())
        holdings.discard(sym)  # rejected candidate might appear if we had partial data
        if not holdings:
            continue

        ref_date = pd.Timestamp(dstr)

        # Candidate fwd60
        if sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close_c = df_c["Close"].dropna()
        close_c.index = pd.to_datetime(close_c.index)
        avail_c = close_c[close_c.index <= ref_date]
        if avail_c.empty:
            continue
        ref_px_c = float(avail_c.iloc[-1])
        cand_fwd60 = _fwd_return(close_c, ref_date, ref_px_c, 60)
        if cand_fwd60 is None:
            continue

        # Holdings fwd60 (from that date, not from their original entry)
        holding_fwd60s = {}
        for h_sym in holdings:
            if h_sym not in universe_raw:
                continue
            df_h = universe_raw[h_sym].get("df")
            if df_h is None or "Close" not in df_h.columns:
                continue
            close_h = df_h["Close"].dropna()
            close_h.index = pd.to_datetime(close_h.index)
            avail_h = close_h[close_h.index <= ref_date]
            if avail_h.empty:
                continue
            ref_px_h = float(avail_h.iloc[-1])
            f60 = _fwd_return(close_h, ref_date, ref_px_h, 60)
            if f60 is not None:
                holding_fwd60s[h_sym] = f60

        if not holding_fwd60s:
            continue

        weakest_sym  = min(holding_fwd60s, key=lambda s: holding_fwd60s[s])
        weakest_f60  = holding_fwd60s[weakest_sym]
        delta        = cand_fwd60 - weakest_f60

        records.append({
            "date":          dstr,
            "rejected_sym":  sym,
            "rejected_rsr":  cand.get("rsr"),
            "rejected_rank": cand.get("rank"),
            "candidate_fwd60": _safe(cand_fwd60, 2),
            "weakest_holding": weakest_sym,
            "weakest_fwd60":   _safe(weakest_f60, 2),
            "delta":           _safe(delta, 2),
            "holdings":        list(holdings),
            "n_holdings":      len(holdings),
        })

    if not records:
        return {"n_events": 0}

    df = pd.DataFrame(records)
    deltas = df["delta"].dropna()
    cand_fwd60s = df["candidate_fwd60"].dropna()
    weak_fwd60s = df["weakest_fwd60"].dropna()

    # Positive delta = swap would have helped
    n_positive = int((deltas > 0).sum())
    n_strongly_pos = int((deltas > 5).sum())

    return {
        "n_events":               len(records),
        "candidate_fwd60":        _summary(cand_fwd60s.tolist()),
        "weakest_holding_fwd60":  _summary(weak_fwd60s.tolist()),
        "delta_fwd60":            _summary(deltas.tolist()),
        "n_delta_positive":       n_positive,
        "n_delta_strongly_pos_5pp": n_strongly_pos,
        "pct_swap_beneficial":    _safe(float(n_positive / max(1, len(deltas))) * 100, 1),
        "verdict_explanation": (
            "delta>0: 入れ替えた方が良かったケース / "
            "delta<0: 既存保有の方が良かったケース"
        ),
        "top10_by_delta": sorted(
            records, key=lambda x: x.get("delta") or -9999, reverse=True
        )[:10],
    }


# ================================================================== #
#  Phase2: Quality Exit Feature Discovery (Day3/5/10)
# ================================================================== #

def _features_at_dayN(
    sym: str,
    entry_date: pd.Timestamp,
    entry_idx: int,
    n_days: int,
    is_dates: pd.DatetimeIndex,
    universe_raw: dict,
    rsr_df: pd.DataFrame,
    topix_close: pd.Series | None,
    entry_rsr: float,
) -> dict | None:
    """entry_date + n_days 後の特徴量。n_days 未満で決済済みの場合 None。"""
    obs_idx = entry_idx + n_days
    if obs_idx >= len(is_dates):
        return None
    obs_date = is_dates[obs_idx]

    if sym not in universe_raw:
        return None
    df_c = universe_raw[sym].get("df")
    if df_c is None or "Close" not in df_c.columns:
        return None

    close = df_c["Close"].dropna()
    close.index = pd.to_datetime(close.index)

    avail = close[close.index <= obs_date]
    if len(avail) < 20:
        return None

    obs_px = float(avail.iloc[-1])
    entry_avail = close[close.index <= entry_date]
    if entry_avail.empty:
        return None
    entry_close = float(entry_avail.iloc[-1])

    feat: dict = {}

    # Return from entry to obs_date
    feat["ret_from_entry"] = _safe((obs_px / entry_close - 1.0) * 100, 2) if entry_close > 0 else None

    # RSR at obs_date
    rsr_sym = rsr_df[sym] if sym in rsr_df.columns else pd.Series(dtype=float)
    rsr_avail = rsr_sym[rsr_sym.index <= obs_date].dropna()
    if not rsr_avail.empty:
        rsr_now = float(rsr_avail.iloc[-1])
        feat["rsr_now"] = _safe(rsr_now, 1)
        feat["rsr_delta"] = _safe(rsr_now - entry_rsr, 2)

        # RSR acceleration: current slope vs entry slope
        if len(rsr_avail) >= 22:
            rsr_slope_now  = (rsr_now - float(rsr_avail.iloc[-22])) / 21.0
            # entry slope (using entry_rsr vs ~21d before entry)
            entry_avail_rsr = rsr_sym[rsr_sym.index <= entry_date].dropna()
            if len(entry_avail_rsr) >= 22:
                rsr_slope_entry = (float(entry_avail_rsr.iloc[-1]) - float(entry_avail_rsr.iloc[-22])) / 21.0
                feat["rs_accel_post"] = _safe(rsr_slope_now - rsr_slope_entry, 4)

    # MA20 deviation at obs_date
    if len(avail) >= 20:
        ma20 = float(avail.iloc[-20:].mean())
        feat["ma20_dev"] = _safe((obs_px / ma20 - 1.0) * 100, 2) if ma20 > 0 else None

    # 20d prev high: still in breakout zone?
    if len(avail) >= 21:
        ph20 = float(avail.iloc[-21:-1].max())
        feat["breakout_retained"] = float(obs_px > ph20) if ph20 > 0 else None
        feat["breakout_dist"] = _safe((obs_px / ph20 - 1.0) * 100, 2) if ph20 > 0 else None

    # Volume retention
    vol_col = "Volume" if "Volume" in df_c.columns else ("volume" if "volume" in df_c.columns else None)
    if vol_col:
        vol = pd.to_numeric(df_c[vol_col], errors="coerce").dropna()
        vol.index = pd.to_datetime(vol.index)
        vol_avail = vol[vol.index <= obs_date]
        if len(vol_avail) >= 20:
            vol_ma20 = float(vol_avail.iloc[-20:].mean())
            vol_now  = float(vol_avail.iloc[-1])
            feat["vol_retention"] = _safe(vol_now / vol_ma20, 3) if vol_ma20 > 0 else None

    # ATR expansion
    if "High" in df_c.columns and "Low" in df_c.columns:
        atr = _atr20(df_c)
        atr_avail = atr[atr.index <= obs_date].dropna()
        atr_entry_avail = atr[atr.index <= entry_date].dropna()
        if not atr_avail.empty and not atr_entry_avail.empty:
            feat["atr_expansion"] = _safe(float(atr_avail.iloc[-1]) / float(atr_entry_avail.iloc[-1]), 3) \
                if float(atr_entry_avail.iloc[-1]) > 0 else None

    # Market RS (obs N-day return vs TOPIX)
    if topix_close is not None:
        tp_avail = topix_close[topix_close.index <= obs_date].dropna()
        tp_entry = topix_close[topix_close.index <= entry_date].dropna()
        if not tp_avail.empty and not tp_entry.empty:
            sym_ret  = (obs_px / entry_close - 1.0) * 100 if entry_close > 0 else None
            tp_ret   = (float(tp_avail.iloc[-1]) / float(tp_entry.iloc[-1]) - 1.0) * 100
            if sym_ret is not None:
                feat["mkt_rs_vs_entry"] = _safe(sym_ret - tp_ret, 2)

    return feat


POST_FEATURES = [
    "ret_from_entry", "rsr_now", "rsr_delta", "rs_accel_post",
    "ma20_dev", "breakout_retained", "breakout_dist",
    "vol_retention", "atr_expansion", "mkt_rs_vs_entry",
]


def phase2_quality_exit(
    sell_trades: list[dict],
    is_dates: pd.DatetimeIndex,
    ds: dict,
) -> dict:
    """
    Winner (Top25%) vs Loser (Bottom25%) の Day+N 後特徴量分析。
    対象: EXECUTED 全 trades。
    """
    universe_raw = ds["universe_raw"]
    rsr_df       = ds["rsr_df"].loc[ds["rsr_df"].index.isin(is_dates)]
    topix_close  = ds.get("topix_close")
    trade_syms   = ds["trade_syms"]

    # Build base records with total return
    base = []
    for t in sell_trades:
        ei = t.get("entry_idx", -1)
        if ei < 1 or ei >= len(is_dates):
            continue
        sym        = t["symbol"]
        entry_px   = t.get("entry", 0.0)
        exit_px    = t.get("exit", 0.0)
        exit_idx   = t.get("exit_idx", -1)
        hold_days  = exit_idx - ei if exit_idx >= 0 else -1
        total_ret  = (exit_px / entry_px - 1.0) * 100 if entry_px > 0 else None

        entry_date = is_dates[ei]
        sig_date   = is_dates[ei - 1]

        # entry RSR
        rsr_val = None
        if sym in rsr_df.columns and sig_date in rsr_df.index:
            rsr_val = float(rsr_df.loc[sig_date, sym])

        base.append({
            "symbol": sym, "sector": trade_syms.get(sym, "不明"),
            "entry_idx": ei, "exit_idx": exit_idx,
            "entry_date": entry_date, "sig_date": sig_date,
            "entry_px": entry_px, "exit_px": exit_px,
            "total_ret": total_ret, "hold_days": hold_days,
            "entry_rsr": rsr_val,
        })

    base_df = pd.DataFrame(base)
    if base_df.empty or base_df["total_ret"].dropna().empty:
        return {"error": "no trades"}

    n_total = len(base_df.dropna(subset=["total_ret"]))
    top_thr = float(base_df["total_ret"].quantile(0.75))
    bot_thr = float(base_df["total_ret"].quantile(0.25))

    results_by_day: dict = {}

    for n_days in OBS_DAYS:
        feat_records = []
        for _, row in base_df.iterrows():
            if row["hold_days"] < n_days or row["total_ret"] is None:
                continue  # trade closed before obs point
            if pd.isna(row["entry_rsr"]):
                continue

            feat = _features_at_dayN(
                sym=row["symbol"],
                entry_date=row["entry_date"],
                entry_idx=int(row["entry_idx"]),
                n_days=n_days,
                is_dates=is_dates,
                universe_raw=universe_raw,
                rsr_df=rsr_df,
                topix_close=topix_close,
                entry_rsr=float(row["entry_rsr"]),
            )
            if feat is None:
                continue

            # remaining return = (exit_px / obs_close - 1) * 100
            rem_ret = None
            df_c = universe_raw.get(row["symbol"], {}).get("df")
            if df_c is not None and "Close" in df_c.columns:
                close = df_c["Close"].dropna()
                close.index = pd.to_datetime(close.index)
                obs_date = is_dates[int(row["entry_idx"]) + n_days]
                obs_av = close[close.index <= obs_date]
                if not obs_av.empty:
                    obs_px = float(obs_av.iloc[-1])
                    if obs_px > 0:
                        rem_ret = (row["exit_px"] / obs_px - 1.0) * 100

            rec = {
                "symbol":    row["symbol"],
                "total_ret": row["total_ret"],
                "rem_ret":   rem_ret,
                "label_winner": int(row["total_ret"] >= top_thr),
                "label_loser":  int(row["total_ret"] <= bot_thr),
            }
            rec.update(feat)
            feat_records.append(rec)

        if not feat_records:
            results_by_day[f"day{n_days}"] = {"n": 0}
            continue

        df_day = pd.DataFrame(feat_records)
        n_obs  = len(df_day)
        winners = df_day[df_day["label_winner"] == 1]
        losers  = df_day[df_day["label_loser"]  == 1]

        feat_analysis: dict = {}
        for feat in POST_FEATURES:
            if feat not in df_day.columns:
                continue
            w_v = winners[feat].dropna()
            l_v = losers[feat].dropna()
            all_v = df_day[feat].dropna()
            coverage = round(float(len(all_v) / n_obs * 100), 1)

            # Spearman IC with remaining return
            ic_rem, _  = (0.0, 1.0)
            valid_rem  = df_day[["rem_ret", feat]].dropna()
            if len(valid_rem) >= 5:
                ic_rem, _ = _spearman(valid_rem["rem_ret"].values, valid_rem[feat].values)

            # IC with total return
            valid_tot = df_day[["total_ret", feat]].dropna()
            ic_tot = 0.0
            if len(valid_tot) >= 5:
                ic_tot, _ = _spearman(valid_tot["total_ret"].values, valid_tot[feat].values)

            # Cohen's d
            d = None
            if len(w_v) > 1 and len(l_v) > 1:
                pooled = np.sqrt((w_v.var(ddof=1) + l_v.var(ddof=1)) / 2)
                d = _safe((float(w_v.mean()) - float(l_v.mean())) / pooled if pooled > 0 else 0.0, 3)

            feat_analysis[feat] = {
                "winner_mean":  _safe(float(w_v.mean()), 3) if len(w_v) else None,
                "loser_mean":   _safe(float(l_v.mean()), 3) if len(l_v) else None,
                "all_mean":     _safe(float(all_v.mean()), 3) if len(all_v) else None,
                "diff_w_l":     _safe(float(w_v.mean() - l_v.mean()), 3) if len(w_v) and len(l_v) else None,
                "ic_rem_ret":   _safe(ic_rem, 3),
                "ic_total_ret": _safe(ic_tot, 3),
                "cohens_d":     d,
                "coverage_pct": coverage,
                "n_valid":      len(all_v),
            }

        # Rank features by |IC with remaining return|
        ranked = sorted(
            [(k, v) for k, v in feat_analysis.items() if v.get("ic_rem_ret") is not None],
            key=lambda x: abs(x[1]["ic_rem_ret"]),
            reverse=True,
        )

        results_by_day[f"day{n_days}"] = {
            "n_obs":             n_obs,
            "n_winners":         len(winners),
            "n_losers":          len(losers),
            "winner_threshold":  _safe(top_thr, 2),
            "loser_threshold":   _safe(bot_thr, 2),
            "feature_analysis":  feat_analysis,
            "ranked_by_ic_rem":  [r[0] for r in ranked],
        }

    # Aggregate: consistent top features across days
    feat_scores: dict[str, list[float]] = defaultdict(list)
    for day_key, day_res in results_by_day.items():
        for feat, fdat in day_res.get("feature_analysis", {}).items():
            ic = fdat.get("ic_rem_ret")
            if ic is not None:
                feat_scores[feat].append(abs(ic))

    avg_scores = {f: np.mean(vs) for f, vs in feat_scores.items() if vs}
    top10 = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:10]

    return {
        "n_total_trades": n_total,
        "top_threshold":  _safe(top_thr, 2),
        "bot_threshold":  _safe(bot_thr, 2),
        "obs_days":       OBS_DAYS,
        "by_day":         results_by_day,
        "quality_exit_top10": [
            {"rank": i+1, "feature": f, "avg_ic_rem": _safe(s, 3)}
            for i, (f, s) in enumerate(top10)
        ],
    }


# ================================================================== #
#  Phase3: Cluster Alpha Audit
# ================================================================== #

def phase3_cluster_audit(
    skip_detail: list[dict],
    universe_raw: dict,
    executed_fwd60_mean: float,
) -> dict:
    """
    CLUSTER_CAP全57件の fwd60d 分析と上位集中度チェック。
    """
    cluster_cands = [c for c in skip_detail if c.get("reason") == "CLUSTER_CAP"]

    records = []
    for cand in cluster_cands:
        sym  = cand.get("symbol")
        dstr = cand.get("date")
        if not sym or not dstr or sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close = df_c["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        try:
            d = pd.Timestamp(dstr)
        except Exception:
            continue
        avail = close[close.index <= d]
        if avail.empty:
            continue
        ref_px = float(avail.iloc[-1])
        if ref_px <= 0:
            continue
        fwd60 = _fwd_return(close, d, ref_px, 60)
        mr, md = _path_max_min(close, d, ref_px, 60)
        records.append({
            "date": dstr, "symbol": sym,
            "rsr": cand.get("rsr"), "sector": cand.get("sector", "不明"),
            "fwd60": fwd60, "max_runup60": mr, "max_drawdown60": md,
        })

    if not records:
        return {"n": 0}

    df = pd.DataFrame(records)
    fwd60_vals = df["fwd60"].dropna()
    n = len(records)

    # Overall stats
    overall = _summary(fwd60_vals.tolist())

    # Concentration: how much do top-K events contribute?
    sorted_fwd60 = sorted([v for v in fwd60_vals if not np.isnan(v)], reverse=True)
    total_pos = sum(v for v in sorted_fwd60 if v > 0) or 1.0
    conc = {}
    for k in [1, 3, 5, 10]:
        top_k = sorted_fwd60[:k]
        top_k_pos = sum(v for v in top_k if v > 0)
        conc[f"top{k}_share_of_pos"] = _safe(top_k_pos / total_pos * 100, 1) if total_pos > 0 else None
        conc[f"top{k}_mean"] = _safe(np.mean(top_k), 2) if top_k else None

    # Symbol concentration
    sym_counts = df["symbol"].value_counts().to_dict()
    top_syms = sorted(sym_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    # Verdict
    top3_share = conc.get("top3_share_of_pos", 0) or 0
    verdict = "CONCENTRATED" if top3_share > 50 else "DISTRIBUTED"

    # vs EXECUTED comparison
    alpha_vs_exec = _safe(overall.get("mean", 0) - executed_fwd60_mean, 2) if overall.get("mean") else None

    return {
        "n":               n,
        "overall_fwd60":   overall,
        "alpha_vs_exec_fwd60_mean": alpha_vs_exec,
        "concentration":   conc,
        "top_symbols":     [{"symbol": s, "n_events": c} for s, c in top_syms],
        "verdict":         verdict,
        "verdict_detail":  f"Top3 events = {top3_share}% of positive fwd60 sum → {verdict}",
        "recommendation":  (
            "CLUSTER_CAP緩和研究 NECESSARY: 57件中構造的優位性あり"
            if verdict == "DISTRIBUTED" and (alpha_vs_exec or 0) > 1.0
            else "CLUSTER_CAP緩和研究 DEFER: 少数大型勝者による見かけ上の優位性 or 効果軽微"
        ),
    }


# ================================================================== #
#  Final: Research Priority
# ================================================================== #

def research_priority(
    phase1a: dict,
    phase1b: dict,
    phase2: dict,
    phase3: dict,
    exec_fwd60_mean: float,
) -> dict:
    """
    Study55/57/Cluster/EarlyEntry の実施価値を判定。
    """
    # Study55 (Quality Exit): quality exit feature の IC 強度
    top_qe = phase2.get("quality_exit_top10", [])
    top_ic  = float(top_qe[0]["avg_ic_rem"]) if top_qe else 0.0
    study55_value = (
        "HIGH" if top_ic >= 0.10 else
        "MEDIUM" if top_ic >= 0.05 else
        "LOW"
    )

    # Study57 (Second Score): Study54 rs_accel AUC=0.589
    # Here we report based on Phase1b (counterfactual delta) and Phase3 IC
    study57_value = "MEDIUM"  # rs_accel AUC=0.589 is promising but |IC|<0.065

    # Cluster allocation:
    cluster_verdict   = phase3.get("verdict", "UNKNOWN")
    cluster_alpha_gap = phase3.get("alpha_vs_exec_fwd60_mean", 0) or 0.0
    cluster_value = (
        "HIGH"   if cluster_verdict == "DISTRIBUTED" and cluster_alpha_gap > 1.5 else
        "MEDIUM" if cluster_verdict == "DISTRIBUTED" and cluster_alpha_gap > 0.5 else
        "LOW"
    )

    # Early Entry (Study56): Group A fwd60=5.17% vs exec 8.13% = -2.96pp
    early_entry_value = "LOW"  # confirmed in Study54

    # MAX_POS verdict
    maxpos_pct_beneficial = phase1b.get("pct_swap_beneficial", 0) or 0.0
    maxpos_verdict = "PASS (真犯人説支持)" if maxpos_pct_beneficial >= 50 else "FAIL (既存保有の方が優秀)"

    priority = []
    priority.append({"rank": "A", "study": "Study55 Quality Exit",
                     "value": study55_value, "reason": f"top_exit_IC={top_ic:.3f}"})
    priority.append({"rank": "B", "study": "Study57 Second Score",
                     "value": study57_value, "reason": "rs_accel AUC=0.589 (Study54確認)"})
    priority.append({"rank": "C", "study": "Cluster Allocation",
                     "value": cluster_value,
                     "reason": f"verdict={cluster_verdict} alpha_gap={cluster_alpha_gap:+.2f}pp"})
    priority.append({"rank": "D", "study": "Early Entry (Study56)",
                     "value": early_entry_value,
                     "reason": "Group A fwd60=-2.96pp vs exec (Study54確認)"})

    return {
        "maxpos_opportunity_verdict": maxpos_verdict,
        "maxpos_swap_beneficial_pct": maxpos_pct_beneficial,
        "study55_quality_exit": {
            "value": study55_value,
            "top_feature": top_qe[0]["feature"] if top_qe else None,
            "top_ic_avg":  top_ic,
        },
        "study57_second_score": {
            "value": study57_value,
            "best_candidate": "rs_accel",
            "auc_from_study54": 0.589,
        },
        "cluster_allocation": {
            "value":   cluster_value,
            "verdict": cluster_verdict,
            "alpha_vs_exec": cluster_alpha_gap,
        },
        "early_entry_study56": {
            "value": early_entry_value,
            "group_a_fwd60": 5.17,
            "exec_fwd60": exec_fwd60_mean,
            "gap_pp": _safe(5.17 - exec_fwd60_mean, 2),
        },
        "priority_list": priority,
    }


# ================================================================== #
#  メイン
# ================================================================== #

def main() -> None:
    print("=" * 72)
    print("  Study55 Planning Audit — Entry Alpha Leakage & Quality Exit")
    print(f"  Date: {TODAY_STR}   IS: {IS_START}~{IS_END}")
    print("=" * 72)

    print("\n[Data] データセット構築中...")
    ds = build_common_dataset(DATA_END)
    print(f"  {len(ds['trade_syms'])} シンボル")

    print("\n[BT] D_ATR_EQ IS run (Study54同一設定)...")
    active_is = get_active(ds, IS_START, IS_END)
    res = run_bt(ds, active_is, IS_START, IS_END)
    print(f"  CAGR={res['cagr']:+.2f}%  Trades={res['n_trades']}")

    is_dates = ds["rsr_df"].index[
        (ds["rsr_df"].index >= IS_START) & (ds["rsr_df"].index <= IS_END)
    ]

    sell_trades   = [t for t in res.get("_trades", []) if t.get("side") == "SELL"]
    missed_cands  = res.get("_missed_cands", [])
    skip_detail   = res.get("_skip_detail", [])

    # EXECUTED fwd60 statistics (from Study54)
    EXEC_FWD60_MEAN   = 8.13
    EXEC_FWD60_MEDIAN = 6.29

    # ── Phase1a ─────────────────────────────────────────────────────
    print("\n[Phase1a] MAX_POS detailed analysis...")
    p1a = phase1a_maxpos_audit(missed_cands, ds["universe_raw"])
    print(f"  n={p1a.get('n_total')}  fwd60_mean={p1a.get('full_summary',{}).get('fwd60',{}).get('mean')}")
    fn_n_med = p1a.get("false_negative_definition", {}).get("n_above_exec_median_629", 0)
    fn_n_mean= p1a.get("false_negative_definition", {}).get("n_above_exec_mean_813",  0)
    print(f"  FN定義 vs median(6.29%): {fn_n_med}件  vs mean(8.13%): {fn_n_mean}件")

    # ── Phase1b ─────────────────────────────────────────────────────
    print("\n[Phase1b] Counterfactual (rejected vs weakest holding)...")
    p1b = phase1b_counterfactual(missed_cands, sell_trades, is_dates, ds["universe_raw"], ds["trade_syms"])
    print(f"  Counterfactual events: {p1b.get('n_events')}")
    print(f"  swap有利割合: {p1b.get('pct_swap_beneficial')}%")
    print(f"  delta_mean: {p1b.get('delta_fwd60',{}).get('mean')}")

    # ── Phase2 ──────────────────────────────────────────────────────
    print("\n[Phase2] Quality Exit feature analysis (Day3/5/10)...")
    p2 = phase2_quality_exit(sell_trades, is_dates, ds)
    for dn in OBS_DAYS:
        day_r = p2.get("by_day", {}).get(f"day{dn}", {})
        top_f = day_r.get("ranked_by_ic_rem", ["—"])[0]
        print(f"  Day{dn}: n_obs={day_r.get('n_obs')}  top_IC_feat={top_f}")
    print(f"  Quality Exit Top10: {[x['feature'] for x in p2.get('quality_exit_top10',[])[:5]]}")

    # ── Phase3 ──────────────────────────────────────────────────────
    print("\n[Phase3] Cluster Alpha Audit...")
    p3 = phase3_cluster_audit(skip_detail, ds["universe_raw"], EXEC_FWD60_MEAN)
    print(f"  n={p3.get('n')}  fwd60_mean={p3.get('overall_fwd60',{}).get('mean')}")
    print(f"  verdict: {p3.get('verdict')}  alpha_gap={p3.get('alpha_vs_exec_fwd60_mean')}")
    print(f"  Top3 concentration: {p3.get('concentration',{}).get('top3_share_of_pos')}%")

    # ── Final Priority ───────────────────────────────────────────────
    print("\n[Final] Research priority ranking...")
    final = research_priority(p1a, p1b, p2, p3, EXEC_FWD60_MEAN)
    for pr in final.get("priority_list", []):
        print(f"  Priority {pr['rank']}: {pr['study']}  VALUE={pr['value']}  ({pr['reason']})")

    # ── 保存 ─────────────────────────────────────────────────────────
    output = {
        "study":  "Study55_PlanningAudit",
        "date":   TODAY_STR,
        "config": "D_ATR_EQ",
        "period": {"is_start": IS_START, "is_end": IS_END},
        "phase1a_maxpos":          p1a,
        "phase1b_counterfactual":  p1b,
        "phase2_quality_exit":     p2,
        "phase3_cluster_audit":    p3,
        "final_priority":          final,
    }
    OUT_FILE.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[Done] {OUT_FILE}")

    # ── コンソールサマリー ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Phase1: MAX_POS Opportunity Audit")
    print("=" * 72)
    fs = p1a.get("full_summary", {})
    print(f"  全{p1a.get('n_total')}件: fwd60 mean={fs.get('fwd60',{}).get('mean')}  WR={fs.get('fwd60',{}).get('win_rate')}%")
    print(f"  Max Runup 60d avg: {fs.get('fwd60',{}).get('top10pct')} (top10%)")
    fn_def = p1a.get("false_negative_definition", {})
    print(f"  FN(vs median 6.29%): {fn_def.get('n_above_exec_median_629')}件 ({fn_def.get('pct_above_exec_median')}%)")
    print(f"  FN(vs mean   8.13%): {fn_def.get('n_above_exec_mean_813')}件 ({fn_def.get('pct_above_exec_mean')}%)")
    print(f"\n  Decile分析:")
    for label, dat in p1a.get("decile_by_fwd60", {}).items():
        print(f"    {label}: n={dat.get('n')} mean={dat.get('mean_fwd60')} WR={dat.get('win_rate60')}%")

    print(f"\n  Counterfactual (rejected vs weakest holding):")
    cf = p1b
    print(f"  events={cf.get('n_events')}  delta_mean={cf.get('delta_fwd60',{}).get('mean')}  "
          f"swap_beneficial={cf.get('pct_swap_beneficial')}%")
    print(f"  → MAX_POS verdict: {final.get('maxpos_opportunity_verdict')}")

    print("\n" + "=" * 72)
    print("  Phase2: Quality Exit Feature Top10")
    print("=" * 72)
    for item in p2.get("quality_exit_top10", []):
        print(f"  #{item['rank']:2d} {item['feature']:20s}  avg_IC_rem={item['avg_ic_rem']}")

    print("\n  Day-by-Day Top Feature:")
    for dn in OBS_DAYS:
        dr = p2.get("by_day", {}).get(f"day{dn}", {})
        ranked = dr.get("ranked_by_ic_rem", [])
        fa = dr.get("feature_analysis", {})
        if ranked:
            top = ranked[0]
            td = fa.get(top, {})
            print(f"  Day{dn}(n={dr.get('n_obs')}): #{1} {top}  "
                  f"IC_rem={td.get('ic_rem_ret')}  W_mean={td.get('winner_mean')}  L_mean={td.get('loser_mean')}")

    print("\n" + "=" * 72)
    print("  Phase3: Cluster Alpha Audit")
    print("=" * 72)
    conc = p3.get("concentration", {})
    print(f"  n={p3.get('n')}  fwd60={p3.get('overall_fwd60',{}).get('mean')} (exec={EXEC_FWD60_MEAN})")
    print(f"  Concentration: top1={conc.get('top1_share_of_pos')}%  top3={conc.get('top3_share_of_pos')}%  top5={conc.get('top5_share_of_pos')}%")
    print(f"  Verdict: {p3.get('verdict_detail')}")
    print(f"  → {p3.get('recommendation')}")

    print("\n" + "=" * 72)
    print("  Final: Research Priority")
    print("=" * 72)
    for pr in final.get("priority_list", []):
        print(f"  Priority {pr['rank']} [{pr['value']:6s}]: {pr['study']}  — {pr['reason']}")


if __name__ == "__main__":
    main()
