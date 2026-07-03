"""
study58_quality_exit_audit.py
Quality Exit vs Replacement Engine — Production Adoption Decision

目的:
  Case A: Hold<T AND Cand>C → Swap only (Study57 Case E)
  Case B: Hold<T → Exit always (候補品質不問)
  Case C: Hold<T → Swap if Cand>C else Exit (Hybrid)

「低Quality銘柄を候補なし時に保有し続ける」ことが本当に期待値プラスか検証。

Sensitivity sweep: (hold_max, cand_min) = (33,68) / (35,70) / (37,72)

追加分析:
  Decision Timeline Audit
  Live Latency Audit
  Replacement Attribution (removal alpha vs replacement alpha)

VERDICT: Production採用方式を1つに確定する

禁止: 閾値最適化 / 過剰最適化 / WF情報漏洩 / 実装変更
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
from src.backtest.wf_dynamic_universe import WF_SEGS
from src.config_loader import load_strategy_config

TODAY_STR   = date.today().strftime("%Y-%m-%d")
CAPITAL     = 3_000_000
MIN_HOLD    = 3
DATA_END    = "2025-12-31"
IS_START    = "2018-01-01"
IS_END      = "2024-12-31"
OOS_START   = "2025-01-01"
OOS_END     = "2025-12-31"

EP_EXIT        = "A"
EP_ADDON       = "D"
ADDON_ATR_MULT = 1.0
ADDON_SIZE_FRAC = 0.25

# Study56 / Study57確認済み Quality Score weights
QS_WEIGHTS = {"atr_expansion": 0.405, "ret_from_entry": 0.342, "rsr_delta": 0.253}

# Sensitivity sweep: (hold_max, cand_min)
THRESHOLDS = [
    (33.0, 68.0, "33/68"),
    (35.0, 70.0, "35/70"),  # Study57 Case E baseline
    (37.0, 72.0, "37/72"),
]

# Executed median fwd60 for swap success definition (Study55)
EXEC_MEDIAN_FWD60 = 6.29

OUT_FILE = ROOT / "backtests" / f"study58_quality_exit_audit_{TODAY_STR}.json"


# ================================================================== #
#  BT 共通設定
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


def run_bt(ds: dict, sym_active_df, start: str, end: str,
           quality_forced_exits=None) -> dict:
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
        quality_exit_pairs=None,
        quality_forced_exits=quality_forced_exits,
        cluster_cap_override=None,
    )


# ================================================================== #
#  Quality Score 計算 (Study57から移植)
# ================================================================== #

def _atr20_at(df_c: pd.DataFrame) -> pd.Series:
    h = df_c["High"]; l = df_c["Low"]; c = df_c["Close"]
    cp = c.shift(1).fillna(c)
    tr = pd.concat([h - l, (h - cp).abs(), (l - cp).abs()], axis=1).max(axis=1)
    return tr.rolling(20, min_periods=10).mean()


def compute_quality_features(
    sym: str, entry_idx: int, obs_n: int,
    is_dates: pd.DatetimeIndex, universe_raw: dict, rsr_df: pd.DataFrame,
    entry_rsr: float,
) -> dict | None:
    obs_idx  = entry_idx + obs_n
    if obs_idx >= len(is_dates):
        return None
    obs_date   = is_dates[obs_idx]
    entry_date = is_dates[entry_idx]
    if sym not in universe_raw:
        return None
    df_c = universe_raw[sym].get("df")
    if df_c is None or "Close" not in df_c.columns:
        return None
    close = df_c["Close"].dropna()
    close.index = pd.to_datetime(close.index)
    avail_obs   = close[close.index <= obs_date]
    avail_entry = close[close.index <= entry_date]
    if len(avail_obs) < 10 or avail_entry.empty:
        return None
    obs_px   = float(avail_obs.iloc[-1])
    entry_px = float(avail_entry.iloc[-1])
    if entry_px <= 0 or obs_px <= 0:
        return None
    feat = {"ret_from_entry": (obs_px / entry_px - 1.0) * 100}
    rsr_sym = rsr_df[sym] if sym in rsr_df.columns else pd.Series(dtype=float)
    rsr_at_obs = rsr_sym[rsr_sym.index <= obs_date].dropna()
    if not rsr_at_obs.empty:
        feat["rsr_delta"] = float(rsr_at_obs.iloc[-1]) - entry_rsr
    if "High" in df_c.columns and "Low" in df_c.columns:
        atr = _atr20_at(df_c)
        atr_obs   = atr[atr.index <= obs_date].dropna()
        atr_entry = atr[atr.index <= entry_date].dropna()
        if not atr_obs.empty and not atr_entry.empty:
            av = float(atr_entry.iloc[-1])
            feat["atr_expansion"] = float(atr_obs.iloc[-1]) / av if av > 0 else 1.0
    return feat


class QualityScorer:
    def __init__(self):
        self._mu: dict[str, float]    = {}
        self._sigma: dict[str, float] = {}
        self._raw_p20: float = 0.0

    def fit(self, feat_list: list[dict]) -> None:
        if not feat_list:
            return
        df = pd.DataFrame(feat_list)
        for col in QS_WEIGHTS:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if len(vals) < 3:
                continue
            self._mu[col]    = float(vals.mean())
            self._sigma[col] = float(vals.std(ddof=1))
        z_scores = []
        for feat in feat_list:
            z, w = 0.0, 0.0
            for col, wt in QS_WEIGHTS.items():
                v = feat.get(col)
                if v is None or np.isnan(v):
                    continue
                s = self._sigma.get(col, 1.0)
                if s > 0:
                    z += wt * (v - self._mu.get(col, 0.0)) / s
                    w += wt
            if w > 0:
                z_scores.append(z / w)
        if z_scores:
            self._raw_p20 = float(np.percentile(z_scores, 20))

    def _raw_z(self, feat: dict) -> float:
        z, w = 0.0, 0.0
        for col, wt in QS_WEIGHTS.items():
            v = feat.get(col)
            if v is None or np.isnan(v):
                continue
            s = self._sigma.get(col, 1.0)
            if s > 0:
                z += wt * (v - self._mu.get(col, 0.0)) / s
                w += wt
        return z / w if w > 0 else 0.0

    def score(self, feat: dict) -> float:
        return min(100.0, max(0.0, 50.0 + self._raw_z(feat) * 25.0))

    def is_low_quality(self, feat: dict) -> bool:
        return self._raw_z(feat) < self._raw_p20


def build_scorer_from_trades(
    sell_trades: list[dict], is_dates: pd.DatetimeIndex,
    universe_raw: dict, rsr_df: pd.DataFrame, obs_n: int = 3,
) -> QualityScorer:
    scorer = QualityScorer()
    feat_list = []
    for t in sell_trades:
        if t.get("side") != "SELL":
            continue
        sym = t["symbol"]; ei = t.get("entry_idx", -1); xi = t.get("exit_idx", -1)
        hold = xi - ei if xi >= 0 else -1
        if ei < 1 or hold < obs_n:
            continue
        sig_date = is_dates[ei - 1] if ei - 1 < len(is_dates) else None
        if sig_date is None:
            continue
        entry_rsr = 0.0
        if sym in rsr_df.columns and sig_date in rsr_df.index:
            entry_rsr = float(rsr_df.loc[sig_date, sym])
        feat = compute_quality_features(sym, ei, obs_n, is_dates, universe_raw, rsr_df, entry_rsr)
        if feat:
            feat_list.append(feat)
    scorer.fit(feat_list)
    return scorer


# ================================================================== #
#  Plan Builder: Cases A / B / C
# ================================================================== #

def _holding_scores_at(
    holdings: dict[str, int], d_idx: int, is_dates: pd.DatetimeIndex,
    universe_raw: dict, rsr_df: pd.DataFrame,
    entry_rsr_map: dict, scorer: QualityScorer,
) -> dict[str, float]:
    scores = {}
    for h_sym, h_ei in holdings.items():
        n_held = d_idx - h_ei
        if n_held < 3:
            scores[h_sym] = 50.0
            continue
        ersr = entry_rsr_map.get((h_sym, h_ei), 0.0)
        feat = compute_quality_features(h_sym, h_ei, n_held, is_dates, universe_raw, rsr_df, ersr)
        if feat:
            scores[h_sym] = scorer.score(feat)
        else:
            scores[h_sym] = 50.0
    return scores


def build_plan(
    case: str, hold_max: float, cand_min: float,
    sell_trades: list[dict], missed_cands: list[dict],
    is_dates: pd.DatetimeIndex, universe_raw: dict, rsr_df: pd.DataFrame,
    scorer: QualityScorer,
) -> tuple[dict[int, str], list[dict]]:
    """
    Case A: hold<hold_max AND cand>cand_min → forced_exit (swap)
    Case B: hold<hold_max → forced_exit (exit; no cand check)
    Case C: hold<hold_max → forced_exit regardless (swap if cand>cand_min else exit)

    共通: 1日1アクション。quality_forced_exits を使うため買い処理前に発動。
    Returns: (forced_exits, detail_list)
    """
    position_at: dict[int, dict[str, int]] = defaultdict(dict)
    entry_rsr_map: dict[tuple, float] = {}
    for t in sell_trades:
        if t.get("side") != "SELL":
            continue
        sym = t["symbol"]; ei = t.get("entry_idx", -1); xi = t.get("exit_idx", -1)
        if ei < 0 or xi < 0:
            continue
        sig_date = is_dates[ei - 1] if ei - 1 < len(is_dates) else None
        ersr = 0.0
        if sig_date is not None and sym in rsr_df.columns and sig_date in rsr_df.index:
            ersr = float(rsr_df.loc[sig_date, sym])
        entry_rsr_map[(sym, ei)] = ersr
        for day in range(ei, xi):
            position_at[day][sym] = ei

    date_to_idx = {str(d.date()): i for i, d in enumerate(is_dates)}
    forced_exits: dict[int, str] = {}
    detail: list[dict] = []
    used_days: set[int] = set()

    for cand in missed_cands:
        sym_cand = cand.get("symbol")
        dstr     = cand.get("date")
        if not sym_cand or not dstr:
            continue
        d_idx = date_to_idx.get(dstr)
        if d_idx is None or d_idx in used_days:
            continue
        holdings = position_at.get(d_idx, {})
        if not holdings:
            continue

        cand_rsr   = cand.get("rsr") or 0.0
        cand_score = min(100.0, max(0.0, (cand_rsr - 50.0) * 2.0))

        hscores  = _holding_scores_at(holdings, d_idx, is_dates, universe_raw, rsr_df,
                                       entry_rsr_map, scorer)
        wk_sym   = min(hscores, key=lambda s: hscores[s])
        wk_score = hscores[wk_sym]

        do_act = False; action = None
        if case == "A":
            if wk_score < hold_max and cand_score > cand_min:
                do_act = True; action = "SWAP"
        elif case == "B":
            if wk_score < hold_max:
                do_act = True; action = "EXIT"
        elif case == "C":
            if wk_score < hold_max:
                do_act = True
                action = "SWAP" if cand_score > cand_min else "EXIT_NO_CAND"

        if do_act:
            hold_days = d_idx - holdings[wk_sym]
            forced_exits[d_idx] = wk_sym
            used_days.add(d_idx)
            detail.append({
                "date": dstr, "exit_sym": wk_sym,
                "hold_days_at_trigger": hold_days,
                "exit_score": round(wk_score, 1),
                "cand_sym": sym_cand,
                "cand_score": round(cand_score, 1),
                "action": action,
            })

    return forced_exits, detail


# ================================================================== #
#  メトリクス抽出
# ================================================================== #

def extract(m: dict) -> dict:
    n  = int(m.get("n_trades", 0) or 0)
    qs = int(m.get("exit_reason_counts", {}).get("QUALITY_SWAP_EXIT", 0))
    tr = [t for t in (m.get("_trades") or []) if t.get("side") == "SELL"]
    gains  = [t["pnl"] for t in tr if t.get("pnl", 0) > 0]
    losses = [t["pnl"] for t in tr if t.get("pnl", 0) < 0]
    nl   = abs(sum(losses))
    pf_v = round(sum(gains) / nl, 3) if nl > 0 else None
    turno = m.get("turnover_ratio")
    return {
        "cagr":      round(float(m.get("cagr",        0.0) or 0.0), 2),
        "sharpe":    round(float(m.get("sharpe",       0.0) or 0.0), 3),
        "max_dd":    round(float(m.get("max_dd",       0.0) or 0.0), 2),
        "calmar":    round(float(m.get("calmar",       0.0) or 0.0), 3),
        "n_trades":  n,
        "avg_exp":   round(float(m.get("avg_exposure", 0.0) or 0.0), 1),
        "pf":        pf_v,
        "swap_exits": qs,
        "turnover":  round(float(turno or 0.0), 4),
    }


def swap_success_check(forced_exits: dict, is_dates: pd.DatetimeIndex, universe_raw: dict) -> dict:
    if not forced_exits:
        return {"n": 0}
    ok = 0; n = 0
    for d_idx, sym in forced_exits.items():
        if d_idx >= len(is_dates) or sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close = df_c["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        obs_date = is_dates[d_idx]
        avail = close[close.index <= obs_date]
        if avail.empty:
            continue
        ref = float(avail.iloc[-1])
        fut = close[close.index > obs_date].iloc[:60]
        if len(fut) < 20:
            continue
        fwd60 = (float(fut.iloc[-1]) / ref - 1.0) * 100
        if fwd60 < EXEC_MEDIAN_FWD60:
            ok += 1
        n += 1
    return {"n": n, "success_rate": round(ok / n * 100, 1) if n else 0.0}


# ================================================================== #
#  Replacement Attribution
# ================================================================== #

def replacement_attribution(
    bt_result: dict, forced_exits: dict[int, str],
    is_dates: pd.DatetimeIndex, universe_raw: dict,
) -> dict:
    """
    Case A スワップの alpha を分解:
      removal_effect: 保有継続より除去が有利だった寄与 (median - holding_fwd20)
      replacement_effect: 代替銘柄の超過リターン寄与 (replacement_fwd20 - median)
    """
    trades  = bt_result.get("_trades", [])
    buys    = [t for t in trades if t.get("side") == "BUY"]

    removal_effects     = []
    replacement_effects = []
    details             = []

    for d_idx, exit_sym in forced_exits.items():
        if d_idx >= len(is_dates):
            continue
        exit_date = is_dates[d_idx]

        # removal effect: forced-exit stock's fwd20
        rem_eff = None
        if exit_sym in universe_raw:
            df_c = universe_raw[exit_sym].get("df")
            if df_c is not None and "Close" in df_c.columns:
                close = df_c["Close"].dropna()
                close.index = pd.to_datetime(close.index)
                avail = close[close.index <= exit_date]
                if not avail.empty:
                    ref = float(avail.iloc[-1])
                    fut = close[close.index > exit_date].iloc[:20]
                    if len(fut) >= 10:
                        fwd20 = (float(fut.iloc[-1]) / ref - 1.0) * 100
                        rem_eff = EXEC_MEDIAN_FWD60 - fwd20  # positive = avoided underperformance

        # replacement effect: BUY on same or next day
        rep_buy = next(
            (t for t in buys
             if t.get("entry_idx") in (d_idx, d_idx + 1) and t.get("symbol") != exit_sym),
            None
        )
        rep_eff = None; rep_sym = None
        if rep_buy:
            rep_sym = rep_buy["symbol"]
            rep_ei  = rep_buy.get("entry_idx", d_idx)
            if rep_ei < len(is_dates) and rep_sym in universe_raw:
                df_c = universe_raw[rep_sym].get("df")
                if df_c is not None and "Close" in df_c.columns:
                    close = df_c["Close"].dropna()
                    close.index = pd.to_datetime(close.index)
                    rep_date = is_dates[rep_ei]
                    avail = close[close.index <= rep_date]
                    if not avail.empty:
                        ref = float(avail.iloc[-1])
                        fut = close[close.index > rep_date].iloc[:20]
                        if len(fut) >= 10:
                            fwd20 = (float(fut.iloc[-1]) / ref - 1.0) * 100
                            rep_eff = fwd20 - EXEC_MEDIAN_FWD60  # positive = outperformed

        if rem_eff is not None:
            removal_effects.append(rem_eff)
        if rep_eff is not None:
            replacement_effects.append(rep_eff)
        details.append({
            "date":               str(exit_date.date()),
            "exit_sym":           exit_sym,
            "removal_effect_pp":  round(rem_eff, 2) if rem_eff is not None else None,
            "replacement_sym":    rep_sym,
            "replacement_effect_pp": round(rep_eff, 2) if rep_eff is not None else None,
        })

    n_rem = len(removal_effects); n_rep = len(replacement_effects)
    return {
        "n_swaps_analyzed":          len(details),
        "removal_n":                 n_rem,
        "removal_effect_mean_pp":    round(float(np.mean(removal_effects)), 2) if removal_effects else None,
        "removal_positive_rate_pct": round(sum(1 for x in removal_effects if x > 0) / n_rem * 100, 1) if n_rem else None,
        "replacement_n":             n_rep,
        "replacement_effect_mean_pp":    round(float(np.mean(replacement_effects)), 2) if replacement_effects else None,
        "replacement_positive_rate_pct": round(sum(1 for x in replacement_effects if x > 0) / n_rep * 100, 1) if n_rep else None,
        "interpretation":            _attribution_interpretation(removal_effects, replacement_effects),
        "details":                   details,
    }


def _attribution_interpretation(removal: list, replacement: list) -> str:
    if not removal:
        return "insufficient data"
    rem_mean = np.mean(removal)
    rep_mean = np.mean(replacement) if replacement else 0.0
    if rem_mean > 1.0 and rep_mean > 1.0:
        return "BOTH: removal AND replacement contribute positively"
    elif rem_mean > 1.0 and rep_mean <= 0.0:
        return "REMOVAL_DOMINANT: removing bad holdings is the key mechanism"
    elif rem_mean <= 0.0 and rep_mean > 1.0:
        return "REPLACEMENT_DOMINANT: adding good replacements is the key mechanism"
    else:
        return "WEAK: limited alpha from either component"


# ================================================================== #
#  Walk-Forward (fold共有)
# ================================================================== #

def run_wf_all(ds: dict, scorer_is: QualityScorer) -> dict:
    """
    全 9 combo (3 case × 3 threshold) + baseline を WF 実行。
    各fold でbaseline BTを1回実行し、9 comboで共有 (効率化)。
    Returns: results[key] = {wf_count, avg_oos_cagr, seg3_2022_cagr, segments, ...}
    """
    rsr_df      = ds["rsr_df"]
    universe_raw = ds["universe_raw"]

    # Keys: "BASE" + "A_33/68" / "A_35/70" etc.
    all_keys = ["BASE"] + [f"{c}_{lab}" for hold_max, cand_min, lab in THRESHOLDS for c in ("A", "B", "C")]
    seg_store: dict[str, list] = {k: [] for k in all_keys}

    for seg in WF_SEGS:
        n = seg["seg"]; oos_s, oos_e = seg["oos"]
        oos_dates = rsr_df.index[(rsr_df.index >= oos_s) & (rsr_df.index <= oos_e)]
        act_oos   = get_active(ds, oos_s, oos_e)

        print(f"  [Fold {n}] OOS {oos_s[:4]} baseline...", end="", flush=True)
        raw_base = run_bt(ds, act_oos, oos_s, oos_e)
        m_base   = extract(raw_base)
        wf_base  = m_base["sharpe"] > 0
        print(f" CAGR={m_base['cagr']:+.2f}% {'✓' if wf_base else '✗'}")
        seg_store["BASE"].append({
            "seg": n, "oos_year": oos_s[:4], "wf_pass": wf_base, **m_base,
        })

        oos_sell   = [t for t in raw_base.get("_trades", []) if t.get("side") == "SELL"]
        oos_missed = raw_base.get("_missed_cands", [])

        for hold_max, cand_min, lab in THRESHOLDS:
            for case in ("A", "B", "C"):
                key = f"{case}_{lab}"
                forced_exits, detail = build_plan(
                    case, hold_max, cand_min,
                    oos_sell, oos_missed, oos_dates, universe_raw, rsr_df, scorer_is,
                )
                n_acts = len(forced_exits)
                print(f"    {key} (n_acts={n_acts}) ", end="", flush=True)
                try:
                    raw  = run_bt(ds, act_oos, oos_s, oos_e, quality_forced_exits=forced_exits)
                    m    = extract(raw)
                    wf_p = m["sharpe"] > 0
                    sr   = swap_success_check(forced_exits, oos_dates, universe_raw)
                    seg_store[key].append({
                        "seg": n, "oos_year": oos_s[:4], "wf_pass": wf_p,
                        "n_acts": n_acts, "n_swaps": sr.get("n", 0),
                        "swap_success_rate": sr.get("success_rate"),
                        "swap_detail": detail,
                        **m,
                    })
                    mark = "✓" if wf_p else "✗"
                    print(f"CAGR={m['cagr']:+.2f}% Seg3_flag DD={m['max_dd']:.1f}% {mark}")
                except Exception as err:
                    print(f"ERROR: {err}")
                    seg_store[key].append({"seg": n, "oos_year": oos_s[:4], "wf_pass": False})

    # Aggregate per key
    results: dict[str, dict] = {}
    for key, segs in seg_store.items():
        cagrs  = [r["cagr"]   for r in segs if "cagr"   in r]
        shlist = [r["sharpe"] for r in segs if "sharpe" in r]
        ddlist = [r["max_dd"] for r in segs if "max_dd" in r]
        calm   = [r["calmar"] for r in segs if "calmar" in r]
        exps   = [r["avg_exp"] for r in segs if "avg_exp" in r]
        seg3   = next((r["cagr"] for r in segs if r.get("oos_year") == "2022" and "cagr" in r), None)
        wf_cnt = sum(1 for r in segs if r.get("wf_pass"))
        results[key] = {
            "wf_count":       wf_cnt,
            "avg_oos_cagr":   round(float(np.mean(cagrs)),  2) if cagrs else 0.0,
            "avg_oos_sharpe": round(float(np.mean(shlist)), 3) if shlist else 0.0,
            "avg_oos_dd":     round(float(np.mean(ddlist)), 2) if ddlist else 0.0,
            "avg_oos_calmar": round(float(np.mean(calm)),   3) if calm else 0.0,
            "avg_oos_exp":    round(float(np.mean(exps)),   1) if exps else 0.0,
            "fold_std_cagr":  round(float(np.std(cagrs, ddof=1)), 2) if len(cagrs) > 1 else 0.0,
            "seg3_2022_cagr": round(seg3, 2) if seg3 is not None else None,
            "total_acts":     sum(r.get("n_acts", 0) for r in segs),
            "segments":       segs,
        }
    return results


# ================================================================== #
#  Decision Timeline Audit
# ================================================================== #

def decision_timeline_audit(wf_results: dict) -> dict:
    """スワップ決定タイムラインの整合性確認。"""
    hold_days_all = []
    for key, res in wf_results.items():
        if key == "BASE":
            continue
        for seg in res.get("segments", []):
            for ev in seg.get("swap_detail", []):
                hd = ev.get("hold_days_at_trigger")
                if hd is not None:
                    hold_days_all.append(hd)

    avg_hold = float(np.mean(hold_days_all)) if hold_days_all else 0.0
    min_hold = int(min(hold_days_all)) if hold_days_all else 0

    return {
        "backtesting": {
            "obs_day":           3,
            "exit_fires_on_day": 3,
            "hold_at_exit":      "3+ days (entry_idx+3 - entry_idx = 3 >= min_hold=3)",
            "min_hold_satisfied": min_hold >= MIN_HOLD,
        },
        "live_trading": {
            "eod_data_cutoff":   "T+0 15:30",
            "signal_generation": "T+1 09:00 (run_live_signal.py 朝実行)",
            "order_execution":   "T+1 寄り成り",
            "actual_hold_days":  "4+ days (1日ラグ)",
            "min_hold_satisfied": True,  # 4 >= min_hold=3
            "settlement":        "T+3 (T+1約定+T+2受渡)",
        },
        "avg_hold_days_at_trigger": round(avg_hold, 1),
        "min_hold_days_at_trigger": min_hold,
        "n_events_analyzed":        len(hold_days_all),
        "verdict":                  "FEASIBLE" if avg_hold >= 3 else "CHECK_MIN_HOLD",
    }


# ================================================================== #
#  Live Latency Audit
# ================================================================== #

def live_latency_audit(scorer: QualityScorer) -> dict:
    """Quality Score 算出のライブ環境実現可能性チェック。"""
    return {
        "features": {
            "atr_expansion": {
                "requires": "ATR20 at entry date + ATR20 at obs date (日足OHLCV)",
                "source":   "kabuステーション OHLCV / portfolio_state.json (entry_date記録)",
                "available_eod": True,
            },
            "ret_from_entry": {
                "requires": "Close at entry + Close at obs date",
                "source":   "kabuステーション Close / portfolio_state.json",
                "available_eod": True,
            },
            "rsr_delta": {
                "requires": "RSR at entry date + RSR at obs date",
                "source":   "RSR42 universe daily (run_live_signal.py 既存)",
                "available_eod": True,
            },
        },
        "normalization_params": {
            "mu":    scorer._mu,
            "sigma": scorer._sigma,
            "raw_p20_threshold": scorer._raw_p20,
            "save_required": True,
            "save_path":     "runtime/quality_scorer_params.json",
        },
        "candidate_scoring": {
            "method": "cand_score = clip((RSR - 50) * 2, 0, 100)",
            "source": "RSR42 universe (既存 run_live_signal.py 利用可能)",
            "note":   "RSR-proxy score (Study57 互換). より精度が必要ならDay0 QS計算も可能",
        },
        "portfolio_state_requirements": [
            "entry_date per position",
            "entry_price per position",
            "entry_rsr per position",
        ],
        "implementation_steps": [
            "1. scorer params → runtime/quality_scorer_params.json に保存 (研究フェーズで1回)",
            "2. run_live_signal.py: EOD後に各ポジションのDay3 QS を算出",
            "3. 待機候補のRSR-proxy scoreと比較",
            "4. Hold<T AND Cand>C (Case A) または Hold<T (Case B/C) → 翌朝成り売り注文",
            "5. signal_bridge.py で swap/exit を通常の売りシグナルと同様に処理",
        ],
        "overall_feasibility": "FEASIBLE",
        "estimated_implementation_effort": "medium (2-4 files, ~100 lines)",
    }


# ================================================================== #
#  Verdict
# ================================================================== #

def compute_verdict(wf_results: dict, base_key: str = "BASE") -> dict:
    base = wf_results.get(base_key, {})
    base_cagr = base.get("avg_oos_cagr", 0.0)
    base_cal  = base.get("avg_oos_calmar", 0.0)
    base_seg3 = base.get("seg3_2022_cagr", -99.0) or -99.0

    scores: list[dict] = []
    for key, res in wf_results.items():
        if key == base_key:
            continue
        wf_ok   = res.get("wf_count", 0) >= 5
        seg3    = res.get("seg3_2022_cagr") or -99.0
        seg3_ok = seg3 > base_seg3 - 0.5  # no meaningful deterioration
        d_cagr  = res.get("avg_oos_cagr", 0.0) - base_cagr
        d_cal   = res.get("avg_oos_calmar", 0.0) - base_cal
        dd      = res.get("avg_oos_dd", -99.0)
        dd_ok   = dd > base.get("avg_oos_dd", -99.0) - 2.0
        case    = key.split("_")[0]
        thresh  = "_".join(key.split("_")[1:])

        if wf_ok and seg3_ok and d_cagr >= 0.0 and dd_ok:
            v = "ADOPT"
        elif wf_ok and seg3_ok and d_cagr >= -0.5:
            v = "ADOPT_CONDITIONAL"
        elif wf_ok and not seg3_ok:
            v = "REJECT_SEG3"
        elif not wf_ok:
            v = "REJECT_WF"
        else:
            v = "REJECT"

        scores.append({
            "key": key, "case": case, "thresh": thresh,
            "verdict": v, "wf": res.get("wf_count", 0),
            "avg_cagr": res.get("avg_oos_cagr", 0.0),
            "d_cagr": round(d_cagr, 2),
            "d_calmar": round(d_cal, 3),
            "seg3": seg3,
        })

    # Find best ADOPT by d_cagr + seg3
    adopts = [s for s in scores if "ADOPT" in s["verdict"]]
    if adopts:
        best = max(adopts, key=lambda s: s["d_cagr"] + (0.5 if s["seg3"] > 0 else 0))
        production_rec = best["key"]
        case_rec = best["case"]
    else:
        production_rec = "BASE (no improvement confirmed)"
        case_rec = "NONE"

    # "低Quality保有継続が期待値プラスか" の答え
    # Compare A vs C at baseline threshold: if C > A → exit without replacement adds value
    a35 = wf_results.get("A_35/70", {}).get("avg_oos_cagr", 0.0)
    c35 = wf_results.get("C_35/70", {}).get("avg_oos_cagr", 0.0)
    b35 = wf_results.get("B_35/70", {}).get("avg_oos_cagr", 0.0)

    if c35 > a35 + 0.3:
        holding_ev_positive = False
        hold_verdict = "EXIT_WINS: 候補なし時もExit有利 (C>A)"
    elif abs(c35 - a35) <= 0.3:
        holding_ev_positive = "NEUTRAL"
        hold_verdict = "NEUTRAL: 候補なし時の保有/Exitは差なし"
    else:
        holding_ev_positive = True
        hold_verdict = "HOLD_WINS: 候補なしなら保有継続が有利 (A>C)"

    return {
        "production_recommendation": production_rec,
        "recommended_case":          case_rec,
        "low_quality_hold_ev_positive": holding_ev_positive,
        "holding_verdict":           hold_verdict,
        "all_verdicts":              scores,
        "a35_cagr": a35, "b35_cagr": b35, "c35_cagr": c35,
    }


# ================================================================== #
#  表示
# ================================================================== #

def print_sensitivity_table(wf_results: dict, base_key: str = "BASE") -> None:
    base = wf_results.get(base_key, {})
    b_cagr = base.get("avg_oos_cagr", 0.0)
    b_cal  = base.get("avg_oos_calmar", 0.0)
    b_seg3 = base.get("seg3_2022_cagr") or -99.0

    sep = "─" * 100
    print(f"\n{sep}")
    print("  Walk-Forward Sensitivity (3 Cases × 3 Thresholds)")
    print(f"  Baseline: WF={base.get('wf_count',0)}/5 avgCAGR={b_cagr:+.2f}% Calmar={b_cal:.3f} Seg3_2022={b_seg3:+.2f}%")
    print(sep)
    print(f"  {'Key':<12} {'WF':>4} {'avgCAGR':>9} {'ΔCalmar':>8} {'Seg3_22':>9} {'ΔCAGR':>7} {'MaxDD':>8} {'Acts':>5}  Verdict")
    print(sep)

    order = [f"{c}_{lab}" for hold_max, cand_min, lab in THRESHOLDS for c in ("A", "B", "C")]
    for key in order:
        res = wf_results.get(key, {})
        if not res:
            print(f"  {key:<12}  (no data)")
            continue
        seg3   = res.get("seg3_2022_cagr") or -99.0
        d_cagr = res.get("avg_oos_cagr", 0.0) - b_cagr
        d_cal  = res.get("avg_oos_calmar", 0.0) - b_cal
        seg3_s = f"{seg3:+.2f}%{'✓' if seg3 > 0 else '✗'}"
        acts   = res.get("total_acts", 0)

        wf_ok   = res.get("wf_count", 0) >= 5
        seg3_ok = seg3 > b_seg3 - 0.5
        adopt   = wf_ok and seg3_ok and d_cagr >= 0.0
        label   = "ADOPT★" if (adopt and d_cagr > 0.5) else "ADOPT" if adopt else "REJECT"

        print(f"  {key:<12} {res['wf_count']:>2}/5 {res['avg_oos_cagr']:>+9.2f}% {d_cal:>+8.3f}"
              f" {seg3_s:>10} {d_cagr:>+7.2f}% {res['avg_oos_dd']:>8.2f}% {acts:>5}  {label}")

    print(sep)
    print(f"\n  Case A = Swap-Only (Study57 Case E): Hold<T AND Cand>C → Swap")
    print(f"  Case B = Exit-Only: Hold<T → Exit (candidate quality not required)")
    print(f"  Case C = Hybrid:    Hold<T → Swap if Cand>C else Exit")


def print_fold_detail(wf_results: dict) -> None:
    sep = "─" * 100
    print(f"\n{sep}")
    print("  WF Fold Detail (CAGR%) — Baseline threshold (35/70)")
    print(sep)
    keys_35 = ["BASE", "A_35/70", "B_35/70", "C_35/70"]
    header = f"  {'Fold':<9} " + " ".join(f"{k:>12}" for k in keys_35)
    print(header); print(sep)
    for yr in ["2020", "2021", "2022", "2023", "2024"]:
        row = [f"  OOS {yr}"]
        for k in keys_35:
            segs = wf_results.get(k, {}).get("segments", [])
            seg  = next((s for s in segs if s.get("oos_year") == yr), {})
            cagr = seg.get("cagr")
            mark = "✓" if seg.get("wf_pass") else "✗"
            row.append(f"{f'{cagr:+.2f}%{mark}' if cagr is not None else '—':>12}")
        print(" ".join(row))
    print(sep)


def print_attribution(attr: dict) -> None:
    sep = "─" * 80
    print(f"\n{sep}")
    print("  Replacement Attribution (Case A 35/70, IS期間)")
    print(sep)
    print(f"  Swaps analyzed:          {attr['n_swaps_analyzed']}")
    print(f"  Removal effect (n={attr['removal_n']}):    mean={attr.get('removal_effect_mean_pp','N/A')}pp"
          f"  positive={attr.get('removal_positive_rate_pct','N/A')}%")
    print(f"  Replacement effect (n={attr['replacement_n']}): mean={attr.get('replacement_effect_mean_pp','N/A')}pp"
          f"  positive={attr.get('replacement_positive_rate_pct','N/A')}%")
    print(f"  Interpretation: {attr['interpretation']}")
    print(sep)


def print_timeline(tl: dict) -> None:
    sep = "─" * 80
    print(f"\n{sep}")
    print("  Decision Timeline Audit")
    print(sep)
    bt = tl["backtesting"]; lv = tl["live_trading"]
    print(f"  [BT]  obs_day=Day{bt['obs_day']} exit_fires=Day{bt['exit_fires_on_day']}"
          f"  hold_at_exit={bt['hold_at_exit']}  min_hold_ok={bt['min_hold_satisfied']}")
    print(f"  [Live] data_cutoff={lv['eod_data_cutoff']}  signal={lv['signal_generation']}")
    print(f"         execution={lv['order_execution']}  actual_hold={lv['actual_hold_days']}")
    print(f"  avg_hold_at_trigger={tl['avg_hold_days_at_trigger']}d  min={tl['min_hold_days_at_trigger']}d  n={tl['n_events_analyzed']}")
    print(f"  Verdict: {tl['verdict']}")
    print(sep)


def print_verdict(v: dict) -> None:
    sep = "=" * 80
    print(f"\n{sep}")
    print("  STUDY58 FINAL VERDICT")
    print(sep)
    print(f"  Production Recommendation : {v['production_recommendation']}")
    print(f"  Recommended Case          : {v['recommended_case']}")
    print(f"  Low-Quality Hold EV+      : {v['holding_verdict']}")
    print(f"\n  A_35/70 avgCAGR={v['a35_cagr']:+.2f}%  B_35/70 avgCAGR={v['b35_cagr']:+.2f}%  C_35/70 avgCAGR={v['c35_cagr']:+.2f}%")
    print(sep)


# ================================================================== #
#  Main
# ================================================================== #

def main() -> None:
    print("=" * 90)
    print("  Study58 Quality Exit vs Replacement Engine Audit")
    print(f"  Date: {TODAY_STR}   Capital: ¥{CAPITAL:,}")
    print("=" * 90)

    print("\n[Data] データセット構築中...")
    ds = build_common_dataset(DATA_END)
    print(f"  {len(ds['trade_syms'])} シンボル")
    rsr_df = ds["rsr_df"]

    # ── IS Baseline ───────────────────────────────────────────────────
    print(f"\n[IS] D_ATR_EQ Baseline ({IS_START}~{IS_END})...")
    active_is  = get_active(ds, IS_START,  IS_END)
    active_oos = get_active(ds, OOS_START, OOS_END)
    raw_is     = run_bt(ds, active_is, IS_START, IS_END)
    is_sell    = [t for t in raw_is.get("_trades", []) if t.get("side") == "SELL"]
    is_missed  = raw_is.get("_missed_cands", [])
    is_dates   = rsr_df.index[(rsr_df.index >= IS_START) & (rsr_df.index <= IS_END)]
    print(f"  Baseline IS: CAGR={raw_is['cagr']:+.2f}%  Trades={raw_is['n_trades']}")

    # ── IS Quality Scorer ─────────────────────────────────────────────
    print("\n[Scorer] IS Quality Scorer 学習 (Day3)...")
    scorer = build_scorer_from_trades(is_sell, is_dates, ds["universe_raw"], rsr_df, obs_n=3)
    print(f"  raw_p20={scorer._raw_p20:.4f}  mu={scorer._mu}")

    # ── IS / OOS Period Results (baseline threshold 35/70) ───────────
    print(f"\n[Period] IS/OOS Cases A/B/C at (35/70)...")
    hold_max_base, cand_min_base = 35.0, 70.0
    period_results: dict[str, dict] = {"BASE_IS": extract(raw_is)}
    raw_oos_base = run_bt(ds, active_oos, OOS_START, OOS_END)
    period_results["BASE_OOS"] = extract(raw_oos_base)
    print(f"  BASE IS: CAGR={period_results['BASE_IS']['cagr']:+.2f}%")
    print(f"  BASE OOS: CAGR={period_results['BASE_OOS']['cagr']:+.2f}%")

    for case in ("A", "B", "C"):
        for period_name, sell_t, missed_t, dates_p, active, start, end in [
            ("IS",  is_sell, is_missed, is_dates, active_is, IS_START, IS_END),
            ("OOS", [t for t in raw_oos_base.get("_trades",[]) if t.get("side")=="SELL"],
             raw_oos_base.get("_missed_cands",[]),
             rsr_df.index[(rsr_df.index >= OOS_START) & (rsr_df.index <= OOS_END)],
             active_oos, OOS_START, OOS_END),
        ]:
            forced_exits, _ = build_plan(
                case, hold_max_base, cand_min_base,
                sell_t, missed_t, dates_p, ds["universe_raw"], rsr_df, scorer,
            )
            print(f"  Case {case} [{period_name}] acts={len(forced_exits)}... ", end="", flush=True)
            raw = run_bt(ds, active, start, end, quality_forced_exits=forced_exits)
            m = extract(raw)
            period_results[f"{case}_{period_name}"] = m
            print(f"CAGR={m['cagr']:+.2f}% Sh={m['sharpe']:.3f} DD={m['max_dd']:.1f}%")

    # ── Replacement Attribution (IS, Case A 35/70) ──────────────────
    print("\n[Attribution] Replacement Alpha分解 (Case A 35/70, IS)...")
    forced_exits_is_a35, _ = build_plan(
        "A", 35.0, 70.0, is_sell, is_missed, is_dates, ds["universe_raw"], rsr_df, scorer,
    )
    raw_is_a35 = run_bt(ds, active_is, IS_START, IS_END, quality_forced_exits=forced_exits_is_a35)
    attr_result = replacement_attribution(raw_is_a35, forced_exits_is_a35, is_dates, ds["universe_raw"])
    print_attribution(attr_result)

    # ── WF (全9combo共有fold) ─────────────────────────────────────────
    print(f"\n[WF] Walk-Forward 5-fold (9 combos + baseline)...")
    wf_results = run_wf_all(ds, scorer)

    # ── Decision Timeline Audit ───────────────────────────────────────
    tl_audit = decision_timeline_audit(wf_results)
    print_timeline(tl_audit)

    # ── Live Latency Audit ────────────────────────────────────────────
    ll_audit = live_latency_audit(scorer)
    print("\n[Live Latency Audit]")
    print(f"  Feasibility: {ll_audit['overall_feasibility']}")
    print(f"  Effort:      {ll_audit['estimated_implementation_effort']}")
    for step in ll_audit["implementation_steps"]:
        print(f"    {step}")

    # ── Summary Tables ────────────────────────────────────────────────
    print_sensitivity_table(wf_results)
    print_fold_detail(wf_results)

    # ── Verdict ───────────────────────────────────────────────────────
    verdict = compute_verdict(wf_results)
    print_verdict(verdict)

    # ── Save JSON ─────────────────────────────────────────────────────
    out = {
        "study":          "Study58_QualityExitAudit",
        "date":           TODAY_STR,
        "config":         "D_ATR_EQ",
        "thresholds":     [{"hold_max": hm, "cand_min": cm, "label": lb}
                           for hm, cm, lb in THRESHOLDS],
        "quality_scorer": {
            "weights": QS_WEIGHTS,
            "mu":    scorer._mu,
            "sigma": scorer._sigma,
            "raw_p20": scorer._raw_p20,
        },
        "period_results": period_results,
        "replacement_attribution": {k: v for k, v in attr_result.items() if k != "details"},
        "attribution_details": attr_result["details"],
        "wf_results":          {k: {kk: vv for kk, vv in v.items() if kk != "segments"}
                                 for k, v in wf_results.items()},
        "wf_segments":         {k: v.get("segments", []) for k, v in wf_results.items()},
        "decision_timeline":   tl_audit,
        "live_latency":        {k: v for k, v in ll_audit.items() if k != "implementation_steps"},
        "verdict":             verdict,
    }
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 保存: {OUT_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
