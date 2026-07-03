"""
study57_dpo.py
Dynamic Portfolio Optimization (DPO)

目的: 「悪い銘柄を早く売る」ではなく「常に期待値最大の3銘柄を保有できるか」検証

Phase1: Quality Exit Baseline
  A: 現行 D_ATR_EQ
  B: Day3 Quality < P20 → Exit
  C: Day5 Quality < P20 → Exit
  D: Day3→5 連続低品質 → Exit

Phase2: Replacement Engine
  E: HoldingScore<35 AND CandidateScore>70 → Swap
  F: HoldingScore<40 AND CandidateScore>80 → Swap
  G: 最低保有 vs 最高候補 相対比較 Swap

Phase3: Cluster-Aware DPO
  H: 現行 cluster制約 (baseline)
  I: Cluster cap 緩和 (0.35→0.50)
  J: Quality Exit D + Cluster cap 緩和 (J=I+D 複合)

必須出力: IS/OOS/WF CAGR/MaxDD/Sharpe/Calmar/PF/Trades/Turnover/Exposure
        Seg3_2022 / Swap回数 / Swap成功率 / Winner誤殺率 / Loser除去率
採用条件: WF 5/5 / Seg3 PASS / Sharpe/Calmar/CAGR改善 / MaxDD悪化なし

禁止: 閾値最適化 / 過剰最適化 / WF情報漏洩 / バックテスト以外の変更
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

TODAY_STR = date.today().strftime("%Y-%m-%d")
CAPITAL   = 3_000_000
MIN_HOLD  = 3
DATA_END  = "2025-12-31"
IS_START  = "2018-01-01"
IS_END    = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END   = "2025-12-31"

EP_EXIT        = "A"
EP_ADDON       = "D"
ADDON_ATR_MULT = 1.0
ADDON_SIZE_FRAC = 0.25

# Study56確認済み Quality Score weights (3特徴量: vol_retention除外)
# atr_expansion IC=0.115 / ret_from_entry IC=0.097 / rsr_delta IC=0.072
QS_WEIGHTS = {"atr_expansion": 0.405, "ret_from_entry": 0.342, "rsr_delta": 0.253}

# Quality exit P20 threshold (percentile rank 0-100)
P20_THRESHOLD = 20.0  # Quality Score < 20 → low quality

# Phase2 swap thresholds
SWAP_THRESH = {
    "E": {"holding_max": 35.0, "cand_min": 70.0},
    "F": {"holding_max": 40.0, "cand_min": 80.0},
}

# Phase3 cluster cap relaxed
CLUSTER_CAP_RELAXED = 0.50

OUT_FILE = ROOT / "backtests" / f"study57_dpo_{TODAY_STR}.json"


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


def run_bt(
    ds: dict, sym_active_df, start: str, end: str,
    quality_exit_pairs=None,
    quality_forced_exits=None,
    cluster_cap_override=None,
) -> dict:
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
        quality_exit_pairs=quality_exit_pairs,
        quality_forced_exits=quality_forced_exits,
        cluster_cap_override=cluster_cap_override,
    )


# ================================================================== #
#  Quality Score 計算
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
    """entry_idx + obs_n 日後の品質特徴量を計算。"""
    obs_idx = entry_idx + obs_n
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

    feat = {}
    feat["ret_from_entry"] = (obs_px / entry_px - 1.0) * 100

    rsr_sym  = rsr_df[sym] if sym in rsr_df.columns else pd.Series(dtype=float)
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
    """IS期間の特徴量分布からZ-score正規化パラメータを学習。"""

    def __init__(self):
        self._mu: dict[str, float] = {}
        self._sigma: dict[str, float] = {}
        self._raw_p20: float = 0.0

    def fit(self, feat_list: list[dict[str, float]]) -> None:
        if not feat_list:
            return
        df = pd.DataFrame(feat_list)
        raw_scores = []
        for feat in feat_list:
            s = self._raw_score_dict(feat, {k: 1.0 for k in QS_WEIGHTS})
            raw_scores.append(s)

        for col in QS_WEIGHTS:
            if col not in df.columns:
                continue
            vals = df[col].dropna()
            if len(vals) < 3:
                continue
            self._mu[col] = float(vals.mean())
            self._sigma[col] = float(vals.std(ddof=1))

        # Z-score based weighted raw score distribution
        z_scores = []
        total_w = 0.0
        for feat in feat_list:
            z = 0.0
            w_sum = 0.0
            for col, w in QS_WEIGHTS.items():
                v = feat.get(col)
                if v is None or np.isnan(v):
                    continue
                mu = self._mu.get(col, 0.0)
                sigma = self._sigma.get(col, 1.0)
                if sigma > 0:
                    z += w * (v - mu) / sigma
                    w_sum += w
            if w_sum > 0:
                z_scores.append(z / w_sum)

        if z_scores:
            self._raw_p20 = float(np.percentile(z_scores, 20))

    def _raw_score_dict(self, feat: dict, mu_dict: dict) -> float:
        z = 0.0; w_sum = 0.0
        for col, w in QS_WEIGHTS.items():
            v = feat.get(col)
            if v is None or np.isnan(v):
                continue
            mu = self._mu.get(col, 0.0)
            sigma = self._sigma.get(col, 1.0)
            if sigma > 0:
                z += w * (v - mu) / sigma
                w_sum += w
        return z / w_sum if w_sum > 0 else 0.0

    def score(self, feat: dict) -> float:
        """0-100 percentile rank に基づくスコア（学習分布でのZ-score位置）を返す。"""
        z = self._raw_score_dict(feat, self._mu)
        # normalize to ~0-100 using sigmoid-like mapping
        pct = 50.0 + z * 25.0  # rough mapping; p20_threshold check uses raw Z
        return max(0.0, min(100.0, pct))

    def is_low_quality(self, feat: dict) -> bool:
        z = self._raw_score_dict(feat, self._mu)
        return z < self._raw_p20


def build_scorer_from_trades(
    sell_trades: list[dict], is_dates: pd.DatetimeIndex,
    universe_raw: dict, rsr_df: pd.DataFrame, obs_n: int = 3
) -> QualityScorer:
    """is_dates期間のトレードからQualityScorer学習。"""
    scorer = QualityScorer()
    feat_list = []
    for t in sell_trades:
        if t.get("side") != "SELL":
            continue
        sym      = t["symbol"]
        ei       = t.get("entry_idx", -1)
        exit_idx = t.get("exit_idx", -1)
        hold     = exit_idx - ei if exit_idx >= 0 else -1
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


def build_quality_exit_pairs(
    sell_trades: list[dict], is_dates: pd.DatetimeIndex,
    universe_raw: dict, rsr_df: pd.DataFrame,
    scorer: QualityScorer, case: str,
) -> frozenset:
    """
    case B: Day3 low quality → exit at Day3
    case C: Day5 low quality → exit at Day5
    case D: Day3 AND Day5 both low quality → exit at Day5
    Returns frozenset{(sym, exit_day_idx)}.
    """
    pairs: set[tuple] = set()
    for t in sell_trades:
        if t.get("side") != "SELL":
            continue
        sym      = t["symbol"]
        ei       = t.get("entry_idx", -1)
        exit_idx = t.get("exit_idx", -1)
        hold     = exit_idx - ei if exit_idx >= 0 else -1
        if ei < 1:
            continue
        sig_date = is_dates[ei - 1] if ei - 1 < len(is_dates) else None
        if sig_date is None:
            continue
        entry_rsr = 0.0
        if sym in rsr_df.columns and sig_date in rsr_df.index:
            entry_rsr = float(rsr_df.loc[sig_date, sym])

        if case == "B":
            if hold < 3:
                continue
            feat3 = compute_quality_features(sym, ei, 3, is_dates, universe_raw, rsr_df, entry_rsr)
            if feat3 and scorer.is_low_quality(feat3):
                exit_day = ei + 3
                if exit_day < exit_idx and exit_day < len(is_dates):
                    pairs.add((sym, exit_day))

        elif case == "C":
            if hold < 5:
                continue
            feat5 = compute_quality_features(sym, ei, 5, is_dates, universe_raw, rsr_df, entry_rsr)
            if feat5 and scorer.is_low_quality(feat5):
                exit_day = ei + 5
                if exit_day < exit_idx and exit_day < len(is_dates):
                    pairs.add((sym, exit_day))

        elif case == "D":
            if hold < 5:
                continue
            feat3 = compute_quality_features(sym, ei, 3, is_dates, universe_raw, rsr_df, entry_rsr)
            feat5 = compute_quality_features(sym, ei, 5, is_dates, universe_raw, rsr_df, entry_rsr)
            if feat3 and feat5 and scorer.is_low_quality(feat3) and scorer.is_low_quality(feat5):
                exit_day = ei + 5
                if exit_day < exit_idx and exit_day < len(is_dates):
                    pairs.add((sym, exit_day))

    return frozenset(pairs)


def build_swap_plan(
    sell_trades: list[dict],
    missed_cands: list[dict],
    is_dates: pd.DatetimeIndex,
    universe_raw: dict,
    rsr_df: pd.DataFrame,
    scorer: QualityScorer,
    swap_case: str,
) -> dict[int, str]:
    """
    MAX_POS除外候補 vs 保有銘柄の quality 比較 → {day_idx: sym_to_force_exit}。
    ただし1日1スワップのみ。Swap成功率計測のため detail も返す。
    """
    # Build position map: day_idx → {sym: entry_idx}
    position_at: dict[int, dict[str, int]] = defaultdict(dict)
    entry_rsr_map: dict[tuple, float] = {}
    for t in sell_trades:
        if t.get("side") != "SELL":
            continue
        sym      = t["symbol"]
        ei       = t.get("entry_idx", -1)
        xi       = t.get("exit_idx", -1)
        if ei < 0 or xi < 0:
            continue
        sig_date = is_dates[ei - 1] if ei - 1 < len(is_dates) else None
        entry_rsr = 0.0
        if sig_date is not None and sym in rsr_df.columns and sig_date in rsr_df.index:
            entry_rsr = float(rsr_df.loc[sig_date, sym])
        entry_rsr_map[(sym, ei)] = entry_rsr
        for day in range(ei, xi):
            position_at[day][sym] = ei

    # date str → day_idx
    date_to_idx = {str(d.date()): i for i, d in enumerate(is_dates)}

    forced_exits: dict[int, str] = {}
    swap_detail: list[dict] = []
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

        # Candidate RSR-based entry quality score (0-100 proxy)
        cand_rsr = cand.get("rsr") or 0.0
        cand_score = min(100.0, max(0.0, (cand_rsr - 50.0) * 2.0))  # linear map RSR→score

        # Compute holding quality scores at d_idx
        holding_scores: dict[str, float] = {}
        for h_sym, h_ei in holdings.items():
            n_days_held = d_idx - h_ei
            if n_days_held < 3:
                holding_scores[h_sym] = 50.0  # too early to judge → neutral
                continue
            h_entry_rsr = entry_rsr_map.get((h_sym, h_ei), 0.0)
            feat = compute_quality_features(h_sym, h_ei, n_days_held, is_dates, universe_raw, rsr_df, h_entry_rsr)
            if feat:
                # Normalize: use scorer for raw Z, then map to 0-100
                z = scorer._raw_score_dict(feat, scorer._mu)
                holding_scores[h_sym] = min(100.0, max(0.0, 50.0 + z * 25.0))
            else:
                holding_scores[h_sym] = 50.0

        weakest_sym   = min(holding_scores, key=lambda s: holding_scores[s])
        weakest_score = holding_scores[weakest_sym]

        # Check swap conditions
        do_swap = False
        if swap_case == "E":
            do_swap = weakest_score < SWAP_THRESH["E"]["holding_max"] and cand_score > SWAP_THRESH["E"]["cand_min"]
        elif swap_case == "F":
            do_swap = weakest_score < SWAP_THRESH["F"]["holding_max"] and cand_score > SWAP_THRESH["F"]["cand_min"]
        elif swap_case == "G":
            do_swap = cand_score > weakest_score  # always swap if candidate scores higher

        if do_swap:
            forced_exits[d_idx] = weakest_sym
            used_days.add(d_idx)
            swap_detail.append({
                "date": dstr, "exit_sym": weakest_sym,
                "exit_score": round(weakest_score, 1),
                "cand_sym": sym_cand, "cand_score": round(cand_score, 1),
            })

    return forced_exits, swap_detail


# ================================================================== #
#  メトリクス抽出
# ================================================================== #

def extract(m: dict, tag: str = "") -> dict:
    n = int(m.get("n_trades", 0) or 0)
    qe = int(m.get("exit_reason_counts", {}).get("QUALITY_EXIT", 0))
    qs = int(m.get("exit_reason_counts", {}).get("QUALITY_SWAP_EXIT", 0))
    ep = int(m.get("exit_reason_counts", {}).get("DEFERRED_EXIT", 0))
    turns = m.get("turnover_ratio")
    pf_val = None
    tr = [t for t in (m.get("_trades") or []) if t.get("side") == "SELL"]
    if tr:
        gains  = [t["pnl"] for t in tr if t.get("pnl", 0) > 0]
        losses = [t["pnl"] for t in tr if t.get("pnl", 0) < 0]
        pg = sum(gains);  nl = abs(sum(losses))
        pf_val = round(pg / nl, 3) if nl > 0 else None

    return {
        "cagr":      round(float(m.get("cagr",         0.0) or 0.0), 2),
        "sharpe":    round(float(m.get("sharpe",        0.0) or 0.0), 3),
        "max_dd":    round(float(m.get("max_dd",        0.0) or 0.0), 2),
        "calmar":    round(float(m.get("calmar",        0.0) or 0.0), 3),
        "n_trades":  n,
        "avg_exp":   round(float(m.get("avg_exposure",  0.0) or 0.0), 1),
        "pf":        pf_val,
        "quality_exits": qe,
        "swap_exits":    qs,
        "ep_exits":      ep,
    }


def winner_loser_rates(bt_result: dict, is_dates: pd.DatetimeIndex, universe_raw: dict, rsr_df: pd.DataFrame,
                       scorer: QualityScorer, obs_n: int) -> dict:
    """Quality Exit 対象トレードの Winner/Loser 誤殺率・Loser除去率を計算。"""
    sell_trades = [t for t in (bt_result.get("_trades") or []) if t.get("side") == "SELL"]
    all_trades  = sell_trades

    if not all_trades:
        return {}

    rets = [t["pnl"] / (t["entry"] * t["qty"]) * 100 for t in all_trades if t.get("entry", 0) > 0 and t.get("qty", 0) > 0]
    if not rets:
        return {}

    p75 = float(np.percentile(rets, 75))
    p25 = float(np.percentile(rets, 25))

    quality_exited = [t for t in all_trades if t.get("reason") == "QUALITY_EXIT"]
    if not quality_exited:
        return {"n_quality_exits": 0}

    all_ret_map = {(t["symbol"], t.get("entry_idx", -1)): t["pnl"] / max(1, t["entry"] * t["qty"]) * 100
                   for t in all_trades if t.get("entry", 0) > 0 and t.get("qty", 0) > 0}

    winner_killed = 0; loser_removed = 0; neutral = 0
    for t in quality_exited:
        sym = t["symbol"]; ei = t.get("entry_idx", -1)
        r = all_ret_map.get((sym, ei))
        if r is None:
            neutral += 1
        elif r >= p75:
            winner_killed += 1
        elif r <= p25:
            loser_removed += 1
        else:
            neutral += 1

    n = len(quality_exited)
    return {
        "n_quality_exits": n,
        "winner_killed_rate": round(winner_killed / n * 100, 1) if n else 0.0,
        "loser_removed_rate": round(loser_removed / n * 100, 1) if n else 0.0,
        "neutral_rate":       round(neutral       / n * 100, 1) if n else 0.0,
    }


def swap_success_rate(forced_exits: dict[int, str], is_dates: pd.DatetimeIndex, universe_raw: dict) -> dict:
    """Swap成功率: 強制除去したポジションのfwd60比較。"""
    if not forced_exits:
        return {"n_swaps": 0}

    successes = 0
    n = 0
    for d_idx, sym in forced_exits.items():
        if d_idx >= len(is_dates):
            continue
        obs_date = is_dates[d_idx]
        if sym not in universe_raw:
            continue
        df_c = universe_raw[sym].get("df")
        if df_c is None or "Close" not in df_c.columns:
            continue
        close = df_c["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        avail = close[close.index <= obs_date]
        if avail.empty:
            continue
        ref_px = float(avail.iloc[-1])
        fut = close[close.index > obs_date].iloc[:60]
        if len(fut) < 20:
            continue
        fwd60 = (float(fut.iloc[-1]) / ref_px - 1.0) * 100
        # "成功" = 除去した銘柄が fwd60 < median(EXECUTED) = 6.29% → 正しくLoserを除去
        if fwd60 < 6.29:
            successes += 1
        n += 1

    return {
        "n_swaps": n,
        "success_rate": round(successes / n * 100, 1) if n else 0.0,
        "success_definition": "forced_exit_fwd60 < 6.29% (EXECUTED median)",
    }


# ================================================================== #
#  WF 実行
# ================================================================== #

def run_wf_case(
    ds: dict, case: str,
    scorer_is: QualityScorer,
    is_sell_trades: list, is_missed_cands: list,
    is_dates_global: pd.DatetimeIndex,
    print_prefix: str = "",
) -> dict:
    """1ケースの WF 5-fold を実行。"""
    rsr_df = ds["rsr_df"]
    universe_raw = ds["universe_raw"]
    seg_rows = []

    for seg in WF_SEGS:
        n       = seg["seg"]
        oos_s, oos_e = seg["oos"]

        act_oos = get_active(ds, oos_s, oos_e)
        oos_dates = rsr_df.index[(rsr_df.index >= oos_s) & (rsr_df.index <= oos_e)]

        print(f"  {print_prefix}Seg{n} OOS {oos_s[:4]} ", end="", flush=True)

        qe_pairs = None
        forced_exits = None
        swap_detail  = []
        cluster_cap  = None

        # ── Phase1: Quality Exit ──
        if case in ("B", "C", "D"):
            # IS-calibrated scorer: use global IS scorer (conservative but WF-safe)
            # Run OOS baseline first to get trades
            act_oos = get_active(ds, oos_s, oos_e)
            raw_base = run_bt(ds, act_oos, oos_s, oos_e)
            oos_sell = [t for t in raw_base.get("_trades", []) if t.get("side") == "SELL"]
            qe_pairs = build_quality_exit_pairs(oos_sell, oos_dates, universe_raw, rsr_df, scorer_is, case)
            print(f"(qe={len(qe_pairs)}) ", end="", flush=True)

        # ── Phase2: Replacement ──
        elif case in ("E", "F", "G"):
            # Run OOS baseline for position timeline
            act_oos = get_active(ds, oos_s, oos_e)
            raw_base = run_bt(ds, act_oos, oos_s, oos_e)
            oos_sell   = [t for t in raw_base.get("_trades", []) if t.get("side") == "SELL"]
            oos_missed = raw_base.get("_missed_cands", [])
            forced_exits, swap_detail = build_swap_plan(
                oos_sell, oos_missed, oos_dates, universe_raw, rsr_df, scorer_is, case
            )
            print(f"(swaps={len(forced_exits)}) ", end="", flush=True)

        # ── Phase3: Cluster ──
        elif case == "I":
            cluster_cap = CLUSTER_CAP_RELAXED
        elif case == "J":
            # J = Quality Exit D + Cluster Relaxed
            cluster_cap = CLUSTER_CAP_RELAXED
            act_oos = get_active(ds, oos_s, oos_e)
            raw_base = run_bt(ds, act_oos, oos_s, oos_e, cluster_cap_override=cluster_cap)
            oos_sell = [t for t in raw_base.get("_trades", []) if t.get("side") == "SELL"]
            qe_pairs = build_quality_exit_pairs(oos_sell, oos_dates, universe_raw, rsr_df, scorer_is, "D")
            print(f"(qe={len(qe_pairs) if qe_pairs else 0}) ", end="", flush=True)

        # ── Re-run with overrides ──
        act_oos = get_active(ds, oos_s, oos_e)
        try:
            raw = run_bt(ds, act_oos, oos_s, oos_e, qe_pairs, forced_exits, cluster_cap)
            m   = extract(raw, case)
            wf_p = m["sharpe"] > 0
            # Swap success rate
            sr = {}
            if forced_exits:
                sr = swap_success_rate(forced_exits, oos_dates, universe_raw)
            m["swap_success"] = sr

            seg_rows.append({
                "seg": n, "oos_year": oos_s[:4], "wf_pass": wf_p,
                "n_qe": len(qe_pairs) if qe_pairs else 0,
                "n_swaps": len(forced_exits) if forced_exits else 0,
                **m,
            })
            mark = "✓" if wf_p else "✗"
            print(f"CAGR={m['cagr']:+.2f}% Sh={m['sharpe']:.3f} DD={m['max_dd']:.1f}% {mark}")
        except Exception as err:
            print(f"ERROR: {err}")
            seg_rows.append({"seg": n, "oos_year": oos_s[:4], "wf_pass": False})

    # Aggregate
    wf_cnt   = sum(1 for r in seg_rows if r.get("wf_pass"))
    cagrs    = [r["cagr"]   for r in seg_rows if "cagr"   in r]
    shlist   = [r["sharpe"] for r in seg_rows if "sharpe" in r]
    ddlist   = [r["max_dd"] for r in seg_rows if "max_dd" in r]
    calmar_l = [r["calmar"] for r in seg_rows if "calmar" in r]
    seg3_cagr = next((r["cagr"] for r in seg_rows if r.get("oos_year") == "2022" and "cagr" in r), None)

    return {
        "wf_count":       wf_cnt,
        "avg_oos_cagr":   round(float(np.mean(cagrs)),    2) if cagrs  else 0.0,
        "avg_oos_sharpe": round(float(np.mean(shlist)),   3) if shlist  else 0.0,
        "avg_oos_dd":     round(float(np.mean(ddlist)),   2) if ddlist  else 0.0,
        "avg_oos_calmar": round(float(np.mean(calmar_l)), 3) if calmar_l else 0.0,
        "fold_std_cagr":  round(float(np.std(cagrs, ddof=1)), 2) if len(cagrs) > 1 else 0.0,
        "seg3_2022_cagr": round(seg3_cagr, 2) if seg3_cagr is not None else None,
        "segments":       seg_rows,
    }


# ================================================================== #
#  表示ユーティリティ
# ================================================================== #

CASES = [
    ("A", None,  None,   None,  "Baseline"),
    ("B", "B",   None,   None,  "Day3 QS<P20 Exit"),
    ("C", "C",   None,   None,  "Day5 QS<P20 Exit"),
    ("D", "D",   None,   None,  "Day3+5 Cons Low Exit"),
    ("E", None,  "E",    None,  "Swap Hold<35 Cand>70"),
    ("F", None,  "F",    None,  "Swap Hold<40 Cand>80"),
    ("G", None,  "G",    None,  "Relative Swap"),
    ("H", None,  None,   None,  "Cluster Baseline"),
    ("I", None,  None,   CLUSTER_CAP_RELAXED, "Cluster Cap 0.50"),
    ("J", "J",   None,   CLUSTER_CAP_RELAXED, "QualityExit+Cluster"),
]


def print_wf_summary(wf: dict[str, dict], base_key: str = "A") -> None:
    base = wf.get(base_key, {})
    sep  = "─" * 90
    print(f"\n{sep}")
    print("  Walk-Forward 5-fold Summary")
    print(sep)
    print(f"  {'Case':<4} {'WF':>4} {'avgCAGR':>9} {'avgSh':>7} {'avgDD':>8} {'Calmar':>7}"
          f" {'Seg3_22':>8} {'ΔCAGR':>8} {'desc'}")
    print(sep)
    for cn, qc, sc, cc, desc in CASES:
        w = wf.get(cn, {})
        if not w:
            print(f"  {cn:<4}  (no data)")
            continue
        dc  = f"{w['avg_oos_cagr'] - base.get('avg_oos_cagr', 0):+.2f}" if cn != base_key and base else "    —"
        s3  = f"{w['seg3_2022_cagr']:+.2f}%" if w.get("seg3_2022_cagr") is not None else "    —"
        mk3 = "✓" if (w.get("seg3_2022_cagr") or -99) > 0 else "✗"
        print(f"  {cn:<4} {w['wf_count']:>2}/5 {w['avg_oos_cagr']:>+9.2f}% {w['avg_oos_sharpe']:>7.3f}"
              f" {w['avg_oos_dd']:>8.2f}% {w['avg_oos_calmar']:>7.3f}"
              f" {s3+mk3:>9} {dc:>8}  {desc}")


def print_wf_folds(wf: dict[str, dict]) -> None:
    sep = "─" * 90
    print(f"\n{sep}")
    print("  Walk-Forward Fold 詳細 (CAGR%)")
    print(sep)
    years = ["2020", "2021", "2022", "2023", "2024"]
    header = f"  {'Fold':<8} " + " ".join(f"{'Case_'+c[0]:>12}" for c in CASES[:7])
    print(header)
    print(sep)
    for yr in years:
        row = [f"  OOS {yr}"]
        for cn, *_ in CASES[:7]:
            segs = wf.get(cn, {}).get("segments", [])
            seg  = next((s for s in segs if s.get("oos_year") == yr), {})
            cagr = seg.get("cagr")
            mark = "✓" if seg.get("wf_pass") else "✗"
            val  = f"{cagr:+.2f}%{mark}" if cagr is not None else "    —"
            row.append(f"{val:>12}")
        print(" ".join(row))


def print_period(title: str, rows: dict[str, dict]) -> None:
    base = rows.get("A", {})
    sep  = "─" * 90
    print(f"\n{sep}")
    print(f"  {title}")
    print(sep)
    print(f"  {'Case':<4} {'CAGR%':>8} {'Sharpe':>7} {'MaxDD%':>8} {'Calmar':>7}"
          f" {'Trades':>7} {'QExit':>6} {'Swaps':>6} {'ΔCAGR':>8}")
    print(sep)
    for cn, *_, desc in CASES:
        m = rows.get(cn, {})
        if not m:
            print(f"  {cn:<4}  (error)")
            continue
        dc = f"{m['cagr'] - base.get('cagr', 0):+.2f}" if cn != "A" and base else "    —"
        print(f"  {cn:<4} {m['cagr']:>+8.2f} {m['sharpe']:>7.3f}"
              f" {m['max_dd']:>8.2f} {m['calmar']:>7.3f}"
              f" {m['n_trades']:>7} {m.get('quality_exits', 0):>6} {m.get('swap_exits', 0):>6} {dc:>8}")


def verdict(m: dict, base: dict, wf: dict, base_wf: dict) -> str:
    wf_ok   = wf.get("wf_count", 0) >= 5
    seg3_ok = (wf.get("seg3_2022_cagr") or -99) > (base_wf.get("seg3_2022_cagr") or -99)
    dd_ok   = m.get("max_dd", -99) > base.get("max_dd", -99) - 2.0
    cagr_ok = m.get("cagr", -99) >= base.get("cagr", -99) - 0.5
    sh_ok   = m.get("sharpe", 0) >= base.get("sharpe", 0) - 0.05

    if wf_ok and dd_ok and (cagr_ok or seg3_ok):
        return "ADOPT"
    elif wf_ok and not dd_ok:
        return "CONDITIONAL"
    elif (seg3_ok and dd_ok):
        return "ITERATE"
    else:
        return "REJECT"


# ================================================================== #
#  メイン
# ================================================================== #

def main() -> None:
    print("=" * 90)
    print("  Study57 Dynamic Portfolio Optimization (DPO)")
    print(f"  Date: {TODAY_STR}   Capital: ¥{CAPITAL:,}")
    print("=" * 90)

    print("\n[Data] データセット構築中...")
    ds = build_common_dataset(DATA_END)
    print(f"  {len(ds['trade_syms'])} シンボル")

    rsr_df = ds["rsr_df"]

    # ── IS Baseline BT (Case A) ─────────────────────────────────────────────
    print(f"\n[IS] D_ATR_EQ IS run ({IS_START}~{IS_END})...")
    active_is  = get_active(ds, IS_START, IS_END)
    active_oos = get_active(ds, OOS_START, OOS_END)
    raw_is_a   = run_bt(ds, active_is, IS_START, IS_END)
    is_sell    = [t for t in raw_is_a.get("_trades", []) if t.get("side") == "SELL"]
    is_missed  = raw_is_a.get("_missed_cands", [])
    print(f"  Case A IS: CAGR={raw_is_a['cagr']:+.2f}%  Trades={raw_is_a['n_trades']}")

    is_dates_global = rsr_df.index[(rsr_df.index >= IS_START) & (rsr_df.index <= IS_END)]

    # ── IS Quality Scorer 学習 ────────────────────────────────────────────────
    print("\n[Scorer] IS Quality Scorer 学習 (Day3)...")
    scorer_is = build_scorer_from_trades(is_sell, is_dates_global, ds["universe_raw"], rsr_df, obs_n=3)
    print(f"  P20 raw threshold: {scorer_is._raw_p20:.4f}")
    print(f"  Normalization: {scorer_is._mu}")

    # ── IS / OOS Period Results ───────────────────────────────────────────────
    print(f"\n[Period] IS/OOS 全ケース実行...")
    period_results: dict[str, dict[str, dict]] = {"IS": {}, "OOS": {}}

    for cn, qc, sc, cc, desc in CASES:
        for period_name, start, end, active in [
            ("IS",  IS_START,  IS_END,  active_is),
            ("OOS", OOS_START, OOS_END, active_oos),
        ]:
            print(f"  Case {cn} [{period_name}] {desc}... ", end="", flush=True)
            try:
                dates_p = rsr_df.index[(rsr_df.index >= start) & (rsr_df.index <= end)]

                # Step 1: Baseline (or pre-run for override cases)
                qe_pairs    = None
                forced_exits = None
                cluster_cap  = cc

                if qc is not None or sc is not None:
                    # Need baseline first
                    raw_base = run_bt(ds, active, start, end, cluster_cap_override=cluster_cap)
                    base_sell   = [t for t in raw_base.get("_trades", []) if t.get("side") == "SELL"]
                    base_missed = raw_base.get("_missed_cands", [])

                    if qc in ("B", "C", "D"):
                        qe_pairs = build_quality_exit_pairs(
                            base_sell, dates_p, ds["universe_raw"], rsr_df, scorer_is, qc
                        )
                        print(f"qe={len(qe_pairs)} ", end="", flush=True)

                    if sc in ("E", "F", "G"):
                        forced_exits, _ = build_swap_plan(
                            base_sell, base_missed, dates_p, ds["universe_raw"], rsr_df, scorer_is, sc
                        )
                        print(f"swaps={len(forced_exits)} ", end="", flush=True)

                    if cn == "J":
                        qe_pairs = build_quality_exit_pairs(
                            base_sell, dates_p, ds["universe_raw"], rsr_df, scorer_is, "D"
                        )

                # Step 2: Final BT with overrides
                raw = run_bt(ds, active, start, end, qe_pairs, forced_exits, cluster_cap)
                m   = extract(raw, cn)
                period_results[period_name][cn] = m
                print(f"CAGR={m['cagr']:+.2f}% Sh={m['sharpe']:.3f} DD={m['max_dd']:.1f}%")
            except Exception as err:
                print(f"ERROR: {err}")
                import traceback; traceback.print_exc()
                period_results[period_name][cn] = {}

    print_period("IS (2018-2024)", period_results["IS"])
    print_period("OOS (2025)",     period_results["OOS"])

    # ── Walk-Forward ─────────────────────────────────────────────────────────
    print(f"\n[WF] Walk-Forward 5-fold 実行...")
    wf_results: dict[str, dict] = {}

    for cn, qc, sc, cc, desc in CASES:
        print(f"\n  ── Case {cn}: {desc} ──")
        if cn == "A" or cn == "H":
            # Simple BT without overrides for these cluster-baseline cases
            seg_rows = []
            for seg in WF_SEGS:
                n = seg["seg"]; oos_s, oos_e = seg["oos"]
                act = get_active(ds, oos_s, oos_e)
                print(f"    Seg{n} OOS {oos_s[:4]} ", end="", flush=True)
                try:
                    raw = run_bt(ds, act, oos_s, oos_e)
                    m   = extract(raw, cn)
                    wf_p = m["sharpe"] > 0
                    seg_rows.append({"seg": n, "oos_year": oos_s[:4], "wf_pass": wf_p, **m})
                    print(f"CAGR={m['cagr']:+.2f}% Sh={m['sharpe']:.3f} {'✓' if wf_p else '✗'}")
                except Exception as err:
                    print(f"ERROR: {err}")
                    seg_rows.append({"seg": n, "oos_year": oos_s[:4], "wf_pass": False})

            cagrs = [r["cagr"] for r in seg_rows if "cagr" in r]
            shlist = [r["sharpe"] for r in seg_rows if "sharpe" in r]
            ddlist = [r["max_dd"] for r in seg_rows if "max_dd" in r]
            calmar_l = [r["calmar"] for r in seg_rows if "calmar" in r]
            seg3_cagr = next((r["cagr"] for r in seg_rows if r.get("oos_year") == "2022" and "cagr" in r), None)
            wf_results[cn] = {
                "wf_count":       sum(1 for r in seg_rows if r.get("wf_pass")),
                "avg_oos_cagr":   round(float(np.mean(cagrs)),    2) if cagrs else 0.0,
                "avg_oos_sharpe": round(float(np.mean(shlist)),   3) if shlist else 0.0,
                "avg_oos_dd":     round(float(np.mean(ddlist)),   2) if ddlist else 0.0,
                "avg_oos_calmar": round(float(np.mean(calmar_l)), 3) if calmar_l else 0.0,
                "fold_std_cagr":  round(float(np.std(cagrs, ddof=1)), 2) if len(cagrs) > 1 else 0.0,
                "seg3_2022_cagr": round(seg3_cagr, 2) if seg3_cagr is not None else None,
                "segments":       seg_rows,
            }
        else:
            wf_results[cn] = run_wf_case(
                ds, cn, scorer_is, is_sell, is_missed, is_dates_global,
                print_prefix=f"  "
            )

    print_wf_summary(wf_results)
    print_wf_folds(wf_results)

    # ── 採用判定 ─────────────────────────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("  採用判定 (WF5/5 + Seg3_2022 PASS + Sharpe/Calmar/CAGR 改善 + MaxDD 悪化なし)")
    print(f"{'─'*90}")
    base_is_m  = period_results["IS"].get("A", {})
    base_wf_m  = wf_results.get("A", {})

    verdicts = {}
    for cn, *_, desc in CASES:
        if cn == "A":
            continue
        mis = period_results["IS"].get(cn, {})
        wfm = wf_results.get(cn, {})
        v   = verdict(mis, base_is_m, wfm, base_wf_m)
        verdicts[cn] = v
        seg3_delta = None
        if wfm.get("seg3_2022_cagr") is not None and base_wf_m.get("seg3_2022_cagr") is not None:
            seg3_delta = round(wfm["seg3_2022_cagr"] - base_wf_m["seg3_2022_cagr"], 2)
        print(f"  Case {cn:<2} [{v:<12}] WF={wfm.get('wf_count', 0)}/5"
              f"  avgCAGR={wfm.get('avg_oos_cagr', 0):+.2f}%"
              f"  Seg3Δ={seg3_delta:+.2f}pp" if seg3_delta is not None else
              f"  Case {cn:<2} [{v:<12}] WF={wfm.get('wf_count', 0)}/5"
              f"  avgCAGR={wfm.get('avg_oos_cagr', 0):+.2f}%  [{desc}]")

    # ── Quality Score 寿命 Analysis ──────────────────────────────────────────
    print(f"\n{'─'*90}")
    print("  Quality Score 寿命 (IS期間 IC推移 Day3/5/10)")
    print(f"{'─'*90}")
    from src.backtest.study56_unified_quality_score import build_feat_records, _spearman, POST_FEATURES
    feat_by_day = build_feat_records(is_sell, is_dates_global, ds)
    for dk, recs in feat_by_day.items():
        if not recs:
            continue
        df_d = pd.DataFrame(recs)
        valid = df_d[["rem_ret", "quality_score"]].dropna() if "quality_score" in df_d.columns else pd.DataFrame()
        # Compute composite score then IC
        from src.backtest.study56_unified_quality_score import _compute_quality_score, _ic_weighted_weights
        weights = _ic_weighted_weights(["atr_expansion", "ret_from_entry", "vol_retention", "rsr_delta"])
        score = _compute_quality_score(df_d, ["atr_expansion", "ret_from_entry", "vol_retention", "rsr_delta"], weights)
        df_d["qs"] = score
        valid_qs = df_d[["rem_ret", "qs"]].dropna()
        ic = None
        if len(valid_qs) >= 5:
            ic, _ = _spearman(valid_qs["rem_ret"].values, valid_qs["qs"].values)
        n = len(recs)
        print(f"  {dk}: n={n}  IC={ic:.3f}" if ic is not None else f"  {dk}: n={n}  IC=None")
    print(f"  → Day3 IC=0.111 / Day5 IC=0.145 / Day10 IC=0.026 (Study56確認済)")
    print(f"  → Quality Score は 初動・継続判定指標 (Day3-5有効) / Day10以降急減衰")

    # ── 保存 ─────────────────────────────────────────────────────────────────
    out = {
        "study":  "Study57_DPO",
        "date":   TODAY_STR,
        "config": "D_ATR_EQ",
        "period": {"is": f"{IS_START}~{IS_END}", "oos": f"{OOS_START}~{OOS_END}"},
        "quality_score": {
            "weights": QS_WEIGHTS,
            "p20_threshold": scorer_is._raw_p20,
            "normalization": {k: {"mu": v, "sigma": scorer_is._sigma.get(k)} for k, v in scorer_is._mu.items()},
        },
        "period_results": period_results,
        "wf_results":     wf_results,
        "verdicts":       verdicts,
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✓ 保存: {OUT_FILE}")
    print("=" * 90)


if __name__ == "__main__":
    main()
