"""
backtest/capital_allocation_abc.py

Capital Deployment Architecture A/B/C Comparison

固定: RSR / Dynamic Universe / Sector Caps / Turtle Entry / Exit Logic
変更: 資本配分ロジックのみ

Pattern A  Baseline   現行 (cash/effective_slots, max_single_weight cap)
Pattern B  FullExp    常にフルポジ均等ウェイト / max_positions到達後は最下位RSRを入替
Pattern C  Conviction 初回エントリー70%集中 / 高確信度ローテーション

実行:
  cd C:/ai-trading
  python src/backtest/capital_allocation_abc.py
"""

from __future__ import annotations

import os, sys, json, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

# ── shared infrastructure ────────────────────────────────────────────
from src.paths import BACKTEST_DATASET_DIR, CONFIGS_DIR, DEFAULT_DATA_VERSION, RESULTS_DIR
from src.config_loader import load_strategy_config
from src.backtest.rsr import calc_universe_rsr, calc_composite_alpha_matrix
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.universe_builder import download_universe
from src.strategy.universe import build_dyn_rsr42_active
from src.strategy.cluster import CLUSTER_MAP_DEFAULT, get_cluster, calc_cluster_weight

# ── Entry Timing (optional; FAIL_OPEN if unavailable) ────────────────
try:
    from src.entry.entry_timing_engine import (
        build_entry_timing_input_from_df as _et_build,
        compute_entry_timing             as _et_score,
        CONFIDENCE_LOW                   as _ET_LOW,
    )
    _ET_AVAILABLE = True
except Exception:
    _ET_AVAILABLE = False

# ── constants (must match composite_alpha_bt.py) ─────────────────────
RSR_UNIVERSE_CSV      = CONFIGS_DIR / "rsr_universe_42.csv"

IS_START  = "2018-01-01"
IS_END    = "2024-12-31"
OOS_START = "2025-01-01"
OOS_END   = "2025-12-31"

LOT           = 100
SLIPPAGE      = 0.001
COMMISSION    = 0.00055
COST_ONE_WAY  = SLIPPAGE + COMMISSION
REENTRY_COOL  = 5
CB_UNLOCK_DAYS = 30
CB_SCALE       = 0.35

# conviction ratio for Pattern C
CONVICTION_FIRST_RATIO = 0.70   # 初回エントリー: 稼働資本の70%
CONVICTION_TARGET_EXP  = 0.85   # 目標稼働率 85%

# Pattern D: Dynamic Capital Concentration
DCC_TARGET_EXPOSURE = 0.90      # 目標稼働率 90%（ポジション数で均等割）
# 1 pos → 90%, 2 pos → 45%+45%, 3 pos → 30%+30%+30%

SECTOR_STRATEGY: dict[str, str] = {
    "海運": "fujiko", "機械": "fujiko", "電機精密": "fujiko",
    "商社": "fujiko", "電機": "fujiko", "ゲーム": "fujiko",
    "レジャー": "fujiko", "食品": "fujiko",
    "ガス": "mean_rev", "鉄鋼": "mean_rev", "銀行": "mean_rev",
    "保険": "mean_rev", "輸送機器": "mean_rev", "化学": "mean_rev",
    "小売": "mean_rev",
}
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

OUTPUT_DIR = Path("backtests")


# ─────────────────────────────────────────────────────────────────────
#  ENTRY TIMING HELPER (backtest-safe, no lookahead)
# ─────────────────────────────────────────────────────────────────────

def _compute_et_bt(
    buy_cands:   list,
    signal_date: "pd.Timestamp",
    universe_raw: dict,
    rsr_df:      "pd.DataFrame",
    rsr_mat:     "np.ndarray",
    i:           int,
    sym_to_i:    dict,
    active_syms: list,
) -> dict:
    """
    Entry Timing scores for backtest buy candidates.
    Slices universe_raw to signal_date (no lookahead).
    Returns {sym: EntryTimingResult}. FAIL_OPEN per symbol → empty dict on error.
    """
    if not _ET_AVAILABLE:
        return {}

    # Universe RSR breadth for current day
    univ_rsr_vals = [
        float(rsr_mat[i, j])
        for j in range(len(active_syms))
        if rsr_mat[i, j] > 0
    ]

    results: dict = {}
    for rsr_val, sym in buy_cands:
        if sym not in universe_raw or sym not in sym_to_i:
            continue
        try:
            df_full  = universe_raw[sym]["df"]
            df_slice = df_full.loc[:signal_date]   # no lookahead
            if len(df_slice) < 20:
                continue

            si      = sym_to_i[sym]
            rsr_mom = float(rsr_mat[i, si] - rsr_mat[max(0, i - 20), si])

            rsr_series = None
            if sym in rsr_df.columns:
                rsr_s = rsr_df[sym].loc[:signal_date].dropna()
                if len(rsr_s) >= 3:
                    rsr_series = [float(v) for v in rsr_s.values[-8:]]

            et_inp = _et_build(
                symbol              = sym,
                df                  = df_slice,
                rsr                 = rsr_val,
                rsr_momentum        = rsr_mom,
                sepa_score          = 0,    # backtest: SEPA略
                rsr_series          = rsr_series,
                trend_cluster_level = 0,    # backtest: cluster略
                universe_rsr_values = univ_rsr_vals,
            )
            results[sym] = _et_score(et_inp)
        except Exception:
            pass  # FAIL_OPEN
    return results


@dataclass
class Position:
    symbol:      str
    sector:      str
    qty:         int
    entry_price: float
    entry_idx:   int
    rsr_at_entry: float = 0.0


def _take(df: pd.DataFrame | pd.Series, dates, active_syms=None,
          dtype=np.float32, fill_value=np.nan):
    if isinstance(df, pd.Series):
        s = df.reindex(pd.DatetimeIndex(dates), method="ffill").fillna(fill_value)
        return s.to_numpy(dtype=dtype, copy=False)
    if active_syms is None:
        active_syms = list(df.columns)
    cols = [c for c in active_syms if c in df.columns]
    sub = df.reindex(index=pd.DatetimeIndex(dates), columns=active_syms, method="ffill")
    sub = sub.fillna(fill_value)
    return sub.to_numpy(dtype=dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────
#  DATA LOADING (same as composite_alpha_bt.py)
# ─────────────────────────────────────────────────────────────────────

def load_data(cfg):
    print("[1/4] データロード中 (RSR42 + TOPIX, 2018-2025)...")
    rsr_csv = RSR_UNIVERSE_CSV
    if rsr_csv.exists():
        rsr_syms_df = pd.read_csv(rsr_csv, dtype=str)
        rsr_syms = {row["symbol"]: row.get("sector", "不明")
                    for _, row in rsr_syms_df.iterrows()}
    else:
        raise FileNotFoundError(f"RSR universe CSV not found: {rsr_csv}")

    all_syms = dict(rsr_syms)
    all_syms.setdefault("1306.T", "ETF")   # TOPIX ETF for regime detection
    universe_raw = download_universe(
        all_syms,
        start="2010-01-01", end=OOS_END,
        verbose=False,
    )

    # TOPIX (1306.T) — extract before removing from universe
    topix_close: pd.Series | None = None
    if "1306.T" in universe_raw:
        topix_close = universe_raw["1306.T"]["df"]["Close"]
        del universe_raw["1306.T"]

    print("[2/4] RSR計算中...")
    raw_close = pd.DataFrame({
        s: d["df"]["Close"]
        for s, d in universe_raw.items()
        if s in rsr_syms
    }).sort_index()
    rsr_df = calc_universe_rsr(raw_close)

    print("[3/4] Composite Alpha計算中... (skipped for speed)")
    # alpha_df not needed (using RSR-only ranking in baseline)
    alpha_df = pd.DataFrame(index=raw_close.index, columns=raw_close.columns, dtype=float).fillna(0.0)

    print("[4/4] 動的ユニバース計算中 (dyn_rsr42_bear_rs0)...")
    bear_exclude = [
        "機械", "鉄鋼", "銀行業", "保険業", "輸送用機器", "海運業", "化学"
    ]
    sym_active_df = build_dyn_rsr42_active(
        universe_raw=universe_raw,
        topix_close=topix_close,
        rsr_df=rsr_df,
        all_syms=list(rsr_syms.keys()),
        start="2010-01-01",
        end=OOS_END,
        bear_exclude_sectors=bear_exclude,
        sym_sector_map=rsr_syms,
    )

    # TOPIX regime arrays
    regime_df = pd.DataFrame(index=raw_close.index)
    if topix_close is not None:
        ma200 = topix_close.rolling(200, min_periods=100).mean()
        ma50  = topix_close.rolling(50,  min_periods=25).mean()
        regime_df["risk_off"] = topix_close < ma200
        regime_df["crash"]    = ma50 < ma200

    print(f"  RSR42: {len(rsr_syms)} 銘柄 / TOPIX: {'あり' if topix_close is not None else 'なし'}")
    print(f"  sym_active_df: {sym_active_df.shape if sym_active_df is not None else 'None'}")
    return universe_raw, rsr_df, alpha_df, sym_active_df, regime_df, rsr_syms, topix_close, cfg


# ─────────────────────────────────────────────────────────────────────
#  METRICS HELPER
# ─────────────────────────────────────────────────────────────────────

def calc_metrics(equity_curve: list[float], trades: list[dict],
                 exposure_list: list[float], capital: float,
                 dates: list) -> dict:
    n = len(equity_curve)
    if n == 0:
        return {}
    eq = pd.Series(equity_curve, index=pd.DatetimeIndex(dates[:n]))
    years = n / 252
    cagr = (eq.iloc[-1] / capital) ** (1 / max(years, 0.01)) - 1

    dr = eq.pct_change().dropna()
    sharpe  = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0
    neg_dr  = dr[dr < 0]
    sortino = float(dr.mean() / neg_dr.std() * np.sqrt(252)) if len(neg_dr) > 0 and neg_dr.std() > 0 else 0.0

    roll_max = eq.expanding().max()
    dd_ser   = (eq - roll_max) / roll_max
    max_dd   = float(dd_ser.min())
    calmar   = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    sells = [t for t in trades if t["side"] == "SELL"]
    wins  = [t for t in sells if t.get("pnl", 0) > 0]
    loss  = [t for t in sells if t.get("pnl", 0) <= 0]
    win_rate = len(wins) / max(1, len(sells))

    avg_win_pct  = float(np.mean([(t["exit"] / t["entry"] - 1) * 100 for t in wins])) if wins else 0.0
    avg_lose_pct = float(np.mean([(t["exit"] / t["entry"] - 1) * 100 for t in loss])) if loss else 0.0
    r_mult = abs(avg_win_pct / avg_lose_pct) if avg_lose_pct != 0 else 0.0

    gross_profit = sum(t["pnl"] for t in wins) if wins else 0.0
    gross_loss   = abs(sum(t["pnl"] for t in loss)) if loss else 0.0
    pf = gross_profit / max(1.0, gross_loss)

    hold_days = [t.get("exit_idx", 0) - t.get("entry_idx", 0) for t in sells if "exit_idx" in t]
    avg_hold  = float(np.mean(hold_days)) if hold_days else 0.0

    annual: dict[str, float] = {}
    for yr, grp in eq.groupby(eq.index.year):
        annual[str(yr)] = round(float(grp.iloc[-1] / grp.iloc[0] - 1) * 100, 2)

    # turnover (2-way, normalized)
    buys = [t for t in trades if t["side"] == "BUY"]
    total_buy_notional = sum(t["entry"] * t["qty"] for t in buys) if buys else 0.0
    avg_equity = eq.mean()
    turnover = total_buy_notional / max(1.0, avg_equity) / max(years, 0.01)

    avg_exp = float(np.mean(exposure_list)) * 100
    idle_pct = 100.0 - avg_exp

    return {
        "cagr":         round(cagr * 100, 2),
        "sharpe":       round(sharpe, 3),
        "sortino":      round(sortino, 3),
        "max_dd":       round(max_dd * 100, 2),
        "calmar":       round(calmar, 3),
        "profit_factor":round(pf, 3),
        "win_rate":     round(win_rate * 100, 1),
        "avg_win_pct":  round(avg_win_pct, 2),
        "avg_lose_pct": round(avg_lose_pct, 2),
        "r_multiple":   round(r_mult, 2),
        "n_trades":     len(sells),
        "n_trades_yr":  round(len(sells) / max(years, 0.01), 1),
        "avg_hold_days":round(avg_hold, 1),
        "avg_exposure": round(avg_exp, 1),
        "idle_cash":    round(idle_pct, 1),
        "turnover_yr":  round(turnover, 2),
        "annual_returns": annual,
        "eq_final": round(eq.iloc[-1], 0),
    }


# ─────────────────────────────────────────────────────────────────────
#  CORE BACKTEST ENGINE
# ─────────────────────────────────────────────────────────────────────

def run_period(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    start: str, end: str,
    pattern: str,                          # "A" | "B" | "C"
    et_enabled: bool = False,              # Entry Timing scoring ON/OFF
    et_block_low: bool = False,            # Block LOW confidence BUY candidates
    et_boost_weight: float = 0.0,          # ET score boost to composite ranking
    et_min_score: float = 0.0,             # Block candidates with ET score < threshold
    return_trades: bool = False,                   # Include raw trades + dates in return dict
    turtle_exit_override: "int | None" = None,    # Override cfg.fujiko.turtle_exit
    rsr_exit_override: "float | None" = None,     # Override rsr_exit_thr (exit-only, not entry)
    rsr_exit_bear: "float | None" = None,         # Bear-regime rsr_exit_thr (TOPIX<=MA200); None=use rsr_exit_thr
    max_new_per_day:      "int | None" = None,    # None=無制限; 1=live現行; 2=研究候補
    max_names_per_sector: "int | None" = None,    # None=weight-onlyのみ; 1=live; 2=研究候補
    cash_buffer:          float = 0.0,            # 0=なし; 0.10=live; 0.05=研究候補
    hold_policy: "str | None" = None,            # "P4"=含み益非対称; None=完全ベースライン
    return_missed_details: bool = False,          # True=blocked BUY候補を _missed_details に収集
    return_cash_skipped: bool = False,            # True=資金不足スキップ(slot有・cash不足)を収集
    return_equity: bool = False,                  # True=equity_curve + exposure_list を返す
    max_single_weight_override: "float | None" = None,  # None=cfg値(0.25); 研究用のみ
    verbose: bool = True,
) -> dict:
    """
    Single backtest period for one allocation pattern.
    Signal generation is IDENTICAL across A/B/C.
    Only position sizing and rotation logic differs.

    Entry Timing (ET) flags — default OFF preserves exact baseline behavior:
      et_enabled=False       → ET not computed, zero effect on ranking/selection
      et_block_low=False     → LOW confidence candidates not blocked
      et_boost_weight=0.0    → No composite score adjustment
    """
    capital = float(cfg.portfolio.capital)
    max_pos = int(cfg.portfolio.max_positions)      # =3 after fix
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold = int(cfg.risk.min_hold_days)
    max_hold = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = (
        float(rsr_exit_override) if rsr_exit_override is not None
        else float(cfg.fujiko.rsr_exit)   # rsr_exit=70; entry min_rsr=75 とは分離
    )
    shock_market_thr = -0.05
    shock_sym_thr    = -0.08

    # risk_controls
    rc = getattr(cfg, "risk_controls", None)
    MAX_SECTOR_W = float(rc.sector_cap)  if rc else 0.25
    MAX_SYMBOL_W = float(rc.symbol_cap)  if rc else 0.40
    MAX_SINGLE_W = (float(max_single_weight_override)
                    if max_single_weight_override is not None
                    else float(cfg.portfolio.max_single_weight))  # default 0.25

    gross_enabled = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gross_normal  = float(getattr(rc, "gross_cap_normal",       1.0)) if rc else 1.0
    gross_dd5     = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    gross_dd8     = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    trade_syms = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())

    # ── build strategy objects ────────────────────────────────────────
    strats: dict = {}
    for sym, sector in trade_syms.items():
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        rule = SECTOR_STRATEGY.get(sector, "fujiko")
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            eff_turtle_exit = (
                turtle_exit_override if turtle_exit_override is not None
                else cfg.fujiko.turtle_exit
            )
            strats[sym] = FujikoStrategy(
                min_rsr=cfg.fujiko.min_rsr,
                turtle_exit=eff_turtle_exit,
                rsr_series=rsr_s,
                min_sepa=cfg.fujiko.min_sepa,
                mom_period=cfg.fujiko.mom_period,
                turtle_entry=cfg.fujiko.turtle_entry,
                use_turtle_entry=cfg.fujiko.use_turtle_entry,
            )

    # ── common date range ─────────────────────────────────────────────
    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    ts = pd.Timestamp(start); te = pd.Timestamp(end)
    common_dates = common_dates[(common_dates >= ts) & (common_dates <= te)]
    if len(common_dates) == 0:
        return {}
    n_dates  = len(common_dates)
    n_syms   = len(active_syms)
    sym_to_i = {s: i for i, s in enumerate(active_syms)}

    # ── price matrices ─────────────────────────────────────────────────
    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    sig_mat   = np.zeros((n_dates, n_syms), dtype=np.int8)
    sig_ready = np.zeros(n_syms, dtype=bool)

    for si, sym in enumerate(active_syms):
        df_src = universe_raw[sym]["df"]
        row_idx = df_src.index.get_indexer(common_dates)
        if np.any(row_idx < 0):
            continue
        open_mat[:, si]  = df_src["Open"].to_numpy(dtype=np.float32)[row_idx]
        close_mat[:, si] = df_src["Close"].to_numpy(dtype=np.float32)[row_idx]
        strat = strats[sym]
        if hasattr(strat, "precompute_signals"):
            required = 252 + getattr(strat, "mom_period", 21) + 2
            if len(df_src) >= required:
                sig_series = strat.precompute_signals(df_src)
                sig_mat[:, si] = sig_series.to_numpy(dtype=np.int8)[row_idx]
                sig_ready[si] = True

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0,
    )
    sym_active_mat = None
    if sym_active_df is not None:
        sym_active_mat = _take(sym_active_df, common_dates, active_syms,
                               dtype=np.float32, fill_value=1.0)

    # ── ATR20 matrix (P4 のみ / High-Low-Close から 20日 SMA-TR) ─────────────
    atr_mat: "np.ndarray | None" = None
    if hold_policy == "P4":
        _ATR_P = 20
        atr_mat = np.zeros((n_dates, n_syms), dtype=np.float32)
        for _si, _sym in enumerate(active_syms):
            _df_s = universe_raw[_sym]["df"]
            _ri   = _df_s.index.get_indexer(common_dates)
            if np.any(_ri < 0):
                continue
            _h  = _df_s["High"].to_numpy(dtype=np.float64)[_ri]
            _l  = _df_s["Low"].to_numpy(dtype=np.float64)[_ri]
            _c  = close_mat[:, _si].astype(np.float64)
            _cp = np.empty_like(_c); _cp[0] = _c[0]; _cp[1:] = _c[:-1]
            _tr = np.maximum(_h - _l, np.maximum(np.abs(_h - _cp), np.abs(_l - _cp)))
            atr_mat[:, _si] = (
                pd.Series(_tr).rolling(_ATR_P, min_periods=1).mean().to_numpy(np.float32)
            )

    # TOPIX arrays
    mkt_ret_arr = np.zeros(n_dates, dtype=np.float32)
    topix_ret20 = topix_ret60 = None
    bear_arr    = None
    if topix_close is not None:
        tr = topix_close.pct_change()
        mkt_ret_arr = _take(tr, common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret20 = _take(topix_close.pct_change(20), common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret60 = _take(topix_close.pct_change(60), common_dates, dtype=np.float32, fill_value=0.0)
        ma200  = topix_close.rolling(200, min_periods=100).mean()
        bear_s = (topix_close < ma200).reindex(pd.DatetimeIndex(common_dates), method="ffill").fillna(False)
        bear_arr = bear_s.values.astype(bool)

    # ── state ─────────────────────────────────────────────────────────
    cash = float(capital)
    positions: dict[str, Position] = {}
    equity_curve: list[float] = []
    exposure_list: list[float] = []
    trades: list[dict] = []
    peak_equity = float(capital)
    cb_active = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    exit_counts: dict[str, int] = {}

    # missed opportunity tracking
    missed_count = 0   # BUY signal blocked by max_positions (no rotation)
    rotation_count = 0 # forced rotations (B/C patterns)
    blocked_gross = 0
    et_blocked_count = 0  # BUY signals filtered out by ET score threshold
    missed_details: list = []  # return_missed_details=True 時に blocked BUY を収集
    cash_skipped:  list = []  # return_cash_skipped=True 時に資金不足スキップを収集

    # ─────────────────────────────────────────────────────────────────
    for i, date in enumerate(common_dates):
        invested = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
                       for s, pos in positions.items())
        cur_equity = cash + invested
        equity_curve.append(cur_equity)
        exposure_list.append(invested / max(1.0, cur_equity))

        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        # CB logic
        if not cb_active:
            if dd <= -max_dd_limit:
                cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0
        cb_scale = CB_SCALE if cb_active else 1.0

        # Gross cap
        gross_cap = gross_normal
        if gross_enabled and topix_ret20 is not None:
            r20 = float(topix_ret20[i]); r60 = float(topix_ret60[i])
            if r20 < -0.05:
                gross_cap = gross_dd5
            elif r60 < -0.08:
                gross_cap = gross_dd8

        # Bear flag
        is_bear = bool(bear_arr[i]) if bear_arr is not None else False
        sec_cap_eff = 0.18 if is_bear else MAX_SECTOR_W
        # Regime-aware RSR exit threshold
        eff_rsr_exit_thr = (
            float(rsr_exit_bear) if (rsr_exit_bear is not None and is_bear)
            else rsr_exit_thr
        )

        sell_sigs: list[tuple] = []
        buy_cands: list[tuple] = []  # (rsr, sym)

        # Market shock
        mkt_shock = float(mkt_ret_arr[i]) <= shock_market_thr

        for sym in active_syms:
            si = sym_to_i[sym]
            is_holding = sym in positions
            hold_idx = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val   = float(rsr_mat[i, si])
            close_t   = float(close_mat[i, si])

            # composite shock exit
            if mkt_shock:
                if not is_holding:
                    continue
                if i > 0:
                    prev_c = float(close_mat[i - 1, si])
                    if prev_c > 0 and (close_t / prev_c - 1.0) <= shock_sym_thr:
                        sell_sigs.append((sym, "MARKET_SHOCK_EXIT"))
                        continue
            if mkt_shock and not is_holding:
                continue

            # time stop
            if is_holding and max_hold is not None and hold_idx > max_hold:
                sell_sigs.append((sym, "TIME_STOP")); continue

            # RSR exit
            if is_holding and rsr_val < eff_rsr_exit_thr and hold_idx >= min_hold:
                if hold_policy == "P4" and atr_mat is not None:
                    _pos = positions[sym]
                    _atr = float(atr_mat[i, si])
                    # 含み益(close > entry + 0.5×ATR)なら閾値を65に緩和
                    _p4  = 65.0 if (_atr > 0 and close_t > _pos.entry_price + 0.5 * _atr) \
                               else eff_rsr_exit_thr
                    if rsr_val >= _p4:
                        pass  # rsr in [65,70): 含み益保有継続 → STRATEGY_EXIT へ
                    else:
                        sell_sigs.append((sym, "RSR_EXIT")); continue
                else:
                    sell_sigs.append((sym, "RSR_EXIT")); continue

            # Strategy exit (turtle / momentum)
            sig = int(sig_mat[i, si]) if sig_ready[si] else 0
            if sig == -1 and is_holding and hold_idx >= min_hold:
                sell_sigs.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not is_holding:
                if i < reentry_ban.get(sym, -1):
                    continue
                # dynamic universe filter
                if sym_active_mat is not None:
                    if float(sym_active_mat[i, si]) < 0.5:
                        continue
                buy_cands.append((rsr_val, sym))

        if i + 1 >= n_dates:
            break
        next_i = i + 1

        # ── SELL EXECUTION ────────────────────────────────────────────
        for sym, reason in sell_sigs:
            if sym not in positions:
                continue
            pos = positions[sym]
            sell_px = float(open_mat[next_i, sym_to_i[sym]])
            proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
            pnl = (sell_px - pos.entry_price) * pos.qty
            cash += proceeds
            trades.append({
                "side": "SELL", "symbol": sym,
                "entry": pos.entry_price, "exit": sell_px,
                "qty": pos.qty, "pnl": pnl,
                "entry_idx": pos.entry_idx, "exit_idx": i,
                "reason": reason,
                "is_bear": is_bear,
            })
            exit_counts[reason] = exit_counts.get(reason, 0) + 1
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if cb_active or not buy_cands:
            continue

        # Sort buy candidates by RSR descending
        buy_cands.sort(key=lambda x: -x[0])

        # ── Entry Timing pre-filter (default OFF → zero effect) ───────
        if et_enabled and buy_cands:
            _et_res = _compute_et_bt(
                buy_cands, date, universe_raw, rsr_df,
                rsr_mat, i, sym_to_i, active_syms,
            )
            # Score threshold filter (et_min_score > 0)
            if et_min_score > 0 and _et_res:
                n_before = len(buy_cands)
                buy_cands = [
                    (rv, s) for rv, s in buy_cands
                    if _et_res.get(s) is None or _et_res[s].score >= et_min_score
                ]
                et_blocked_count += n_before - len(buy_cands)
            # Block LOW confidence entries (block_low_confidence gate)
            if et_block_low and _et_res:
                buy_cands = [
                    (rv, s) for rv, s in buy_cands
                    if _et_res.get(s) is None or _et_res[s].confidence != _ET_LOW
                ]
            # Optional: ET score boost to composite ranking
            if et_boost_weight > 0 and _et_res:
                buy_cands.sort(key=lambda x: -(
                    x[0] + (_et_res[x[1]].score - 50.0) * et_boost_weight
                    if x[1] in _et_res else x[0]
                ))

        # ─────────────────────────────────────────────────────────────
        # PATTERN-SPECIFIC BUY LOGIC
        # ─────────────────────────────────────────────────────────────

        if pattern == "A":
            # ── Pattern A: Baseline ───────────────────────────────────
            # alloc = cash / effective_slots, max_single_weight cap
            n_remaining = len(buy_cands)
            _new_today = 0  # max_new_per_day counter
            for _bi, (rsr_val, sym) in enumerate(buy_cands):
                open_slots = max_pos - len(positions)
                if open_slots <= 0:
                    if return_missed_details:
                        for _bj, (_rv, _s) in enumerate(buy_cands[_bi:]):
                            _si_b = sym_to_i.get(_s, -1)
                            if _si_b >= 0:
                                _bp = float(open_mat[next_i, _si_b])
                                missed_details.append({
                                    "date": str(date.date()),
                                    "sym":  _s,
                                    "rsr":  round(float(_rv), 2),
                                    "buy_px": round(_bp, 2),
                                    "lot_cost": round(_bp * LOT, 0),
                                    "n_held": len(positions),
                                    "rank_blocked": _bj,
                                })
                    missed_count += len(buy_cands) - _bi
                    break
                if max_new_per_day is not None and _new_today >= max_new_per_day:
                    break  # daily new-entry limit reached
                buy_px = float(open_mat[next_i, sym_to_i[sym]])
                if buy_px <= 0:
                    continue
                # sector cap check (weight-based)
                if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                    continue
                # sector names check (count-based; None=skip)
                if not _sector_name_ok(sym, positions, trade_syms, max_names_per_sector):
                    continue
                # gross exposure check
                if gross_enabled:
                    cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                    for p in positions.values()) / max(1.0, capital)
                    order_w = buy_px * LOT / max(1.0, capital)
                    if cur_gross + order_w > gross_cap:
                        blocked_gross += 1; continue
                # sizing: equal allocation — capital / max_pos (scale invariant)
                # each position targets 1/max_pos of initial capital regardless
                # of how many slots are open or how many candidates remain.
                alloc = (capital / max_pos) * cb_scale
                qty = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    if return_cash_skipped:
                        cash_skipped.append({
                            "date":      str(date.date()),
                            "sym":       sym,
                            "rank":      _bi,
                            "rsr":       round(float(rsr_val), 2),
                            "buy_px":    round(buy_px, 2),
                            "lot_cost":  round(buy_px * LOT, 0),
                            "alloc":     round(alloc, 0),
                            "cash":      round(cash, 0),
                            "open_slots": open_slots,
                        })
                    continue
                # cash gate: lot rounding → cash check → final size
                if qty * buy_px * (1.0 + COST_ONE_WAY) > cash * (1.0 - cash_buffer):
                    continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)
                _new_today += 1

        elif pattern == "B":
            # ── Pattern B: Equal Weight Full Exposure ─────────────────
            # Target: always fill max_positions with equal weight
            # When slots open: buy and size to equity / max_positions
            # When full: rotate out lowest RSR if new > worst held
            cur_equity_b = cash + sum(p.qty * float(close_mat[i, sym_to_i[s]])
                                      for s, p in positions.items())
            target_alloc = cur_equity_b / max_pos  # equal weight target

            # First: fill open slots
            for rsr_val, sym in buy_cands:
                open_slots = max_pos - len(positions)
                if open_slots <= 0:
                    break
                buy_px = float(open_mat[next_i, sym_to_i[sym]])
                if buy_px <= 0:
                    continue
                if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                    continue
                if gross_enabled:
                    cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                    for p in positions.values()) / max(1.0, cur_equity_b)
                    order_w = buy_px * LOT / max(1.0, cur_equity_b)
                    if cur_gross + order_w > gross_cap:
                        blocked_gross += 1; continue
                # equal weight sizing
                alloc = min(target_alloc, cash) * cb_scale
                qty = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)

            # Rotation: if still have buy candidates and slots still full
            if len(positions) >= max_pos:
                for rsr_val, sym in buy_cands:
                    if sym in positions:
                        continue
                    # Find worst held position by RSR
                    worst_sym = min(positions.keys(),
                                    key=lambda s: float(rsr_mat[i, sym_to_i[s]]))
                    worst_rsr = float(rsr_mat[i, sym_to_i[worst_sym]])
                    if rsr_val <= worst_rsr:
                        continue   # new candidate not better than worst held
                    if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                        continue
                    # Sell worst
                    w_pos = positions[worst_sym]
                    sell_px = float(open_mat[next_i, sym_to_i[worst_sym]])
                    proceeds = w_pos.qty * sell_px * (1 - COST_ONE_WAY)
                    pnl = (sell_px - w_pos.entry_price) * w_pos.qty
                    cash += proceeds
                    trades.append({
                        "side": "SELL", "symbol": worst_sym,
                        "entry": w_pos.entry_price, "exit": sell_px,
                        "qty": w_pos.qty, "pnl": pnl,
                        "entry_idx": w_pos.entry_idx, "exit_idx": i,
                        "reason": "ROTATION_B",
                    })
                    del positions[worst_sym]
                    rotation_count += 1
                    # Buy new at equal weight
                    buy_px = float(open_mat[next_i, sym_to_i[sym]])
                    if buy_px <= 0:
                        continue
                    alloc = min(target_alloc, cash) * cb_scale
                    qty = int(alloc / buy_px / LOT) * LOT
                    if qty <= 0:
                        continue
                    _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                    cash -= qty * buy_px * (1 + COST_ONE_WAY)
                    break   # one rotation per day maximum

        elif pattern == "C":
            # ── Pattern C: Conviction Rotation ────────────────────────
            # First entry: CONVICTION_FIRST_RATIO × deployable capital
            # Target: CONVICTION_TARGET_EXP exposure
            # Active rotation: always swap lowest for better when detected
            cur_equity_c = cash + sum(p.qty * float(close_mat[i, sym_to_i[s]])
                                      for s, p in positions.items())
            deployable = cur_equity_c * (1 - 0.10)  # 10% cash buffer

            # Conviction sizing
            if len(positions) == 0:
                # First entry: 70% conviction
                alloc_per = deployable * CONVICTION_FIRST_RATIO * cb_scale
            else:
                # Subsequent: target 85% exposure equally
                current_inv = sum(p.qty * float(close_mat[i, sym_to_i[s]])
                                  for s, p in positions.items())
                target_inv = cur_equity_c * CONVICTION_TARGET_EXP
                remaining_budget = max(0.0, target_inv - current_inv)
                n_open = max_pos - len(positions)
                alloc_per = (remaining_budget / max(1, n_open)) * cb_scale

            # Fill open slots
            for rsr_val, sym in buy_cands:
                open_slots = max_pos - len(positions)
                if open_slots <= 0:
                    break
                buy_px = float(open_mat[next_i, sym_to_i[sym]])
                if buy_px <= 0:
                    continue
                if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                    continue
                if gross_enabled:
                    cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                    for p in positions.values()) / max(1.0, cur_equity_c)
                    order_w = buy_px * LOT / max(1.0, cur_equity_c)
                    if cur_gross + order_w > gross_cap:
                        blocked_gross += 1; continue
                alloc = min(alloc_per, cash * 0.99)
                qty = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)

            # Rotation: swap lowest conviction held with better candidate
            if len(positions) >= max_pos:
                for rsr_val, sym in buy_cands:
                    if sym in positions:
                        continue
                    worst_sym = min(positions.keys(),
                                    key=lambda s: float(rsr_mat[i, sym_to_i[s]]))
                    worst_rsr = float(rsr_mat[i, sym_to_i[worst_sym]])
                    # More aggressive rotation threshold for C: rotate even at +5 RSR diff
                    if rsr_val <= worst_rsr + 5.0:
                        continue
                    if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                        continue
                    # Sell worst
                    w_pos = positions[worst_sym]
                    sell_px = float(open_mat[next_i, sym_to_i[worst_sym]])
                    proceeds = w_pos.qty * sell_px * (1 - COST_ONE_WAY)
                    pnl = (sell_px - w_pos.entry_price) * w_pos.qty
                    cash += proceeds
                    trades.append({
                        "side": "SELL", "symbol": worst_sym,
                        "entry": w_pos.entry_price, "exit": sell_px,
                        "qty": w_pos.qty, "pnl": pnl,
                        "entry_idx": w_pos.entry_idx, "exit_idx": i,
                        "reason": "ROTATION_C",
                    })
                    del positions[worst_sym]
                    rotation_count += 1
                    # Buy new at conviction sizing
                    buy_px = float(open_mat[next_i, sym_to_i[sym]])
                    if buy_px <= 0:
                        continue
                    # Re-compute conviction alloc after sale
                    alloc_rot = proceeds * 0.95 * cb_scale  # reuse 95% of proceeds
                    qty = int(alloc_rot / buy_px / LOT) * LOT
                    if qty <= 0:
                        continue
                    _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                    cash -= qty * buy_px * (1 + COST_ONE_WAY)
                    break

        elif pattern == "D":
            # ── Pattern D: Dynamic Capital Concentration (DCC) ────────
            # target total exposure = 90% equally divided by n_positions
            # 1 pos → 90%   2 pos → 45%+45%   3 pos → 30%+30%+30%
            # No rotation — same exit-only logic as Pattern A
            cur_equity_d = cash + sum(p.qty * float(close_mat[i, sym_to_i[s]])
                                      for s, p in positions.items())
            for rsr_val, sym in buy_cands:
                open_slots = max_pos - len(positions)
                if open_slots <= 0:
                    missed_count += len(buy_cands) - buy_cands.index((rsr_val, sym))
                    break
                buy_px = float(open_mat[next_i, sym_to_i[sym]])
                if buy_px <= 0:
                    continue
                if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                    continue
                if gross_enabled:
                    cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                    for p in positions.values()) / max(1.0, cur_equity_d)
                    order_w = buy_px * LOT / max(1.0, cur_equity_d)
                    if cur_gross + order_w > gross_cap:
                        blocked_gross += 1; continue
                # DCC sizing: 90% × equity / n_positions_after_this_entry
                n_after = len(positions) + 1
                target_per = cur_equity_d * DCC_TARGET_EXPOSURE / n_after * cb_scale
                alloc = min(target_per, cash * 0.99)
                qty = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue
                _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
                cash -= qty * buy_px * (1 + COST_ONE_WAY)

    metrics = calc_metrics(equity_curve, trades, exposure_list, capital,
                           list(common_dates))
    metrics["exit_reasons"]    = exit_counts
    metrics["rotation_count"]  = rotation_count
    metrics["missed_signals"]  = missed_count
    metrics["blocked_gross"]   = blocked_gross
    metrics["et_blocked_count"] = et_blocked_count
    if return_trades:
        metrics["_trades_raw"]    = trades
        metrics["_common_dates"]  = [str(d.date()) for d in common_dates]
    if return_equity:
        metrics["_equity_curve"]  = equity_curve
        metrics["_equity_dates"]  = [str(d.date()) for d in common_dates]
        metrics["_exposure_list"] = exposure_list
    if return_missed_details:
        metrics["_missed_details"] = missed_details
    if return_cash_skipped:
        metrics["_cash_skipped"] = cash_skipped
    return metrics


def _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms,
               capital, sec_cap_eff) -> bool:
    sector = trade_syms.get(sym, "不明")
    sec_val = sum(
        p.qty * float(close_mat[i, sym_to_i[p.symbol]])
        for p in positions.values()
        if trade_syms.get(p.symbol, "不明") == sector and p.symbol in sym_to_i
    )
    return sec_val / max(1.0, capital) < sec_cap_eff


def _sector_name_ok(sym, positions, trade_syms, max_names) -> bool:
    """count-based sector constraint. max_names=None → always True."""
    if max_names is None:
        return True
    sector = trade_syms.get(sym, "不明")
    n_same = sum(1 for p in positions.values()
                 if trade_syms.get(p.symbol, "不明") == sector)
    return n_same < max_names


def _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val):
    trades.append({
        "side": "BUY", "symbol": sym,
        "entry": buy_px, "exit": None,
        "qty": qty, "pnl": None,
        "entry_idx": next_i, "reason": f"RSR={rsr_val:.1f}",
    })
    positions[sym] = Position(
        symbol=sym, sector=trade_syms.get(sym, "不明"),
        qty=qty, entry_price=buy_px, entry_idx=next_i, rsr_at_entry=rsr_val,
    )


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main():
    cfg = load_strategy_config()
    assert cfg.portfolio.max_positions == 3, \
        f"max_positions={cfg.portfolio.max_positions} — must be 3 (PARAMS_LOCKED)"

    print("=" * 70)
    print("  Capital Deployment A/B/C Backtest")
    print(f"  IS: {IS_START}〜{IS_END}  /  OOS: {OOS_START}〜{OOS_END}")
    print(f"  max_pos={cfg.portfolio.max_positions}  capital=¥{cfg.portfolio.capital:,.0f}")
    print("=" * 70 + "\n")

    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    patterns = ["A", "B", "C", "D"]
    pattern_labels = {
        "A": "A Baseline   (現行)",
        "B": "B FullExp    (均等フルポジ)",
        "C": "C Conviction (信念ローテーション)",
        "D": "D DCC        (動的集中配分)",
    }

    results: dict[str, dict] = {}

    for pattern in patterns:
        print(f"\n{'='*60}")
        print(f"  パターン {pattern_labels[pattern]}")
        print(f"{'='*60}")

        print(f"  [IS 2018-2024] 実行中...")
        t0 = time.time()
        is_res = run_period(
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
            start=IS_START, end=IS_END,
            pattern=pattern, verbose=False,
        )
        is_time = time.time() - t0

        print(f"  [OOS 2025]     実行中...")
        t0 = time.time()
        oos_res = run_period(
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
            start=OOS_START, end=OOS_END,
            pattern=pattern, verbose=False,
        )
        oos_time = time.time() - t0

        results[pattern] = {"IS": is_res, "OOS": oos_res}

        # ── print summary ─────────────────────────────────────────────
        def _p(label, key, fmt="{:.1f}"):
            iv = is_res.get(key, "-")
            ov = oos_res.get(key, "-")
            if isinstance(iv, (int, float)):
                iv_s = fmt.format(iv)
            else:
                iv_s = str(iv)
            if isinstance(ov, (int, float)):
                ov_s = fmt.format(ov)
            else:
                ov_s = str(ov)
            print(f"    {label:<22} IS={iv_s:>8}   OOS={ov_s:>8}")

        print(f"\n  ── 収益性 ─────────────────────────────────────────")
        _p("CAGR (%)",        "cagr")
        _p("Profit Factor",   "profit_factor", "{:.3f}")
        _p("Win Rate (%)",    "win_rate")
        _p("Avg R multiple",  "r_multiple", "{:.2f}")
        print(f"  ── リスク ─────────────────────────────────────────")
        _p("Sharpe",          "sharpe", "{:.3f}")
        _p("Sortino",         "sortino", "{:.3f}")
        _p("MaxDD (%)",       "max_dd")
        _p("Calmar",          "calmar", "{:.3f}")
        print(f"  ── 資本効率 ───────────────────────────────────────")
        _p("Avg Exposure (%)", "avg_exposure")
        _p("Idle Cash (%)",    "idle_cash")
        _p("Turnover/yr",      "turnover_yr", "{:.2f}")
        print(f"  ── 売買構造 ───────────────────────────────────────")
        _p("Trades/yr",        "n_trades_yr")
        _p("Avg Hold Days",    "avg_hold_days")
        rot  = is_res.get("rotation_count", 0)
        miss = is_res.get("missed_signals", 0)
        print(f"    {'Rotation count':<22} IS={rot:>8}   OOS={oos_res.get('rotation_count',0):>8}")
        print(f"    {'Missed (no slot)':<22} IS={miss:>8}   OOS={oos_res.get('missed_signals',0):>8}")
        print(f"\n  IS 年次リターン:")
        for yr in sorted(is_res.get("annual_returns", {}).keys()):
            print(f"    {yr}: {is_res['annual_returns'][yr]:+.1f}%")
        print(f"  ({is_time:.1f}s IS / {oos_time:.1f}s OOS)")

    # ─────────────────────────────────────────────────────────────────
    # COMPARATIVE SUMMARY TABLE
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  ★ COMPARATIVE SUMMARY")
    print("=" * 80)

    hdr = f"  {'指標':<24} {'A 現行':>11} {'B フルポジ':>11} {'C 信念回転':>11} {'D DCC':>11}"
    print(hdr)
    print("  " + "-" * 70)

    def row(label, key, period, fmt="{:.1f}"):
        vals = []
        for p in patterns:
            v = results[p][period].get(key, "-")
            vals.append(fmt.format(v) if isinstance(v, (int, float)) else str(v))
        print(f"  {label:<24} {vals[0]:>11} {vals[1]:>11} {vals[2]:>11} {vals[3]:>11}")

    print(f"\n  IS 2018-2024")
    row("CAGR (%)",           "cagr",         "IS")
    row("Sharpe",             "sharpe",        "IS", "{:.3f}")
    row("Sortino",            "sortino",       "IS", "{:.3f}")
    row("MaxDD (%)",          "max_dd",        "IS")
    row("Calmar",             "calmar",        "IS", "{:.3f}")
    row("Profit Factor",      "profit_factor", "IS", "{:.3f}")
    row("Win Rate (%)",       "win_rate",      "IS")
    row("Avg Hold Days",      "avg_hold_days", "IS")
    row("Avg Exposure (%)",   "avg_exposure",  "IS")
    row("Idle Cash (%)",      "idle_cash",     "IS")
    row("Trades/yr",          "n_trades_yr",   "IS")
    row("Rotation count",     "rotation_count","IS","{:.0f}")
    row("Missed signals",     "missed_signals","IS","{:.0f}")

    print(f"\n  OOS 2025")
    row("CAGR (%)",           "cagr",         "OOS")
    row("Sharpe",             "sharpe",        "OOS", "{:.3f}")
    row("MaxDD (%)",          "max_dd",        "OOS")
    row("Avg Exposure (%)",   "avg_exposure",  "OOS")

    # ─────────────────────────────────────────────────────────────────
    # INCREMENTAL ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  ★ 差分分析")
    print("=" * 80)

    def delta(key, pa, pb, period, fmt="{:+.1f}"):
        va = results[pa][period].get(key, 0)
        vb = results[pb][period].get(key, 0)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = vb - va
            return fmt.format(d)
        return "n/a"

    print(f"\n  ① A → B (フルポジ化のみで CAGR はどう変わるか)")
    print(f"    IS  CAGR:     {delta('cagr',        'A','B','IS')}")
    print(f"    IS  Sharpe:   {delta('sharpe',      'A','B','IS', '{:+.3f}')}")
    print(f"    IS  MaxDD:    {delta('max_dd',      'A','B','IS')}")
    print(f"    IS  Exposure: {delta('avg_exposure','A','B','IS')}")
    print(f"    OOS CAGR:     {delta('cagr',        'A','B','OOS')}")
    print(f"    OOS Sharpe:   {delta('sharpe',      'A','B','OOS', '{:+.3f}')}")
    print(f"    OOS MaxDD:    {delta('max_dd',      'A','B','OOS')}")

    print(f"\n  ② B → C (均等 vs 信念加重 — 上乗せ効果)")
    print(f"    IS  CAGR:     {delta('cagr',        'B','C','IS')}")
    print(f"    IS  Sharpe:   {delta('sharpe',      'B','C','IS', '{:+.3f}')}")
    print(f"    IS  MaxDD:    {delta('max_dd',      'B','C','IS')}")
    print(f"    IS  Exposure: {delta('avg_exposure','B','C','IS')}")
    print(f"    OOS CAGR:     {delta('cagr',        'B','C','OOS')}")
    print(f"    OOS Sharpe:   {delta('sharpe',      'B','C','OOS', '{:+.3f}')}")

    print(f"\n  ③ A → C (現行 vs 最大稼働モデル)")
    print(f"    IS  CAGR:     {delta('cagr',        'A','C','IS')}")
    print(f"    IS  Sharpe:   {delta('sharpe',      'A','C','IS', '{:+.3f}')}")
    print(f"    IS  MaxDD:    {delta('max_dd',      'A','C','IS')}")
    print(f"    IS  Exposure: {delta('avg_exposure','A','C','IS')}")
    print(f"    OOS CAGR:     {delta('cagr',        'A','C','OOS')}")
    print(f"    OOS Sharpe:   {delta('sharpe',      'A','C','OOS', '{:+.3f}')}")

    print(f"\n  ④ A → D (現行 vs DCC 動的集中配分)")
    print(f"    IS  CAGR:     {delta('cagr',        'A','D','IS')}")
    print(f"    IS  Sharpe:   {delta('sharpe',      'A','D','IS', '{:+.3f}')}")
    print(f"    IS  MaxDD:    {delta('max_dd',      'A','D','IS')}")
    print(f"    IS  Exposure: {delta('avg_exposure','A','D','IS')}")
    print(f"    OOS CAGR:     {delta('cagr',        'A','D','OOS')}")
    print(f"    OOS Sharpe:   {delta('sharpe',      'A','D','OOS', '{:+.3f}')}")

    # ─────────────────────────────────────────────────────────────────
    # CONCLUSIONS
    # ─────────────────────────────────────────────────────────────────
    print("\n\n" + "=" * 80)
    print("  ★ 結論")
    print("=" * 80)

    # Find best by each criterion
    is_cagr   = {p: results[p]["IS"].get("cagr", -999)   for p in patterns}
    is_sharpe = {p: results[p]["IS"].get("sharpe", -999) for p in patterns}
    oos_cagr  = {p: results[p]["OOS"].get("cagr", -999)  for p in patterns}
    oos_sh    = {p: results[p]["OOS"].get("sharpe", -999) for p in patterns}

    best_cagr   = max(is_cagr,   key=is_cagr.get)
    best_sharpe = max(is_sharpe, key=is_sharpe.get)

    # "Balanced" = max Calmar or (sharpe * (1 + cagr/100))
    balance_score = {p: results[p]["IS"].get("calmar", 0) for p in patterns}
    best_balance  = max(balance_score, key=balance_score.get)

    cagr30_score = {p: abs(results[p]["IS"].get("cagr", 0) - 30) for p in patterns}
    closest_30   = min(cagr30_score, key=cagr30_score.get)

    print(f"\n  1. CAGR最大モデル         : Pattern {best_cagr}  "
          f"(IS {is_cagr[best_cagr]:+.1f}% / OOS {oos_cagr[best_cagr]:+.1f}%)")
    print(f"  2. Sharpe最大モデル        : Pattern {best_sharpe}  "
          f"(IS Sharpe {is_sharpe[best_sharpe]:.3f} / OOS {oos_sh[best_sharpe]:.3f})")
    print(f"  3. 実運用推奨モデル        : Pattern {best_balance}  "
          f"(IS Calmar {balance_score[best_balance]:.3f})")
    print(f"  4. CAGR 30%最近傍モデル    : Pattern {closest_30}  "
          f"(IS {is_cagr[closest_30]:+.1f}%)")

    print(f"\n  ⑤ Exposure増加 vs DD増加 トレードオフ (vs Pattern A)")
    for p in patterns:
        is_r = results[p]["IS"]
        exp_gain = is_r.get("avg_exposure", 0) - results["A"]["IS"].get("avg_exposure", 0)
        dd_cost  = abs(is_r.get("max_dd", 0))  - abs(results["A"]["IS"].get("max_dd", 0))
        cagr_g   = is_r.get("cagr", 0) - results["A"]["IS"].get("cagr", 0)
        print(f"    Pattern {p}: Exposure {exp_gain:+.1f}pp / MaxDD {dd_cost:+.1f}pp / CAGR {cagr_g:+.1f}pp")

    # ─────────────────────────────────────────────────────────────────
    # SAVE JSON
    # ─────────────────────────────────────────────────────────────────
    today = time.strftime("%Y-%m-%d")
    out_path = OUTPUT_DIR / f"capital_allocation_abc_{today}.json"
    OUTPUT_DIR.mkdir(exist_ok=True)

    save_data: dict = {}
    for p in patterns:
        save_data[f"pattern_{p}"] = {
            "IS":  {k: v for k, v in results[p]["IS"].items()  if k != "equity_curve"},
            "OOS": {k: v for k, v in results[p]["OOS"].items() if k != "equity_curve"},
        }
    out_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  結果保存: {out_path}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
