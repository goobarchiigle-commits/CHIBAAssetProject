"""
backtest/rsr_state_matrix.py

研究専用 / 実装変更禁止

RSR絶対水準 × 方向性状態 2D マトリクス分析

仮説:
H1: 高RSRほど良い
H2: 最適帯が存在
H3: 同じRSRでも上昇中と下降中で期待値が逆転

対象: 2018-2025 全 buy signal 候補 (selected / not_selected 両方)

特徴量:
  RSR_slope_5  = (RSR_t - RSR_t-5) / 5
  RSR_slope_20 = (RSR_t - RSR_t-20) / 20
  RSR_peak_20  = max(RSR[-20:t])
  RSR_drawdown = RSR_t - RSR_peak_20
  RSR_accel    = slope5 - slope20

状態:
  RISING      slope20 > FLAT_THR AND slope5 > 0
  FLAT        |slope20| <= FLAT_THR
  COOLING     slope20 > FLAT_THR AND slope5 < 0
  FALLING     slope20 < -FLAT_THR
  POST_EXTREME RSR>=95 AND drawdown<0

Output:
  reports/rsr_state_matrix.csv
  reports/entry_state_distribution.csv
  reports/rsr_state_matrix.md

Run:
    cd C:/ai-trading
    python src/backtest/rsr_state_matrix.py
"""

from __future__ import annotations

import sys, time, warnings, csv
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.config_loader import load_strategy_config
from src.backtest.fujiko_strategy import FujikoStrategy
from src.backtest.mean_reversion_strategy import MeanReversionStrategy
from src.backtest.capital_allocation_abc import (
    load_data, _sector_ok, _execute_buy, Position, _take,
    LOT, COST_ONE_WAY, REENTRY_COOL, CB_UNLOCK_DAYS, CB_SCALE,
    SECTOR_STRATEGY, MR_PARAMS,
)

REPORTS_DIR = Path("reports")
FULL_START  = "2018-01-01"
FULL_END    = "2025-12-31"

# ── 状態分類パラメータ ────────────────────────────────────────────────
FLAT_THRESHOLD  = 0.15   # RSR points/day (3pp over 20d → FLAT)
POST_EXT_RSR    = 95.0   # RSR >= 95 for POST_EXTREME

# ── RSR bin 定義 ─────────────────────────────────────────────────────
RSR_BINS   = ["<70", "70-74", "75-79", "80-84", "85-89", "90-94", "95-100"]
RSR_STATES = ["RISING", "FLAT", "COOLING", "FALLING", "POST_EXTREME"]

MIN_CELL_N = 5   # 最小サンプル数（未満は N/A 表示）


# ─────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────

def get_rsr_bin(rsr: float) -> str:
    if rsr < 70:   return "<70"
    if rsr < 75:   return "70-74"
    if rsr < 80:   return "75-79"
    if rsr < 85:   return "80-84"
    if rsr < 90:   return "85-89"
    if rsr < 95:   return "90-94"
    return "95-100"


def classify_state(rsr_t: float, slope5, slope20: float, drawdown: float) -> str:
    if np.isnan(slope20):
        return "UNKNOWN"
    # POST_EXTREME: 高値圏で既に下落開始
    if rsr_t >= POST_EXT_RSR and drawdown < 0:
        return "POST_EXTREME"
    # FLAT: 20d トレンドが微小
    if abs(slope20) <= FLAT_THRESHOLD:
        return "FLAT"
    if slope20 > FLAT_THRESHOLD:
        if slope5 is not None and not np.isnan(slope5):
            return "RISING" if slope5 > 0 else "COOLING"
        return "RISING"
    # slope20 < -FLAT_THRESHOLD
    return "FALLING"


def safe_mean(vals) -> float | None:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return round(float(np.mean(v)), 2) if v else None


def safe_pf(vals) -> float | None:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return None
    pos = sum(x for x in v if x > 0)
    neg = abs(sum(x for x in v if x <= 0))
    return round(pos / max(neg, 1e-6), 3) if neg > 0 else None


def safe_sharpe(vals) -> float | None:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if len(v) < MIN_CELL_N:
        return None
    m = np.mean(v); s = np.std(v)
    return round(float(m / s), 3) if s > 1e-6 else None


def hit_rate(vals) -> float | None:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    if not v:
        return None
    return round(sum(1 for x in v if x > 0) / len(v) * 100, 1)


def effect_size(vals_a, vals_b) -> float | None:
    a = [x for x in vals_a if x is not None and not np.isnan(x)]
    b = [x for x in vals_b if x is not None and not np.isnan(x)]
    if len(a) < 3 or len(b) < 3:
        return None
    pooled_std = np.sqrt((np.var(a) * (len(a)-1) + np.var(b) * (len(b)-1))
                         / (len(a) + len(b) - 2))
    return round((np.mean(a) - np.mean(b)) / max(pooled_std, 1e-6), 3) if pooled_std > 1e-6 else None


# ─────────────────────────────────────────────────────────────────────
#  SIGNAL COLLECTOR + PORTFOLIO SIMULATION
# ─────────────────────────────────────────────────────────────────────

def collect_signals(
    universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
    topix_close, rsr_syms, cfg,
    start: str, end: str,
) -> list[dict]:
    """
    全期間を通じて全 buy signal 候補を収集。
    portfolio simulation により selected / not_selected を判定。
    post-hoc で fwd20/60/120 を付与して返す。
    """
    capital      = float(cfg.portfolio.capital)
    max_pos      = int(cfg.portfolio.max_positions)
    max_dd_limit = float(cfg.portfolio.max_dd_limit)
    min_hold     = int(cfg.risk.min_hold_days)
    max_hold     = getattr(cfg.risk, "max_hold_days", None)
    rsr_exit_thr = float(cfg.fujiko.rsr_exit)
    MAX_SINGLE_W = float(cfg.portfolio.max_single_weight)

    shock_market_thr = -0.05
    shock_sym_thr    = -0.08

    rc            = getattr(cfg, "risk_controls", None)
    MAX_SECTOR_W  = float(rc.sector_cap) if rc else 0.25
    gross_enabled = bool(getattr(rc, "gross_exposure_enabled", True)) if rc else True
    gross_normal  = float(getattr(rc, "gross_cap_normal",       1.0)) if rc else 1.0
    gross_dd5     = float(getattr(rc, "gross_cap_drawdown_5pct", 0.6)) if rc else 0.6
    gross_dd8     = float(getattr(rc, "gross_cap_drawdown_8pct", 0.4)) if rc else 0.4

    trade_syms  = {s: v for s, v in rsr_syms.items() if s in universe_raw}
    active_syms = list(trade_syms.keys())

    strats = {}
    for sym, sector in trade_syms.items():
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        rule  = SECTOR_STRATEGY.get(sector, "fujiko")
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(
                min_rsr=cfg.fujiko.min_rsr, turtle_exit=cfg.fujiko.turtle_exit,
                rsr_series=rsr_s, min_sepa=cfg.fujiko.min_sepa,
                mom_period=cfg.fujiko.mom_period, turtle_entry=cfg.fujiko.turtle_entry,
                use_turtle_entry=cfg.fujiko.use_turtle_entry,
            )

    common_dates = None
    for sym in active_syms:
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    ts = pd.Timestamp(start); te = pd.Timestamp(end)
    common_dates = common_dates[(common_dates >= ts) & (common_dates <= te)]
    n_dates  = len(common_dates)
    n_syms   = len(active_syms)
    sym_to_i = {s: idx for idx, s in enumerate(active_syms)}

    open_mat  = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    close_mat = np.full((n_dates, n_syms), np.nan, dtype=np.float32)
    sig_mat   = np.zeros((n_dates, n_syms), dtype=np.int8)
    sig_ready = np.zeros(n_syms, dtype=bool)

    for si, sym in enumerate(active_syms):
        df_src  = universe_raw[sym]["df"]
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
                sig_ready[si]  = True

    rsr_mat = np.nan_to_num(
        _take(rsr_df, common_dates, active_syms, dtype=np.float32, fill_value=np.nan),
        nan=0.0,
    )
    sym_active_mat = None
    if sym_active_df is not None:
        sym_active_mat = _take(sym_active_df, common_dates, active_syms,
                               dtype=np.float32, fill_value=1.0)

    mkt_ret_arr = np.zeros(n_dates, dtype=np.float32)
    topix_ret20 = topix_ret60 = bear_arr = None
    if topix_close is not None:
        tr          = topix_close.pct_change()
        mkt_ret_arr = _take(tr, common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret20 = _take(topix_close.pct_change(20), common_dates, dtype=np.float32, fill_value=0.0)
        topix_ret60 = _take(topix_close.pct_change(60), common_dates, dtype=np.float32, fill_value=0.0)
        ma200  = topix_close.rolling(200, min_periods=100).mean()
        bear_s = (topix_close < ma200).reindex(pd.DatetimeIndex(common_dates), method="ffill").fillna(False)
        bear_arr = bear_s.values.astype(bool)

    # ── Portfolio simulation state ────────────────────────────────────
    cash         = float(capital)
    positions: dict[str, Position] = {}
    pos_meta: dict[str, dict] = {}
    peak_equity  = float(capital)
    cb_active    = False; cb_days = 0
    reentry_ban: dict[str, int] = {}
    trades: list[dict] = []

    # ── Signal records ────────────────────────────────────────────────
    all_signals: list[dict] = []

    for i, date in enumerate(common_dates):
        if i < 21:   # need 20 days lookback + 1 safety margin
            continue

        invested = sum(pos.qty * float(close_mat[i, sym_to_i[s]])
                       for s, pos in positions.items())
        cur_equity = cash + invested

        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        if not cb_active:
            if dd <= -max_dd_limit:
                cb_active = True; cb_days = 0
        else:
            cb_days += 1
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False; cb_days = 0
        cb_scale = CB_SCALE if cb_active else 1.0

        gross_cap = gross_normal
        if gross_enabled and topix_ret20 is not None:
            r20 = float(topix_ret20[i]); r60 = float(topix_ret60[i])
            if r20 < -0.05:
                gross_cap = gross_dd5
            elif r60 < -0.08:
                gross_cap = gross_dd8

        is_bear     = bool(bear_arr[i]) if bear_arr is not None else False
        sec_cap_eff = 0.18 if is_bear else MAX_SECTOR_W

        sell_sigs: list[tuple] = []
        buy_cands: list[tuple] = []
        mkt_shock = float(mkt_ret_arr[i]) <= shock_market_thr

        for sym in active_syms:
            si         = sym_to_i[sym]
            is_holding = sym in positions
            hold_idx   = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_mat[i, si])
            close_t    = float(close_mat[i, si])

            if mkt_shock:
                if not is_holding:
                    continue
                if i > 0:
                    prev_c = float(close_mat[i - 1, si])
                    if prev_c > 0 and (close_t / prev_c - 1.0) <= shock_sym_thr:
                        sell_sigs.append((sym, "MARKET_SHOCK_EXIT")); continue
            if mkt_shock and not is_holding:
                continue

            if is_holding and max_hold is not None and hold_idx > max_hold:
                sell_sigs.append((sym, "TIME_STOP")); continue
            if is_holding and rsr_val < rsr_exit_thr and hold_idx >= min_hold:
                sell_sigs.append((sym, "RSR_EXIT")); continue

            sig = int(sig_mat[i, si]) if sig_ready[si] else 0
            if sig == -1 and is_holding and hold_idx >= min_hold:
                sell_sigs.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not is_holding:
                if i < reentry_ban.get(sym, -1):
                    continue
                if sym_active_mat is not None:
                    if float(sym_active_mat[i, si]) < 0.5:
                        continue
                buy_cands.append((rsr_val, sym))

        if i + 1 >= n_dates:
            break
        next_i = i + 1

        for sym, reason in sell_sigs:
            if sym not in positions:
                continue
            pos      = positions[sym]
            sell_px  = float(open_mat[next_i, sym_to_i[sym]])
            proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
            pnl      = (sell_px - pos.entry_price) * pos.qty
            cash    += proceeds
            trades.append({"side": "SELL", "symbol": sym, "pnl": pnl})
            del positions[sym]
            del pos_meta[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        if not buy_cands:
            continue

        buy_cands.sort(key=lambda x: -x[0])

        # ── RSR feature computation + signal recording ─────────────────
        selected_syms_today = set()

        for rsr_val, sym in buy_cands:
            si = sym_to_i[sym]

            # RSR features (lookback safe: i >= 21)
            rsr_t   = float(rsr_mat[i, si])
            rsr_t5  = float(rsr_mat[i-5, si])  if i >= 5  else np.nan
            rsr_t10 = float(rsr_mat[i-10, si]) if i >= 10 else np.nan
            rsr_t20 = float(rsr_mat[i-20, si]) if i >= 20 else np.nan

            slope5  = (rsr_t - rsr_t5)  / 5  if not np.isnan(rsr_t5)  else np.nan
            slope20 = (rsr_t - rsr_t20) / 20 if not np.isnan(rsr_t20) else np.nan
            peak20  = float(np.nanmax(rsr_mat[i-20:i+1, si]))
            drawdown= rsr_t - peak20
            accel   = (slope5 - slope20) if not (np.isnan(slope5) or np.isnan(slope20)) else np.nan

            state   = classify_state(rsr_t, slope5, slope20, drawdown)
            rsr_bin = get_rsr_bin(rsr_t)

            sector     = trade_syms.get(sym, "不明")
            strat_name = SECTOR_STRATEGY.get(sector, "fujiko")

            sig_record = {
                "date":        str(date.date()),
                "ticker":      sym,
                "sector":      sector,
                "strategy":    strat_name,
                "rsr_t":       round(rsr_t, 1),
                "rsr_t5":      round(rsr_t5, 1)  if not np.isnan(rsr_t5)  else None,
                "rsr_t10":     round(rsr_t10, 1) if not np.isnan(rsr_t10) else None,
                "rsr_t20":     round(rsr_t20, 1) if not np.isnan(rsr_t20) else None,
                "slope5":      round(slope5, 4)  if not np.isnan(slope5)  else None,
                "slope20":     round(slope20, 4) if not np.isnan(slope20) else None,
                "peak20":      round(peak20, 1),
                "drawdown":    round(drawdown, 1),
                "accel":       round(accel, 4)   if not np.isnan(accel)   else None,
                "state":       state,
                "rsr_bin":     rsr_bin,
                "selected":    False,
                "skip_reason": "CAPACITY",   # default; updated below
                "entry_idx":   next_i,
                "fwd20":       None,
                "fwd60":       None,
                "fwd120":      None,
            }

            # ── Portfolio simulation: attempt buy ─────────────────────
            if cb_active:
                sig_record["skip_reason"] = "CB_ACTIVE"
                all_signals.append(sig_record)
                continue

            open_slots = max_pos - len(positions)
            if open_slots <= 0:
                sig_record["skip_reason"] = "CAPACITY"
                all_signals.append(sig_record)
                continue

            buy_px = float(open_mat[next_i, si])
            if buy_px <= 0:
                sig_record["skip_reason"] = "NO_PRICE"
                all_signals.append(sig_record)
                continue
            if not _sector_ok(sym, positions, close_mat, i, sym_to_i, trade_syms, capital, sec_cap_eff):
                sig_record["skip_reason"] = "SECTOR"
                all_signals.append(sig_record)
                continue
            if gross_enabled:
                cur_gross = sum(p.qty * float(close_mat[i, sym_to_i[p.symbol]])
                                for p in positions.values()) / max(1.0, capital)
                if cur_gross + buy_px * LOT / max(1.0, capital) > gross_cap:
                    sig_record["skip_reason"] = "GROSS_CAP"
                    all_signals.append(sig_record)
                    continue

            n_rem     = sum(1 for _, s in buy_cands if s not in positions)
            eff_slots = min(open_slots, max(1, n_rem))
            alloc     = (cash / eff_slots) * cb_scale
            alloc     = min(alloc, capital * MAX_SINGLE_W)
            qty       = int(alloc / buy_px / LOT) * LOT
            if qty <= 0:
                sig_record["skip_reason"] = "LOT_ZERO"
                all_signals.append(sig_record)
                continue

            # ── Execute ──────────────────────────────────────────────
            _execute_buy(sym, buy_px, qty, i, next_i, trade_syms, trades, positions, rsr_val)
            cash -= qty * buy_px * (1 + COST_ONE_WAY)
            pos_meta[sym] = {"rsr": rsr_val, "entry_date": str(date.date())}
            selected_syms_today.add(sym)

            sig_record["selected"]    = True
            sig_record["skip_reason"] = None
            all_signals.append(sig_record)

    # ── Post-hoc: forward returns ──────────────────────────────────────
    for rec in all_signals:
        ei  = rec["entry_idx"]
        sym = rec["ticker"]
        if sym not in sym_to_i:
            continue
        si = sym_to_i[sym]
        base_px = float(close_mat[ei, si]) if ei < n_dates and close_mat[ei, si] > 0 else 0
        if base_px <= 0:
            continue
        for h, key in [(20, "fwd20"), (60, "fwd60"), (120, "fwd120")]:
            fi = ei + h
            if fi < n_dates and close_mat[fi, si] > 0:
                rec[key] = round((float(close_mat[fi, si]) / base_px - 1) * 100, 2)

    return all_signals


# ─────────────────────────────────────────────────────────────────────
#  MATRIX BUILDER
# ─────────────────────────────────────────────────────────────────────

def build_matrix(df: pd.DataFrame, fwd_col: str = "fwd60") -> pd.DataFrame:
    """
    RSR bin × state のマトリクスを構築。
    各セル: n, avg_fwd20, avg_fwd60, avg_fwd120, hit_rate, PF, Sharpe_like
    """
    rows = []
    for rsr_bin in RSR_BINS:
        for state in RSR_STATES:
            cell = df[(df["rsr_bin"] == rsr_bin) & (df["state"] == state)]
            n = len(cell)
            f20  = cell["fwd20"].dropna().tolist()
            f60  = cell["fwd60"].dropna().tolist()
            f120 = cell["fwd120"].dropna().tolist()
            rows.append({
                "rsr_bin":   rsr_bin,
                "state":     state,
                "n":         n,
                "avg_fwd20": safe_mean(f20),
                "avg_fwd60": safe_mean(f60),
                "avg_fwd120":safe_mean(f120),
                "hit_rate":  hit_rate(f60),
                "pf":        safe_pf(f60),
                "sharpe":    safe_sharpe(f60),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────

def fmt_cell(v, decimals=2, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if isinstance(v, (int, np.integer)):
        return str(v)
    return f"{v:+.{decimals}f}{suffix}"


def write_report(
    df_signals: pd.DataFrame,
    mat: pd.DataFrame,
    output_path: Path,
) -> str:
    """Returns final answer A/B/C/D as string."""
    L = []; w = L.append

    w("# RSR 絶対水準 × 方向性状態 2D マトリクス研究")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  期間: 2018-2025  |  研究専用（実装変更なし）")
    w(f"\n全シグナル数: **{len(df_signals):,}件**  "
      f"(selected={df_signals['selected'].sum():,}件 "
      f"/ not_selected={len(df_signals)-df_signals['selected'].sum():,}件)")

    # ── Section 1: データ概要 ─────────────────────────────────────────
    w("\n---\n## 1. データ概要\n")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| 総シグナル件数 | {len(df_signals):,} |")
    w(f"| うち selected | {df_signals['selected'].sum():,} |")
    w(f"| うち not_selected | {(~df_signals['selected']).sum():,} |")

    # State distribution
    w("\n**状態別シグナル分布**\n")
    w("| State | N | % |")
    w("|---|---|---|")
    for s in RSR_STATES + ["UNKNOWN"]:
        cnt = (df_signals["state"] == s).sum()
        if cnt == 0:
            continue
        w(f"| {s} | {cnt:,} | {cnt/len(df_signals)*100:.1f}% |")

    # RSR bin distribution
    w("\n**RSR bin 別シグナル分布**\n")
    w("| RSR bin | N | avg_RSR | % |")
    w("|---|---|---|---|")
    for b in RSR_BINS:
        sub = df_signals[df_signals["rsr_bin"] == b]
        if len(sub) == 0:
            continue
        w(f"| {b} | {len(sub):,} | {sub['rsr_t'].mean():.1f} | {len(sub)/len(df_signals)*100:.1f}% |")

    # ── Section 2: 2D マトリクス (fwd60) ─────────────────────────────
    w("\n---\n## 2. 2D マトリクス: avg_fwd60 (%)\n")
    w("_行=RSR bin / 列=状態 / 値=avg_fwd60_\n")

    # avg_fwd60 table
    header = "| RSR bin |" + "".join(f" {s} |" for s in RSR_STATES)
    w(header)
    w("|---|" + "---|" * len(RSR_STATES))
    for rsr_bin in RSR_BINS:
        row = f"| **{rsr_bin}** |"
        for state in RSR_STATES:
            cell_row = mat[(mat["rsr_bin"] == rsr_bin) & (mat["state"] == state)]
            if len(cell_row) == 0:
                row += " — |"
                continue
            cr = cell_row.iloc[0]
            n = cr["n"]
            v = cr["avg_fwd60"]
            if n < MIN_CELL_N or v is None:
                row += f" (N={n}) |"
            else:
                row += f" {v:+.1f}% (N={n}) |"
        w(row)

    # hit rate table
    w("\n**hit_rate (fwd60>0) %**\n")
    w("| RSR bin |" + "".join(f" {s} |" for s in RSR_STATES))
    w("|---|" + "---|" * len(RSR_STATES))
    for rsr_bin in RSR_BINS:
        row = f"| **{rsr_bin}** |"
        for state in RSR_STATES:
            cr = mat[(mat["rsr_bin"] == rsr_bin) & (mat["state"] == state)]
            if len(cr) == 0:
                row += " — |"; continue
            cr = cr.iloc[0]
            v = cr["hit_rate"]
            row += (f" {v:.0f}% |" if v is not None and cr["n"] >= MIN_CELL_N else f" — |")
        w(row)

    # PF table
    w("\n**Profit Factor (fwd60 based)**\n")
    w("| RSR bin |" + "".join(f" {s} |" for s in RSR_STATES))
    w("|---|" + "---|" * len(RSR_STATES))
    for rsr_bin in RSR_BINS:
        row = f"| **{rsr_bin}** |"
        for state in RSR_STATES:
            cr = mat[(mat["rsr_bin"] == rsr_bin) & (mat["state"] == state)]
            if len(cr) == 0:
                row += " — |"; continue
            cr = cr.iloc[0]
            v = cr["pf"]
            row += (f" {v:.2f} |" if v is not None and cr["n"] >= MIN_CELL_N else f" — |")
        w(row)

    # Sharpe_like table
    w("\n**Sharpe_like (mean/std, fwd60)**\n")
    w("| RSR bin |" + "".join(f" {s} |" for s in RSR_STATES))
    w("|---|" + "---|" * len(RSR_STATES))
    for rsr_bin in RSR_BINS:
        row = f"| **{rsr_bin}** |"
        for state in RSR_STATES:
            cr = mat[(mat["rsr_bin"] == rsr_bin) & (mat["state"] == state)]
            if len(cr) == 0:
                row += " — |"; continue
            cr = cr.iloc[0]
            v = cr["sharpe"]
            row += (f" {v:+.2f} |" if v is not None and cr["n"] >= MIN_CELL_N else f" — |")
        w(row)

    # ── Section 3: avg_fwd20 / avg_fwd120 サブマトリクス ──────────────
    w("\n---\n## 3. 時間軸別 avg_fwd: 20d / 120d\n")
    for fwd_key, fwd_label in [("avg_fwd20", "avg_fwd20 (%)"), ("avg_fwd120", "avg_fwd120 (%)")]:
        w(f"\n**{fwd_label}**\n")
        w("| RSR bin |" + "".join(f" {s} |" for s in RSR_STATES))
        w("|---|" + "---|" * len(RSR_STATES))
        for rsr_bin in RSR_BINS:
            row = f"| **{rsr_bin}** |"
            for state in RSR_STATES:
                cr = mat[(mat["rsr_bin"] == rsr_bin) & (mat["state"] == state)]
                if len(cr) == 0:
                    row += " — |"; continue
                cr = cr.iloc[0]
                n  = cr["n"]
                v  = cr[fwd_key]
                row += (f" {v:+.1f}% |" if v is not None and n >= MIN_CELL_N else f" — |")
            w(row)

    # ── Section 4: H1 検証 — RSR水準 vs 期待値 ────────────────────────
    w("\n---\n## 4. H1 検証: RSR水準 vs fwd60期待値\n")
    w("_全状態合算。RSR高 → 期待値高 なら H1 支持。_\n")
    w("| RSR bin | N | avg_fwd60 | hit_rate | PF | Sharpe |")
    w("|---|---|---|---|---|---|")
    for rsr_bin in RSR_BINS:
        sub = df_signals[df_signals["rsr_bin"] == rsr_bin]
        f60 = sub["fwd60"].dropna().tolist()
        n   = len(f60)
        if n < MIN_CELL_N:
            w(f"| {rsr_bin} | {len(sub)} | — | — | — | — |")
            continue
        w(f"| {rsr_bin} | {n} "
          f"| {safe_mean(f60):+.2f}% "
          f"| {hit_rate(f60):.1f}% "
          f"| {safe_pf(f60):.2f} "
          f"| {safe_sharpe(f60) or 'N/A'} |")

    # ── Section 5: H3 検証 — RISING vs FALLING ────────────────────────
    w("\n---\n## 5. H3 検証: 同RSR帯での方向性比較\n")

    comparisons = [
        ("RSR≥75 RISING vs FALLING", "rsr_t>=75", "RISING", "FALLING"),
        ("RSR≥75 RISING vs POST_EXTREME", "rsr_t>=75", "RISING", "POST_EXTREME"),
        ("RSR75-89 RISING vs FALLING", "rsr_t>=75 and rsr_t<90", "RISING", "FALLING"),
        ("RSR≥90 RISING vs FALLING", "rsr_t>=90", "RISING", "FALLING"),
        ("RSR≥95 RISING vs POST_EXTREME", "rsr_t>=95", "RISING", "POST_EXTREME"),
        ("RSR≥75 COOLING vs FALLING", "rsr_t>=75", "COOLING", "FALLING"),
    ]

    w("| 比較 | N_A | avg_fwd60_A | N_B | avg_fwd60_B | return_diff | effect_size | 判定 |")
    w("|---|---|---|---|---|---|---|---|")
    h3_evidence = []
    for label, q_filter, state_a, state_b in comparisons:
        base = df_signals.query(q_filter) if q_filter else df_signals
        ga = base[base["state"] == state_a]["fwd60"].dropna().tolist()
        gb = base[base["state"] == state_b]["fwd60"].dropna().tolist()
        if len(ga) < MIN_CELL_N or len(gb) < MIN_CELL_N:
            w(f"| {label} | {len(ga)} | — | {len(gb)} | — | — | — | N/A |")
            continue
        ma = safe_mean(ga); mb = safe_mean(gb)
        diff  = round(ma - mb, 2) if ma is not None and mb is not None else None
        eff   = effect_size(ga, gb)
        sig   = "✅" if diff is not None and diff > 0 and eff is not None and eff > 0.1 else "❌"
        h3_evidence.append((label, diff, eff, sig))
        w(f"| {label} "
          f"| {len(ga)} | {ma:+.2f}% "
          f"| {len(gb)} | {mb:+.2f}% "
          f"| {diff:+.2f}% "
          f"| {eff:.3f} "
          f"| {sig} |")

    # ── Section 6: H2 検証 — 最適帯の探索 ───────────────────────────
    w("\n---\n## 6. H2 検証: 最適RSR帯の探索\n")
    w("_RISINGステートのみで RSR帯別 fwd60 を比較。_\n")
    w("| RSR bin | N | avg_fwd60 | hit_rate | PF | 最適帯候補 |")
    w("|---|---|---|---|---|---|")
    RELIABLE_N  = 20   # best_bin 選定に最低必要なサンプル数
    best_avg    = -999; best_bin = ""
    rising_df = df_signals[df_signals["state"] == "RISING"]
    for rsr_bin in RSR_BINS:
        sub = rising_df[rising_df["rsr_bin"] == rsr_bin]
        f60 = sub["fwd60"].dropna().tolist()
        n   = len(f60)
        if n < MIN_CELL_N:
            w(f"| {rsr_bin} | {n} | — | — | — | — |")
            continue
        pf_v = safe_pf(f60)
        avg_v = safe_mean(f60)
        # n>=RELIABLE_N の帯のみ "最適帯" 候補とする（少サンプル外れ値を除外）
        if n >= RELIABLE_N and avg_v is not None and avg_v > best_avg:
            best_avg = avg_v; best_bin = rsr_bin
        best_mark = "★" if rsr_bin == best_bin else ""
        w(f"| {rsr_bin} | {n} "
          f"| {avg_v:+.2f}% "
          f"| {hit_rate(f60):.1f}% "
          f"| {pf_v:.2f} "
          f"| {best_mark} |")

    # ── Section 7: POST_EXTREME 分析 ─────────────────────────────────
    w("\n---\n## 7. POST_EXTREME (RSR≥95かつ drawdown<0) 分析\n")
    pe_df = df_signals[df_signals["state"] == "POST_EXTREME"]
    ri95  = df_signals[(df_signals["state"] == "RISING") & (df_signals["rsr_t"] >= 95)]
    w(f"- POST_EXTREME シグナル数: {len(pe_df)} 件  "
      f"(avg drawdown: {pe_df['drawdown'].mean():.1f} RSR pt)\n")
    pe_f60  = pe_df["fwd60"].dropna().tolist()
    ri_f60  = ri95["fwd60"].dropna().tolist()
    diff95  = round(safe_mean(ri_f60) - safe_mean(pe_f60), 2) if ri_f60 and pe_f60 else None
    eff95   = effect_size(ri_f60, pe_f60)
    w("| グループ | N | avg_fwd60 | hit_rate | PF |")
    w("|---|---|---|---|---|")
    for label, f60 in [("RSR≥95 RISING", ri_f60), ("POST_EXTREME", pe_f60)]:
        if len(f60) >= MIN_CELL_N:
            w(f"| {label} | {len(f60)} | {safe_mean(f60):+.2f}% "
              f"| {hit_rate(f60):.1f}% | {safe_pf(f60):.2f} |")
        else:
            w(f"| {label} | {len(f60)} | — | — | — |")
    if diff95 is not None:
        w(f"\nRISING vs POST_EXTREME: return_diff={diff95:+.2f}%  effect_size={eff95:.3f}")
    c_h3 = "RSR高値圏(≥95)では上昇中 > ピーク後が確認" if diff95 and diff95 > 1 else "RSR高値 vs ピーク後: 差異不明確"
    w(f"→ {c_h3}")

    # ── Section 8: 現行戦略エントリー状態分布 ───────────────────────
    w("\n---\n## 8. 現行戦略 エントリー状態分布 (selected のみ)\n")
    sel = df_signals[df_signals["selected"] == True]
    w(f"総エントリー数: **{len(sel):,}件**\n")
    w("| State | N | % | avg_RSR | avg_fwd60 |")
    w("|---|---|---|---|---|")
    for state in RSR_STATES:
        sub = sel[sel["state"] == state]
        if len(sub) == 0:
            continue
        f60 = sub["fwd60"].dropna().tolist()
        w(f"| {state} | {len(sub)} | {len(sub)/max(len(sel),1)*100:.1f}% "
          f"| {sub['rsr_t'].mean():.1f} "
          f"| {safe_mean(f60) if len(f60)>=MIN_CELL_N else 'N/A'} |")
    w("\n| RSR bin | N | % |")
    w("|---|---|---|")
    for b in RSR_BINS:
        sub = sel[sel["rsr_bin"] == b]
        if len(sub) == 0:
            continue
        w(f"| {b} | {len(sub)} | {len(sub)/max(len(sel),1)*100:.1f}% |")

    # ── Section 9: 仮説検証サマリ + 最終回答 ─────────────────────────
    w("\n---\n## 9. 仮説検証サマリ\n")

    # H1: RSR monotone?
    rsr_bins_valid = []
    for rsr_bin in RSR_BINS:
        sub = df_signals[df_signals["rsr_bin"] == rsr_bin]
        f60 = sub["fwd60"].dropna().tolist()
        if len(f60) >= MIN_CELL_N:
            rsr_bins_valid.append((rsr_bin, safe_mean(f60), len(f60)))

    h1_monotone = True
    prev_avg = None
    for _, avg, _ in rsr_bins_valid:
        if prev_avg is not None and avg is not None and avg < prev_avg - 2.0:
            h1_monotone = False
            break
        prev_avg = avg

    # H3: direction matters?
    h3_support = sum(1 for _, diff, eff, sig in h3_evidence if sig == "✅")
    h3_ratio   = h3_support / max(len(h3_evidence), 1)

    # H2: optimal band?
    h2_nonmonotone = not h1_monotone

    w("| 仮説 | 内容 | 証拠 | 判定 |")
    w("|---|---|---|---|")
    h1_j = "✅ 支持" if h1_monotone else "⚠ 非単調（最適帯あり）"
    h2_j = "✅ 確認" if h2_nonmonotone else "❌ 証拠なし"
    h3_j = f"✅ 支持 ({h3_support}/{len(h3_evidence)}比較で有意)" if h3_ratio >= 0.5 else f"❌ 弱い ({h3_support}/{len(h3_evidence)})"
    w(f"| H1: 高RSR=良い | 単調増加か | {'単調' if h1_monotone else '非単調'} | {h1_j} |")
    w(f"| H2: 最適帯あり | RSR帯別差異 | {'非単調' if h2_nonmonotone else '単調'} | {h2_j} |")
    w(f"| H3: 方向性効果 | RISING>FALLING | {h3_support}/{len(h3_evidence)} | {h3_j} |")

    # POST_EXTREME penalty?
    pe_avg = safe_mean(pe_df["fwd60"].dropna().tolist()) or 0
    ri95_avg = safe_mean(ri95["fwd60"].dropna().tolist()) or 0
    pe_penalty = pe_avg < ri95_avg - 2.0

    w("\n---\n## 10. 最終回答\n")

    if h3_ratio >= 0.5 and h2_nonmonotone:
        answer = "D"
        reason = f"最適RSR帯({best_bin})と方向性(RISING優位)の組み合わせが必要。H2+H3 両方支持。"
    elif h3_ratio >= 0.5:
        answer = "B"
        reason = f"RSR水準は必要条件だが方向性(RISING vs FALLING)で期待値が反転。H3支持。"
    elif pe_penalty:
        answer = "C"
        reason = f"RSR高値圏(≥95)は過熱 → POST_EXTREME(ピーク後)の期待値が低い。H3部分支持。"
    elif h1_monotone:
        answer = "A"
        reason = "RSR単独で単調に期待値が改善。方向性追加は不要。H1支持。"
    else:
        answer = "D"
        reason = "最適RSR帯が存在し、方向性効果も複合的。"

    answer_labels = {
        "A": "RSR単独で十分",
        "B": "RSR×方向性が必要",
        "C": "RSR高値は過熱",
        "D": "最適帯＋方向性が必要",
    }
    w(f"### 最終回答: **{answer}: {answer_labels[answer]}**\n")
    w(f"根拠: {reason}\n")

    w("**実装示唆:**")
    if answer in ("B", "D"):
        w("- エントリーフィルタに slope20 / slope5 を追加することで期待値改善の可能性")
        w("- 特に COOLING / FALLING 状態のシグナルをスキップすることを検討（ASK_FIRST）")
    if answer == "C":
        w("- RSR≥95 のシグナルは POST_EXTREME 判定時にスキップを検討")
    if answer == "A":
        w("- 現行 RSR≥75 entry threshold で十分。方向性フィルタ追加は過学習リスク。")
    if answer == "D":
        w(f"- 最適帯: RSR {best_bin} × RISING が最高 PF")
        w("- ただし実装前に WF 検証必須（過学習チェック）")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"  MD レポート保存: {output_path}")
    return answer


# ─────────────────────────────────────────────────────────────────────
#  CSV OUTPUT
# ─────────────────────────────────────────────────────────────────────

def save_csvs(df_signals: pd.DataFrame, mat: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # rsr_state_matrix.csv
    mat_pivot = mat.copy()
    mat_path = REPORTS_DIR / "rsr_state_matrix.csv"
    mat_pivot.to_csv(mat_path, index=False, encoding="utf-8-sig")
    print(f"  Matrix CSV: {mat_path}")

    # entry_state_distribution.csv
    sel = df_signals[df_signals["selected"] == True].copy()
    dist_rows = []
    for state in RSR_STATES:
        for rsr_bin in RSR_BINS:
            sub = sel[(sel["state"] == state) & (sel["rsr_bin"] == rsr_bin)]
            if len(sub) == 0:
                continue
            f60 = sub["fwd60"].dropna().tolist()
            dist_rows.append({
                "state": state,
                "rsr_bin": rsr_bin,
                "n": len(sub),
                "pct_of_entries": round(len(sub) / max(len(sel), 1) * 100, 1),
                "avg_rsr": round(sub["rsr_t"].mean(), 1),
                "avg_fwd60": safe_mean(f60),
                "hit_rate": hit_rate(f60),
                "pf": safe_pf(f60),
            })
    dist_path = REPORTS_DIR / "entry_state_distribution.csv"
    pd.DataFrame(dist_rows).to_csv(dist_path, index=False, encoding="utf-8-sig")
    print(f"  Distribution CSV: {dist_path}")

    # raw signals (サマリー用 - fwd60あるもののみ)
    raw_path = REPORTS_DIR / "rsr_state_signals_raw.csv"
    cols = ["date","ticker","sector","strategy","rsr_t","slope5","slope20",
            "drawdown","accel","state","rsr_bin","selected","skip_reason",
            "fwd20","fwd60","fwd120"]
    df_signals[cols].to_csv(raw_path, index=False, encoding="utf-8-sig")
    print(f"  Raw signals CSV: {raw_path}")


# ─────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────

def main() -> int:
    cfg = load_strategy_config()

    print("=" * 68)
    print("  RSR 絶対水準 × 方向性状態 2D マトリクス研究")
    print("=" * 68 + "\n")

    print("[1/4] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/4] シグナル収集 + ポートフォリオシミュレーション中...")
    t0 = time.time()
    all_signals = collect_signals(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=FULL_START, end=FULL_END,
    )
    print(f"  完了: {len(all_signals):,}件  ({time.time()-t0:.1f}s)")

    df = pd.DataFrame(all_signals)
    print(f"  selected={df['selected'].sum():,}  not_selected={(~df['selected']).sum():,}")

    # State distribution console
    print("\n  State 分布:")
    for state in RSR_STATES + ["UNKNOWN"]:
        cnt = (df["state"] == state).sum()
        if cnt > 0:
            print(f"    {state:<15}: {cnt:>5}件  ({cnt/len(df)*100:.1f}%)")

    print("\n[3/4] マトリクス構築中...")
    mat = build_matrix(df, fwd_col="fwd60")

    # Quick console matrix preview
    print("\n  ★ 2D Matrix Preview (avg_fwd60)")
    header_str = f"  {'RSR bin':<10}"
    for s in RSR_STATES:
        header_str += f" {s:<14}"
    print(header_str)
    for rsr_bin in RSR_BINS:
        row_str = f"  {rsr_bin:<10}"
        for state in RSR_STATES:
            cr = mat[(mat["rsr_bin"] == rsr_bin) & (mat["state"] == state)]
            if len(cr) == 0:
                row_str += f" {'—':<14}"; continue
            cr = cr.iloc[0]
            n = cr["n"]; v = cr["avg_fwd60"]
            if n < MIN_CELL_N or v is None:
                row_str += f" {'n<'+str(MIN_CELL_N):<14}"
            else:
                cell = f"{v:+.1f}%(N={n})"
                row_str += f" {cell:<14}"
        print(row_str)

    print("\n[4/4] CSV + レポート生成中...")
    save_csvs(df, mat)
    answer = write_report(df, mat, REPORTS_DIR / "rsr_state_matrix.md")

    print(f"\n{'='*68}")
    print(f"  最終回答: {answer}")
    print(f"{'='*68}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
