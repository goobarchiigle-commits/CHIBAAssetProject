"""
backtest/composite_alpha_bt.py

3ステップ改善バックテスト（比較実験）

BASELINE : RSR≥75 → RSR降順ランキング → top-3 均等ウェイト（現行）
STEP1    : RSR≥75 → (slope×r2)² × RSR 複合スコアランキング → top-3 均等ウェイト
           ※ alpha² × RSR: alphaを主役にしてRSRは乗数として機能させる
STEP2    : STEP1  + 2段階マクロレジームフィルター
            A（Risk Off）: TOPIX < MA200 → 新規BUY停止
            B（Crash）   : MA50 < MA200  → ポジションサイズ半減
           + CB修正: 30営業日タイムアウト解除 + CB時35%スケール（完全停止廃止）
STEP3    : STEP2  + αウェイトポジションサイジング（均等→αランク比例）

実行:
  python -m backtest.composite_alpha_bt
  python -m backtest.composite_alpha_bt --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from dataclasses import dataclass, field
from datetime import date as date_type
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from paths import CONFIGS_DIR, UNIVERSE_DIR, RESULTS_DIR, BACKTEST_DATASET_DIR

import numpy as np
import pandas as pd
import yfinance as yf

from backtest.rsr                     import calc_universe_rsr, calc_composite_alpha_matrix
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.universe_builder        import download_universe


# ------------------------------------------------------------------ #
# 定数（live_equivalent.py 確定値と揃える）
# ------------------------------------------------------------------ #
RSR_UNIVERSE_CSV      = CONFIGS_DIR / "rsr_universe_42.csv"
TRADING_UNIVERSE_JSON = UNIVERSE_DIR / "2026Q1_temporal24.json"

START          = "2018-01-01"
END            = "2024-12-31"
CAPITAL        = 3_000_000
MAX_POSITIONS  = 3
MAX_HOLD_DAYS  = 60
MIN_RSR        = 75.0
MAX_DD_LIMIT   = 0.15
REENTRY_COOL   = 5
LOT            = 100
SLIPPAGE       = 0.001
COMMISSION     = 0.00055
COST_ONE_WAY   = SLIPPAGE + COMMISSION
MIN_SEPA       = 6
MOM_PERIOD     = 21
TURTLE_ENTRY   = 20
TURTLE_EXIT    = 55   # 2026-03-31: exit感度テストで20→55に変更（IS Sharpe +12.6%）

COMP_ALPHA_WINDOW = 90   # Composite Alpha 計算ウィンドウ

# Step3: αウェイト上限（ユーザー指定 0.40）
MAX_POS_WEIGHT = 0.40

# CBデッドロック防止
CB_UNLOCK_DAYS = 30    # CB発動後30営業日で強制解除（市場回復の可能性を担保）
CB_SCALE       = 0.35  # CB時のポジションスケール（完全停止→縮小に変更）

# STEP4: ブレイクアウトスコアボーナス
BREAKOUT_LOOKBACK = 200   # 200日高値ブレイク判定期間（フィルターではなく加点）
BREAKOUT_BONUS    = 1.25  # ブレイクアウト時のスコア乗数
# STEP5: トレーリングストップ
TRAIL_PERIOD      = 50    # トレーリング参照期間（50日最高終値）
TRAIL_ATR_MULT    = 3.0   # トレーリングストップのATR倍率
# STEP6: Market Breadthレジーム
BREADTH_STOP      = 0.25  # breadth < 25%: 新規BUY停止
BREADTH_REDUCE    = 0.15  # breadth < 15%: ポジション縮小（regime_step=0.5）

SECTOR_STRATEGY: dict[str, str] = {
    "海運":     "fujiko", "機械":     "fujiko", "電機精密": "fujiko",
    "商社":     "fujiko", "電機":     "fujiko", "ゲーム":   "fujiko",
    "レジャー": "fujiko", "食品":     "fujiko",
    "ガス":     "mean_rev","鉄鋼":    "mean_rev","銀行":     "mean_rev",
    "保険":     "mean_rev","輸送機器":"mean_rev","化学":     "mean_rev",
    "小売":     "mean_rev",
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

OUTPUT_DIR = RESULTS_DIR
TOPIX_SYMBOL = "1306.T"   # NEXT FUNDS TOPIX ETF（TOPIXプロキシ）


# ------------------------------------------------------------------ #
# データ構造
# ------------------------------------------------------------------ #
@dataclass
class Position:
    symbol:      str
    sector:      str
    qty:         int
    entry_price: float
    entry_idx:   int
    alpha_score: float = 0.0   # エントリー時のcomposite alpha（Step3用）


# ------------------------------------------------------------------ #
# ユーティリティ
# ------------------------------------------------------------------ #
def _load_rsr_universe() -> dict[str, str]:
    df = pd.read_csv(RSR_UNIVERSE_CSV)
    return {row["symbol"]: row.get("sector", "不明") for _, row in df.iterrows()}


def _load_trading_universe() -> dict[str, str]:
    data = json.loads(TRADING_UNIVERSE_JSON.read_text(encoding="utf-8"))
    return data["symbols"]


def _download_topix(start: str, end: str) -> pd.Series:
    """TOPIX ETF (1306.T) を取得して日次終値を返す。"""
    try:
        df = yf.download(TOPIX_SYMBOL, start=start, end=end, progress=False)
        if df.empty:
            return pd.Series(dtype=float)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df["Close"].dropna()
        close.index = pd.to_datetime(close.index)
        return close
    except Exception as e:
        print(f"  ⚠ TOPIX取得失敗（{e}）→ レジームフィルター無効化")
        return pd.Series(dtype=float)


def _calc_regime(topix: pd.Series) -> pd.DataFrame:
    """
    2段階マクロレジームフィルターを計算する。

    Returns DataFrame with columns:
      risk_off : True = TOPIX < MA200（新規BUY停止）
      crash    : True = MA50 < MA200（ポジションサイズ半減）
    """
    if topix.empty:
        idx = pd.date_range(START, END, freq="B")
        return pd.DataFrame({"risk_off": False, "crash": False}, index=idx)

    ma200 = topix.rolling(200, min_periods=100).mean()
    ma50  = topix.rolling(50,  min_periods=25).mean()

    risk_off = topix < ma200
    crash    = ma50  < ma200

    regime = pd.DataFrame({"risk_off": risk_off, "crash": crash})
    regime = regime.fillna(False)
    return regime


def _precompute_tech_matrices(universe_raw: dict, symbols: list) -> dict:
    """
    STEP4/5用テクニカル指標をまとめて事前計算。

    Returns
    -------
    dict with keys:
      atr20        : ATR20（先読み防止なし・当日値）
      atr20_med90  : ATR20の90日中央値（shift(1)済み）
      high200      : 200日最高値 High（shift(1)済み）- ブレイクアウトボーナス用
      high50_close : 50日最高終値 Close（shift(1)なし）- トレーリング用
    """
    atr20_d        = {}
    atr20_med90_d  = {}
    high200_d      = {}
    high50_close_d = {}

    for sym in symbols:
        if sym not in universe_raw:
            continue
        df = universe_raw[sym]["df"]
        h = df["High"]
        l = df["Low"]
        c = df["Close"]
        c_prev = c.shift(1)

        tr   = pd.concat([h - l, (h - c_prev).abs(), (l - c_prev).abs()], axis=1).max(axis=1)
        atr20 = tr.rolling(20, min_periods=10).mean()

        atr20_d[sym]        = atr20
        atr20_med90_d[sym]  = atr20.rolling(90, min_periods=45).median().shift(1)
        high200_d[sym]      = h.rolling(BREAKOUT_LOOKBACK, min_periods=BREAKOUT_LOOKBACK // 2).max().shift(1)
        high50_close_d[sym] = c.rolling(TRAIL_PERIOD, min_periods=TRAIL_PERIOD // 2).max()

    return {
        "atr20":        pd.DataFrame(atr20_d),
        "atr20_med90":  pd.DataFrame(atr20_med90_d),
        "high200":      pd.DataFrame(high200_d),
        "high50_close": pd.DataFrame(high50_close_d),
    }


def _calc_breadth(rsr_df: pd.DataFrame, threshold: float = 75.0) -> pd.Series:
    """
    RSR≥threshold の割合を毎日計算（Market Breadth インジケータ）。
    shift(1) で先読み防止済み。
    """
    strong = (rsr_df >= threshold).sum(axis=1)
    total  = rsr_df.shape[1]
    return (strong / max(1, total)).shift(1).fillna(0.0)


def _alpha_weights(candidates: list[tuple], max_weight: float = MAX_POS_WEIGHT) -> dict[str, float]:
    """
    候補銘柄のαスコアに比例したウェイトを計算する。

    Parameters
    ----------
    candidates : [(alpha_score, sym), ...]  ← alphaは正値のみ渡す想定
    max_weight : 1銘柄の最大ウェイト上限

    Returns
    -------
    {sym: weight}  ← 合計=1.0, 各 ≤ max_weight
    """
    if not candidates:
        return {}
    scores = np.array([max(0.0, a) for a, _ in candidates])
    syms   = [s for _, s in candidates]

    if scores.sum() < 1e-8:
        # 全スコアゼロ → 均等ウェイト
        w = np.ones(len(syms)) / len(syms)
    else:
        w = scores / scores.sum()
        w = np.clip(w, 0, max_weight)
        w = w / w.sum()  # 正規化（clip後）

    return {sym: float(wt) for sym, wt in zip(syms, w)}


# ------------------------------------------------------------------ #
# 汎用バックテスト
# ------------------------------------------------------------------ #
def run_scenario(
    scenario:       str,   # "BASELINE"|"STEP1"|"STEP2"|"STEP3"|"STEP4"|"STEP5"|"STEP6"
    universe_raw:   dict,
    rsr_df:         pd.DataFrame,
    alpha_df:       pd.DataFrame,  # composite alpha matrix（BASELINE時は未使用）
    regime_df:      pd.DataFrame,
    trade_syms:     dict[str, str],
    rsr_syms:       dict[str, str],
    start:          str = START,
    end:            str = END,
    capital:        float = CAPITAL,
    verbose:        bool = True,
    tech_matrices:  dict | None = None,   # STEP4+: 事前計算テクニカル指標
    breadth_series: pd.Series | None = None,  # STEP6+: Market Breadth
    breadth_stop:        float = BREADTH_STOP,    # STEP6+: BUY停止閾値（調整可能）
    breadth_reduce:      float = BREADTH_REDUCE,  # STEP6+: 縮小閾値（調整可能）
    min_hold:            int   = 0,               # 最低保有日数（0=無効）
    rsr_exit_threshold:  float = MIN_RSR,         # RSR exitの閾値（デフォルト=MIN_RSR=75）
) -> dict:
    """
    汎用バックテストループ。

    scenario別の挙動:
      BASELINE : RSR降順ランキング / 均等ウェイト / レジームなし
      STEP1    : alpha複合スコアランキング / 均等ウェイト / レジームなし
      STEP2    : alpha複合スコアランキング / 均等ウェイト / 2段階レジームフィルター
      STEP3    : alpha複合スコアランキング / αウェイト / 2段階レジームフィルター
    """
    use_alpha_rank      = (scenario != "BASELINE")
    use_regime          = (scenario in ("STEP2", "STEP3"))
    use_alpha_weight    = (scenario == "STEP3")
    use_breakout_filter = (scenario in ("STEP4", "STEP5", "STEP6", "STEP6A", "STEP6B"))
    use_trail_exit      = (scenario in ("STEP5", "STEP6", "STEP6A", "STEP6B"))
    use_breadth_regime  = (scenario in ("STEP6", "STEP6A", "STEP6B"))

    # ---- 戦略初期化 ----
    strats: dict[str, object] = {}
    for sym, sector in trade_syms.items():
        if sym not in universe_raw:
            continue
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        rule  = SECTOR_STRATEGY.get(sector, "fujiko")
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(
                rsr_series       = rsr_s,
                min_sepa         = MIN_SEPA,
                min_rsr          = MIN_RSR,
                mom_period       = MOM_PERIOD,
                turtle_entry     = TURTLE_ENTRY,
                turtle_exit      = TURTLE_EXIT,
                use_turtle_entry = True,
            )

    # ---- 共通取引日 ----
    common_dates: pd.DatetimeIndex | None = None
    for sym in trade_syms:
        if sym not in universe_raw:
            continue
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    if common_dates is None or len(common_dates) == 0:
        return {}
    common_dates = common_dates.sort_values()
    ts_f = pd.Timestamp(start)
    te_f = pd.Timestamp(end)
    common_dates = common_dates[(common_dates >= ts_f) & (common_dates <= te_f)]

    # ---- ポートフォリオ変数 ----
    cash         = float(capital)
    positions:   dict[str, Position] = {}
    equity_curve = []
    exposure_list= []
    pos_list     = []
    cand_list    = []
    trades:      list[dict] = []
    exit_reason_counts: dict[str, int] = {}   # exit reason 集計
    peak_equity  = float(capital)
    cb_active    = False
    cb_days      = 0     # CB発動からの経過営業日（30日で強制解除）
    reentry_ban: dict[str, int] = {}

    regime_step = 1.0   # Step2: crash時に0.5

    for i, date in enumerate(common_dates):

        # ── Equity & DD ─────────────────────────────────────────────
        invested = sum(
            pos.qty * float(universe_raw[sym]["df"].loc[date, "Close"])
            for sym, pos in positions.items()
            if sym in universe_raw and date in universe_raw[sym]["df"].index
        )
        cur_equity = cash + invested
        equity_curve.append(cur_equity)
        exposure_list.append(invested / max(1.0, cur_equity))
        pos_list.append(len(positions))

        if cur_equity > peak_equity:
            peak_equity = cur_equity
        dd = (cur_equity - peak_equity) / peak_equity

        # CB状態機械（デッドロック防止）
        # 毎日上書きをやめ、発動・解除を明示的に管理する
        if not cb_active:
            if dd <= -MAX_DD_LIMIT:
                cb_active = True
                cb_days   = 0
        else:
            cb_days += 1
            # 解除条件: 30営業日経過 OR equity回復（DD > -5%）
            if cb_days >= CB_UNLOCK_DAYS or dd > -0.05:
                cb_active = False
                cb_days   = 0

        # ── マクロレジーム取得（Step2/3: TOPIX MA / Step6: Breadth） ──────
        risk_off    = False
        crash       = False
        regime_step = 1.0

        if use_regime and date in regime_df.index:
            risk_off    = bool(regime_df.loc[date, "risk_off"])
            crash       = bool(regime_df.loc[date, "crash"])
            regime_step = 0.5 if crash else 1.0

        if use_breadth_regime and breadth_series is not None and date in breadth_series.index:
            bval = float(breadth_series.loc[date])
            if bval < breadth_stop:
                risk_off = True                        # breadth低下: 新規BUY停止
            if bval < breadth_reduce:
                regime_step = min(regime_step, 0.5)   # breadth急落: 縮小

        # ── RSR行取得 ────────────────────────────────────────────
        rsr_row = rsr_df.loc[date] if date in rsr_df.index else pd.Series(dtype=float)

        # ── Composite Alpha行取得（Step1/2/3） ─────────────────────
        if use_alpha_rank and date in alpha_df.index:
            alpha_row = alpha_df.loc[date]
        else:
            alpha_row = pd.Series(dtype=float)

        # ── シグナル生成 ────────────────────────────────────────
        sell_signals:    list[tuple]  = []
        buy_candidates:  list[tuple]  = []   # (rank_score, alpha_score, sym)

        for sym in trade_syms:
            if sym not in universe_raw or sym not in strats:
                continue
            df_sym = universe_raw[sym]["df"]
            if date not in df_sym.index:
                continue

            is_holding = sym in positions
            hold_idx   = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_row.get(sym, 0.0)) if sym in rsr_row.index else 0.0

            # 時間ストップ
            if is_holding and hold_idx > MAX_HOLD_DAYS:
                sell_signals.append((sym, "TIME_STOP"))
                continue

            # RSR低下エグジット（min_hold: 最低保有日数を満たした場合のみ）
            if is_holding and rsr_val < rsr_exit_threshold and hold_idx >= min_hold:
                sell_signals.append((sym, "RSR_EXIT"))
                continue

            # トレーリングストップ（STEP5+）
            if use_trail_exit and is_holding and tech_matrices is not None:
                tm = tech_matrices
                if (date in tm["high50_close"].index and sym in tm["high50_close"].columns
                        and date in tm["atr20"].index and sym in tm["atr20"].columns):
                    h50c  = float(tm["high50_close"].loc[date, sym])
                    atr20 = float(tm["atr20"].loc[date, sym])
                    close_today = float(df_sym.loc[date, "Close"])
                    if not (np.isnan(h50c) or np.isnan(atr20) or atr20 <= 0):
                        trail_stop = h50c - TRAIL_ATR_MULT * atr20
                        if close_today < trail_stop:
                            sell_signals.append((sym, "TRAIL_EXIT"))
                            continue

            # 戦略シグナル
            past = df_sym.loc[:date]
            if len(past) < 30:
                continue
            sig = strats[sym].generate_signal(past)

            if sig == -1 and is_holding and hold_idx >= min_hold:
                sell_signals.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not is_holding:
                ban_until = reentry_ban.get(sym, -1)
                if i < ban_until:
                    continue

                # ブレイクアウトスコアボーナス（STEP4+）
                # フィルターではなく加点: 200日高値ブレイク銘柄のスコアを1.25倍
                breakout_mult = 1.0
                if use_breakout_filter and tech_matrices is not None:
                    tm = tech_matrices
                    close_today = float(df_sym.loc[date, "Close"])
                    if date in tm["high200"].index and sym in tm["high200"].columns:
                        h200 = float(tm["high200"].loc[date, sym])
                        if not np.isnan(h200) and close_today >= h200:
                            breakout_mult = BREAKOUT_BONUS

                # composite alpha スコア
                a_val = float(alpha_row.get(sym, 0.0)) if sym in alpha_row.index else 0.0
                if use_alpha_rank:
                    # alpha² × RSR × breakout_bonus（STEP4+は乗数追加）
                    rank_score = (max(0.0, a_val) ** 2) * rsr_val * breakout_mult
                else:
                    rank_score = rsr_val
                buy_candidates.append((rank_score, a_val, sym))

        cand_list.append(len(buy_candidates))

        # ── 翌日 ────────────────────────────────────────────────────
        if i + 1 >= len(common_dates):
            break
        next_date = common_dates[i + 1]

        # ── SELL 実行 ────────────────────────────────────────────────
        for sym, reason in sell_signals:
            if sym not in positions:
                continue
            pos    = positions[sym]
            df_sym = universe_raw[sym]["df"]
            if next_date not in df_sym.index:
                continue
            sell_px  = float(df_sym.loc[next_date, "Open"])
            proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
            pnl      = (sell_px - pos.entry_price) * pos.qty
            cash    += proceeds
            trades.append({"symbol": sym, "side": "SELL",
                           "entry": pos.entry_price, "exit": sell_px,
                           "qty": pos.qty, "pnl": pnl, "reason": reason,
                           "entry_idx": pos.entry_idx, "exit_idx": i})
            exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOL

        # ── BUY 実行 ─────────────────────────────────────────────────
        # レジームA（risk_off）: 新規BUY完全停止
        # CB: 完全停止ではなくスケールダウン（CB_SCALE=0.35）に変更
        buy_blocked = use_regime and risk_off

        if not buy_blocked and buy_candidates:
            # ランクスコア降順
            buy_candidates.sort(key=lambda x: -x[0])

            if use_alpha_weight:
                # Step3: αスコアに比例したウェイト
                top_cands = buy_candidates[:MAX_POSITIONS]
                wt_map    = _alpha_weights(
                    [(a, s) for _, a, s in top_cands],
                    max_weight=MAX_POS_WEIGHT,
                )
            else:
                wt_map = {}

            # CB時はポジションサイズを35%に縮小（完全停止からの変更）
            cb_scale = CB_SCALE if cb_active else 1.0

            new_buys = 0
            for rank_score, a_val, sym in buy_candidates:
                open_slots = MAX_POSITIONS - len(positions)
                if open_slots <= 0:
                    break

                df_sym = universe_raw[sym]["df"]
                if next_date not in df_sym.index:
                    continue
                buy_px = float(df_sym.loc[next_date, "Open"])
                if buy_px <= 0:
                    continue

                # ウェイト計算（CB時はcb_scaleを適用）
                if use_alpha_weight and sym in wt_map:
                    target_weight = wt_map[sym] * regime_step
                    alloc = capital * target_weight * cb_scale
                else:
                    # 均等ウェイト（Step2のcrashは資金半減）
                    n_remaining = sum(1 for _, _, s in buy_candidates
                                      if s not in positions)
                    effective_slots = min(open_slots, max(1, n_remaining))
                    alloc = (cash / effective_slots) * regime_step * cb_scale

                # 価格フィルター上限（現行: ¥600,000 per lot）
                alloc = min(alloc, capital * 0.25)

                qty = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue

                cash -= qty * buy_px * (1 + COST_ONE_WAY)
                positions[sym] = Position(
                    symbol=sym, sector=trade_syms.get(sym, "不明"),
                    qty=qty, entry_price=buy_px, entry_idx=i + 1,
                    alpha_score=a_val,
                )
                trades.append({"symbol": sym, "side": "BUY",
                               "entry": buy_px, "exit": None,
                               "qty": qty, "pnl": None,
                               "reason": f"alpha={a_val:.4f}/RSR={rsr_val:.1f}"})
                new_buys += 1

    # ---- 指標計算 ----
    n_days = len(equity_curve)
    if n_days == 0:
        return {}
    eq     = pd.Series(equity_curve, index=common_dates[:n_days])
    if eq.empty:
        return {}
    years  = n_days / 252
    cagr   = (eq.iloc[-1] / capital) ** (1 / max(years, 0.01)) - 1

    dr     = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0

    roll_max = eq.expanding().max()
    dd_ser   = (eq - roll_max) / roll_max
    max_dd   = float(dd_ser.min())
    calmar   = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    sells    = [t for t in trades if t["side"] == "SELL" and t["pnl"] is not None]
    win_rate = sum(1 for t in sells if t["pnl"] > 0) / max(1, len(sells))

    # 勝ちトレードの平均リターン（R multiple的な指標）
    win_trades  = [t for t in sells if t["pnl"] > 0]
    lose_trades = [t for t in sells if t["pnl"] <= 0]
    avg_win_pct  = float(np.mean([(t["exit"] / t["entry"] - 1) * 100
                                  for t in win_trades])) if win_trades else 0.0
    avg_lose_pct = float(np.mean([(t["exit"] / t["entry"] - 1) * 100
                                  for t in lose_trades])) if lose_trades else 0.0
    r_multiple   = abs(avg_win_pct / avg_lose_pct) if avg_lose_pct != 0 else float("inf")

    # 年ごとの損益
    annual = {}
    for yr, grp in eq.groupby(eq.index.year):
        y_ret = float(grp.iloc[-1] / grp.iloc[0] - 1)
        annual[str(yr)] = round(y_ret * 100, 2)

    # データセットバージョン（再現性チェック用）
    import os as _os
    _dataset_version = _os.environ.get("DATA_VERSION", "live_yfinance")

    return {
        "scenario":       scenario,
        "dataset_version": _dataset_version,   # ← 再現性チェック用
        "cagr":           round(cagr * 100, 2),
        "sharpe":         round(sharpe, 3),
        "max_dd":         round(max_dd * 100, 2),
        "calmar":         round(calmar, 3),
        "n_trades":       len(sells),
        "n_trades_yr":    round(len(sells) / max(years, 0.01), 1),
        "avg_hold_days":  round(float(np.mean([t["exit_idx"] - t["entry_idx"]
                                               for t in trades if t["side"] == "SELL"
                                               and "exit_idx" in t and "entry_idx" in t]))
                                if sells else 0, 1),
        "win_rate":       round(win_rate * 100, 1),
        "avg_win_pct":    round(avg_win_pct, 2),
        "avg_lose_pct":   round(avg_lose_pct, 2),
        "r_multiple":     round(r_multiple, 2),
        "avg_exposure":   round(float(np.mean(exposure_list)) * 100, 1),
        "avg_candidates":              round(float(np.mean(cand_list)), 2),
        "avg_simultaneous_holdings":  round(float(np.mean(pos_list)), 2),
        "annual_returns":     annual,
        "equity_curve":       eq,
        "exit_reason_counts": exit_reason_counts,
        "_trades":            [t for t in trades if t["side"] == "SELL"],
    }


# ------------------------------------------------------------------ #
# メイン
# ------------------------------------------------------------------ #
def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--start",       default=START)
    p.add_argument("--end",         default=END)
    args = p.parse_args()

    print("=" * 70)
    print("  Composite Alpha × Regime Filter 3ステップ改善バックテスト")
    print(f"  期間: {args.start} 〜 {args.end}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 70)

    # ---- 1. ユニバース読み込み ----
    # 2026-03-30: RSR42に統一（TEMPORAL24廃止）
    rsr_syms   = _load_rsr_universe()
    trade_syms = rsr_syms   # RSR42を取引・RSR両方のユニバースとして使用
    print(f"\n[1/6] ユニバース: RSR42統一（取引・RSR共通 {len(trade_syms)}銘柄）")

    # ---- 再現性ログ（Step1 基準確認用） ----
    import os as _os
    _dataset_version = _os.environ.get("DATA_VERSION", "live_yfinance")
    _dataset_hash    = "unknown"
    _meta_path       = BACKTEST_DATASET_DIR / _dataset_version / "_meta.json"
    if _meta_path.exists():
        import json as _json
        with open(_meta_path, encoding="utf-8") as _f:
            _dataset_hash = _json.load(_f).get("snapshot_hash", "unknown")
    print(f"\nDATASET_VERSION  {_dataset_version}")
    print(f"DATASET_HASH     {_dataset_hash}")
    print(f"CAPITAL          {CAPITAL:,}")
    print(f"UNIVERSE         {len(trade_syms)}")
    print()

    # ---- 2. 価格データ取得 ----
    all_syms = {**rsr_syms, **trade_syms}   # RSR42 trade時は重複するが問題なし
    print(f"[2/6] 価格データ取得中（{len(all_syms)}銘柄 + TOPIX ETF）...")
    universe_raw = download_universe(all_syms, start=args.start, end=args.end, verbose=False)
    topix_close  = _download_topix(args.start, args.end)
    print(f"  取得完了: {len(universe_raw)}銘柄, TOPIX={len(topix_close)}日")

    # ---- 3. RSR 計算 ----
    print("[3/6] RSR計算中（42銘柄コンテキスト）...")
    rsr42_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in rsr_syms if sym in universe_raw
    }
    rsr_df = calc_universe_rsr(rsr42_prices)
    print(f"  RSR: {rsr_df.shape[1]}銘柄 × {rsr_df.shape[0]}日")

    # ---- 4. Composite Alpha 計算 ----
    print(f"[4/6] Composite Alpha計算中（window={COMP_ALPHA_WINDOW}日）...")
    trade_prices = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in trade_syms if sym in universe_raw
    }
    alpha_df = calc_composite_alpha_matrix(trade_prices, window=COMP_ALPHA_WINDOW)
    # 先読み防止: shift(1) して前日データを当日判定に使用
    alpha_df = alpha_df.shift(1)
    print(f"  Alpha matrix: {alpha_df.shape[1]}銘柄 × {alpha_df.shape[0]}日")

    # ---- レジームフィルター計算 ----
    regime_df = _calc_regime(topix_close)
    risk_off_days = int(regime_df["risk_off"].sum())
    crash_days    = int(regime_df["crash"].sum())
    total_days    = len(regime_df)
    print(f"  レジーム: risk_off={risk_off_days}日({risk_off_days/total_days:.0%}), "
          f"crash={crash_days}日({crash_days/total_days:.0%})")

    # ---- 5. STEP4/5/6用テクニカル指標を事前計算 ----
    print("[5/6] テクニカル指標事前計算中（STEP4/5/6用）...")
    tech_matrices = _precompute_tech_matrices(universe_raw, list(trade_syms.keys()))
    breadth_series = _calc_breadth(rsr_df)
    print(f"  ATR20: {tech_matrices['atr20'].shape[1]}銘柄  "
          f"High200: {tech_matrices['high200'].shape[1]}銘柄  "
          f"Breadth中央値: {breadth_series.median():.2f}")

    # ---- 6. 7シナリオ実行 ----
    print("[6/6] 7シナリオ実行中...\n")

    # STEP6バリアント: breadth閾値の感度テスト
    _breadth_params = {
        "STEP6":  (BREADTH_STOP,  BREADTH_REDUCE),   # 0.25 / 0.15（デフォルト）
        "STEP6A": (0.30,          0.20),              # 0.30 / 0.20
        "STEP6B": (0.35,          0.25),              # 0.35 / 0.25
    }

    results = {}
    for scenario in ("BASELINE", "STEP1", "STEP2", "STEP3", "STEP4", "STEP5",
                      "STEP6", "STEP6A", "STEP6B"):
        print(f"  ─── {scenario} ───────────────────────────")
        b_stop, b_reduce = _breadth_params.get(scenario, (BREADTH_STOP, BREADTH_REDUCE))
        res = run_scenario(
            scenario       = scenario,
            universe_raw   = universe_raw,
            rsr_df         = rsr_df,
            alpha_df       = alpha_df,
            regime_df      = regime_df,
            trade_syms     = trade_syms,
            rsr_syms       = rsr_syms,
            start          = args.start,
            end            = args.end,
            verbose        = False,
            tech_matrices  = tech_matrices,
            breadth_series = breadth_series,
            breadth_stop   = b_stop,
            breadth_reduce = b_reduce,
            min_hold       = 3,   # 2026-03-31: min_hold感度テストで採用
        )
        results[scenario] = res

        print(f"    CAGR={res['cagr']:+.1f}%  Sharpe={res['sharpe']:.3f}  "
              f"MaxDD={res['max_dd']:.1f}%  Calmar={res['calmar']:.3f}")
        print(f"    取引={res['n_trades']}回({res['n_trades_yr']:.0f}/年)  "
              f"勝率={res['win_rate']:.1f}%  R倍率={res['r_multiple']:.2f}x")
        print(f"    avgExp={res['avg_exposure']:.1f}%  "
              f"avgWin={res['avg_win_pct']:+.2f}%  avgLose={res['avg_lose_pct']:+.2f}%")

    # ---- 集計表示 ----
    all_scenarios = ("BASELINE", "STEP1", "STEP2", "STEP3", "STEP4", "STEP5",
                      "STEP6", "STEP6A", "STEP6B")
    col_w = 9

    print("\n" + "=" * 80)
    print("  改善ステップ比較サマリー（7シナリオ）")
    print("=" * 80)
    hdr = f"  {'指標':<20}" + "".join(f" {sc:>{col_w}}" for sc in all_scenarios)
    print(hdr)
    print(f"  {'-'*20}" + f" {'-'*col_w}" * len(all_scenarios))

    metrics = [
        ("CAGR (%)",   "cagr",         "{:+.1f}"),
        ("Sharpe",     "sharpe",        "{:.3f}"),
        ("MaxDD (%)",  "max_dd",        "{:.1f}"),
        ("Calmar",     "calmar",        "{:.3f}"),
        ("勝率 (%)",   "win_rate",      "{:.1f}"),
        ("R倍率",      "r_multiple",    "{:.2f}"),
        ("avgExp (%)", "avg_exposure",  "{:.1f}"),
        ("取引数/年",  "n_trades_yr",   "{:.0f}"),
        ("avgHoldings", "avg_simultaneous_holdings", "{:.2f}"),
    ]
    for label, key, fmt in metrics:
        row = f"  {label:<20}"
        for sc in all_scenarios:
            v = results[sc].get(key, 0)
            row += f" {fmt.format(v):>{col_w}}"
        print(row)

    print("\n  ── 年次リターン ───────────────────────────────────────────────────")
    all_years = sorted(set(yr for r in results.values()
                           for yr in r.get("annual_returns", {})))
    print(f"  {'年':<6}" + "".join(f" {sc:>{col_w}}" for sc in all_scenarios))
    for yr in all_years:
        row = f"  {yr:<6}"
        for sc in all_scenarios:
            v = results[sc].get("annual_returns", {}).get(str(yr), 0.0)
            row += f" {v:>+8.1f}%"
        print(row)

    # ---- 判定 ----
    print("\n  ── 判定 ──────────────────────────────────────────────────────────")
    baseline_sharpe = results["BASELINE"]["sharpe"]
    for sc in ("STEP1", "STEP2", "STEP3", "STEP4", "STEP5", "STEP6", "STEP6A", "STEP6B"):
        diff_sharpe = results[sc]["sharpe"] - baseline_sharpe
        diff_cagr   = results[sc]["cagr"]   - results["BASELINE"]["cagr"]
        diff_dd     = results[sc]["max_dd"]  - results["BASELINE"]["max_dd"]
        verdict = "✅ 採用推奨" if diff_sharpe > 0.05 and results[sc]["max_dd"] > -20 else \
                  "△ 条件付き" if diff_sharpe > 0 else "❌ 逆効果"
        print(f"  {sc}: CAGR{diff_cagr:+.1f}pp  Sharpe{diff_sharpe:+.3f}  "
              f"MaxDD{diff_dd:+.1f}pp  → {verdict}")

    # ---- 保存 ----
    mode_tag  = "rsr42" if args.rsr42_trade else "temporal24"
    save_path = OUTPUT_DIR / f"composite_alpha_bt_{mode_tag}_{date_type.today()}.json"
    save_data = {}
    for sc, res in results.items():
        d = {k: v for k, v in res.items() if k != "equity_curve"}
        save_data[sc] = d
    save_path.write_text(json.dumps(save_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  結果保存: {save_path}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
