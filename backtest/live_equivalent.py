"""
backtest/live_equivalent.py

ライブアーキテクチャ完全再現バックテスト（RSR統一42銘柄コンテキスト版）

research / live / backtest の3つを統一した設定:
  RSR context  = 42銘柄（configs/rsr_universe_42.csv）
  min_rsr      = 75.0（filter-first / research 確定値）
  architecture = filter → rank → top_k（select_top_k は最後）

ライブ signal_bridge.py との対応:
  RSR ≥ min_rsr → FujikoStrategy → BUY候補をRSR降順 → top max_positions

確定パラメータ:
  universe          = TEMPORAL24 (configs/universe/2026Q1_temporal24.json)
  max_positions     = 3  (research 確定値)
  max_hold_days     = 60
  max_single_weight = 0.25
  max_new_positions_per_day = 2
  min_sepa = 6, min_rsr = 75.0
  reentry_cooldown  = 5 営業日

実行:
  python -m backtest.live_equivalent
  python -m backtest.live_equivalent --start 2018-01-01 --end 2024-12-31
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.universe_builder        import download_universe

# ------------------------------------------------------------------ #
# 定数（research 確定値）
# ------------------------------------------------------------------ #
UNIVERSE_JSON     = Path("configs/universe/2026Q1_temporal24.json")
RSR_UNIVERSE_CSV  = Path("configs/rsr_universe_42.csv")
ROLLING_JSON      = Path("configs/rolling_universe.json")
START             = "2018-01-01"
END               = "2024-12-31"
CAPITAL           = 2_000_000
MAX_POSITIONS     = 3        # research 確定値
MAX_HOLD_DAYS     = 60
MAX_SINGLE_WEIGHT = 0.25
MAX_NEW_PER_DAY   = 2
MIN_SEPA          = 6
MIN_RSR           = 75.0     # filter-first 閾値（research 確定値）
MAX_DD_LIMIT      = 0.15
REENTRY_COOLDOWN  = 5
LOT               = 100

SLIPPAGE     = 0.001
COMMISSION   = 0.00055
COST_ONE_WAY = SLIPPAGE + COMMISSION

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


# ------------------------------------------------------------------ #
# ユーティリティ
# ------------------------------------------------------------------ #
@dataclass
class Position:
    symbol:      str
    sector:      str
    qty:         int
    entry_price: float
    entry_idx:   int
    entry_atr:   float = 0.0   # ATR20 at entry（False Breakout計測用）


def _load_temporal24() -> dict[str, str]:
    data = json.loads(UNIVERSE_JSON.read_text(encoding="utf-8"))
    return data["symbols"]


def _load_rsr_universe_42() -> dict[str, str]:
    df = pd.read_csv(RSR_UNIVERSE_CSV)
    return {row["symbol"]: row.get("sector", "不明") for _, row in df.iterrows()}


def _load_rolling_universe() -> dict[int, dict[str, str]]:
    """configs/rolling_universe.json から {year: {sym: sector}} を返す。"""
    if not ROLLING_JSON.exists():
        raise FileNotFoundError(
            f"{ROLLING_JSON} が見つかりません。先に rolling_universe.py を実行してください。"
        )
    raw = json.loads(ROLLING_JSON.read_text(encoding="utf-8"))
    return {int(yr): d for yr, d in raw.items()}


# ------------------------------------------------------------------ #
# バックテスト本体
# ------------------------------------------------------------------ #
def run_backtest(
    start:                    str   = START,
    end:                      str   = END,
    capital:                  float = CAPITAL,
    min_rsr:                  float = MIN_RSR,
    max_positions:            int   = MAX_POSITIONS,
    max_hold_days:            int   = MAX_HOLD_DAYS,
    max_single_weight:        float = MAX_SINGLE_WEIGHT,
    max_new_positions_per_day: int  = MAX_NEW_PER_DAY,
    use_rolling:              bool  = False,   # True: ローリング選定ユニバースを使用
    use_broad_rsr:            bool  = False,   # True: RSR context = TOPIX100全件(~76), False: 年別選定銘柄
    trade_universe:           dict[str, str] | None = None,  # 取引対象ユニバースを外部から指定
    turtle_entry:             int   = 20,      # Turtle エントリー期間（デフォルト=確定値）
    weekly_trend_filter:      str | None = None,  # MTFフィルター: "ma20" | "cross" | None
    verbose:                  bool  = True,
) -> dict:
    """
    filter-first アーキテクチャでバックテストを実行する。

    use_rolling=False: TEMPORAL24 固定ユニバース（デフォルト）
    use_rolling=True : ローリング選定ユニバース（look-ahead bias 除去版）

    RSR コンテキスト設計:
      use_broad_rsr=False（デフォルト）:
        ローリングモード: 各年の選定銘柄のみでRSRパーセンタイルを計算
        → context サイズ 9〜30（分布が薄く、min_rsr=75の閾値が実質的に緩くなる）
      use_broad_rsr=True（推奨実験）:
        Ranking universe = TOPIX100全件（~76銘柄）でRSR計算
        Trading universe = ローリング選定銘柄（20〜30銘柄）
        → cross-sectional alpha が強化される（プロ設計）

    Returns: 性能指標の辞書
    """
    # ---- 1. ユニバース読み込み ----
    if use_rolling:
        rolling_univs = _load_rolling_universe()
        # 全年のユニークな銘柄を集めてデータ取得
        all_rolling_syms: dict[str, str] = {}
        for d in rolling_univs.values():
            all_rolling_syms.update(d)
        trade_syms_fixed = None   # ローリングモードでは年別に決まる
        if verbose:
            print(f"[1/4] ローリングユニバース: {len(FOLDS) if 'FOLDS' in dir() else 7}フォールド")
            for yr, d in rolling_univs.items():
                print(f"  {yr}: {len(d)}銘柄")
    else:
        if trade_universe is not None:
            temporal24 = trade_universe
        else:
            temporal24 = _load_temporal24()
        trade_syms_fixed = list(temporal24.keys())
        all_rolling_syms = temporal24
        rolling_univs    = None
        if verbose:
            print(f"[1/4] 取引ユニバース: {len(trade_syms_fixed)}銘柄")

    rsr_univ_42 = _load_rsr_universe_42()

    # ---- 2. データ取得 ----
    all_tickers = {**rsr_univ_42, **all_rolling_syms}
    if verbose:
        print(f"[2/4] データ取得中（{len(all_tickers)}銘柄）...")
    universe_raw = download_universe(all_tickers, start=start, end=end, verbose=False)
    if verbose:
        print(f"  取得完了: {len(universe_raw)} 銘柄")

    # ---- 3. RSR 計算 ----
    if use_rolling:
        # ローリングモード: 年別 RSR
        if verbose:
            ctx_label = "TOPIX100全件(broad)" if use_broad_rsr else "年別選定銘柄(narrow)"
            print(f"[3/4] RSR 計算（{ctx_label}コンテキスト）...")
        rsr_by_year: dict[int, pd.DataFrame] = {}

        if use_broad_rsr:
            # Ranking universe = 全ダウンロード済み銘柄（~76）でRSR計算
            # → 各年共通の広いコンテキストで RSR パーセンタイルを計算
            broad_prices = {
                sym: universe_raw[sym]["df"]["Close"]
                for sym in universe_raw
            }
            rsr_broad = calc_universe_rsr(broad_prices)
            for yr in rolling_univs:
                rsr_by_year[yr] = rsr_broad   # 全年同じ広いコンテキストを使う
            if verbose:
                print(f"  全年共通: RSR context={rsr_broad.shape[1]}銘柄（TOPIX100 broad）")
        else:
            # Ranking universe = 各年の選定銘柄のみ（従来動作）
            for yr, yr_syms in rolling_univs.items():
                yr_prices = {
                    sym: universe_raw[sym]["df"]["Close"]
                    for sym in yr_syms
                    if sym in universe_raw
                }
                if yr_prices:
                    rsr_by_year[yr] = calc_universe_rsr(yr_prices)
            if verbose:
                for yr, df in rsr_by_year.items():
                    print(f"  {yr}: RSR context={df.shape[1]}銘柄")
    else:
        # 固定モード: 42銘柄コンテキスト
        if verbose:
            print("[3/4] RSR 計算（42銘柄コンテキスト）...")
        rsr42_prices = {
            sym: universe_raw[sym]["df"]["Close"]
            for sym in rsr_univ_42
            if sym in universe_raw
        }
        rsr_df = calc_universe_rsr(rsr42_prices)
        rsr_by_year = None
        if verbose:
            print(f"  RSR context: {rsr_df.shape[1]}銘柄 × {rsr_df.shape[0]}日")

    # ---- 3b. 週足トレンド事前計算（MTFフィルター） ----
    weekly_trend_map: dict[str, pd.Series] = {}
    if weekly_trend_filter:
        trade_syms_for_mtf = list(all_rolling_syms.keys())
        if verbose:
            print(f"[MTF] 週足トレンド計算中（filter={weekly_trend_filter}, {len(trade_syms_for_mtf)}銘柄）...")
        for sym in trade_syms_for_mtf:
            if sym not in universe_raw:
                continue
            daily_close  = universe_raw[sym]["df"]["Close"]
            weekly_close = daily_close.resample("W-FRI").last().dropna()
            try:
                if weekly_trend_filter == "ma20":
                    wma20 = weekly_close.rolling(20).mean()
                    signal = (weekly_close > wma20)
                elif weekly_trend_filter == "cross":
                    wma10 = weekly_close.rolling(10).mean()
                    wma30 = weekly_close.rolling(30).mean()
                    signal = (wma10 > wma30)
                else:
                    continue
                # 週足シグナルを日足インデックスに前向き補完
                weekly_trend_map[sym] = signal.reindex(daily_close.index, method="ffill").fillna(False)
            except Exception:
                pass

    # ---- 4. 戦略初期化（固定モードのみ事前生成） ----
    def _make_strats(syms: dict[str, str], rsr_source: pd.DataFrame) -> dict:
        strats = {}
        for sym, sector in syms.items():
            if sym not in universe_raw:
                continue
            rsr_s = rsr_source[sym] if rsr_source is not None and sym in rsr_source.columns else None
            rule  = SECTOR_STRATEGY.get(sector, "fujiko")
            if rule == "mean_rev":
                strats[sym] = MeanReversionStrategy(**MR_PARAMS)
            else:
                strats[sym] = FujikoStrategy(
                    rsr_series       = rsr_s,
                    min_sepa         = MIN_SEPA,
                    min_rsr          = min_rsr,
                    mom_period       = 21,
                    turtle_entry     = turtle_entry,
                    turtle_exit      = 20,
                    use_turtle_entry = True,
                )
        return strats

    if not use_rolling:
        strats_fixed = _make_strats(temporal24, rsr_df)
        if verbose:
            print(f"[4/4] 戦略初期化完了: {len(strats_fixed)}銘柄")
    else:
        strats_fixed = None
        if verbose:
            print("[4/4] 戦略初期化: ローリングモード（年別動的生成）")

    # ---- 5. 共通取引日 ----
    ref_syms = list(all_rolling_syms.keys()) if use_rolling else trade_syms_fixed
    common_dates: pd.DatetimeIndex = None
    for sym in ref_syms:
        if sym not in universe_raw:
            continue
        idx = universe_raw[sym]["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    ts_f = pd.Timestamp(start)
    te_f = pd.Timestamp(end)
    common_dates = common_dates[(common_dates >= ts_f) & (common_dates <= te_f)]

    # ---- 6. ポートフォリオ ループ ----
    cash           = float(capital)
    positions:     dict[str, Position] = {}
    equity_curve:  list[float]         = []
    exposure_list: list[float]         = []
    pos_list:      list[float]         = []
    cand_list:     list[int]           = []   # 1日のBUY候補数
    hhi_list:      list[float]         = []   # Herfindahl 集中度
    trades:        list[dict]          = []
    cb_active      = False
    peak_equity    = float(capital)
    reentry_ban:   dict[str, int]      = {}
    false_breakout_count = 0   # MTF効果測定用

    current_year   = -1
    active_syms:    dict[str, str]    = {}
    active_strats:  dict[str, object] = {}
    active_rsr_df:  pd.DataFrame | None = None

    for i, date in enumerate(common_dates):

        # ── ローリングモード: 年が変わったら universe / RSR / strats を切り替え ──
        yr = date.year
        if use_rolling and yr != current_year:
            current_year = yr
            active_syms   = rolling_univs.get(yr, {})
            active_rsr_df = rsr_by_year.get(yr)
            active_strats = _make_strats(active_syms, active_rsr_df)
        elif not use_rolling and current_year == -1:
            current_year  = yr
            active_syms   = temporal24
            active_strats = strats_fixed
            active_rsr_df = rsr_df

        # ── RSR 取得 ────────────────────────────────────────────────
        rsr_row = (
            active_rsr_df.loc[date]
            if active_rsr_df is not None and date in active_rsr_df.index
            else pd.Series(dtype=float)
        )

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
        cb_active = dd < -MAX_DD_LIMIT

        # HHI（保有時のみ計算）
        if positions:
            weights = [
                pos.qty * float(universe_raw[sym]["df"].loc[date, "Close"])
                / max(1.0, cur_equity)
                for sym, pos in positions.items()
                if sym in universe_raw and date in universe_raw[sym]["df"].index
            ]
            hhi_list.append(sum(w * w for w in weights))
        else:
            hhi_list.append(0.0)

        # ── シグナル生成（filter-first） ─────────────────────────────
        sell_signals: list[tuple]  = []   # (sym, reason)
        buy_candidates: list[tuple] = []  # (rsr_val, sym)

        for sym in active_syms:
            if sym not in universe_raw or sym not in active_strats:
                continue
            df_sym = universe_raw[sym]["df"]
            if date not in df_sym.index:
                continue

            is_holding = sym in positions
            hold_idx   = (i - positions[sym].entry_idx) if is_holding else 0
            rsr_val    = float(rsr_row.get(sym, 0.0)) if sym in rsr_row.index else 0.0

            # False Breakout診断（entry後5日以内 かつ -2ATR到達）
            if is_holding and hold_idx <= 5 and positions[sym].entry_atr > 0:
                try:
                    _c = float(df_sym.loc[date, "Close"])
                    if _c < positions[sym].entry_price - 2.0 * positions[sym].entry_atr:
                        false_breakout_count += 1
                except Exception:
                    pass

            # 時間ストップ
            if is_holding and hold_idx > max_hold_days:
                sell_signals.append((sym, "TIME_STOP"))
                continue

            # RSR 低下エグジット（filter-first の rank_exit 相当）
            if is_holding and rsr_val < min_rsr:
                sell_signals.append((sym, "RSR_EXIT"))
                continue

            # 戦略シグナル（FujikoStrategy が RSR ≥ min_rsr を内部チェック）
            past = df_sym.loc[:date]
            if len(past) < 30:
                continue
            sig = active_strats[sym].generate_signal(past)

            if sig == -1 and is_holding:
                sell_signals.append((sym, "STRATEGY_EXIT"))
            elif sig == 1 and not is_holding:
                ban_until = reentry_ban.get(sym, -1)
                if i >= ban_until:
                    # MTFフィルター: weekly trend が上向きのときのみ BUY
                    if weekly_trend_filter and sym in weekly_trend_map:
                        wt = weekly_trend_map[sym]
                        if date in wt.index and not bool(wt.loc[date]):
                            continue   # weekly trend ≠ up → BUY スキップ
                    buy_candidates.append((rsr_val, sym))

        cand_list.append(len(buy_candidates))

        # ── 翌日 ────────────────────────────────────────────────────
        if i + 1 >= len(common_dates):
            break
        next_date = common_dates[i + 1]

        # ── SELL 実行 ────────────────────────────────────────────────
        for sym, reason in sell_signals:
            if sym not in positions:
                continue
            pos = positions[sym]
            df_sym = universe_raw[sym]["df"]
            if next_date not in df_sym.index:
                continue
            sell_px = float(df_sym.loc[next_date, "Open"])
            proceeds = pos.qty * sell_px * (1 - COST_ONE_WAY)
            pnl      = (sell_px - pos.entry_price) * pos.qty
            cash    += proceeds
            trades.append({"symbol": sym, "side": "SELL",
                           "entry": pos.entry_price, "exit": sell_px,
                           "qty": pos.qty, "pnl": pnl, "reason": reason})
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOLDOWN

        # ── BUY 実行（RSR降順 → top max_positions） ─────────────────
        if not cb_active and buy_candidates:
            # eligible.sort() → eligible[:max_positions]（research 仕様）
            buy_candidates.sort(key=lambda x: -x[0])
            new_buys = 0
            for rsr_val, sym in buy_candidates:
                if new_buys >= max_new_positions_per_day:
                    break
                open_slots = max_positions - len(positions)
                if open_slots <= 0:
                    break
                df_sym = universe_raw[sym]["df"]
                if next_date not in df_sym.index:
                    continue
                buy_px = float(df_sym.loc[next_date, "Open"])
                if buy_px <= 0:
                    continue

                n_remaining     = sum(1 for _, s in buy_candidates if s not in positions)
                effective_slots = min(open_slots, max(1, n_remaining))
                alloc           = cash / effective_slots
                alloc           = min(alloc, capital * max_single_weight)

                qty = int(alloc / buy_px / LOT) * LOT
                if qty <= 0:
                    continue

                # ATR20 at entry（True Range平均）
                _atr20_bt = 0.0
                try:
                    _df_b = df_sym.loc[:date]
                    _tr = pd.concat([
                        _df_b["High"] - _df_b["Low"],
                        (_df_b["High"] - _df_b["Close"].shift()).abs(),
                        (_df_b["Low"]  - _df_b["Close"].shift()).abs(),
                    ], axis=1).max(axis=1)
                    _v = float(_tr.rolling(20).mean().iloc[-1])
                    if not np.isnan(_v):
                        _atr20_bt = _v
                except Exception:
                    pass

                cash -= qty * buy_px * (1 + COST_ONE_WAY)
                positions[sym] = Position(
                    symbol=sym, sector=active_syms.get(sym, "不明"),
                    qty=qty, entry_price=buy_px, entry_idx=i + 1,
                    entry_atr=_atr20_bt,
                )
                trades.append({"symbol": sym, "side": "BUY",
                               "entry": buy_px, "exit": None,
                               "qty": qty, "pnl": None, "reason": f"RSR={rsr_val:.1f}"})
                new_buys += 1

    # ---- 7. 指標計算 ----
    n_days = len(equity_curve)
    eq     = pd.Series(equity_curve, index=common_dates[:n_days])
    years  = n_days / 252
    cagr   = (eq.iloc[-1] / capital) ** (1 / max(years, 0.01)) - 1

    dr     = eq.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * np.sqrt(252)) if dr.std() > 0 else 0.0

    roll_max = eq.expanding().max()
    dd_ser   = (eq - roll_max) / roll_max
    max_dd   = float(dd_ser.min())
    calmar   = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    avg_exposure   = float(np.mean(exposure_list))
    avg_positions  = float(np.mean(pos_list))
    avg_candidates = float(np.mean(cand_list))
    zero_exp_rate  = sum(1 for e in exposure_list if e == 0) / max(1, len(exposure_list)) * 100

    # HHI（保有日のみ）
    hhi_invested   = [h for h in hhi_list if h > 0]
    avg_hhi        = float(np.mean(hhi_invested)) if hhi_invested else 0.0
    max_hhi        = float(np.max(hhi_invested))  if hhi_invested else 0.0

    sells      = [t for t in trades if t["side"] == "SELL" and t["pnl"] is not None]
    win_rate   = sum(1 for t in sells if t["pnl"] > 0) / max(1, len(sells))
    false_bo_rate = false_breakout_count / max(1, len(sells))

    return {
        "cagr":                cagr,
        "sharpe":              sharpe,
        "max_dd":              max_dd,
        "calmar":              calmar,
        "avg_exposure":        avg_exposure,
        "avg_positions":       avg_positions,
        "avg_candidates":      avg_candidates,
        "zero_exp_rate":       zero_exp_rate,
        "trade_count":         len(sells),
        "win_rate":            win_rate,
        "avg_hhi":             avg_hhi,
        "max_hhi":             max_hhi,
        "false_breakout_rate": false_bo_rate,
        "equity_curve":        eq,
    }


# ------------------------------------------------------------------ #
# 判定
# ------------------------------------------------------------------ #
def _judge(m: dict) -> str:
    e = m["avg_exposure"]
    s = m["sharpe"]
    c = m["avg_candidates"]
    lines = []
    lines.append(f"  avg_exposure  {e:.1%}  {'✅ 正常(25~35%)' if 0.25<=e<=0.35 else '⚠️ '+('低すぎ' if e<0.15 else '高すぎ' if e>0.45 else '許容範囲')}")
    lines.append(f"  avg_candidates {c:.2f}/日  {'✅ 研究値≈0.4' if 0.3<=c<=0.6 else '⚠️ '+('低い' if c<0.3 else '高い')}")
    lines.append(f"  zero_exp      {m['zero_exp_rate']:.1f}%  {'✅ 研究値~19%' if 15<=m['zero_exp_rate']<=25 else '⚠️ 差異あり'}")
    lines.append(f"  Sharpe        {s:.3f}  {'✅ >1.0' if s>1.0 else '⚠️ 目標未達'}")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="ライブ等価バックテスト")
    p.add_argument("--start",   default=START)
    p.add_argument("--end",     default=END)
    p.add_argument("--min-rsr", type=float, default=MIN_RSR)
    p.add_argument("--rolling", action="store_true",
                   help="ローリング選定ユニバースを使用（look-ahead bias 除去）")
    p.add_argument("--broad-rsr", action="store_true",
                   help="RSR context を TOPIX100全件(~76銘柄)に拡大（ranking/trading 分離）")
    args = p.parse_args()

    mode = "ローリング選定" if args.rolling else "TEMPORAL24 固定"
    if args.rolling and args.broad_rsr:
        mode += " / broad RSR(~76)"
    print("=" * 64)
    print(f"  ライブ等価バックテスト（{mode} / filter-first）")
    print(f"  期間: {args.start} 〜 {args.end}  /  初期資本: ¥{CAPITAL:,}")
    print(f"  min_rsr={args.min_rsr}  max_pos={MAX_POSITIONS}  max_hold={MAX_HOLD_DAYS}d")
    print("=" * 64)

    m = run_backtest(start=args.start, end=args.end,
                     min_rsr=args.min_rsr, use_rolling=args.rolling,
                     use_broad_rsr=args.broad_rsr, verbose=True)

    print(f"\n  CAGR      {m['cagr']:>+8.2%}")
    print(f"  Sharpe    {m['sharpe']:>8.3f}")
    print(f"  MaxDD     {m['max_dd']:>8.2%}")
    print(f"  Calmar    {m['calmar']:>8.3f}")
    print(f"  取引数    {m['trade_count']:>8d}")
    print(f"  勝率      {m['win_rate']:>8.1%}")
    print(f"\n  ── 目標指標チェック ──")
    print(_judge(m))
    print("\n" + "=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
