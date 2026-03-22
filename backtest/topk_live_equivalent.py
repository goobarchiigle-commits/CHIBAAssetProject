"""
backtest/topk_live_equivalent.py

ライブアーキテクチャ完全再現バックテスト

ライブ signal_bridge.py との対応:
  select_top_k()          → _select_top_k()          （RSR上位k選出）
  _generate_all_signals() → シグナルループ           （セクター別戦略適用）
  _build_orders()         → 注文生成ロジック         （ポジションサイジング）

確定パラメータ（run_live_signal.py と一致）:
  universe          = TEMPORAL24 (configs/universe/2026Q1_temporal24.json)
  top_k             = 4 (デフォルト / sweep 対象)
  max_positions     = top_k と同値
  max_hold_days     = 60
  max_single_weight = 0.25 → 1銘柄最大 50 万円
  max_new_positions_per_day = 2
  min_sepa          = 6, min_rsr = 0.0 (top_k で代替)
  reentry_cooldown  = 5 営業日

RSR コンテキスト:
  TEMPORAL24 銘柄内でパーセンタイルを計算。
  ライブ（TOPIX100 コンテキスト）との差異: ランキング順位は十分近似。
  sweep 比較（k=3/4/5 の相対比較）には有効。

実行:
  python -m backtest.topk_live_equivalent
  python -m backtest.topk_live_equivalent --k 3
  python -m backtest.topk_live_equivalent --k 3 4 5   # sweep
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from backtest.rsr                     import calc_universe_rsr
from backtest.fujiko_strategy         import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.engine                  import TradeCost
from backtest.universe_builder        import download_universe

# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
UNIVERSE_JSON     = Path("configs/universe/2026Q1_temporal24.json")
START             = "2018-01-01"
END               = "2024-12-31"
CAPITAL           = 2_000_000
MAX_POSITIONS     = 4        # top_k と一致させる
MAX_HOLD_DAYS     = 60       # 時間ストップ（営業日）
MAX_SINGLE_WEIGHT = 0.25     # 1銘柄最大25%
MAX_NEW_PER_DAY   = 2        # 1日の新規BUY上限
MIN_SEPA          = 6
MAX_DD_LIMIT      = 0.15     # サーキットブレーカー
REENTRY_COOLDOWN  = 5        # 時間ストップ後の再エントリー禁止（営業日）
LOT               = 100      # 単元株数

SLIPPAGE    = 0.001          # 0.1% 片道
COMMISSION  = 0.00055        # 0.055% 片道
COST_ONE_WAY = SLIPPAGE + COMMISSION  # 0.00155

# セクター → 戦略マッピング（signal_bridge.py / portfolio_v2.py と同一）
SECTOR_STRATEGY: dict[str, str] = {
    "海運":     "fujiko", "機械":     "fujiko", "電機精密": "fujiko",
    "商社":     "fujiko", "電機":     "fujiko", "ゲーム":   "fujiko",
    "レジャー": "fujiko", "食品":     "fujiko",
    "ガス":     "mean_rev","鉄鋼":    "mean_rev","銀行":     "mean_rev",
    "保険":     "mean_rev","輸送機器":"mean_rev","化学":     "mean_rev",
    "小売":     "mean_rev",
    # dynamic → ライブと同じくフジコ法を適用
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
    symbol:     str
    sector:     str
    qty:        int
    entry_price: float
    entry_idx:  int   # 取引日インデックス（hold_days 計算用）


def _select_top_k(rsr_series: pd.Series, k: int) -> list[str]:
    """RSR 上位 k 銘柄を返す（signal_bridge.select_top_k と同等）。"""
    valid = rsr_series.dropna()
    if valid.empty:
        return []
    return valid.sort_values(ascending=False).head(k).index.tolist()


def _load_temporal24() -> dict[str, str]:
    """TEMPORAL24 ユニバースを JSON から読み込む。{symbol: sector} を返す。"""
    if not UNIVERSE_JSON.exists():
        raise FileNotFoundError(
            f"{UNIVERSE_JSON} が見つかりません。プロジェクトルートから実行してください。"
        )
    data = json.loads(UNIVERSE_JSON.read_text(encoding="utf-8"))
    return data["symbols"]


# ------------------------------------------------------------------ #
# メイン バックテスト関数
# ------------------------------------------------------------------ #
def run_backtest(
    top_k:                    int   = 4,
    start:                    str   = START,
    end:                      str   = END,
    capital:                  float = CAPITAL,
    max_hold_days:            int   = MAX_HOLD_DAYS,
    max_single_weight:        float = MAX_SINGLE_WEIGHT,
    max_new_positions_per_day: int  = MAX_NEW_PER_DAY,
    min_sepa:                 int   = MIN_SEPA,
    verbose:                  bool  = True,
) -> dict:
    """
    top_k パラメータでライブ等価バックテストを実行し、性能指標を返す。

    Returns:
        {
          "top_k": int,
          "cagr": float, "sharpe": float, "max_dd": float, "calmar": float,
          "avg_exposure": float, "avg_positions": float,
          "trade_count": int, "win_rate": float,
          "equity_curve": pd.Series,
        }
    """
    max_positions = top_k  # ライブと同じく max_pos = top_k

    # ---- 1. ユニバース読み込み ----
    if verbose:
        print(f"\n[1/4] TEMPORAL24 ユニバース読み込み...")
    temporal24 = _load_temporal24()
    temporal24_syms = list(temporal24.keys())
    if verbose:
        print(f"  {len(temporal24_syms)} 銘柄: {temporal24_syms[:5]} ...")

    # ---- 2. データ取得 ----
    if verbose:
        print(f"[2/4] データ取得中（{len(temporal24_syms)}銘柄 / {start}〜{end}）...")
    universe_raw = download_universe(temporal24, start=start, end=end, verbose=False)
    if verbose:
        print(f"  取得完了: {len(universe_raw)} 銘柄")

    # ---- 3. RSR 計算（TEMPORAL24 コンテキスト）----
    if verbose:
        print("[3/4] RSR 計算中...")
    prices_all = {sym: info["df"]["Close"] for sym, info in universe_raw.items()}
    rsr_df     = calc_universe_rsr(prices_all)   # shape: (dates, symbols)
    if verbose:
        print(f"  RSR 計算完了: {rsr_df.shape}")

    # ---- 4. 戦略インスタンス生成 ----
    if verbose:
        print(f"[4/4] 戦略初期化 → top_k={top_k} バックテスト開始...")
    strats: dict[str, object] = {}
    for sym, sector in temporal24.items():
        if sym not in universe_raw:
            continue
        rsr_s = rsr_df[sym] if sym in rsr_df.columns else None
        rule  = SECTOR_STRATEGY.get(sector, "fujiko")
        if rule == "mean_rev":
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:  # fujiko or dynamic
            strats[sym] = FujikoStrategy(
                rsr_series       = rsr_s,
                min_sepa         = min_sepa,
                min_rsr          = 0.0,    # top_k で代替（min_rsr 閾値は無効化）
                mom_period       = 21,
                turtle_entry     = 20,
                turtle_exit      = 20,
                use_turtle_entry = True,
            )

    # ---- 5. 共通取引日 ----
    common_dates: pd.DatetimeIndex = None
    for sym, info in universe_raw.items():
        idx = info["df"].index
        common_dates = idx if common_dates is None else common_dates.intersection(idx)
    common_dates = common_dates.sort_values()

    # 期間フィルター
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)
    common_dates = common_dates[(common_dates >= ts_start) & (common_dates <= ts_end)]

    # ---- 6. ポートフォリオ ループ ----
    cash          = float(capital)
    positions:    dict[str, Position] = {}
    equity_curve: list[float]         = []
    exposure_list: list[float]        = []
    pos_count_list: list[float]       = []

    trades:       list[dict]          = []   # 全売買履歴
    cb_active     = False
    peak_equity   = float(capital)
    reentry_ban:  dict[str, int] = {}        # symbol → ban_until_idx

    rsr_syms = set(rsr_df.columns)

    for i, date in enumerate(common_dates):

        # ── RSR top-k 選出 ──────────────────────────────────────────
        if date in rsr_df.index:
            rsr_row   = rsr_df.loc[date]
            rsr_t24   = rsr_row.reindex(temporal24_syms).dropna()
            top_k_set = set(_select_top_k(rsr_t24, k=top_k))
        else:
            top_k_set = set()

        # ── 現在 equity と DD 計算 ──────────────────────────────────
        invested = sum(
            pos.qty * float(universe_raw[sym]["df"].loc[date, "Close"])
            for sym, pos in positions.items()
            if sym in universe_raw and date in universe_raw[sym]["df"].index
        )
        current_equity = cash + invested
        equity_curve.append(current_equity)
        exposure_list.append(invested / max(1.0, current_equity))
        pos_count_list.append(len(positions))

        # CB 状態更新
        if current_equity > peak_equity:
            peak_equity = current_equity
        dd = (current_equity - peak_equity) / peak_equity
        cb_active = dd < -MAX_DD_LIMIT

        # ── シグナル生成 ─────────────────────────────────────────────
        sell_signals: list[str]        = []
        buy_signals:  list[tuple]      = []  # (rsr_val, sym)

        for sym in temporal24_syms:
            if sym not in universe_raw or sym not in strats:
                continue
            df_sym = universe_raw[sym]["df"]
            if date not in df_sym.index:
                continue

            is_holding = sym in positions
            hold_idx   = (i - positions[sym].entry_idx) if is_holding else 0
            in_top_k   = sym in top_k_set

            # 時間ストップ（max_hold_days 超過）
            if is_holding and hold_idx > max_hold_days:
                sell_signals.append((sym, "TIME_STOP"))
                continue

            # ランク圏外エグジット（top_k 外に落ちた保有銘柄）
            if is_holding and not in_top_k:
                sell_signals.append((sym, "RANK_EXIT"))
                continue

            # top_k 内: 個別戦略シグナル
            if in_top_k:
                past_data = df_sym.loc[:date]
                if len(past_data) < 30:    # ウォームアップ未満はスキップ
                    continue
                sig = strats[sym].generate_signal(past_data)

                if sig == -1 and is_holding:
                    sell_signals.append((sym, "STRATEGY_EXIT"))
                elif sig == 1 and not is_holding:
                    # リエントリー禁止チェック
                    ban_until = reentry_ban.get(sym, -1)
                    if i >= ban_until:
                        rsr_val = float(rsr_row.get(sym, 0)) if date in rsr_df.index else 0
                        buy_signals.append((rsr_val, sym))

        # ── 翌日の取引日 ─────────────────────────────────────────────
        if i + 1 >= len(common_dates):
            break
        next_date = common_dates[i + 1]

        # ── SELL 実行（翌日始値） ────────────────────────────────────
        for item in sell_signals:
            sym, reason = item
            if sym not in positions:
                continue
            pos = positions[sym]
            df_sym = universe_raw[sym]["df"]
            if next_date not in df_sym.index:
                continue
            sell_price = float(df_sym.loc[next_date, "Open"])
            proceeds   = pos.qty * sell_price * (1 - COST_ONE_WAY)
            cash      += proceeds
            pnl        = (sell_price - pos.entry_price) * pos.qty
            trades.append({
                "symbol":   sym,
                "side":     "SELL",
                "entry":    pos.entry_price,
                "exit":     sell_price,
                "qty":      pos.qty,
                "pnl":      pnl,
                "reason":   reason,
            })
            del positions[sym]
            if reason == "TIME_STOP":
                reentry_ban[sym] = i + 1 + REENTRY_COOLDOWN

        # ── BUY 実行（翌日始値 / CB 時は禁止） ─────────────────────
        if not cb_active and buy_signals:
            # RSR 降順（高い方が優先）
            buy_signals.sort(key=lambda x: -x[0])
            new_buys = 0
            for rsr_val, sym in buy_signals:
                if new_buys >= max_new_positions_per_day:
                    break
                open_slots = max_positions - len(positions)
                if open_slots <= 0:
                    break
                df_sym = universe_raw[sym]["df"]
                if next_date not in df_sym.index:
                    continue
                buy_price = float(df_sym.loc[next_date, "Open"])
                if buy_price <= 0:
                    continue

                # ライブと同じ alloc 計算
                n_remaining     = sum(1 for _, s in buy_signals if s not in positions)
                effective_slots = min(open_slots, max(1, n_remaining))
                alloc           = cash / effective_slots
                alloc           = min(alloc, capital * max_single_weight)

                qty = int(alloc / buy_price / LOT) * LOT
                if qty <= 0:
                    continue

                cost = qty * buy_price * COST_ONE_WAY
                cash -= qty * buy_price + cost
                positions[sym] = Position(
                    symbol=sym,
                    sector=temporal24.get(sym, "不明"),
                    qty=qty,
                    entry_price=buy_price,
                    entry_idx=i + 1,
                )
                trades.append({
                    "symbol": sym,
                    "side":   "BUY",
                    "entry":  buy_price,
                    "exit":   None,
                    "qty":    qty,
                    "pnl":    None,
                    "reason": f"BUY[top{top_k}] RSR={rsr_val:.1f}",
                })
                new_buys += 1

    # ---- 7. 期末精算 ----
    # equity_curve を Series に変換（date インデックス付き）
    n_days = len(equity_curve)
    eq_dates = common_dates[:n_days]
    equity_series = pd.Series(equity_curve, index=eq_dates)

    # ---- 8. 指標計算 ----
    total_days  = len(equity_series)
    years       = total_days / 252
    final_eq    = equity_series.iloc[-1]
    total_ret   = final_eq / capital - 1
    cagr        = (final_eq / capital) ** (1 / max(years, 0.01)) - 1

    daily_ret   = equity_series.pct_change().dropna()
    sharpe      = float(daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0.0

    rolling_max = equity_series.expanding().max()
    dd_series   = (equity_series - rolling_max) / rolling_max
    max_dd      = float(dd_series.min())
    calmar      = cagr / abs(max_dd) if max_dd < 0 else float("inf")

    avg_exposure = float(np.mean(exposure_list)) if exposure_list else 0.0
    avg_positions = float(np.mean(pos_count_list)) if pos_count_list else 0.0
    zero_exp_rate = sum(1 for e in exposure_list if e == 0) / max(1, len(exposure_list)) * 100

    sell_trades = [t for t in trades if t["side"] == "SELL" and t["pnl"] is not None]
    win_rate    = sum(1 for t in sell_trades if t["pnl"] > 0) / max(1, len(sell_trades))

    return {
        "top_k":         top_k,
        "cagr":          cagr,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "calmar":        calmar,
        "avg_exposure":  avg_exposure,
        "avg_positions": avg_positions,
        "zero_exp_rate": zero_exp_rate,
        "trade_count":   len(sell_trades),
        "win_rate":      win_rate,
        "equity_curve":  equity_series,
    }


# ------------------------------------------------------------------ #
# 単体実行
# ------------------------------------------------------------------ #
def _print_result(m: dict) -> None:
    print(f"\n  top_k = {m['top_k']}")
    print(f"  {'CAGR':<20} {m['cagr']:>+8.2%}")
    print(f"  {'Sharpe':<20} {m['sharpe']:>8.3f}")
    print(f"  {'MaxDD':<20} {m['max_dd']:>8.2%}")
    print(f"  {'Calmar':<20} {m['calmar']:>8.3f}")
    print(f"  {'avg_exposure':<20} {m['avg_exposure']:>8.1%}   （正常: 25〜35%）")
    print(f"  {'avg_positions':<20} {m['avg_positions']:>8.2f}   （研究値: 1.91）")
    print(f"  {'zero_exp_rate':<20} {m['zero_exp_rate']:>8.1f}%  （研究値: ~19%）")
    print(f"  {'取引数':<20} {m['trade_count']:>8d}")
    print(f"  {'勝率':<20} {m['win_rate']:>8.1%}")


def main() -> int:
    p = argparse.ArgumentParser(description="ライブ等価バックテスト")
    p.add_argument("--k",     type=int, nargs="+", default=[4], help="top_k 値（複数可）")
    p.add_argument("--start", default=START)
    p.add_argument("--end",   default=END)
    args = p.parse_args()

    print("=" * 64)
    print("  ライブアーキテクチャ完全再現バックテスト")
    print(f"  期間: {args.start} 〜 {args.end}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 64)

    for k in args.k:
        result = run_backtest(top_k=k, start=args.start, end=args.end, verbose=True)
        _print_result(result)

    print("\n" + "=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
