"""
backtest/portfolio_stress_test.py
ストレステスト — 意図的DDを作り案AC制御の有効性を検証

課題: G27+V2の実DD(-7.5%)では案AC閾値(-8%/-12%/-25%)に未到達
解決: レバレッジ(1.0x/1.5x/2.0x) + ポジション緩和でDDを-15%〜-30%まで拡大
     その上で制御なし vs 案ACを比較

案AC（保険型）= 案Aの構造 × 案Cの閾値
  -8%:  ポジションサイズ×0.50
  -12%: entry_stop（新規エントリー禁止）
  -25%: strategy kill（全決済）
  entry_stop解除: DD > -6%
  kill解除: 60営業日 OR TOPIX > MA200
"""

from __future__ import annotations
import os, sys, warnings, json, datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from backtest.universe_builder  import download_universe
from backtest.rsr               import calc_universe_rsr
from backtest.fujiko_strategy   import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.engine            import TradeCost
from backtest.portfolio_engine  import Position
from backtest.portfolio_dd_control import (
    DDScheme, DDControlEngine, DDControlResult,
    build_strategies, make_top2_universe, SECTOR_STRATEGY, MR_PARAMS,
)

START        = "2018-01-01"
END          = "2024-12-31"
CAPITAL      = 2_000_000
TOPIX_TICKER = "1306.T"

# ------------------------------------------------------------------ #
# 案AC: 案Aの構造 × 案Cの閾値
# ------------------------------------------------------------------ #
SCHEME_NONE = DDScheme(
    name="制御なし",
    size_levels={},
    entry_stop_dd=-0.99,
    kill_dd=-0.99,
)

SCHEME_AC = DDScheme(
    name="案AC（保険型）",
    size_levels={-0.08: 0.50},
    entry_stop_dd=-0.12,
    entry_stop_reentry_dd=-0.06,
    kill_dd=-0.25,
    kill_reentry_days=60,
    kill_reentry_topix_ma200=True,
)

SCHEME_CURRENT_CB = DDScheme(
    name="現行CB(-15%/-7.5%)",
    size_levels={},
    entry_stop_dd=-0.99,
    kill_dd=-0.15,
    entry_stop_reentry_dd=-0.075,
)

# ------------------------------------------------------------------ #
# レバレッジ対応エンジン（DDControlEngineを拡張）
# ------------------------------------------------------------------ #
class LeveragedDDEngine(DDControlEngine):
    """
    レバレッジ対応版。
    leverage > 1.0 のとき、ポジションサイズをleverage倍にする。
    キャッシュがマイナスになることを許容（証拠金取引の模擬）。
    """

    def __init__(self, leverage: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.leverage = leverage

    def _position_alloc(self, dd: float) -> float:
        base = (self.capital * self.leverage) / self.max_positions
        return base * self.scheme.size_scalar(dd)

    def run(self) -> DDControlResult:
        """
        DDControlEngine.run() のオーバーライド。
        キャッシュ下限チェックをなくしてレバレッジを実現。
        """
        cash           = self.capital
        positions: dict[str, Position] = {}
        equity_records = []
        trades         = []
        peak_equity    = float(self.capital)

        entry_stop   = False
        kill_mode    = False
        days_in_kill = 0
        kill_events  = 0
        lock_days    = 0

        dates = self.all_dates

        for i, date in enumerate(dates):

            # 1. 時価評価
            mkt_value = sum(
                pos.market_value(self.universe[sym]["df"].loc[date, "Close"])
                for sym, pos in positions.items()
                if date in self.universe[sym]["df"].index
            )
            port_value  = cash + mkt_value
            peak_equity = max(peak_equity, port_value)
            current_dd  = (port_value - peak_equity) / peak_equity

            # 2. ストラテジーキル発動
            if not kill_mode and self.scheme.kill_triggered(current_dd):
                for sym in list(positions.keys()):
                    price = self.universe[sym]["df"].loc[date, "Close"]
                    cash  = self._close_position(
                        sym, positions[sym], price, "KILL", date, cash, trades
                    )
                positions.clear()
                kill_mode    = True
                entry_stop   = True
                days_in_kill = 0
                kill_events += 1

            # 3. キル解除チェック
            if kill_mode:
                days_in_kill += 1
                if self._kill_resolved(current_dd, date, days_in_kill):
                    kill_mode    = False
                    entry_stop   = False
                    days_in_kill = 0

            # 4. エントリー停止発動
            if not kill_mode and not entry_stop:
                if self.scheme.entry_stop_triggered(current_dd):
                    entry_stop = True

            # 5. エントリー停止解除
            if entry_stop and not kill_mode:
                if self._entry_stop_resolved(current_dd, date):
                    entry_stop = False

            if entry_stop or kill_mode:
                lock_days += 1

            # 6. 損切りチェック
            if not kill_mode:
                for sym in list(positions.keys()):
                    df_sym = self.universe[sym]["df"]
                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal == -1:
                        price = df_sym.loc[date, "Close"]
                        cash  = self._close_position(
                            sym, positions[sym], price, "STOP", date, cash, trades
                        )
                        del positions[sym]

            # 7. エントリーチェック（キャッシュ下限なし = レバレッジ）
            if not kill_mode and not entry_stop and i + 1 < len(dates):
                next_date = dates[i + 1]
                alloc = self._position_alloc(current_dd)

                for sym, info in self.universe.items():
                    if sym in positions:
                        continue
                    if len(positions) >= self.max_positions:
                        break

                    df_sym = info["df"]
                    if date not in df_sym.index or next_date not in df_sym.index:
                        continue

                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal != 1:
                        continue

                    exec_price = (
                        df_sym.loc[next_date, "Open"] * (1 + self.cost.slippage_rate)
                    )
                    qty = int(alloc // (exec_price * 100)) * 100
                    if qty <= 0:
                        continue

                    total_cost = qty * exec_price + max(
                        qty * exec_price * self.cost.commission_rate,
                        self.cost.min_commission,
                    )
                    # レバレッジ: キャッシュ下限チェックを緩和
                    # （cash < total_cost でもポジションを取れる = マージン借入）
                    if cash + self.capital * (self.leverage - 1.0) < total_cost:
                        continue

                    cash -= total_cost
                    positions[sym] = Position(
                        symbol=sym, sector=info["sector"],
                        qty=qty, entry_price=exec_price, entry_date=next_date,
                    )
                    trades.append({
                        "date": next_date, "symbol": sym, "sector": info["sector"],
                        "side": "BUY", "qty": qty, "price": exec_price, "pnl": 0.0,
                    })

            # 8. 記録
            equity_records.append({
                "date":       date,
                "value":      port_value,
                "dd":         current_dd,
                "n_pos":      len(positions),
                "entry_stop": entry_stop,
                "kill_mode":  kill_mode,
            })

        rec = pd.DataFrame(equity_records).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["date","symbol","sector","side","qty","price","pnl"]
        )
        return DDControlResult(
            equity_curve    = rec["value"],
            dd_series       = rec["dd"],
            n_positions     = rec["n_pos"],
            trades          = trades_df,
            initial_capital = self.capital,
            scheme_name     = f"{self.scheme.name} {self.leverage:.1f}x",
            kill_events     = kill_events,
            lock_days       = lock_days,
        )


# ------------------------------------------------------------------ #
# メトリクス行生成
# ------------------------------------------------------------------ #
def row(res: DDControlResult, label: str) -> dict:
    yd = res.yearly_dd()
    return {
        "シナリオ":      label,
        "CAGR%":        f"{res.cagr*100:+.2f}",
        "Sharpe":       f"{res.sharpe:.3f}",
        "MaxDD%":       f"{res.max_drawdown*100:.2f}",
        "Calmar":       f"{res.calmar:.3f}",
        "平均保有":     f"{res.avg_positions:.2f}",
        "年間取引":     f"{res.n_trades_year:.1f}",
        "CB/ロック":    f"{res.kill_events}回/{res.lock_days}日",
        "2021DD%":      f"{yd.get(2021,0):.1f}",
        "2022DD%":      f"{yd.get(2022,0):.1f}",
        "2023DD%":      f"{yd.get(2023,0):.1f}",
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 80)
    print("  ストレステスト — 案AC保険型 DD制御有効性検証")
    print("  レバレッジ 1.0x / 1.5x / 2.0x × 制御なし vs 案AC vs 現行CB")
    print(f"  期間: {START}〜{END}  初期資本: ¥{CAPITAL:,}")
    print("=" * 80)

    # ---- データ取得 ----
    df_sel       = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
    mask_27      = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27   = {r["symbol"]: r["sector"] for _, r in df_sel[mask_27].iterrows()}

    all_tickers  = {**tickers_27, TOPIX_TICKER: "ETF"}
    print(f"\n[1/3] データ取得中（{len(all_tickers)}銘柄）...")
    universe_all = download_universe(all_tickers, start=START, end=END, verbose=False)

    topix        = universe_all[TOPIX_TICKER]["df"]["Close"] if TOPIX_TICKER in universe_all \
                   else pd.Series(dtype=float)
    universe_g27 = {sym: universe_all[sym] for sym in tickers_27 if sym in universe_all}
    print(f"  G27: {len(universe_g27)}銘柄  TOPIX: {'取得済' if len(topix)>0 else '取得失敗'}")

    prices_all = {sym: info["df"]["Close"] for sym, info in universe_all.items()
                  if sym != TOPIX_TICKER}
    rsr_uni    = calc_universe_rsr(prices_all)

    strats_g27 = build_strategies(universe_g27, rsr_uni, sym_to_strat)
    cost       = TradeCost()

    # ---- シナリオ定義 ----
    # (label, leverage, max_pos, scheme)
    scenarios = []

    # [A] レバレッジ比較（max_pos=3固定、G27）
    for lev in [1.0, 1.5, 2.0]:
        for scheme, s_name in [
            (SCHEME_NONE,       "制御なし"),
            (SCHEME_AC,         "案AC"),
            (SCHEME_CURRENT_CB, "現行CB"),
        ]:
            scenarios.append((
                f"{s_name} {lev:.1f}x / max3",
                lev, 3, scheme, universe_g27, strats_g27
            ))

    # [B] ポジション数緩和（2.0xレバ、max_pos=5、G27）
    for scheme, s_name in [
        (SCHEME_NONE, "制御なし"),
        (SCHEME_AC,   "案AC"),
    ]:
        scenarios.append((
            f"{s_name} 2.0x / max5",
            2.0, 5, scheme, universe_g27, strats_g27
        ))

    print(f"\n[2/3] {len(scenarios)}シナリオ実行...")

    rows = []
    for label, lev, max_pos, scheme, univ, strats in scenarios:
        print(f"  {label} ...", end=" ", flush=True)
        eng = LeveragedDDEngine(
            leverage=lev, universe=univ, strategies=strats,
            scheme=scheme, topix=topix, capital=CAPITAL,
            max_positions=max_pos, max_single_weight=0.25, cost=cost,
        )
        res = eng.run()
        r = row(res, label)
        rows.append(r)
        print(f"CAGR={r['CAGR%']}%  MaxDD={r['MaxDD%']}%  Calmar={r['Calmar']}  {r['CB/ロック']}")

    # ---- 結果表示 ----
    print(f"\n[3/3] 結果サマリー")
    print("=" * 110)
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    # セクション別に見やすく表示
    print("\n" + "=" * 80)
    print("  ▼ レバレッジ別 案AC効果（DD制御なし vs 案AC）")
    print("=" * 80)
    hdr = f"{'シナリオ':<32} {'MaxDD%':>7} {'CAGR%':>7} {'Sharpe':>7} {'Calmar':>7} {'CB/ロック':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['シナリオ']:<32} {r['MaxDD%']:>7} {r['CAGR%']:>7} "
              f"{r['Sharpe']:>7} {r['Calmar']:>7} {r['CB/ロック']:>14}")

    # 2022年フォーカス
    print("\n  ▼ 2022年前後のMaxDD（758日問題の文脈）")
    print(f"  {'シナリオ':<32} {'2021DD%':>8} {'2022DD%':>8} {'2023DD%':>8}")
    print("  " + "-" * 58)
    for r in rows:
        print(f"  {r['シナリオ']:<32} {r['2021DD%']:>8} {r['2022DD%']:>8} {r['2023DD%']:>8}")

    # JSON保存
    today    = datetime.date.today().strftime("%Y-%m-%d")
    out_path = f"results/stress_test_{today}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")

    # 推奨値サマリー
    print("\n" + "=" * 80)
    print("  ▼ 案AC閾値評価（何xレバで初めて発動するか）")
    print("=" * 80)
    for r in rows:
        if "案AC" in r["シナリオ"] and r["CB/ロック"] != "0回/0日":
            print(f"  ✅ {r['シナリオ']}: MaxDD={r['MaxDD%']}%  {r['CB/ロック']}  Calmar={r['Calmar']}")
    print("  ※ 上記が空の場合: 2.0xでもDD制御が不要なほど戦略が堅牢")


if __name__ == "__main__":
    main()
