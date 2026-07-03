"""
backtest/portfolio_dd_control.py
段階型DDリスク制御バックテスト

課題: 2022年下落相場での758日ロック問題（現行CB: -15%発動 / -7.5%解除）
目的: CBデッドロック回避 + Sharpe/CAGRの安定化

■ 案A（標準型）
  -10%: size×0.50 / -15%: entry_stop / -25%: size×0.25 / -40%: kill
  entry_stop解除: DD > -7%
  kill解除: 60営業日 OR TOPIX > MA200

■ 案B（シンプル型）
  -12%: entry_stop / -35%: kill
  解除: 30営業日 OR TOPIX > MA200

■ 案C（資本保護型）
  -8%: size×0.50 / -12%: entry_stop / -25%: kill
  解除: DD > -6%

ユニバース:
  G27  - 現行27銘柄
  Top2 - G27から各セクター上位2銘柄（Sharpe順）
"""

from __future__ import annotations
import os, sys, warnings
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
from backtest.engine            import TradeCost, BacktestEngine
from backtest.portfolio_engine  import Position, PortfolioResult

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000
TOPIX_TICKER = "1306.T"   # TOPIXのETF（MA200フィルター用）

# ------------------------------------------------------------------ #
# セクター→戦略マッピング（portfolio_v2.py と同一）
# ------------------------------------------------------------------ #
SECTOR_STRATEGY = {
    "海運":"fujiko","機械":"fujiko","電機精密":"fujiko","商社":"fujiko",
    "電機":"fujiko","ゲーム":"fujiko","レジャー":"fujiko","食品":"fujiko",
    "ガス":"mean_rev","鉄鋼":"mean_rev","銀行":"mean_rev","保険":"mean_rev",
    "輸送機器":"mean_rev","化学":"mean_rev","小売":"mean_rev",
    "サービス":"dynamic","医薬品":"dynamic","不動産":"dynamic",
    "情報通信":"dynamic","陸運":"dynamic",
}
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

# ------------------------------------------------------------------ #
# DDスキーム定義
# ------------------------------------------------------------------ #
@dataclass
class DDScheme:
    name:                  str
    # サイズ縮小 {DD閾値: スケーラー}  例: {-0.10: 0.50, -0.25: 0.25}
    size_levels:           dict = field(default_factory=dict)
    # エントリー停止発動DD
    entry_stop_dd:         float = -0.99
    # エントリー停止解除DD（None = DD条件なし）
    entry_stop_reentry_dd: Optional[float] = None
    # ストラテジーキル発動DD
    kill_dd:               float = -0.99
    # キル解除：営業日数（None = 日数条件なし）
    kill_reentry_days:     Optional[int] = None
    # キル解除：TOPIX > MA200 条件を使うか
    kill_reentry_topix_ma200: bool = False

    def size_scalar(self, dd: float) -> float:
        """現在DDに対するサイズスケーラーを返す（0〜1）"""
        scalar = 1.0
        for threshold, scale in sorted(self.size_levels.items()):
            if dd <= threshold:
                scalar = scale
        return scalar

    def entry_stop_triggered(self, dd: float) -> bool:
        return dd <= self.entry_stop_dd

    def kill_triggered(self, dd: float) -> bool:
        return dd <= self.kill_dd


SCHEME_A = DDScheme(
    name="案A（標準型）",
    size_levels={-0.10: 0.50, -0.25: 0.25},
    entry_stop_dd=-0.15,
    entry_stop_reentry_dd=-0.07,
    kill_dd=-0.40,
    kill_reentry_days=60,
    kill_reentry_topix_ma200=True,
)

SCHEME_B = DDScheme(
    name="案B（シンプル型）",
    size_levels={},
    entry_stop_dd=-0.12,
    entry_stop_reentry_dd=None,  # entry_stopはkillと同じ条件で解除
    kill_dd=-0.35,
    kill_reentry_days=30,
    kill_reentry_topix_ma200=True,
)

SCHEME_C = DDScheme(
    name="案C（資本保護型）",
    size_levels={-0.08: 0.50},
    entry_stop_dd=-0.12,
    entry_stop_reentry_dd=-0.06,
    kill_dd=-0.25,
    kill_reentry_days=None,
    kill_reentry_topix_ma200=False,  # DD > -6% のみ
)

# 現行ベースライン（比較用）
SCHEME_CURRENT = DDScheme(
    name="現行CB(-15%/-7.5%)",
    size_levels={},
    entry_stop_dd=-0.99,   # entry_stopなし
    kill_dd=-0.15,
    kill_reentry_days=None,
    kill_reentry_topix_ma200=False,
    # kill解除 = DD > -7.5%（現行ロジック）で entry_stop_reentry_ddを流用
    entry_stop_reentry_dd=-0.075,
)


# ------------------------------------------------------------------ #
# 段階型DDエンジン
# ------------------------------------------------------------------ #
class DDControlEngine:
    """
    段階型DDリスク制御ポートフォリオエンジン。
    PortfolioEngineのrunループを参考に再実装し、多段階CB制御を追加。
    """

    def __init__(
        self,
        universe:          dict,
        strategies:        dict,
        scheme:            DDScheme,
        topix:             pd.Series,
        capital:           float     = CAPITAL,
        max_positions:     int       = 3,
        max_single_weight: float     = 0.25,
        cost:              TradeCost = None,
    ):
        self.universe          = universe
        self.strategies        = strategies
        self.scheme            = scheme
        self.topix             = topix
        self.capital           = capital
        self.max_positions     = max_positions
        self.max_single_weight = max_single_weight
        self.cost              = cost or TradeCost()

        # 共通取引日
        dates = None
        for info in universe.values():
            idx = info["df"].index
            dates = idx if dates is None else dates.intersection(idx)
        self.all_dates = dates.sort_values()

    def _topix_above_ma200(self, date: pd.Timestamp) -> bool:
        past = self.topix.loc[:date]
        if len(past) < 200:
            return True  # データ不足 → フィルターなし
        return float(past.iloc[-1]) > float(past.tail(200).mean())

    def _entry_stop_resolved(self, dd: float, date: pd.Timestamp) -> bool:
        """エントリー停止が解除できる状態か"""
        s = self.scheme
        if s.entry_stop_reentry_dd is not None:
            if dd > s.entry_stop_reentry_dd:
                return True
        return False

    def _kill_resolved(
        self,
        dd:             float,
        date:           pd.Timestamp,
        days_in_kill:   int,
    ) -> bool:
        """ストラテジーキルが解除できる状態か"""
        s = self.scheme
        # 案C: DD回復のみ（entry_stop_reentry_ddを流用）
        if not s.kill_reentry_topix_ma200 and s.kill_reentry_days is None:
            return s.entry_stop_reentry_dd is not None and dd > s.entry_stop_reentry_dd
        # 案A/B: 日数 OR TOPIX>MA200
        if s.kill_reentry_days is not None and days_in_kill >= s.kill_reentry_days:
            return True
        if s.kill_reentry_topix_ma200 and self._topix_above_ma200(date):
            return True
        return False

    def _position_alloc(self, dd: float) -> float:
        base = self.capital / self.max_positions
        return base * self.scheme.size_scalar(dd)

    def _close_position(self, sym, pos, price, reason, date, cash, trades):
        exec_price  = price * (1 - self.cost.slippage_rate)
        trade_value = pos.qty * exec_price
        commission  = max(trade_value * self.cost.commission_rate, self.cost.min_commission)
        pnl  = (exec_price - pos.entry_price) * pos.qty - commission
        cash += trade_value - commission
        trades.append({
            "date": date, "symbol": sym, "sector": pos.sector,
            "side": f"SELL_{reason}", "qty": pos.qty,
            "price": exec_price, "pnl": pnl,
        })
        return cash

    def run(self) -> "DDControlResult":
        cash           = self.capital
        positions: dict[str, Position] = {}
        equity_records = []
        trades         = []
        peak_equity    = float(self.capital)

        entry_stop  = False   # 新規エントリー禁止（既存は保有継続）
        kill_mode   = False   # 全決済 + 待機
        days_in_kill = 0      # kill発動からの営業日数
        kill_events  = 0      # kill発動回数
        lock_days    = 0      # entry_stop or kill_mode の日数合計

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

            # 2. ストラテジーキル発動（まだ未発動のとき）
            if not kill_mode and self.scheme.kill_triggered(current_dd):
                for sym in list(positions.keys()):
                    price = self.universe[sym]["df"].loc[date, "Close"]
                    cash  = self._close_position(sym, positions[sym], price,
                                                 "KILL", date, cash, trades)
                positions.clear()
                kill_mode   = True
                entry_stop  = True
                days_in_kill = 0
                kill_events += 1

            # 3. キルモード中: 解除チェック
            if kill_mode:
                days_in_kill += 1
                if self._kill_resolved(current_dd, date, days_in_kill):
                    kill_mode   = False
                    entry_stop  = False
                    days_in_kill = 0

            # 4. エントリー停止発動（キルに至らない中程度DD）
            if not kill_mode and not entry_stop:
                if self.scheme.entry_stop_triggered(current_dd):
                    entry_stop = True

            # 5. エントリー停止解除（DD回復）
            if entry_stop and not kill_mode:
                if self._entry_stop_resolved(current_dd, date):
                    entry_stop = False

            # ロック日カウント
            if entry_stop or kill_mode:
                lock_days += 1

            # 6. 損切りチェック（キルモード解除後も含む）
            if not kill_mode:
                for sym in list(positions.keys()):
                    df_sym = self.universe[sym]["df"]
                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal == -1:
                        price = df_sym.loc[date, "Close"]
                        cash  = self._close_position(sym, positions[sym], price,
                                                     "STOP", date, cash, trades)
                        del positions[sym]

            # 7. エントリーチェック
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

                    exec_price = df_sym.loc[next_date, "Open"] * (1 + self.cost.slippage_rate)
                    qty = int(alloc // (exec_price * 100)) * 100
                    if qty <= 0:
                        continue

                    total_cost = qty * exec_price + max(
                        qty * exec_price * self.cost.commission_rate,
                        self.cost.min_commission,
                    )
                    if total_cost > cash:
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
            equity_curve  = rec["value"],
            dd_series     = rec["dd"],
            n_positions   = rec["n_pos"],
            trades        = trades_df,
            initial_capital = self.capital,
            scheme_name   = self.scheme.name,
            kill_events   = kill_events,
            lock_days     = lock_days,
        )


# ------------------------------------------------------------------ #
# 結果クラス
# ------------------------------------------------------------------ #
@dataclass
class DDControlResult:
    equity_curve:    pd.Series
    dd_series:       pd.Series
    n_positions:     pd.Series
    trades:          pd.DataFrame
    initial_capital: float
    scheme_name:     str
    kill_events:     int
    lock_days:       int

    @property
    def total_return(self) -> float:
        return self.equity_curve.iloc[-1] / self.initial_capital - 1.0

    @property
    def cagr(self) -> float:
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        return (1 + self.total_return) ** (1.0 / max(years, 0.01)) - 1

    @property
    def sharpe(self) -> float:
        r = self.equity_curve.pct_change().dropna()
        return float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0.0

    @property
    def max_drawdown(self) -> float:
        return float(self.dd_series.min())

    @property
    def calmar(self) -> float:
        dd = abs(self.max_drawdown)
        return self.cagr / dd if dd > 0 else float("inf")

    @property
    def n_trades_year(self) -> float:
        sells = self.trades[self.trades["side"].str.startswith("SELL")]
        if sells.empty:
            return 0.0
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        return len(sells) / max(years, 0.01)

    @property
    def avg_positions(self) -> float:
        return float(self.n_positions.mean())

    def yearly_dd(self) -> dict:
        """年別MaxDD（2022年の状況確認用）"""
        result = {}
        for yr in range(2018, 2025):
            yd = self.dd_series[self.dd_series.index.year == yr]
            result[yr] = round(float(yd.min()) * 100, 1) if not yd.empty else 0.0
        return result


# ------------------------------------------------------------------ #
# ヘルパー
# ------------------------------------------------------------------ #
def build_strategies(universe: dict, rsr_uni: pd.DataFrame, sym_to_strat: dict) -> dict:
    strats = {}
    for sym, info in universe.items():
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        sector = info["sector"]
        rule   = SECTOR_STRATEGY.get(sector, "dynamic")
        lbl    = sym_to_strat.get(sym, "フジコ法")
        if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in lbl):
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=6, min_rsr=70.0,
                mom_period=21, turtle_entry=20, turtle_exit=10,
                use_turtle_entry=True,
            )
    return strats


def make_top2_universe(universe: dict, df_sel: pd.DataFrame) -> dict:
    """G27から各セクター上位2銘柄（Sharpe順）を選択"""
    sym_sharpe = dict(zip(df_sel["symbol"], df_sel["sharpe"]))
    sector_map: dict[str, list] = {}
    for sym, info in universe.items():
        sec = info["sector"]
        sh  = sym_sharpe.get(sym, 0.0)
        sector_map.setdefault(sec, []).append((sh, sym))

    selected = {}
    for sec, items in sector_map.items():
        items.sort(reverse=True)
        for _, sym in items[:2]:
            selected[sym] = universe[sym]
    return selected


def metrics_row(res: DDControlResult, label: str) -> dict:
    return {
        "シナリオ":       label,
        "CAGR%":         f"{res.cagr*100:+.2f}",
        "Sharpe":        f"{res.sharpe:.3f}",
        "MaxDD%":        f"{res.max_drawdown*100:.2f}",
        "Calmar":        f"{res.calmar:.3f}",
        "平均保有銘柄":  f"{res.avg_positions:.2f}",
        "年間取引数":    f"{res.n_trades_year:.1f}",
        "CB数/ロック日": f"{res.kill_events}回/{res.lock_days}日",
        # 年別MaxDD（2022年検証）
        "2021DD%":       f"{res.yearly_dd().get(2021, 0):.1f}",
        "2022DD%":       f"{res.yearly_dd().get(2022, 0):.1f}",
        "2023DD%":       f"{res.yearly_dd().get(2023, 0):.1f}",
    }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 72)
    print("  段階型DDリスク制御バックテスト")
    print("  課題: 2022年下落相場での758日ロック問題の解消")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 72)

    # ---- 1. データ取得 ----
    df_sel       = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
    mask_27      = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27   = {r["symbol"]: r["sector"] for _, r in df_sel[mask_27].iterrows()}

    all_tickers = {**tickers_27, TOPIX_TICKER: "ETF"}
    print(f"\n[1/4] データ取得中（{len(all_tickers)}銘柄 + TOPIX）...")
    universe_all = download_universe(all_tickers, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe_all)} 銘柄")

    # TOPIX（MA200フィルター用）
    topix = universe_all[TOPIX_TICKER]["df"]["Close"] if TOPIX_TICKER in universe_all else pd.Series(dtype=float)
    universe_g27 = {sym: universe_all[sym] for sym in tickers_27 if sym in universe_all}
    print(f"  G27ユニバース: {len(universe_g27)} 銘柄")

    # ---- 2. RSR計算 ----
    print("\n[2/4] RSR計算...")
    prices_all = {sym: info["df"]["Close"] for sym, info in universe_all.items()
                  if sym != TOPIX_TICKER}
    rsr_uni    = calc_universe_rsr(prices_all)

    # ---- 3. ユニバース準備 ----
    universe_top2 = make_top2_universe(universe_g27, df_sel)
    print(f"  Top2/セクターユニバース: {len(universe_top2)} 銘柄")
    top2_sectors = {}
    for sym, info in universe_top2.items():
        top2_sectors.setdefault(info["sector"], []).append(sym)
    for sec, syms in sorted(top2_sectors.items()):
        print(f"    {sec}: {', '.join(syms)}")

    # 戦略辞書
    strats_g27  = build_strategies(universe_g27,  rsr_uni, sym_to_strat)
    strats_top2 = build_strategies(universe_top2, rsr_uni, sym_to_strat)

    cost = TradeCost()

    # ---- 4. バックテスト実行 ----
    print("\n[3/4] バックテスト実行（7シナリオ）...")

    scenarios = [
        ("現行CB(-15%/-7.5%) G27",  universe_g27,  strats_g27,  SCHEME_CURRENT),
        ("案A G27",                  universe_g27,  strats_g27,  SCHEME_A),
        ("案A Top2",                 universe_top2, strats_top2, SCHEME_A),
        ("案B G27",                  universe_g27,  strats_g27,  SCHEME_B),
        ("案B Top2",                 universe_top2, strats_top2, SCHEME_B),
        ("案C G27",                  universe_g27,  strats_g27,  SCHEME_C),
        ("案C Top2",                 universe_top2, strats_top2, SCHEME_C),
    ]

    rows = []
    for label, univ, strats, scheme in scenarios:
        print(f"  [{label}] ...", end=" ", flush=True)
        eng = DDControlEngine(
            universe=univ, strategies=strats, scheme=scheme,
            topix=topix, capital=CAPITAL, max_positions=3,
            max_single_weight=0.25, cost=cost,
        )
        res = eng.run()
        row = metrics_row(res, label)
        rows.append(row)
        print(f"CAGR={row['CAGR%']}%  MaxDD={row['MaxDD%']}%  "
              f"Calmar={row['Calmar']}  {row['CB数/ロック日']}")

    # ---- 5. 結果表示 ----
    print("\n[4/4] 結果サマリー")
    print("=" * 100)
    df_out = pd.DataFrame(rows)
    print(df_out.to_string(index=False))

    # 2022年フォーカス表示
    print("\n--- 2022年前後 年別MaxDD ---")
    hdr = f"{'シナリオ':<30} {'2021DD%':>8} {'2022DD%':>8} {'2023DD%':>8} {'ロック日':>8}"
    print(hdr)
    print("-" * len(hdr))
    for row in rows:
        print(f"{row['シナリオ']:<30} {row['2021DD%']:>8} {row['2022DD%']:>8} "
              f"{row['2023DD%']:>8} {row['CB数/ロック日']:>12}")

    # JSON保存
    import json, datetime
    today = datetime.date.today().strftime("%Y-%m-%d")
    out_path = f"results/dd_control_{today}.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")


if __name__ == "__main__":
    main()
