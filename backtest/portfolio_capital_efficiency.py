"""
backtest/portfolio_capital_efficiency.py
資本効率最大化バックテスト

目的: 「資本効率の最大化」「CAGR向上」
実装:
  ① RSRランキング加重配分（均等配分禁止）
  ② MIN_POSITIONS=3 強制保有（稼働率向上）
  ③ RSRベース強制ローテーション（資金枯渇対策）

比較シナリオ:
  BASE  : 均等配分 / min_pos無し / ローテーション無し
  ①    : ランキング加重のみ
  ②    : MIN_POSITIONS=3 のみ
  ③    : 強制ローテーションのみ
  ①+②  : ランキング加重 + MIN_POSITIONS
  ①+③  : ランキング加重 + ローテーション
  FULL  : 3つすべて統合
  感度分析: RSR閾値(60/65/70) × ローテーション差分(3/5/8)
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
    build_strategies, SECTOR_STRATEGY, MR_PARAMS,
)

START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000


# ------------------------------------------------------------------ #
# 設定クラス
# ------------------------------------------------------------------ #
@dataclass
class CEConfig:
    """Capital Efficiency 設定"""
    name:                str   = "BASE"
    # ① ランキング加重
    use_rank_weight:     bool  = False
    # ② 最低保有銘柄数
    min_positions:       int   = 0        # 0 = 無効
    min_pos_rsr:         float = 60.0     # MIN_POSITIONS補完エントリーのRSR下限
    min_pos_weight:      float = 0.08     # 補完エントリーのサイズ（資本比）
    min_pos_ma_period:   int   = 50       # トレンド条件 MA期間
    # ③ 強制ローテーション
    use_rotation:        bool  = False
    rotation_rsr_diff:   float = 5.0      # new_rsr - worst_rsr > この値で入替
    # 共通
    max_positions:       int   = 3
    max_pos_weight:      float = 0.40     # 1銘柄最大40%（ランキング加重時）
    dd_entry_suppress:   float = -0.10    # DD < -10% で新規抑制
    max_orders_per_day:  int   = 2        # 1日あたり最大発注数（過剰防止）


# ------------------------------------------------------------------ #
# ランキング加重計算
# ------------------------------------------------------------------ #
def rank_weights(rsr_values: list[tuple[str, float]]) -> dict[str, float]:
    """
    RSR値のリストから順位加重辞書を返す。
    1位に最大配分、n位に最小配分（線形加重）。

    例: 3銘柄の場合
      1位: 3/6 = 50%
      2位: 2/6 = 33%
      3位: 1/6 = 17%
    """
    n = len(rsr_values)
    if n == 0:
        return {}
    sorted_syms = sorted(rsr_values, key=lambda x: x[1], reverse=True)
    total = n * (n + 1) / 2
    return {sym: (n - rank) / total for rank, (sym, _) in enumerate(sorted_syms)}


# ------------------------------------------------------------------ #
# 資本効率エンジン
# ------------------------------------------------------------------ #
class CapitalEfficiencyEngine:
    """
    ① RSRランキング加重配分
    ② MIN_POSITIONS強制保有
    ③ RSRベース強制ローテーション
    を実装したポートフォリオエンジン。
    """

    def __init__(
        self,
        universe:   dict,
        strategies: dict,
        rsr_series: pd.DataFrame,     # shape: (日付, 銘柄) の事前計算済みRSR
        config:     CEConfig,
        capital:    float     = CAPITAL,
        cost:       TradeCost = None,
    ):
        self.universe   = universe
        self.strategies = strategies
        self.rsr_series = rsr_series
        self.cfg        = config
        self.capital    = capital
        self.cost       = cost or TradeCost()

        # 全銘柄共通取引日
        dates = None
        for info in universe.values():
            idx = info["df"].index
            dates = idx if dates is None else dates.intersection(idx)
        self.all_dates = dates.sort_values()

    def _get_rsr(self, sym: str, date: pd.Timestamp) -> float:
        """指定日時点のRSR値を返す（データなしは0.0）"""
        if sym not in self.rsr_series.columns:
            return 0.0
        past = self.rsr_series[sym].loc[:date]
        return float(past.iloc[-1]) if len(past) > 0 else 0.0

    def _get_ma(self, sym: str, date: pd.Timestamp, period: int) -> Optional[float]:
        """指定日時点のMA値を返す"""
        df = self.universe[sym]["df"]
        past = df["Close"].loc[:date]
        if len(past) < period:
            return None
        return float(past.tail(period).mean())

    def _position_vol(self, sym: str, date: pd.Timestamp, window: int = 20) -> float:
        """直近ボラティリティ（低いほど補完エントリー優先）"""
        df = self.universe[sym]["df"]
        past = df["Close"].loc[:date].pct_change().dropna()
        if len(past) < window:
            return 1.0
        return float(past.tail(window).std())

    def _alloc_rank_weighted(
        self,
        buy_signals: list[tuple[str, float]],  # [(sym, rsr), ...]
        available_cash: float,
        current_dd: float,
    ) -> dict[str, float]:
        """
        ① ランキング加重配分。
        利用可能資金を RSR 順位加重で割り当てる。
        DD -10%以下では配分を半減。
        """
        weights = rank_weights(buy_signals)
        dd_scalar = 0.5 if current_dd <= self.cfg.dd_entry_suppress else 1.0
        max_alloc = self.capital * self.cfg.max_pos_weight

        allocs = {}
        for sym, w in weights.items():
            alloc = available_cash * w * dd_scalar
            alloc = min(alloc, max_alloc)
            allocs[sym] = alloc
        return allocs

    def _alloc_equal(
        self,
        n_signals: int,
        current_dd: float,
    ) -> float:
        """均等配分（BASE用）"""
        dd_scalar = 0.5 if current_dd <= self.cfg.dd_entry_suppress else 1.0
        return (self.capital / self.cfg.max_positions) * dd_scalar

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

    def _open_position(self, sym, alloc, exec_price, sector, next_date, cash, positions, trades):
        """ポジションをオープンして cash を返す"""
        qty = int(alloc // (exec_price * 100)) * 100
        if qty <= 0:
            return cash, False
        total_cost = qty * exec_price + max(
            qty * exec_price * self.cost.commission_rate,
            self.cost.min_commission,
        )
        if total_cost > cash:
            return cash, False
        cash -= total_cost
        positions[sym] = Position(
            symbol=sym, sector=sector,
            qty=qty, entry_price=exec_price, entry_date=next_date,
        )
        trades.append({
            "date": next_date, "symbol": sym, "sector": sector,
            "side": "BUY", "qty": qty, "price": exec_price, "pnl": 0.0,
        })
        return cash, True

    def run(self) -> "CEResult":
        cash           = self.capital
        positions: dict[str, Position] = {}
        equity_records = []
        trades         = []
        peak_equity    = float(self.capital)
        rotation_count = 0

        cfg   = self.cfg
        dates = self.all_dates

        for i, date in enumerate(dates):

            # ── 1. 時価評価 ──────────────────────────────────────────
            mkt_value = sum(
                pos.market_value(self.universe[sym]["df"].loc[date, "Close"])
                for sym, pos in positions.items()
                if date in self.universe[sym]["df"].index
            )
            port_value  = cash + mkt_value
            peak_equity = max(peak_equity, port_value)
            current_dd  = (port_value - peak_equity) / peak_equity
            exposure    = mkt_value / port_value if port_value > 0 else 0.0

            # ── 2. 損切りチェック ─────────────────────────────────────
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

            # ── 3. エントリー / ローテーション ──────────────────────
            orders_today = 0
            if i + 1 < len(dates):
                next_date = dates[i + 1]

                # BUYシグナル収集（全候補）
                buy_sigs: list[tuple[str, float]] = []
                for sym, info in self.universe.items():
                    if sym in positions:
                        continue
                    df_sym = info["df"]
                    if date not in df_sym.index or next_date not in df_sym.index:
                        continue
                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal == 1:
                        rsr = self._get_rsr(sym, date)
                        buy_sigs.append((sym, rsr))

                # DD -10%以下でエントリー抑制
                entry_ok = current_dd > cfg.dd_entry_suppress

                if entry_ok and buy_sigs:
                    # ── ③ 強制ローテーション ──────────────────────
                    if cfg.use_rotation and positions and cash < self.capital * 0.15:
                        # 最低RSR保有銘柄を特定
                        worst_sym  = min(
                            positions.keys(),
                            key=lambda s: self._get_rsr(s, date)
                        )
                        worst_rsr  = self._get_rsr(worst_sym, date)
                        best_sig   = max(buy_sigs, key=lambda x: x[1])
                        best_sym, best_rsr = best_sig

                        if best_rsr - worst_rsr > cfg.rotation_rsr_diff:
                            # 入替実行
                            price = self.universe[worst_sym]["df"].loc[date, "Close"]
                            cash  = self._close_position(
                                worst_sym, positions[worst_sym], price,
                                "ROTATION", date, cash, trades
                            )
                            del positions[worst_sym]
                            rotation_count += 1
                            # buy_sigsから除外して続く通常エントリーで処理

                    # ── ① ランキング加重 or 均等配分 ───────────────
                    if entry_ok and buy_sigs and len(positions) < cfg.max_positions:
                        if cfg.use_rank_weight:
                            allocs = self._alloc_rank_weighted(buy_sigs, cash, current_dd)
                        else:
                            eq_alloc = self._alloc_equal(len(buy_sigs), current_dd)
                            allocs   = {sym: eq_alloc for sym, _ in buy_sigs}

                        # RSR降順でエントリー
                        for sym, rsr in sorted(buy_sigs, key=lambda x: x[1], reverse=True):
                            if sym in positions:
                                continue
                            if len(positions) >= cfg.max_positions:
                                break
                            if orders_today >= cfg.max_orders_per_day:
                                break
                            info = self.universe[sym]
                            exec_price = (
                                info["df"].loc[next_date, "Open"]
                                * (1 + self.cost.slippage_rate)
                            )
                            alloc = allocs.get(sym, 0.0)
                            cash, opened = self._open_position(
                                sym, alloc, exec_price, info["sector"],
                                next_date, cash, positions, trades,
                            )
                            if opened:
                                orders_today += 1

                # ── ② MIN_POSITIONS 補完エントリー ──────────────────
                if (entry_ok
                        and cfg.min_positions > 0
                        and len(positions) < cfg.min_positions
                        and orders_today < cfg.max_orders_per_day):

                    # 未保有かつ RSR>min_pos_rsr かつ price>MA50 かつ 低ボラ優先
                    candidates: list[tuple[float, str]] = []
                    for sym, info in self.universe.items():
                        if sym in positions:
                            continue
                        df_sym = info["df"]
                        if date not in df_sym.index or next_date not in df_sym.index:
                            continue

                        rsr = self._get_rsr(sym, date)
                        if rsr < cfg.min_pos_rsr:
                            continue

                        # トレンド条件: price > MA50
                        ma50 = self._get_ma(sym, date, cfg.min_pos_ma_period)
                        if ma50 is None:
                            continue
                        price_now = float(info["df"].loc[date, "Close"])
                        if price_now <= ma50:
                            continue

                        vol = self._position_vol(sym, date)
                        candidates.append((vol, sym))   # 低ボラ順

                    candidates.sort()   # ボラ昇順（低ボラ優先）

                    for vol, sym in candidates:
                        if len(positions) >= cfg.min_positions:
                            break
                        if orders_today >= cfg.max_orders_per_day:
                            break
                        info = self.universe[sym]
                        exec_price = (
                            info["df"].loc[next_date, "Open"]
                            * (1 + self.cost.slippage_rate)
                        )
                        min_alloc = self.capital * cfg.min_pos_weight
                        cash, opened = self._open_position(
                            sym, min_alloc, exec_price, info["sector"],
                            next_date, cash, positions, trades,
                        )
                        if opened:
                            orders_today += 1

            # ── 4. 記録 ───────────────────────────────────────────────
            equity_records.append({
                "date":     date,
                "value":    port_value,
                "dd":       current_dd,
                "n_pos":    len(positions),
                "exposure": exposure,
            })

        rec = pd.DataFrame(equity_records).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["date","symbol","sector","side","qty","price","pnl"]
        )
        return CEResult(
            equity_curve    = rec["value"],
            dd_series       = rec["dd"],
            n_positions     = rec["n_pos"],
            exposure_series = rec["exposure"],
            trades          = trades_df,
            initial_capital = self.capital,
            config          = cfg,
            rotation_count  = rotation_count,
        )


# ------------------------------------------------------------------ #
# 結果クラス
# ------------------------------------------------------------------ #
@dataclass
class CEResult:
    equity_curve:    pd.Series
    dd_series:       pd.Series
    n_positions:     pd.Series
    exposure_series: pd.Series
    trades:          pd.DataFrame
    initial_capital: float
    config:          CEConfig
    rotation_count:  int

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
    def avg_positions(self) -> float:
        return float(self.n_positions.mean())

    @property
    def avg_exposure(self) -> float:
        return float(self.exposure_series.mean())

    @property
    def n_trades_year(self) -> float:
        sells = self.trades[self.trades["side"].str.startswith("SELL")]
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        return len(sells) / max(years, 0.01)

    @property
    def rotation_per_year(self) -> float:
        years = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        return self.rotation_count / max(years, 0.01)

    def to_row(self, label: str) -> dict:
        return {
            "シナリオ":       label,
            "CAGR%":         f"{self.cagr*100:+.2f}",
            "Sharpe":        f"{self.sharpe:.3f}",
            "MaxDD%":        f"{self.max_drawdown*100:.2f}",
            "Calmar":        f"{self.calmar:.3f}",
            "平均保有":      f"{self.avg_positions:.2f}",
            "稼働率%":       f"{self.avg_exposure*100:.1f}",
            "年間取引":      f"{self.n_trades_year:.1f}",
            "ローテ/年":     f"{self.rotation_per_year:.1f}",
        }


# ================================================================
# MAIN
# ================================================================
def main():
    print("=" * 80)
    print("  資本効率最大化バックテスト")
    print("  ① RSRランキング加重  ② MIN_POSITIONS=3  ③ 強制ローテーション")
    print(f"  期間: {START}〜{END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 80)

    # ── データ取得 ──────────────────────────────────────────────────
    df_sel       = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
    mask_27      = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27   = {r["symbol"]: r["sector"] for _, r in df_sel[mask_27].iterrows()}

    print(f"\n[1/4] データ取得中（{len(tickers_27)}銘柄）...")
    universe = download_universe(tickers_27, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe)} 銘柄")

    prices = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_uni = calc_universe_rsr(prices)
    print(f"  RSR計算完了: {rsr_uni.shape[1]} 銘柄")

    strats = build_strategies(universe, rsr_uni, sym_to_strat)
    cost   = TradeCost()

    # ── シナリオ定義 ─────────────────────────────────────────────────
    # [label, config]
    scenarios = [
        ("BASE（均等/制御なし）",     CEConfig(name="BASE")),
        ("① ランキング加重",          CEConfig(name="①",
                                               use_rank_weight=True)),
        ("② MIN_POS=3",              CEConfig(name="②",
                                               min_positions=3)),
        ("③ ローテーション(diff=5)", CEConfig(name="③",
                                               use_rotation=True,
                                               rotation_rsr_diff=5.0)),
        ("①+② 加重+MIN_POS",        CEConfig(name="①+②",
                                               use_rank_weight=True,
                                               min_positions=3)),
        ("①+③ 加重+ローテ",          CEConfig(name="①+③",
                                               use_rank_weight=True,
                                               use_rotation=True,
                                               rotation_rsr_diff=5.0)),
        ("FULL ①+②+③",             CEConfig(name="FULL",
                                               use_rank_weight=True,
                                               min_positions=3,
                                               use_rotation=True,
                                               rotation_rsr_diff=5.0)),
    ]

    print(f"\n[2/4] {len(scenarios)}シナリオ実行...")
    base_rows = []
    for label, cfg in scenarios:
        print(f"  [{label}] ...", end=" ", flush=True)
        eng = CapitalEfficiencyEngine(
            universe=universe, strategies=strats, rsr_series=rsr_uni,
            config=cfg, capital=CAPITAL, cost=cost,
        )
        res = eng.run()
        r   = res.to_row(label)
        base_rows.append(r)
        print(f"CAGR={r['CAGR%']}%  Sharpe={r['Sharpe']}  "
              f"稼働率={r['稼働率%']}%  ローテ={r['ローテ/年']}/年")

    # ── 感度分析 ─────────────────────────────────────────────────────
    print(f"\n[3/4] 感度分析（RSR閾値 × ローテ差分）...")
    sensitivity_rows = []
    for rsr_th in [60.0, 65.0, 70.0]:
        for rot_diff in [3.0, 5.0, 8.0]:
            label = f"FULL RSR{int(rsr_th)} diff{int(rot_diff)}"
            cfg   = CEConfig(
                name=label,
                use_rank_weight=True,
                min_positions=3,
                min_pos_rsr=rsr_th,
                use_rotation=True,
                rotation_rsr_diff=rot_diff,
            )
            print(f"  [{label}] ...", end=" ", flush=True)
            eng = CapitalEfficiencyEngine(
                universe=universe, strategies=strats, rsr_series=rsr_uni,
                config=cfg, capital=CAPITAL, cost=cost,
            )
            res = eng.run()
            r   = res.to_row(label)
            sensitivity_rows.append(r)
            print(f"CAGR={r['CAGR%']}%  Sharpe={r['Sharpe']}  "
                  f"稼働率={r['稼働率%']}%  ローテ={r['ローテ/年']}/年")

    # ── 結果表示 ──────────────────────────────────────────────────────
    print(f"\n[4/4] 結果サマリー")

    print("\n" + "=" * 90)
    print("  ▼ メイン比較（7シナリオ）")
    print("=" * 90)
    df_base = pd.DataFrame(base_rows)
    print(df_base.to_string(index=False))

    # 分析①: ランキング加重によるCAGR変化
    print("\n" + "─" * 90)
    print("  【分析①】ランキング加重によるCAGR変化（BASEとの差分）")
    base_cagr = float(base_rows[0]["CAGR%"].replace("+",""))
    for r in base_rows[1:]:
        diff = float(r["CAGR%"].replace("+","")) - base_cagr
        flag = "✅" if diff > 0 else "❌"
        print(f"  {flag} {r['シナリオ']:<28} CAGR差: {diff:+.2f}pp  "
              f"稼働率: {r['稼働率%']}%")

    # 分析②: MIN_POSITIONS の稼働率 vs Sharpe トレードオフ
    print("\n  【分析②】MIN_POSITIONS導入による稼働率とSharpeのトレードオフ")
    for r in base_rows:
        if "BASE" in r["シナリオ"] or "②" in r["シナリオ"] or "FULL" in r["シナリオ"]:
            print(f"  {r['シナリオ']:<28} 稼働率={r['稼働率%']}%  Sharpe={r['Sharpe']}  "
                  f"平均保有={r['平均保有']}")

    # 分析③: ローテーションのCAGR影響
    print("\n  【分析③】強制ローテーションがCAGRに与える影響")
    for r in base_rows:
        if "BASE" in r["シナリオ"] or "③" in r["シナリオ"] or "FULL" in r["シナリオ"]:
            print(f"  {r['シナリオ']:<28} CAGR={r['CAGR%']}%  "
                  f"ローテ={r['ローテ/年']}/年  MaxDD={r['MaxDD%']}%")

    # 感度分析テーブル
    print("\n" + "=" * 90)
    print("  ▼ 感度分析（RSR閾値 × ローテ差分）")
    print("=" * 90)
    df_sens = pd.DataFrame(sensitivity_rows)
    print(df_sens.to_string(index=False))

    # 最適パラメータ特定（Calmar最大）
    best = max(sensitivity_rows, key=lambda r: float(r["Calmar"]))
    print(f"\n  ★ 最適パラメータ（Calmar最大）: {best['シナリオ']}")
    print(f"    CAGR={best['CAGR%']}%  Sharpe={best['Sharpe']}  "
          f"MaxDD={best['MaxDD%']}%  Calmar={best['Calmar']}")

    # JSON保存
    today    = datetime.date.today().strftime("%Y-%m-%d")
    out_path = f"results/capital_efficiency_{today}.json"
    os.makedirs("results", exist_ok=True)
    all_rows = base_rows + sensitivity_rows
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")


if __name__ == "__main__":
    main()
