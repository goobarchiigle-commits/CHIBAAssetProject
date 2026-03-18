"""
backtest/portfolio_entry_stop_v2.py
entry_stop改修版バックテスト

【改修内容】
1. entry_stop中もポジション維持（旧CBは全決済 → 新設計は保有継続）
2. entry_stop中の部分クローズ（追加DD閾値超過時）
3. 解除条件を擬似AND（DD回復 AND (日数経過 OR トレンドシグナル)）
4. 再エントリー時のポジションサイズ縮小（just_released期間）
5. クールダウン期間（解除後N日は再発動しない）
6. 詳細ログ出力（発動/解除/部分クローズ/DD推移）
7. 同一バー内の重複発注防止
8. バックテスト指標: MaxDD / CAGR / Sharpe / DD回復日数 / 発動回数 / 2x破綻検証

設計メモ:
  - entry_stop 発動閾値はデフォルト -10%（max_dd_limit=15% より早め）
  - 既存ポジションは損切りシグナルが出るまで保有継続
  - 部分クローズ: entry_stop発動時のDDから追加で additional_dd_limit 下落で起動
    重複防止のため last_partial_close_dd をトラッキング
  - 解除: dd_recovered AND (days_in_stop >= max_days OR trend_up)
    trend_up = 直近 trend_ma_short の equity 移動平均 > trend_ma_long の equity 移動平均
  - クールダウン中は entry_stop の再発動閾値チェックをスキップ
"""

from __future__ import annotations

import logging
import os
import sys
import json
import warnings
import datetime
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from backtest.universe_builder import download_universe
from backtest.rsr import calc_universe_rsr
from backtest.fujiko_strategy import FujikoStrategy
from backtest.mean_reversion_strategy import MeanReversionStrategy
from backtest.engine import TradeCost
from backtest.portfolio_engine import Position

# ─── ロガー設定 ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("entry_stop")

# ─── 定数 ────────────────────────────────────────────────────────────────────
START   = "2018-01-01"
END     = "2024-12-31"
CAPITAL = 2_000_000

SECTOR_STRATEGY: dict[str, str] = {
    "海運": "fujiko", "機械": "fujiko", "電機精密": "fujiko", "商社": "fujiko",
    "電機": "fujiko", "ゲーム": "fujiko", "レジャー": "fujiko", "食品": "fujiko",
    "ガス": "mean_rev", "鉄鋼": "mean_rev", "銀行": "mean_rev", "保険": "mean_rev",
    "輸送機器": "mean_rev", "化学": "mean_rev", "小売": "mean_rev",
    "サービス": "dynamic", "医薬品": "dynamic", "不動産": "dynamic",
    "情報通信": "dynamic", "陸運": "dynamic",
}

MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)


# ─── コンフィグ ───────────────────────────────────────────────────────────────
@dataclass
class EntryStopConfig:
    """全設定値を一元管理。変更時は必ずここを編集する。"""

    # ─── entry_stop 発動 ───────────────────────────────────────────
    entry_stop_dd: float = 0.10
    """entry_stop 発動 DD 閾値（0.10 = -10%）"""

    # ─── entry_stop 中の部分クローズ ──────────────────────────────
    additional_dd_limit: float = 0.05
    """entry_stop 発動後の追加 DD（0.05 = さらに -5%）で部分クローズ起動"""
    partial_close_ratio: float = 0.30
    """部分クローズ比率（0.30 = 各ポジションの30%を決済）"""

    # ─── entry_stop 解除条件 ──────────────────────────────────────
    entry_stop_release_ratio: float = 0.70
    """DD が entry_stop_dd * release_ratio まで回復したら dd_recovered=True"""
    entry_stop_max_days: int = 20
    """この日数経過で日数条件成立（dd_recovered AND (days OR trend)）"""

    # ─── トレンド判定（equity MA クロス） ────────────────────────
    trend_ma_short: int = 10
    """equity カーブの短期 MA 期間"""
    trend_ma_long: int = 30
    """equity カーブの長期 MA 期間"""

    # ─── 再エントリー制御 ─────────────────────────────────────────
    cooldown_days: int = 3
    """entry_stop 解除後、再発動しないクールダウン日数"""
    just_released_days: int = 5
    """解除後にポジションサイズを縮小する日数"""
    just_released_scale: float = 0.50
    """just_released 期間中のサイズスケーラー"""

    # ─── ポートフォリオ ───────────────────────────────────────────
    capital: float = 2_000_000
    max_positions: int = 3
    max_single_weight: float = 0.25
    """1銘柄最大ウェイト（資本比）"""

    # ─── コスト ───────────────────────────────────────────────────
    slippage_rate: float = 0.001
    commission_rate: float = 0.00055
    min_commission: float = 99.0


# ─── エンジン ─────────────────────────────────────────────────────────────────
class EntryStopEngine:
    """
    entry_stop 改修版ポートフォリオエンジン。

    先読みリーク防止:
      - シグナルは当日終値で判定
      - 約定は翌営業日始値（スリッページ込み）
    """

    def __init__(
        self,
        universe:   dict,
        strategies: dict,
        cfg:        EntryStopConfig = None,
        label:      str = "default",
        leverage:   float = 1.0,
    ) -> None:
        self.universe   = universe
        self.strategies = strategies
        self.cfg        = cfg or EntryStopConfig()
        self.label      = label
        self.leverage   = leverage  # 1.0 = 通常, 2.0 = 2倍レバレッジ検証

        # 共通取引日
        dates = None
        for info in universe.values():
            idx = info["df"].index
            dates = idx if dates is None else dates.intersection(idx)
        self.all_dates: pd.DatetimeIndex = dates.sort_values()

        self._cost = TradeCost(
            slippage_rate   = cfg.slippage_rate   if cfg else 0.001,
            commission_rate = cfg.commission_rate if cfg else 0.00055,
            min_commission  = cfg.min_commission  if cfg else 99.0,
        )

    # ─── ユーティリティ ──────────────────────────────────────────────
    def _alloc_base(self) -> float:
        """1ポジション基本配分（均等）"""
        return self.cfg.capital * self.leverage / self.cfg.max_positions

    def _close_position(
        self,
        sym: str,
        pos: Position,
        price: float,
        reason: str,
        date: pd.Timestamp,
        cash: float,
        trades: list,
        qty_override: int = None,
    ) -> tuple[float, int]:
        """
        ポジションを決済してキャッシュを返す。
        qty_override が指定された場合は部分クローズ。
        Returns: (new_cash, closed_qty)
        """
        close_qty   = qty_override if qty_override is not None else pos.qty
        close_qty   = min(close_qty, pos.qty)
        exec_price  = price * (1 - self._cost.slippage_rate)
        trade_value = close_qty * exec_price
        commission  = max(
            trade_value * self._cost.commission_rate,
            self._cost.min_commission,
        )
        pnl  = (exec_price - pos.entry_price) * close_qty - commission
        cash += trade_value - commission
        trades.append({
            "date":   date,
            "symbol": sym,
            "sector": pos.sector,
            "side":   f"SELL_{reason}",
            "qty":    close_qty,
            "price":  exec_price,
            "pnl":    pnl,
        })
        return cash, close_qty

    def _trend_up(self, equity_values: list[float]) -> bool:
        """
        equity カーブの短期 MA > 長期 MA → 上昇トレンド。
        データ不足の場合は False（保守的判定）。
        """
        n_long  = self.cfg.trend_ma_long
        n_short = self.cfg.trend_ma_short
        if len(equity_values) < n_long:
            return False
        short_ma = sum(equity_values[-n_short:]) / n_short
        long_ma  = sum(equity_values[-n_long:])  / n_long
        return short_ma > long_ma

    # ─── メインループ ────────────────────────────────────────────────
    def run(self) -> "EntryStopResult":
        cfg  = self.cfg
        cost = self._cost

        cash            = cfg.capital * self.leverage
        positions: dict[str, Position] = {}
        equity_records: list[dict]     = []
        trades:         list[dict]     = []
        peak_equity     = cash

        # entry_stop 状態管理
        entry_stop          = False
        entry_stop_date: Optional[pd.Timestamp] = None
        entry_stop_dd_val   = 0.0   # 発動時の DD 値
        last_partial_close_dd = 0.0  # 部分クローズ後のDD（重複防止）
        days_in_stop        = 0
        cooldown_remaining  = 0
        just_released_remaining = 0

        # メトリクス
        entry_stop_count    = 0
        entry_stop_events: list[dict] = []  # 各発動の詳細
        partial_close_count = 0

        equity_values: list[float] = []  # トレンド判定用

        dates = self.all_dates

        for i, date in enumerate(dates):

            # ─── 1. 時価評価 ────────────────────────────────────────
            mkt_value = sum(
                pos.market_value(self.universe[sym]["df"].loc[date, "Close"])
                for sym, pos in positions.items()
                if date in self.universe[sym]["df"].index
            )
            port_value  = cash + mkt_value
            peak_equity = max(peak_equity, port_value)
            current_dd  = (port_value - peak_equity) / peak_equity

            # ─── 2. クールダウンカウントダウン ──────────────────────
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # ─── 3. entry_stop 発動チェック ─────────────────────────
            if (
                not entry_stop
                and cooldown_remaining == 0
                and current_dd <= -cfg.entry_stop_dd
            ):
                entry_stop           = True
                entry_stop_date      = date
                entry_stop_dd_val    = current_dd
                last_partial_close_dd = current_dd  # 基準点をリセット
                days_in_stop         = 0
                entry_stop_count    += 1
                logger.info(
                    "📛 entry_stop 発動 | %s | DD=%.2f%% | 資産=¥%.0f",
                    date.date(), current_dd * 100, port_value,
                )
                entry_stop_events.append({
                    "activate_date": str(date.date()),
                    "activate_dd":   round(current_dd * 100, 2),
                    "release_date":  None,
                    "release_dd":    None,
                    "days_in_stop":  None,
                })

            # ─── 4. entry_stop 中: 部分クローズチェック ──────────────
            if entry_stop and positions and cfg.partial_close_ratio > 0:
                # 追加DD が additional_dd_limit を超えたら部分クローズ
                additional_dd = current_dd - last_partial_close_dd
                if additional_dd < -cfg.additional_dd_limit:
                    logger.info(
                        "⚠️  部分クローズ発動 | %s | 追加DD=%.2f%% | 対象=%d銘柄",
                        date.date(),
                        additional_dd * 100,
                        len(positions),
                    )
                    for sym in list(positions.keys()):
                        pos   = positions[sym]
                        price = self.universe[sym]["df"].loc[date, "Close"]
                        # 100株単位で30%を計算
                        close_qty = max(100, int(pos.qty * cfg.partial_close_ratio / 100) * 100)
                        close_qty = min(close_qty, pos.qty)  # 保有数を超えない

                        if close_qty >= pos.qty:
                            # 全決済
                            cash, _ = self._close_position(
                                sym, pos, price, "PARTIAL_FULL", date, cash, trades
                            )
                            del positions[sym]
                            logger.info(
                                "  全決済 %s @ ¥%.0f (部分クローズが100%%に丸め込まれた)",
                                sym, price,
                            )
                        else:
                            # 部分クローズ
                            cash, _ = self._close_position(
                                sym, pos, price, "PARTIAL", date, cash, trades,
                                qty_override=close_qty,
                            )
                            positions[sym].qty -= close_qty
                            logger.info(
                                "  部分クローズ %s %d株→残%d株 @ ¥%.0f",
                                sym, close_qty, positions[sym].qty, price,
                            )

                    last_partial_close_dd = current_dd  # 次の判定基準を更新
                    partial_close_count  += 1

            # ─── 5. entry_stop 解除チェック ─────────────────────────
            if entry_stop:
                days_in_stop += 1

                dd_recovered  = current_dd > -(cfg.entry_stop_dd * cfg.entry_stop_release_ratio)
                days_ok       = days_in_stop >= cfg.entry_stop_max_days
                trend_ok      = self._trend_up(equity_values)

                release = dd_recovered and (days_ok or trend_ok)
                if release:
                    logger.info(
                        "✅ entry_stop 解除 | %s | DD=%.2f%% | 在停日数=%d | "
                        "dd_rec=%s days=%s trend=%s",
                        date.date(), current_dd * 100, days_in_stop,
                        dd_recovered, days_ok, trend_ok,
                    )
                    # イベント記録を更新
                    if entry_stop_events:
                        entry_stop_events[-1].update({
                            "release_date": str(date.date()),
                            "release_dd":   round(current_dd * 100, 2),
                            "days_in_stop": days_in_stop,
                        })
                    entry_stop              = False
                    cooldown_remaining      = cfg.cooldown_days
                    just_released_remaining = cfg.just_released_days
                    days_in_stop            = 0

            # ─── 6. 損切りチェック（entry_stop 中も実行） ───────────
            for sym in list(positions.keys()):
                df_sym = self.universe[sym]["df"]
                past   = df_sym.loc[:date]
                signal = self.strategies[sym].generate_signal(past)
                if signal == -1:
                    price = df_sym.loc[date, "Close"]
                    cash, _ = self._close_position(
                        sym, positions[sym], price, "STOP", date, cash, trades
                    )
                    del positions[sym]
                    logger.debug("損切り %s @ %s", sym, date.date())

            # ─── 7. エントリーチェック ──────────────────────────────
            if not entry_stop and i + 1 < len(dates):
                next_date = dates[i + 1]

                # just_released: サイズを縮小
                size_scale = 1.0
                if just_released_remaining > 0:
                    size_scale = cfg.just_released_scale
                    just_released_remaining -= 1

                # 同一バー内の重複発注防止
                pending_orders: set[str] = set()

                for sym, info in self.universe.items():
                    if sym in positions:
                        continue
                    if len(positions) >= cfg.max_positions:
                        break
                    if sym in pending_orders:
                        continue

                    df_sym = info["df"]
                    if date not in df_sym.index or next_date not in df_sym.index:
                        continue

                    past   = df_sym.loc[:date]
                    signal = self.strategies[sym].generate_signal(past)
                    if signal != 1:
                        continue

                    # 翌日始値で約定（先読みリーク防止）
                    exec_price = df_sym.loc[next_date, "Open"] * (1 + cost.slippage_rate)

                    alloc = self._alloc_base() * size_scale
                    # 1銘柄ウェイト上限
                    alloc = min(alloc, cfg.capital * self.leverage * cfg.max_single_weight)
                    qty   = int(alloc // (exec_price * 100)) * 100
                    if qty <= 0:
                        continue

                    total_cost = qty * exec_price + max(
                        qty * exec_price * cost.commission_rate,
                        cost.min_commission,
                    )
                    if total_cost > cash:
                        continue

                    cash -= total_cost
                    positions[sym] = Position(
                        symbol      = sym,
                        sector      = info["sector"],
                        qty         = qty,
                        entry_price = exec_price,
                        entry_date  = next_date,
                    )
                    trades.append({
                        "date":   next_date,
                        "symbol": sym,
                        "sector": info["sector"],
                        "side":   "BUY",
                        "qty":    qty,
                        "price":  exec_price,
                        "pnl":    0.0,
                    })
                    pending_orders.add(sym)
                    if size_scale < 1.0:
                        logger.info(
                            "📉 縮小エントリー %s %d株 (x%.1f) @ %s",
                            sym, qty, size_scale, next_date.date(),
                        )

            # ─── 8. 記録 ────────────────────────────────────────────
            equity_values.append(port_value)
            equity_records.append({
                "date":              date,
                "value":             port_value,
                "dd":                current_dd,
                "n_positions":       len(positions),
                "entry_stop":        entry_stop,
                "cooldown":          cooldown_remaining > 0,
                "just_released":     just_released_remaining > 0,
                "partial_close_cnt": partial_close_count,
            })

            # DD 警告ログ（1%刻みで変化があったとき）
            if i > 0 and abs(current_dd) > 0.05:
                prev_dd = equity_records[-2]["dd"] if len(equity_records) >= 2 else 0.0
                if int(current_dd * 100) != int(prev_dd * 100):
                    logger.debug(
                        "DD推移 %s: %.1f%%", date.date(), current_dd * 100
                    )

        # ─── 最終記録（未クローズのイベント） ──────────────────────
        if entry_stop and entry_stop_events and entry_stop_events[-1]["release_date"] is None:
            entry_stop_events[-1].update({
                "release_date": "N/A (未解除)",
                "release_dd":   round(current_dd * 100, 2),
                "days_in_stop": days_in_stop,
            })

        rec_df    = pd.DataFrame(equity_records).set_index("date")
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=["date", "symbol", "sector", "side", "qty", "price", "pnl"]
        )
        return EntryStopResult(
            equity_curve        = rec_df["value"],
            dd_series           = rec_df["dd"],
            n_positions         = rec_df["n_positions"],
            entry_stop_series   = rec_df["entry_stop"],
            trades              = trades_df,
            initial_capital     = cfg.capital * self.leverage,
            label               = self.label,
            entry_stop_count    = entry_stop_count,
            partial_close_count = partial_close_count,
            entry_stop_events   = entry_stop_events,
        )


# ─── 結果クラス ──────────────────────────────────────────────────────────────
@dataclass
class EntryStopResult:
    equity_curve:        pd.Series
    dd_series:           pd.Series
    n_positions:         pd.Series
    entry_stop_series:   pd.Series
    trades:              pd.DataFrame
    initial_capital:     float
    label:               str
    entry_stop_count:    int
    partial_close_count: int
    entry_stop_events:   list

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
    def n_trades(self) -> int:
        if self.trades.empty:
            return 0
        return int(self.trades["side"].str.startswith("SELL").sum())

    @property
    def win_rate(self) -> float:
        sells = self.trades[self.trades["side"].str.startswith("SELL")]
        if sells.empty:
            return 0.0
        return float((sells["pnl"] > 0).sum() / len(sells))

    @property
    def dd_recovery_days(self) -> list[int]:
        """各 entry_stop イベントの在停日数リスト"""
        return [
            e["days_in_stop"]
            for e in self.entry_stop_events
            if isinstance(e.get("days_in_stop"), int)
        ]

    @property
    def bankrupt(self) -> bool:
        """equity が初期資本の 10% 以下に到達したら破綻とみなす"""
        return bool((self.equity_curve < self.initial_capital * 0.10).any())

    def summary(self) -> None:
        w = 60
        print("=" * w)
        print(f"  {self.label}")
        print("-" * w)
        print(f"  CAGR        : {self.cagr * 100:>+8.2f}%")
        print(f"  MaxDD       : {self.max_drawdown * 100:>+8.2f}%")
        print(f"  Calmar      : {self.calmar:>10.3f}")
        print(f"  Sharpe      : {self.sharpe:>10.3f}")
        print(f"  総リターン   : {self.total_return * 100:>+8.2f}%")
        print(f"  決済回数     : {self.n_trades:>10}")
        print(f"  勝率         : {self.win_rate * 100:>9.1f}%")
        print("-" * w)
        print(f"  entry_stop発動: {self.entry_stop_count:>7} 回")
        print(f"  部分クローズ   : {self.partial_close_count:>7} 回")
        rec = self.dd_recovery_days
        if rec:
            print(f"  DD回復日数    : 平均={sum(rec)/len(rec):.1f}日  "
                  f"最大={max(rec)}日  min={min(rec)}日")
        print(f"  破綻           : {'YES ⚠️' if self.bankrupt else 'NO ✓'}")
        print("=" * w)
        if self.entry_stop_events:
            print("  ─── entry_stop イベント詳細 ───")
            for ev in self.entry_stop_events:
                print(f"    発動:{ev['activate_date']} DD={ev['activate_dd']}%"
                      f"  → 解除:{ev['release_date']} DD={ev['release_dd']}%"
                      f"  在停={ev['days_in_stop']}日")


# ─── ヘルパー ─────────────────────────────────────────────────────────────────
def build_strategies(
    universe: dict,
    rsr_uni: pd.DataFrame,
    sym_to_strat: dict,
) -> dict:
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


def metrics_row(res: EntryStopResult) -> dict:
    rec = res.dd_recovery_days
    avg_rec = round(sum(rec) / len(rec), 1) if rec else 0
    return {
        "シナリオ":        res.label,
        "CAGR%":          f"{res.cagr*100:+.2f}",
        "MaxDD%":         f"{res.max_drawdown*100:.2f}",
        "Calmar":         f"{res.calmar:.3f}",
        "Sharpe":         f"{res.sharpe:.3f}",
        "勝率%":          f"{res.win_rate*100:.1f}",
        "entry_stop回数": res.entry_stop_count,
        "部分クローズ回数": res.partial_close_count,
        "DD回復日数(平均)": avg_rec,
        "2x破綻":         "YES" if res.bankrupt else "NO",
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("  portfolio_entry_stop_v2.py — entry_stop 改修版バックテスト")
    print(f"  期間: {START} 〜 {END}  /  初期資本: ¥{CAPITAL:,}")
    print("=" * 72)

    # ── 1. データ取得 ───────────────────────────────────────────────
    df_sel       = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
    mask_27      = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27   = {r["symbol"]: r["sector"] for _, r in df_sel[mask_27].iterrows()}

    print(f"\n[1/4] データ取得中（{len(tickers_27)}銘柄）...")
    universe = download_universe(tickers_27, start=START, end=END, verbose=False)
    print(f"  取得完了: {len(universe)} 銘柄")

    # ── 2. RSR 計算 ─────────────────────────────────────────────────
    print("\n[2/4] RSR 計算中...")
    prices = {sym: info["df"]["Close"] for sym, info in universe.items()}
    rsr_uni = calc_universe_rsr(prices)
    strategies = build_strategies(universe, rsr_uni, sym_to_strat)

    # ── 3. バックテスト実行（4シナリオ） ─────────────────────────
    print("\n[3/4] バックテスト実行（4シナリオ）...")

    # 各シナリオのコンフィグを定義
    scenarios: list[tuple[str, EntryStopConfig, float]] = [
        # (label, config, leverage)
        (
            "現行CB（-15%全決済・比較用）",
            EntryStopConfig(
                entry_stop_dd=0.15,
                additional_dd_limit=0.99,  # 実質無効化（全決済との比較）
                partial_close_ratio=0.0,
                entry_stop_release_ratio=0.5,
                entry_stop_max_days=9999,
                cooldown_days=0,
                just_released_days=0,
            ),
            1.0,
        ),
        (
            "新entry_stop（-10%停止/部分クローズ）",
            EntryStopConfig(),  # デフォルト設定
            1.0,
        ),
        (
            "新entry_stop（-8%早期停止）",
            EntryStopConfig(
                entry_stop_dd=0.08,
                additional_dd_limit=0.04,
                entry_stop_release_ratio=0.60,
                entry_stop_max_days=15,
            ),
            1.0,
        ),
        (
            "新entry_stop x2レバレッジ（破綻検証）",
            EntryStopConfig(),
            2.0,
        ),
        # ─── ロジック動作確認用: 閾値を -2% に設定 ─────────────────
        # NOTE: 実運用パラメータではない。entry_stop が発動することを確認するためのデバッグシナリオ。
        (
            "【動作確認】-2%閾値（ロジック検証用）",
            EntryStopConfig(
                entry_stop_dd=0.02,
                additional_dd_limit=0.01,
                partial_close_ratio=0.30,
                entry_stop_release_ratio=0.50,
                entry_stop_max_days=5,
                trend_ma_short=5,
                trend_ma_long=10,
                cooldown_days=3,
                just_released_days=3,
                just_released_scale=0.50,
            ),
            1.0,
        ),
    ]

    results: list[EntryStopResult] = []
    for label, cfg, leverage in scenarios:
        print(f"  [{label}]...", end=" ", flush=True)
        eng = EntryStopEngine(
            universe=universe, strategies=strategies,
            cfg=cfg, label=label, leverage=leverage,
        )
        res = eng.run()
        results.append(res)
        print(f"CAGR={res.cagr*100:+.2f}%  MaxDD={res.max_drawdown*100:.2f}%  "
              f"Calmar={res.calmar:.3f}  ES発動={res.entry_stop_count}回  "
              f"破綻={'YES⚠️' if res.bankrupt else 'NO✓'}")

    # ── 4. 結果表示 ─────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  比較サマリー")
    print("=" * 72)
    rows = [metrics_row(r) for r in results]
    df_cmp = pd.DataFrame(rows)
    print(df_cmp.to_string(index=False))

    print("\n  ─── 詳細サマリー ───")
    for res in results:
        print()
        res.summary()

    # ── 5. JSON 保存 ────────────────────────────────────────────────
    today    = datetime.date.today().strftime("%Y-%m-%d")
    out_path = f"results/entry_stop_v2_{today}.json"
    os.makedirs("results", exist_ok=True)

    output = {
        "run_date":  today,
        "period":    f"{START} ~ {END}",
        "capital":   CAPITAL,
        "scenarios": [],
    }
    for res in results:
        output["scenarios"].append({
            "label":               res.label,
            "cagr_pct":            round(res.cagr * 100, 2),
            "max_dd_pct":          round(res.max_drawdown * 100, 2),
            "calmar":              round(res.calmar, 3),
            "sharpe":              round(res.sharpe, 3),
            "win_rate_pct":        round(res.win_rate * 100, 1),
            "n_trades":            res.n_trades,
            "entry_stop_count":    res.entry_stop_count,
            "partial_close_count": res.partial_close_count,
            "dd_recovery_days":    res.dd_recovery_days,
            "bankrupt_2x":         res.bankrupt,
            "entry_stop_events":   res.entry_stop_events,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print("完了。")


if __name__ == "__main__":
    main()
