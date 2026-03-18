"""
backtest/portfolio_entry_stop_v3.py
entry_stop v3 — 回復判定再定義 + ロックアウト根絶版

【v2からの改善点】
1. dd_recovered 定義を「局所回復率」に変更（ピーク比ではなく底値からの回復率）
   recovery_ratio = (current_equity - bottom_equity) / (peak_equity - bottom_equity)
   → entry_stop発動後の部分クローズ・キャッシュ化と構造的に整合

2. ハードタイムアウト（永久ロック根絶）
   release = (dd_recovered AND (days OR trend)) OR (days >= hard_timeout)
   → hard_timeout日経過で無条件解除

3. entry_stop中の限定プローブエントリー
   条件: trend_up AND days_in_stop >= probe_min_days
   サイズ: 通常の probe_scale（デフォルト25%）
   → equity更新の機会を作り dd_recovered を前進させる

4. 部分クローズのやりすぎ防止
   cash_ratio > max_cash_ratio_for_partial → 部分クローズを停止
   → キャッシュ60%超では保護せず「回復に賭ける」モードへ

5. レバレッジ連動閾値
   leverage >= 2.0 時に entry_stop_dd *= leverage_dd_scale
   → 2x環境では自動的に閾値が厳しくなり「同じDD水準で先に発動」
"""

from __future__ import annotations

import logging
import os
import sys
import json
import warnings
import datetime
from dataclasses import dataclass
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

# ─── ロガー ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("es_v3")

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
class ESConfig:
    """entry_stop v3 全設定。configs/strategy.yaml の entry_stop セクションと対応。"""

    # ─── entry_stop 発動 ───────────────────────────────────────────
    entry_stop_dd: float = 0.06
    """発動閾値。leverage>=2.0 時は自動で *= leverage_dd_scale される。"""

    # ─── 部分クローズ ──────────────────────────────────────────────
    additional_dd_limit: float = 0.03
    """entry_stop 発動後にさらに何% 下落したら部分クローズ（デフォルト3%）"""
    partial_close_ratio: float = 0.30
    """部分クローズ比率"""
    max_cash_ratio_for_partial: float = 0.60
    """【改善4】キャッシュがこの比率を超えたら部分クローズ停止（やりすぎ防止）"""

    # ─── dd_recovered 定義（【改善1】局所回復率） ─────────────────
    recovery_mode: str = "local"
    """'local'=局所回復率(v3推奨) / 'peak'=ピーク比DD(v2旧方式)"""
    local_recovery_threshold: float = 0.30
    """local モード: (current-bottom)/(peak-bottom) > この値で dd_recovered=True"""
    peak_recovery_ratio: float = 0.70
    """peak モード(互換): DD < entry_stop_dd * peak_recovery_ratio で dd_recovered=True"""

    # ─── 解除条件 ─────────────────────────────────────────────────
    entry_stop_max_days: int = 20
    """(dd_recovered AND days) のうちの日数条件"""
    hard_timeout_days: int = 45
    """【改善2】無条件解除（永久ロック根絶）。0 で無効化。"""

    # ─── トレンド判定（equity MA クロス） ────────────────────────
    trend_ma_short: int = 10
    trend_ma_long: int = 30

    # ─── プローブエントリー（【改善3】） ─────────────────────────
    probe_entry_enabled: bool = True
    """entry_stop中 trend_up のとき小サイズ再エントリーを許可"""
    probe_entry_min_days: int = 5
    """entry_stop 発動後何日経過でプローブ許可"""
    probe_entry_scale: float = 0.25
    """プローブエントリーのサイズ（通常の25%）"""

    # ─── 再エントリー制御 ─────────────────────────────────────────
    cooldown_days: int = 3
    just_released_days: int = 5
    just_released_scale: float = 0.50

    # ─── レバレッジ連動（【改善5】） ──────────────────────────────
    leverage_dd_scale: float = 0.70
    """leverage >= 2.0 時に entry_stop_dd に掛ける係数"""

    # ─── ポートフォリオ ───────────────────────────────────────────
    capital: float = 2_000_000
    max_positions: int = 3
    max_single_weight: float = 0.25
    slippage_rate: float = 0.001
    commission_rate: float = 0.00055
    min_commission: float = 99.0


# ─── エンジン ─────────────────────────────────────────────────────────────────
class ESEngineV3:
    """entry_stop v3 エンジン。5つの改善を統合。"""

    def __init__(
        self,
        universe:   dict,
        strategies: dict,
        cfg:        ESConfig = None,
        label:      str = "",
        leverage:   float = 1.0,
    ) -> None:
        self.universe   = universe
        self.strategies = strategies
        self.cfg        = cfg or ESConfig()
        self.label      = label
        self.leverage   = leverage

        # 【改善5】レバレッジ連動閾値
        self._eff_dd = self.cfg.entry_stop_dd
        if leverage >= 2.0:
            self._eff_dd *= self.cfg.leverage_dd_scale

        dates = None
        for info in universe.values():
            idx = info["df"].index
            dates = idx if dates is None else dates.intersection(idx)
        self.all_dates: pd.DatetimeIndex = dates.sort_values()

        self._cost = TradeCost(
            slippage_rate   = cfg.slippage_rate,
            commission_rate = cfg.commission_rate,
            min_commission  = cfg.min_commission,
        )

    # ─── ユーティリティ ──────────────────────────────────────────────
    def _alloc(self, scale: float = 1.0) -> float:
        base = self.cfg.capital * self.leverage / self.cfg.max_positions
        return min(base * scale,
                   self.cfg.capital * self.leverage * self.cfg.max_single_weight)

    def _close_pos(
        self, sym: str, pos: Position, price: float,
        reason: str, date, cash: float, trades: list,
        qty_override: int = None,
    ) -> tuple[float, int]:
        qty         = min(qty_override or pos.qty, pos.qty)
        exec_price  = price * (1 - self._cost.slippage_rate)
        trade_val   = qty * exec_price
        commission  = max(trade_val * self._cost.commission_rate, self._cost.min_commission)
        pnl         = (exec_price - pos.entry_price) * qty - commission
        cash       += trade_val - commission
        trades.append(dict(date=date, symbol=sym, sector=pos.sector,
                           side=f"SELL_{reason}", qty=qty, price=exec_price, pnl=pnl))
        return cash, qty

    def _trend_up(self, eq_vals: list[float]) -> bool:
        n_l = self.cfg.trend_ma_long
        n_s = self.cfg.trend_ma_short
        if len(eq_vals) < n_l:
            return False
        return (sum(eq_vals[-n_s:]) / n_s) > (sum(eq_vals[-n_l:]) / n_l)

    def _dd_recovered(
        self,
        current: float,
        peak:    float,
        bottom:  float,
    ) -> bool:
        """
        【改善1】回復判定。

        local モード（推奨）:
          recovery_ratio = (current - bottom) / (peak - bottom)
          → 底値からどれだけ回復したかを測定
          → entry_stop発動・部分クローズ後のキャッシュ化と整合する

        peak モード（旧 v2 互換）:
          current_dd = (current - peak) / peak
          dd_recovered = current_dd > -(entry_stop_dd * peak_recovery_ratio)
        """
        cfg = self.cfg
        if cfg.recovery_mode == "local":
            span = peak - bottom
            if span <= 0:
                return current >= peak  # ピーク回復済み
            ratio = (current - bottom) / span
            return ratio >= cfg.local_recovery_threshold
        else:  # "peak"
            dd = (current - peak) / peak
            return dd > -(self._eff_dd * cfg.peak_recovery_ratio)

    # ─── メインループ ────────────────────────────────────────────────
    def run(self) -> "ESResultV3":
        cfg  = self.cfg
        cost = self._cost

        cash              = cfg.capital * self.leverage
        positions: dict[str, Position] = {}
        equity_records    = []
        trades: list[dict] = []
        peak_equity       = cash
        bottom_since_stop = cash  # entry_stop 発動後の最低 equity

        # 状態
        entry_stop          = False
        entry_stop_date: Optional[pd.Timestamp] = None
        last_partial_dd     = 0.0
        days_in_stop        = 0
        cooldown_remaining  = 0
        just_released_rem   = 0

        # メトリクス
        es_count            = 0
        partial_count       = 0
        es_events: list[dict] = []
        eq_vals: list[float]  = []

        dates = self.all_dates

        for i, date in enumerate(dates):

            # ─── 1. 時価評価 ─────────────────────────────────────────
            mkt = sum(
                pos.market_value(self.universe[s]["df"].loc[date, "Close"])
                for s, pos in positions.items()
                if date in self.universe[s]["df"].index
            )
            port_val    = cash + mkt
            peak_equity = max(peak_equity, port_val)
            current_dd  = (port_val - peak_equity) / peak_equity

            if entry_stop:
                bottom_since_stop = min(bottom_since_stop, port_val)

            # ─── 2. クールダウン ─────────────────────────────────────
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # ─── 3. entry_stop 発動 ──────────────────────────────────
            if not entry_stop and cooldown_remaining == 0:
                if current_dd <= -self._eff_dd:
                    entry_stop        = True
                    entry_stop_date   = date
                    last_partial_dd   = current_dd
                    bottom_since_stop = port_val
                    days_in_stop      = 0
                    es_count         += 1
                    logger.info(
                        "📛 entry_stop発動 | %s | DD=%.2f%% | 実効閾値=%.1f%% | "
                        "資産=¥%.0f | leverage=%.1fx",
                        date.date(), current_dd * 100,
                        self._eff_dd * 100, port_val, self.leverage,
                    )
                    es_events.append(dict(
                        activate_date = str(date.date()),
                        activate_dd   = round(current_dd * 100, 2),
                        release_date  = None, release_dd=None,
                        days_in_stop  = None, release_reason=None,
                    ))

            # ─── 4. 部分クローズ（【改善4】cash_ratio 上限チェック付き） ─
            if entry_stop and positions and cfg.partial_close_ratio > 0:
                cash_ratio     = cash / port_val if port_val > 0 else 1.0
                allow_partial  = cash_ratio <= cfg.max_cash_ratio_for_partial
                add_dd         = current_dd - last_partial_dd

                if allow_partial and add_dd < -cfg.additional_dd_limit:
                    logger.info(
                        "⚠️  部分クローズ | %s | 追加DD=%.2f%% | cash比率=%.1f%% | %d銘柄",
                        date.date(), add_dd * 100, cash_ratio * 100, len(positions),
                    )
                    for sym in list(positions.keys()):
                        pos   = positions[sym]
                        price = self.universe[sym]["df"].loc[date, "Close"]
                        cq    = max(100, int(pos.qty * cfg.partial_close_ratio / 100) * 100)
                        cq    = min(cq, pos.qty)
                        if cq >= pos.qty:
                            cash, _ = self._close_pos(sym, pos, price, "PARTIAL_FULL",
                                                      date, cash, trades)
                            del positions[sym]
                            logger.info("  全決済 %s @ ¥%.0f", sym, price)
                        else:
                            cash, _ = self._close_pos(sym, pos, price, "PARTIAL",
                                                      date, cash, trades, cq)
                            positions[sym].qty -= cq
                            logger.info("  部分 %s -%d株(残%d) @ ¥%.0f",
                                        sym, cq, positions[sym].qty, price)
                    last_partial_dd = current_dd
                    partial_count  += 1

                elif not allow_partial and add_dd < -cfg.additional_dd_limit:
                    logger.info(
                        "⚡ 部分クローズ抑制（cash比率 %.1f%% > 上限 %.0f%%）",
                        cash_ratio * 100, cfg.max_cash_ratio_for_partial * 100,
                    )

            # ─── 5. entry_stop 解除チェック ─────────────────────────
            if entry_stop:
                days_in_stop += 1
                trend_ok  = self._trend_up(eq_vals)
                rec_ok    = self._dd_recovered(port_val, peak_equity, bottom_since_stop)
                days_ok   = days_in_stop >= cfg.entry_stop_max_days
                hard_ok   = (cfg.hard_timeout_days > 0
                             and days_in_stop >= cfg.hard_timeout_days)

                # 【改善2】release = (rec_ok AND (days OR trend)) OR hard_timeout
                release = (rec_ok and (days_ok or trend_ok)) or hard_ok
                reason  = ("hard_timeout"  if hard_ok
                           else "rec+days" if (rec_ok and days_ok)
                           else "rec+trend" if (rec_ok and trend_ok)
                           else None)

                if release:
                    logger.info(
                        "✅ entry_stop解除 | %s | DD=%.2f%% | 在停=%d日 | "
                        "理由=%s | rec=%.3f",
                        date.date(), current_dd * 100, days_in_stop,
                        reason,
                        (port_val - bottom_since_stop) / max(peak_equity - bottom_since_stop, 1),
                    )
                    if es_events:
                        es_events[-1].update(dict(
                            release_date   = str(date.date()),
                            release_dd     = round(current_dd * 100, 2),
                            days_in_stop   = days_in_stop,
                            release_reason = reason,
                        ))
                    entry_stop              = False
                    cooldown_remaining      = cfg.cooldown_days
                    just_released_rem       = cfg.just_released_days
                    days_in_stop            = 0
                    bottom_since_stop       = port_val  # リセット

            # ─── 6. 損切り（entry_stop 中も実行） ────────────────────
            for sym in list(positions.keys()):
                df_sym = self.universe[sym]["df"]
                sig    = self.strategies[sym].generate_signal(df_sym.loc[:date])
                if sig == -1:
                    price = df_sym.loc[date, "Close"]
                    cash, _ = self._close_pos(sym, positions[sym], price, "STOP",
                                              date, cash, trades)
                    del positions[sym]
                    logger.debug("損切 %s @ %s DD=%.2f%%",
                                 sym, date.date(), current_dd * 100)

            # ─── 7. エントリー（通常 + プローブ） ────────────────────
            if i + 1 < len(dates):
                next_date  = dates[i + 1]
                trend_ok   = self._trend_up(eq_vals)

                # 通常エントリー条件
                normal_ok  = not entry_stop
                # 【改善3】プローブエントリー条件
                probe_ok   = (
                    entry_stop
                    and cfg.probe_entry_enabled
                    and trend_ok
                    and days_in_stop >= cfg.probe_entry_min_days
                )

                if normal_ok or probe_ok:
                    scale = (cfg.just_released_scale if just_released_rem > 0
                             else cfg.probe_entry_scale if probe_ok
                             else 1.0)
                    if just_released_rem > 0:
                        just_released_rem -= 1

                    pending: set[str] = set()
                    for sym, info in self.universe.items():
                        if sym in positions or sym in pending:
                            continue
                        if len(positions) >= cfg.max_positions:
                            break

                        df_sym = info["df"]
                        if date not in df_sym.index or next_date not in df_sym.index:
                            continue

                        sig = self.strategies[sym].generate_signal(df_sym.loc[:date])
                        if sig != 1:
                            continue

                        exec_p = df_sym.loc[next_date, "Open"] * (1 + cost.slippage_rate)
                        alloc  = self._alloc(scale)
                        qty    = int(alloc // (exec_p * 100)) * 100
                        if qty <= 0:
                            continue

                        total_c = qty * exec_p + max(
                            qty * exec_p * cost.commission_rate, cost.min_commission)
                        if total_c > cash:
                            continue

                        cash -= total_c
                        positions[sym] = Position(
                            symbol=sym, sector=info["sector"],
                            qty=qty, entry_price=exec_p, entry_date=next_date,
                        )
                        trades.append(dict(
                            date=next_date, symbol=sym, sector=info["sector"],
                            side="BUY_PROBE" if probe_ok else "BUY",
                            qty=qty, price=exec_p, pnl=0.0,
                        ))
                        pending.add(sym)
                        if probe_ok:
                            logger.info("🔍 プローブエントリー %s %d株(x%.2f) @ %s",
                                        sym, qty, scale, next_date.date())

            # ─── 8. 記録 ──────────────────────────────────────────────
            eq_vals.append(port_val)
            equity_records.append(dict(
                date         = date,
                value        = port_val,
                dd           = current_dd,
                n_positions  = len(positions),
                entry_stop   = entry_stop,
                cash         = cash,
                cash_ratio   = cash / port_val if port_val > 0 else 1.0,
                bottom_ratio = ((port_val - bottom_since_stop) /
                                max(peak_equity - bottom_since_stop, 1))
                               if entry_stop else 0.0,
            ))

        # 未解除イベントの後処理
        if entry_stop and es_events and es_events[-1]["release_date"] is None:
            es_events[-1].update(dict(
                release_date   = "N/A(未解除)",
                release_dd     = round(current_dd * 100, 2),
                days_in_stop   = days_in_stop,
                release_reason = "未解除",
            ))

        rec_df    = pd.DataFrame(equity_records).set_index("date")
        trades_df = (pd.DataFrame(trades) if trades
                     else pd.DataFrame(columns=["date","symbol","sector",
                                                "side","qty","price","pnl"]))
        return ESResultV3(
            equity_curve        = rec_df["value"],
            dd_series           = rec_df["dd"],
            n_positions         = rec_df["n_positions"],
            entry_stop_series   = rec_df["entry_stop"],
            cash_ratio_series   = rec_df["cash_ratio"],
            trades              = trades_df,
            initial_capital     = cfg.capital * self.leverage,
            label               = self.label,
            eff_dd_threshold    = self._eff_dd,
            es_count            = es_count,
            partial_count       = partial_count,
            es_events           = es_events,
        )


# ─── 結果 ─────────────────────────────────────────────────────────────────────
@dataclass
class ESResultV3:
    equity_curve:      pd.Series
    dd_series:         pd.Series
    n_positions:       pd.Series
    entry_stop_series: pd.Series
    cash_ratio_series: pd.Series
    trades:            pd.DataFrame
    initial_capital:   float
    label:             str
    eff_dd_threshold:  float
    es_count:          int
    partial_count:     int
    es_events:         list

    @property
    def total_return(self) -> float:
        return self.equity_curve.iloc[-1] / self.initial_capital - 1.0

    @property
    def cagr(self) -> float:
        y = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days / 365.25
        return (1 + self.total_return) ** (1 / max(y, 0.01)) - 1

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
    def win_rate(self) -> float:
        sells = self.trades[self.trades["side"].str.startswith("SELL")]
        return float((sells["pnl"] > 0).sum() / len(sells)) if len(sells) > 0 else 0.0

    @property
    def n_trades(self) -> int:
        return int(self.trades["side"].str.startswith("SELL").sum())

    @property
    def avg_lock_days(self) -> float:
        days = [e["days_in_stop"] for e in self.es_events
                if isinstance(e.get("days_in_stop"), int)]
        return sum(days) / len(days) if days else 0.0

    @property
    def max_lock_days(self) -> int:
        days = [e["days_in_stop"] for e in self.es_events
                if isinstance(e.get("days_in_stop"), int)]
        return max(days) if days else 0

    @property
    def bankrupt(self) -> bool:
        return bool((self.equity_curve < self.initial_capital * 0.10).any())

    def summary(self) -> None:
        w = 62
        print("=" * w)
        print(f"  {self.label}")
        print(f"  実効DD閾値: -{self.eff_dd_threshold*100:.1f}%")
        print("-" * w)
        print(f"  CAGR      : {self.cagr*100:>+8.2f}%")
        print(f"  MaxDD     : {self.max_drawdown*100:>+8.2f}%")
        print(f"  Calmar    : {self.calmar:>10.3f}")
        print(f"  Sharpe    : {self.sharpe:>10.3f}")
        print(f"  総リターン : {self.total_return*100:>+8.2f}%")
        print(f"  決済回数   : {self.n_trades:>10}")
        print(f"  勝率       : {self.win_rate*100:>9.1f}%")
        print("-" * w)
        print(f"  ES発動     : {self.es_count:>7} 回")
        print(f"  部分クローズ: {self.partial_count:>6} 回")
        print(f"  ロック日数  : avg={self.avg_lock_days:.1f}日 / max={self.max_lock_days}日")
        print(f"  破綻        : {'YES ⚠️' if self.bankrupt else 'NO  ✓'}")
        print("=" * w)
        if self.es_events:
            print("  ─── ES イベント ───")
            for ev in self.es_events:
                print(f"    {ev['activate_date']} DD={ev['activate_dd']:.1f}%"
                      f" → {ev['release_date']} DD={ev.get('release_dd','?'):.1f}%"
                      f" [{ev['days_in_stop']}日 / {ev.get('release_reason','-')}]")


# ─── ヘルパー ─────────────────────────────────────────────────────────────────
def build_strategies(universe: dict, rsr_uni: pd.DataFrame, sym_to_strat: dict) -> dict:
    strats = {}
    for sym, info in universe.items():
        rsr_s  = rsr_uni[sym] if sym in rsr_uni.columns else None
        rule   = SECTOR_STRATEGY.get(info["sector"], "dynamic")
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


def metrics_row(res: ESResultV3) -> dict:
    return {
        "シナリオ":          res.label,
        "CAGR%":            f"{res.cagr*100:+.2f}",
        "MaxDD%":           f"{res.max_drawdown*100:.2f}",
        "Calmar":           f"{res.calmar:.3f}",
        "Sharpe":           f"{res.sharpe:.3f}",
        "勝率%":            f"{res.win_rate*100:.1f}",
        "ES発動":           res.es_count,
        "部分CL":           res.partial_count,
        "avg_lock日":       f"{res.avg_lock_days:.1f}",
        "max_lock日":       res.max_lock_days,
        "2x破綻":           "YES" if res.bankrupt else "NO",
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("  portfolio_entry_stop_v3.py — 回復判定再定義 + ロックアウト根絶版")
    print(f"  期間: {START}〜{END}  /  資本: ¥{CAPITAL:,}")
    print("=" * 72)

    # ── データ取得 ──────────────────────────────────────────────────
    df_sel       = pd.read_csv("data/dynamic_selection.csv", encoding="utf-8-sig")
    sym_to_strat = dict(zip(df_sel["symbol"], df_sel["strategy"]))
    mask_27      = (df_sel["sharpe"] > 0.30) & (df_sel["maxdd"].abs() < 0.30)
    tickers_27   = {r["symbol"]: r["sector"] for _, r in df_sel[mask_27].iterrows()}

    print(f"\n[1/4] データ取得（{len(tickers_27)}銘柄）...")
    universe = download_universe(tickers_27, start=START, end=END, verbose=False)
    print(f"  完了: {len(universe)} 銘柄")

    print("\n[2/4] RSR 計算...")
    prices    = {s: info["df"]["Close"] for s, info in universe.items()}
    rsr_uni   = calc_universe_rsr(prices)
    strats    = build_strategies(universe, rsr_uni, sym_to_strat)

    # ── シナリオ定義 ────────────────────────────────────────────────
    # ① v2旧方式（比較ベースライン）
    cfg_v2_compat = ESConfig(
        entry_stop_dd=0.06,
        recovery_mode="peak",       # 旧 peak-based DD
        peak_recovery_ratio=0.70,
        hard_timeout_days=0,        # ハードタイムアウトなし
        probe_entry_enabled=False,  # プローブなし
        partial_close_ratio=0.30,
        max_cash_ratio_for_partial=1.0,  # 上限なし（v2挙動）
    )

    # ② v3 改善版（-6%, 1x）
    cfg_v3_1x = ESConfig(
        entry_stop_dd=0.06,
        recovery_mode="local",
        local_recovery_threshold=0.30,
        hard_timeout_days=45,
        probe_entry_enabled=True,
        probe_entry_min_days=5,
        probe_entry_scale=0.25,
        partial_close_ratio=0.30,
        max_cash_ratio_for_partial=0.60,
    )

    # ③ v3 改善版（-6%, 2x レバレッジ）— 実効閾値 -4.2%
    cfg_v3_2x = ESConfig(
        entry_stop_dd=0.06,
        leverage_dd_scale=0.70,     # 2x時: 0.06*0.7 = -4.2%
        recovery_mode="local",
        local_recovery_threshold=0.30,
        hard_timeout_days=45,
        probe_entry_enabled=True,
        probe_entry_min_days=5,
        probe_entry_scale=0.25,
        partial_close_ratio=0.30,
        max_cash_ratio_for_partial=0.60,
    )

    # ④ ストレステスト（-4%, 1x）— ES発動を複数回確認
    cfg_stress = ESConfig(
        entry_stop_dd=0.04,
        recovery_mode="local",
        local_recovery_threshold=0.25,
        hard_timeout_days=30,
        probe_entry_enabled=True,
        probe_entry_min_days=3,
        probe_entry_scale=0.20,
        partial_close_ratio=0.30,
        max_cash_ratio_for_partial=0.60,
        entry_stop_max_days=10,
    )

    scenarios: list[tuple[str, ESConfig, float]] = [
        ("v2旧方式（peak-based / no-timeout）", cfg_v2_compat, 1.0),
        ("v3改善（local-recovery / hard45d）1x", cfg_v3_1x, 1.0),
        ("v3改善（local-recovery / hard45d）2x ←実効-4.2%", cfg_v3_2x, 2.0),
        ("ストレス（-4%閾値 / hard30d）1x",     cfg_stress,   1.0),
    ]

    print("\n[3/4] バックテスト実行（4シナリオ）...")
    results: list[ESResultV3] = []
    for label, cfg, lev in scenarios:
        print(f"  [{label}]...", end=" ", flush=True)
        eng = ESEngineV3(universe=universe, strategies=strats,
                         cfg=cfg, label=label, leverage=lev)
        res = eng.run()
        results.append(res)
        print(f"CAGR={res.cagr*100:+.2f}%  MaxDD={res.max_drawdown*100:.2f}%  "
              f"ES={res.es_count}回  avg_lock={res.avg_lock_days:.1f}日  "
              f"破綻={'YES⚠️' if res.bankrupt else 'NO✓'}")

    # ── 結果表示 ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  比較サマリー")
    print("=" * 72)
    rows = [metrics_row(r) for r in results]
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n  ─── 詳細 ───")
    for res in results:
        print()
        res.summary()

    # ── KPI チェック ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  KPI 達成チェック")
    print("=" * 72)
    kpi_targets = {
        "ES発動回数 >= 2回":      lambda r: r.es_count >= 2,
        "avg_lock < 30日":        lambda r: r.avg_lock_days < 30 or r.es_count == 0,
        "max_lock < 60日":        lambda r: r.max_lock_days < 60 or r.es_count == 0,
        "MaxDD < -10%（v3本番）": lambda r: r.max_drawdown > -0.10,
        "破綻なし":               lambda r: not r.bankrupt,
    }
    for kpi, fn in kpi_targets.items():
        row = []
        for r in results:
            ok = fn(r)
            row.append(f"{'✅' if ok else '❌'} {r.label[:18]:<18}")
        print(f"  {kpi:<26}: " + "  ".join(row))

    # ── JSON 保存 ────────────────────────────────────────────────────
    today    = datetime.date.today().strftime("%Y-%m-%d")
    out_path = f"results/entry_stop_v3_{today}.json"
    os.makedirs("results", exist_ok=True)
    output = {
        "run_date": today, "period": f"{START}~{END}", "capital": CAPITAL,
        "scenarios": [
            dict(
                label          = r.label,
                eff_dd_pct     = round(r.eff_dd_threshold * 100, 2),
                cagr_pct       = round(r.cagr * 100, 2),
                max_dd_pct     = round(r.max_drawdown * 100, 2),
                calmar         = round(r.calmar, 3),
                sharpe         = round(r.sharpe, 3),
                win_rate_pct   = round(r.win_rate * 100, 1),
                es_count       = r.es_count,
                partial_count  = r.partial_count,
                avg_lock_days  = round(r.avg_lock_days, 1),
                max_lock_days  = r.max_lock_days,
                bankrupt       = r.bankrupt,
                es_events      = r.es_events,
            )
            for r in results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print("完了。")


if __name__ == "__main__":
    main()
