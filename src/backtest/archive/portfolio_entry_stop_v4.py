"""
backtest/portfolio_entry_stop_v4.py
entry_stop v4 — 「機能するが期待値を毀損しない」設計

【v3からの改善点】
① DD velocity トリガー
   急落のみを捕捉（5日DDの変化率 < -velocity_threshold で発動）
   → 緩やかな下落でのフライング発動を排除

② DD Z-score トリガー
   252日ローリングで平均・標準偏差を算出し、偏差ベースで判定
   → 固定閾値をやめ相場環境に自動適応

③ 段階的エクスポージャ制御
   binary on/off → -4%: 50%、-6%: 20%、-8%: 0% の連続縮小
   → フル停止によるCAGR劣化を防止
   → エクスポージャ低下時は最悪パフォーマンスのポジションから自動部分クローズ

④ 指数回復ベース解除（EMA クロス）
   線形回復率に加え、equity の EMA(short) > EMA(long) クロスで解除判定
   → 底値付近でのノイズに惑わされず「実際に回復傾向が出た」時点で解除

設計方針:
  - ① ② は velocity/z-score トリガーで entry_stop を強制発動（exposure=0）
  - ③ は entry_stop 外では常時稼働（DD に応じてスケールを調整）
  - entry_stop 中の解除は ④ + hard_timeout（v3継承）
  - ④ は解除条件を "EMA回復 OR (線形回復 AND 日数/トレンド) OR hard_timeout" に統合

注: ④の仕様は原文が途中で切れていたため、指数EMAクロス＋回復率として実装。
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("es_v4")

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
class ESConfigV4:
    """v4 全設定。4つの機能ブロックに対応。"""

    # ─── ① velocity トリガー ─────────────────────────────────────
    use_velocity:       bool  = True
    velocity_window:    int   = 5        # DDの変化率を見る期間（営業日）
    velocity_threshold: float = 0.02     # 5日間で-2%以上の急落でトリガー

    # ─── ② Z-score トリガー ──────────────────────────────────────
    use_zscore:         bool  = True
    zscore_window:      int   = 252      # ローリングウィンドウ（1年）
    zscore_min_periods: int   = 126      # 最低データ数（半年）
    zscore_threshold:   float = 1.5      # -1.5σ 以下でトリガー

    # ─── ③ 段階的エクスポージャ ──────────────────────────────────
    stage1_dd:    float = 0.04   # -4%以下で exposure = stage1_scale
    stage1_scale: float = 0.5    # 50%
    stage2_dd:    float = 0.06   # -6%以下で exposure = stage2_scale
    stage2_scale: float = 0.2    # 20%
    stage3_dd:    float = 0.08   # -8%以下で entry_stop（exposure=0）
    # ③ 部分クローズ戦略: 最悪パフォーマンスから順に閉じる

    # ─── ④ 指数回復ベース解除 ────────────────────────────────────
    ema_short:              int   = 10   # 短期EMAスパン
    ema_long:               int   = 30   # 長期EMAスパン
    ema_recovery_threshold: float = 0.20 # EMAクロスに加えて必要な回復率（底値比）

    # 線形回復（v3 互換フォールバック）
    linear_recovery_threshold: float = 0.30

    # 日数・ハードタイムアウト
    max_days:          int  = 20
    hard_timeout_days: int  = 45  # 0で無効

    # ─── 再エントリー制御 ─────────────────────────────────────────
    cooldown_days:       int   = 3
    just_released_days:  int   = 5
    just_released_scale: float = 0.7   # v4: 解除直後も0.7倍（v3の0.5より緩和）

    # ─── レバレッジ連動 ────────────────────────────────────────────
    leverage_dd_scale: float = 0.70  # 2x時に全閾値に掛ける係数

    # ─── ポートフォリオ ───────────────────────────────────────────
    capital:           float = 2_000_000
    max_positions:     int   = 3
    max_single_weight: float = 0.25
    slippage_rate:     float = 0.001
    commission_rate:   float = 0.00055
    min_commission:    float = 99.0


# ─── エンジン ─────────────────────────────────────────────────────────────────
class ESEngineV4:
    """v4 エンジン。4機能を統合したポートフォリオバックテスト。"""

    def __init__(
        self,
        universe:   dict,
        strategies: dict,
        cfg:        ESConfigV4 = None,
        label:      str = "",
        leverage:   float = 1.0,
    ) -> None:
        self.universe   = universe
        self.strategies = strategies
        self.cfg        = cfg or ESConfigV4()
        self.label      = label
        self.leverage   = leverage

        # レバレッジ連動閾値スケーリング
        s = cfg.leverage_dd_scale if leverage >= 2.0 else 1.0
        self._s1 = cfg.stage1_dd    * s
        self._s2 = cfg.stage2_dd    * s
        self._s3 = cfg.stage3_dd    * s
        self._v_th = cfg.velocity_threshold  # velocity は資産規模非依存

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

    # ─── 内部ユーティリティ ───────────────────────────────────────
    def _staged_exposure(self, dd: float) -> float:
        """③ 段階的エクスポージャ: DD水準に対応する通常エクスポージャ係数"""
        cfg = self.cfg
        dd_abs = abs(dd)
        if dd_abs < self._s1:  return 1.0
        if dd_abs < self._s2:  return cfg.stage1_scale
        if dd_abs < self._s3:  return cfg.stage2_scale
        return 0.0  # -8%以下 → entry_stop

    def _dd_velocity(self, dd_history: list[float]) -> float:
        """
        ① 5日DD変化率（lookahead ゼロ）
        dd_history = [dd_{t-4}, ..., dd_{t}] の最新5件以上が必要
        """
        n = self.cfg.velocity_window
        if len(dd_history) < n + 1:
            return 0.0
        return dd_history[-1] - dd_history[-1 - n]

    def _dd_zscore(self, dd_history: list[float]) -> Optional[float]:
        """
        ② DDのZ-score化（252日ローリング、min_periods=126）
        現在バーのDD値が過去252日の分布で何σかを返す。
        データ不足の場合は None（トリガーしない）。
        """
        cfg = self.cfg
        min_p = cfg.zscore_min_periods
        if len(dd_history) < min_p:
            return None
        window = dd_history[-cfg.zscore_window:]
        mu  = sum(window) / len(window)
        var = sum((x - mu) ** 2 for x in window) / max(len(window) - 1, 1)
        std = var ** 0.5
        if std < 1e-8:  # ゼロ除算防止
            return None
        return (dd_history[-1] - mu) / std

    def _ema(self, values: list[float], span: int) -> float:
        """指数移動平均（オンライン計算）。最後の span*2 件だけ使用。"""
        alpha = 2.0 / (span + 1)
        seq   = values[-min(len(values), span * 3):]
        ema   = seq[0]
        for v in seq[1:]:
            ema = alpha * v + (1 - alpha) * ema
        return ema

    def _ema_recovery(
        self,
        eq_vals:    list[float],
        bottom_eq:  float,
        peak_eq:    float,
    ) -> bool:
        """
        ④ 指数回復ベース解除判定。
        条件: EMA(short) > EMA(long) クロス AND 底値から recovery_threshold 以上回復
        """
        cfg = self.cfg
        if len(eq_vals) < cfg.ema_long:
            return False
        ema_s = self._ema(eq_vals, cfg.ema_short)
        ema_l = self._ema(eq_vals, cfg.ema_long)
        cross_up = ema_s > ema_l
        span  = peak_eq - bottom_eq
        ratio = (eq_vals[-1] - bottom_eq) / max(span, 1)
        return cross_up and ratio >= cfg.ema_recovery_threshold

    def _linear_recovery(
        self, current: float, peak: float, bottom: float
    ) -> bool:
        """線形回復率チェック（フォールバック）"""
        span = peak - bottom
        if span <= 0:
            return current >= peak
        return (current - bottom) / span >= self.cfg.linear_recovery_threshold

    def _alloc(self, exposure: float) -> float:
        """exposure 係数を反映した1ポジション配分額"""
        cfg  = self.cfg
        base = cfg.capital * self.leverage / cfg.max_positions
        return min(base * exposure,
                   cfg.capital * self.leverage * cfg.max_single_weight)

    def _close_pos(
        self, sym: str, pos: Position, price: float,
        reason: str, date, cash: float, trades: list,
        qty_override: int = None,
    ) -> tuple[float, int]:
        qty        = min(qty_override or pos.qty, pos.qty)
        exec_price = price * (1 - self._cost.slippage_rate)
        trade_val  = qty * exec_price
        commission = max(trade_val * self._cost.commission_rate,
                         self._cost.min_commission)
        pnl  = (exec_price - pos.entry_price) * qty - commission
        cash += trade_val - commission
        trades.append(dict(date=date, symbol=sym, sector=pos.sector,
                           side=f"SELL_{reason}", qty=qty,
                           price=exec_price, pnl=pnl))
        return cash, qty

    def _reduce_positions_to_target(
        self,
        target_n:   int,
        positions:  dict,
        date,
        cash:       float,
        trades:     list,
        reason:     str,
    ) -> float:
        """
        ③ エクスポージャ低下時の自動部分クローズ。
        最悪パフォーマンス（含み損が大きい順）から target_n になるまで閉じる。
        """
        if len(positions) <= target_n:
            return cash
        sorted_syms = sorted(
            positions.keys(),
            key=lambda s: positions[s].unrealized_pnl(
                self.universe[s]["df"].loc[date, "Close"]
            ),
        )
        n_close = len(positions) - target_n
        for sym in sorted_syms[:n_close]:
            pos   = positions[sym]
            price = self.universe[sym]["df"].loc[date, "Close"]
            cash, _ = self._close_pos(sym, pos, price, reason, date, cash, trades)
            del positions[sym]
            logger.info("  ③縮小CL %s @ ¥%.0f (%s)", sym, price, reason)
        return cash

    # ─── メインループ ────────────────────────────────────────────────
    def run(self) -> "ESResultV4":
        cfg  = self.cfg
        cost = self._cost

        cash            = cfg.capital * self.leverage
        positions: dict[str, Position] = {}
        records         = []
        trades          = []
        peak_equity     = cash
        bottom_stop_eq  = cash  # entry_stop 発動後の最低 equity

        # entry_stop 状態
        entry_stop          = False
        days_in_stop        = 0
        cooldown_remaining  = 0
        just_released_rem   = 0
        prev_exposure       = 1.0  # 前バーのエクスポージャ（③部分CL用）
        es_count            = 0
        es_events: list[dict] = []

        # ローリング用バッファ
        dd_history:  list[float] = []
        eq_vals:     list[float] = []

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

            dd_history.append(current_dd)
            if entry_stop:
                bottom_stop_eq = min(bottom_stop_eq, port_val)

            # ─── 2. クールダウン ──────────────────────────────────────
            if cooldown_remaining > 0:
                cooldown_remaining -= 1

            # ─── 3. トリガー判定 ─────────────────────────────────────

            # ① velocity トリガー
            velocity        = self._dd_velocity(dd_history)
            velocity_hit    = cfg.use_velocity and velocity < -self._v_th

            # ② Z-score トリガー
            dd_z            = self._dd_zscore(dd_history)
            zscore_hit      = (cfg.use_zscore
                               and dd_z is not None
                               and dd_z < -cfg.zscore_threshold)

            # ③ 段階的エクスポージャ（現在 DD に対応する通常露出）
            stage_exposure  = self._staged_exposure(current_dd)
            stage_trigger   = (stage_exposure == 0.0)  # -8%以下で強制発動

            any_trigger = (stage_trigger or velocity_hit or zscore_hit)
            trigger_reasons = (
                (["velocity"] if velocity_hit else [])
                + (["z_score"] if zscore_hit  else [])
                + (["stage3"]  if stage_trigger else [])
            )

            # ─── 4. entry_stop 発動 ──────────────────────────────────
            if any_trigger and not entry_stop and cooldown_remaining == 0:
                entry_stop     = True
                bottom_stop_eq = port_val
                days_in_stop   = 0
                es_count      += 1
                logger.info(
                    "📛 entry_stop発動 | %s | DD=%.2f%% | v=%.2f%% | z=%s | 理由=%s",
                    date.date(), current_dd * 100,
                    velocity * 100,
                    f"{dd_z:.2f}" if dd_z is not None else "n/a",
                    "+".join(trigger_reasons) or "none",
                )
                es_events.append(dict(
                    activate_date   = str(date.date()),
                    activate_dd     = round(current_dd * 100, 2),
                    trigger_reasons = trigger_reasons,
                    release_date    = None,
                    release_dd      = None,
                    days_in_stop    = None,
                    release_reason  = None,
                ))

            # ─── 5. ③ エクスポージャ自動縮小（entry_stop 外） ────────
            if not entry_stop:
                target_exposure = stage_exposure
                if just_released_rem > 0:
                    target_exposure = min(target_exposure, cfg.just_released_scale)
                    just_released_rem -= 1

                # エクスポージャが下がったら最悪ポジを先に閉じる
                if target_exposure < prev_exposure and positions:
                    target_n = max(0, round(cfg.max_positions * target_exposure))
                    cash = self._reduce_positions_to_target(
                        target_n, positions, date, cash, trades,
                        f"STAGE_DOWN_x{target_exposure:.1f}",
                    )
                prev_exposure = target_exposure
            else:
                target_exposure = 0.0
                prev_exposure   = 0.0

            # ─── 6. entry_stop 解除チェック ─────────────────────────
            if entry_stop:
                days_in_stop += 1

                # ④ 指数回復ベース解除
                ema_rec  = self._ema_recovery(eq_vals, bottom_stop_eq, peak_equity)
                # 線形回復 + 日数/トレンドフォールバック
                lin_rec  = self._linear_recovery(port_val, peak_equity, bottom_stop_eq)
                days_ok  = days_in_stop >= cfg.max_days
                trend_ok = (len(eq_vals) >= cfg.ema_long and
                            self._ema(eq_vals, cfg.ema_short) >
                            self._ema(eq_vals, cfg.ema_long))
                hard_ok  = (cfg.hard_timeout_days > 0
                            and days_in_stop >= cfg.hard_timeout_days)

                release = ema_rec or (lin_rec and (days_ok or trend_ok)) or hard_ok
                reason  = ("ema_cross"   if ema_rec
                           else "hard_timeout" if hard_ok
                           else "lin+days"     if (lin_rec and days_ok)
                           else "lin+trend"    if (lin_rec and trend_ok)
                           else None)

                # velocity が継続している間は解除しない（ただし hard_timeout は除く）
                if release and not hard_ok:
                    if velocity < -self._v_th:
                        release = False
                        reason  = None
                        logger.debug("解除抑制: velocity=%.2f%% (閾値=%.2f%%)",
                                     velocity * 100, self._v_th * 100)

                if release:
                    recovery_ratio = ((port_val - bottom_stop_eq) /
                                      max(peak_equity - bottom_stop_eq, 1))
                    logger.info(
                        "✅ entry_stop解除 | %s | DD=%.2f%% | 在停=%d日 | "
                        "理由=%s | ema_s/l=%.0f/%.0f | rec=%.3f",
                        date.date(), current_dd * 100, days_in_stop, reason,
                        self._ema(eq_vals, cfg.ema_short) if eq_vals else 0,
                        self._ema(eq_vals, cfg.ema_long)  if eq_vals else 0,
                        recovery_ratio,
                    )
                    if es_events:
                        es_events[-1].update(dict(
                            release_date   = str(date.date()),
                            release_dd     = round(current_dd * 100, 2),
                            days_in_stop   = days_in_stop,
                            release_reason = reason,
                        ))
                    entry_stop        = False
                    cooldown_remaining = cfg.cooldown_days
                    just_released_rem  = cfg.just_released_days
                    days_in_stop       = 0
                    target_exposure    = self._staged_exposure(current_dd)
                    prev_exposure      = target_exposure

            # ─── 7. 損切り（entry_stop 中も実行） ─────────────────────
            for sym in list(positions.keys()):
                sig = self.strategies[sym].generate_signal(
                    self.universe[sym]["df"].loc[:date]
                )
                if sig == -1:
                    price = self.universe[sym]["df"].loc[date, "Close"]
                    cash, _ = self._close_pos(sym, positions[sym], price,
                                              "STOP", date, cash, trades)
                    del positions[sym]

            # ─── 8. エントリー ───────────────────────────────────────
            if not entry_stop and target_exposure > 0 and i + 1 < len(dates):
                next_date = dates[i + 1]
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
                    alloc  = self._alloc(target_exposure)
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
                        side="BUY", qty=qty, price=exec_p, pnl=0.0,
                    ))
                    pending.add(sym)
                    if target_exposure < 1.0:
                        logger.debug("縮小エントリー %s x%.1f @ %s",
                                     sym, target_exposure, next_date.date())

            # ─── 9. 記録 ──────────────────────────────────────────────
            eq_vals.append(port_val)
            records.append(dict(
                date           = date,
                value          = port_val,
                dd             = current_dd,
                dd_velocity    = velocity,
                dd_z           = dd_z if dd_z is not None else float("nan"),
                n_positions    = len(positions),
                entry_stop     = entry_stop,
                exposure       = target_exposure,
                cash_ratio     = cash / port_val if port_val > 0 else 1.0,
            ))

        # 未解除イベント後処理
        if entry_stop and es_events and es_events[-1]["release_date"] is None:
            es_events[-1].update(dict(
                release_date   = "N/A(未解除)",
                release_dd     = round(current_dd * 100, 2),
                days_in_stop   = days_in_stop,
                release_reason = "未解除",
            ))

        rec_df    = pd.DataFrame(records).set_index("date")
        trades_df = (pd.DataFrame(trades) if trades
                     else pd.DataFrame(columns=["date","symbol","sector",
                                                "side","qty","price","pnl"]))
        return ESResultV4(
            equity_curve      = rec_df["value"],
            dd_series         = rec_df["dd"],
            velocity_series   = rec_df["dd_velocity"],
            zscore_series     = rec_df["dd_z"],
            exposure_series   = rec_df["exposure"],
            n_positions       = rec_df["n_positions"],
            entry_stop_series = rec_df["entry_stop"],
            trades            = trades_df,
            initial_capital   = cfg.capital * self.leverage,
            label             = self.label,
            eff_thresholds    = dict(s1=self._s1, s2=self._s2, s3=self._s3),
            es_count          = es_count,
            es_events         = es_events,
        )


# ─── 結果 ─────────────────────────────────────────────────────────────────────
@dataclass
class ESResultV4:
    equity_curve:      pd.Series
    dd_series:         pd.Series
    velocity_series:   pd.Series
    zscore_series:     pd.Series
    exposure_series:   pd.Series
    n_positions:       pd.Series
    entry_stop_series: pd.Series
    trades:            pd.DataFrame
    initial_capital:   float
    label:             str
    eff_thresholds:    dict
    es_count:          int
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

    @property
    def trigger_breakdown(self) -> dict:
        """各トリガー種別の発動回数集計"""
        cnt = {"velocity": 0, "z_score": 0, "stage3": 0}
        for ev in self.es_events:
            for r in ev.get("trigger_reasons", []):
                cnt[r] = cnt.get(r, 0) + 1
        return cnt

    def summary(self) -> None:
        w = 66
        th = self.eff_thresholds
        print("=" * w)
        print(f"  {self.label}")
        print(f"  実効閾値: s1=-{th['s1']*100:.1f}%  s2=-{th['s2']*100:.1f}%"
              f"  s3=-{th['s3']*100:.1f}%")
        print("-" * w)
        print(f"  CAGR      : {self.cagr*100:>+8.2f}%")
        print(f"  MaxDD     : {self.max_drawdown*100:>+8.2f}%")
        print(f"  Calmar    : {self.calmar:>10.3f}")
        print(f"  Sharpe    : {self.sharpe:>10.3f}")
        print(f"  総リターン : {self.total_return*100:>+8.2f}%")
        print(f"  決済回数   : {len(self.trades[self.trades['side'].str.startswith('SELL')]):>10}")
        print(f"  勝率       : {self.win_rate*100:>9.1f}%")
        print(f"  avg_exp    : {self.exposure_series.mean():>9.3f}")
        print("-" * w)
        tb = self.trigger_breakdown
        print(f"  ES発動     : {self.es_count:>6}回  "
              f"(vel={tb.get('velocity',0)} / z={tb.get('z_score',0)} / s3={tb.get('stage3',0)})")
        print(f"  avg_lock   : {self.avg_lock_days:.1f}日  /  max={self.max_lock_days}日")
        print(f"  破綻        : {'YES ⚠️' if self.bankrupt else 'NO  ✓'}")
        print("=" * w)
        if self.es_events:
            print("  ─── ES イベント ───")
            for ev in self.es_events:
                tr = "+".join(ev.get("trigger_reasons", ["-"]))
                print(f"    {ev['activate_date']} DD={ev['activate_dd']:.1f}%"
                      f" [{tr}]"
                      f" → {ev['release_date']} DD={ev.get('release_dd','?')}%"
                      f" [{ev['days_in_stop']}日/{ev.get('release_reason','-')}]")


# ─── ヘルパー ─────────────────────────────────────────────────────────────────
def build_strategies(universe: dict, rsr_uni: pd.DataFrame, sym_to_strat: dict) -> dict:
    strats = {}
    for sym, info in universe.items():
        rsr_s = rsr_uni[sym] if sym in rsr_uni.columns else None
        rule  = SECTOR_STRATEGY.get(info["sector"], "dynamic")
        lbl   = sym_to_strat.get(sym, "フジコ法")
        if rule == "mean_rev" or (rule == "dynamic" and "平均回帰" in lbl):
            strats[sym] = MeanReversionStrategy(**MR_PARAMS)
        else:
            strats[sym] = FujikoStrategy(
                rsr_series=rsr_s, min_sepa=6, min_rsr=70.0,
                mom_period=21, turtle_entry=20, turtle_exit=10,
                use_turtle_entry=True,
            )
    return strats


def metrics_row(res: ESResultV4) -> dict:
    tb = res.trigger_breakdown
    return {
        "シナリオ":       res.label[:30],
        "CAGR%":         f"{res.cagr*100:+.2f}",
        "MaxDD%":        f"{res.max_drawdown*100:.2f}",
        "Calmar":        f"{res.calmar:.3f}",
        "Sharpe":        f"{res.sharpe:.3f}",
        "avg_exp":       f"{res.exposure_series.mean():.3f}",
        "ES発動":        res.es_count,
        "vel/z/s3":      f"{tb.get('velocity',0)}/{tb.get('z_score',0)}/{tb.get('stage3',0)}",
        "avg_lock":      f"{res.avg_lock_days:.1f}",
        "max_lock":      res.max_lock_days,
        "破綻":          "YES" if res.bankrupt else "NO",
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("  portfolio_entry_stop_v4.py")
    print("  ① velocity  ② Z-score  ③ 段階的エクスポージャ  ④ EMA回復解除")
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
    prices  = {s: info["df"]["Close"] for s, info in universe.items()}
    rsr_uni = calc_universe_rsr(prices)
    strats  = build_strategies(universe, rsr_uni, sym_to_strat)

    # ── シナリオ ────────────────────────────────────────────────────
    # ① ベースライン: 機能なし（純粋な segment比較用）
    cfg_base = ESConfigV4(
        use_velocity=False, use_zscore=False,
        stage1_dd=0.99, stage2_dd=0.99, stage3_dd=0.99,  # 実質無効
        hard_timeout_days=0,
    )

    # ② ③のみ（段階的エクスポージャ、velocity/z-scoreなし）
    cfg_stage_only = ESConfigV4(
        use_velocity=False, use_zscore=False,
        stage1_dd=0.04, stage1_scale=0.5,
        stage2_dd=0.06, stage2_scale=0.2,
        stage3_dd=0.08,
        hard_timeout_days=45,
    )

    # ③ v4 フル（①②③④全て）1x
    cfg_v4_full = ESConfigV4(
        use_velocity=True,  velocity_window=5, velocity_threshold=0.02,
        use_zscore=True,    zscore_window=252, zscore_min_periods=126,
                            zscore_threshold=1.5,
        stage1_dd=0.04, stage1_scale=0.5,
        stage2_dd=0.06, stage2_scale=0.2,
        stage3_dd=0.08,
        ema_short=10, ema_long=30, ema_recovery_threshold=0.20,
        linear_recovery_threshold=0.30,
        max_days=20, hard_timeout_days=45,
        cooldown_days=3, just_released_days=5, just_released_scale=0.7,
    )

    # ④ v4 フル 2x（全閾値自動スケール）
    cfg_v4_2x = ESConfigV4(
        use_velocity=True,  velocity_threshold=0.02,
        use_zscore=True,    zscore_threshold=1.5,
        stage1_dd=0.04, stage1_scale=0.5,
        stage2_dd=0.06, stage2_scale=0.2,
        stage3_dd=0.08,
        leverage_dd_scale=0.70,
        ema_short=10, ema_long=30, ema_recovery_threshold=0.20,
        max_days=20, hard_timeout_days=45,
        cooldown_days=3, just_released_days=5, just_released_scale=0.7,
    )

    # ⑤ 感度確認: velocity 感度を上げる（threshold=0.015）
    cfg_sensitive = ESConfigV4(
        use_velocity=True,  velocity_threshold=0.015,
        use_zscore=True,    zscore_threshold=1.2,
        stage1_dd=0.03, stage1_scale=0.5,
        stage2_dd=0.05, stage2_scale=0.2,
        stage3_dd=0.07,
        ema_short=10, ema_long=30, ema_recovery_threshold=0.15,
        max_days=15, hard_timeout_days=30,
        cooldown_days=3, just_released_days=3, just_released_scale=0.7,
    )

    scenarios: list[tuple[str, ESConfigV4, float]] = [
        ("ベースライン（ES機能なし）",        cfg_base,       1.0),
        ("③ 段階的EXPのみ（vel/z-score無）", cfg_stage_only, 1.0),
        ("v4 フル（①②③④） 1x",            cfg_v4_full,    1.0),
        ("v4 フル 2x（実効-2.8%/-4.2%/-5.6%）", cfg_v4_2x,  2.0),
        ("v4 高感度（vel=1.5% / z=1.2σ）",   cfg_sensitive,  1.0),
    ]

    print("\n[3/4] バックテスト実行（5シナリオ）...")
    results: list[ESResultV4] = []
    for label, cfg, lev in scenarios:
        print(f"  [{label}]...", end=" ", flush=True)
        eng = ESEngineV4(universe=universe, strategies=strats,
                         cfg=cfg, label=label, leverage=lev)
        res = eng.run()
        results.append(res)
        tb  = res.trigger_breakdown
        print(f"CAGR={res.cagr*100:+.2f}%  MaxDD={res.max_drawdown*100:.2f}%  "
              f"Calmar={res.calmar:.3f}  ES={res.es_count}回"
              f"(v={tb.get('velocity',0)}/z={tb.get('z_score',0)}/s3={tb.get('stage3',0)})"
              f"  avg_lock={res.avg_lock_days:.1f}日  破綻={'YES⚠️' if res.bankrupt else 'NO✓'}")

    # ── 結果表示 ────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  比較サマリー")
    print("=" * 72)
    print(pd.DataFrame([metrics_row(r) for r in results]).to_string(index=False))

    print("\n  ─── 詳細 ───")
    for res in results:
        print()
        res.summary()

    # ── KPI チェック ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  KPI チェック")
    print("=" * 72)
    ref_calmar = results[0].calmar  # ベースライン
    ref_cagr   = results[0].cagr
    kpis = {
        "Calmar >= baseline*0.90": lambda r: r.calmar >= ref_calmar * 0.90,
        "CAGR >= baseline-3%":     lambda r: r.cagr   >= ref_cagr   - 0.03,
        "ES発動 >= 1回（v4）":     lambda r: r.es_count >= 1 or "ベース" in r.label,
        "avg_lock < 30日":         lambda r: r.avg_lock_days < 30 or r.es_count == 0,
        "max_lock < 60日":         lambda r: r.max_lock_days < 60,
        "永久ロックなし":           lambda r: "未解除" not in str(r.es_events) or r.es_count==0,
        "破綻なし":                 lambda r: not r.bankrupt,
    }
    for kpi, fn in kpis.items():
        row = "  ".join(f"{'✅' if fn(r) else '❌'} {r.label[:16]}" for r in results)
        print(f"  {kpi:<30}: {row}")

    # ── JSON 保存 ────────────────────────────────────────────────────
    today    = datetime.date.today().strftime("%Y-%m-%d")
    out_path = f"results/entry_stop_v4_{today}.json"
    os.makedirs("results", exist_ok=True)
    output = dict(
        run_date  = today,
        period    = f"{START}~{END}",
        capital   = CAPITAL,
        scenarios = [
            dict(
                label          = r.label,
                eff_thresholds = r.eff_thresholds,
                cagr_pct       = round(r.cagr * 100, 2),
                max_dd_pct     = round(r.max_drawdown * 100, 2),
                calmar         = round(r.calmar, 3),
                sharpe         = round(r.sharpe, 3),
                avg_exposure   = round(float(r.exposure_series.mean()), 3),
                es_count       = r.es_count,
                trigger_breakdown = r.trigger_breakdown,
                avg_lock_days  = round(r.avg_lock_days, 1),
                max_lock_days  = r.max_lock_days,
                bankrupt       = r.bankrupt,
                es_events      = r.es_events,
            )
            for r in results
        ],
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n結果保存: {out_path}")
    print("完了。")


if __name__ == "__main__":
    main()
