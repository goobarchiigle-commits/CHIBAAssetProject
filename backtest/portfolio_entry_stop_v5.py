"""
backtest/portfolio_entry_stop_v5.py
entry_stop v5 — 状態遷移ステートマシン + ヒステリシス + 段階復帰

【v4からの5つの修正】
① ステージ遷移にヒステリシス（境界振動根絶）
   DOWN: DD絶対値 >= stage_threshold で即遷移
   UP:   DD絶対値  < stage_threshold * hyst_ratio になるまで遷移しない
   → 2xループの根本原因を排除

② stage3 = "縮小して生きる"（完全停止を廃止）
   stage3: 既存ポジを30%まで削減 + 新規エントリー停止
   → exposure=0の"死んだ状態"から"縮小継続"へ変更

③ hard_timeout → stage+1ではなく「1段階ずつ復帰」
   stage3 → stage2（NORMAL直帰禁止）
   stage2 → stage1 （if hard_timeout again）
   → 急速復帰によるリトリガーループを防止

④ エクスポージャ回復速度の制御
   actual_new_scale は recovery_speed（10%/日）で目標に向けて徐々に増加
   2x では recovery_speed * 0.5（倍の日数で慎重に回復）
   → whiplash防止、avg_expの安定化

⑤ 2x専用制御
   stage3_dd *= 0.8（2xは-8%でstage3 → 1x の-10%より早期発動）
   recovery_speed *= 0.5（回復も倍の日数）
   → 2xと1xを別戦略として取り扱う

パラメータ改善（v4分析から）
   velocity_threshold: 0.02 → 0.04
   zscore_threshold:   1.5  → 2.0
   zscore_min_periods: 126  → 200
   stage1_dd:          -4%  → -5%
   stage3_dd:          -8%  → -10%（1xのみ、2x=-8%）
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("es_v5")

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

# ─── ステージ定義 ─────────────────────────────────────────────────────────────
#
# Stage 0 NORMAL  : DD < -stage1_dd
# Stage 1 CAUTION : stage1_dd <= DD < stage2_dd  → 新規50%
# Stage 2 WARNING : stage2_dd <= DD < stage3_dd  → 新規20%
# Stage 3 ALERT   : DD >= stage3_dd              → 新規停止 + ポジ縮小30%
#
STAGE_LABEL = {0: "NORMAL", 1: "CAUTION", 2: "WARNING", 3: "ALERT"}

# ─── コンフィグ ───────────────────────────────────────────────────────────────
@dataclass
class ESConfigV5:

    # ─── ステージ閾値 ──────────────────────────────────────────────
    stage1_dd: float = 0.05    # -5%
    stage2_dd: float = 0.06    # -6%
    stage3_dd: float = 0.10    # -10%（2x時は *= leverage_stage3_scale）

    # ─── ステージ別エクスポージャ ──────────────────────────────────
    # 新規エントリーの最大スケール
    stage1_new_scale: float = 0.50
    stage2_new_scale: float = 0.20
    stage3_new_scale: float = 0.00   # stage3は新規停止

    # ポートフォリオの最大保有比率（閾値超過時はここまで強制クローズ）
    stage1_port_frac: float = 0.50   # max_positions * 0.5 = 1.5 → 2ポジ
    stage2_port_frac: float = 0.33   # max_positions * 0.33 = 1ポジ
    stage3_port_frac: float = 0.33   # stage3も1ポジ維持（縮小して生きる）

    # ─── ① ヒステリシス（UP遷移バッファ） ────────────────────────
    hyst_3_to_2: float = 0.70   # stage3→2: DD > -stage3_dd * 0.70 が必要
    hyst_2_to_1: float = 0.80   # stage2→1: DD > -stage2_dd * 0.80 が必要
    hyst_1_to_0: float = 0.60   # stage1→0: DD > -stage1_dd * 0.60 が必要

    # ─── ③ hard_timeout → 1段階ずつ復帰 ──────────────────────────
    hard_timeout_days: int = 45  # この日数経過で1段階UP（stage3→2、2→1）

    # ─── ④ エクスポージャ回復速度 ─────────────────────────────────
    recovery_speed: float = 0.10    # 1日で+10%ずつ目標に近づく（1x）
    # 2x: recovery_speed * leverage_recovery_speed_scale

    # ─── velocity トリガー（感度修正済み） ───────────────────────
    use_velocity: bool = True
    velocity_window: int = 5
    velocity_threshold: float = 0.04   # 5日で4%以上の急落

    # ─── Z-score トリガー（安定化修正済み） ──────────────────────
    use_zscore: bool = True
    zscore_window: int = 252
    zscore_min_periods: int = 200      # 1年相当のデータ後に有効化
    zscore_threshold: float = 2.0     # 2σ（旧1.5σ）

    # ─── ⑤ 2x専用制御 ────────────────────────────────────────────
    leverage_stage3_dd_scale: float    = 0.80   # 2x: stage3_dd *= 0.8 → -8%
    leverage_recovery_speed_scale: float = 0.50  # 2x: recovery_speed *= 0.5

    # ─── ポートフォリオ ───────────────────────────────────────────
    capital:           float = 2_000_000
    max_positions:     int   = 3
    max_single_weight: float = 0.25
    slippage_rate:     float = 0.001
    commission_rate:   float = 0.00055
    min_commission:    float = 99.0

    # クールダウン（velocity/z-scoreトリガー後の再発動防止）
    cooldown_days: int = 0   # v5ではヒステリシスで代替、通常は0


# ─── エンジン ─────────────────────────────────────────────────────────────────
class ESEngineV5:

    def __init__(
        self,
        universe:   dict,
        strategies: dict,
        cfg:        ESConfigV5 = None,
        label:      str = "",
        leverage:   float = 1.0,
    ) -> None:
        self.universe   = universe
        self.strategies = strategies
        self.cfg        = cfg or ESConfigV5()
        self.label      = label
        self.leverage   = leverage

        # ⑤ 2x専用閾値スケーリング
        is_2x = leverage >= 2.0
        cfg   = self.cfg
        self._s1   = cfg.stage1_dd
        self._s2   = cfg.stage2_dd
        self._s3   = cfg.stage3_dd * (cfg.leverage_stage3_dd_scale if is_2x else 1.0)
        self._rspd = cfg.recovery_speed * (cfg.leverage_recovery_speed_scale if is_2x else 1.0)

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
    def _dd_velocity(self, dd_hist: list[float]) -> float:
        n = self.cfg.velocity_window
        return (dd_hist[-1] - dd_hist[-1 - n]) if len(dd_hist) > n else 0.0

    def _dd_zscore(self, dd_hist: list[float]) -> Optional[float]:
        cfg = self.cfg
        if len(dd_hist) < cfg.zscore_min_periods:
            return None
        w   = dd_hist[-cfg.zscore_window:]
        mu  = sum(w) / len(w)
        var = sum((x - mu) ** 2 for x in w) / max(len(w) - 1, 1)
        std = var ** 0.5
        return None if std < 1e-8 else (dd_hist[-1] - mu) / std

    def _target_stage(
        self,
        dd_abs:        float,
        velocity:      float,
        dd_z:          Optional[float],
        current_stage: int,
    ) -> int:
        """
        ① ヒステリシス付きステージ計算。
        DOWN遷移: DD絶対値 >= 閾値 で即座に悪化
        UP遷移:  DD絶対値 <  閾値*hyst_ratio になるまで改善しない
        """
        cfg = self.cfg

        # --- 自然ステージ（DD絶対値ベース）---
        if   dd_abs >= self._s3: natural = 3
        elif dd_abs >= self._s2: natural = 2
        elif dd_abs >= self._s1: natural = 1
        else:                    natural = 0

        # --- velocity/z-score トリガー: 最低でも stage2 まで押し下げ ---
        forced = 0
        if cfg.use_velocity and velocity < -cfg.velocity_threshold:
            forced = max(forced, 2)
        if cfg.use_zscore and dd_z is not None and dd_z < -cfg.zscore_threshold:
            forced = max(forced, 2)

        proposed = max(natural, forced)

        # --- ① ヒステリシス: UP遷移時のみ適用 ---
        if proposed < current_stage:
            hyst_table = {
                (3, 2): self._s3 * cfg.hyst_3_to_2,
                (2, 1): self._s2 * cfg.hyst_2_to_1,
                (1, 0): self._s1 * cfg.hyst_1_to_0,
            }
            stage = current_stage
            while stage > proposed:
                threshold = hyst_table.get((stage, stage - 1), 0.0)
                if dd_abs > threshold:
                    break   # DDがまだ閾値以上 → これ以上UP遷移できない
                stage -= 1
            proposed = stage

        return proposed

    def _stage_max_positions(self, stage: int) -> int:
        """ステージごとの最大保有ポジション数"""
        fracs = {
            0: 1.0,
            1: self.cfg.stage1_port_frac,
            2: self.cfg.stage2_port_frac,
            3: self.cfg.stage3_port_frac,
        }
        return max(1, round(self.cfg.max_positions * fracs[stage]))

    def _stage_new_scale(self, stage: int) -> float:
        scales = {
            0: 1.0,
            1: self.cfg.stage1_new_scale,
            2: self.cfg.stage2_new_scale,
            3: self.cfg.stage3_new_scale,
        }
        return scales[stage]

    def _close_pos(
        self, sym: str, pos: Position, price: float,
        reason: str, date, cash: float, trades: list,
    ) -> float:
        exec_p = price * (1 - self._cost.slippage_rate)
        tv     = pos.qty * exec_p
        comm   = max(tv * self._cost.commission_rate, self._cost.min_commission)
        pnl    = (exec_p - pos.entry_price) * pos.qty - comm
        cash  += tv - comm
        trades.append(dict(date=date, symbol=sym, sector=pos.sector,
                           side=f"SELL_{reason}", qty=pos.qty,
                           price=exec_p, pnl=pnl))
        return cash

    def _close_excess_positions(
        self,
        target_n:  int,
        positions: dict,
        date,
        cash:      float,
        trades:    list,
        reason:    str,
    ) -> float:
        """最悪パフォーマンスから順に target_n になるまでクローズ"""
        if len(positions) <= target_n:
            return cash
        ranked = sorted(
            positions.keys(),
            key=lambda s: positions[s].unrealized_pnl(
                self.universe[s]["df"].loc[date, "Close"]
            ),
        )
        for sym in ranked[:len(positions) - target_n]:
            price = self.universe[sym]["df"].loc[date, "Close"]
            cash  = self._close_pos(sym, positions[sym], price, reason, date, cash, trades)
            del positions[sym]
            logger.info("  ③縮小CL %s @ ¥%.0f [%s]", sym, price, reason)
        return cash

    # ─── メインループ ────────────────────────────────────────────────
    def run(self) -> "ESResultV5":
        cfg  = self.cfg
        cost = self._cost

        cash              = cfg.capital * self.leverage
        positions: dict[str, Position] = {}
        records: list[dict]  = []
        trades:  list[dict]  = []
        peak_equity          = cash

        # ステートマシン変数
        current_stage        = 0     # 0=NORMAL, 1=CAUTION, 2=WARNING, 3=ALERT
        days_in_stage        = 0
        actual_new_scale     = 1.0   # ④ 実際の新規エントリースケール（徐々に回復）

        # メトリクス
        stage_history: list[int]       = []
        transition_log: list[dict]     = []
        dd_history:    list[float]     = []
        eq_vals:       list[float]     = []

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
            dd_abs      = abs(current_dd)

            dd_history.append(current_dd)

            # ─── 2. シグナル計算 ─────────────────────────────────────
            velocity = self._dd_velocity(dd_history)
            dd_z     = self._dd_zscore(dd_history)

            # ─── 3. ① ヒステリシス付きステージ遷移 ──────────────────
            new_stage = self._target_stage(dd_abs, velocity, dd_z, current_stage)

            if new_stage != current_stage:
                direction = "▼" if new_stage > current_stage else "▲"
                logger.info(
                    "%s %s stage %d[%s]→%d[%s] | DD=%.2f%% | v=%.2f%% | z=%s",
                    direction, date.date(),
                    current_stage, STAGE_LABEL[current_stage],
                    new_stage,     STAGE_LABEL[new_stage],
                    current_dd * 100, velocity * 100,
                    f"{dd_z:.2f}" if dd_z is not None else "n/a",
                )
                transition_log.append(dict(
                    date      = str(date.date()),
                    from_stage = current_stage,
                    to_stage  = new_stage,
                    direction = direction,
                    dd        = round(current_dd * 100, 2),
                    velocity  = round(velocity * 100, 2),
                    reason    = "dd" if new_stage == 3 else
                                ("vel" if velocity < -cfg.velocity_threshold else "z_score"),
                ))
                current_stage = new_stage
                days_in_stage = 0

                # ② ステージ悪化時: 即時ポジション削減
                if new_stage > 0:
                    target_n = self._stage_max_positions(new_stage)
                    cash = self._close_excess_positions(
                        target_n, positions, date, cash, trades,
                        f"STAGE{new_stage}_ENTER",
                    )
                    # ④ 悪化時は actual_new_scale を即座に下げる
                    actual_new_scale = min(actual_new_scale,
                                          self._stage_new_scale(new_stage))
            else:
                days_in_stage += 1

            # ─── 4. ③ hard_timeout → 1段階ずつ復帰 ──────────────────
            if current_stage >= 2 and days_in_stage >= cfg.hard_timeout_days:
                prev_stage    = current_stage
                current_stage -= 1   # 1段階だけ上げる（NORMAL直帰禁止）
                days_in_stage = 0
                logger.info(
                    "⏰ hard_timeout | %s | stage %d[%s]→%d[%s] | DD=%.2f%%",
                    date.date(), prev_stage, STAGE_LABEL[prev_stage],
                    current_stage, STAGE_LABEL[current_stage],
                    current_dd * 100,
                )
                transition_log.append(dict(
                    date=str(date.date()), from_stage=prev_stage, to_stage=current_stage,
                    direction="⏰", dd=round(current_dd * 100, 2),
                    velocity=round(velocity * 100, 2), reason="hard_timeout",
                ))

            # ─── 5. ④ エクスポージャ回復速度制御 ─────────────────────
            target_new_scale = self._stage_new_scale(current_stage)
            if actual_new_scale < target_new_scale:
                actual_new_scale = min(target_new_scale,
                                       actual_new_scale + self._rspd)
            elif actual_new_scale > target_new_scale:
                actual_new_scale = target_new_scale  # 悪化時は即座

            # ─── 6. 損切り（全ステージで実行） ───────────────────────
            for sym in list(positions.keys()):
                sig = self.strategies[sym].generate_signal(
                    self.universe[sym]["df"].loc[:date]
                )
                if sig == -1:
                    price = self.universe[sym]["df"].loc[date, "Close"]
                    cash  = self._close_pos(sym, positions[sym], price,
                                            "STOP", date, cash, trades)
                    del positions[sym]

            # ─── 7. 新規エントリー（stage3以外、actual_new_scale>0） ──
            allow_entry = (current_stage < 3) and (actual_new_scale > 0)
            if allow_entry and i + 1 < len(dates):
                next_date  = dates[i + 1]
                max_n_this = self._stage_max_positions(current_stage)
                pending: set[str] = set()

                for sym, info in self.universe.items():
                    if sym in positions or sym in pending:
                        continue
                    if len(positions) >= max_n_this:
                        break

                    df_sym = info["df"]
                    if date not in df_sym.index or next_date not in df_sym.index:
                        continue

                    sig = self.strategies[sym].generate_signal(df_sym.loc[:date])
                    if sig != 1:
                        continue

                    exec_p = df_sym.loc[next_date, "Open"] * (1 + cost.slippage_rate)
                    # 配分: actual_new_scale を適用
                    base_alloc = cfg.capital * self.leverage / cfg.max_positions
                    alloc = min(base_alloc * actual_new_scale,
                                cfg.capital * self.leverage * cfg.max_single_weight)
                    qty = int(alloc // (exec_p * 100)) * 100
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

            # ─── 8. 記録 ──────────────────────────────────────────────
            eq_vals.append(port_val)
            stage_history.append(current_stage)
            records.append(dict(
                date            = date,
                value           = port_val,
                dd              = current_dd,
                dd_velocity     = velocity,
                dd_z            = dd_z if dd_z is not None else float("nan"),
                stage           = current_stage,
                actual_scale    = actual_new_scale,
                n_positions     = len(positions),
                cash_ratio      = cash / port_val if port_val > 0 else 1.0,
            ))

        rec_df    = pd.DataFrame(records).set_index("date")
        trades_df = (pd.DataFrame(trades) if trades
                     else pd.DataFrame(columns=["date","symbol","sector",
                                                "side","qty","price","pnl"]))
        stage_series = pd.Series(stage_history, index=rec_df.index)
        return ESResultV5(
            equity_curve      = rec_df["value"],
            dd_series         = rec_df["dd"],
            stage_series      = stage_series,
            actual_scale_series = rec_df["actual_scale"],
            n_positions       = rec_df["n_positions"],
            trades            = trades_df,
            initial_capital   = cfg.capital * self.leverage,
            label             = self.label,
            eff_s3            = self._s3,
            eff_rspd          = self._rspd,
            transition_log    = transition_log,
        )


# ─── 結果 ─────────────────────────────────────────────────────────────────────
@dataclass
class ESResultV5:
    equity_curve:        pd.Series
    dd_series:           pd.Series
    stage_series:        pd.Series
    actual_scale_series: pd.Series
    n_positions:         pd.Series
    trades:              pd.DataFrame
    initial_capital:     float
    label:               str
    eff_s3:              float
    eff_rspd:            float
    transition_log:      list

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
    def avg_stage(self) -> float:
        return float(self.stage_series.mean())

    @property
    def avg_scale(self) -> float:
        return float(self.actual_scale_series.mean())

    @property
    def pct_time_normal(self) -> float:
        return float((self.stage_series == 0).mean())

    @property
    def pct_time_stage3(self) -> float:
        return float((self.stage_series == 3).mean())

    @property
    def n_stage3_entries(self) -> int:
        return sum(1 for t in self.transition_log if t["to_stage"] == 3)

    @property
    def bankrupt(self) -> bool:
        return bool((self.equity_curve < self.initial_capital * 0.10).any())

    @property
    def stage_breakdown_pct(self) -> dict:
        total = len(self.stage_series)
        return {
            f"stage{s}%": round(float((self.stage_series == s).sum() / total * 100), 1)
            for s in range(4)
        }

    def summary(self) -> None:
        w = 68
        print("=" * w)
        print(f"  {self.label}")
        print(f"  実効stage3閾値: -{self.eff_s3*100:.1f}%  / 回復速度: {self.eff_rspd*100:.1f}%/日")
        print("-" * w)
        print(f"  CAGR      : {self.cagr*100:>+8.2f}%")
        print(f"  MaxDD     : {self.max_drawdown*100:>+8.2f}%")
        print(f"  Calmar    : {self.calmar:>10.3f}")
        print(f"  Sharpe    : {self.sharpe:>10.3f}")
        print(f"  総リターン : {self.total_return*100:>+8.2f}%")
        print(f"  勝率       : {self.win_rate*100:>9.1f}%")
        print("-" * w)
        sb = self.stage_breakdown_pct
        print(f"  ステージ分布: "
              f"NORMAL={sb['stage0%']}%  CAUTION={sb['stage1%']}%  "
              f"WARNING={sb['stage2%']}%  ALERT={sb['stage3%']}%")
        print(f"  avg_stage  : {self.avg_stage:.3f}  /  avg_scale : {self.avg_scale:.3f}")
        print(f"  NORMAL率   : {self.pct_time_normal*100:.1f}%  "
              f"ALERT率: {self.pct_time_stage3*100:.1f}%")
        print(f"  ALERT発動  : {self.n_stage3_entries}回")
        print(f"  破綻        : {'YES ⚠️' if self.bankrupt else 'NO  ✓'}")
        print("=" * w)
        if self.transition_log:
            print("  ─── ステージ遷移ログ（上位20件） ───")
            for t in self.transition_log[:20]:
                print(f"    {t['direction']} {t['date']}  "
                      f"{t['from_stage']}→{t['to_stage']}  "
                      f"DD={t['dd']:.1f}%  v={t['velocity']:.2f}%  [{t['reason']}]")
            if len(self.transition_log) > 20:
                print(f"    ...（計{len(self.transition_log)}件）")


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


def metrics_row(res: ESResultV5) -> dict:
    sb = res.stage_breakdown_pct
    return {
        "シナリオ":    res.label[:28],
        "CAGR%":      f"{res.cagr*100:>+6.2f}",
        "MaxDD%":     f"{res.max_drawdown*100:>6.2f}",
        "Calmar":     f"{res.calmar:>6.3f}",
        "Sharpe":     f"{res.sharpe:>6.3f}",
        "avg_scale":  f"{res.avg_scale:>6.3f}",
        "NORMAL%":    sb["stage0%"],
        "ALERT%":     sb["stage3%"],
        "ALERT入":    res.n_stage3_entries,
        "破綻":       "YES" if res.bankrupt else "NO",
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=" * 72)
    print("  portfolio_entry_stop_v5.py")
    print("  ヒステリシス + 段階復帰 + 縮小継続 + 回復速度制御 + 2x専用")
    print(f"  期間: {START}〜{END}  /  資本: ¥{CAPITAL:,}")
    print("=" * 72)

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

    # ── シナリオ定義 ─────────────────────────────────────────────────
    # ① ベースライン
    cfg_base = ESConfigV5(
        use_velocity=False, use_zscore=False,
        stage1_dd=99.0, stage2_dd=99.0, stage3_dd=99.0,
    )

    # ② v4の勝者（③段階EXPのみ）との比較ベース
    cfg_stage_only_v4 = ESConfigV5(
        use_velocity=False, use_zscore=False,
        stage1_dd=0.04, stage2_dd=0.06, stage3_dd=99.0,  # stage3無効
        stage1_new_scale=0.5, stage2_new_scale=0.2,
        hard_timeout_days=9999,  # timeoutなし
        recovery_speed=9999.0,   # 即時回復（v4互換）
        hyst_3_to_2=0.0, hyst_2_to_1=0.0, hyst_1_to_0=0.0,  # ヒステリシスなし
    )

    # ③ v5 フル（ヒステリシス + 段階復帰 + velocity/z-score）1x
    cfg_v5_full = ESConfigV5(
        use_velocity=True,  velocity_threshold=0.04,
        use_zscore=True,    zscore_threshold=2.0, zscore_min_periods=200,
        stage1_dd=0.05, stage2_dd=0.06, stage3_dd=0.10,
        stage1_new_scale=0.5, stage2_new_scale=0.2, stage3_new_scale=0.0,
        hyst_3_to_2=0.70, hyst_2_to_1=0.80, hyst_1_to_0=0.60,
        hard_timeout_days=45,
        recovery_speed=0.10,
    )

    # ④ v5 段階EXPのみ + ヒステリシス（velocity/z-score なし）
    cfg_v5_stage = ESConfigV5(
        use_velocity=False, use_zscore=False,
        stage1_dd=0.05, stage2_dd=0.06, stage3_dd=0.10,
        stage1_new_scale=0.5, stage2_new_scale=0.2, stage3_new_scale=0.0,
        hyst_3_to_2=0.70, hyst_2_to_1=0.80, hyst_1_to_0=0.60,
        hard_timeout_days=45,
        recovery_speed=0.10,
    )

    # ⑤ v5 フル 2x（⑤専用制御: stage3_dd=-8%, recovery_speed=5%/日）
    cfg_v5_2x = ESConfigV5(
        use_velocity=True,  velocity_threshold=0.04,
        use_zscore=True,    zscore_threshold=2.0, zscore_min_periods=200,
        stage1_dd=0.05, stage2_dd=0.06, stage3_dd=0.10,
        stage1_new_scale=0.5, stage2_new_scale=0.2, stage3_new_scale=0.0,
        hyst_3_to_2=0.70, hyst_2_to_1=0.80, hyst_1_to_0=0.60,
        hard_timeout_days=45,
        recovery_speed=0.10,
        leverage_stage3_dd_scale=0.80,      # 2x: stage3=-8%
        leverage_recovery_speed_scale=0.50,  # 2x: 5%/日
    )

    scenarios: list[tuple[str, ESConfigV5, float]] = [
        ("ベースライン",                                 cfg_base,        1.0),
        ("v4 段階EXPのみ（比較用）",                     cfg_stage_only_v4, 1.0),
        ("v5 段階EXP+ヒステリシス（vel/z-score無）",    cfg_v5_stage,    1.0),
        ("v5 フル（①②③④⑤統合）1x",                  cfg_v5_full,     1.0),
        ("v5 フル 2x（stage3=-8%/回復5%/日）",          cfg_v5_2x,       2.0),
    ]

    print("\n[3/4] バックテスト実行（5シナリオ）...")
    results: list[ESResultV5] = []
    for label, cfg, lev in scenarios:
        print(f"  [{label}]...", end=" ", flush=True)
        eng = ESEngineV5(universe=universe, strategies=strats,
                         cfg=cfg, label=label, leverage=lev)
        res = eng.run()
        results.append(res)
        sb  = res.stage_breakdown_pct
        print(f"CAGR={res.cagr*100:+.2f}%  MaxDD={res.max_drawdown*100:.2f}%  "
              f"Calmar={res.calmar:.3f}  avg_scale={res.avg_scale:.3f}  "
              f"NORMAL={sb['stage0%']}%  ALERT入={res.n_stage3_entries}回  "
              f"破綻={'YES⚠️' if res.bankrupt else 'NO✓'}")

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
    print("  KPI チェック（ベースラインとの比較）")
    print("=" * 72)
    b = results[0]
    kpis = {
        f"Calmar >= {b.calmar*0.90:.3f} (BL×90%)": lambda r: r.calmar >= b.calmar * 0.90,
        f"CAGR   >= {(b.cagr-0.03)*100:.1f}% (BL-3%)":  lambda r: r.cagr   >= b.cagr - 0.03,
        "NORMAL率 >= 70%":                               lambda r: r.pct_time_normal >= 0.70,
        "avg_scale >= 0.80":                             lambda r: r.avg_scale >= 0.80,
        "max_lock < 45日":                               lambda r: all(
                                                              t.get("days","inf") != "inf"
                                                              for t in r.transition_log
                                                          ),
        "破綻なし":                                      lambda r: not r.bankrupt,
    }
    for kpi, fn in kpis.items():
        row = "  ".join(f"{'✅' if fn(r) else '❌'} {r.label[:14]}" for r in results)
        print(f"  {kpi:<40}: {row}")

    # ── JSON 保存 ────────────────────────────────────────────────────
    today    = datetime.date.today().strftime("%Y-%m-%d")
    out_path = f"results/entry_stop_v5_{today}.json"
    os.makedirs("results", exist_ok=True)
    output = dict(
        run_date=today, period=f"{START}~{END}", capital=CAPITAL,
        scenarios=[
            dict(
                label             = r.label,
                eff_stage3_pct    = round(r.eff_s3 * 100, 2),
                eff_rspd_pct      = round(r.eff_rspd * 100, 2),
                cagr_pct          = round(r.cagr * 100, 2),
                max_dd_pct        = round(r.max_drawdown * 100, 2),
                calmar            = round(r.calmar, 3),
                sharpe            = round(r.sharpe, 3),
                win_rate_pct      = round(r.win_rate * 100, 1),
                avg_scale         = round(r.avg_scale, 3),
                pct_time_normal   = round(r.pct_time_normal * 100, 1),
                pct_time_alert    = round(r.pct_time_stage3 * 100, 1),
                n_stage3_entries  = r.n_stage3_entries,
                bankrupt          = r.bankrupt,
                stage_breakdown   = r.stage_breakdown_pct,
                transition_log    = r.transition_log,
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
