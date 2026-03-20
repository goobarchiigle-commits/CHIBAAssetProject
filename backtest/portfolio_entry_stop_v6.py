"""
backtest/portfolio_entry_stop_v6.py
entry_stop v6c — Z-score regime-gating + 回帰ベースvelocity + DD持続確認

【v6b からの追加修正（3点）】

① Z-score regime-gating（ブートストラップ破綻対策）
  問題: TEMPORAL ユニバースは初期 200 日が静穏 → rolling std ≈ 0.002
        → z 発散（z=-12）→ median floor を入れても初期サンプルが低分散で汚染
  修正: rolling_std > z_min_std(0.006) かつ len(std_history) >= z_min_history(252)
        を満たす場合のみ z-score を有効化。それ以外は velocity+DD のみ。

② velocity: 20日線形回帰スロープ（ノイズ低減）
  問題: 静穏期では 5日差分が小さすぎてノイズに埋もれ遅延する
  修正: np.polyfit(equity[-20:]) の slope / equity[-1] を velocity とする
        単位: fraction/day（0.001/day = 月約2%ペースで下落）
        threshold: 0.002/day（20日で4%ペース下落 = 旧5日4%と同等の感度）

③ DD持続確認（単発スパイク回避）
  問題: 1日の大引けに閾値を超えるだけでステージが変化する
  修正: dd_confirm_days=3 → 3日連続で閾値超を維持した場合のみ自然ステージ変化
        velocity/z-score トリガーは即時（持続確認なし）

④ 発注制御（state lock）
  backtest では1日1回のため自然に保証。live 運用では ORDER_LOCK_FILE で制御済み。

z-scoreの極性（実装上の注意）:
  dd_history は current_dd（負値）を格納。velocity も equity 回帰スロープ（負=下落）。
  - DD悪化 → velocity < 0、z < 0
  - z-score trigger: z < -Z_ENTRY AND dd_abs > 1% AND velocity < 0
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
    level=logging.WARNING,   # v6はWARNING以上のみ（実行時ノイズ削減）
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("es_v6")

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
    "石油": "mean_rev",
}
MR_PARAMS = dict(
    rsi_period=5, rsi_entry=25.0, rsi_exit=65.0,
    ma_long=200, stop_loss_pct=0.07, max_hold_days=10, knife_threshold=0.15,
)

STAGE_LABEL = {0: "NORMAL", 1: "CAUTION", 2: "WARNING", 3: "ALERT"}


# ─── コンフィグ ───────────────────────────────────────────────────────────────
@dataclass
class ESConfigV6:

    # ─── ステージ閾値（v5と同一） ──────────────────────────────────
    stage1_dd: float = 0.05
    stage2_dd: float = 0.06
    stage3_dd: float = 0.10

    # ─── ステージ別エクスポージャ（v5と同一） ────────────────────
    stage1_new_scale: float = 0.50
    stage2_new_scale: float = 0.20
    stage3_new_scale: float = 0.00

    stage1_port_frac: float = 0.50
    stage2_port_frac: float = 0.33
    stage3_port_frac: float = 0.33

    # ─── ヒステリシス（v5と同一） ─────────────────────────────────
    hyst_3_to_2: float = 0.70
    hyst_2_to_1: float = 0.80
    hyst_1_to_0: float = 0.60

    # ─── hard_timeout / 回復速度（v5と同一） ─────────────────────
    hard_timeout_days: int   = 45
    recovery_speed:    float = 0.10

    # ─── velocity トリガー（★v6c: 回帰ベースに変更） ───────────────
    use_velocity:              bool  = True
    velocity_regression_window: int  = 20     # 線形回帰ウィンドウ（日数）
    velocity_threshold:        float = 0.002  # slope/equity[%/day]、20日で4%下落ペース

    # ─── Z-score トリガー（★v6変更点） ──────────────────────────
    use_zscore:          bool  = True
    zscore_window:       int   = 252
    zscore_min_periods:  int   = 200

    # ① z-score 閾値（レジーム別）
    zscore_threshold_normal: float = 1.8
    zscore_threshold_quiet:  float = 2.2

    # ② DD 下限（微小DD除外）
    dd_zscore_min_abs: float = 0.01

    # ③ 静穏レジーム検出（floor前生値で判定）
    quiet_regime_dd_std: float = 0.005

    # ④ regime-gating（★v6c追加: ブートストラップ破綻対策）
    z_min_std:     float = 0.006  # rolling_std がこれ以上の場合のみ z-score 有効
    z_min_history: int   = 252    # std_history がこれ以上蓄積後のみ z-score 有効

    # ─── DD 持続確認（★v6c追加: 単発スパイク回避） ────────────
    dd_confirm_days: int = 3      # 絶対DD閾値を N日連続で超えた場合のみ自然ステージ変化

    # ─── exposure-adjusted DD（★v6d: TEMPORAL モード） ──────────
    # effective_dd = raw_dd / max(avg_exposure_60d, exposure_floor)
    # → 低稼働ポートフォリオでの DD 歪みを補正
    use_exposure_adjusted_dd: bool  = False
    exposure_floor:           float = 0.25   # min clamp（0除算防止）
    exposure_window:          int   = 60     # 移動平均ウィンドウ（営業日）

    # ─── 2x 専用制御（v5と同一） ─────────────────────────────────
    leverage_stage3_dd_scale:     float = 0.80
    leverage_recovery_speed_scale: float = 0.50

    # ─── ポートフォリオ ───────────────────────────────────────────
    capital:           float = 2_000_000
    max_positions:     int   = 3
    max_single_weight: float = 0.25
    slippage_rate:     float = 0.001
    commission_rate:   float = 0.00055
    min_commission:    float = 99.0

    cooldown_days: int = 0


# ─── エンジン ─────────────────────────────────────────────────────────────────
class ESEngineV6:

    def __init__(
        self,
        universe:   dict,
        strategies: dict,
        cfg:        ESConfigV6 = None,
        label:      str = "",
        leverage:   float = 1.0,
    ) -> None:
        self.universe   = universe
        self.strategies = strategies
        self.cfg        = cfg or ESConfigV6()
        self.label      = label
        self.leverage   = leverage

        is_2x = leverage >= 2.0
        cfg   = self.cfg
        self._s1   = cfg.stage1_dd
        self._s2   = cfg.stage2_dd
        self._s3   = cfg.stage3_dd * (cfg.leverage_stage3_dd_scale if is_2x else 1.0)
        self._rspd = cfg.recovery_speed * (cfg.leverage_recovery_speed_scale if is_2x else 1.0)

        dates = None
        for info in universe.values():
            idx   = info["df"].index
            dates = idx if dates is None else dates.intersection(idx)
        self.all_dates: pd.DatetimeIndex = dates.sort_values()
        self._cost = TradeCost(
            slippage_rate   = cfg.slippage_rate,
            commission_rate = cfg.commission_rate,
            min_commission  = cfg.min_commission,
        )

    # ─── ユーティリティ ──────────────────────────────────────────────

    def _equity_velocity(self, eq_hist: list[float]) -> float:
        """
        20日線形回帰スロープ / 現在equity = velocity（fraction/day）。

        - 負値: equity が下落トレンド（例 -0.002 = 0.2%/day 下落 ≈ 20日で4%）
        - 正値: equity が上昇トレンド
        - velocity < -threshold でトリガー（threshold デフォルト 0.002）

        静穏期の 5日差分ノイズ問題を回帰スロープで平滑化。
        """
        n = self.cfg.velocity_regression_window
        if len(eq_hist) < n:
            return 0.0
        y    = np.array(eq_hist[-n:], dtype=float)
        x    = np.arange(n, dtype=float)
        slope = np.polyfit(x, y, 1)[0]   # yen/day
        return float(slope / y[-1])       # fraction/day（現在equity で正規化）

    def _dd_zscore_v6(
        self,
        dd_hist:     list[float],
        std_history: list[float],   # 過去の window_std 履歴（median floor 計算用）
    ) -> tuple[Optional[float], Optional[float]]:
        """
        rolling z-score と window std（生値）を返す。

        Volatility floor:
          std_eff = max(std_raw, median(std_history))
          市場レジームや低分散ユニバースが変わっても z が発散しない。
          floor は過去の観測値の中央値なので look-ahead なし。

        Returns:
            (z, std_raw):
              z        : floor適用済みのz-score（None = データ不足）
              std_raw  : 今日の window_std（生値・静穏レジーム検出用）
        """
        cfg = self.cfg
        if len(dd_hist) < cfg.zscore_min_periods:
            return None, None

        w   = dd_hist[-cfg.zscore_window:]
        mu  = sum(w) / len(w)
        var = sum((x - mu) ** 2 for x in w) / max(len(w) - 1, 1)
        std_raw = var ** 0.5

        if std_raw < 1e-8:
            return None, None

        # ─── Regime-gate（★v6c追加: ブートストラップ破綻対策）──────
        # rolling_std が統計的に意味ある水準に達していない、または
        # std_history が十分蓄積されていない場合は z-score を無効化する
        if (std_raw <= cfg.z_min_std or len(std_history) < cfg.z_min_history):
            return None, std_raw   # z=None → z-scoreトリガー不発

        # ─── Volatility floor（★v6b追加）────────────────────────────
        # 過去の window_std の中央値を floor として使用
        # → z-score が統計的意味を失う「std 発散」問題を防止
        if std_history:
            std_floor = float(np.median(std_history))
            std_eff   = max(std_raw, std_floor)
        else:
            std_eff = std_raw

        z = (dd_hist[-1] - mu) / std_eff
        return z, std_raw   # std_raw を返す（静穏判定はfloor前の生値で行う）

    def _target_stage(
        self,
        natural_stage: int,           # ★v6c: confirm 済み自然ステージ（run loop で計算）
        dd_abs:        float,
        velocity:      float,
        dd_z:          Optional[float],
        dd_win_std:    Optional[float],
        current_stage: int,
    ) -> int:
        """
        ヒステリシス付きステージ計算。

        natural_stage : DD持続確認済みの自然ステージ（run loop で dd_confirm_counts から算出）
        velocity/z    : 即時トリガー（持続確認なし）
        """
        cfg = self.cfg

        # --- velocity トリガー（即時・独立） ---
        forced = 0
        if cfg.use_velocity and velocity < -cfg.velocity_threshold:
            forced = max(forced, 2)

        # --- Z-scoreトリガー（★v6: 3条件すべて必要） ---
        if cfg.use_zscore and dd_z is not None:
            # 静穏レジーム検出: std が小さければ閾値を厳しくする
            quiet = (dd_win_std is not None and dd_win_std < cfg.quiet_regime_dd_std)
            z_thresh = cfg.zscore_threshold_quiet if quiet else cfg.zscore_threshold_normal

            zscore_fires = (
                dd_z < -z_thresh            # ① z-score超過
                and dd_abs > cfg.dd_zscore_min_abs   # ② DD > 1%（微小DD除外）
                and velocity < 0            # ③ DD悪化中（velocity < 0 = 負値storage）
            )
            if zscore_fires:
                forced = max(forced, 2)

        proposed = max(natural_stage, forced)

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
                    break
                stage -= 1
            proposed = stage

        return proposed

    def _stage_max_positions(self, stage: int) -> int:
        fracs = {
            0: 1.0,
            1: self.cfg.stage1_port_frac,
            2: self.cfg.stage2_port_frac,
            3: self.cfg.stage3_port_frac,
        }
        return max(1, int(self.cfg.max_positions * fracs[stage]))

    def _stage_new_scale(self, stage: int) -> float:
        scales = {
            0: 1.0,
            1: self.cfg.stage1_new_scale,
            2: self.cfg.stage2_new_scale,
            3: self.cfg.stage3_new_scale,
        }
        return scales[stage]

    def _close_pos(self, sym, pos, price, reason, date, cash, trades):
        proceeds = price * pos.qty
        cost_val = max(proceeds * self.cfg.commission_rate, self.cfg.min_commission)
        cash    += proceeds - cost_val
        pnl      = (price - pos.entry_price) * pos.qty - cost_val
        trades.append(dict(
            date=date, symbol=sym, sector=pos.sector,
            side=f"SELL_{reason}", qty=pos.qty, price=price, pnl=pnl,
        ))
        return cash

    def _close_excess_positions(self, target_n, positions, date, cash, trades, reason):
        syms_by_pnl = sorted(
            positions.keys(),
            key=lambda s: (
                self.universe[s]["df"].loc[date, "Close"] - positions[s].entry_price
            ) * positions[s].qty
        )
        while len(positions) > target_n and syms_by_pnl:
            sym   = syms_by_pnl.pop(0)
            price = self.universe[sym]["df"].loc[date, "Close"]
            cash  = self._close_pos(sym, positions[sym], price, reason, date, cash, trades)
            del positions[sym]
        return cash

    # ─── メインループ ────────────────────────────────────────────────
    def run(self) -> "ESResultV6":
        cfg  = self.cfg
        cost = self._cost

        cash              = cfg.capital * self.leverage
        positions: dict[str, Position] = {}
        records: list[dict]  = []
        trades:  list[dict]  = []
        peak_equity          = cash

        current_stage        = 0
        days_in_stage        = 0
        actual_new_scale     = 1.0

        stage_history:    list[int]   = []
        transition_log:   list[dict] = []
        dd_history:       list[float] = []
        std_history:      list[float] = []   # window_std 履歴（median floor 用）
        eq_history:       list[float] = []   # equity 履歴（回帰velocity用）
        exp_history:      list[float] = []   # exposure 履歴（exposure-adjusted DD用）
        eq_vals:          list[float] = []
        quiet_history:    list[bool]  = []
        dd_confirm_counts = {1: 0, 2: 0, 3: 0}  # DD持続確認カウンタ

        dates = self.all_dates

        for i, date in enumerate(dates):

            # 1. 時価評価
            mkt = sum(
                pos.market_value(self.universe[s]["df"].loc[date, "Close"])
                for s, pos in positions.items()
                if date in self.universe[s]["df"].index
            )
            port_val    = cash + mkt
            peak_equity = max(peak_equity, port_val)
            current_dd  = (port_val - peak_equity) / peak_equity
            raw_dd_abs  = abs(current_dd)

            # ★v6d: exposure-adjusted DD
            current_exposure = len(positions) / cfg.max_positions  # 0〜1
            exp_history.append(current_exposure)
            if cfg.use_exposure_adjusted_dd and len(exp_history) >= cfg.exposure_window:
                avg_exp_60d = sum(exp_history[-cfg.exposure_window:]) / cfg.exposure_window
                exp_denom   = max(avg_exp_60d, cfg.exposure_floor)
                dd_abs      = raw_dd_abs / exp_denom   # 低稼働時の DD 歪みを補正
            else:
                dd_abs = raw_dd_abs

            dd_history.append(current_dd)
            eq_history.append(port_val)

            # 2. シグナル計算
            velocity         = self._equity_velocity(eq_history)             # ★v6c: 回帰
            dd_z, dd_win_std = self._dd_zscore_v6(dd_history, std_history)   # regime-gated
            if dd_win_std is not None:
                std_history.append(dd_win_std)
            is_quiet = (dd_win_std is not None
                        and dd_win_std < cfg.quiet_regime_dd_std)
            quiet_history.append(is_quiet)

            # ★v6c: DD持続確認カウンタ更新
            thresh_map = {1: self._s1, 2: self._s2, 3: self._s3}
            for s in [1, 2, 3]:
                if dd_abs >= thresh_map[s]:
                    dd_confirm_counts[s] += 1
                else:
                    dd_confirm_counts[s] = 0
            # 自然ステージ: dd_confirm_days 連続超えのみ確定（単発スパイク除外）
            natural_stage = 0
            for s in [3, 2, 1]:
                if dd_confirm_counts[s] >= cfg.dd_confirm_days:
                    natural_stage = s
                    break

            # 3. ステージ遷移
            new_stage = self._target_stage(
                natural_stage, dd_abs, velocity, dd_z, dd_win_std, current_stage
            )

            if new_stage != current_stage:
                direction = "▼" if new_stage > current_stage else "▲"
                reason = (
                    "dd"    if new_stage >= 3 else
                    "vel"   if (cfg.use_velocity and velocity < -cfg.velocity_threshold) else
                    "z_v6"  if (dd_z is not None and dd_z < -(cfg.zscore_threshold_quiet
                                if is_quiet else cfg.zscore_threshold_normal)) else
                    "hyst_up"
                )
                transition_log.append(dict(
                    date       = str(date.date()),
                    from_stage = current_stage,
                    to_stage   = new_stage,
                    direction  = direction,
                    dd         = round(current_dd * 100, 2),
                    velocity   = round(velocity * 100, 2),
                    dd_z       = round(dd_z, 3) if dd_z is not None else None,
                    dd_abs_pct = round(dd_abs * 100, 2),
                    quiet      = is_quiet,
                    reason     = reason,
                ))
                current_stage = new_stage
                days_in_stage = 0

                if new_stage > 0:
                    target_n = self._stage_max_positions(new_stage)
                    cash = self._close_excess_positions(
                        target_n, positions, date, cash, trades,
                        f"STAGE{new_stage}_ENTER",
                    )
            else:
                days_in_stage += 1

            # 4. hard_timeout → 1段階ずつ復帰
            if current_stage >= 2 and days_in_stage >= cfg.hard_timeout_days:
                prev_stage    = current_stage
                current_stage -= 1
                days_in_stage = 0
                transition_log.append(dict(
                    date=str(date.date()), from_stage=prev_stage, to_stage=current_stage,
                    direction="⏰", dd=round(current_dd * 100, 2),
                    velocity=round(velocity * 100, 2),
                    dd_z=None, dd_abs_pct=round(dd_abs*100, 2),
                    quiet=is_quiet, reason="hard_timeout",
                ))

            # 5. エクスポージャ回復速度制御
            target_new_scale = self._stage_new_scale(current_stage)
            if actual_new_scale < target_new_scale:
                actual_new_scale = min(target_new_scale, actual_new_scale + self._rspd)
            elif actual_new_scale > target_new_scale:
                actual_new_scale = target_new_scale

            # 6. 損切り
            for sym in list(positions.keys()):
                sig = self.strategies[sym].generate_signal(
                    self.universe[sym]["df"].loc[:date]
                )
                if sig == -1:
                    price = self.universe[sym]["df"].loc[date, "Close"]
                    cash  = self._close_pos(sym, positions[sym], price,
                                            "STOP", date, cash, trades)
                    del positions[sym]

            # 7. 新規エントリー
            allow_entry = (current_stage < 3) and (actual_new_scale > 0)
            if allow_entry and i + 1 < len(dates):
                next_date  = dates[i + 1]
                max_n_this = self._stage_max_positions(current_stage)

                for sym, info in self.universe.items():
                    if sym in positions:
                        continue
                    if len(positions) >= max_n_this:
                        break

                    df_sym = info["df"]
                    if date not in df_sym.index or next_date not in df_sym.index:
                        continue

                    sig = self.strategies[sym].generate_signal(df_sym.loc[:date])
                    if sig != 1:
                        continue

                    exec_p   = df_sym.loc[next_date, "Open"] * (1 + cost.slippage_rate)
                    base_alloc = cfg.capital * self.leverage / cfg.max_positions
                    alloc    = min(base_alloc * actual_new_scale,
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

            # 8. 記録
            eq_vals.append(port_val)
            stage_history.append(current_stage)
            records.append(dict(
                date         = date,
                value        = port_val,
                dd           = current_dd,
                dd_velocity  = velocity,
                dd_z         = dd_z if dd_z is not None else float("nan"),
                dd_win_std   = dd_win_std if dd_win_std is not None else float("nan"),
                quiet        = is_quiet,
                stage        = current_stage,
                actual_scale = actual_new_scale,
                n_positions  = len(positions),
                cash_ratio   = cash / port_val if port_val > 0 else 1.0,
            ))

        rec_df    = pd.DataFrame(records).set_index("date")
        trades_df = (pd.DataFrame(trades) if trades
                     else pd.DataFrame(columns=["date","symbol","sector",
                                                "side","qty","price","pnl"]))
        stage_series = pd.Series(stage_history, index=rec_df.index)
        return ESResultV6(
            equity_curve        = rec_df["value"],
            dd_series           = rec_df["dd"],
            stage_series        = stage_series,
            actual_scale_series = rec_df["actual_scale"],
            n_positions         = rec_df["n_positions"],
            quiet_series        = pd.Series(quiet_history, index=rec_df.index),
            trades              = trades_df,
            initial_capital     = cfg.capital * self.leverage,
            label               = self.label,
            eff_s3              = self._s3,
            eff_rspd            = self._rspd,
            transition_log      = transition_log,
        )


# ─── 結果 ─────────────────────────────────────────────────────────────────────
@dataclass
class ESResultV6:
    equity_curve:        pd.Series
    dd_series:           pd.Series
    stage_series:        pd.Series
    actual_scale_series: pd.Series
    n_positions:         pd.Series
    quiet_series:        pd.Series
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
    def pct_time_quiet(self) -> float:
        return float(self.quiet_series.mean())

    @property
    def n_stage3_entries(self) -> int:
        return sum(1 for t in self.transition_log if t["to_stage"] == 3)

    @property
    def n_zscore_triggers(self) -> int:
        return sum(1 for t in self.transition_log if t.get("reason") == "z_v6")

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
        w = 72
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
        print(f"  静穏レジーム: {self.pct_time_quiet*100:.1f}%")
        print(f"  ALERT発動  : {self.n_stage3_entries}回  "
              f"（うちZ-score v6: {self.n_zscore_triggers}回）")
        print(f"  破綻        : {'YES ⚠️' if self.bankrupt else 'NO  ✓'}")
        print("=" * w)
        if self.transition_log:
            print("  ─── ステージ遷移ログ（上位20件）───")
            for t in self.transition_log[:20]:
                z_str = f"z={t['dd_z']:.2f}" if t.get("dd_z") is not None else "z=n/a"
                q_str = "🔇" if t.get("quiet") else "  "
                print(f"  {q_str}{t['direction']} {t['date']}  "
                      f"{t['from_stage']}→{t['to_stage']}  "
                      f"DD={t['dd']:.1f}%  abs={t['dd_abs_pct']:.2f}%  "
                      f"v={t['velocity']:.2f}%  {z_str}  [{t['reason']}]")
            if len(self.transition_log) > 20:
                print(f"  ...（計{len(self.transition_log)}件）")


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
                mom_period=21, turtle_entry=20, turtle_exit=20,
                use_turtle_entry=True,
            )
    return strats


def metrics_row(res: ESResultV6) -> dict:
    sb = res.stage_breakdown_pct
    return {
        "シナリオ":    res.label[:30],
        "CAGR%":      f"{res.cagr*100:>+6.2f}",
        "MaxDD%":     f"{res.max_drawdown*100:>6.2f}",
        "Calmar":     f"{res.calmar:>6.3f}",
        "Sharpe":     f"{res.sharpe:>6.3f}",
        "avg_scale":  f"{res.avg_scale:>6.3f}",
        "NORMAL%":    sb["stage0%"],
        "ALERT%":     sb["stage3%"],
        "ALERT入":    res.n_stage3_entries,
        "Z_v6発動":   res.n_zscore_triggers,
        "破綻":       "YES" if res.bankrupt else "NO",
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main() -> None:
    import os
    from backtest.universe_builder import download_universe
    from backtest.portfolio_temporal_separation import (
        SEL_START, SEL_END,
        EVAL_START, EVAL_END, SHARPE_THRESH, MAXDD_THRESH,
    )

    print("=" * 72)
    print("  portfolio_entry_stop_v6.py")
    print("  Z-score 3条件化 + 静穏レジーム適応")
    print("=" * 72)

    # ユニバース: TEMPORAL 24銘柄（2015-2017選択 / 2018-2024評価）
    import json
    from pathlib import Path
    uni_path = Path("configs/universe/2026Q1_temporal24.json")
    if uni_path.exists():
        uni_data = json.loads(uni_path.read_text(encoding="utf-8"))
        tickers_raw = uni_data["symbols"]
        print(f"\n  ユニバース: {uni_data['version']} ({len(tickers_raw)}銘柄)")
    else:
        # フォールバック: portfolio_v2 のG27銘柄
        from backtest.portfolio_v2 import G29_UNIVERSE
        tickers_raw = G29_UNIVERSE
        print(f"\n  ユニバース: G29フォールバック ({len(tickers_raw)}銘柄)")

    # セクター情報を TEMPORAL から取得
    tickers_with_sector = {sym: sec for sym, sec in tickers_raw.items()}

    print(f"\n[1/4] データ取得（{len(tickers_with_sector)}銘柄）...")
    universe = download_universe(
        tickers_with_sector, start=EVAL_START, end=EVAL_END, verbose=False
    )
    print(f"  完了: {len(universe)} 銘柄")

    print("\n[2/4] RSR 計算...")
    prices   = {s: info["df"]["Close"] for s, info in universe.items()}
    rsr_uni  = calc_universe_rsr(prices)
    # TEMPORAL 24銘柄のセクター→戦略マッピングは SECTOR_STRATEGY 共通テーブルを使用
    strats   = build_strategies(universe, rsr_uni, sym_to_strat={})

    # ── シナリオ定義 ─────────────────────────────────────────────────

    # ① ベースライン（velocity/z-score無効）
    cfg_base = ESConfigV6(
        use_velocity=False, use_zscore=False,
        stage1_dd=99.0, stage2_dd=99.0, stage3_dd=99.0,
    )

    # ② velocity のみ（z-score無効: TEMPORAL ユニバース基本設定）
    cfg_vel_only = ESConfigV6(
        use_velocity=True,  velocity_threshold=0.002,  # 回帰: 0.2%/day ≈ 20日で4%
        use_zscore=False,
        dd_confirm_days=3,
        stage1_dd=0.05, stage2_dd=0.06, stage3_dd=0.10,
        stage1_new_scale=0.5, stage2_new_scale=0.2, stage3_new_scale=0.0,
        hyst_3_to_2=0.70, hyst_2_to_1=0.80, hyst_1_to_0=0.60,
        hard_timeout_days=45, recovery_speed=0.10,
    )

    # ③ v6c フル（regime-gating + 回帰velocity + DD持続確認）
    cfg_v6c = ESConfigV6(
        use_velocity=True,  velocity_threshold=0.002,
        use_zscore=True,
        zscore_threshold_normal=1.8,
        zscore_threshold_quiet=2.2,
        dd_zscore_min_abs=0.01,
        quiet_regime_dd_std=0.005,
        z_min_std=0.006,          # regime-gate: std < 0.006 → z-score 無効
        z_min_history=252,        # regime-gate: 252件蓄積後のみ有効
        dd_confirm_days=3,
        stage1_dd=0.05, stage2_dd=0.06, stage3_dd=0.10,
        stage1_new_scale=0.5, stage2_new_scale=0.2, stage3_new_scale=0.0,
        hyst_3_to_2=0.70, hyst_2_to_1=0.80, hyst_1_to_0=0.60,
        hard_timeout_days=45, recovery_speed=0.10,
    )

    # ④ v6c フル 2x
    cfg_v6c_2x = ESConfigV6(
        use_velocity=True,  velocity_threshold=0.002,
        use_zscore=True,
        zscore_threshold_normal=1.8,
        zscore_threshold_quiet=2.2,
        dd_zscore_min_abs=0.01,
        quiet_regime_dd_std=0.005,
        z_min_std=0.006,
        z_min_history=252,
        dd_confirm_days=3,
        stage1_dd=0.05, stage2_dd=0.06, stage3_dd=0.10,
        stage1_new_scale=0.5, stage2_new_scale=0.2, stage3_new_scale=0.0,
        hyst_3_to_2=0.70, hyst_2_to_1=0.80, hyst_1_to_0=0.60,
        hard_timeout_days=45, recovery_speed=0.10,
        leverage_stage3_dd_scale=0.80,
        leverage_recovery_speed_scale=0.50,
    )

    # ⑤ v6d: exposure-adjusted DD（TEMPORAL モード）
    cfg_v6d = ESConfigV6(
        use_velocity=True,  velocity_threshold=0.002,
        use_zscore=True,
        zscore_threshold_normal=1.8,
        zscore_threshold_quiet=2.2,
        dd_zscore_min_abs=0.01,
        quiet_regime_dd_std=0.005,
        z_min_std=0.006,
        z_min_history=252,
        dd_confirm_days=3,
        # exposure-adjusted DD（★v6d）
        use_exposure_adjusted_dd=True,
        exposure_floor=0.25,
        exposure_window=60,
        # TEMPORAL 用に adjusted 閾値を引き上げ
        stage1_dd=0.06, stage2_dd=0.08, stage3_dd=0.12,
        stage1_new_scale=0.5, stage2_new_scale=0.2, stage3_new_scale=0.0,
        hyst_3_to_2=0.70, hyst_2_to_1=0.80, hyst_1_to_0=0.60,
        hard_timeout_days=45, recovery_speed=0.10,
    )

    scenarios: list[tuple[str, ESConfigV6, float]] = [
        ("ベースライン",                          cfg_base,    1.0),
        ("velocity only（z無効）",               cfg_vel_only, 1.0),
        ("v6c フル（regime-gate）1x",            cfg_v6c,     1.0),
        ("v6d exp-adjusted（TEMPORAL mode）",    cfg_v6d,     1.0),
    ]

    print("\n[3/4] バックテスト実行（4シナリオ）...")
    results: list[ESResultV6] = []
    for label, cfg, lev in scenarios:
        print(f"  [{label}]...", end=" ", flush=True)
        eng = ESEngineV6(universe=universe, strategies=strats,
                         cfg=cfg, label=label, leverage=lev)
        res = eng.run()
        results.append(res)
        sb  = res.stage_breakdown_pct
        print(f"CAGR={res.cagr*100:+.2f}%  MaxDD={res.max_drawdown*100:.2f}%  "
              f"Calmar={res.calmar:.3f}  avg_scale={res.avg_scale:.3f}  "
              f"NORMAL={sb['stage0%']}%  ALERT入={res.n_stage3_entries}回  "
              f"Z発動={res.n_zscore_triggers}回  "
              f"静穏={res.pct_time_quiet*100:.1f}%")

    print("\n" + "=" * 72)
    print("  比較サマリー")
    print("=" * 72)
    print(pd.DataFrame([metrics_row(r) for r in results]).to_string(index=False))

    print("\n  ─── 詳細 ───")
    for res in results:
        print()
        res.summary()

    # KPI チェック
    print("\n" + "=" * 72)
    print("  KPI チェック（ベースラインとの比較）")
    print("=" * 72)
    b = results[0]
    kpis = {
        f"Calmar >= {b.calmar*0.90:.3f} (BL×90%)": lambda r: r.calmar >= b.calmar * 0.90,
        f"CAGR   >= {(b.cagr-0.03)*100:.1f}% (BL-3%)": lambda r: r.cagr >= b.cagr - 0.03,
        "NORMAL率 >= 70%":                              lambda r: r.pct_time_normal >= 0.70,
        "avg_scale >= 0.80":                            lambda r: r.avg_scale >= 0.80,
        "破綻なし":                                     lambda r: not r.bankrupt,
    }
    for kpi, fn in kpis.items():
        row = "  ".join(f"{'✅' if fn(r) else '❌'} {r.label[:16]}" for r in results)
        print(f"  {kpi:<40}: {row}")

    # JSON 保存
    today    = datetime.date.today().strftime("%Y-%m-%d")
    out_path = f"results/entry_stop_v6_{today}.json"
    os.makedirs("results", exist_ok=True)
    out = {
        "script":    "portfolio_entry_stop_v6.py",
        "run_date":  today,
        "universe":  "2026Q1_temporal24 (24銘柄)",
        "eval_period": f"{EVAL_START} / {EVAL_END}",
        "scenarios": [
            {
                "label":      r.label,
                "cagr_pct":   round(r.cagr * 100, 2),
                "max_dd_pct": round(r.max_drawdown * 100, 2),
                "calmar":     round(r.calmar, 3),
                "sharpe":     round(r.sharpe, 3),
                "avg_scale":  round(r.avg_scale, 3),
                "pct_normal": round(r.pct_time_normal * 100, 1),
                "n_alert":    r.n_stage3_entries,
                "n_zscore_v6": r.n_zscore_triggers,
                "pct_quiet":  round(r.pct_time_quiet * 100, 1),
            }
            for r in results
        ],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  結果保存: {out_path}")
    print("\n[4/4] 完了")


if __name__ == "__main__":
    main()
