"""
src/portfolio/equity_peak_tracker.py

設計のみ。既存ライブ実行フロー（signal_bridge.py / run_live_signal.py / sync_positions.py）
には未統合。CB/DD判定ロジックも変更しない。

目的:
  現行の equity_peak（state["equity_peak"]）は balance（cash + market_value）の歴代最大値のみを
  追跡し、入出金イベントによる balance 変化と運用損益による変化を区別しない。これが本調査で
  確認した equity_peak 汚染（入金 → balance 上昇 → peak 誤上書き → 誤DD）の構造的原因である。

  本モジュールは balance_peak と performance_peak を分離して追跡する設計を提供する。
  どちらを CB/DD 判定の基準にするかは未確定（ASK_FIRST 対象）。本設計は追跡機構のみを
  提供し、既存の判定ロジックを置き換えない。

定義:
  balance_peak      = cash + market_value の歴代最大値
                       （現行の equity_peak と同義。入出金の影響をそのまま受ける）
  performance_peak  = cumulative_trading_pnl（運用損益の累積）の歴代最大値
                       入出金イベントとして識別された balance 変化分を除外して算出する

依存（未統合）:
  cash_event_delta は Task6 の入出金検知（[EQUITY_CASH_EVENT]）の出力を渡す想定。
  本モジュール単体では入出金を自動検知しない（呼び出し側が判定結果を渡す）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta

JST = timezone(timedelta(hours=9))


@dataclass
class EquityPeakSnapshot:
    timestamp:          str
    cash:                float
    market_value:        float
    balance_equity:      float
    cash_event_delta:    float   # 入出金と判定された balance 変化分（0 = 通常運用のみ）
    trading_pnl_delta:   float   # balance_delta - cash_event_delta
    balance_peak:        float
    performance_peak:    float


class EquityPeakTracker:
    """
    balance_peak と performance_peak を分離管理する状態追跡器（設計サンプル）。

    NOTE: 既存の state["equity_peak"] を置き換えるものではない。
    将来 CB/DD 判定への統合を検討する場合は ASK_FIRST 対象
    （PARAMS_LOCKED / CIRCUIT セクションに影響するため）。
    """

    def __init__(self, initial_cash: float, initial_market_value: float) -> None:
        initial_balance               = initial_cash + initial_market_value
        self.balance_peak: float      = initial_balance
        self.cumulative_trading_pnl: float = 0.0
        self.performance_peak: float  = 0.0
        self._prev_balance: float     = initial_balance
        self._history: list[EquityPeakSnapshot] = []

    def update(
        self,
        cash:             float,
        market_value:     float,
        cash_event_delta: float = 0.0,
    ) -> EquityPeakSnapshot:
        """
        balance_peak / performance_peak を更新する。

        Args:
            cash_event_delta: 入出金と判定された cash 変化量（Task6 検知結果）。
                               0 の場合、balance 変化分は全て運用損益として扱う。
        """
        balance_equity    = cash + market_value
        balance_delta     = balance_equity - self._prev_balance
        trading_pnl_delta = balance_delta - cash_event_delta

        self.cumulative_trading_pnl += trading_pnl_delta
        self._prev_balance           = balance_equity

        if balance_equity > self.balance_peak:
            self.balance_peak = balance_equity
        if self.cumulative_trading_pnl > self.performance_peak:
            self.performance_peak = self.cumulative_trading_pnl

        snap = EquityPeakSnapshot(
            timestamp         = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            cash              = cash,
            market_value      = market_value,
            balance_equity    = balance_equity,
            cash_event_delta  = cash_event_delta,
            trading_pnl_delta = trading_pnl_delta,
            balance_peak      = self.balance_peak,
            performance_peak  = self.performance_peak,
        )
        self._history.append(snap)
        return snap

    def balance_drawdown(self, current_balance: float) -> float:
        """現行 DD 計算と同義（入出金補正なし）。"""
        if self.balance_peak <= 0:
            return 0.0
        return (current_balance - self.balance_peak) / self.balance_peak

    def performance_drawdown(self, current_pnl: float) -> float:
        """入出金を除いた運用損益ベースの DD（performance_peak 基準）。"""
        if self.performance_peak <= 0:
            return 0.0
        return (current_pnl - self.performance_peak) / self.performance_peak
