"""
diagnostics/daily_state_logger.py
バックテストから exposure_root_cause.py 用の日次ログを出力するヘルパー

【バックテスト側での使い方】

    from diagnostics.daily_state_logger import DailyStateLogger

    logger = DailyStateLogger(k=3, capital=2_000_000)

    # シミュレーションループの各日に呼ぶ
    logger.record(
        date             = current_date,
        equity           = portfolio_equity,
        invested_capital = invested,
        positions        = n_positions,
        candidates       = len(buy_candidates),
        blocked_by_cb    = cb_active,
        # 拡張（任意）
        target_weight    = 1.0 / k,
        actual_weight    = actual_avg_weight,
        entry_signals    = n_entry_signals,
        skipped_due_to_price = n_price_skip,
        skipped_due_to_lot   = n_lot_skip,
        skipped_due_to_cash  = n_cash_skip,
    )

    # バックテスト完了後に保存
    logger.save("logs/backtest_daily_state.csv")

【出力カラム】
  必須: date, equity, invested_capital, positions,
        topk_target, candidates, blocked_by_cb
  拡張: target_weight, actual_weight,
        entry_signals, skipped_due_to_price,
        skipped_due_to_lot, skipped_due_to_cash
"""

from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pathlib import Path
from typing import Any

import pandas as pd


class DailyStateLogger:
    """
    バックテスト日次状態を蓄積し CSV に書き出すロガー。

    Args:
        k:       Top-k の k（max_positions と同値が通常）
        capital: 初期資本（参考値として記録）
    """

    COLUMNS_REQUIRED = [
        "date", "equity", "invested_capital", "positions",
        "topk_target", "candidates", "blocked_by_cb",
    ]
    COLUMNS_EXTENDED = [
        "target_weight", "actual_weight",
        "entry_signals", "skipped_due_to_price",
        "skipped_due_to_lot", "skipped_due_to_cash",
    ]

    def __init__(self, k: int, capital: float = 2_000_000):
        self.k       = k
        self.capital = capital
        self._rows: list[dict] = []

    def record(
        self,
        date:             Any,
        equity:           float,
        invested_capital: float,
        positions:        int,
        candidates:       int,
        blocked_by_cb:    bool | int,
        # 拡張カラム（省略可）
        target_weight:         float | None = None,
        actual_weight:         float | None = None,
        entry_signals:         int   | None = None,
        skipped_due_to_price:  int   | None = None,
        skipped_due_to_lot:    int   | None = None,
        skipped_due_to_cash:   int   | None = None,
    ) -> None:
        """1 営業日分の状態を記録する。"""
        row: dict = {
            "date":             pd.Timestamp(date).date().isoformat(),
            "equity":           round(float(equity), 0),
            "invested_capital": round(float(invested_capital), 0),
            "positions":        int(positions),
            "topk_target":      self.k,
            "candidates":       int(candidates),
            "blocked_by_cb":    int(bool(blocked_by_cb)),
        }
        # 拡張カラム（None でない場合のみ追記）
        if target_weight        is not None: row["target_weight"]        = round(float(target_weight), 4)
        if actual_weight        is not None: row["actual_weight"]        = round(float(actual_weight), 4)
        if entry_signals        is not None: row["entry_signals"]        = int(entry_signals)
        if skipped_due_to_price is not None: row["skipped_due_to_price"] = int(skipped_due_to_price)
        if skipped_due_to_lot   is not None: row["skipped_due_to_lot"]   = int(skipped_due_to_lot)
        if skipped_due_to_cash  is not None: row["skipped_due_to_cash"]  = int(skipped_due_to_cash)

        self._rows.append(row)

    def save(self, path: str | Path = "logs/backtest_daily_state.csv") -> Path:
        """蓄積データを CSV に保存する。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._rows)
        df.to_csv(path, index=False, encoding="utf-8")
        return path

    def to_dataframe(self) -> pd.DataFrame:
        """蓄積データを DataFrame で返す（保存せずに診断したい場合）。"""
        return pd.DataFrame(self._rows)

    def __len__(self) -> int:
        return len(self._rows)
