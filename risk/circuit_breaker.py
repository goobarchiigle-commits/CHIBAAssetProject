"""
risk/circuit_breaker.py
サーキットブレーカー（CB）モジュール

状態機械:
    NORMAL
      ↓ DD <= -15%（DD_TRIGGER）
    ENTRY_STOP_ONLY  ← BUY 停止 / SELL は引き続き許可
      ↓ 30営業日経過 OR DD >= -5%（RECOVERY_DD）に回復
    NORMAL

設計ポイント（デッドロック防止）:
  - peak は日次で更新（equity が新高値のとき更新）
  - CB 解除は「営業日数 OR 回復 DD」の先着方式
  - cb_start が None のまま ENTRY_STOP_ONLY になるバグを防ぐ guard 付き
  - 直列更新を前提（1日1回 update_cb_state を呼ぶ）
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Optional

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from market.jpx_calendar import JPXCalendar

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
DD_TRIGGER   = -0.15   # -15%: ENTRY_STOP_ONLY へ移行
RECOVERY_DD  = -0.05   # -5%: NORMAL へ回復
MAX_CB_DAYS  = 30      # 最大 30 営業日で強制解除

_calendar = JPXCalendar()


# ------------------------------------------------------------------ #
# 状態
# ------------------------------------------------------------------ #
@dataclass
class CBState:
    mode:         str            = "NORMAL"          # "NORMAL" | "ENTRY_STOP_ONLY"
    peak_equity:  float          = 0.0
    cb_start:     Optional[str]  = None              # ISO date string（例: "2026-03-20"）


# ------------------------------------------------------------------ #
# ユーティリティ
# ------------------------------------------------------------------ #
def calc_drawdown(equity: float, peak: float) -> float:
    """現在 DD（0 以下の分数値）を返す。peak=0 のときは 0。"""
    if peak <= 0:
        return 0.0
    return (equity - peak) / peak


def _update_peak(state: CBState, equity: float) -> None:
    """equity が過去最高値を更新したら peak を上書き（インプレース）。"""
    if equity > state.peak_equity:
        state.peak_equity = equity


# ------------------------------------------------------------------ #
# メイン更新関数（日次で 1 回呼ぶ）
# ------------------------------------------------------------------ #
def update_cb_state(state: CBState, equity: float, today: date) -> CBState:
    """
    CB 状態を 1 営業日分更新して返す。

    Args:
        state:  現在の CB 状態（インプレース更新）
        equity: 当日終値ベースのポートフォリオ評価額
        today:  当日の日付

    Returns:
        更新後の CBState（同一オブジェクト）
    """
    _update_peak(state, equity)
    dd = calc_drawdown(equity, state.peak_equity)

    if state.mode == "NORMAL":
        if dd <= DD_TRIGGER:
            state.mode     = "ENTRY_STOP_ONLY"
            state.cb_start = today.isoformat()
            logger.warning(
                "CB TRIGGERED: DD=%.1f%% → ENTRY_STOP_ONLY（%s）",
                dd * 100, today
            )

    elif state.mode == "ENTRY_STOP_ONLY":
        # guard: cb_start が None の場合は当日を起点に設定
        if state.cb_start is None:
            state.cb_start = today.isoformat()
            logger.warning("CB: cb_start が None → 当日 %s を起点に設定", today)

        cb_start_date = date.fromisoformat(state.cb_start)
        elapsed_days  = _calendar.trading_days_between(cb_start_date, today)

        if elapsed_days >= MAX_CB_DAYS:
            logger.warning(
                "CB RESET（強制）: %d 営業日経過 → NORMAL（%s）",
                elapsed_days, today
            )
            _reset(state)

        elif dd >= RECOVERY_DD:
            logger.info(
                "CB RESET（回復）: DD=%.1f%% >= %.1f%% → NORMAL（%s）",
                dd * 100, RECOVERY_DD * 100, today
            )
            _reset(state)

        else:
            logger.debug(
                "CB 継続: mode=%s / DD=%.1f%% / elapsed=%d日",
                state.mode, dd * 100, elapsed_days
            )

    return state


def _reset(state: CBState) -> None:
    """CB を NORMAL に戻す（インプレース）。"""
    state.mode     = "NORMAL"
    state.cb_start = None


# ------------------------------------------------------------------ #
# 発注許可判定
# ------------------------------------------------------------------ #
def can_enter(state: CBState) -> bool:
    """BUY 可能かどうかを返す。ENTRY_STOP_ONLY のときは False。"""
    return state.mode == "NORMAL"


def can_exit(state: CBState) -> bool:
    """SELL は常に許可（True）。"""
    return True


# ------------------------------------------------------------------ #
# 永続化（runtime/cb_state.json）
# ------------------------------------------------------------------ #
_STATE_FILE = Path(__file__).parent.parent / "runtime" / "cb_state.json"


def load_cb_state(path: Path = _STATE_FILE) -> CBState:
    """
    JSON ファイルから CB 状態を読み込む。
    ファイルが存在しない場合はデフォルト状態を返す。
    """
    if not path.exists():
        return CBState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return CBState(**data)
    except Exception as e:
        logger.warning("CB 状態ファイルの読み込み失敗 (%s) → 初期化", e)
        return CBState()


def save_cb_state(state: CBState, path: Path = _STATE_FILE) -> None:
    """CB 状態を JSON ファイルに保存する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ------------------------------------------------------------------ #
# CLI（単体テスト用）
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    state = load_cb_state()
    print(f"現在の CB 状態: {state}")

    today = date.today()
    equity = float(sys.argv[1]) if len(sys.argv) > 1 else state.peak_equity or 2_000_000

    state = update_cb_state(state, equity, today)
    save_cb_state(state)

    print(f"更新後: {state}")
    print(f"BUY 許可: {can_enter(state)}")
    print(f"SELL 許可: {can_exit(state)}")
