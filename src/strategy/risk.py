"""
risk.py — リスク・ストップ・状態管理・時間制御（純粋関数）

目的: リスク・ストップ・状態管理・時間制御を統合し、
     実運用での損失制御と一貫性を保証する。

Environment variables:
    ATR_MULT             float ATR倍率（hard stop距離）               (default 2.0)
    TRAIL_ATR_MULT       float ATR倍率（trailing stop距離）           (default 1.5)
    MIN_STOP_PCT         float 最小ストップ距離（価格比）              (default 0.01)
    RISK_PER_TRADE       float 1トレード当たりリスク比率（equity比）   (default 0.01)
    GAP_BUFFER           float ワーストケース損失の上乗せ率            (default 0.05)
    MIN_UPDATE_INTERVAL  int   trailing更新の最小間隔（秒）            (default 60)
    MAX_HOLD_TIME        int   最大保有期間（periods）                  (default 55)
    MIN_NOTIONAL         float 最小注文金額（円）                       (default 50000)
    MAX_PRICE_DIVERGENCE float entry vs market 価格乖離 HALT 閾値      (default 0.20)
"""

from __future__ import annotations

import datetime
import math
import os
from typing import Any

# ── Environment variables ──────────────────────────────────────────────────────
ATR_MULT             = float(os.environ.get("ATR_MULT",              2.0))
TRAIL_ATR_MULT       = float(os.environ.get("TRAIL_ATR_MULT",        1.5))
MIN_STOP_PCT         = float(os.environ.get("MIN_STOP_PCT",           0.01))
RISK_PER_TRADE       = float(os.environ.get("RISK_PER_TRADE",         0.01))
GAP_BUFFER           = float(os.environ.get("GAP_BUFFER",             0.05))
MIN_UPDATE_INTERVAL  = int(os.environ.get("MIN_UPDATE_INTERVAL",      60))
MAX_HOLD_TIME        = int(os.environ.get("MAX_HOLD_TIME",            55))
MIN_NOTIONAL         = float(os.environ.get("MIN_NOTIONAL",           50_000.0))
MAX_PRICE_DIVERGENCE = float(os.environ.get("MAX_PRICE_DIVERGENCE",   0.20))

# EXIT_PRIORITY は変更禁止。同時発火時はインデックスが小さい方のみ採用する。
EXIT_PRIORITY: list[str] = ["hard_stop", "trailing_stop", "logic", "time"]


# ── Utility helpers ────────────────────────────────────────────────────────────

def _safe(value: Any, fallback: float = 0.0) -> float:
    """NaN / inf / None → fallback。"""
    try:
        v = float(value)
        return v if math.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def _aware_utc(dt: Any) -> datetime.datetime:
    """datetime を UTC-aware に統一する。epoch 秒も受け付ける。"""
    if isinstance(dt, datetime.datetime):
        return dt if dt.tzinfo is not None else dt.replace(tzinfo=datetime.timezone.utc)
    return datetime.datetime.fromtimestamp(float(dt), tz=datetime.timezone.utc)


# ── Stop 計算 ──────────────────────────────────────────────────────────────────

def _stop_distance(atr: float, price: float) -> float:
    """
    stop_distance = max(ATR_MULT * ATR, MIN_STOP_PCT * price)

    atr=0  → MIN_STOP_PCT * price が下限として機能する。
    price=0 → ATR_MULT * atr が下限として機能する。
    両方 0  → 0.0（→ worst_case_loss ガードが qty=0 を保証）。
    """
    return max(_safe(ATR_MULT * atr), _safe(MIN_STOP_PCT * price))


def _hard_stop_price(entry_price: float, side: str, atr: float, price: float) -> float:
    """
    hard_stop 価格を計算する（entry_price 基準の固定距離）。

    long : entry - stop_distance
    short: entry + stop_distance
    """
    dist = _stop_distance(atr, price)
    if side == "long":
        return _safe(entry_price - dist, fallback=0.0)
    return _safe(entry_price + dist, fallback=0.0)


def _init_trailing(entry_price: float, side: str, atr: float) -> float:
    """
    初期 trailing_stop 価格を計算する。

    long : entry - TRAIL_ATR_MULT * ATR   (hard_stop より entry 寄り)
    short: entry + TRAIL_ATR_MULT * ATR
    """
    dist = _safe(TRAIL_ATR_MULT * atr)
    if side == "long":
        return _safe(entry_price - dist, fallback=0.0)
    return _safe(entry_price + dist, fallback=0.0)


def _advance_trailing(
    current_trailing:  float,
    price:             float,
    side:              str,
    atr:               float,
    timestamp:         datetime.datetime,
    last_update_time:  datetime.datetime,
) -> float:
    """
    trailing_stop を更新する。

    - MIN_UPDATE_INTERVAL 秒未満は更新禁止。
    - 有利方向のみ更新:
        long  → trailing = max(current, price - TRAIL_ATR_MULT * ATR)
        short → trailing = min(current, price + TRAIL_ATR_MULT * ATR)
    """
    elapsed = (timestamp - last_update_time).total_seconds()
    if elapsed < MIN_UPDATE_INTERVAL:
        return current_trailing

    candidate = (
        _safe(price - TRAIL_ATR_MULT * atr) if side == "long"
        else _safe(price + TRAIL_ATR_MULT * atr)
    )
    if side == "long":
        return max(current_trailing, candidate)
    return min(current_trailing, candidate)


# ── Exit 判定 ─────────────────────────────────────────────────────────────────

def _is_stop_hit(price: float, side: str, stop_price: float) -> bool:
    if side == "long":
        return price <= stop_price
    return price >= stop_price


def _check_exit(
    price:         float,
    side:          str,
    hard_stop:     float,
    trailing_stop: float,
    logic_exit:    bool,
    holding_time:  int,
) -> str | None:
    """
    EXIT_PRIORITY 順に評価し、最初にトリガーされた exit 名を返す。
    同時発火時は最初の 1 つのみ採用する。トリガーなし → None。

    ["hard_stop", "trailing_stop", "logic", "time"]
    """
    triggers: dict[str, bool] = {
        "hard_stop":     _is_stop_hit(price, side, hard_stop),
        "trailing_stop": _is_stop_hit(price, side, trailing_stop),
        "logic":         bool(logic_exit),
        "time":          holding_time >= MAX_HOLD_TIME,
    }
    for key in EXIT_PRIORITY:
        if triggers[key]:
            return key
    return None


# ── Position sizing ───────────────────────────────────────────────────────────

def size_position(
    equity:        float,
    stop_distance: float,
    corr_penalty:  float = 0.0,
) -> tuple[float, float, float]:
    """
    risk_amount 基準でポジションサイズを計算する（株数）。

    qty = target_risk / worst_case_loss * (1 - corr_penalty)

    corr_penalty は qty にのみ適用。stop / risk_amount には影響しない。

    Returns
    -------
    (qty, target_risk, worst_case_loss)
    qty ≥ 0 を保証する。
    """
    target_risk     = _safe(equity * RISK_PER_TRADE)
    worst_case_loss = _safe(stop_distance * (1.0 + GAP_BUFFER))

    if worst_case_loss < 1e-10:
        return 0.0, target_risk, worst_case_loss

    raw_qty = target_risk / worst_case_loss
    scale   = max(0.0, 1.0 - _safe(corr_penalty))
    qty     = max(0.0, _safe(raw_qty * scale))

    return qty, target_risk, worst_case_loss


# ── State management ──────────────────────────────────────────────────────────

def init_state(
    entry_price: float,
    side:        str,
    atr:         float,
    timestamp:   datetime.datetime,
) -> dict[str, Any]:
    """
    新規ポジションの初期 state を生成する。trailing_stop は必ず設定される。
    """
    return {
        "entry_price":      _safe(entry_price),
        "side":             side,
        "holding_time":     0,
        "max_favorable":    0.0,
        "max_adverse":      0.0,
        "trailing_stop":    _init_trailing(entry_price, side, atr),
        "last_update_time": _aware_utc(timestamp),
    }


# ── Sanity / filter checks ────────────────────────────────────────────────────

def _run_sanity(
    entry_price:        float,
    price:              float,
    trailing_stop_val:  Any,
) -> str:
    """
    整合性チェック。HALT 条件を評価する。

    trailing_stop is None              → "halt:trailing_none"
    |price - entry| / entry > MAX_PCT  → "halt:price_divergence"
    else                               → "ok"
    """
    if trailing_stop_val is None:
        return "halt:trailing_none"
    if entry_price > 1e-10:
        divergence = abs(price - entry_price) / entry_price
        if divergence > MAX_PRICE_DIVERGENCE:
            return "halt:price_divergence"
    return "ok"


def _run_preorder_filter(
    qty:        float,
    stop_price: float,
    price:      float,
) -> str:
    """
    注文前フィルタ。SKIP 条件を評価する。

    qty == 0                          → "skip:qty_zero"
    stop_price <= 0                   → "skip:stop_invalid"
    abs(qty * price) < MIN_NOTIONAL   → "skip:min_notional"
    else                              → "ok"
    """
    if qty == 0.0:
        return "skip:qty_zero"
    if stop_price <= 0.0:
        return "skip:stop_invalid"
    if abs(qty * price) < MIN_NOTIONAL:
        return "skip:min_notional"
    return "ok"


# ── Public API ────────────────────────────────────────────────────────────────

def evaluate_risk(
    price:        float,
    side:         str,
    equity:       float,
    atr:          float,
    entry_price:  float,
    state:        dict[str, Any] | None,
    logic_exit:   bool,
    corr_penalty: float,
    timestamp:    datetime.datetime,
) -> dict[str, Any]:
    """
    リスク評価・ストップ計算・ポジションサイジングを一元管理する。
    純粋関数 — 副作用なし・外部API呼び出しなし・決定論的。

    Parameters
    ----------
    price        : 現在価格
    side         : "long" | "short"
    equity       : 現在の資産額
    atr          : Average True Range（現在値）
    entry_price  : エントリー価格（state=None 時に state を初期化する）
    state        : 現在のポジション state。None = 新規ポジション
    logic_exit   : RSRシグナルによる退出フラグ
    corr_penalty : 相関ペナルティ ∈ [0, 1]（qty のみに適用）
    timestamp    : 現在時刻（UTC datetime 推奨）

    Returns
    -------
    {
      "qty":           float,        # 推奨ポジションサイズ（株数）≥ 0
      "stop_price":    float,        # hard_stop 価格
      "trailing_stop": float,        # trailing_stop 価格（更新後）
      "risk_amount":   float,        # target_risk 金額
      "exit_signal":   str | None,   # "hard_stop"|"trailing_stop"|"logic"|"time"|None
      "updated_state": dict,         # 更新済み state（必ず返却）
      "sanity_status": str,          # "ok"|"skip:*"|"halt:*"
      "log_info":      dict,         # {state, inputs, decision, qty, stop}
    }

    State 更新順序（厳守）:
      1) 新価格受信 + 入力サニタイズ
      2) state 初期化 or 既存 state 継承
      3) 整合性チェック（trailing_none → full halt）
      4) max_favorable / max_adverse / holding_time 更新
      5) trailing_stop 更新（MIN_UPDATE_INTERVAL 制約）
      6) exit 判定（EXIT_PRIORITY 順: hard → trailing → logic → time）
      7) risk / qty 再計算
      8) 注文前フィルタ（price_divergence halt + qty/stop/notional skip）
      9) 出力構築

    Guarantees
    ----------
    - ゼロ除算なし（worst_case_loss=0 → qty=0）
    - NaN / inf 入力はすべて fallback=0.0 に変換
    - qty は負にならない
    - corr_penalty は qty にのみ適用
    - updated_state は必ず返却（None 入力でも初期化して返す）
    - trailing_stop は None を返さない（init_state が保証）
    """
    ts = _aware_utc(timestamp)

    # ── 1. 入力サニタイズ ──────────────────────────────────────────────────────
    price       = _safe(price,       fallback=0.0)
    equity      = _safe(equity,      fallback=0.0)
    atr         = _safe(atr,         fallback=0.0)
    entry_price = _safe(entry_price, fallback=price)
    side        = side if side in ("long", "short") else "long"

    inputs_snap: dict[str, Any] = {
        "price":        price,
        "side":         side,
        "equity":       equity,
        "atr":          atr,
        "entry_price":  entry_price,
        "logic_exit":   logic_exit,
        "corr_penalty": corr_penalty,
        "timestamp":    ts.isoformat(),
    }

    # ── 2. state 初期化 ────────────────────────────────────────────────────────
    if state is None:
        state = init_state(entry_price, side, atr, ts)

    # 既存ポジションは state の entry_price / side を優先
    ep   = _safe(state.get("entry_price", entry_price), fallback=price)
    side = state.get("side", side)

    state_snap = dict(state)  # log 用スナップショット（更新前）

    # ── 3. 整合性チェック: trailing_stop is None → full halt ──────────────────
    raw_trailing = state.get("trailing_stop")
    if raw_trailing is None:
        log_info: dict[str, Any] = {
            "state":    state_snap,
            "inputs":   inputs_snap,
            "decision": "halt",
            "qty":      0.0,
            "stop":     0.0,
        }
        return {
            "qty":           0.0,
            "stop_price":    0.0,
            "trailing_stop": 0.0,
            "risk_amount":   0.0,
            "exit_signal":   None,
            "updated_state": dict(state),
            "sanity_status": "halt:trailing_none",
            "log_info":      log_info,
        }

    # ── 4. state 更新: favorable / adverse / holding_time ─────────────────────
    movement   = (price - ep) if side == "long" else (ep - price)
    new_state  = dict(state)
    new_state["holding_time"]  = int(_safe(state.get("holding_time", 0))) + 1
    new_state["max_favorable"] = max(_safe(state.get("max_favorable", 0.0)),  movement)
    new_state["max_adverse"]   = max(_safe(state.get("max_adverse",   0.0)), -movement)

    # ── 5. trailing_stop 更新（有利方向のみ / MIN_UPDATE_INTERVAL 制約） ─────
    current_trailing = _safe(raw_trailing, fallback=_init_trailing(ep, side, atr))
    last_upd = state.get("last_update_time", ts)
    if not isinstance(last_upd, datetime.datetime):
        last_upd = ts
    last_upd = _aware_utc(last_upd)

    new_trailing = _advance_trailing(current_trailing, price, side, atr, ts, last_upd)
    new_state["trailing_stop"]    = new_trailing
    new_state["last_update_time"] = ts

    # ── 6. exit 判定（EXIT_PRIORITY 順） ──────────────────────────────────────
    hard_stop   = _hard_stop_price(ep, side, atr, price)
    exit_signal = _check_exit(
        price, side, hard_stop, new_trailing,
        logic_exit, new_state["holding_time"],
    )

    # ── 7. risk / qty 再計算 ──────────────────────────────────────────────────
    dist             = _stop_distance(atr, price)
    qty, risk_amt, _ = size_position(equity, dist, corr_penalty)

    # ── 8. フィルタ: price_divergence halt → preorder skip ────────────────────
    sanity = _run_sanity(ep, price, new_trailing)

    if sanity.startswith("halt"):
        # price_divergence HALT: exit_signal は維持（既存ポジション保護）
        decision  = "halt"
        final_qty = 0.0
    else:
        filter_res = _run_preorder_filter(qty, hard_stop, price)
        if filter_res != "ok":
            sanity    = filter_res
            decision  = "skip"
            final_qty = 0.0
        else:
            sanity    = "ok"
            decision  = "execute"
            final_qty = qty

    # ── 9. 出力構築 ────────────────────────────────────────────────────────────
    log_info = {
        "state":    state_snap,
        "inputs":   inputs_snap,
        "decision": decision,
        "qty":      final_qty,
        "stop":     hard_stop,
    }

    return {
        "qty":           round(final_qty,    6),
        "stop_price":    round(hard_stop,    6),
        "trailing_stop": round(new_trailing, 6),
        "risk_amount":   round(risk_amt,     6),
        "exit_signal":   exit_signal,
        "updated_state": new_state,
        "sanity_status": sanity,
        "log_info":      log_info,
    }
