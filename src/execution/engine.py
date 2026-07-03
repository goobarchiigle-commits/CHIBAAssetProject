"""
engine.py — 実行エンジン 強化版（適応 + リスク統治）

目的: 実運用における実現Sharpeの劣化を最小化する。
     オンラインパラメータ更新・段階型ドローダウン制御・ボラティリティ急変対応を含む。

Environment variables:
    MAX_SIGNAL_AGE    int   シグナル有効期限（秒）                  (default 60)
    N_SLICES          int   TWAP分割数                              (default 5)
    COST_BPS          float 市場インパクト基準コスト (bps)           (default 10)
    MIN_EXPECTED_EDGE float 最小期待エッジ                          (default 0.0)
    MAX_PARAM_CHANGE  float パラメータ変化率の上限（絶対値）         (default 0.1)
    VOL_SPIKE         float realized/rolling vol比率の急変閾値      (default 2.0)
    VOL_SHORT_WIN     int   実現ボラティリティの短期ウィンドウ       (default 5)
    VOL_LONG_WIN      int   ローリングボラティリティの長期ウィンドウ (default 20)
    VOL_SCALE         float ボラティリティ急変時のサイズ縮小率       (default 0.5)
    MIN_POLY_OBS      int   polyfit に必要な最小観測数               (default 5)
    RANDOM_SKIP_PROB  float ランダムSKIP確率（探索・アンチフロントラン）(default 0.05)
    RANDOM_NOISE_PROB float ランダムノイズ付与確率                   (default 0.05)
    RANDOM_NOISE_FRAC float ノイズ幅（±分率）                        (default 0.10)
"""

from __future__ import annotations

import datetime
import math
import os
import random
import subprocess
from typing import Any

import numpy as np

# ── Environment variables ──────────────────────────────────────────────────────
MAX_SIGNAL_AGE    = int(os.environ.get("MAX_SIGNAL_AGE",     60))
N_SLICES          = int(os.environ.get("N_SLICES",            5))
COST_BPS          = float(os.environ.get("COST_BPS",          10.0))
MIN_EXPECTED_EDGE = float(os.environ.get("MIN_EXPECTED_EDGE",  0.0))
MAX_PARAM_CHANGE  = float(os.environ.get("MAX_PARAM_CHANGE",   0.1))
VOL_SPIKE         = float(os.environ.get("VOL_SPIKE",          2.0))
VOL_SHORT_WIN     = int(os.environ.get("VOL_SHORT_WIN",        5))
VOL_LONG_WIN      = int(os.environ.get("VOL_LONG_WIN",         20))
VOL_SCALE         = float(os.environ.get("VOL_SCALE",          0.5))
MIN_POLY_OBS      = int(os.environ.get("MIN_POLY_OBS",         5))
RANDOM_SKIP_PROB  = float(os.environ.get("RANDOM_SKIP_PROB",   0.05))
RANDOM_NOISE_PROB = float(os.environ.get("RANDOM_NOISE_PROB",  0.05))
RANDOM_NOISE_FRAC = float(os.environ.get("RANDOM_NOISE_FRAC",  0.10))

# ── Internal constants ─────────────────────────────────────────────────────────
_MAX_PARTICIPATION  = 0.10
_PENALTY_DECAY      = 0.05
_PENALTY_GROWTH     = 0.30
_FILL_GOOD_THRESH   = 0.90
_DD_REDUCE_1        = -0.05    # → size × 0.50
_DD_REDUCE_2        = -0.08    # → size × 0.20
_DD_HALT            = -0.10    # → HALT
_SHARPE_REVERT_MIN  = 0.0     # これ未満のSharpeでパラメータ更新をリバート
_AUDIT_ENV_KEYS     = frozenset({
    "MAX_SIGNAL_AGE", "N_SLICES", "COST_BPS", "MIN_EXPECTED_EDGE",
    "MAX_PARAM_CHANGE", "VOL_SPIKE", "VOL_SHORT_WIN", "VOL_LONG_WIN",
    "VOL_SCALE", "MIN_POLY_OBS",
})

# デフォルトパラメータ（params=None 時に使用）
DEFAULT_PARAMS: dict[str, Any] = {
    "impact_coef":       1.0,    # sqrt市場インパクトモデルの係数
    "poly_coefs":        None,   # [a, b, c] from polyfit(2), or None
    "max_participation": _MAX_PARTICIPATION,
}


# ── Utility helpers ────────────────────────────────────────────────────────────

def _safe(value: Any, fallback: float = 0.0) -> float:
    """NaN / inf / None → fallback。ゼロ除算ガード用。"""
    try:
        v = float(value)
        return v if math.isfinite(v) else fallback
    except (TypeError, ValueError):
        return fallback


def _aware_utc(dt: datetime.datetime) -> datetime.datetime:
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=datetime.timezone.utc)


def _now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _git_hash() -> str:
    """git rev-parse --short HEAD。失敗時は "unknown"。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# ── Statistical helpers ────────────────────────────────────────────────────────

def _compute_sharpe(equity_curve: list[float]) -> float:
    """equity_curve の日次リターンから年次化Sharpeを推定する。"""
    if len(equity_curve) < 3:
        return 0.0
    arr  = np.array(equity_curve, dtype=float)
    rets = np.diff(arr) / arr[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return 0.0
    std = float(np.std(rets, ddof=1))
    if std < 1e-10:
        return 0.0
    return float(np.mean(rets) / std * math.sqrt(252))


def _compute_drawdown(equity_curve: list[float]) -> float:
    """現時点のドローダウン比率（ピーク比）。"""
    if len(equity_curve) < 2:
        return 0.0
    arr = np.array(equity_curve, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 2:
        return 0.0
    peak    = float(np.maximum.accumulate(arr)[-1])
    current = float(arr[-1])
    if peak <= 0.0:
        return 0.0
    return (current - peak) / peak


def _compute_vols(equity_curve: list[float]) -> tuple[float, float]:
    """(realized_vol, rolling_vol) を返す。データ不足時は (0.0, 0.0)。"""
    if len(equity_curve) < 3:
        return 0.0, 0.0
    arr  = np.array(equity_curve, dtype=float)
    rets = np.diff(arr) / arr[:-1]
    rets = rets[np.isfinite(rets)]
    if len(rets) < 2:
        return 0.0, 0.0
    rolling_vol  = float(np.std(rets, ddof=1))
    short_window = rets[-VOL_SHORT_WIN:] if len(rets) >= VOL_SHORT_WIN else rets
    realized_vol = float(np.std(short_window, ddof=1)) if len(short_window) >= 2 else 0.0
    return realized_vol, rolling_vol


def _compute_risk_state(
    equity_curve: list[float],
) -> tuple[str, float, float, float, float]:
    """
    Returns (risk_state, size_multiplier, drawdown, realized_vol, rolling_vol).

    段階型ドローダウン制御:
      DD > -5%         → normal   multiplier=1.0
      -8% < DD ≤ -5%  → reduced  multiplier=0.50
      -10% < DD ≤ -8% → reduced  multiplier=0.20
      DD ≤ -10%        → halt     multiplier=0.0

    ボラティリティ急変:
      realized_vol > rolling_vol * VOL_SPIKE → multiplier × VOL_SCALE
    """
    dd = _compute_drawdown(equity_curve) if equity_curve else 0.0
    realized_vol, rolling_vol = _compute_vols(equity_curve) if equity_curve else (0.0, 0.0)

    if dd <= _DD_HALT:
        return "halt", 0.0, dd, realized_vol, rolling_vol

    multiplier = 1.0
    state      = "normal"

    if dd <= _DD_REDUCE_2:
        multiplier, state = 0.20, "reduced"
    elif dd <= _DD_REDUCE_1:
        multiplier, state = 0.50, "reduced"

    # ボラティリティ急変（DD 制御と独立して乗算）
    if rolling_vol > 0.0 and realized_vol > rolling_vol * VOL_SPIKE:
        multiplier = max(0.0, multiplier * VOL_SCALE)
        state      = "reduced"

    return state, multiplier, dd, realized_vol, rolling_vol


# ── Slippage model ─────────────────────────────────────────────────────────────

def _estimate_slippage(
    participation: float,
    poly_coefs:    list[float] | None,
    impact_coef:   float,
    base_cost:     float,
) -> float:
    """
    参加率からスリッページを推定する。

    poly_coefs が利用可能: a*p^2 + b*p + c  (polyfit由来)
    else                 : base_cost * impact_coef * sqrt(p)  (sqrt モデル)
    """
    p = max(0.0, participation)
    if poly_coefs is not None:
        a, b, c = poly_coefs
        return max(0.0, _safe(a * p ** 2 + b * p + c))
    return _safe(base_cost * impact_coef * math.sqrt(p))


def _fit_slippage_poly(
    execution_log: list[dict[str, Any]],
) -> list[float] | None:
    """
    execution_log から 2次多項式を当てはめる。
    "slippage" キーがなければ "expected_cost" をフォールバックに使用。
    EXECUTE 決定のみを対象とする。
    観測数 < MIN_POLY_OBS → None。
    """
    ps, ss = [], []
    for rec in execution_log:
        if rec.get("decision") not in (None, "EXECUTE"):
            continue
        p = _safe(rec.get("participation", 0.0))
        s = _safe(rec.get("slippage", rec.get("expected_cost", 0.0)))
        if p > 0.0 and s >= 0.0:
            ps.append(p)
            ss.append(s)
    if len(ps) < MIN_POLY_OBS:
        return None
    try:
        coefs = np.polyfit(ps, ss, 2).tolist()
        if not all(math.isfinite(c) for c in coefs):
            return None
        return coefs
    except Exception:
        return None


def _update_impact_coef(
    current_coef:  float,
    base_cost:     float,
    execution_log: list[dict[str, Any]],
) -> float:
    """
    実績スリッページ / (base_cost * sqrt(participation)) の平均から
    経験的 impact_coef を推定する。観測なし → current_coef を返す。
    """
    implied: list[float] = []
    for rec in execution_log:
        if rec.get("decision") not in (None, "EXECUTE"):
            continue
        p = _safe(rec.get("participation", 0.0))
        s = _safe(rec.get("slippage", rec.get("expected_cost", 0.0)))
        denom = base_cost * math.sqrt(p) if p > 0.0 else 0.0
        if denom > 1e-12 and s >= 0.0:
            implied.append(s / denom)
    if not implied:
        return current_coef
    return float(np.mean(implied))


def _estimate_cost_improvement(
    execution_log: list[dict[str, Any]],
    old_coefs:     list[float] | None,
    new_coefs:     list[float] | None,
    old_impact:    float,
    new_impact:    float,
    base_cost:     float,
) -> float:
    """
    新パラメータの予測精度改善量を返す（正 = 新モデルの方が観測値に近い）。

    improvement = Σ |old_error| - |new_error|
    where error = model_prediction - observed_slippage

    コスト削減ではなく予測精度で判定する。
    観測スリッページが sqrt モデルを大きく上回る場合、
    polyfit モデルの方が正確 → acceptance が正当化される。
    """
    total = 0.0
    for rec in execution_log:
        p = _safe(rec.get("participation", 0.0))
        s = _safe(rec.get("slippage", rec.get("expected_cost", 0.0)))
        if p <= 0.0:
            continue
        err_old = abs(_estimate_slippage(p, old_coefs, old_impact, base_cost) - s)
        err_new = abs(_estimate_slippage(p, new_coefs, new_impact, base_cost) - s)
        total  += err_old - err_new
    return total


# ── Public: online parameter update ───────────────────────────────────────────

def update_param(
    params:        dict[str, Any],
    execution_log: list[dict[str, Any]],
    equity_curve:  list[float],
) -> dict[str, Any]:
    """
    実行ログからスリッページモデルパラメータをオンライン更新する。

    更新ロジック
    ────────────
    1. execution_log から polyfit(2次) で poly_coefs 候補を推定
    2. execution_log から経験的 impact_coef 候補を推定
    3. MAX_PARAM_CHANGE で impact_coef の変化率をクランプ
    4. 検証ゲート:
       - equity_curve のSharpe < _SHARPE_REVERT_MIN  → リバート
       - 新パラメータのコスト改善量 < 0               → リバート
       - 両条件をパスした場合のみ更新を採用

    Parameters
    ----------
    params        : 現在のパラメータ dict (DEFAULT_PARAMS 参照)
    execution_log : [{participation, slippage/expected_cost, decision, ...}]
    equity_curve  : 累積資産額の時系列リスト

    Returns
    -------
    更新済みパラメータ dict（呼び出し元の dict は変更しない）
    """
    new_params = dict(params)
    base_cost  = COST_BPS / 10_000.0
    old_impact = _safe(params.get("impact_coef", 1.0), fallback=1.0)
    old_coefs  = params.get("poly_coefs")

    # 1. Sharpe 検証（リバート判定基準）
    pre_sharpe = _compute_sharpe(equity_curve)

    # 2. 新パラメータ候補
    new_coefs  = _fit_slippage_poly(execution_log)
    raw_impact = _update_impact_coef(old_impact, base_cost, execution_log)

    # 3. MAX_PARAM_CHANGE クランプ
    delta         = raw_impact - old_impact
    clamped_delta = max(-MAX_PARAM_CHANGE, min(MAX_PARAM_CHANGE, delta))
    new_impact    = old_impact + clamped_delta

    # 4. コスト改善チェック
    improvement = _estimate_cost_improvement(
        execution_log, old_coefs, new_coefs, old_impact, new_impact, base_cost,
    )

    # 5. 検証ゲート（両条件クリアで採用）
    if pre_sharpe >= _SHARPE_REVERT_MIN and improvement >= 0.0:
        new_params["impact_coef"] = round(new_impact, 6)
        if new_coefs is not None:
            new_params["poly_coefs"] = [round(c, 8) for c in new_coefs]

    return new_params


# ── Public: execution ──────────────────────────────────────────────────────────

def run_execution(
    target_sizes:      dict[str, int],
    market_data:       dict[str, dict[str, Any]],
    signal_timestamp:  datetime.datetime,
    prev_fills:        dict[str, float],
    liquidity_penalty: dict[str, float],
    equity_curve:      list[float] | None          = None,
    execution_log:     list[dict[str, Any]] | None = None,
    params:            dict[str, Any] | None       = None,
    now:               datetime.datetime | None    = None,
    seed:              int | None                  = None,
) -> dict[str, Any]:
    """
    実行エンジン 強化版。純粋関数 — 外部API呼び出しなし・スリープなし。

    Parameters
    ----------
    target_sizes      : {symbol: target_qty (株数)}
    market_data       : {symbol: {"open": float, "volume": float,
                                  "expected_alpha": float}}
    signal_timestamp  : シグナル生成時刻 (UTC datetime)
    prev_fills        : {symbol: fill_ratio ∈ [0.0, 1.0]}。初回は {}
    liquidity_penalty : {symbol: penalty_factor ≥ 0.0}。初回は {}
    equity_curve      : 累積資産額リスト（DD・ボラティリティ制御用）
    execution_log     : 過去実行記録リスト（スリッページモデル更新用）
    params            : 可変パラメータ dict（None → DEFAULT_PARAMS）
    now               : 現在時刻 (UTC)。None → UTC now
    seed              : ランダム化の再現性用シード（None → 非決定的）

    Returns
    -------
    {
      "executed_orders": [{"symbol": str, "child_qty": int}, ...],
      "updated_penalty": {symbol: float},
      "logs":            [{symbol, decision, expected_cost, expected_alpha,
                           participation, fill_ratio}, ...],
      "updated_params":  dict,
      "risk_state":      "normal" | "reduced" | "halt",
      "audit_logs":      {timestamp, env, code_version, params_snapshot,
                          risk_state, drawdown, realized_vol, rolling_vol},
    }

    Decision ladder (銘柄ごと・先着一致):
      HALT        DD ≤ -10%                     → 全銘柄ブロック
      STALE       age > MAX_SIGNAL_AGE           → qty=0
      RANDOM_SKIP 5%確率                         → qty=0（探索）
      SKIP_COST   alpha ≤ expected_cost          → qty=0
      SKIP_DEPTH  depth-constrained qty == 0     → qty=0
      SKIP_RISK   リスク縮小後 qty == 0          → qty=0
      EXECUTE     TWAPスプリット（±5%確率ノイズ）

    Guarantees
    ----------
    - ゼロ除算なし（volume=0 は max_participation にフォールバック）
    - NaN/inf 入力はすべて 0.0 にコアース後に判定
    - liquidity_penalty の元 dict は変更しない（shallow copy）
    - seed 指定時はランダム化が決定的
    """
    if now is None:
        now = _now_utc()
    now              = _aware_utc(now)
    signal_timestamp = _aware_utc(signal_timestamp)

    equity_curve  = equity_curve  or []
    execution_log = execution_log or []

    effective_params = dict(DEFAULT_PARAMS)
    effective_params.update(params or {})

    # ── パラメータ更新 ─────────────────────────────────────────────────────────
    updated_params = update_param(effective_params, execution_log, equity_curve)

    # ── リスク状態 ─────────────────────────────────────────────────────────────
    risk_state, size_mult, dd, realized_vol, rolling_vol = _compute_risk_state(equity_curve)

    # ── Audit log ──────────────────────────────────────────────────────────────
    audit_logs: dict[str, Any] = {
        "timestamp":       now.isoformat(),
        "env":             {k: os.environ.get(k, "") for k in sorted(_AUDIT_ENV_KEYS)},
        "code_version":    _git_hash(),
        "params_snapshot": dict(updated_params),
        "risk_state":      risk_state,
        "drawdown":        round(dd, 6),
        "realized_vol":    round(realized_vol, 6),
        "rolling_vol":     round(rolling_vol, 6),
    }

    # ── HALT: 全注文をブロック ─────────────────────────────────────────────────
    if risk_state == "halt":
        return {
            "executed_orders": [],
            "updated_penalty": dict(liquidity_penalty),
            "logs": [
                _log(sym, "HALT", 0.0, 0.0, 0.0, _safe(prev_fills.get(sym, 1.0)))
                for sym in target_sizes
            ],
            "updated_params": updated_params,
            "risk_state":     "halt",
            "audit_logs":     audit_logs,
        }

    # ── RNG（シード指定可能） ──────────────────────────────────────────────────
    rng = random.Random(seed)

    # ── 実行パラメータ ─────────────────────────────────────────────────────────
    base_cost   = COST_BPS / 10_000.0
    impact_coef = _safe(updated_params.get("impact_coef", 1.0), fallback=1.0)
    poly_coefs  = updated_params.get("poly_coefs")
    max_part    = _safe(
        updated_params.get("max_participation", _MAX_PARTICIPATION),
        fallback=_MAX_PARTICIPATION,
    )

    age_seconds  = (now - signal_timestamp).total_seconds()
    signal_stale = age_seconds > MAX_SIGNAL_AGE

    executed_orders: list[dict[str, Any]] = []
    updated_penalty: dict[str, float]     = dict(liquidity_penalty)
    logs: list[dict[str, Any]]            = []

    for sym, target_qty in target_sizes.items():
        sym_data       = market_data.get(sym, {})
        expected_alpha = _safe(sym_data.get("expected_alpha", MIN_EXPECTED_EDGE))
        volume         = _safe(sym_data.get("volume", 0.0))
        penalty        = _safe(liquidity_penalty.get(sym, 0.0))
        prev_fill      = _safe(prev_fills.get(sym, 1.0), fallback=1.0)

        # ── 1. シグナル鮮度 ────────────────────────────────────────────────────
        if signal_stale:
            _update_penalty(updated_penalty, sym, prev_fill)
            logs.append(_log(sym, "STALE", 0.0, expected_alpha, 0.0, prev_fill))
            continue

        # ── 2. ランダムSKIP（探索・アンチフロントラン） ───────────────────────
        r = rng.random()
        if r < RANDOM_SKIP_PROB:
            logs.append(_log(sym, "RANDOM_SKIP", 0.0, expected_alpha, 0.0, prev_fill))
            continue

        # ── 3. 参加率（ブックデプス制約前） ────────────────────────────────────
        participation_raw = float(target_qty) / volume if volume > 0.0 else max_part

        # ── 4. コスト推定 ──────────────────────────────────────────────────────
        impact        = _estimate_slippage(participation_raw, poly_coefs, impact_coef, base_cost)
        expected_cost = _safe(impact + base_cost * penalty)

        # ── 5. アルファゲート ──────────────────────────────────────────────────
        if expected_alpha <= expected_cost:
            _update_penalty(updated_penalty, sym, prev_fill)
            logs.append(_log(sym, "SKIP_COST", expected_cost, expected_alpha,
                             participation_raw, prev_fill))
            continue

        # ── 6. ブックデプス制約 ────────────────────────────────────────────────
        if volume > 0.0:
            adj_qty = min(int(target_qty), int(volume * max_part))
        else:
            adj_qty = int(target_qty)

        if adj_qty <= 0:
            _update_penalty(updated_penalty, sym, prev_fill)
            logs.append(_log(sym, "SKIP_DEPTH", expected_cost, expected_alpha,
                             participation_raw, prev_fill))
            continue

        # ── 7. リスク縮小（ドローダウン / ボラティリティ急変） ─────────────────
        adj_qty = max(0, int(adj_qty * size_mult))
        if adj_qty == 0:
            logs.append(_log(sym, "SKIP_RISK", expected_cost, expected_alpha,
                             participation_raw, prev_fill))
            continue

        # ── 8. ランダムノイズ（±RANDOM_NOISE_FRAC） ───────────────────────────
        # r ∈ [RANDOM_SKIP_PROB, RANDOM_SKIP_PROB + RANDOM_NOISE_PROB): noise applied
        if r < RANDOM_SKIP_PROB + RANDOM_NOISE_PROB:
            noise   = rng.uniform(-RANDOM_NOISE_FRAC, RANDOM_NOISE_FRAC)
            adj_qty = max(1, int(adj_qty * (1.0 + noise)))

        # ── 9. 制約後の参加率 ──────────────────────────────────────────────────
        participation = float(adj_qty) / volume if volume > 0.0 else participation_raw

        # ── 10. TWAP スプリット ────────────────────────────────────────────────
        executed_orders.extend(_twap_split(sym, adj_qty))

        # ── 11. フィルフィードバック ────────────────────────────────────────────
        _update_penalty(updated_penalty, sym, prev_fill)
        logs.append(_log(sym, "EXECUTE", expected_cost, expected_alpha,
                         participation, prev_fill))

    return {
        "executed_orders": executed_orders,
        "updated_penalty": updated_penalty,
        "logs":            logs,
        "updated_params":  updated_params,
        "risk_state":      risk_state,
        "audit_logs":      audit_logs,
    }


# ── Internal helpers ───────────────────────────────────────────────────────────

def _twap_split(sym: str, qty: int) -> list[dict[str, Any]]:
    """qty を N_SLICES 個の子注文に均等分割する。端数は最終スライスに加算。"""
    if N_SLICES <= 0 or qty <= 0:
        return []
    base   = qty // N_SLICES
    rem    = qty % N_SLICES
    result = []
    for i in range(N_SLICES):
        child_qty = base + (rem if i == N_SLICES - 1 else 0)
        if child_qty > 0:
            result.append({"symbol": sym, "child_qty": child_qty})
    return result


def _update_penalty(
    penalty_map: dict[str, float],
    sym:         str,
    fill_ratio:  float,
) -> None:
    """
    fill_ratio に応じてペナルティを更新（in-place）。
    fill_ratio ≥ _FILL_GOOD_THRESH → 指数減衰
    fill_ratio <  _FILL_GOOD_THRESH → shortfall 比例増加
    """
    current = _safe(penalty_map.get(sym, 0.0))
    if fill_ratio >= _FILL_GOOD_THRESH:
        updated = max(0.0, current * (1.0 - _PENALTY_DECAY))
    else:
        shortfall = max(0.0, 1.0 - fill_ratio)
        updated   = current + _PENALTY_GROWTH * shortfall
    penalty_map[sym] = round(updated, 6)


def _log(
    sym:            str,
    decision:       str,
    expected_cost:  float,
    expected_alpha: float,
    participation:  float,
    fill_ratio:     float,
) -> dict[str, Any]:
    return {
        "symbol":         sym,
        "decision":       decision,
        "expected_cost":  round(_safe(expected_cost),  8),
        "expected_alpha": round(_safe(expected_alpha), 8),
        "participation":  round(_safe(participation),  6),
        "fill_ratio":     round(_safe(fill_ratio),     6),
    }
