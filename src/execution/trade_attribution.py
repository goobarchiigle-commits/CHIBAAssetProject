"""
src/execution/trade_attribution.py
トレード収益の要因分解記録。副作用なし。先読みなし。O(1) 追加コスト/トレード。

出力列 (必須):
    pnl, alpha_score_at_entry, regime_label, position_size,
    holding_period, entry_volatility, exit_reason

派生メトリクス (O(1) 計算):
    pnl_per_time     = pnl / holding_period
    pnl_per_risk     = pnl / (entry_volatility * position_size)
    alpha_efficiency = pnl / alpha_score_at_entry

制約:
    NaN禁止（default埋め） / dtype明示（float32, int32） / append-only
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ── デフォルト値（NaN/ゼロ除算回避） ────────────────────────────────────────
_DEF_FLOAT: float = 0.0
_DEF_INT:   int   = 1       # holding_period=0 は除算エラーになるため最小 1
_DEF_STR:   str   = "unknown"

# ── 列定義 ───────────────────────────────────────────────────────────────────
_FLOAT32_COLS: tuple[str, ...] = (
    "pnl",
    "alpha_score_at_entry",
    "position_size",
    "entry_volatility",
    "pnl_per_time",
    "pnl_per_risk",
    "alpha_efficiency",
)
_INT32_COLS: tuple[str, ...] = ("holding_period",)
_STR_COLS:   tuple[str, ...] = ("regime_label", "exit_reason")

_ALL_COLS: tuple[str, ...] = (
    "pnl",
    "alpha_score_at_entry",
    "regime_label",
    "position_size",
    "holding_period",
    "entry_volatility",
    "exit_reason",
    "pnl_per_time",
    "pnl_per_risk",
    "alpha_efficiency",
)


# ── 入力安全変換 ─────────────────────────────────────────────────────────────

def _safe_f(v: Any, fallback: float = _DEF_FLOAT) -> float:
    """NaN / inf / None → fallback (float)。"""
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _safe_i(v: Any, fallback: int = _DEF_INT) -> int:
    """None / 非整数 → fallback。最小値 1 を保証（ゼロ除算回避）。"""
    try:
        return max(1, int(v))
    except (TypeError, ValueError):
        return fallback


def _safe_s(v: Any, fallback: str = _DEF_STR) -> str:
    """None / 空文字 → fallback (str)。"""
    if v is None:
        return fallback
    s = str(v).strip()
    return s if s else fallback


# ── O(1) 派生メトリクス ──────────────────────────────────────────────────────

def _pnl_per_time(pnl: float, holding_period: int) -> float:
    """pnl / holding_period。holding_period は最小 1 保証済み。"""
    return _safe_f(pnl / float(holding_period))


def _pnl_per_risk(pnl: float, entry_vol: float, position_size: float) -> float:
    """pnl / (entry_volatility × position_size)。ゼロ除算 → 0.0。"""
    denom = entry_vol * position_size
    return _safe_f(pnl / denom) if abs(denom) > 1e-12 else 0.0


def _alpha_efficiency(pnl: float, alpha_score: float) -> float:
    """pnl / alpha_score_at_entry。ゼロ除算 → 0.0。"""
    return _safe_f(pnl / alpha_score) if abs(alpha_score) > 1e-12 else 0.0


# ── 公開 API ─────────────────────────────────────────────────────────────────

def make_trade_record(
    pnl:                  float | Any,
    alpha_score_at_entry: float | Any,
    regime_label:         str   | Any,
    position_size:        float | Any,
    holding_period:       int   | Any,
    entry_volatility:     float | Any,
    exit_reason:          str   | Any,
) -> dict[str, Any]:
    """
    1 トレード分の収益要因分解レコードを生成する（純粋関数）。

    - 全入力を NaN / inf / None 安全に処理（default 埋め）
    - 派生メトリクス 3 本を O(1) で計算
    - 既存 execution pipeline に副作用なし
    - 先読みなし（エントリー時点の情報のみ使用）

    Args:
        pnl                  : 実現損益
        alpha_score_at_entry : エントリー時のアルファスコア
        regime_label         : 市場レジームラベル（例: "bull_trend"）
        position_size        : ポジションサイズ（株数・数量）
        holding_period       : 保有期間（バー数）
        entry_volatility     : エントリー時ボラティリティ（ATR 等）
        exit_reason          : 決済理由（例: "turtle_exit", "stop_loss"）

    Returns:
        _ALL_COLS 順序の dict。全 float 値が有限。
    """
    p   = _safe_f(pnl)
    a   = _safe_f(alpha_score_at_entry)
    r   = _safe_s(regime_label)
    ps  = _safe_f(position_size)
    hp  = _safe_i(holding_period)
    ev  = _safe_f(entry_volatility)
    er  = _safe_s(exit_reason)

    return {
        "pnl":                  np.float32(p),
        "alpha_score_at_entry": np.float32(a),
        "regime_label":         r,
        "position_size":        np.float32(ps),
        "holding_period":       np.int32(hp),
        "entry_volatility":     np.float32(ev),
        "exit_reason":          er,
        "pnl_per_time":         np.float32(_pnl_per_time(p, hp)),
        "pnl_per_risk":         np.float32(_pnl_per_risk(p, ev, ps)),
        "alpha_efficiency":     np.float32(_alpha_efficiency(p, a)),
    }


def records_to_df(records: list[dict[str, Any]]) -> pd.DataFrame:
    """
    make_trade_record() の出力リストを型付き DataFrame に変換する。

    空リスト → 空 DataFrame（正しい dtype 付き）。
    """
    if not records:
        dtype_map = (
            {c: "float32" for c in _FLOAT32_COLS}
            | {c: "int32"  for c in _INT32_COLS}
        )
        return pd.DataFrame(columns=list(_ALL_COLS)).astype(dtype_map)

    df = pd.DataFrame(records, columns=list(_ALL_COLS))
    for c in _FLOAT32_COLS:
        df[c] = df[c].astype("float32")
    for c in _INT32_COLS:
        df[c] = df[c].astype("int32")
    return df


def append_trade_log(path: str | Path, record: dict[str, Any]) -> None:
    """
    1 レコードを CSV に追記する（append-only）。

    - ファイル未存在 / 空 → ヘッダー付きで新規作成
    - 既存ファイルには末尾追記のみ（過去行変更禁止）
    - numpy スカラーを Python ネイティブ型に変換して書き込む

    Args:
        path   : 出力 CSV パス
        record : make_trade_record() の戻り値
    """
    p = Path(path)
    write_header = (not p.exists()) or (p.stat().st_size == 0)

    row = {k: (v.item() if hasattr(v, "item") else v) for k, v in record.items()}

    with p.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(_ALL_COLS))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_trade_log(path: str | Path) -> pd.DataFrame:
    """
    append_trade_log() で作成した CSV を型付き DataFrame として読み込む。

    ファイル未存在 / 空 → 空 DataFrame（records_to_df([]) と同じ）。
    破損値は dtype ごとのデフォルト値で埋める（NaN禁止）。
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return records_to_df([])

    df = pd.read_csv(p)
    for c in _FLOAT32_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(_DEF_FLOAT).astype("float32")
    for c in _INT32_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(_DEF_INT).clip(lower=1).astype("int32")
    for c in _STR_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(_DEF_STR).astype(str)
    return df


# ── 使用例 ───────────────────────────────────────────────────────────────────

def _example() -> None:
    import os
    import sys
    import tempfile
    sys.stdout.reconfigure(encoding="utf-8")

    # 正常系レコード
    rec = make_trade_record(
        pnl=1_500.0,
        alpha_score_at_entry=0.75,
        regime_label="bull_trend",
        position_size=1_000.0,
        holding_period=10,
        entry_volatility=0.015,
        exit_reason="turtle_exit",
    )
    print("=== make_trade_record ===")
    for k, v in rec.items():
        print(f"  {k}: {v!r:>20}  ({type(v).__name__})")

    # NaN / None / ゼロ holding_period テスト
    bad = make_trade_record(
        pnl=float("nan"),
        alpha_score_at_entry=None,
        regime_label=None,
        position_size=float("inf"),
        holding_period=0,
        entry_volatility=None,
        exit_reason="",
    )
    print("\n=== NaN/None default-fill テスト ===")
    for k, v in bad.items():
        print(f"  {k}: {v!r:>20}  ({type(v).__name__})")

    # DataFrame 変換
    df = records_to_df([rec, bad])
    print("\n=== records_to_df dtypes ===")
    print(df.dtypes.to_string())
    print(df.to_string(index=False))

    # append-only CSV テスト
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        tmp = f.name
    try:
        append_trade_log(tmp, rec)
        append_trade_log(tmp, bad)
        loaded = load_trade_log(tmp)
        print("\n=== load_trade_log ===")
        print(loaded.dtypes.to_string())
        print(loaded.to_string(index=False))
        # append-only 確認: 3 件目追記
        append_trade_log(tmp, rec)
        print(f"\n追記後行数: {len(load_trade_log(tmp))}  (期待値: 3)")
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    _example()
