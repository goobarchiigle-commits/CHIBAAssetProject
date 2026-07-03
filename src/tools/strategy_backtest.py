"""
strategy_backtest.py v2 — DeepSeek strategy JSON → バックテスト実行エンジン

使い方:
    from src.tools.strategy_backtest import run
    result = run(strategy_json, df, capital=3_000_000)

    python src/tools/strategy_backtest.py \\
        --strategy docs/research/deepseek_prompt_2026-04-25.json \\
        --data backtests/trades_log.csv --symbol 7203

出力:
    {
      "metrics":     {"sharpe": float, "max_drawdown": float, "win_rate": float,
                      "n_trades": int, "avg_holding_period": float, ...},
      "constraints": {"valid": bool, "errors": [], "warnings": []},
      "equity_curve": [...],
      "trades":      [...]
    }

設計方針:
    - _safe_eval 廃止 → AST ベース安全評価器（関数呼び出し・属性アクセス禁止）
    - 約定ロジック: min(requested_qty, volume * fill_rate) によるボリュームベース部分約定
    - 評価順序: stop_loss → exit → entry（リスク優先）
    - Sharpe > 3.0 は errors → valid=False（v1 は warning のみ）
    - pandas + 標準ライブラリのみ / 乱数なし / 関数分割
"""
from __future__ import annotations

import ast
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ─────────────────────────────────────────────────────────────────────────────
# デフォルトパラメータ（CLAUDE.md PARAMS_LOCKED 準拠）
# ─────────────────────────────────────────────────────────────────────────────
_CAP      = 3_000_000   # 初期資金
_MAX_POS  = 3           # 最大同時保有数
_COMM     = 0.00055     # 片道手数料率
_LOT      = 100         # 最小売買単位（株）


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: AST ベース安全評価器
# ─────────────────────────────────────────────────────────────────────────────

_ALLOWED_NODES: frozenset[type] = frozenset({
    ast.Expression,
    # リテラル
    ast.Constant,
    # 変数参照（ast.Name の ctx フィールドは ast.Load のみ許可）
    ast.Name, ast.Load,
    # 二項演算
    ast.BinOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    # 単項演算
    ast.UnaryOp, ast.USub, ast.UAdd, ast.Not,
    # 論理演算
    ast.BoolOp, ast.And, ast.Or,
    # 比較演算
    ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
})

# 禁止: ast.Call, ast.Attribute, ast.Import, ast.Subscript, ast.Lambda, ...


def _preprocess(expr: str) -> str:
    """大文字 AND/OR/NOT → Python 演算子に変換"""
    expr = re.sub(r'\bAND\b', 'and', expr)
    expr = re.sub(r'\bOR\b',  'or',  expr)
    expr = re.sub(r'\bNOT\b', 'not', expr)
    return expr.strip()


def _validate_ast(node: ast.AST) -> None:
    """
    AST 全ノードを再帰検証。
    _ALLOWED_NODES 以外のノードがあれば即 ValueError。
    関数呼び出し・属性アクセス・import・__系変数を完全遮断。
    """
    if type(node) not in _ALLOWED_NODES:
        raise ValueError(
            f"禁止ノード '{type(node).__name__}': {ast.dump(node)[:80]}"
        )
    if isinstance(node, ast.Name):
        if node.id.startswith('_'):
            raise ValueError(f"禁止変数名（アンダースコア）: '{node.id}'")
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float, bool)):
            raise ValueError(f"禁止リテラル（数値以外）: {node.value!r}")
    for child in ast.iter_child_nodes(node):
        _validate_ast(child)


def _compile_expr(expr: str) -> ast.Expression:
    """
    式を解析・検証し、再利用可能な AST を返す。
    不正な式は SyntaxError / ValueError を送出する。
    """
    processed = _preprocess(expr)
    try:
        tree = ast.parse(processed, mode='eval')
    except SyntaxError as e:
        raise SyntaxError(f"構文エラー [{expr!r}]: {e}") from e
    _validate_ast(tree)
    return tree  # type: ignore[return-value]


def _eval_node(node: ast.AST, ctx: dict[str, Any]) -> Any:
    """
    AST ノードを再帰評価（_validate_ast 通過済みを前提）。
    未定義変数は ValueError。ゼロ除算は ZeroDivisionError。
    """
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, ctx)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in ctx:
            raise ValueError(f"未定義変数: '{node.id}'")
        return ctx[node.id]

    if isinstance(node, ast.BinOp):
        left  = _eval_node(node.left,  ctx)
        right = _eval_node(node.right, ctx)
        op    = node.op
        if isinstance(op, ast.Add):      return left + right
        if isinstance(op, ast.Sub):      return left - right
        if isinstance(op, ast.Mult):     return left * right
        if isinstance(op, ast.Div):
            if right == 0: raise ZeroDivisionError(f"ゼロ除算 (Div): {ast.dump(node)}")
            return left / right
        if isinstance(op, ast.FloorDiv):
            if right == 0: raise ZeroDivisionError(f"ゼロ除算 (FloorDiv): {ast.dump(node)}")
            return int(left) // int(right)
        if isinstance(op, ast.Mod):  return left % right
        if isinstance(op, ast.Pow):  return left ** right
        raise ValueError(f"未対応 BinOp 演算子: {type(op).__name__}")

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, ctx)
        op      = node.op
        if isinstance(op, ast.USub): return -operand
        if isinstance(op, ast.UAdd): return +operand
        if isinstance(op, ast.Not):  return not operand
        raise ValueError(f"未対応 UnaryOp 演算子: {type(op).__name__}")

    if isinstance(node, ast.BoolOp):
        op = node.op
        if isinstance(op, ast.And):
            for v in node.values:
                if not _eval_node(v, ctx): return False
            return True
        if isinstance(op, ast.Or):
            for v in node.values:
                if _eval_node(v, ctx): return True
            return False
        raise ValueError(f"未対応 BoolOp 演算子: {type(op).__name__}")

    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, ctx)
        for cmp_op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, ctx)
            if   isinstance(cmp_op, ast.Lt):    r = left <  right
            elif isinstance(cmp_op, ast.LtE):   r = left <= right
            elif isinstance(cmp_op, ast.Gt):    r = left >  right
            elif isinstance(cmp_op, ast.GtE):   r = left >= right
            elif isinstance(cmp_op, ast.Eq):    r = left == right
            elif isinstance(cmp_op, ast.NotEq): r = left != right
            else: raise ValueError(f"未対応比較演算子: {type(cmp_op).__name__}")
            if not r: return False
            left = right
        return True

    raise ValueError(f"未対応ノード: {type(node).__name__}")  # safety net


def eval_expr(tree: ast.Expression, ctx: dict[str, Any]) -> Any:
    """コンパイル済み AST を評価する（公開 API・ユニットテスト用）"""
    return _eval_node(tree, ctx)


# ─────────────────────────────────────────────────────────────────────────────
# Section 1.5: 文字列レベル事前バリデーション（ASTパース前）
# ─────────────────────────────────────────────────────────────────────────────

_FORBIDDEN_PATTERNS: list[str] = ["min(", "max(", "abs(", "if "]
_REQUIRED_FIELDS: frozenset[str] = frozenset({"entry", "exit", "stop_loss", "position_size"})


def pre_validate_strategy(strategy: dict) -> tuple[bool, list[str]]:
    """
    ASTパース前に文字列レベルで即reject。
    - 必須フィールド欠損チェック
    - 禁止パターン（min/max/abs/if）チェック
    Returns (is_valid, errors)
    """
    errors: list[str] = []
    missing = _REQUIRED_FIELDS - set(strategy.keys())
    if missing:
        errors.append(f"必須フィールド欠損: {sorted(missing)}")
        return False, errors
    for field in _REQUIRED_FIELDS:
        expr = strategy[field]
        if not isinstance(expr, str):
            errors.append(f"[{field}] 式が文字列ではありません: {type(expr).__name__}")
            continue
        for pat in _FORBIDDEN_PATTERNS:
            if pat in expr:
                errors.append(f"[{field}] 禁止パターン '{pat}': {expr!r}")
    return len(errors) == 0, errors


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: コンパイル済み戦略
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _CompiledStrategy:
    entry_expr: ast.Expression
    exit_expr:  ast.Expression
    sl_expr:    ast.Expression
    size_expr:  ast.Expression


def compile_strategy(strategy: dict) -> tuple[_CompiledStrategy | None, list[str]]:
    """
    4 式を一括コンパイル・AST 検証。
    エラーがある場合は (None, errors)。
    正常な場合は (_CompiledStrategy, [])。
    公開 API: ユニットテストで直接呼び出し可能。
    """
    required = ("entry", "exit", "stop_loss", "position_size")
    missing  = [k for k in required if k not in strategy]
    if missing:
        return None, [f"必須キー不足: {missing}"]

    errors: list[str] = []
    trees:  dict[str, ast.Expression] = {}

    for key in required:
        try:
            trees[key] = _compile_expr(strategy[key])
        except (SyntaxError, ValueError) as e:
            errors.append(f"[{key}] {e}")

    if errors:
        return None, errors

    return _CompiledStrategy(
        entry_expr=trees["entry"],
        exit_expr=trees["exit"],
        sl_expr=trees["stop_loss"],
        size_expr=trees["position_size"],
    ), []


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: DataFrame 前処理・指標自動補完
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    必須列確認 + 共通指標を自動計算。列名を小文字統一。
    close 列がなければ ValueError。
    """
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    cols = set(df.columns)

    if "close" not in cols:
        raise ValueError("DataFrame に 'close' 列が必要です")

    # ATR_20
    if "atr_20" not in cols:
        if {"high", "low"}.issubset(cols):
            pc = df["close"].shift(1)
            tr = pd.concat([
                df["high"] - df["low"],
                (df["high"] - pc).abs(),
                (df["low"]  - pc).abs(),
            ], axis=1).max(axis=1)
            df["atr_20"] = tr.ewm(span=20, adjust=False).mean()
        else:
            df["atr_20"] = df["close"] * 0.02

    # MA 系
    for n in (5, 10, 20, 50):
        key = f"ma_{n}"
        if key not in cols:
            df[key] = df["close"].rolling(n).mean()

    # RSI_14
    if "rsi_14" not in cols:
        d  = df["close"].diff()
        up = d.clip(lower=0)
        dn = (-d).clip(lower=0)
        rs = up.ewm(com=13, adjust=False).mean() / dn.ewm(com=13, adjust=False).mean()
        df["rsi_14"] = 100 - 100 / (1 + rs)

    if "open" not in cols:
        df["open"] = df["close"]

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: データ品質検証
# ─────────────────────────────────────────────────────────────────────────────

def _validate_data(df: pd.DataFrame) -> list[str]:
    """
    ATR/price の NaN・ゼロ・負値を検出。
    violations リストを返す（空なら問題なし）。
    """
    errors: list[str] = []

    nan_close = int(df["close"].isna().sum())
    if nan_close > 0:
        errors.append(f"close に NaN {nan_close} 件")

    neg_close = int((df["close"].fillna(0) <= 0).sum())
    if neg_close > 0:
        errors.append(f"close <= 0 が {neg_close} 件")

    if "atr_20" in df.columns:
        # 初期化期間（先頭 20 行）は除外
        tail = df["atr_20"].iloc[20:]
        nan_atr = int(tail.isna().sum())
        if nan_atr > 0:
            errors.append(f"atr_20 に NaN {nan_atr} 件（初期化後）")
        neg_atr = int((tail.fillna(1) <= 0).sum())
        if neg_atr > 0:
            errors.append(f"atr_20 <= 0 が {neg_atr} 件")

    return errors


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: 執行コスト
# ─────────────────────────────────────────────────────────────────────────────

def _buy_price(raw: float, spread_bps: float, slip_bps: float) -> float:
    """買い執行価格（spread + slippage 分だけ高くなる）"""
    return raw * (1 + (spread_bps + slip_bps) / 10_000)


def _sell_price(raw: float, spread_bps: float, slip_bps: float) -> float:
    """売り執行価格（spread + slippage 分だけ低くなる）"""
    return raw * (1 - (spread_bps + slip_bps) / 10_000)


def _trade_ret(entry_px: float, exit_px: float) -> float:
    """往復手数料込み実質リターン"""
    cost_in  = entry_px * (1 + _COMM)
    cost_out = exit_px  * (1 - _COMM)
    return (cost_out - cost_in) / cost_in


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: ボリュームベース部分約定
# ─────────────────────────────────────────────────────────────────────────────

def calc_fill_qty(
    requested_qty: int,
    volume: float,
    fill_rate: float,
    lot_size: int,
) -> int:
    """
    ボリュームベース部分約定（公開 API・ユニットテスト可能）。

    約定量 = min(requested_qty, floor(volume * fill_rate / lot_size) * lot_size)
    volume が inf（列なし）または NaN/0 の場合はフル約定（volume 制約なし）。
    """
    if math.isinf(volume) or math.isnan(volume) or volume <= 0:
        return requested_qty
    max_by_vol = int((volume * fill_rate) // lot_size) * lot_size
    return min(requested_qty, max(0, max_by_vol))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: ポジション
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _Position:
    symbol:      str
    entry_date:  Any
    entry_price: float   # 執行コスト込み
    qty:         int
    hold_days:   int = 0


# ─────────────────────────────────────────────────────────────────────────────
# Section 8: コンテキスト構築
# ─────────────────────────────────────────────────────────────────────────────

def _build_ctx(
    row: "pd.Series[Any]",
    entry_price: float,
    hold_days:   int,
    cash:        float,
    capital:     float,
) -> dict[str, Any]:
    """
    DataFrame 行 + 状態変数を eval コンテキストに変換。
    全列名は小文字。数値変換できない値は文字列として格納。
    """
    ctx: dict[str, Any] = {}
    for col in row.index:
        v = row[col]
        try:
            fv = float(v)
            ctx[col] = fv
        except (TypeError, ValueError):
            ctx[col] = v
    ctx.update({
        "entry_price": entry_price,
        "hold_days":   hold_days,
        "cash":        cash,
        "capital":     capital,
    })
    return ctx


# ─────────────────────────────────────────────────────────────────────────────
# Section 9: 個別ルール評価関数（公開 API・ユニットテスト可能）
# ─────────────────────────────────────────────────────────────────────────────

def eval_entry(tree: ast.Expression, ctx: dict[str, Any]) -> bool:
    """エントリーシグナル評価。評価エラー時は False を返す"""
    try:
        return bool(_eval_node(tree, ctx))
    except Exception:
        return False


def eval_exit(tree: ast.Expression, ctx: dict[str, Any]) -> bool:
    """イグジットシグナル評価。評価エラー時は False を返す"""
    try:
        return bool(_eval_node(tree, ctx))
    except Exception:
        return False


def eval_stop_loss(
    tree:  ast.Expression,
    ctx:   dict[str, Any],
    close: float,
) -> bool:
    """
    ストップロス評価（公開 API）。

    式が価格水準（float）を返す場合: close <= sl_level で判定。
    式が bool を返す場合: そのまま使用。
    評価エラー時は False。
    """
    try:
        result = _eval_node(tree, ctx)
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    if isinstance(result, (int, float)) and not math.isnan(float(result)):
        return close <= float(result)
    return False


def eval_position_size(
    tree: ast.Expression,
    ctx:  dict[str, Any],
) -> float:
    """
    ポジションサイズ計算（公開 API）。
    返値は株数（entry_price で割り算済みの式を想定）。
    評価エラー・負値・NaN は 0.0 を返す。
    """
    try:
        v = float(_eval_node(tree, ctx))
        return v if not math.isnan(v) and v > 0 else 0.0
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Section 10: バックテストループ
# ─────────────────────────────────────────────────────────────────────────────

def _get_volume(row: "pd.Series[Any]", has_volume: bool) -> float:
    """volume 列がなければ inf（ボリューム制約なし）を返す"""
    if not has_volume:
        return float('inf')
    v = row["volume"]
    try:
        fv = float(v)
        return fv if not math.isnan(fv) and fv > 0 else float('inf')
    except (TypeError, ValueError):
        return float('inf')


def _make_trade_record(
    symbol:      str,
    pos:         _Position,
    exit_date:   Any,
    exit_px:     float,
    ret:         float,
    reason:      str,
) -> dict[str, Any]:
    return {
        "symbol":      symbol,
        "entry_date":  str(pos.entry_date),
        "exit_date":   str(exit_date),
        "entry_price": round(pos.entry_price, 2),
        "exit_price":  round(exit_px, 2),
        "qty":         pos.qty,
        "ret":         round(ret, 6),
        "hold_days":   pos.hold_days,
        "exit_reason": reason,
    }


def _run_loop(
    df:                  pd.DataFrame,
    cs:                  _CompiledStrategy,
    capital:             float,
    max_positions:       int,
    spread_bps:          float,
    slip_bps:            float,
    fill_rate:           float,
    lot_size:            int,
    max_single_yen:      float,
    symbol:              str,
    max_position_ratio:  float = 0.25,
) -> tuple[list[dict], list[dict]]:
    """
    バックテストのメインループ。
    評価順序（リスク優先）: stop_loss → exit → entry
    max_position_ratio: 1 トレードの最大資金割合（capital 基準）
    """
    cash:         float              = capital
    positions:    list[_Position]    = []
    trades:       list[dict[str, Any]] = []
    equity_curve: list[dict[str, Any]] = []
    has_volume    = "volume" in df.columns

    for i in range(len(df)):
        row    = df.iloc[i]
        date   = df.index[i]
        close  = float(row["close"])
        volume = _get_volume(row, has_volume)

        if math.isnan(close) or close <= 0:
            continue

        still_open: list[_Position] = []

        for pos in positions:
            ctx = _build_ctx(row, pos.entry_price, pos.hold_days, cash, capital)

            # ─ ① STOP LOSS（最優先） ──────────────────────────────────
            if eval_stop_loss(cs.sl_expr, ctx, close):
                raw_px  = float(row["open"])
                exec_px = _sell_price(raw_px, spread_bps, slip_bps)
                cash   += exec_px * pos.qty * (1 - _COMM)
                trades.append(_make_trade_record(
                    symbol, pos, date, exec_px,
                    _trade_ret(pos.entry_price, exec_px), "stop_loss"
                ))
                continue

            # ─ ② EXIT シグナル ───────────────────────────────────────
            if eval_exit(cs.exit_expr, ctx):
                exec_px = _sell_price(close, spread_bps, slip_bps)
                cash   += exec_px * pos.qty * (1 - _COMM)
                trades.append(_make_trade_record(
                    symbol, pos, date, exec_px,
                    _trade_ret(pos.entry_price, exec_px), "signal"
                ))
                continue

            pos.hold_days += 1
            still_open.append(pos)

        positions = still_open

        # ─ ③ ENTRY ──────────────────────────────────────────────────
        if len(positions) < max_positions:
            ctx_entry = _build_ctx(row, close, 0, cash, capital)
            if eval_entry(cs.entry_expr, ctx_entry):
                exec_px = _buy_price(close, spread_bps, slip_bps)

                # ポジションサイズ計算（exec_px ベースコンテキスト）
                ctx_size = _build_ctx(row, exec_px, 0, cash, capital)
                qty_raw  = eval_position_size(cs.size_expr, ctx_size)
                qty_req  = max(lot_size, int(qty_raw // lot_size) * lot_size)

                # 資金・1ポジション上限クランプ
                max_by_cash   = int((cash / (exec_px * (1 + _COMM))) // lot_size) * lot_size
                max_by_single = int((max_single_yen / exec_px)        // lot_size) * lot_size
                # max_position_ratio: position_value <= capital * ratio
                max_by_ratio  = int((capital * max_position_ratio / exec_px) // lot_size) * lot_size
                qty_req = min(qty_req, max_by_cash, max_by_single, max_by_ratio)

                # ボリュームベース部分約定
                qty  = calc_fill_qty(qty_req, volume, fill_rate, lot_size)
                cost = exec_px * qty * (1 + _COMM)

                if qty >= lot_size and cost <= cash:
                    cash -= cost
                    positions.append(_Position(
                        symbol=symbol, entry_date=date,
                        entry_price=exec_px, qty=qty,
                    ))

        # ─ ④ equity curve 記録 ──────────────────────────────────────
        open_pnl = sum((close - p.entry_price) * p.qty for p in positions)
        equity   = cash + sum(p.entry_price * p.qty for p in positions) + open_pnl
        equity_curve.append({
            "date":     str(date),
            "equity":   round(equity, 0),
            "cash":     round(cash, 0),
            "open_pnl": round(open_pnl, 0),
            "n_pos":    len(positions),
        })

    # 期末強制クローズ
    if len(df) > 0 and positions:
        last_close = float(df.iloc[-1]["close"])
        last_date  = df.index[-1]
        for pos in positions:
            exec_px = _sell_price(last_close, spread_bps, slip_bps)
            trades.append(_make_trade_record(
                symbol, pos, last_date, exec_px,
                _trade_ret(pos.entry_price, exec_px), "forced_close"
            ))

    return trades, equity_curve


# ─────────────────────────────────────────────────────────────────────────────
# Section 11: 指標計算（標準ライブラリのみ）
# ─────────────────────────────────────────────────────────────────────────────

def _sharpe(daily_rets: list[float]) -> float:
    n = len(daily_rets)
    if n < 2:
        return float("nan")
    mean = sum(daily_rets) / n
    var  = sum((r - mean) ** 2 for r in daily_rets) / (n - 1)
    std  = math.sqrt(var)
    return float("nan") if std == 0 else mean / std * math.sqrt(252)


def _max_drawdown(equity: list[float]) -> float:
    if not equity:
        return float("nan")
    peak   = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        if peak > 0:
            dd = (v - peak) / peak
            if dd < max_dd:
                max_dd = dd
    return max_dd


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2


def _fmt(v: float, decimals: int = 4) -> float | None:
    return None if (math.isnan(v) or math.isinf(v)) else round(v, decimals)


def calc_metrics(
    trades:       list[dict],
    equity_curve: list[dict],
    capital:      float,
) -> dict[str, Any]:
    """
    Sharpe / MaxDrawdown / 勝率 / トレード数 / 平均保有期間 を計算（公開 API）。
    trades が空の場合は None 埋め。
    """
    if not trades:
        return {
            "sharpe":             None,
            "max_drawdown":       None,
            "win_rate":           None,
            "n_trades":           0,
            "avg_holding_period": None,
            "mean_ret":           None,
            "total_return":       None,
            "final_equity":       round(capital, 0),
        }

    rets       = [t["ret"] for t in trades]
    hold_days  = [t["hold_days"] for t in trades]
    wins       = [r for r in rets if r > 0]
    eq_vals    = [e["equity"] for e in equity_curve]

    daily_rets = [
        (eq_vals[i] - eq_vals[i - 1]) / eq_vals[i - 1]
        for i in range(1, len(eq_vals))
        if eq_vals[i - 1] > 0
    ]

    final_e = eq_vals[-1] if eq_vals else capital

    return {
        "sharpe":             _fmt(_sharpe(daily_rets), 3),
        "max_drawdown":       _fmt(_max_drawdown(eq_vals), 4),
        "win_rate":           round(len(wins) / len(rets), 4),
        "n_trades":           len(trades),
        "avg_holding_period": round(sum(hold_days) / len(hold_days), 1),
        "mean_ret":           round(sum(rets) / len(rets), 4),
        "total_return":       round((final_e - capital) / capital, 4),
        "final_equity":       round(final_e, 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 12: 制約チェック
# ─────────────────────────────────────────────────────────────────────────────

def check_constraints(
    metrics:     dict[str, Any],
    data_errors: list[str],
    expr_errors: list[str],
    stability:   "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    """
    全 violations を集約（公開 API）。

    errors   → valid=False
      - n_trades < 50
      - Sharpe > 3.0 AND n_trades < 200（低頻度偶然ヒット排除）
      - max_drawdown < -50%
      - データ異常 / 式エラー
      - stability errors（有効時）
    warnings → valid に影響しない（avg_holding_period 異常値）
    stability → constraints["stability"] に格納
    """
    errors:   list[str] = list(data_errors) + list(expr_errors)
    warnings: list[str] = []

    n = metrics.get("n_trades", 0)
    if n < 50 and not expr_errors:
        errors.append(f"n_trades={n} < 50: 統計的に無効（最低 50 件必要）")

    # Sharpe 過学習フィルタ: 高頻度（n>=200）は許容、低頻度の偶然ヒットのみ排除
    sharpe = metrics.get("sharpe")
    if sharpe is not None and sharpe > 3.0 and n < 200:
        errors.append(
            f"Sharpe={sharpe:.3f} > 3.0 かつ n_trades={n} < 200: "
            "低頻度過学習疑い（先読みリーク確認必須）[VALIDATION.sharpe_max=3.0]"
        )

    max_dd = metrics.get("max_drawdown")
    if max_dd is not None and max_dd < -0.50:
        errors.append(f"max_drawdown={max_dd:.2%} < -50%: 戦略として非現実的")

    avg_hold = metrics.get("avg_holding_period")
    if avg_hold is not None:
        if avg_hold < 1.0:
            warnings.append(f"avg_holding_period={avg_hold:.1f} < 1: 超高頻度の可能性")
        elif avg_hold > 100.0:
            warnings.append(f"avg_holding_period={avg_hold:.1f} > 100: 過度な長期保有リスク")

    # stability チェック（run_stability の結果を統合）
    stability_out: dict[str, Any] = {}
    if stability is not None:
        errors.extend(stability.get("errors", []))
        stability_out = {
            "sharpe_std":  stability.get("sharpe_std"),
            "sharpe_min":  stability.get("sharpe_min"),
            "sharpe_mean": stability.get("sharpe_mean"),
            "sharpe_list": stability.get("sharpe_list", []),
        }

    return {
        "valid":     not bool(errors),
        "errors":    errors,
        "warnings":  warnings,
        "stability": stability_out,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 12.5: 安定性評価（時系列分割）
# ─────────────────────────────────────────────────────────────────────────────

def run_stability(
    strategy_input: dict,
    df:             pd.DataFrame,
    n_splits:       int = 5,
    **run_kwargs: Any,
) -> dict[str, Any]:
    """
    同一戦略を時系列 n 分割で検証し、Sharpe の安定性を測定（公開 API）。

    「勝てる戦略を探す」のではなく「ダメな戦略を落とす」ためのフィルター。

    Args:
        n_splits  : 分割数（2〜10 に制限。デフォルト: 5）
        run_kwargs: run() に渡す追加引数（capital, spread_bps 等）

    Returns:
        {
          "sharpe_list":    [各期間 Sharpe],
          "sharpe_mean":    float | None,
          "sharpe_std":     float | None,
          "sharpe_min":     float | None,
          "errors":         [invalid 判定理由],
          "period_results": [各期間の概要],
        }

    判定基準:
        sharpe_std > 1.0 → invalid（期間間安定性不足）
        sharpe_min < 0   → invalid（損失期間あり）
    """
    # 分割数を安全範囲に制限
    n_splits = max(2, min(n_splits, 10))

    # stability_splits=0 を強制してループを防止
    run_kwargs["stability_splits"] = 0

    total      = len(df)
    split_size = total // n_splits

    if split_size < 60:
        return {
            "sharpe_list":    [],
            "sharpe_mean":    None,
            "sharpe_std":     None,
            "sharpe_min":     None,
            "errors":         [
                f"分割後行数 {split_size} < 60（n_splits={n_splits}）: "
                "分割数を減らすか、データ期間を延ばしてください"
            ],
            "period_results": [],
        }

    sharpe_list:    list[float | None]  = []
    period_results: list[dict[str, Any]] = []

    for k in range(n_splits):
        start = k * split_size
        end   = (k + 1) * split_size if k < n_splits - 1 else total
        df_k  = df.iloc[start:end]

        res     = run(strategy_input, df_k, **run_kwargs)
        sharpe  = res["metrics"].get("sharpe")
        n_t     = res["metrics"].get("n_trades", 0)

        sharpe_list.append(sharpe)
        period_results.append({
            "period":   k + 1,
            "rows":     len(df_k),
            "n_trades": n_t,
            "sharpe":   sharpe,
            "valid":    res["constraints"]["valid"],
        })

    valid_sharpes = [s for s in sharpe_list if s is not None]

    if not valid_sharpes:
        return {
            "sharpe_list":    [_fmt(s, 3) if s is not None else None for s in sharpe_list],
            "sharpe_mean":    None,
            "sharpe_std":     None,
            "sharpe_min":     None,
            "errors":         ["全期間でバックテスト結果なし（データまたは戦略に問題あり）"],
            "period_results": period_results,
        }

    vn    = len(valid_sharpes)
    mean  = sum(valid_sharpes) / vn
    var   = sum((s - mean) ** 2 for s in valid_sharpes) / max(vn - 1, 1)
    std   = math.sqrt(var)
    min_s = min(valid_sharpes)

    errors: list[str] = []
    if not math.isnan(std) and std > 1.0:
        errors.append(
            f"sharpe_std={std:.3f} > 1.0: 期間間安定性不足（過学習の可能性）"
        )
    if min_s < 0:
        errors.append(
            f"sharpe_min={min_s:.3f} < 0: 損失期間あり（戦略の頑健性に懸念）"
        )

    return {
        "sharpe_list":    [_fmt(s, 3) if s is not None else None for s in sharpe_list],
        "sharpe_mean":    _fmt(mean, 3),
        "sharpe_std":     _fmt(std, 3),
        "sharpe_min":     _fmt(min_s, 3),
        "errors":         errors,
        "period_results": period_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 13: メインエントリー
# ─────────────────────────────────────────────────────────────────────────────

def _empty_result(errors: list[str]) -> dict[str, Any]:
    return {
        "metrics":      {},
        "constraints":  {"valid": False, "errors": errors, "warnings": [], "stability": {}},
        "equity_curve": [],
        "trades":       [],
    }


def run(
    strategy_input:      dict,
    df:                  pd.DataFrame,
    *,
    symbol:              str   = "unknown",
    capital:             float = float(_CAP),
    max_positions:       int   = _MAX_POS,
    spread_bps:          float = 3.0,
    slip_bps:            float = 10.0,
    fill_rate:           float = 0.01,
    lot_size:            int   = _LOT,
    max_single_yen:      float = 750_000.0,
    max_position_ratio:  float = 0.25,       # 1トレード最大資金割合
    stability_splits:    int   = 0,           # 0=OFF, 2〜10=ON（時系列分割数）
) -> dict[str, Any]:
    """
    DeepSeek strategy JSON をバックテスト実行。

    Args:
        max_position_ratio : 1トレードの最大資金割合（position_value <= capital * ratio）
        stability_splits   : 時系列分割数（0=無効, 2〜10 で安定性検証を実行）
        spread_bps         : スプレッド (bps)
        slip_bps           : スリッページ (bps)
        fill_rate          : ボリューム制約係数（約定量 = min(qty, volume * fill_rate)）
        max_single_yen     : 1ポジション最大投資額

    Returns:
        {"metrics": {...}, "constraints": {...}, "equity_curve": [...], "trades": [...]}
        constraints["stability"] に安定性指標を格納（stability_splits >= 2 の場合）
    """
    strategy = strategy_input.get("strategy", strategy_input)
    if not isinstance(strategy, dict):
        return _empty_result(["strategy が dict ではありません"])

    # ─ 文字列レベル事前バリデーション ──────────────────────────────
    pre_ok, pre_errors = pre_validate_strategy(strategy)
    if not pre_ok:
        return _empty_result(pre_errors)

    # ─ 式コンパイル（upfront validation） ──────────────────────────
    cs, expr_errors = compile_strategy(strategy)

    # ─ DataFrame 前処理 ─────────────────────────────────────────────
    try:
        df = _ensure_indicators(df)
    except ValueError as e:
        return _empty_result([str(e)])

    if len(df) < 60:
        return _empty_result([f"データ行数={len(df)} < 60: バックテスト不可"])

    # ─ データ品質検証 ───────────────────────────────────────────────
    data_errors = _validate_data(df)

    # ─ バックテスト実行 ─────────────────────────────────────────────
    if cs is not None and not data_errors:
        trades, equity_curve = _run_loop(
            df, cs, capital, max_positions,
            spread_bps, slip_bps, fill_rate, lot_size, max_single_yen, symbol,
            max_position_ratio=max_position_ratio,
        )
        metrics = calc_metrics(trades, equity_curve, capital)
    else:
        trades       = []
        equity_curve = []
        metrics      = {}

    # ─ 安定性評価（オプション） ─────────────────────────────────────
    stability: "dict[str, Any] | None" = None
    if stability_splits >= 2:
        stability = run_stability(
            strategy_input, df,
            n_splits=stability_splits,
            symbol=symbol, capital=capital, max_positions=max_positions,
            spread_bps=spread_bps, slip_bps=slip_bps, fill_rate=fill_rate,
            lot_size=lot_size, max_single_yen=max_single_yen,
            max_position_ratio=max_position_ratio,
            # stability_splits=0 は run_stability 内部で強制設定済み
        )

    constraints = check_constraints(metrics, data_errors, expr_errors, stability)

    return {
        "metrics":      metrics,
        "constraints":  constraints,
        "equity_curve": equity_curve,
        "trades":       trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section 14: CLI
# ─────────────────────────────────────────────────────────────────────────────

def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col = next(
        (c for c in df.columns if c.lower() in ("date", "datetime", "entry_date")), None
    )
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col).sort_index()
    return df


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek strategy JSON → バックテスト v2")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--data",     required=True)
    parser.add_argument("--symbol",   default="unknown")
    parser.add_argument("--capital",  type=float, default=float(_CAP))
    parser.add_argument("--spread",   type=float, default=3.0)
    parser.add_argument("--slip",     type=float, default=10.0)
    parser.add_argument("--fill",      type=float, default=0.01)
    parser.add_argument("--max-pos-ratio", type=float, default=0.25, help="1トレード最大資金割合")
    parser.add_argument("--splits",   type=int,   default=0,    help="安定性評価の分割数 (0=OFF)")
    parser.add_argument("--out",      default=None)
    args = parser.parse_args()

    _BASE = Path(__file__).resolve().parent.parent.parent

    sp = Path(args.strategy)
    dp = Path(args.data)
    if not sp.is_absolute(): sp = _BASE / sp
    if not dp.is_absolute(): dp = _BASE / dp

    for p in (sp, dp):
        if not p.exists():
            print(f"[ERROR] {p} が見つかりません", file=sys.stderr)
            sys.exit(1)

    strategy_json = json.loads(sp.read_text(encoding="utf-8"))
    df            = _load_df(dp)

    result = run(
        strategy_json, df,
        symbol=args.symbol, capital=args.capital,
        spread_bps=args.spread, slip_bps=args.slip, fill_rate=args.fill,
        max_position_ratio=args.max_pos_ratio, stability_splits=args.splits,
    )

    out_str = json.dumps(result, ensure_ascii=False, indent=2, default=str)
    if args.out:
        Path(args.out).write_text(out_str, encoding="utf-8")
        print(f"[INFO] 保存: {args.out}")
    else:
        print(out_str)


if __name__ == "__main__":
    main()
