"""
trade_prompt_gen.py — トレード履歴CSV → DeepSeek分析プロンプト自動生成

使い方:
    python src/tools/trade_prompt_gen.py --input backtests/trades_log.csv
    python src/tools/trade_prompt_gen.py --input logs/ce_compare_daily.csv
    python src/tools/trade_prompt_gen.py --input logs/trades.jsonl
    python src/tools/trade_prompt_gen.py --input logs/execution_quality/20260421.jsonl

出力:
    logs/research/deepseek_prompt_<YYYY-MM-DD>.json
    標準出力にプロンプト文字列も表示
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import pandas as pd

# パス解決
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from src.paths import DOCS_DIR, LOGS_DIR

# ──────────────────────────────────────────
# スキーマ判定
# ──────────────────────────────────────────
_SCHEMA_TRADES   = "trades_log"       # entry_date, exit_bar, ret
_SCHEMA_CE       = "ce_compare"       # pnl_base, pnl_ce, delta_pnl
_SCHEMA_LIVE     = "live_trades"      # symbol, side, amount (JSONL)
_SCHEMA_EXECQUAL = "exec_quality"     # slippage_pct, fill_status (JSONL)

def detect_schema(df: pd.DataFrame, source_path: Path) -> str:
    cols = set(df.columns)
    if {"entry_date", "ret"}.issubset(cols):
        return _SCHEMA_TRADES
    if {"pnl_base", "pnl_ce", "delta_pnl"}.issubset(cols):
        return _SCHEMA_CE
    if {"symbol", "side", "amount"}.issubset(cols):
        return _SCHEMA_LIVE
    if {"slippage_pct", "fill_status"}.issubset(cols):
        return _SCHEMA_EXECQUAL
    raise ValueError(f"未対応スキーマ: {cols}")


# ──────────────────────────────────────────
# ロード（CSV / JSONL 自動判別）
# ──────────────────────────────────────────
def load_file(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return pd.DataFrame(rows)
    return pd.read_csv(path)


# ──────────────────────────────────────────
# 統計集計
# ──────────────────────────────────────────
def _safe_sharpe(rets: pd.Series) -> float:
    std = rets.std()
    if std == 0 or math.isnan(std):
        return float("nan")
    return float(rets.mean() / std * math.sqrt(252))


def _max_drawdown(rets: pd.Series) -> float:
    cum = (1 + rets).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max
    return float(dd.min())


def stats_trades(df: pd.DataFrame) -> dict:
    r = df["ret"].dropna()
    return {
        "n_trades":    int(len(r)),
        "win_rate":    round(float((r > 0).mean()), 4),
        "mean_ret":    round(float(r.mean()), 4),
        "median_ret":  round(float(r.median()), 4),
        "sharpe":      round(_safe_sharpe(r), 3),
        "max_dd":      round(_max_drawdown(r), 4),
        "hold_days_mean": round(float(df["exit_bar"].mean()), 1) if "exit_bar" in df else None,
        "ret_adj_mean": round(float(df["ret_adj"].mean()), 4) if "ret_adj" in df else None,
        "worst_5": r.nsmallest(5).round(4).tolist(),
        "best_5":  r.nlargest(5).round(4).tolist(),
    }


def _nan_to_none(v: float | None) -> float | None:
    if v is None:
        return None
    return None if (math.isnan(v) or math.isinf(v)) else v


def stats_ce(df: pd.DataFrame) -> dict:
    delta = df["delta_pnl"].dropna()
    cum_delta_final = float(df["cum_delta"].iloc[-1]) if ("cum_delta" in df and len(df) > 0) else None
    outlier_days: list = []
    if "date" in df and len(delta) > 1:
        std = delta.std()
        if std > 0:
            outlier_days = df[delta.abs() > std * 2]["date"].tolist()
    return {
        "n_days":          int(len(df)),
        "delta_mean":      _nan_to_none(round(float(delta.mean()), 4)) if len(delta) else None,
        "delta_std":       _nan_to_none(round(float(delta.std()), 4))  if len(delta) else None,
        "cum_delta_final": round(cum_delta_final, 4) if cum_delta_final is not None else None,
        "fill_rate_diff_mean":  _nan_to_none(round(float(df["fill_rate_diff"].mean()), 4))   if "fill_rate_diff"   in df else None,
        "sign_match_rate_mean": _nan_to_none(round(float(df["sign_match_rate"].mean()), 4))  if "sign_match_rate"  in df else None,
        "outlier_days_2sigma": outlier_days[:10],
    }


def stats_live(df: pd.DataFrame) -> dict:
    out: dict = {"n_trades": int(len(df))}
    if "sector" in df:
        out["by_sector"] = df.groupby("sector")["amount"].sum().round(0).astype(int).to_dict()
    if "entry_regime" in df:
        out["by_regime"] = df["entry_regime"].value_counts().to_dict()
    if "side" in df:
        out["buy_sell"] = df["side"].value_counts().to_dict()
    if "amount" in df:
        out["total_amount"] = int(df["amount"].sum())
        out["avg_amount"] = int(df["amount"].mean())
    return out


def stats_execqual(df: pd.DataFrame) -> dict:
    slip = df["slippage_pct"].dropna()
    fill = df["fill_status"].value_counts().to_dict() if "fill_status" in df else {}
    gap = df["gap_pct"].dropna() if "gap_pct" in df else pd.Series([], dtype=float)
    return {
        "n_orders":       int(len(df)),
        "fill_status":    fill,
        "slippage_mean":  _nan_to_none(round(float(slip.mean()), 5)) if len(slip) else None,
        "slippage_max":   _nan_to_none(round(float(slip.max()),  5)) if len(slip) else None,
        "slippage_std":   _nan_to_none(round(float(slip.std()),  5)) if len(slip) else None,
        "gap_pct_mean":   _nan_to_none(round(float(gap.mean()),  5)) if len(gap)  else None,
    }


# ──────────────────────────────────────────
# プロンプト生成
# ──────────────────────────────────────────

# 出力スキーマ（DeepSeekに厳守させる）
_OUTPUT_SCHEMA = """\
{
  "strategy": {
    "entry":         "<使用可能変数: close, open, high, low, ma_5/10/20/50, rsi_14, atr_20, hold_days 等。例: close > ma_20 AND rsi_14 < 55>",
    "exit":          "<例: hold_days >= 20 OR rsi_14 > 70>",
    "stop_loss":     "<entry_price に対する掛け算のみ。例: entry_price * 0.95>",
    "position_size": "<形式固定: capital * X / entry_price  (0 < X <= 0.25 の数値のみ)。例: capital * 0.25 / entry_price>"
  },
  "reason": "<根拠。統計の具体的数値を必ず引用>",
  "risk":   "<実運用上の最大リスク。具体的数値で記述>",
  "strategy_score": {
    "robustness":            "<0.0〜1.0>",
    "overfitting_risk":      "<0.0〜1.0。Sharpe>3なら>=0.7>",
    "execution_feasibility": "<0.0〜1.0>"
  }
}"""

_SYSTEM_STRICT = """\
あなたは定量トレーディング戦略の専門アナリストです。

【式の文法制約 — 1違反で出力全体無効】
使用可能演算子: + - * / > < >= <= AND OR のみ
禁止: min() max() abs() int() float() その他すべての関数呼び出し
禁止: 三項演算子 (X if cond else Y)
禁止: ネスト式（括弧は計算順序のみ使用可、式は1階層まで）
禁止: 文字列・リスト・辞書リテラル

【position_size 固定形式】
必ず "capital * X / entry_price" の形式のみ（X は 0 < X <= 0.25 の小数）
例: capital * 0.25 / entry_price  ← 唯一の許容形式
例 NG: min(capital * 0.25, 600000) / entry_price  ← 関数呼び出し禁止

【戦略要件】
- n_trades >= 200 を満たせるシグナル頻度を優先せよ（閾値を緩めに設定）
- 1年以上の期間でシグナルが途切れない条件を選べ
- Sharpe > 3.0 のデータは overfitting_risk >= 0.7 必須

【出力形式】
- 有効な JSON のみ。前置き・後書き・マークダウン・説明文一切禁止
- 実行環境: kabuステーション REST API / 東証現物 / 資金300万円 / max_positions=3"""

# 安定性制約（build_prompt 内で system_content に注入）
_STABILITY_CONSTRAINT = """\
【安定性制約 — OOS複数期間での安定性が最優先】
- 目的: 単一期間のSharpe最大化ではなく、複数OOS期間でSharpeが安定すること
- トレンド依存禁止: 上昇・下落トレンド期のみで機能する条件は使わない
- 過剰パラメータ最適化禁止: 閾値は大台の数値（ma_20, rsi_14, hold_days=20 等）を優先

【エントリー構造要件 — entry + filter の2層構造を必須とする】
エントリー条件の単体使用禁止。以下のいずれかの組み合わせを使用すること:
  - MAクロス + ボラティリティフィルタ（例: close > ma_20 AND atr_20 > X）
  - RSI逆張り + トレンドフィルタ（例: rsi_14 < 40 AND close > ma_50）
  - ブレイクアウト + ボラ抑制（例: close > ma_20 AND atr_20 < X）

【取引数制約】
- OOS分割した各セグメントで最低30トレード以上を想定できる条件を選べ
- 1年以内のOOS期間でシグナルが0件になる条件は禁止

【exit制約 — 時間exitを必須とする】
- hold_days による時間exit を必ず含めること（例: hold_days >= N）
- stop_loss のみに依存する exit は禁止
- 推奨形式: hold_days >= N OR rsi_14 > X（時間 + 利益確定の組み合わせ）"""

# スキーマ別の分析フォーカス（systemに追加する補足）
_ANALYSIS_FOCUS: dict[str, str] = {
    _SCHEMA_TRADES: """\
【分析フォーカス: バックテスト履歴】
- win_rate・sharpe・max_dd から entry/exit 条件の閾値を導出せよ
- worst_5 の損失幅から stop_loss 水準を決定せよ
- hold_days_mean から exit の保有日数上限を設定せよ
- n_trades >= 200 を達成できる閾値を選択せよ（厳しすぎる条件は禁止）""",

    _SCHEMA_CE: """\
【分析フォーカス: CE vs Base 日次比較】
- delta_pnl の平均・標準偏差から CE モデルの有効性を定量評価せよ
- sign_match_rate から entry シグナルの信頼度を閾値化せよ
- fill_rate_diff から position_size の X 係数を算出せよ
- n_trades >= 200 を達成するシグナル頻度を確保せよ""",

    _SCHEMA_LIVE: """\
【分析フォーカス: ライブトレード履歴】
- by_sector の偏りから entry 条件に業種フィルタを組み込め
- buy_sell 比率から entry の方向性バイアスを条件式に明示せよ
- avg_amount と capital=3_000_000 / max_positions=3 の整合性を確認せよ
- n_trades >= 200 を達成するシグナル頻度を優先せよ""",

    _SCHEMA_EXECQUAL: """\
【分析フォーカス: 執行品質】
- slippage_mean から stop_loss のマージンを調整せよ（entry_price * X の X を増やす）
- gap_pct_mean が 0.005 超の場合、entry 条件に寄り付き回避フィルタを追加せよ
- execution_feasibility はスリッページ・約定率の実測値から算出せよ
- n_trades >= 200 を達成するシグナル頻度を確保せよ""",
}

# スキーマ → 統計関数マップ（pipeline.py から参照）
STAT_FN_MAP: dict[str, Any] = {
    _SCHEMA_TRADES:   stats_trades,
    _SCHEMA_CE:       stats_ce,
    _SCHEMA_LIVE:     stats_live,
    _SCHEMA_EXECQUAL: stats_execqual,
}


def build_prompt(
    schema:        str,
    stats:         dict,
    source_name:   str,
    date_str:      str,
    valid_summary: "dict | None" = None,
) -> dict:
    """
    DeepSeek 用プロンプトペイロードを生成。

    Args:
        valid_summary: pipeline の再学習ループから渡される有効戦略フィードバック。
                       None の場合はフィードバックなし（初回など）。
    """
    focus      = _ANALYSIS_FOCUS[schema]
    stats_json = json.dumps(stats, ensure_ascii=False, indent=2)
    system_content = (
        _SYSTEM_STRICT
        + f"\n\n{_STABILITY_CONSTRAINT}"
        + f"\n\n【出力スキーマ（厳守）】\n{_OUTPUT_SCHEMA}"
    )

    user_parts = [
        f"【データソース】{source_name}（集計日: {date_str}）",
        "",
        f"【統計サマリー】\n```json\n{stats_json}\n```",
        "",
        focus,
    ]

    # 有効戦略フィードバック（再学習ループ）
    if valid_summary and valid_summary.get("n_valid", 0) > 0:
        user_parts += [
            "",
            f"【過去の有効戦略フィードバック（参考のみ・過剰適合禁止）】",
            f"- 累計有効数: {valid_summary['n_valid']} 件 / 平均Sharpe: {valid_summary.get('avg_sharpe', 'N/A')}",
            "- 有効だった条件パターン（上位5件）:",
        ]
        for feat in valid_summary.get("common_features", [])[:5]:
            user_parts.append(f"  * {feat}")
        user_parts.append("【注意】上記はヒントであり丸コピー禁止。新しい条件を導出せよ。")

    return {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_content},
            {"role": "user",   "content": "\n".join(user_parts)},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
        "response_format": {"type": "json_object"},
    }


# ──────────────────────────────────────────
# 出力保存
# ──────────────────────────────────────────
def save_output(prompt_payload: dict, date_str: str) -> Path:
    out_dir = DOCS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"deepseek_prompt_{date_str}.json"
    out_path.write_text(
        json.dumps(prompt_payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return out_path


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="トレード履歴 → DeepSeekプロンプト生成")
    parser.add_argument("--input", required=True, help="入力CSV/JSONLパス")
    parser.add_argument("--out",   default=None,  help="出力JSONパス（省略時: docs/research/）")
    parser.add_argument("--print-prompt", action="store_true", help="プロンプトをstdoutに表示")
    args = parser.parse_args()

    src_path = Path(args.input)
    if not src_path.is_absolute():
        # cwd=C:/ai-trading 前提でフォールバック
        from src.paths import BASE_DIR
        src_path = BASE_DIR / src_path
    if not src_path.exists():
        print(f"[ERROR] ファイルが見つかりません: {src_path}", file=sys.stderr)
        sys.exit(1)

    date_str = datetime.now().strftime("%Y-%m-%d")

    df = load_file(src_path)
    schema = detect_schema(df, src_path)

    stat_fn = {
        _SCHEMA_TRADES:   stats_trades,
        _SCHEMA_CE:       stats_ce,
        _SCHEMA_LIVE:     stats_live,
        _SCHEMA_EXECQUAL: stats_execqual,
    }[schema]
    stats = stat_fn(df)

    print(f"[INFO] スキーマ判定: {schema} ({len(df)} rows)")
    print(f"[INFO] 統計: {json.dumps(stats, ensure_ascii=False)}")

    payload = build_prompt(schema, stats, src_path.name, date_str)

    out_path = Path(args.out) if args.out else save_output(payload, date_str)
    if not args.out:
        print(f"[INFO] 保存: {out_path}")
    else:
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[INFO] 保存: {out_path}")

    if args.print_prompt:
        print("\n" + "─" * 60)
        print(payload["messages"][1]["content"])
        print("─" * 60)


if __name__ == "__main__":
    main()
