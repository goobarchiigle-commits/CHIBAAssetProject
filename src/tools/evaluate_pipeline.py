"""
evaluate_pipeline.py
data/trades_log.csv を入力に pipeline.py を実行し、valid戦略を抽出・評価する
"""
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import json
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd

# ── パス設定 ─────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))
from src.paths import BASE_DIR

INPUT_DATA   = BASE_DIR / "data" / "trades_log.csv"
RESEARCH_DIR = BASE_DIR / "research"
N_ROUNDS     = 50


# ── スコア計算 ────────────────────────────────────
def _score(entry: dict) -> float:
    """
    strategy_score から複合スコアを算出。
    score = robustness * (1 - overfitting_risk) * execution_feasibility
    フィールド欠損時はメトリクスの sharpe を代替利用。
    """
    strat = entry.get("strategy", {})
    ss = strat.get("strategy_score", {}) if isinstance(strat, dict) else {}

    if ss:
        r   = float(ss.get("robustness", 0.5))
        o   = float(ss.get("overfitting_risk", 0.5))
        e   = float(ss.get("execution_feasibility", 0.5))
        return round(r * (1 - o) * e, 4)

    # strategy_score がない場合は sharpe フォールバック
    metrics = entry.get("metrics", {})
    sharpe  = metrics.get("sharpe") if metrics else None
    return float(sharpe) if sharpe is not None else 0.0


def _extract_metrics(entry: dict) -> dict:
    metrics = entry.get("metrics") or {}
    strat   = entry.get("strategy", {}) if isinstance(entry.get("strategy"), dict) else {}
    ss      = strat.get("strategy_score", {}) if isinstance(strat, dict) else {}
    inner   = strat.get("strategy", {}) if isinstance(strat, dict) else {}

    return {
        "score":        _score(entry),
        "sharpe":       metrics.get("sharpe"),
        "n_trades":     metrics.get("n_trades"),
        "win_rate":     metrics.get("win_rate"),
        "max_drawdown": metrics.get("max_drawdown"),
        "mean_ret":     metrics.get("mean_ret"),
        "robustness":   ss.get("robustness"),
        "overfitting_risk": ss.get("overfitting_risk"),
        "entry":        inner.get("entry", strat.get("entry", "N/A")),
        "exit":         inner.get("exit",  strat.get("exit",  "N/A")),
    }


# ── pipeline.py 実行 ──────────────────────────────
def run_pipeline(data_path: Path, n: int) -> tuple[int, str, Path | None]:
    """
    pipeline.py をサブプロセスで実行。
    Returns: (returncode, stdout+stderr, valid_strategies_json_path)
    """
    cmd = [
        sys.executable, str(_HERE / "pipeline.py"),
        "--data", str(data_path),
        "--n",    str(n),
    ]

    print(f"[EXEC] {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(BASE_DIR),
    )

    output = proc.stdout + proc.stderr

    # 本日付の valid_strategies JSON を探す
    date_str   = datetime.now().strftime("%Y-%m-%d")
    json_path  = RESEARCH_DIR / f"valid_strategies_{date_str}.json"
    found_path = json_path if json_path.exists() else None

    return proc.returncode, output, found_path


# ── valid戦略の読み込み・集計 ─────────────────────
def load_valid_strategies(json_path: Path) -> list[dict]:
    if not json_path.exists():
        return []
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return [e for e in raw if e.get("valid", False)]


# ── 結果表示 ──────────────────────────────────────
def print_results(valid_entries: list[dict], n_rounds: int) -> None:
    valid_count = len(valid_entries)
    print("\n" + "=" * 60)
    print("EVALUATE PIPELINE — 結果サマリー")
    print("=" * 60)
    print(f"実行ラウンド数 : {n_rounds}")
    print(f"valid_count   : {valid_count}")

    if valid_count == 0:
        print("valid戦略     : なし（top_5 = []）")
        print("=" * 60)
        return

    # スコア降順ソート
    scored = sorted(valid_entries, key=_score, reverse=True)
    top5   = scored[:5]

    print(f"\n--- top_{min(5, valid_count)} strategies (score降順) ---")
    rows = []
    for rank, entry in enumerate(top5, start=1):
        m = _extract_metrics(entry)
        rows.append({
            "rank":      rank,
            "score":     m["score"],
            "sharpe":    m["sharpe"],
            "n_trades":  m["n_trades"],
            "win_rate":  m["win_rate"],
            "max_dd":    m["max_drawdown"],
            "mean_ret":  m["mean_ret"],
            "robust":    m["robustness"],
            "overfit":   m["overfitting_risk"],
        })

    df = pd.DataFrame(rows).set_index("rank")
    print(df.to_string())

    print("\n--- top_5 entry/exit 条件 ---")
    for rank, entry in enumerate(top5, start=1):
        m = _extract_metrics(entry)
        print(f"[{rank}] entry : {m['entry']}")
        print(f"    exit  : {m['exit']}")

    print("=" * 60)


# ── メイン ────────────────────────────────────────
def main() -> None:
    # 入力確認
    if not INPUT_DATA.exists():
        raise FileNotFoundError(f"入力データが見つかりません: {INPUT_DATA}")

    df_in = pd.read_csv(INPUT_DATA)
    print(f"[INPUT] {INPUT_DATA}")
    print(f"  行数: {len(df_in)}  列: {list(df_in.columns)}")

    # pipeline.py 実行
    rc, output, json_path = run_pipeline(INPUT_DATA, N_ROUNDS)

    # パイプライン出力をそのまま表示
    print("\n[PIPELINE OUTPUT]")
    print(output.strip())

    if rc != 0:
        print(f"\n[WARN] pipeline.py 終了コード: {rc}")

    # valid戦略の読み込み
    if json_path is None:
        print("\n[WARN] valid_strategies JSON が見つかりません")
        print_results([], N_ROUNDS)
        return

    valid_entries = load_valid_strategies(json_path)
    print_results(valid_entries, N_ROUNDS)


if __name__ == "__main__":
    main()
