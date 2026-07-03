"""
analysis/live_exposure_report.py

Phase2 ライブ稼働状況レポート

集計対象: data/signals/signal_*.json（*_executed.json は除外）
1 data_as_of 日付につき最後の run を採用（同日複数実行の場合は最新を使う）。

出力指標:
  avg_exposure      : 平均エクスポージャー（1 - available_cash / CAPITAL）
  avg_positions     : 平均保有銘柄数
  avg_candidates    : 平均BUY候補数（signal=+1 かつ非保有）
  entry_block_rate  : BUY候補あり → 発注なし の発生率（%）
  zero_exposure_rate: ポジションゼロの日の割合（%）

判断基準（研究値との比較）:
  avg_exposure  0.25〜0.35 → 正常（バックテスト avg_exp ≈ 32.7%）
  avg_exposure < 0.15     → 戦略弱体化（シグナルが少なすぎる）
  avg_exposure >= 0.45    → リスク過大（ポジションサイズ見直し）

実行:
  python analysis/live_exposure_report.py
"""

import sys
import os
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from src.paths import SIGNALS_DIR

CAPITAL       = 2_000_000
SIGNAL_DIR    = SIGNALS_DIR


def load_runs() -> list[dict]:
    """
    data/signals/ からシグナル JSON を読み込み、
    data_as_of 日付ごとに最後の run を返す（重複除去済み）。
    """
    files = sorted(SIGNAL_DIR.glob("signal_*.json"))
    files = [f for f in files if not f.name.endswith("_executed.json")]

    by_date: dict[str, dict] = {}
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [WARN] 読み込み失敗: {f.name} — {e}")
            continue

        date = data.get("data_as_of", "")
        gen  = data.get("generated_at", "")
        if not date:
            continue
        # 同日で複数 run がある場合は最後の run を採用
        if date not in by_date or gen > by_date[date]["generated_at"]:
            by_date[date] = data

    return [by_date[d] for d in sorted(by_date.keys())]


def analyze(runs: list[dict]) -> dict:
    exposures  = []
    positions  = []
    candidates = []
    blocks     = 0  # BUY候補あり → 発注なし の件数

    for run in runs:
        ps     = run.get("portfolio_summary", {})
        sigs   = run.get("signals", [])
        orders = run.get("orders", [])

        # エクスポージャー: 1 - available_cash / CAPITAL
        cash = float(ps.get("available_cash", CAPITAL))
        expo = 1.0 - cash / CAPITAL
        exposures.append(expo)

        # 保有銘柄数
        n_pos = int(ps.get("current_positions", 0))
        positions.append(n_pos)

        # BUY候補数（signal=+1 かつ非保有）
        buy_cands = [
            s for s in sigs
            if s.get("signal") == 1 and not s.get("currently_holding", False)
        ]
        candidates.append(len(buy_cands))

        # エントリーブロック: 候補あり & BUY注文なし
        buy_orders = [o for o in orders if o.get("side") == "BUY"]
        if len(buy_cands) > 0 and len(buy_orders) == 0:
            blocks += 1

    n = len(runs)
    if n == 0:
        return {}

    return {
        "n_runs":             n,
        "date_from":          runs[0].get("data_as_of", "?"),
        "date_to":            runs[-1].get("data_as_of", "?"),
        "avg_exposure":       sum(exposures) / n,
        "avg_positions":      sum(positions) / n,
        "avg_candidates":     sum(candidates) / n,
        "entry_block_rate":   blocks / n * 100,
        "zero_exposure_rate": sum(1 for e in exposures if e == 0) / n * 100,
        "exposures":          exposures,
        "positions":          positions,
        "candidates":         candidates,
        "dates":              [r.get("data_as_of", "?") for r in runs],
    }


def judge(avg_exposure: float) -> str:
    if avg_exposure < 0.10:
        return "🔴 CRITICAL: エクスポージャーが極めて低い（シグナル生成を確認）"
    elif avg_exposure < 0.15:
        return "⚠️  WARNING: 戦略弱体化の疑い"
    elif avg_exposure >= 0.45:
        return "⚠️  WARNING: リスク過大（ポジションサイズ見直し）"
    elif 0.25 <= avg_exposure <= 0.35:
        return "✅  NORMAL: 研究値と一致（バックテスト avg_exp ≈ 32.7%）"
    else:
        return "⚡  CAUTION: 許容範囲内（継続観察）"


def candidate_block_reason(avg_cands: float, block_rate: float) -> str:
    """候補 0 vs ブロックの切り分け"""
    if avg_cands < 0.1:
        return "→ 戦略自体がシグナルを出していない（市場環境 or フィルター過剰）"
    elif block_rate > 50:
        return "→ シグナルは出ているが発注制約でブロック（max_pos / 現金不足）"
    else:
        return "→ 通常稼働中"


def main() -> int:
    if not SIGNAL_DIR.exists():
        print(f"[FATAL] {SIGNAL_DIR} が見つかりません。プロジェクトルートから実行してください。")
        return 1

    runs = load_runs()
    if not runs:
        print("シグナル JSON がありません。先に python run_live_signal.py を実行してください。")
        return 1

    m = analyze(runs)

    print("=" * 64)
    print("  Phase2 ライブ稼働状況レポート")
    print("=" * 64)
    print(f"\n  集計対象: {m['n_runs']} 日 / {m['date_from']} 〜 {m['date_to']}\n")

    avg_e = m["avg_exposure"]
    avg_p = m["avg_positions"]
    avg_c = m["avg_candidates"]
    blk   = m["entry_block_rate"]
    zero  = m["zero_exposure_rate"]

    print(f"  {'指標':<26} {'実測値':>10}   {'研究値 / 基準'}")
    print(f"  {'-'*62}")
    print(f"  {'avg_exposure':<26} {avg_e:>9.1%}   研究値: 32.7%  / 正常: 25〜35%")
    print(f"  {'avg_positions':<26} {avg_p:>10.2f}   研究値: 1.91")
    print(f"  {'avg_candidates (BUY候補/日)':<26} {avg_c:>10.2f}   0 なら市場環境が悪いか戦略停止")
    print(f"  {'entry_block_rate':<26} {blk:>9.1f}%   候補あり→発注なしの率")
    print(f"  {'zero_exposure_rate':<26} {zero:>9.1f}%   研究値: ~19%")

    print(f"\n  エクスポージャー判定: {judge(avg_e)}")
    print(f"  候補・ブロック診断 : {candidate_block_reason(avg_c, blk)}")

    # 直近 10 営業日の時系列
    n_show = min(10, len(runs))
    last_n = list(zip(
        m["dates"][-n_show:],
        m["exposures"][-n_show:],
        m["positions"][-n_show:],
        m["candidates"][-n_show:],
    ))

    print(f"\n  直近 {n_show} 日の推移:")
    print(f"  {'日付':<12} {'exposure':>9} {'positions':>10} {'candidates':>11}")
    print(f"  {'-'*50}")
    for date, expo, pos, cand in last_n:
        flag = " ⚠" if expo == 0 else ""
        print(f"  {date:<12} {expo:>8.1%}  {pos:>10.1f}  {cand:>11.1f}{flag}")

    print("\n" + "=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
