"""
src/analytics/live_stage_audit.py

恒久的なLIVEステージ監査ログ（2026-06-29 EVS RCA follow-up）。

背景: 既存の EVS(executed_vs_skipped_expectancy) フックは
skip_reason を "capital_constraint"/"slot_full" の2値ヒューリスティックで
判定するのみで、実際にどのステージ（Ranking/Capital/Sizing/Sector/Risk/
Capacity/Order）で落ちたかを記録していなかった。また "executed" 判定に
既知の不整合がある（2026-06-23 2802.T で確認: 同一runのorders.jsonには
BUY注文が存在するのに EVS store は executed=False, skip_reason=slot_full
として記録していた）。

本モジュールは SignalBridge._build_orders() から呼ばれる audit_sink 経由で、
BUY候補ごとに各ステージの PASS/FAIL を観測専用（observation_only）で
append-only JSONL に記録する。発注ロジックには一切影響しない。

ステージ一覧:
  RANKING               : top_k カットオフで除外されたか
  CAPACITY              : max_positions 枠が空いていたか
  DAILY_LIMIT           : 1日の新規BUY上限に達していたか
  CAPITAL               : 配分上限キャップ（1単元コスト vs alloc_cap）
  SIZING                : サイジング結果 qty>0 か
  SECTOR_CONCENTRATION  : セクター集中制限（adaptive degradation）
  RISK                  : pre_trade_risk_check（symbol/sector/cluster cap）
  ORDER_BUILT           : 最終的に発注可能な OrderInstruction が組まれたか

実行:
    python -m src.analytics.live_stage_audit --report --days 30
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

DEFAULT_AUDIT_DIR = Path("logs/live_stage_audit")


def append_stage_audit(
    today_str: str,
    decisions: list[dict],
    audit_dir: Path = DEFAULT_AUDIT_DIR,
) -> None:
    """
    1回のrunで収集したステージ判定リストを、日付ごとのJSONLファイルに
    1行(1run分)として追記する。観測専用 — 例外は握りつぶし発注ロジックに
    一切影響させない。
    """
    if not decisions:
        return
    try:
        audit_dir.mkdir(parents=True, exist_ok=True)
        path = audit_dir / f"{today_str.replace('-', '')}.jsonl"
        record = {
            "run_at": datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "eval_date": today_str,
            "decisions": decisions,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning("[LIVE_STAGE_AUDIT] append failed: %s", exc)


def load_stage_audit_runs(
    audit_dir: Path = DEFAULT_AUDIT_DIR,
    since_date: "str | None" = None,
) -> list[dict]:
    """audit_dir内の全JSONLファイルからrunレコードを読み込む（新しい順ではない）。"""
    if not audit_dir.exists():
        return []
    runs: list[dict] = []
    for path in sorted(audit_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if since_date and rec.get("eval_date", "") < since_date:
                continue
            runs.append(rec)
    return runs


def summarize_stage_drops(runs: list[dict]) -> dict:
    """
    Phase3相当: ステージ別のdrop件数と割合を集計する。
    同一symbol×同一stageの重複（1日に複数run）は最終判定のみ残す
    （同じ日の最後のrun内での判定を採用）。
    """
    # (eval_date, symbol, stage) -> passed（最後に見たものを採用）
    latest: dict[tuple, bool] = {}
    reasons: dict[tuple, str] = {}
    for run in runs:
        d = run.get("eval_date", "")
        for dec in run.get("decisions", []):
            key = (d, dec.get("symbol", ""), dec.get("stage", ""))
            latest[key] = dec.get("passed", False)
            reasons[key] = dec.get("reason", "")

    stage_totals: dict[str, int] = {}
    stage_fails: dict[str, int] = {}
    reason_counts: dict[str, int] = {}

    for (d, sym, stage), passed in latest.items():
        stage_totals[stage] = stage_totals.get(stage, 0) + 1
        if not passed:
            stage_fails[stage] = stage_fails.get(stage, 0) + 1
            r = reasons.get((d, sym, stage), "unknown")
            reason_counts[r] = reason_counts.get(r, 0) + 1

    stage_drop_pct = {
        stage: round(100.0 * stage_fails.get(stage, 0) / max(1, stage_totals[stage]), 1)
        for stage in stage_totals
    }

    return {
        "stage_totals":    stage_totals,
        "stage_fails":     stage_fails,
        "stage_drop_pct":  stage_drop_pct,
        "reason_counts":   dict(sorted(reason_counts.items(), key=lambda kv: -kv[1])),
    }
