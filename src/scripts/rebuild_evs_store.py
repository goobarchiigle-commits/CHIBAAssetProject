"""
src/scripts/rebuild_evs_store.py

EVS完全性修正 Phase3/5/6/7: 2026-06-01〜現在の全runを実データのみから
schema_version=2で再構築し、真のボトルネックランキングを作成する。

【重要な限界（推測禁止の原則に基づき明記）】
過去のrunでは _build_orders() の audit_sink（stage_audit）が存在しなかった
ため、当時のstage判定を完全再現することはできない。本スクリプトは
data/signals/*.json に保存された実測値（signals/orders/warnings/
send_results）のみから、以下の優先順位で final_stage を再構成する:
  1. send_results に success=True の同symbol/BUYがあれば → ORDER_SENT (executed)
  2. warnings文字列にそのsymbolを含む一致があれば → 該当stage
  3. held(currently_holding数) >= max_positions なら → CAPACITY(推定)
  4. それ以外は → UNTRACKED（stage監査ログが存在しない過去run。
     2026-07-08以降の新規runからは _build_orders() のaudit_sinkにより
     完全なstage追跡が可能になる — 本スクリプトはあくまで暫定復元）

出力: runtime/analytics/executed_vs_skipped_rebuilt.jsonl
      （本番の executed_vs_skipped.jsonl は上書きしない — 安全のため別ファイル）

実行:
    python -m src.scripts.rebuild_evs_store
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.analytics.executed_vs_skipped_expectancy import (
    TradeOpportunityRecord, append_opportunity,
    STAGE_TO_SKIP_REASON, SKIP_UNKNOWN, SKIP_POSITION_FULL,
)

SIGNALS_DIR  = _ROOT / "data" / "signals"
REBUILT_PATH = _ROOT / "runtime" / "analytics" / "executed_vs_skipped_rebuilt.jsonl"
MAX_POS      = 3
START_DATE   = "2026-06-01"


def _classify_from_warning(w: str) -> str:
    if "最大ポジション数" in w:
        return "CAPACITY"
    if "配分上限キャップ" in w:
        return "CAPITAL"
    if "サイジング結果qty=0" in w:
        return "SIZING"
    if "セクター集中制限" in w:
        return "SECTOR_CONCENTRATION"
    if "pre_trade_risk_check" in w:
        return "RISK"
    if "新規 BUY 上限" in w:
        return "DAILY_LIMIT"
    return "UNKNOWN_WARNING_TEXT"


def rebuild() -> list[TradeOpportunityRecord]:
    files = sorted(SIGNALS_DIR.glob("signal_*.json"))
    records: list[TradeOpportunityRecord] = []
    n_runs = 0

    for f in files:
        m = re.match(r"signal_(\d{8})_(\d{6})", f.name)
        if not m:
            continue
        date_compact, time_compact = m.group(1), m.group(2)
        iso_date = f"{date_compact[:4]}-{date_compact[4:6]}-{date_compact[6:]}"
        if iso_date < START_DATE:
            continue

        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue

        signals = d.get("signals", [])
        orders  = d.get("orders", [])
        send_results = d.get("send_results", [])
        warnings = d.get("warnings", []) or []
        mode = d.get("mode", "unknown")

        n_held = sum(1 for s in signals if s.get("currently_holding"))
        # send_results があれば真の約定判定に使う。無ければ orders(候補)から
        # 「発注が試みられたか」だけ分かる（本番の新フックはsend_resultsを使うが、
        # 過去ログにsend_resultsが無いrunはordersを次善のソースとして扱う。
        # ただし executed=True は send_results.success=True でのみ許可する
        # （result.orders在中=候補、ではない、というPhase1の教訓を過去復元にも適用）。
        executed_syms = {
            r.get("symbol") for r in send_results
            if str(r.get("side", "")).upper() == "BUY" and r.get("success")
        }

        run_id = f"{date_compact}_{time_compact}"
        run_ts = f"{iso_date}T{time_compact[:2]}:{time_compact[2:4]}:{time_compact[4:]}+0900"

        buy_cands = [s for s in signals if s.get("signal") == 1 and not s.get("currently_holding")]
        if not buy_cands:
            continue
        n_runs += 1

        for s in buy_cands:
            sym = s["symbol"]
            executed = sym in executed_syms
            if executed:
                final_stage, skip_reason = "ORDER_SENT", None
            else:
                w_match = next((w for w in warnings if sym in w), None)
                if w_match:
                    final_stage = _classify_from_warning(w_match)
                    skip_reason = STAGE_TO_SKIP_REASON.get(final_stage, SKIP_UNKNOWN)
                elif n_held >= MAX_POS:
                    final_stage, skip_reason = "CAPACITY", SKIP_POSITION_FULL
                else:
                    final_stage, skip_reason = "UNTRACKED", SKIP_UNKNOWN

            rec = TradeOpportunityRecord.create(
                eval_date=iso_date,
                symbol=sym,
                executed=executed,
                skip_reason=skip_reason,
                capital_available_pct=0.0,  # 過去ログには保存されておらず復元不可(推測しない)
                portfolio_heat=0.0,
                slot_utilization=n_held / max(1, MAX_POS),
                sector=str(s.get("sector") or "不明"),
                market_regime="unknown",
                atr_pct=0.02,
                rs_rank=int(s.get("rsr_rank") or 50),
                entry_score=float(s.get("rsr") or 50) / 100.0,
                liquidity_score=0.5,
                position_lifecycle_available=False,
                run_id=run_id,
                run_timestamp=run_ts,
                mode=str(mode),
                final_stage=final_stage,
                source_script="rebuild_evs_store(retroactive)",
            )
            records.append(rec)

    print(f"再構築対象run数(候補ありrunのみ): {n_runs}")
    print(f"再構築レコード数: {len(records)}")
    return records


def report_phase5_6_7(records: list[TradeOpportunityRecord]) -> None:
    print("\n" + "=" * 78)
    print("  Phase5: UNCLASSIFIED / UNTRACKED 内訳")
    print("=" * 78)
    untracked = [r for r in records if r.final_stage == "UNTRACKED"]
    unknown_warn = [r for r in records if r.final_stage == "UNKNOWN_WARNING_TEXT"]
    print(f"  UNTRACKED（stage監査ログ無し・容量にも余裕あり）: {len(untracked)}件")
    print(f"  UNKNOWN_WARNING_TEXT（warningsはあるが未分類文字列）: {len(unknown_warn)}件")
    if unknown_warn:
        print("  未分類warning文字列サンプル:")
        seen = set()
        for r in unknown_warn[:10]:
            print(f"    {r.symbol} {r.eval_date} stage={r.final_stage}")

    print("\n" + "=" * 78)
    print("  Phase6: Stage別ランキング（排他的分類 — 各レコード1つのfinal_stageのみ）")
    print("=" * 78)
    stage_counts: dict[str, int] = defaultdict(int)
    for r in records:
        stage_counts[r.final_stage if not r.executed else "ORDER_SENT"] += 1
    total = len(records)
    for stage, n in sorted(stage_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {stage:<25} n={n:>4}  ({round(100*n/max(1,total),1)}%)")

    print("\n" + "=" * 78)
    print("  Phase7: 2026-06-01〜現在 真のボトルネックランキング")
    print("=" * 78)
    skip_reason_counts: dict[str, int] = defaultdict(int)
    for r in records:
        if not r.executed:
            skip_reason_counts[r.skip_reason or SKIP_UNKNOWN] += 1
    for rank, (reason, n) in enumerate(sorted(skip_reason_counts.items(), key=lambda kv: -kv[1]), 1):
        print(f"  {rank}. {reason:<25} n={n}  ({round(100*n/max(1,total),1)}%)")

    n_executed = sum(1 for r in records if r.executed)
    print(f"\n  executed(真の約定)件数 = {n_executed} / {total}"
          f" ({round(100*n_executed/max(1,total),1)}%)")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    records = rebuild()

    REBUILT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REBUILT_PATH.exists():
        REBUILT_PATH.unlink()
    for rec in records:
        append_opportunity(rec, REBUILT_PATH)
    print(f"\n再構築ストア出力先: {REBUILT_PATH}")

    report_phase5_6_7(records)
    return 0


if __name__ == "__main__":
    sys.exit(main())
