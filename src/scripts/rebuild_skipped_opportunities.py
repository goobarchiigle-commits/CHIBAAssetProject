"""
src/scripts/rebuild_skipped_opportunities.py

Opportunity Capture Phase5: 2026-06-01〜現在の skipped_opportunities.jsonl
を実データのみから再構築し、forward_return enrichmentまで実行する。

runtime/analytics/skipped_opportunities.jsonl は元々空（一度もwriterが
実行されていなかった — Phase1参照）なので、このファイルへの直接書き込みは
既存データの上書きにはならない。

実行:
    python -m src.scripts.rebuild_skipped_opportunities
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.analytics.skipped_opportunity_analytics import (
    SkippedOpportunityRecord, append_skipped_opportunity, enrich_and_rewrite_store,
    STAGE_TO_REJECTION_REASON, REJECTION_STALE_SIGNAL, REJECTION_POSITION_FULL,
)

SIGNALS_DIR  = _ROOT / "data" / "signals"
OHLCV_DIR    = _ROOT / "cache" / "ohlcv"
TARGET_PATH  = _ROOT / "runtime" / "analytics" / "skipped_opportunities.jsonl"
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


def rebuild() -> list[SkippedOpportunityRecord]:
    files = sorted(SIGNALS_DIR.glob("signal_*.json"))
    records: list[SkippedOpportunityRecord] = []

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
        send_results = d.get("send_results", [])
        warnings = d.get("warnings", []) or []
        mode = d.get("mode", "unknown")
        available_cash = float((d.get("portfolio_summary") or {}).get("available_cash", 0.0))

        n_held = sum(1 for s in signals if s.get("currently_holding"))
        executed_syms = {
            r.get("symbol") for r in send_results
            if str(r.get("side", "")).upper() == "BUY" and r.get("success")
        }
        run_id = f"{date_compact}_{time_compact}"
        run_ts = f"{iso_date}T{time_compact[:2]}:{time_compact[2:4]}:{time_compact[4:]}+0900"

        buy_cands = [s for s in signals if s.get("signal") == 1 and not s.get("currently_holding")]
        for s in buy_cands:
            sym = s["symbol"]
            if sym in executed_syms:
                continue  # executed — not a skip
            w_match = next((w for w in warnings if sym in w), None)
            if w_match:
                stage = _classify_from_warning(w_match)
            elif n_held >= MAX_POS:
                stage = "CAPACITY"
            else:
                stage = "UNTRACKED"
            rejection_reason = STAGE_TO_REJECTION_REASON.get(stage, REJECTION_STALE_SIGNAL)

            rec = SkippedOpportunityRecord.create(
                run_id=run_id,
                timestamp=run_ts,
                symbol=sym,
                strategy_id="fujiko_v2",
                signal_strength=float(s.get("rsr") or 50.0),
                predicted_rank=int(s.get("rsr_rank") or 50),
                rejection_reason=rejection_reason,
                available_cash=available_cash,
                alloc_cap=0,
                intended_position_size=0,
                sector_state=str(s.get("sector") or "不明"),
                concentration_state=0.0,
                price_at_signal=float(s.get("entry_price") or 0.0),
                mode=str(mode),
                final_stage=stage,
                source_script="rebuild_skipped_opportunities(retroactive)",
            )
            records.append(rec)

    return records


def _price_fetcher(sym: str, iso_date: str) -> "list[float] | None":
    p = OHLCV_DIR / f"{sym}.parquet"
    if not p.exists():
        return None
    try:
        df = pd.read_parquet(p)
        df.index = pd.to_datetime(df.index)
        mask = df.index >= pd.Timestamp(iso_date)
        if not mask.any():
            return None
        return df.loc[mask, "Close"].tolist()[:6]
    except Exception:
        return None


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    records = rebuild()
    print(f"再構築レコード数: {len(records)}")

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    if TARGET_PATH.exists():
        TARGET_PATH.unlink()
    for rec in records:
        append_skipped_opportunity(rec, TARGET_PATH)
    print(f"出力先: {TARGET_PATH}")

    n_enriched = enrich_and_rewrite_store(TARGET_PATH, _price_fetcher)
    print(f"forward_return enrichment 完了: {n_enriched}件")

    # price_at_signal が 0 の場合 forward_return 計算はできない点に注意
    # （過去ログには当時の株価が保存されていないケースがある — 捏造しない）。
    n_with_price = sum(1 for r in records if r.price_at_signal > 0)
    print(f"price_at_signal>0（enrichment対象になり得た件数）: {n_with_price}/{len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
