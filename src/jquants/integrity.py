"""
src/jquants/integrity.py
取得結果の整合性サマリーレポート生成。

出力項目（要件どおり）:
  - 取得件数（銘柄別行数 + 合計）
  - 更新日時
  - 対象期間
  - 取得日数（対象期間内のJPX営業日数・実測ではなく参照値）
  - 欠損日数
  - 対象銘柄数
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from src.market.jpx_calendar import JPXCalendar
from src.paths import JQUANTS_METADATA_DIR

logger = logging.getLogger(__name__)
_JST = timezone(timedelta(hours=9))


def _trading_day_count(start: str, end: str) -> int:
    try:
        calendar = JPXCalendar()
        return sum(1 for d in pd.bdate_range(start, end) if calendar.is_trading_day(d))
    except (ValueError, TypeError) as e:
        logger.warning("[JQUANTS_INTEGRITY] trading_day_count 計算失敗（fail-open）: %s", e)
        return 0


def build_integrity_report(
    symbol_reports: list[dict],
    requested_start: str,
    requested_end: str,
) -> dict:
    """
    validator.validate_symbol() のレポート群から統合サマリーを作る。
    """
    total_rows = sum(r["row_count"] for r in symbol_reports)
    total_missing_gap_days = sum(
        gap["gap_days"] for r in symbol_reports for gap in r["missing_day_gaps"]
    )
    symbols_with_issues = [r["symbol"] for r in symbol_reports if r["status"] != "ok"]

    return {
        "generated_at":          datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
        "requested_period":      {"start": requested_start, "end": requested_end},
        "requested_trading_days": _trading_day_count(requested_start, requested_end),
        "symbol_count":          len(symbol_reports),
        "symbol_count_with_issues": len(symbols_with_issues),
        "symbols_with_issues":   symbols_with_issues,
        "total_row_count":       total_rows,
        "total_missing_gap_days": total_missing_gap_days,
        "per_symbol": {
            r["symbol"]: {
                "row_count": r["row_count"],
                "status":    r["status"],
                "issue_count": r["issue_count"],
            }
            for r in symbol_reports
        },
    }


def save_integrity_report(report: dict, label: str = "integrity") -> Path:
    JQUANTS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(_JST).strftime("%Y%m%d_%H%M%S")
    path = JQUANTS_METADATA_DIR / f"{label}_{date_str}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[JQUANTS_INTEGRITY] レポート保存: %s", path)
    return path


def save_validation_reports(symbol_reports: list[dict], label: str = "validation") -> Path:
    JQUANTS_METADATA_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now(_JST).strftime("%Y%m%d_%H%M%S")
    path = JQUANTS_METADATA_DIR / f"{label}_{date_str}.json"
    path.write_text(json.dumps(symbol_reports, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[JQUANTS_VALIDATION] レポート保存: %s", path)
    return path
