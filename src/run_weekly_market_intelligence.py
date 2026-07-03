"""
src/run_weekly_market_intelligence.py — Weekly market intelligence entry point

Usage:
    python src/run_weekly_market_intelligence.py
    python src/run_weekly_market_intelligence.py --dry-run
    python src/run_weekly_market_intelligence.py --no-email
    python src/run_weekly_market_intelligence.py --date 2026-05-16

Scheduled via: scripts/run_weekly_market_intelligence.ps1
  → Windows Task Scheduler: WeeklyMarketIntelligence / FRI 15:30
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.analytics.weekly_market_intelligence import (
    WeeklyMarketIntelligenceEngine,
    WeeklyIntelligenceError,
)
from src.paths import BASE_DIR, RESEARCH_PRIORITY_REPORT_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run_weekly_market_intelligence")


def main() -> int:
    p = argparse.ArgumentParser(description="週次市場インテリジェンス エンジン")
    p.add_argument("--dry-run", action="store_true",
                   help="ファイル書き出し・メール送信なし（動作確認用）")
    p.add_argument("--no-email", action="store_true",
                   help="メール送信をスキップ")
    p.add_argument("--date", default="",
                   help="基準日 YYYY-MM-DD（省略時: 直前の金曜日）")
    args = p.parse_args()

    reference_date: date | None = None
    if args.date:
        try:
            reference_date = date.fromisoformat(args.date)
        except ValueError:
            logger.error("--date の形式が不正です: %s (YYYY-MM-DD を指定)", args.date)
            return 2

    engine = WeeklyMarketIntelligenceEngine(
        base_dir=BASE_DIR,
        report_dir=RESEARCH_PRIORITY_REPORT_DIR,
    )

    try:
        report = engine.run(
            reference_date=reference_date,
            send_email=(not args.dry_run and not args.no_email),
            dry_run=args.dry_run,
        )
    except WeeklyIntelligenceError as exc:
        logger.critical("取引履歴破損によりAbort: %s", exc)
        return 1
    except Exception as exc:
        logger.critical("予期しないエラー: %s", exc, exc_info=True)
        return 1

    if args.dry_run:
        logger.info("[DRY RUN] 完了。ファイルは書き出されていません。")
    else:
        logger.info("MD  : %s", report.report_md_path)
        logger.info("HTML: %s", report.report_html_path)
        logger.info("JSON: %s", report.report_json_path)
        logger.info("優先度 Top3:")
        for pr in report.research_priorities[:3]:
            logger.info("  [%d] %s (%s)", pr.rank, pr.title, pr.confidence)

    return 0


if __name__ == "__main__":
    sys.exit(main())
