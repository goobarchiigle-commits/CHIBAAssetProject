"""
src/scripts/jquants_sync.py
J-Quants データ同期 CLI（Study75 データ取得基盤）。

使い方:
  # 小規模ユニバース限定・初回フル取得（後方互換モード）
  python -m src.scripts.jquants_sync --mode full --universe src/configs/rsr_universe_42.csv --start 2010-01-01 --end 2025-12-31

  # 全上場銘柄（上場廃止含む）を対象にした差分更新（旧・銘柄別エンジン）
  python -m src.scripts.jquants_sync --full-market --end 2026-07-09

  # Study75 Full Download（Strategy C・日次イテレーション・正本。中断しても再実行で自動再開）
  python -m src.scripts.jquants_sync --study75-download --end 2026-07-09

  # Universeイベントログの全営業日フル復元（初回のみ・重い処理）
  python -m src.scripts.jquants_sync --rebuild-universe --start 2016-01-01 --end 2026-07-09

  # Study76互換の銘柄別 processed/{symbol}.parquet を書き出す
  python -m src.scripts.jquants_sync --materialize src/configs/universe/rsr42_trading.json

  # 再取得なしでステージング→年パーティションの再構築のみ
  python -m src.scripts.jquants_sync --compact-only

  # 整合性チェックのみ（再取得なし・認証設定のみ確認）
  python -m src.scripts.jquants_sync --check-only

  # Study75日次ステージング（cache/daily/day_*.parquet）の整合性検証（再取得なし・API通信なし）
  python -m src.scripts.jquants_sync --verify

  # Full Download事前見積もり（API通信なし・ダウンロード実行なし）
  python -m src.scripts.jquants_sync --preflight

前提:
  .env に JQUANTS_API_KEY（J-Quantsダッシュボード [設定 » APIキー] で発行）が設定済みであること。
  未設定の場合は明確なエラーメッセージで終了する（ダミー認証は使わない）。
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd

from src.jquants import (
    cache, catalog, compaction, manifest, preflight as preflight_module,
    study75_downloader, universe, verify as verify_module,
)
from src.jquants.config import load_config
from src.jquants.downloader import JQuantsDownloader
from src.jquants.exceptions import JQuantsError
from src.jquants.provider import JQuantsProvider, estimate_subscription_floor
from src.jquants.study75_adapter import materialize_symbol_panels
from src.paths import JQUANTS_LOGS_DIR

JQUANTS_LOGS_DIR.mkdir(parents=True, exist_ok=True)  # FileHandler より前に作成する必要がある

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(JQUANTS_LOGS_DIR / "jquants_sync.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("jquants_sync")


def load_universe(path: Path) -> dict[str, str]:
    """CSV(symbol,sector) または JSON({"symbols": {...}}) からユニバースを読む。"""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "sector" in df.columns:
            return dict(zip(df["symbol"], df["sector"]))
        return {sym: "不明" for sym in df["symbol"]}
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("symbols", data)
    raise ValueError(f"未対応のユニバースファイル形式: {path}")


def _print_integrity(integrity: dict) -> None:
    print("=" * 60)
    print(f"  対象銘柄数     : {integrity['symbol_count']}")
    print(f"  合計取得行数   : {integrity['total_row_count']}")
    print(f"  取得日数（営業日参照値）: {integrity.get('requested_trading_days', 'N/A')}")
    print(f"  問題あり銘柄数 : {integrity['symbol_count_with_issues']}")
    print(f"  欠損ギャップ日数合計: {integrity['total_missing_gap_days']}")
    print(f"  キャッシュ済みスキップ: {len(integrity['skipped_cached_symbols'])}")
    if integrity["symbols_with_issues"]:
        print(f"  要確認銘柄: {integrity['symbols_with_issues']}")
    print("=" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(description="J-Quants データ同期")
    parser.add_argument("--universe", help="ユニバースファイル（CSV or JSON）。省略時は全上場銘柄が対象")
    parser.add_argument(
        "--start", default=None,
        help="取得開始日。省略時は契約データ提供開始日の推定値（今日から10年前・ローリング窓と仮定）",
    )
    parser.add_argument("--end", help="取得終了日")
    parser.add_argument("--mode", choices=["full", "incremental"], default="incremental")
    parser.add_argument("--full-market", action="store_true", help="[旧・銘柄別エンジン] 全上場銘柄（上場廃止含む）を対象に同期")
    parser.add_argument(
        "--study75-download", action="store_true",
        help="[Strategy C・正本] 日次イテレーションでFull Download。中断しても再実行で自動再開",
    )
    parser.add_argument(
        "--catalog", action="store_true",
        help="metadata/catalog.json 生成 + manifest.json へのランレコード追記（再取得なし・API通信なし）",
    )
    parser.add_argument(
        "--rebuild-universe", action="store_true",
        help="[Option B・正本] Universeイベントログをcache/daily/の検証済みステージングから完全オフラインで復元（API通信ゼロ）",
    )
    parser.add_argument(
        "--rebuild-universe-live", action="store_true",
        help="[Option B・ライブAPI] daily barsを日毎に取得しながらUniverseイベントログを復元",
    )
    parser.add_argument(
        "--rebuild-universe-legacy", action="store_true",
        help="[Option A・レガシー] listed/masterを日毎にポーリングしてUniverseイベントログを復元（比較用）",
    )
    parser.add_argument("--enrich-universe", action="store_true", help="listed/masterでUniverse参照テーブルの会社名・セクター等を銘柄単位で補完")
    parser.add_argument("--materialize", help="Study76互換の銘柄別parquetを書き出す（ユニバースファイル指定）")
    parser.add_argument("--compact-only", action="store_true", help="再取得せずステージング→年パーティションの再構築のみ")
    parser.add_argument("--check-only", action="store_true", help="再取得せず設定・接続前提のみ確認する")
    parser.add_argument("--verify", action="store_true", help="Study75日次ステージングの整合性検証（再取得なし・API通信なし）")
    parser.add_argument("--preflight", action="store_true", help="Full Download事前見積もり（API通信なし・ダウンロード実行なし）")
    args = parser.parse_args()
    user_specified_start = args.start is not None  # --preflightのcoverage_is_estimated判定に使う

    if args.check_only:
        cfg = load_config()
        print(f"[CHECK] JQUANTS_API_KEY 設定済み: {bool(cfg.api_key)}")
        print(f"[CHECK] base_url: {cfg.base_url}")
        state = cache.load_state()
        print(f"[CHECK] キャッシュ済み銘柄数: {len(state)}")
        return 0

    if args.preflight:
        report = preflight_module.run_preflight(
            start=args.start if user_specified_start else None, end=args.end,
        )
        print("=" * 60)
        print(f"  想定契約データ提供開始日: {report['coverage_start']}"
              f"{'（推定値・detect_subscription_floor()で実測確認推奨）' if report['coverage_is_estimated'] else '（指定値）'}")
        print(f"  想定契約データ提供終了日: {report['coverage_end']}")
        print(f"  対象営業日数         : {report['trading_day_count']}")
        print(f"  推定リクエスト数     : {report['estimated_requests']}")
        print(f"  推定所要時間         : {report['estimated_runtime_human']}"
              f"（{report['estimated_runtime_sec']}秒・base_sleep=0.05s想定）")
        print(f"  推定ディスク使用量   : {report['estimated_disk_gb']} GB")
        print(f"  空きディスク容量     : {report['disk']['free_gb']} GB"
              f"（最低必要: {report['disk']['minimum_required_gb']} GB・"
              f"{'十分' if report['disk']['sufficient'] else '不足'}）")
        print(f"  対象年               : {report['expected_download_years']}")
        print("=" * 60)
        return 0 if report["disk"]["sufficient"] else 1

    if args.verify:
        report = verify_module.verify_daily_staging()
        print("=" * 60)
        print(f"  検証対象日数     : {report['checked_dates']}")
        print(f"  欠落ファイル     : {len(report['missing_files'])}")
        print(f"  破損Parquet      : {len(report['corrupted_files'])}")
        print(f"  行数不一致       : {len(report['row_count_mismatches'])}")
        print(f"  ハッシュ不一致   : {len(report['hash_mismatches'])}")
        print(f"  総合判定         : {report['status']}")
        print("=" * 60)
        for label, key in (
            ("欠落ファイル", "missing_files"), ("破損Parquet", "corrupted_files"),
            ("行数不一致", "row_count_mismatches"), ("ハッシュ不一致", "hash_mismatches"),
        ):
            if report[key]:
                print(f"  [{label}] {report[key]}")
        return 0 if report["status"] == "ok" else 1

    if args.compact_only:
        result = compaction.compact_years(None)
        print(f"[COMPACT] 再構築した年: {result['years']}")
        for year, counts in result["row_counts"].items():
            print(f"  {year}: raw={counts['raw']} processed={counts['processed']}")
        return 0

    if args.materialize:
        universe_path = Path(args.materialize)
        if not universe_path.exists():
            logger.error("ユニバースファイルが存在しません: %s", universe_path)
            return 1
        symbols = list(load_universe(universe_path).keys())
        result = materialize_symbol_panels(symbols)
        print(f"[MATERIALIZE] 生成銘柄数: {len(result['materialized'])} / 欠落: {len(result['missing'])}")
        if result["missing"]:
            print(f"  欠落銘柄: {result['missing']}")
        return 0

    if args.rebuild_universe:
        # Option B・オフライン正本（2026-07-10採用）: cache/daily/ の検証済みステージングのみを
        # 読む・API通信ゼロ。--end省略時はステージング済みデータの全期間が対象。
        # Full Download（--full-market）未実行/未完了だと途中で停止する。
        result = universe.rebuild_universe_events_from_staged_bars(
            args.start if user_specified_start else None, args.end, force=(args.mode == "full"),
        )
        print(f"[REBUILD_UNIVERSE] (Option B・オフライン) 処理営業日数: {result['processed_days']} "
              f"/ 追記イベント数: {result['events_written']}")
        if result["stopped_at_missing_day"]:
            print(f"  未ステージングの営業日で停止: {result['stopped_at_missing_day']}"
                  f"（先に --full-market でFull Downloadを進めてから再実行してください）")
        print(f"  最終処理日: {result['last_processed_date']}")
        universe.materialize_universe_reference()
        return 0

    if args.enrich_universe:
        # listed/master = メタデータ補完専用。コード単位で1回だけ問い合わせる（日次ではない）。
        provider = JQuantsProvider()
        ref = universe.enrich_universe_reference_with_listed_info(provider)
        enriched = int((ref["company_name"] != "").sum())
        print(f"[ENRICH_UNIVERSE] 補完済み銘柄数: {enriched} / 全体: {len(ref)}")
        return 0

    if args.catalog:
        from datetime import datetime, timezone
        catalog_snapshot = catalog.build_catalog()
        catalog.save_catalog()
        record = manifest.record_run(
            download_started=datetime.now(timezone.utc),
            download_finished=datetime.now(timezone.utc),
            first_date=catalog_snapshot.get("coverage_start"),
            last_date=catalog_snapshot.get("coverage_end"),
            symbol_count=catalog_snapshot.get("total_symbols", 0),
            record_count=catalog_snapshot.get("total_rows", 0),
        )
        print("=" * 60)
        print(f"  coverage_start   : {catalog_snapshot.get('coverage_start')}")
        print(f"  coverage_end     : {catalog_snapshot.get('coverage_end')}")
        print(f"  total_rows       : {catalog_snapshot.get('total_rows')}")
        print(f"  total_symbols    : {catalog_snapshot.get('total_symbols')}")
        print(f"  dataset_hash     : {catalog_snapshot.get('dataset_hash', '')[:16]}...")
        print(f"  git_commit       : {record.get('git_commit', '')[:12]}")
        print("=" * 60)
        return 0

    if not args.end:
        logger.error(
            "--end は必須です"
            "（--check-only/--verify/--preflight/--compact-only/--materialize/--rebuild-universe/--enrich-universe を除く）"
        )
        return 1

    if args.start is None:
        args.start = estimate_subscription_floor()

    if args.study75_download:
        provider = JQuantsProvider()
        try:
            result = study75_downloader.download_full_market_by_day(
                provider, args.start, args.end, force_full=(args.mode == "full"),
            )
        except JQuantsError as e:
            logger.error("Study75 Full Download失敗: %s", e)
            return 1
        print("=" * 60)
        print(f"  処理日数         : {result['days_processed']}")
        print(f"  スキップ済み日数 : {result['days_skipped']}")
        print(f"  検証エラー日数   : {result['days_failed_validation']}")
        print(f"  総レコード数     : {result['total_rows']}")
        print(f"  再構築した年     : {result['years_touched']}")
        if result["validation_issues"]:
            print(f"  検証issues件数   : {len(result['validation_issues'])}")
        print("=" * 60)
        return 0 if result["days_failed_validation"] == 0 else 1

    if args.rebuild_universe_live:
        # Option B・ライブAPI版: daily barsを日毎に取得しながらUniverseを導出する
        # （価格ダウンロードと別に単独実行すると--rebuild-universe-legacyと同等のリクエスト数がかかる）。
        provider = JQuantsProvider()
        result = universe.rebuild_universe_events_from_daily_bars(provider, args.start, args.end, force=(args.mode == "full"))
        print(f"[REBUILD_UNIVERSE] (Option B・ライブAPI) 処理営業日数: {result['processed_days']} "
              f"/ 追記イベント数: {result['events_written']}")
        print(f"  最終処理日: {result['last_processed_date']}")
        universe.materialize_universe_reference()
        return 0

    if args.rebuild_universe_legacy:
        # Option A・レガシー: listed/masterを日毎にポーリング（比較・フォールバック用に残置）。
        provider = JQuantsProvider()
        result = universe.rebuild_universe_events(provider, args.start, args.end, force=(args.mode == "full"))
        print(f"[REBUILD_UNIVERSE] (Option A・レガシー) 処理営業日数: {result['processed_days']} "
              f"/ 追記イベント数: {result['events_written']}")
        print(f"  最終処理日: {result['last_processed_date']}")
        universe.materialize_universe_reference()
        return 0

    if args.full_market:
        universe_codes = None
        logger.info("全上場銘柄モード（Universeイベントログから復元）")
    else:
        if not args.universe:
            logger.error("--universe は必須です（--full-market 指定時を除く）")
            return 1
        universe_path = Path(args.universe)
        if not universe_path.exists():
            logger.error("ユニバースファイルが存在しません: %s", universe_path)
            return 1
        universe_codes = load_universe(universe_path)
        logger.info("ユニバース読込: %d銘柄 (%s)", len(universe_codes), universe_path)

    try:
        downloader = JQuantsDownloader()
        result = downloader.sync_full_market(
            start=args.start,
            end=args.end,
            force_full=(args.mode == "full"),
            universe_codes=universe_codes,
        )
    except JQuantsError as e:
        logger.error("同期失敗: %s", e)
        return 1

    _print_integrity(result["integrity"])
    print(f"[MANIFEST] git_commit={result['manifest'].get('git_commit', '')[:12]} "
          f"dataset_hash={result['manifest'].get('dataset_hash', '')[:12]}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
