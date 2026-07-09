"""
src/jquants/downloader.py
取得オーケストレーション: Universe解決 → Incremental Update → ステージング取得 → 検証 →
TOPIX同期 → コンパクション（raw/processed 年パーティション再構築）→ 整合性レポート →
catalog/manifest 更新。

責務分離の原則どおり、取得（provider）・保存（cache）・分割再構築（compaction）・
Universe（universe）・検証（validator）を本モジュールが束ねるだけで、各責務を混在させない。
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.jquants import cache, catalog, compaction, manifest, universe
from src.jquants.exceptions import JQuantsError
from src.jquants.integrity import build_integrity_report, save_integrity_report, save_validation_reports
from src.jquants.provider import JQuantsProvider, raw_records_to_frame
from src.jquants.validator import validate_symbol
from src.paths import JQUANTS_PROCESSED_DIR, JQUANTS_RAW_DIR

logger = logging.getLogger(__name__)

_TOPIX_RAW_PATH = JQUANTS_RAW_DIR / "topix.parquet"
_TOPIX_PROCESSED_PATH = JQUANTS_PROCESSED_DIR / "topix.parquet"


class JQuantsDownloader:
    def __init__(self, provider: JQuantsProvider | None = None) -> None:
        self.provider = provider or JQuantsProvider()

    # ------------------------------------------------------------------ #
    def sync_full_market(
        self,
        start: str,
        end: str,
        force_full: bool = False,
        universe_codes: dict[str, str] | None = None,
    ) -> dict:
        """
        universe_codes を指定すると従来通り小規模ユニバース限定の同期（後方互換）。
        None なら universe.py が復元したイベントログから全上場銘柄（上場廃止含む）を対象にする
        （事前に universe.rebuild_universe_events() 等でイベントログが構築済みであること）。

        Returns: 統合レポート dict（per-symbol validation + integrity + manifest記録）。
        """
        cache.ensure_dirs()
        download_started = datetime.now(timezone.utc)

        if universe_codes is None:
            symbols = self._resolve_full_market_symbols(end)
        else:
            symbols = list(universe_codes.keys())

        state = {} if force_full else cache.load_state()
        symbol_reports: list[dict] = []
        skipped: list[str] = []
        touched_years: set[int] = set()

        for symbol in symbols:
            fetch_range = cache.compute_fetch_range(symbol, start, end, state)
            if fetch_range is None:
                skipped.append(symbol)
                logger.info("[JQUANTS_DOWNLOADER] %s: キャッシュ済み・スキップ", symbol)
                continue

            fetch_start, fetch_end = fetch_range
            code = symbol.split(".")[0]  # "1301.T" -> "1301"（J-Quants形式）
            try:
                records = self.provider.daily_quotes_raw(code, fetch_start, fetch_end)
            except Exception as e:
                logger.error("[JQUANTS_DOWNLOADER] %s: 取得失敗 %s", symbol, e)
                symbol_reports.append({
                    "symbol": symbol, "row_count": 0, "status": "fetch_failed",
                    "issue_count": 1, "duplicates": [], "date_order_ok": True,
                    "missing_day_gaps": [], "null_counts": {}, "adjustment_factor_issues": [],
                    "ohlc_consistency_issues": [], "volume_anomalies": {"negative": [], "spikes": []},
                    "error": str(e),
                })
                continue

            cache.save_raw_response(symbol, fetch_start, fetch_end, records)
            new_df = raw_records_to_frame(records)
            if new_df.empty:
                skipped.append(symbol)
                logger.info("[JQUANTS_DOWNLOADER] %s: 新規データなし", symbol)
                continue

            merged_df = cache.merge_and_store_staging(symbol, new_df)
            cache.update_state_entry(state, symbol, merged_df)
            touched_years.update(int(y) for y in merged_df["Date"].dt.year.unique().tolist())

            report = validate_symbol(symbol, merged_df.set_index("Date"))
            symbol_reports.append(report)
            logger.info(
                "[JQUANTS_DOWNLOADER] %s: 新規%d件 合計%d件 status=%s",
                symbol, len(new_df), report["row_count"], report["status"],
            )

        cache.save_state(state)

        self._sync_topix(start, end)

        if universe_codes is None:
            universe.sync_latest_snapshot_diff(self.provider)
        universe.materialize_universe_reference()

        if touched_years:
            compaction.compact_years(touched_years)

        integrity = build_integrity_report(symbol_reports, start, end)
        integrity["skipped_cached_symbols"] = skipped
        save_validation_reports(symbol_reports)
        save_integrity_report(integrity)

        catalog.save_catalog()

        download_finished = datetime.now(timezone.utc)
        catalog_snapshot = catalog.build_catalog()
        manifest_record = manifest.record_run(
            download_started=download_started,
            download_finished=download_finished,
            first_date=catalog_snapshot.get("coverage_start"),
            last_date=catalog_snapshot.get("coverage_end"),
            symbol_count=len(symbol_reports),
            record_count=integrity["total_row_count"],
        )

        return {"symbol_reports": symbol_reports, "integrity": integrity, "manifest": manifest_record}

    # ------------------------------------------------------------------ #
    def _resolve_full_market_symbols(self, as_of: str) -> list[str]:
        ref = universe.reconstruct_universe_asof(as_of)
        if ref.empty:
            raise JQuantsError(
                "Universeイベントログが空です。先に "
                "universe.rebuild_universe_events(provider, start, end) を実行してユニバースを"
                "復元してください（--rebuild-universe）。"
            )
        codes = sorted(ref["code"].astype(str).unique())
        return [f"{c}.T" for c in codes]

    def _sync_topix(self, start: str, end: str) -> None:
        """raw/topix.parquet（生値・列名リネームなし）と processed/topix.parquet
        （Study76互換: DatetimeIndex + Open/High/Low/Close）を差分更新する。"""
        JQUANTS_RAW_DIR.mkdir(parents=True, exist_ok=True)
        JQUANTS_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

        existing = pd.read_parquet(_TOPIX_RAW_PATH) if _TOPIX_RAW_PATH.exists() else pd.DataFrame()
        if not existing.empty:
            last_date = pd.Timestamp(existing["Date"].max())
            fetch_start = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            fetch_start = start

        if pd.Timestamp(fetch_start) <= pd.Timestamp(end):
            try:
                new_raw = self.provider.topix_df(fetch_start, end)
            except Exception as e:
                logger.error("[JQUANTS_DOWNLOADER] TOPIX取得失敗（fail-open・既存データ維持）: %s", e)
                new_raw = pd.DataFrame()
            if not new_raw.empty:
                merged = pd.concat([existing, new_raw], ignore_index=True)
                merged = merged.drop_duplicates(subset=["Date"], keep="last").sort_values("Date").reset_index(drop=True)
                merged.to_parquet(_TOPIX_RAW_PATH)
                existing = merged

        if existing.empty:
            return

        processed = existing.copy()
        processed["Date"] = pd.to_datetime(processed["Date"])
        for col in ("Open", "High", "Low", "Close"):
            if col in processed.columns:
                processed[col] = pd.to_numeric(processed[col], errors="coerce")
        keep_cols = [c for c in ("Open", "High", "Low", "Close") if c in processed.columns]
        processed = processed.set_index("Date").sort_index()[keep_cols]
        # Windows(NTFS既定)は大小文字区別なしのため、Study76が読む "TOPIX.parquet" とも一致する。
        processed.to_parquet(_TOPIX_PROCESSED_PATH)
