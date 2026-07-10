# Study75 Dataset Snapshot — Protection Record

**日付**: 2026-07-10
**目的**: Study75A/B着手前の時点でデータセット状態を凍結記録する。本書自体はデータセットへの
変更を一切伴わない（read-onlyでcatalog.json/manifest.jsonを読んだのみ）。

---

## 1. catalog.json サマリー

| 項目 | 値 |
|---|---|
| coverage_start | 2016-07-11 |
| coverage_end | 2026-07-09 |
| total_rows（processed daily_bars_*のみ） | 10,084,970 |
| total_symbols（survivorship-free・上場廃止含む） | 5,376 |
| dataset_hash | `c736b5027a52bc09d2901dfc76c6a34d0437c7ff1edb1eac39ab7fc506e4015d` |
| catalog生成日時 | 2026-07-10T09:16:50+0900 |

### raw/processed 年別ファイル（両者とも行数一致・正規化前後で欠落なし）

| 年 | raw行数 | processed行数 | 対象期間 |
|---|---|---|---|
| 2016 | 446,916 | 446,916 | 07-11 〜 12-30 |
| 2017 | 953,757 | 953,757 | 01-04 〜 12-29 |
| 2018 | 964,473 | 964,473 | 01-04 〜 12-28 |
| 2019 | 963,118 | 963,118 | 01-04 〜 12-30 |
| 2020 | 977,021 | 977,021 | 01-06 〜 12-30 |
| 2021 | 1,004,780 | 1,004,780 | 01-04 〜 12-30 |
| 2022 | 1,023,708 | 1,023,708 | 01-04 〜 12-30 |
| 2023 | 1,052,339 | 1,052,339 | 01-04 〜 12-29 |
| 2024 | 1,070,937 | 1,070,937 | 01-04 〜 12-30 |
| 2025 | 1,068,092 | 1,068,092 | 01-06 〜 12-30 |
| 2026（〜07-09） | 559,829 | 559,829 | 01-05 〜 07-09 |

processed/ には上記11年ファイルに加え `universe.parquet`（5,376行・Universeイベントログからの
マテリアライズドビュー。company_name等の記述列はenrichment未実行のため空欄）が存在する。

## 2. manifest.json サマリー（実行履歴・追記専用）

2件のランレコードが記録済み（Full Download完了直後・universe.parquet追加後の2回のcatalog生成時点）。
最新レコード:

| フィールド | 値 |
|---|---|
| download_started / finished | 2026-07-10T09:16:54+0900（catalog再生成のみ・実ダウンロードなし） |
| first_date / last_date | 2016-07-11 / 2026-07-09 |
| symbol_count | 5,376 |
| record_count | 10,084,970 |
| generator_version | jquants_sync/1.0.0 |
| git_commit | `01fa5a798e8747bcb2cfee1640467d18c46a71cf` |
| dataset_hash | `c736b5027a52bc09d2901dfc76c6a34d0437c7ff1edb1eac39ab7fc506e4015d`（catalog.jsonと一致） |

## 3. ディスク使用量（実測）

| ディレクトリ | サイズ | ファイル数 |
|---|---|---|
| raw/ | 273.0 MB | 11 |
| processed/ | 194.0 MB | 12 |
| cache/（日次ステージング・Strategy Cチェックポイント） | 558.8 MB | 2,440 |
| metadata/ | 0.6 MB | 5 |
| **合計** | **1,026.4 MB** | **2,468** |

## 4. バックアップ推奨事項

- `data/jquants/` は `.gitignore` により**git管理対象外**（`data_gitignore=true`・CLAUDE.md規則）。
  Study75A/B実装中に万一データセットが破損・誤削除された場合、**現状は再取得（Full Download再実行）
  以外の復旧手段がない**。
- **推奨**: Study75A/B着手前に `data/jquants/raw/` と `data/jquants/metadata/`
  （合計約274MB・processedは`compaction`で`raw`から再生成可能なため必須ではない）を
  git管理外の別ロケーション（外部ドライブ・クラウドストレージ等）へ1回コピーしておくこと。
  `cache/`（558.8MB・日次ステージング）は`raw`さえあれば再コンパクション不要で失っても実害小さいが、
  Universe再構築（`rebuild_universe_events_from_staged_bars`）が`cache/daily/`のみを参照するため、
  **`cache/daily/` も保全対象に含めることを推奨**（`raw`からの逆算では復元できないため）。
- dataset_hash（`c736b5027a52bc09...`）を記録済み。将来の整合性確認は
  `python -m src.scripts.jquants_sync --catalog` で再計算し、この値と一致するかで検証できる。
- 本タスク（Study75A/B）はデータセットを**読み取り専用**で使用する設計とする
  （Universe Generator・Bias Measurementとも新規ファイルを`backtests/`配下に生成するのみで、
  `data/jquants/`配下への書き込みは行わない）。

## 5. 確認事項

本書作成のために実行したコマンドはすべて read-only（`catalog.json`/`manifest.json`の読込・
`data/jquants/`のディスク使用量計測）であり、データセットへの変更は一切発生していない。
