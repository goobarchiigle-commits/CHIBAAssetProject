# database/market — 日本株分析データベース（Single Source of Truth）

J-Quants APIから取得した日本株の日足/分足/Tick/財務/信用/空売り/指数/ETFデータを保存する。
本ディレクトリ配下のみが分析・バックテスト・機械学習の参照対象（`data/`・`cache/`・
`data/jquants/`（Legacy）とは完全独立）。

## 取得日・取得範囲

| 項目 | 内容 |
|---|---|
| 主要バックフィル実施日 | 2026-07-23 |
| **J-Quants契約解約日** | **2026-08-09**（本データベースは解約前提で運用終了に向けて固定化） |
| 最終同期日 | **2026-08-08**（`src/database/sync_bulk.py`実行・2026-08-07分まで反映。解約前最終差分取得） |
| 日足OHLCV/財務summary/信用/空売り | 2016-07-01 〜 2026-08-07（継続更新終了・以後更新不可） |
| 財務summary追加項目 | 2026-08-08、`ShEq`/`NCShEq`/`ROE`/`NCROE`が実データに新規出現していることを確認し、
  2016-07〜2026-08の全期間を`force=True`で再取得・遡及統合済み（列追加のみ・既存列は無変更）。 |
| 分足・Tick | 2024-07-23 〜 2026-08-07（J-Quants Add-on契約の提供範囲=過去2年分のみ。これより古いデータは存在しない） |
| 指数（TOPIX/TOPIX-17業種等） | 2016-07-25〜2016-08-01（系列により異なる・Standardプラン10年ローリング窓の実測下限）〜2026-08-07。
  2026-08-08よりISSUE-001対策済みの安全差分マージ版（`--safe`）で更新・全26系列でshrink検知ゼロ。 |
| 決算発表予定日（`/equities/earnings-calendar`） | 2026-08-08新規実装。前方参照専用API（翌営業日分のみ）のため、解約前に取得できるのは
  2026-08-08時点の1スナップショット（186銘柄）のみ。`fundamentals/earnings_calendar/2026-08-08.parquet`。 |
| 最終監査 | 2026-08-08（`docs/research/jquants_final_sync_2026-08-08.md`・verify_database.py --full-archive-check含め全PASS・全データセットで行数減少ゼロを確認） |

既知の問題は`database/market/KNOWN_ISSUES.md`を参照（ISSUE-001は2026-08-08修正済み）。

進行中の当月（例: 2026-07）は日次ファイルを束ねた`status=partial`として保存され、月が
historicalへロールアップされ次第、次回`sync_bulk.py`実行で自動的に`status=ok`へ昇格する
（`database/market/metadata/bulk_ingest_state.parquet`で追跡）。

## 保存形式

- **Parquet（派生データ）**: `pyarrow`エンジン。大容量ドメイン（`minute/`・`tick/`）は**ZSTD**圧縮、
  それ以外は**Snappy**圧縮（`database_version.json`の`compression`列で常に最新値を確認可能）。
  読み込み側はコーデックをファイルフッタから自動判定するため、将来圧縮方式を変更しても
  既存コードの変更は不要（`pd.read_parquet(engine="pyarrow")`のみ使用・`fastparquet`や
  `pyarrow.dataset`は不使用であることを確認済み）。
- **CSV.GZ（原本・一次データ）**: `archive/bulk/`配下に永久保存（`database/market/`とは別の
  トップレベルディレクトリ）。**J-Quants解約後もこれが唯一の一次データになる**。Parquetは
  ここから再生成可能な派生データという位置付け。

## 更新方法

```bash
# 日次インクリメンタル更新（OHLCV + 分足/Tick/summary/margin/shortselling 全て・冪等）
python -m src.database.sync_bulk

# 整合性監査（ファイル実在・アーカイブSHA256照合・スキーマ適合・重複検出）
python -m src.database.verify_database

# スナップショット生成（全ファイルSHA256付き状態記録）
python -m src.database.snapshot

# 個別バックフィル（例: 特定月のみ再取得）
python -m src.database.minute_bars --month 202607 --force
python -m src.database.ticks --month 202607 --force
python -m src.database.markets_bulk --domain margin_interest --start-year 2016 --start-month 7
```

J-Quants解約後は`sync_bulk.py`実行はエラーになる（認証不可）。解約前が最後の更新チャンス。

## ディレクトリ構成

| ディレクトリ | 内容 | 実装状況 |
|---|---|---|
| `ohlcv/{year}.parquet` | 日足OHLCV（全銘柄） | 実装済み |
| `minute/{year}/{yyyymm}.parquet` | 分足OHLCV（全銘柄・月次） | 実装済み（過去2年分） |
| `tick/{year}/{yyyymm}.parquet` | Tick（全銘柄・月次） | 実装済み（過去2年分） |
| `master/companies.parquet` 等 | 銘柄マスタ・分類・ユニバース・指数マスタ | 実装済み |
| `fundamentals/summary/{year}.parquet` | 決算短信サマリ | 実装済み |
| `fundamentals/statements,forecasts,dividend/` | 財務諸表詳細・予想・配当 | 未実装（Premium限定・ディレクトリのみ） |
| `margin/margin_interest,margin_alert/{year}.parquet` | 信用取引週末残高・日々公表信用残高 | 実装済み |
| `shortselling/short_ratio,short_sale_report/{year}.parquet` | 業種別空売り比率・空売り残高報告 | 実装済み |
| `index/prices/{code}.parquet` + `master/indices.parquet` | TOPIX・TOPIX-17業種・REIT・Growth250・市場区分別等26指数 | 実装済み |
| `etf/master.parquet` `etf/prices/{year}.parquet` | ETF銘柄マスタ・価格（ohlcvからの派生・追加API不要） | 実装済み |
| `etf/constituents/` | ETF構成銘柄 | **J-Quants非提供のため実装不可**（運用会社PCF等が必要） |
| `factor/` | クロスセクショナルファクタースコア | 未実装（設計のみ） |
| `features/{daily,intraday,factor}/` | 銘柄別生特徴量（ML入力） | 未実装（設計のみ） |
| `macro/` | マクロ指標 | 未実装（設計のみ） |
| `metadata/` | メタデータ一式（下記） | 実装済み |
| `cache/` | （未使用） | - |

### metadata/ の中身

| ファイル | 内容 |
|---|---|
| `database_version.json` | スキーマバージョン・API version・圧縮方式・Python version・git commit |
| `dataset_info.json` | 現在状態スナップショット（毎回上書き） |
| `schema.json` | `src/database/schema.py`定義から自動生成 |
| `update_history.parquet` | 実行ごとの追記専用ログ |
| `bulk_ingest_state.parquet` | Bulk取り込みの冪等性/resume/トレーサビリティ台帳（原本CSV.GZ↔Parquetの対応・SHA256） |
| `coverage_report.json` | 各ドメインの実測カバレッジ（期間・行数・銘柄数） |
| `data_quality.json` | 営業日欠落・未完了期間の検出結果 |
| `source_catalog.json` | J-Quants全エンドポイントの棚卸し（取得済み/Premium限定/未提供） |
| `bulk_catalog/{date}.json` | Bulk API（`/v2/bulk/list`）生応答の保存（「当時何が提供されていたか」の一次証拠） |
| `snapshot_{date}.json` | 全Parquetのsha256・行数・サイズを含む完全な状態記録 |
| `verify_report_{date}.json` | `verify_database.py`実行結果 |

## API契約

- **契約プラン**: J-Quants Standard（¥3,300/月）+ 株価分足/Tick Add-on（¥5,500/月）。
- **Premium限定（未契約・取得不可）**: 財務諸表詳細(`/fins/details`)・配当(`/fins/dividend`)・
  前場四本値・先物/一部オプション四本値・売買内訳データ(`/markets/breakdown`)。
- **別Add-on（未契約）**: TDnet適時開示(`/td/list`,`/td/files`,`/td/bulk`・5年分)。
- **日経225**: J-Quantsでは提供されていない（Nikkei Inc独自ライセンスのため。実測確認済み）。
- 詳細は`metadata/source_catalog.json`（実装状況付き一覧）・`metadata/bulk_catalog/`（生カタログ）参照。

## データソース・取得方式

全データはJ-Quants Bulk API（`/v2/bulk/list` + `/v2/bulk/get`、1リクエストで全銘柄分の
CSV.GZを取得）経由。銘柄別REST逐次取得（Study82時代の429レート制限問題の原因）は使用していない。
唯一の例外は`index/prices/`（`/v2/indices/bars/daily`の通常REST、指数単位で軽量なため）。

実装: `src/jquants/bulk_client.py`（低レベルクライアント）、
`src/database/{minute_bars,ticks,fundamentals,markets_bulk,etf,indices,index_prices}.py`
（ドメイン別取り込み）、`src/database/{bulk_state,reports,snapshot,verify_database,bulk_catalog}.py`
（トレーサビリティ・監査基盤）。

## 注意事項

- **分足/Tickは直近2年分のみ**（J-Quants側の提供制限。契約継続してももっと遡れるわけではない）。
- **`archive/`は絶対に手動削除しない**（J-Quants解約後の唯一の一次データ）。`.gitignore`で
  `/archive/`除外済み・Gitにはコミットされない（`database/`も同様）。
- ETF/REITは通常銘柄として`ohlcv/`に含まれる（`companies.parquet`の`ProductCategory`
  014=国内ETF・023=海外ETF・013=REITで判別）。`etf/`はそこからの派生ビュー。
- TOPIX-17業種別指数のコードは`0080`-`0090`（16進）。**`0040`-`0056`ではない**
  （2026-07-15時点の実装は誤りだった。2026-07-23に実データ相関検証で発見・修正済み）。
- 整合性に疑問があれば`python -m src.database.verify_database --full-archive-check`で
  archive/全件のSHA256を再検証できる（既定はサンプル20件のみ・高速）。
