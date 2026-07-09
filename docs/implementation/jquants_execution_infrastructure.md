# J-Quants Execution Infrastructure（Study75 前提インフラ）

作成日: 2026-07-04 / 改訂: 2026-07-09（全上場銘柄・年パーティション化・イベントソーシングUniverse対応）
関連: Study75（Survivorship-free Universe）/ Study76 Runner（`src/backtest/study76_clenow_benchmark_wf.py`）

## 1. 目的

Study75 開始に必要な**全上場銘柄（上場廃止銘柄含む）・2016年〜現在**の価格データ基盤を構築する。
認証情報を `.env` に設定するだけで、追加のコード変更なしに初回Full Download → 以後は差分更新のみで
運用できる状態を作る。

## 2. アーキテクチャ

```
src/jquants/
  config.py         .env から認証情報・接続設定を読込（JQUANTS_API_KEY・API v2の静的キー認証）
  exceptions.py     例外階層（Config / Auth / API / RateLimit）
  auth.py           [DEPRECATED] 旧v1想定（mailaddress/password→refreshToken→idToken）・現行API未使用
  client.py         JQuantsClient: HTTP GET + x-api-keyヘッダー + Retry(指数バックオフ) + Rate Limit
                     + pagination_key追走。401/403は静的キー認証エラーとして即送出（再試行しない）
  provider.py       JQuantsProvider: /listed/info, /prices/daily_quotes, /indices/topix の意味的ラッパー。
                     返り値は常に「APIレスポンスをできる限りそのまま」保つ（列リネームはしない）。
  universe.py       全上場銘柄ユニバースのイベントソーシング復元（ADD/REMOVE・営業日粒度）
  cache.py          Incremental Update差分判定 + ステージング（銘柄別 cache/staging/{symbol}.parquet）
  normalize.py      raw → processed の固定スキーマ正規化（生値・調整値を両方列として保持）
  compaction.py     ステージング → raw/processed 年パーティション（daily_bars_{year}.parquet）再構築
  study75_adapter.py  Study76が要求する銘柄別 processed/{symbol}.parquet への変換（Study76側は無改修）
  catalog.py        metadata/catalog.json（raw/processed の現在状態を1ファイルに集約・毎回上書き）
  manifest.py       metadata/manifest.json（同期実行ごとのランレコード・追記専用・再現性確認用）
  validator.py      重複/日付順/欠損/NULL/調整係数異常/OHLC整合性/出来高異常を検証（例外化せずレポート化）
  integrity.py      取得件数・対象期間・取得日数（営業日参照値）・欠損日数・対象銘柄数の統合レポート生成
  downloader.py     上記を束ねるオーケストレーション（sync_full_market）

src/scripts/jquants_sync.py   CLI エントリーポイント
```

責務分離: 取得（provider）・保存（cache）・分割再構築（compaction）・Universe（universe）を混在させない。

## 3. raw / processed の責務分離（2026-07-09改訂）

- **raw**（`data/jquants/raw/daily_bars_{year}.parquet`）: APIレスポンスを可能な限りそのまま保持する
  （列リネームなし。Open/High/Low/Close/Volume と AdjustmentOpen/High/Low/Close/Volume が両方入る）。
- **processed**（`data/jquants/processed/daily_bars_{year}.parquet`）: Study75以降が参照する研究用
  正規化データセット。列は固定: `Date, Code, Open, High, Low, Close, Volume, AdjustmentFactor,
  AdjustmentOpen, AdjustmentHigh, AdjustmentLow, AdjustmentClose, AdjustmentVolume`。
  **Study75以降は processed のみを参照する**（raw への直接依存を作らない）。
- **Study76互換ファイル**（`processed/{symbol}.parquet`・`processed/topix.parquet`）: 上記とは別物。
  `study75_adapter.py` が正規化データセットから調整済み値だけを選んでリネーム・DatetimeIndex化した、
  既存の `study76_clenow_benchmark_wf.py` 専用の狭いスキーマ（Study76側コードは変更しない）。
  processed/ 配下に「フルスキーマ年パーティション」と「銘柄別互換ファイル」が共存する。

## 4. ディレクトリ構成（`data/` 配下・`.gitignore` により自動的にコミット対象外）

```
data/jquants/
  raw/
    daily_bars_{year}.parquet   全銘柄・年単位・APIレスポンスそのまま
    topix.parquet                TOPIX日足生データ
    _audit/{symbol}_{from}_{to}.json  生レスポンス監査保存（fail-open）
  cache/
    staging/{symbol}.parquet    銘柄別ステージング（Checkpoint単位・compactionの入力）
  processed/
    daily_bars_{year}.parquet   正規化済み研究用データセット（Study75以降はここだけ参照）
    universe.parquet            銘柄参照テーブル（universe_events.parquetから導出・再生成可能）
    topix.parquet                Study76互換TOPIX（DatetimeIndex + Open/High/Low/Close）
    {symbol}.parquet             Study76互換・銘柄別（study75_adapter.py が生成）
  metadata/
    universe_events.parquet      Universe ADD/REMOVEイベントログ（正本・追記専用）
    universe_reconstruction_state.json  Universe復元のチェックポイント
    incremental_state.json       銘柄別「取得済み最終日」
    catalog.json                 raw/processed の現在状態一覧（毎回上書き）
    manifest.json                 同期実行ごとのランレコード（追記専用・再現性確認用）
    integrity_*.json / validation_*.json  実行ごとの整合性・検証レポート
  logs/
    jquants_sync.log
```

## 5. 認証（API v2・2026-07-09改訂）

- J-Quants API v2 は **`x-api-key` ヘッダーの静的APIキー**認証（トークンリフレッシュ不要）。
  ダッシュボード [設定 » APIキー] で発行し `.env` に `JQUANTS_API_KEY` として設定する。
- 旧実装（`src/jquants/auth.py`）は v1 想定（mailaddress/password → refreshToken → idToken）で
  現行APIでは使用されない（DEPRECATED・削除せず残置のみ）。
- 401/403は「キーが無効」の即時エラーとして扱う（静的キーのため再試行しても解決しない）。
- **1ページ最大件数・pagination_keyの正確な仕様・実効Rate Limit・429時のRetry挙動は
  smoke testで実測確認する**（§11 Open Questions参照）。

## 6. Universeイベントソーシング（上場廃止銘柄を含む全銘柄復元）

営業日粒度で上場銘柄一覧APIのスナップショットを比較し、ADD（新規出現）/REMOVE（消失）イベントを
`metadata/universe_events.parquet` に追記する。任意日時点のUniverseはイベントのリプレイで求める
（`universe.reconstruct_universe_asof(date)`）ため、別途「現在状態」を二重管理しない。

- 初回フル復元（`--rebuild-universe`）: 対象期間の全営業日を1日ずつ処理（2016-01-01〜現在で約2,600件）。
  `metadata/universe_reconstruction_state.json` にチェックポイントを保持し中断・再開が安全。
- 以後の更新: 最新スナップショット1回のみを叩く差分更新（`universe.sync_latest_snapshot_diff()`）。
- `processed/universe.parquet`: イベントログから導出するマテリアライズドビュー（再生成可能なキャッシュ）。

## 7. Retry / Backoff / Rate Limit

- 429 / 500 / 502 / 503 / 504: 指数バックオフ（`Retry-After` ヘッダ優先、なければ 1s→2s→4s→...最大60s）
- リトライ上限: `JQUANTS_RETRY_MAX`（デフォルト5回） / リクエスト間隔: `JQUANTS_RATE_LIMIT_SEC`（デフォルト0.2秒）

## 8. データ検証・整合性・Catalog・Manifest

| 項目 | 内容 |
|---|---|
| validator.py | 重複/日付順/欠損/NULL/調整係数異常/OHLC整合性/出来高異常（例外化せずレポート化） |
| integrity.py | 取得件数・対象期間・取得日数（営業日参照値）・欠損日数・対象銘柄数 → `metadata/integrity_*.json` |
| catalog.py | raw/processed 配下の各parquetの行数・対象期間・銘柄数・最終更新日時 → `metadata/catalog.json`（単一ファイル・毎回上書き） |
| manifest.py | 同期実行ごとのランレコード（download_started/finished, first_date, last_date, symbol_count, record_count, generator_version, git_commit, dataset_hash）→ `metadata/manifest.json`（追記専用） |

## 9. PowerShell 手順書（ゼロから実行）

### 9.1 事前準備

```powershell
cd C:\ai-trading
python -c "import requests, pandas, pyarrow; print('deps OK')"
```

### 9.2 .env 設定

```powershell
notepad src\.env
```

以下1行を追加（J-Quantsダッシュボード [設定 » APIキー] で発行した静的APIキー）:

```
JQUANTS_API_KEY=your_jquants_api_key_here
```

保存後、設定確認（実通信なし）:

```powershell
python -m src.scripts.jquants_sync --check-only
```

`JQUANTS_API_KEY 設定済み: True` になっていることを確認する。

### 9.3 Universeイベントログのフル復元（初回のみ・重い処理）

```powershell
python -m src.scripts.jquants_sync --rebuild-universe --start 2016-01-01 --end (Get-Date -Format "yyyy-MM-dd")
```

中断した場合は同じコマンドを再実行すればチェックポイントから再開する。

### 9.4 初回Full Download（全上場銘柄）

```powershell
python -m src.scripts.jquants_sync --full-market --mode full --start 2016-01-01 --end (Get-Date -Format "yyyy-MM-dd")
```

完了後、以下が生成される:

```
data\jquants\raw\daily_bars_{year}.parquet         … 年単位・全銘柄・生データ
data\jquants\raw\topix.parquet                      … TOPIX生データ
data\jquants\processed\daily_bars_{year}.parquet    … 年単位・全銘柄・正規化済み
data\jquants\processed\universe.parquet             … 銘柄参照テーブル
data\jquants\processed\topix.parquet                … Study76互換TOPIX
data\jquants\metadata\catalog.json                  … データセット現在状態
data\jquants\metadata\manifest.json                 … 実行履歴（再現性確認用）
data\jquants\metadata\integrity_*.json / validation_*.json
```

### 9.5 更新同期（差分のみ）

```powershell
python -m src.scripts.jquants_sync --full-market --end (Get-Date -Format "yyyy-MM-dd")
```

### 9.6 Study76互換ファイルの生成

Study76が使うユニバース（例: 暫定的に RSR42）が決まったら:

```powershell
python -m src.scripts.jquants_sync --materialize src\configs\universe\rsr42_trading.json
```

### 9.7 ログ確認

```powershell
Get-Content data\jquants\logs\jquants_sync.log -Tail 50
Get-Content data\jquants\logs\jquants_sync.log -Tail 200 | Select-String "ERROR|WARNING"
Get-Content data\jquants\metadata\catalog.json | ConvertFrom-Json
```

## 10. Study75/76 との接続点

Study76（`src/backtest/study76_clenow_benchmark_wf.py`）は正典
（`reports/study76_execution_plan.md` 等）により実装済み・変更しない。要求データ:

1. **規則ユニバース**: `src/configs/universe/study75_survivorship_free.json`（Study75本体が生成・本インフラの対象外）。
2. **価格パネル**: `data/jquants/processed/{symbol}.parquet`（`--materialize` で生成）。
3. **TOPIX**: `data/jquants/processed/topix.parquet`（Windows NTFSは大小文字区別なしのため `TOPIX.parquet` とも一致）。

## 11. Open Questions（smoke testで実測確認が必要な事項・2026-07-09時点）

1. **上場銘柄一覧・TOPIXの正確なパス**: 確定しているのは `/v2/equities/bars/daily`（公式クイックスタート実例）のみ。
   `LISTED_INFO_PATH_CANDIDATE = "/v2/equities/info"` / `TOPIX_PATH_CANDIDATE = "/v2/indices/bars/daily"`
   （`provider.py`）は未確認の暫定パス。
2. 1ページ最大件数・`pagination_key` の正確な仕様。
3. 実効Rate Limit（プラン別の公式上限値）。現状は保守的なデフォルト（0.2秒間隔）。
4. 429発生時の `Retry-After` ヘッダー有無・実際のバックオフ挙動。
5. `/v2/equities/bars/daily` のコード桁数（4桁 vs 5桁）。`study75_adapter.py` は5桁への簡易フォールバックのみ実装済み。
6. Survivorship-free Universe 復元が実際に必要な期間・銘柄をカバーしているか（`--rebuild-universe` 実行後に検証）。
