"""
src/jquants/ — J-Quants データ取得基盤（Study75 前提インフラ）

責務分離:
  config.py         : .env からの認証情報（JQUANTS_API_KEY）・設定読込
  exceptions.py      : 例外階層
  auth.py            : [DEPRECATED] 旧v1トークンリフレッシュ実装・現行API v2では未使用
  client.py          : 低レベル HTTP クライアント（x-api-key認証・Retry/Backoff/Rate Limit）
  provider.py        : J-Quants エンドポイントのラッパー（listed_info / daily_quotes / topix）
  schema.py           : raw(v2略記フィールド) ⇔ 標準名マッピング（normalize.py・validator.py共用）
  universe.py          : Universeイベントソーシング復元（ADD/REMOVE・上場廃止銘柄含む）
  cache.py              : ステージング（銘柄別）・Incremental Update 差分判定
  normalize.py           : raw → processed 正規化（固定スキーマ）
  compaction.py            : ステージング → raw/processed 年パーティション再構築
                             （銘柄別 compact_years / 日次 compact_years_from_daily 両対応）
  study75_adapter.py        : Study76互換の銘柄別processedファイル生成
  catalog.py                  : metadata/catalog.json（データセット現在状態）
  manifest.py                   : metadata/manifest.json（実行ごとの再現性レコード）
  validator.py                    : データ検証（例外化せずレポートへ記録）
  integrity.py                      : 取得件数・欠損日数等の整合性レポート生成
  downloader.py                       : 銘柄別エンジン（sync_full_market・後方互換のため維持）
  study75_downloader.py                 : Strategy C（日次イテレーション）専用ダウンロードエンジン
                                           （cache/daily/day_YYYY-MM-DD.parquet・完了済み営業日
                                           マニフェスト・再開対応）

認証情報未設定でも import 可能（NameError/ImportError を出さない）。
実際の認証・通信は JQuantsClient.get() を呼んだ時点で初めて発生する（x-api-keyヘッダー・v2 API）。
"""
from __future__ import annotations
