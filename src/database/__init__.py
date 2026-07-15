"""
src/database/ — 日本株分析データベース（database/market/・Single Source of Truth）

責務分離:
  schema.py       : ohlcv/master各テーブルの列名・dtype定義・validate_schema()
  dtypes.py        : メモリ効率dtype変換（optimize_dtypes）
  sources/          : データソース抽象化層（J-Quants・JPX公式CSV等、プロバイダ非依存）
  ohlcv.py            : OHLCV年次Parquet管理（split_by_year/save/load/update_current_year）
  master.py             : companies/classifications/universe/indices マスタ構築
  metadata.py             : dataset_info.json/schema.json/update_history.parquet
  cache.py                  : 分析専用キャッシュ（database/market/cache/ のみに書く）
  repository.py               : 消費者向け唯一の公開データアクセスAPI（MarketDataRepository）
  migrate.py                    : data/jquants/processed からの一回限りの移行（読み取り専用）
  sync.py                         : 日次更新エントリポイント（database/market のみ更新・data/jquants不可侵）

設計原則:
  - data/（バックテスト生成物）・cache/（売買システム専用）・data/jquants/（取り込みLegacy、
    移行後は読み取り専用）とは完全独立。database/market/ を唯一の分析データベースとする。
  - J-Quants固有にしない: sources/ 層を介してのみデータソースにアクセスし、新規プロバイダは
    sources/ に実装を追加するだけで良い設計にする。
  - 消費側（バックテスト・セクター/RS/ファクター分析・ML）は repository.py のみを import し、
    database/market/ 配下のファイルパスを直接知らない。
"""
from __future__ import annotations
