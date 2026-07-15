"""
src/database/sources/ — データソース抽象化層。

database/market/ 本体（ohlcv.py・master.py・sync.py・migrate.py）はこの層のみに依存し、
個々のデータソース（J-Quants API・JPX公式CSV・将来のEDINET/ETF/マクロ等）の実装詳細
（エンドポイントパス・認証方式・レート制限・HTMLパース等）を知らない。

新しいデータソースを追加する場合は SourceAdapter プロトコル（base.py）に沿った
モジュールをこのディレクトリに追加するだけでよい。
"""
from __future__ import annotations
