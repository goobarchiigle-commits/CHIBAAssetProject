"""
src/database/exceptions.py
database/market/ 分析データベース専用の例外階層。

方針:
  - スキーマ不整合・書き込み境界違反（Legacy領域への誤書き込み等）→ 例外化してfail-closed
  - データソース側の一時的失敗（HTTP等）は src.jquants.exceptions をそのまま伝播させる
    （二重の例外階層は作らない）。
"""
from __future__ import annotations


class DatabaseError(Exception):
    """database/market 関連エラーの基底クラス。"""


class SchemaValidationError(DatabaseError):
    """Parquetテーブルが schema.json / schema.py の定義と一致しない場合。"""


class WriteBoundaryError(DatabaseError):
    """database/market 以外の場所（data/jquants/等のLegacy領域）へ書き込もうとした場合。"""


class MigrationError(DatabaseError):
    """既存 data/jquants/processed からの移行処理に失敗した場合。"""
