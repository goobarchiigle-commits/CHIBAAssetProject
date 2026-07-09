"""
src/jquants/exceptions.py
J-Quants インフラ専用例外階層。

方針:
  - 認証/通信の致命的失敗 → 例外を送出（呼び出し元で fail-closed）
  - データ品質異常（欠損/重複/NULL等）→ 例外化せず validator.py がレポートに記録
"""
from __future__ import annotations


class JQuantsError(Exception):
    """J-Quants 関連エラーの基底クラス。"""


class JQuantsConfigError(JQuantsError):
    """認証情報・設定が不足/不正な場合。"""


class JQuantsAuthError(JQuantsError):
    """認証（refreshToken / idToken 取得）に失敗した場合。"""


class JQuantsAPIError(JQuantsError):
    """API がエラー応答（4xx/5xx、リトライ上限到達後）を返した場合。"""

    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class JQuantsRateLimitError(JQuantsAPIError):
    """429 応答がリトライ上限を超えて継続した場合。"""
