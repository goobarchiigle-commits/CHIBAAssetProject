"""
src/jquants/client.py
J-Quants 低レベル HTTP クライアント（API v2・x-api-key 静的キー認証）。

責務: HTTP 通信のみ（取得したデータの解釈・保存は provider.py / downloader.py の責務）。

実装済み耐障害性:
  - 認証     : x-api-key ヘッダー（静的キー・トークンリフレッシュ不要）
  - Rate Limit: リクエスト間隔を config.rate_limit_sec 以上空ける
  - 401/403   : 認証エラーとして即座に送出（静的キーのため再試行しても解決しない）
  - 429       : Retry-After ヘッダ優先、なければ指数バックオフ
  - 5xx       : 指数バックオフ（最大 config.retry_max 回）
  - pagination_key: get_paginated() で自動追走

Open Question: 1ページ最大件数・pagination_keyの正確な仕様・実効Rate Limit・429時の
Retry-After有無は契約後smoke testで実測確認すること（docs/implementation/jquants_execution_infrastructure.md §11）。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterator

import requests

from src.jquants.config import JQuantsConfig, load_config
from src.jquants.exceptions import JQuantsAPIError, JQuantsAuthError, JQuantsRateLimitError

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
BACKOFF_BASE_SEC = 1.0
BACKOFF_MAX_SEC  = 60.0


class JQuantsClient:
    """
    使い方:
        client = JQuantsClient()
        data = client.get("/v2/equities/bars/daily", {"code": "86970", "date": "20240104"})
        for page in client.get_paginated("/v2/equities/bars/daily", {"code": "86970"}, list_key="data"):
            ...
    """

    def __init__(self, config: JQuantsConfig | None = None) -> None:
        self.config = config or load_config()
        self._session = requests.Session()
        self._last_request_ts = 0.0

    # ------------------------------------------------------------------ #
    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_ts
        wait = self.config.rate_limit_sec - elapsed
        if wait > 0:
            time.sleep(wait)

    def _backoff_delay(self, attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        return min(BACKOFF_MAX_SEC, BACKOFF_BASE_SEC * (2 ** attempt))

    # ------------------------------------------------------------------ #
    def get(self, path: str, params: dict[str, Any] | None = None) -> dict:
        """
        単発 GET リクエスト。429/5xxは指数バックオフでリトライ。401/403は静的キーの認証エラーとして即送出。
        """
        self.config.validate()
        url = f"{self.config.base_url}{path}"

        for attempt in range(self.config.retry_max):
            self._throttle()
            self._last_request_ts = time.time()

            try:
                resp = self._session.get(
                    url,
                    headers={"x-api-key": self.config.api_key},
                    params=params or {},
                    timeout=self.config.timeout_sec,
                )
            except requests.RequestException as e:
                if attempt < self.config.retry_max - 1:
                    wait = self._backoff_delay(attempt, None)
                    logger.warning("[JQUANTS_CLIENT] network_error=%s retry_in=%.1fs path=%s", e, wait, path)
                    time.sleep(wait)
                    continue
                raise JQuantsAPIError(f"通信失敗（リトライ上限到達）: {e}") from e

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (401, 403):
                raise JQuantsAuthError(
                    f"認証エラー: status={resp.status_code} path={path} body={resp.text[:300]}\n"
                    "  JQUANTS_API_KEY が正しいか、ダッシュボードで有効か確認してください。"
                )

            if resp.status_code in RETRYABLE_STATUS and attempt < self.config.retry_max - 1:
                wait = self._backoff_delay(attempt, resp.headers.get("Retry-After"))
                logger.warning(
                    "[JQUANTS_CLIENT] status=%d attempt=%d/%d retry_in=%.1fs path=%s",
                    resp.status_code, attempt + 1, self.config.retry_max, wait, path,
                )
                time.sleep(wait)
                continue

            if resp.status_code == 429:
                raise JQuantsRateLimitError(
                    f"レート制限（リトライ上限到達）: path={path}",
                    status_code=resp.status_code, body=resp.text[:300],
                )
            raise JQuantsAPIError(
                f"APIエラー: status={resp.status_code} path={path} body={resp.text[:300]}",
                status_code=resp.status_code, body=resp.text[:300],
            )

        raise JQuantsAPIError(f"リトライ上限到達・応答なし: path={path}")

    # ------------------------------------------------------------------ #
    def get_paginated(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        list_key: str = "",
    ) -> Iterator[dict]:
        """
        pagination_key を自動追走しながら全ページの要素を1件ずつ yield する。
        list_key を省略した場合、"pagination_key" 以外の最初のリスト値キーを使う。
        """
        query = dict(params or {})
        while True:
            data = self.get(path, query)
            key = list_key or next(
                (k for k, v in data.items() if k != "pagination_key" and isinstance(v, list)),
                "",
            )
            for item in data.get(key, []):
                yield item

            next_key = data.get("pagination_key")
            if not next_key:
                break
            query["pagination_key"] = next_key
