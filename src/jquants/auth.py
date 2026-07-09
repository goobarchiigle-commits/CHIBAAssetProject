"""
src/jquants/auth.py
[DEPRECATED 2026-07-09] J-Quants API v2 は x-api-key 静的キー認証（src/jquants/config.py・client.py参照）。
本モジュールは旧v1想定（mailaddress/password → refreshToken → idToken）で、現行APIでは使用されない。
どこからもimportされていない（client.pyから参照除去済み）。履歴保持のため削除せず残置。

--- 以下は旧v1実装（未使用） ---

J-Quants 認証: リフレッシュトークン / IDトークン取得・TTL管理・自動更新。

認証経路は2通り:
  (a) config.refresh_token が設定済み（推奨・J-Quantsダッシュボードで発行）
      -> /token/auth_user 呼び出しをスキップし、そのrefreshTokenをそのまま使う。
  (b) mailaddress/password のみ設定済み
      -> POST {base_url}/token/auth_user  body={"mailaddress":..., "password":...}
             -> {"refreshToken": "..."}

共通:
  POST {base_url}/token/auth_refresh?refreshtoken=<token>
       -> {"idToken": "..."}

TTL（公式仕様の一般的な目安。契約後に実測値で要確認）:
  refreshToken: 約7日  -> マージンを取り 6日 でキャッシュ失効とみなす
  idToken     : 約24時間 -> マージンを取り 23時間 でキャッシュ失効とみなす

kabusapi/client.py の _TokenCache パターン（process内 singleton・TTL 再利用）を踏襲。
"""
from __future__ import annotations

import logging
import time

import requests

from src.jquants.config import JQuantsConfig
from src.jquants.exceptions import JQuantsAuthError

logger = logging.getLogger(__name__)

REFRESH_TOKEN_TTL_SEC = 6 * 24 * 3600   # 6日（公式7日の1日マージン）
ID_TOKEN_TTL_SEC       = 23 * 3600       # 23時間（公式24時間の1時間マージン）
AUTH_RETRY_MAX = 3
AUTH_RETRY_BASE_SEC = 1.0


def _post_with_retry(url: str, timeout: float, **kwargs) -> dict:
    """認証エンドポイント専用の簡易リトライ（指数バックオフ）。"""
    last_exc: Exception | None = None
    for attempt in range(AUTH_RETRY_MAX):
        try:
            resp = requests.post(url, timeout=timeout, **kwargs)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < AUTH_RETRY_MAX - 1:
                wait = AUTH_RETRY_BASE_SEC * (2 ** attempt)
                logger.warning(
                    "[JQUANTS_AUTH] status=%d attempt=%d/%d retry_in=%.1fs",
                    resp.status_code, attempt + 1, AUTH_RETRY_MAX, wait,
                )
                time.sleep(wait)
                continue
            raise JQuantsAuthError(
                f"認証失敗: status={resp.status_code} body={resp.text[:300]}"
            )
        except requests.RequestException as e:
            last_exc = e
            if attempt < AUTH_RETRY_MAX - 1:
                wait = AUTH_RETRY_BASE_SEC * (2 ** attempt)
                logger.warning("[JQUANTS_AUTH] network_error=%s retry_in=%.1fs", e, wait)
                time.sleep(wait)
                continue
    raise JQuantsAuthError(f"認証リクエストが{AUTH_RETRY_MAX}回失敗しました") from last_exc


class TokenManager:
    """
    process内 singleton 前提で使う想定（呼び出し側で使い回すこと）。
    refreshToken / idToken を TTL 内は再取得しない。
    """

    def __init__(self, config: JQuantsConfig):
        self._config = config
        self._refresh_token: str = ""
        self._refresh_fetched_at: float = 0.0
        self._id_token: str = ""
        self._id_fetched_at: float = 0.0

    # ------------------------------------------------------------------ #
    def _refresh_token_valid(self) -> bool:
        return bool(self._refresh_token) and (
            time.time() - self._refresh_fetched_at
        ) < REFRESH_TOKEN_TTL_SEC

    def _id_token_valid(self) -> bool:
        return bool(self._id_token) and (
            time.time() - self._id_fetched_at
        ) < ID_TOKEN_TTL_SEC

    # ------------------------------------------------------------------ #
    def fetch_refresh_token(self, force: bool = False) -> str:
        """refreshToken を取得（TTL内はキャッシュ再利用）。

        config.refresh_token が注入済みならそれを直接使い、/token/auth_user は呼ばない
        （password保存を避けられる推奨経路）。
        """
        self._config.validate()
        if not force and self._refresh_token_valid():
            return self._refresh_token

        if self._config.refresh_token:
            self._refresh_token = self._config.refresh_token
            self._refresh_fetched_at = time.time()
            logger.info("[JQUANTS_AUTH] refreshToken を設定値から使用（auth_user呼び出しなし）")
            return self._refresh_token

        data = _post_with_retry(
            f"{self._config.base_url}/token/auth_user",
            timeout=self._config.timeout_sec,
            json={
                "mailaddress": self._config.mail_address,
                "password":    self._config.api_password,
            },
        )
        token = data.get("refreshToken", "")
        if not token:
            raise JQuantsAuthError(f"refreshToken が応答に含まれていません: {data}")
        self._refresh_token = token
        self._refresh_fetched_at = time.time()
        logger.info("[JQUANTS_AUTH] refreshToken 取得成功")
        return token

    def fetch_id_token(self, force: bool = False) -> str:
        """refreshToken から idToken を取得（TTL内はキャッシュ再利用・失効時は自動更新）。"""
        if not force and self._id_token_valid():
            return self._id_token

        refresh_token = self.fetch_refresh_token()
        data = _post_with_retry(
            f"{self._config.base_url}/token/auth_refresh",
            timeout=self._config.timeout_sec,
            params={"refreshtoken": refresh_token},
        )
        token = data.get("idToken", "")
        if not token:
            raise JQuantsAuthError(f"idToken が応答に含まれていません: {data}")
        self._id_token = token
        self._id_fetched_at = time.time()
        logger.info("[JQUANTS_AUTH] idToken 取得成功")
        return token

    def invalidate_id_token(self) -> None:
        self._id_token = ""
        self._id_fetched_at = 0.0
        logger.debug("[JQUANTS_AUTH] idToken キャッシュ破棄")

    def invalidate_all(self) -> None:
        self.invalidate_id_token()
        self._refresh_token = ""
        self._refresh_fetched_at = 0.0
        logger.debug("[JQUANTS_AUTH] 全トークンキャッシュ破棄")
