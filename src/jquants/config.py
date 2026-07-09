"""
src/jquants/config.py
J-Quants 認証情報・接続設定（.env から取得。ハードコード禁止）。

J-Quants API v2（2026-07時点の公式クイックスタート準拠）: 認証は x-api-key ヘッダーの静的APIキー
（ダッシュボード [設定 » APIキー] で発行）。mailaddress/password や refreshToken/idToken のトークン
リフレッシュフローは使わない（旧v1想定で実装されていた分は廃止・2026-07-09）。

必須環境変数（.env）:
  JQUANTS_API_KEY

任意環境変数（.env）:
  JQUANTS_BASE_URL          デフォルト: https://api.jquants.com
  JQUANTS_RATE_LIMIT_SEC    デフォルト: 0.05（base_sleep・1リクエストあたりの最小間隔・秒。
                             2026-07-09 Download Strategy Validationでレート制限ヘッダーが
                             一切露出しないことを確認済み・応答自体が0.4-3.6秒かかるため
                             base_sleepは補助的。429時はclient.pyが指数バックオフで別途対応する）
  JQUANTS_RETRY_MAX         デフォルト: 5
  JQUANTS_TIMEOUT_SEC       デフォルト: 30.0

import 時点では検証しない（認証情報未設定でもビルド可能にするため）。
実際に接続する直前に validate() を呼ぶこと。
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from src.paths import JQUANTS_API_KEY
from src.jquants.exceptions import JQuantsConfigError

DEFAULT_BASE_URL = "https://api.jquants.com"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class JQuantsConfig:
    api_key:        str
    base_url:       str
    rate_limit_sec: float
    retry_max:      int
    timeout_sec:    float

    def validate(self) -> None:
        """認証必須項目の存在チェック。通信直前に呼ぶ。"""
        if not self.api_key:
            raise JQuantsConfigError(
                "J-Quants 認証情報が未設定です: JQUANTS_API_KEY\n"
                "  src/.env に JQUANTS_API_KEY=<ダッシュボードで発行したAPIキー> を追加してください"
                "（src/.env.example 参照）。"
            )

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)


def load_config() -> JQuantsConfig:
    """現在の環境変数から JQuantsConfig を構築する（検証は行わない）。"""
    return JQuantsConfig(
        api_key=JQUANTS_API_KEY,
        base_url=os.environ.get("JQUANTS_BASE_URL", "").strip() or DEFAULT_BASE_URL,
        rate_limit_sec=_env_float("JQUANTS_RATE_LIMIT_SEC", 0.05),
        retry_max=_env_int("JQUANTS_RETRY_MAX", 5),
        timeout_sec=_env_float("JQUANTS_TIMEOUT_SEC", 30.0),
    )
