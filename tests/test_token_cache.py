"""
tests/test_token_cache.py
_TokenCache singleton + fetch_token キャッシュ + 401 self-heal のユニットテスト
"""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

import src.kabusapi.client as client_module
from src.kabusapi.client import _CACHE, KabuClient


def _reset_cache():
    """テスト間でキャッシュ状態をリセットするヘルパ。"""
    _CACHE._token = ""
    _CACHE._fetched_at = 0.0
    _CACHE._fetch_count = 0


def _make_client(api_password: str = "testpass") -> KabuClient:
    """KABU_API_PASSWORD を設定した KabuClient を返す。"""
    with patch.dict("os.environ", {"KABU_API_PASSWORD": api_password,
                                    "KABU_TRADE_PASSWORD": "",
                                    "KABU_ACCOUNT_TYPE": "4"}):
        return KabuClient()


class TestTokenCacheCase1_FirstFetch(unittest.TestCase):
    """テスト1: 初回 fetch_token() は HTTP POST を 1 回発行し _CACHE に保存される"""

    def setUp(self):
        _reset_cache()

    def test_first_fetch_posts_once_and_stores_cache(self):
        client = _make_client()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Token": "tok_abc"}
        mock_resp.raise_for_status.return_value = None

        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            token = client.fetch_token()

        mock_post.assert_called_once()
        self.assertEqual(token, "tok_abc")
        self.assertEqual(_CACHE._token, "tok_abc")
        self.assertEqual(_CACHE._fetch_count, 1)


class TestTokenCacheCase2_Reuse(unittest.TestCase):
    """テスト2: TTL 内の 2 回目 fetch_token() は HTTP POST を発行しない"""

    def setUp(self):
        _reset_cache()

    def test_second_fetch_reuses_cache(self):
        # キャッシュを有効状態に手動設定
        _CACHE._token = "tok_cached"
        _CACHE._fetched_at = time.time()
        _CACHE._fetch_count = 1

        client = _make_client()

        with patch.object(client._session, "post") as mock_post:
            token = client.fetch_token()

        mock_post.assert_not_called()
        self.assertEqual(token, "tok_cached")
        self.assertEqual(client._token, "tok_cached")
        self.assertEqual(_CACHE._fetch_count, 1)  # 増えない


class TestTokenCacheCase3_TtlExpiry(unittest.TestCase):
    """テスト3: TTL 切れ後は再 fetch が発生する"""

    def setUp(self):
        _reset_cache()

    def test_expired_cache_refetches(self):
        # TTL を超過した状態にする（fetched_at を 3100 秒前に設定）
        _CACHE._token = "tok_old"
        _CACHE._fetched_at = time.time() - 3100
        _CACHE._fetch_count = 1

        client = _make_client()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Token": "tok_new"}
        mock_resp.raise_for_status.return_value = None

        with patch.object(client._session, "post", return_value=mock_resp) as mock_post:
            token = client.fetch_token()

        mock_post.assert_called_once()
        self.assertEqual(token, "tok_new")
        self.assertEqual(_CACHE._fetch_count, 2)


class TestTokenCacheCase4_MultipleInstances(unittest.TestCase):
    """テスト4: 複数の KabuClient インスタンスが _CACHE を共有する"""

    def setUp(self):
        _reset_cache()

    def test_two_clients_share_cache(self):
        client1 = _make_client()
        client2 = _make_client()

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"Token": "tok_shared"}
        mock_resp.raise_for_status.return_value = None

        # client1 が fetch
        with patch.object(client1._session, "post", return_value=mock_resp) as mock_post1:
            t1 = client1.fetch_token()
        mock_post1.assert_called_once()
        self.assertEqual(_CACHE._fetch_count, 1)

        # client2 はキャッシュを使う（POST しない）
        with patch.object(client2._session, "post") as mock_post2:
            t2 = client2.fetch_token()
        mock_post2.assert_not_called()

        self.assertEqual(t1, t2)
        self.assertEqual(t1, "tok_shared")
        self.assertEqual(_CACHE._fetch_count, 1)


class TestTokenCacheCase5_ResetBetweenTests(unittest.TestCase):
    """テスト5: _reset_cache() でキャッシュが完全にリセットされる"""

    def setUp(self):
        _reset_cache()

    def test_cache_is_empty_after_reset(self):
        self.assertEqual(_CACHE._token, "")
        self.assertEqual(_CACHE._fetched_at, 0.0)
        self.assertEqual(_CACHE._fetch_count, 0)
        self.assertFalse(_CACHE.is_valid())

    def test_cache_populated_then_reset(self):
        _CACHE._token = "some_token"
        _CACHE._fetched_at = time.time()
        _CACHE._fetch_count = 3
        self.assertTrue(_CACHE.is_valid())

        _reset_cache()

        self.assertFalse(_CACHE.is_valid())
        self.assertEqual(_CACHE._fetch_count, 0)


class TestTokenCacheCase6_MissingPassword(unittest.TestCase):
    """テスト6: KABU_API_PASSWORD 未設定時は ValueError が発生し _CACHE は更新されない"""

    def setUp(self):
        _reset_cache()

    def test_missing_password_raises_and_cache_unchanged(self):
        with patch.dict("os.environ", {"KABU_API_PASSWORD": "",
                                        "KABU_TRADE_PASSWORD": "",
                                        "KABU_ACCOUNT_TYPE": "4"}):
            client = KabuClient()

        with self.assertRaises(ValueError):
            client.fetch_token()

        self.assertEqual(_CACHE._fetch_count, 0)
        self.assertEqual(_CACHE._token, "")


class TestTokenCacheCase7_401SelfHeal(unittest.TestCase):
    """テスト7: API call が 401 を返した場合にキャッシュを破棄して 1 回だけリトライする"""

    def setUp(self):
        _reset_cache()

    def test_401_triggers_cache_invalidate_and_retry(self):
        client = _make_client()

        # キャッシュを有効状態に設定（古い token が残っているシナリオ）
        _CACHE._token = "tok_stale"
        _CACHE._fetched_at = time.time()
        _CACHE._fetch_count = 1
        client._token = "tok_stale"
        client._session.headers.update({"X-API-KEY": "tok_stale"})

        # 1回目 GET /positions → 401
        resp_401 = MagicMock()
        resp_401.status_code = 401
        resp_401.raise_for_status.side_effect = None  # raise_for_status は retry 後に呼ばれる

        # 2回目 GET /positions → 200
        resp_200 = MagicMock()
        resp_200.status_code = 200
        resp_200.json.return_value = [{"Symbol": "7203", "LeavesQty": 100, "Price": 3000.0}]
        resp_200.raise_for_status.return_value = None

        # POST /token（self-heal 時の再 fetch）→ 200
        resp_token = MagicMock()
        resp_token.json.return_value = {"Token": "tok_fresh"}
        resp_token.raise_for_status.return_value = None

        get_calls = [resp_401, resp_200]

        with patch.object(client._session, "get", side_effect=get_calls) as mock_get, \
             patch.object(client._session, "post", return_value=resp_token) as mock_post:
            positions = client.get_positions()

        # GET は 2 回呼ばれる（1回目: 401、2回目: リトライ）
        self.assertEqual(mock_get.call_count, 2)
        # POST /token は 1 回（self-heal による再取得）
        mock_post.assert_called_once()
        # キャッシュが新しい token で更新されている
        self.assertEqual(_CACHE._token, "tok_fresh")
        self.assertEqual(_CACHE._fetch_count, 2)
        # 戻り値は正しい
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["Symbol"], "7203")

    def test_401_does_not_infinite_retry(self):
        """401 が連続しても 2 回目の GET は raise_for_status() で例外になり無限ループしない"""
        client = _make_client()

        _CACHE._token = "tok_stale"
        _CACHE._fetched_at = time.time()
        _CACHE._fetch_count = 1
        client._token = "tok_stale"

        # GET が常に 401 を返す（リトライ後も 401）
        resp_401a = MagicMock()
        resp_401a.status_code = 401
        resp_401a.raise_for_status.return_value = None

        resp_401b = MagicMock()
        resp_401b.status_code = 200  # status_code は 200 だが raise_for_status で例外
        resp_401b.raise_for_status.side_effect = Exception("HTTP Error 401")
        resp_401b.json.return_value = []

        resp_token = MagicMock()
        resp_token.json.return_value = {"Token": "tok_fresh"}
        resp_token.raise_for_status.return_value = None

        get_calls = [resp_401a, resp_401b]

        with patch.object(client._session, "get", side_effect=get_calls) as mock_get, \
             patch.object(client._session, "post", return_value=resp_token):
            with self.assertRaises(Exception):
                client.get_positions()

        # GET は最大 2 回（リトライは 1 回のみ）
        self.assertEqual(mock_get.call_count, 2)


def test_module_identity():
    """kabusapi.client が sys.modules に存在しないこと（二重ロードなし）"""
    import sys
    assert "kabusapi.client" not in sys.modules, \
        f"二重ロード検出: kabusapi.client が sys.modules に存在する"
    assert "src.kabusapi.client" in sys.modules


def test_singleton_shared():
    """src.kabusapi.client 経由の _CACHE は同一オブジェクト"""
    from src.kabusapi.client import _CACHE as cache1
    from src.kabusapi import client as mod
    assert cache1 is mod._CACHE, "_CACHE が別オブジェクト（singleton 分裂）"


if __name__ == "__main__":
    unittest.main()
