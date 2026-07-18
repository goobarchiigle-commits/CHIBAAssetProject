"""
tests/test_send_order_sell_payload.py
現物 SELL 注文の payload が kabu API 仕様に準拠しているかを検証するユニットテスト

【背景】
  - BUY 注文: DelivType=2（お預り金）, FundType="AA"（特定預り）
  - SELL注文: DelivType=0（受渡なし）, FundType="  "（出金/空白2文字）
  - この差を無視して固定値を送ると SELL 時に 400 Bad Request になる
  - kabu API は Qty を整数で要求する。float (100.0) を送ると 500 Internal Server Error になる

【テスト範囲】
  1. BUY payload: DelivType=2, FundType="AA", Exchange=9(SOR)
  2. SELL payload: DelivType=0, FundType="  ", Exchange=1(TSE)
  3. SELL でも CashMargin=1（現物）が保持される
  4. SELL でも Side="1" が正しく送信される
  5. 成功レスポンス → OrderResult.success=True
  6. logger.debug("SEND_ORDER_PAYLOAD qty=... qty_type=...") が呼ばれること
  7. qty=100.0 (float) を渡しても payload Qty は int になること
  8. BUY でも qty float → int 変換が行われること（回帰）
"""
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import json

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

# _CACHE をリセットするためインポートしてから書き換える
import src.kabusapi.client as client_module
from src.kabusapi.client import (
    _CACHE,
    KabuClient,
    Side,
    Exchange,
    CashMargin,
    OrderType,
)


# ------------------------------------------------------------------ #
# テスト共通ヘルパ
# ------------------------------------------------------------------ #

_entry_freeze_env_patch = None


def setUpModule() -> None:
    """Entry Freeze Mode（2026-07-17〜既定enabled）は本テストの対象外なので、
    モジュール全体で明示的に無効化する（send_order()のpayload構築ロジック自体を
    検証するテストであり、freezeゲートの挙動検証はtest_send_order_entry_freeze_guard.py
    が別途担当する）。load_strategy_config()はlru_cache済みのため、他モジュールで
    既にキャッシュされていた場合に備えcache_clear()も併用する。"""
    global _entry_freeze_env_patch
    from src.config_loader import load_strategy_config
    _entry_freeze_env_patch = patch.dict("os.environ", {"ENTRY_FREEZE_ENABLED": "0"})
    _entry_freeze_env_patch.start()
    load_strategy_config.cache_clear()


def tearDownModule() -> None:
    from src.config_loader import load_strategy_config
    if _entry_freeze_env_patch is not None:
        _entry_freeze_env_patch.stop()
    load_strategy_config.cache_clear()


def _reset_cache() -> None:
    _CACHE._token = ""
    _CACHE._fetched_at = 0.0
    _CACHE._fetch_count = 0


def _make_client() -> KabuClient:
    with patch.dict("os.environ", {
        "KABU_API_PASSWORD":   "testpass",
        "KABU_TRADE_PASSWORD": "tradepass",
        "KABU_ACCOUNT_TYPE":   "4",
    }):
        c = KabuClient()
    # token をセットして fetch_token() 呼び出し不要にする
    c._token = "tok_test"
    return c


def _ok_response(order_id: str = "20260414A02N00000001") -> MagicMock:
    """sendorder の正常レスポンスを模倣する。"""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = {"OrderId": order_id}
    resp.raise_for_status.return_value = None
    return resp


def _capture_payload(client: KabuClient, resp: MagicMock) -> dict:
    """send_order を呼び出し、POST された JSON payload を返す。"""
    with patch.object(client._session, "post", return_value=resp) as mock_post:
        # Exchange=1(TSE) / SELL を明示して呼ぶ（signal_bridge が渡す形式）
        client.send_order(
            symbol     = "5401",
            exchange   = Exchange.TSE,
            side       = Side.SELL,
            qty        = 100,
            order_type = OrderType.MARKET,
            price      = 0.0,
        )
    # requests.Session.post(url, json=payload) の kwargs を取得
    _, kwargs = mock_post.call_args
    return kwargs.get("json", {})


# ------------------------------------------------------------------ #
# テストケース
# ------------------------------------------------------------------ #

class TestSellPayloadDelivType(unittest.TestCase):
    """SELL 注文時は DelivType=0 が送信されること"""

    def setUp(self):
        _reset_cache()
        self.client = _make_client()

    def test_sell_deliv_type_is_zero(self):
        payload = _capture_payload(self.client, _ok_response())
        self.assertEqual(
            payload["DelivType"], 0,
            f"SELL時 DelivType は 0（受渡なし）でなければならない。実際: {payload['DelivType']}"
        )


class TestSellPayloadFundType(unittest.TestCase):
    """SELL 注文時は FundType='  '（空白2文字）が送信されること"""

    def setUp(self):
        _reset_cache()
        self.client = _make_client()

    def test_sell_fund_type_is_two_spaces(self):
        payload = _capture_payload(self.client, _ok_response())
        self.assertEqual(
            payload["FundType"], "  ",
            f"SELL時 FundType は '  '（空白2文字）でなければならない。実際: repr={repr(payload['FundType'])}"
        )

    def test_sell_fund_type_length(self):
        payload = _capture_payload(self.client, _ok_response())
        self.assertEqual(
            len(payload["FundType"]), 2,
            "FundType は 2文字でなければならない"
        )


class TestSellPayloadSideAndCashMargin(unittest.TestCase):
    """SELL 注文時に Side='1', CashMargin=1 が保持されること"""

    def setUp(self):
        _reset_cache()
        self.client = _make_client()

    def test_sell_side_is_one(self):
        payload = _capture_payload(self.client, _ok_response())
        self.assertEqual(
            payload["Side"], "1",
            "SELL は Side='1' でなければならない"
        )

    def test_sell_cash_margin_is_cash(self):
        payload = _capture_payload(self.client, _ok_response())
        self.assertEqual(
            payload["CashMargin"], CashMargin.CASH,  # 1
            "現物SELL は CashMargin=1 でなければならない"
        )


class TestSellPayloadExchange(unittest.TestCase):
    """SELL 注文時に Exchange=1（TSE）が送信されること"""

    def setUp(self):
        _reset_cache()
        self.client = _make_client()

    def test_sell_exchange_is_tse(self):
        resp = _ok_response()
        with patch.object(self.client._session, "post", return_value=resp) as mock_post:
            self.client.send_order(
                symbol   = "5401",
                exchange = Exchange.TSE,   # signal_bridge が明示指定
                side     = Side.SELL,
                qty      = 100,
            )
        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        self.assertEqual(
            payload["Exchange"], Exchange.TSE,
            f"SELL時 Exchange は {Exchange.TSE}（TSE）でなければならない"
        )


class TestBuyPayloadUnchanged(unittest.TestCase):
    """BUY 注文時は従来どおり DelivType=2, FundType='AA' が送信されること（回帰）"""

    def setUp(self):
        _reset_cache()
        self.client = _make_client()

    def _buy_payload(self) -> dict:
        resp = _ok_response()
        with patch.object(self.client._session, "post", return_value=resp) as mock_post:
            self.client.send_order(
                symbol   = "7203",
                exchange = Exchange.SOR,
                side     = Side.BUY,
                qty      = 100,
            )
        _, kwargs = mock_post.call_args
        return kwargs.get("json", {})

    def test_buy_deliv_type_is_two(self):
        payload = self._buy_payload()
        self.assertEqual(payload["DelivType"], 2)

    def test_buy_fund_type_is_aa(self):
        payload = self._buy_payload()
        self.assertEqual(payload["FundType"], "AA")

    def test_buy_exchange_is_sor(self):
        payload = self._buy_payload()
        self.assertEqual(payload["Exchange"], Exchange.SOR)


class TestSellOrderResult(unittest.TestCase):
    """SELL の正常レスポンス → OrderResult.success=True になること"""

    def setUp(self):
        _reset_cache()
        self.client = _make_client()

    def test_sell_success_result(self):
        resp = _ok_response("20260414A02N99999999")
        with patch.object(self.client._session, "post", return_value=resp):
            result = self.client.send_order(
                symbol   = "5411",
                exchange = Exchange.TSE,
                side     = Side.SELL,
                qty      = 100,
            )
        self.assertTrue(result.success)
        self.assertEqual(result.order_id, "20260414A02N99999999")
        self.assertEqual(result.result_code, 0)


class TestQtyIntConversion(unittest.TestCase):
    """qty が float で来ても payload["Qty"] は int になること（500 対策）"""

    def setUp(self):
        _reset_cache()
        self.client = _make_client()

    def _get_payload(self, side: str, qty) -> dict:
        exchange = Exchange.TSE if side == Side.SELL else Exchange.SOR
        resp = _ok_response()
        with patch.object(self.client._session, "post", return_value=resp) as mock_post:
            self.client.send_order(
                symbol   = "5401",
                exchange = exchange,
                side     = side,
                qty      = qty,
            )
        _, kwargs = mock_post.call_args
        return kwargs.get("json", {})

    def test_sell_float_qty_becomes_int_in_payload(self):
        """SELL: qty=100.0 → payload["Qty"] == 100 (int)"""
        payload = self._get_payload(Side.SELL, 100.0)
        self.assertEqual(payload["Qty"], 100)
        self.assertIsInstance(
            payload["Qty"], int,
            f"Qty の型は int でなければならない。実際: {type(payload['Qty']).__name__}"
        )

    def test_sell_int_qty_stays_int(self):
        """SELL: qty=100 (int) → payload["Qty"] == 100 (int)"""
        payload = self._get_payload(Side.SELL, 100)
        self.assertEqual(payload["Qty"], 100)
        self.assertIsInstance(payload["Qty"], int)

    def test_buy_float_qty_becomes_int_in_payload(self):
        """BUY: qty=200.0 → payload["Qty"] == 200 (int)（回帰）"""
        payload = self._get_payload(Side.BUY, 200.0)
        self.assertEqual(payload["Qty"], 200)
        self.assertIsInstance(
            payload["Qty"], int,
            f"BUY でも Qty は int でなければならない。実際: {type(payload['Qty']).__name__}"
        )

    def test_buy_int_qty_stays_int(self):
        """BUY: qty=300 (int) → payload["Qty"] == 300 (int)"""
        payload = self._get_payload(Side.BUY, 300)
        self.assertEqual(payload["Qty"], 300)
        self.assertIsInstance(payload["Qty"], int)


class TestSendOrderHttpErrorLogging(unittest.TestCase):
    """HTTPError 発生時に SEND_ORDER_ERROR ログが出て例外が re-raise されること"""

    def setUp(self):
        _reset_cache()
        self.client = _make_client()

    def _make_http_error(self, status_code: int, body: str) -> "requests.HTTPError":
        import requests as req
        mock_resp = MagicMock(spec=req.Response)
        mock_resp.status_code = status_code
        mock_resp.text = body
        err = req.HTTPError(response=mock_resp)
        return err

    def test_500_error_is_logged_and_reraised(self):
        """500 を受けたとき SEND_ORDER_ERROR がログ出力され HTTPError が伝播すること"""
        import requests as req
        http_err = self._make_http_error(500, '{"Code":4002002,"Message":"Internal Error"}')

        with patch.object(self.client, "_request_with_token_retry", side_effect=http_err), \
             patch("src.kabusapi.client.logger") as mock_logger:
            with self.assertRaises(req.HTTPError):
                self.client.send_order(
                    symbol   = "5401",
                    exchange = Exchange.TSE,
                    side     = Side.SELL,
                    qty      = 100,
                )

        error_calls = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(
            any("SEND_ORDER_ERROR" in c for c in error_calls),
            f"SEND_ORDER_ERROR ログが出ていない: {error_calls}"
        )

    def test_error_log_contains_status_and_body(self):
        """ログに status_code と body が含まれること"""
        import requests as req
        body = '{"Code":4002002,"Message":"Qty must be integer"}'
        http_err = self._make_http_error(500, body)

        with patch.object(self.client, "_request_with_token_retry", side_effect=http_err), \
             patch("src.kabusapi.client.logger") as mock_logger:
            with self.assertRaises(req.HTTPError):
                self.client.send_order(
                    symbol   = "5401",
                    exchange = Exchange.TSE,
                    side     = Side.SELL,
                    qty      = 100,
                )

        error_calls = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(
            any("500" in c for c in error_calls),
            f"status_code=500 がログに含まれていない: {error_calls}"
        )
        self.assertTrue(
            any(body in c for c in error_calls),
            f"レスポンス本文がログに含まれていない: {error_calls}"
        )

    def test_400_error_is_also_logged(self):
        """400 でも同様に SEND_ORDER_ERROR が出ること（回帰）"""
        import requests as req
        http_err = self._make_http_error(400, '{"Code":4001001,"Message":"Bad Request"}')

        with patch.object(self.client, "_request_with_token_retry", side_effect=http_err), \
             patch("src.kabusapi.client.logger") as mock_logger:
            with self.assertRaises(req.HTTPError):
                self.client.send_order(
                    symbol   = "5401",
                    exchange = Exchange.TSE,
                    side     = Side.SELL,
                    qty      = 100,
                )

        error_calls = [str(c) for c in mock_logger.error.call_args_list]
        self.assertTrue(any("SEND_ORDER_ERROR" in c for c in error_calls))


class TestSendOrderDebugLog(unittest.TestCase):
    """send_order が qty_type を含む DEBUG ログを出力すること"""

    def setUp(self):
        _reset_cache()
        self.client = _make_client()

    def _debug_calls(self, side: str, qty) -> list[str]:
        exchange = Exchange.TSE if side == Side.SELL else Exchange.SOR
        resp = _ok_response()
        with patch.object(self.client._session, "post", return_value=resp), \
             patch("src.kabusapi.client.logger") as mock_logger:
            self.client.send_order(
                symbol   = "5401",
                exchange = exchange,
                side     = side,
                qty      = qty,
            )
        return [str(c) for c in mock_logger.debug.call_args_list]

    def test_debug_log_emitted_for_sell(self):
        calls = self._debug_calls(Side.SELL, 100.0)
        self.assertTrue(
            any("SEND_ORDER_PAYLOAD" in c for c in calls),
            f"SEND_ORDER_PAYLOAD のデバッグログが見つからない: {calls}"
        )

    def test_debug_log_contains_qty_type_for_sell(self):
        calls = self._debug_calls(Side.SELL, 100.0)
        self.assertTrue(
            any("qty_type" in c for c in calls),
            f"qty_type がデバッグログに含まれていない: {calls}"
        )

    def test_debug_log_emitted_for_buy(self):
        calls = self._debug_calls(Side.BUY, 100)
        self.assertTrue(
            any("SEND_ORDER_PAYLOAD" in c for c in calls),
            f"BUY 時も SEND_ORDER_PAYLOAD のデバッグログが必要: {calls}"
        )


# ------------------------------------------------------------------ #
# dry-run 検証用スクリプト相当（コマンドラインで直接実行した場合）
# ------------------------------------------------------------------ #
def _dryrun_print_payload():
    """
    kabuステーション未起動でも payload の内容を確認できる dry-run ユーティリティ。
    実行:
        python tests/test_send_order_sell_payload.py --dryrun
    """
    import logging
    logging.basicConfig(level=logging.DEBUG)

    _reset_cache()
    client = _make_client()

    for side, exchange, label in [
        (Side.SELL, Exchange.TSE, "SELL[現物]"),
        (Side.BUY,  Exchange.SOR, "BUY[現物]"),
    ]:
        resp = _ok_response()
        captured = {}

        original_post = client._session.post

        def _fake_post(url, **kwargs):
            captured.update(kwargs.get("json", {}))
            return resp

        client._session.post = _fake_post
        client.send_order(
            symbol   = "5401",
            exchange = exchange,
            side     = side,
            qty      = 100,
        )
        client._session.post = original_post

        print(f"\n{'='*55}")
        print(f"  {label} payload")
        print(f"{'='*55}")
        for k, v in captured.items():
            if k == "Password":
                v = "***"
            print(f"  {k:20s}: {repr(v)}")


if __name__ == "__main__":
    import sys as _sys
    if "--dryrun" in _sys.argv:
        _dryrun_print_payload()
    else:
        unittest.main()
