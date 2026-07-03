"""
tests/test_api_response.py
P0-2: APIレスポンス判定ロジックのユニットテスト
"""
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "src"))

from src.kabusapi.client import parse_order_response


class TestParseOrderResponse(unittest.TestCase):

    def test_order_id_without_result_key(self):
        """OrderId が存在して Result キーが欠落した場合を成功とみなす（3/16 実運用再現）"""
        mock_response = {"OrderId": "20260316A02N80057402"}
        result = parse_order_response(mock_response)
        self.assertTrue(result["success"])
        self.assertEqual(result["order_id"], "20260316A02N80057402")
        self.assertEqual(result["result_code"], 0)

    def test_result_zero_success(self):
        """Result=0 は成功"""
        mock_response = {"Result": 0, "OrderId": "TEST123"}
        result = parse_order_response(mock_response)
        self.assertTrue(result["success"])
        self.assertEqual(result["order_id"], "TEST123")
        self.assertEqual(result["result_code"], 0)

    def test_result_nonzero_failure(self):
        """Result=4001013 は失敗（認証エラー）"""
        mock_response = {"Result": 4001013, "Message": "Auth error"}
        result = parse_order_response(mock_response)
        self.assertFalse(result["success"])
        self.assertEqual(result["result_code"], 4001013)

    def test_empty_response_failure(self):
        """空レスポンスは失敗"""
        mock_response = {}
        result = parse_order_response(mock_response)
        self.assertFalse(result["success"])
        self.assertEqual(result["result_code"], -1)
        self.assertEqual(result["order_id"], "")

    def test_result_code_alias(self):
        """ResultCode（旧仕様）も正しく処理する"""
        mock_response = {"ResultCode": 0, "OrderId": "OLD_STYLE"}
        result = parse_order_response(mock_response)
        self.assertTrue(result["success"])

    def test_result_nonzero_with_order_id(self):
        """OrderId があっても Result が非0なら失敗"""
        mock_response = {"OrderId": "FAIL_ID", "Result": 9999}
        result = parse_order_response(mock_response)
        self.assertFalse(result["success"])
        self.assertEqual(result["result_code"], 9999)


if __name__ == "__main__":
    unittest.main()
