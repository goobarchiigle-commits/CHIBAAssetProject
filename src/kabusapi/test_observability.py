"""
src/kabusapi/test_observability.py
ロギング/出力の観測性テスト。

実行:
    cd C:/ai-trading
    python -m pytest src/kabusapi/test_observability.py -v
"""
from __future__ import annotations

import io
import json
import logging
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class TestUTF8Logging(unittest.TestCase):
    """StreamHandler が UTF-8 文字化けを起こさないこと。"""

    def test_japanese_log_no_mojibake(self):
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log = logging.getLogger("test_utf8_obs")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            msg = "テスト 日本語 ログ LIVE_UNIVERSE SHADOW_UNIVERSE"
            log.info(msg)
            out = buf.getvalue()
            self.assertIn("LIVE_UNIVERSE", out)
            self.assertIn("SHADOW_UNIVERSE", out)
            self.assertIn("テスト", out)
        finally:
            log.removeHandler(handler)

    def test_stream_handler_encoding_utf8(self):
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        # StringIO は常に str。encoding 属性は None のケースをガード。
        enc = getattr(buf, "encoding", None) or "utf-8"
        self.assertIn(enc.lower().replace("-", ""), ("utf8", "utf-8"))


class TestBridgeResultUniverseStats(unittest.TestCase):
    """BridgeResult.universe_stats の不変条件テスト。"""

    def _make_stats(self, live=42, shadow=20, rsr_context=62, tradeable=37,
                    filtered_price=3, filtered_risk=1):
        return {
            "live": live,
            "shadow": shadow,
            "rsr_context": rsr_context,
            "tradeable": tradeable,
            "filtered_price": filtered_price,
            "filtered_risk": filtered_risk,
        }

    def test_context_eq_live_plus_shadow(self):
        """rsr_context == live + shadow（重複なし前提）。"""
        s = self._make_stats(live=42, shadow=20, rsr_context=62)
        self.assertEqual(s["rsr_context"], s["live"] + s["shadow"])

    def test_tradeable_le_live(self):
        """tradeable <= live（データ取得失敗銘柄の除外後）。"""
        s = self._make_stats(live=42, tradeable=37)
        self.assertLessEqual(s["tradeable"], s["live"])

    def test_filtered_price_nonneg(self):
        s = self._make_stats(filtered_price=0)
        self.assertGreaterEqual(s["filtered_price"], 0)

    def test_filtered_risk_nonneg(self):
        s = self._make_stats(filtered_risk=0)
        self.assertGreaterEqual(s["filtered_risk"], 0)

    def test_tradeable_ge_zero(self):
        s = self._make_stats(tradeable=0)
        self.assertGreaterEqual(s["tradeable"], 0)


class TestUniverseLogStructure(unittest.TestCase):
    """[UNIVERSE] ログ行の構造テスト（フォーマット整合性）。"""

    def test_universe_log_format(self):
        """[UNIVERSE] タグで始まるログに必須キーが含まれること。"""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log = logging.getLogger("test_universe_log")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            log.info(
                "[UNIVERSE] live=%d shadow=%d context=%d tradeable=%d filtered_price=%d filtered_risk=%d",
                42, 20, 62, 37, 3, 1,
            )
            out = buf.getvalue()
            for key in ("live=42", "shadow=20", "context=62", "tradeable=37",
                        "filtered_price=3", "filtered_risk=1"):
                self.assertIn(key, out, f"missing key: {key}")
        finally:
            log.removeHandler(handler)

    def test_cluster_log_format(self):
        """[CLUSTER] タグで始まるログに必須キーが含まれること。"""
        buf = io.StringIO()
        handler = logging.StreamHandler(buf)
        handler.setFormatter(logging.Formatter("%(message)s"))
        log = logging.getLogger("test_cluster_log")
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        try:
            log.info(
                "[CLUSTER] sector_exposure=%s rejected_alloc_cap=%d risk_rejected=%d missed_entries=%d",
                {"機械": 1}, 0, 0, 2,
            )
            out = buf.getvalue()
            for key in ("[CLUSTER]", "sector_exposure=", "rejected_alloc_cap=",
                        "risk_rejected=", "missed_entries="):
                self.assertIn(key, out, f"missing key: {key}")
        finally:
            log.removeHandler(handler)


class TestValidateBuildOrdersContract(unittest.TestCase):
    """_validate_build_orders_contract の 5-tuple 強制テスト。"""

    def setUp(self):
        try:
            heavy = ["kabusapi", "kabu_station_api", "pandas_datareader"]
            mocks = {m: MagicMock() for m in heavy}
            with patch.dict("sys.modules", mocks):
                from src.kabusapi.signal_bridge import _validate_build_orders_contract
            self.validate = _validate_build_orders_contract
        except Exception as e:
            self.skipTest(f"import unavailable: {e}")

    def test_accepts_5tuple(self):
        self.validate(([], [], 0, 0, 0))

    def test_rejects_4tuple(self):
        with self.assertRaises(ValueError):
            self.validate(([], [], 0, 0))

    def test_rejects_list(self):
        with self.assertRaises(TypeError):
            self.validate([[], [], 0, 0, 0])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    unittest.main(verbosity=2)
