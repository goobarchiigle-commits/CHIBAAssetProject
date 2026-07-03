"""
test_position_normalizer.py
filter_live_positions() のユニットテスト。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from src.common.position_normalizer import filter_live_positions, _extract_qty


class TestExtractQty:
    def test_leaves_qty_preferred(self):
        assert _extract_qty({"LeavesQty": 100, "Qty": 0}) == 100.0

    def test_fallback_to_qty(self):
        assert _extract_qty({"Qty": 200}) == 200.0

    def test_none_value_skipped(self):
        assert _extract_qty({"LeavesQty": None, "Qty": 50}) == 50.0

    def test_missing_all_keys(self):
        assert _extract_qty({"Symbol": "1234"}) == 0.0

    def test_non_numeric_string_zero(self):
        assert _extract_qty({"LeavesQty": "abc"}) == 0.0

    def test_string_zero(self):
        assert _extract_qty({"LeavesQty": "0"}) == 0.0

    def test_string_positive(self):
        assert _extract_qty({"LeavesQty": "100"}) == 100.0

    def test_float_positive(self):
        assert _extract_qty({"LeavesQty": 100.5}) == 100.5


class TestFilterLivePositions:
    def test_excludes_zero_qty(self):
        positions = [
            {"Symbol": "5401", "LeavesQty": 0},
            {"Symbol": "5411", "LeavesQty": 200},
        ]
        result = filter_live_positions(positions)
        assert len(result) == 1
        assert result[0]["Symbol"] == "5411"

    def test_excludes_none_qty(self):
        positions = [
            {"Symbol": "5401", "LeavesQty": None},
            {"Symbol": "5411", "LeavesQty": 100},
        ]
        result = filter_live_positions(positions)
        assert len(result) == 1
        assert result[0]["Symbol"] == "5411"

    def test_excludes_string_zero(self):
        positions = [
            {"Symbol": "5401", "LeavesQty": "0"},
            {"Symbol": "5411", "LeavesQty": "100"},
        ]
        result = filter_live_positions(positions)
        assert len(result) == 1
        assert result[0]["Symbol"] == "5411"

    def test_excludes_missing_qty_keys(self):
        positions = [
            {"Symbol": "5401"},           # no qty keys at all
            {"Symbol": "5411", "Qty": 50},
        ]
        result = filter_live_positions(positions)
        assert len(result) == 1
        assert result[0]["Symbol"] == "5411"

    def test_all_live(self):
        positions = [
            {"Symbol": "5401", "LeavesQty": 100},
            {"Symbol": "5411", "LeavesQty": 200},
        ]
        result = filter_live_positions(positions)
        assert len(result) == 2

    def test_all_zero(self):
        positions = [
            {"Symbol": "5401", "LeavesQty": 0},
            {"Symbol": "5411", "LeavesQty": 0},
        ]
        result = filter_live_positions(positions)
        assert result == []

    def test_empty_input(self):
        assert filter_live_positions([]) == []

    def test_returns_original_dicts(self):
        """filter は辞書を変更せずそのまま返す"""
        pos = {"Symbol": "5411", "LeavesQty": 100, "Price": 2345.0}
        result = filter_live_positions([pos])
        assert result[0] is pos

    def test_mixed_qty_fields(self):
        """HoldQty フォールバックも拾う"""
        positions = [
            {"Symbol": "5401", "HoldQty": 0},
            {"Symbol": "5411", "HoldQty": 300},
        ]
        result = filter_live_positions(positions)
        assert len(result) == 1
        assert result[0]["Symbol"] == "5411"
