"""Tests for src/database/schema.py"""
from __future__ import annotations

import pandas as pd
import pytest

from src.database.exceptions import SchemaValidationError
from src.database.schema import TABLE_SCHEMAS, to_schema_json, validate_schema


class TestValidateSchema:
    def test_valid_ohlcv_passes(self):
        df = pd.DataFrame({"Date": ["2024-01-01"], "Code": ["1301"], "Open": [100.0]})
        validate_schema(df, "ohlcv")  # 例外を送出しないこと

    def test_missing_required_column_raises(self):
        df = pd.DataFrame({"Date": ["2024-01-01"]})  # Code列欠落
        with pytest.raises(SchemaValidationError):
            validate_schema(df, "ohlcv")

    def test_unknown_table_raises(self):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(SchemaValidationError):
            validate_schema(df, "not_a_real_table")

    def test_extra_columns_allowed(self):
        df = pd.DataFrame({"Date": ["2024-01-01"], "Code": ["1301"], "FutureColumn": [1]})
        validate_schema(df, "ohlcv")  # 未知の追加列はエラーにしない


class TestSchemaJson:
    def test_to_schema_json_covers_all_tables(self):
        out = to_schema_json()
        assert set(out.keys()) == set(TABLE_SCHEMAS.keys())
        for table_name, table_def in out.items():
            assert "columns" in table_def
            assert len(table_def["columns"]) > 0

    def test_classification_flags_have_source_and_last_updated(self):
        cols = {c["name"] for c in to_schema_json()["classifications"]["columns"]}
        for flag in ("IsJPXPrime150", "IsJPX400", "IsNikkei225"):
            assert flag in cols
        assert "jpx_prime150_source" in cols
        assert "jpx_prime150_last_updated" in cols
        assert "jpx400_source" in cols
        assert "nikkei225_source" in cols
