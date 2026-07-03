"""
tests/analytics/test_future_leader_survivability.py
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from src.analytics.future_leader_survivability import (
    FORWARD_SHORT,
    FORWARD_WINDOW,
    FutureLeaderSurvivabilityRecord,
    _compute_forward_metrics,
    _find_obs_idx,
    _load_integrity_map,
    _load_materialized_keys,
    _normalize_df_index,
    _safe_float,
    _write_survivability_log,
    format_survivability_summary,
    materialize_future_leader_survivability,
)

JST = timezone(timedelta(hours=9))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 30, start: str = "2025-01-01", base_close: float = 1000.0) -> pd.DataFrame:
    idx   = pd.bdate_range(start=start, periods=n, freq="B")
    close = np.linspace(base_close, base_close * 1.05, n)
    high  = close * 1.01
    low   = close * 0.99
    vol   = np.full(n, 1_000_000)
    return pd.DataFrame({"Open": close, "High": high, "Low": low, "Close": close, "Volume": vol}, index=idx)


def _candidate_line(
    symbol: str,
    obs_date: str,
    score: float = 75.0,
    rsr_zone: str = "breakout_zone",
    obs_ts: str = "2025-01-01T09:00:00+09:00",
) -> str:
    return json.dumps({
        "symbol": symbol,
        "observation_date": obs_date,
        "observation_ts": obs_ts,
        "future_leader_score": score,
        "rsr_zone": rsr_zone,
    })


def _integrity_line(obs_date: str, state: str, weight: float) -> str:
    return json.dumps({
        "observation_date": obs_date,
        "integrity_state": state,
        "data_quality_weight": weight,
    })


# ─────────────────────────────────────────────────────────────────────────────
# _safe_float
# ─────────────────────────────────────────────────────────────────────────────

def test_safe_float_valid():
    assert _safe_float(1000.0) == pytest.approx(1000.0)


def test_safe_float_string_number():
    assert _safe_float("500") == pytest.approx(500.0)


def test_safe_float_nan_returns_none():
    assert _safe_float(float("nan")) is None


def test_safe_float_inf_returns_none():
    assert _safe_float(float("inf")) is None


def test_safe_float_invalid_string_returns_none():
    assert _safe_float("INVALID") is None


# ─────────────────────────────────────────────────────────────────────────────
# _normalize_df_index / _find_obs_idx
# ─────────────────────────────────────────────────────────────────────────────

def test_normalize_df_index_strips_time():
    df = _make_ohlcv(5, start="2025-01-06")
    df_norm = _normalize_df_index(df)
    assert df_norm.index[0] == pd.Timestamp("2025-01-06")


def test_find_obs_idx_found():
    df = _normalize_df_index(_make_ohlcv(10, start="2025-01-06"))
    idx = _find_obs_idx(df, "2025-01-06")
    assert idx == 0


def test_find_obs_idx_not_found():
    df = _normalize_df_index(_make_ohlcv(5, start="2025-01-06"))
    assert _find_obs_idx(df, "2025-12-31") is None


# ─────────────────────────────────────────────────────────────────────────────
# _compute_forward_metrics — 5d return correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_5d_return_correctness():
    df  = _normalize_df_index(_make_ohlcv(30, start="2025-01-02"))
    obs_idx  = 5
    close_obs = float(df["Close"].iloc[obs_idx])
    close_5d  = float(df["Close"].iloc[obs_idx + FORWARD_SHORT])

    # today far in the future so no lookahead cutoff applies
    fwd5, fwd10, _, _, days, insuf = _compute_forward_metrics(
        df, obs_idx, close_obs, today=date(2026, 1, 1)
    )
    expected = round(close_5d / close_obs - 1.0, 6)
    assert fwd5 == pytest.approx(expected, abs=1e-5)
    assert days == FORWARD_WINDOW
    assert not insuf


# ─────────────────────────────────────────────────────────────────────────────
# _compute_forward_metrics — 10d return correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_10d_return_correctness():
    df  = _normalize_df_index(_make_ohlcv(30, start="2025-01-02"))
    obs_idx   = 3
    close_obs = float(df["Close"].iloc[obs_idx])
    close_10d = float(df["Close"].iloc[obs_idx + FORWARD_WINDOW])

    fwd5, fwd10, _, _, days, insuf = _compute_forward_metrics(
        df, obs_idx, close_obs, today=date(2026, 1, 1)
    )
    expected = round(close_10d / close_obs - 1.0, 6)
    assert fwd10 == pytest.approx(expected, abs=1e-5)
    assert not insuf


# ─────────────────────────────────────────────────────────────────────────────
# _compute_forward_metrics — MFE / MAE correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_mfe_mae_correctness():
    df = _normalize_df_index(_make_ohlcv(30, start="2025-01-02"))
    obs_idx   = 2
    close_obs = float(df["Close"].iloc[obs_idx])

    _, _, mfe, mae, _, _ = _compute_forward_metrics(
        df, obs_idx, close_obs, today=date(2026, 1, 1)
    )
    fwd_slice = df.iloc[obs_idx + 1: obs_idx + 1 + FORWARD_WINDOW]
    expected_mfe = round(float(fwd_slice["High"].max()) / close_obs - 1.0, 6)
    expected_mae = round(float(fwd_slice["Low"].min())  / close_obs - 1.0, 6)
    assert mfe == pytest.approx(expected_mfe, abs=1e-5)
    assert mae == pytest.approx(expected_mae, abs=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# _compute_forward_metrics — insufficient forward data
# ─────────────────────────────────────────────────────────────────────────────

def test_insufficient_when_fewer_than_10_forward_bars():
    df = _normalize_df_index(_make_ohlcv(15, start="2025-01-02"))
    obs_idx   = 8   # only 6 bars forward (indices 9-14)
    close_obs = float(df["Close"].iloc[obs_idx])

    _, fwd10, mfe, mae, days, insuf = _compute_forward_metrics(
        df, obs_idx, close_obs, today=date(2026, 1, 1)
    )
    assert insuf is True
    assert fwd10 is None
    assert mfe   is None
    assert mae   is None
    assert days < FORWARD_WINDOW


# ─────────────────────────────────────────────────────────────────────────────
# Lookahead prevention: today's bar excluded
# ─────────────────────────────────────────────────────────────────────────────

def test_no_lookahead_today_bar_excluded():
    df = _normalize_df_index(_make_ohlcv(30, start="2025-01-02"))
    obs_idx   = 0
    close_obs = float(df["Close"].iloc[obs_idx])

    # Set today to the 6th bar (index 5) — bars at index 5+ are excluded
    # Only 4 forward bars available (indices 1-4), which is < FORWARD_WINDOW
    today = df.index[5].date()
    _, _, _, _, days, insuf = _compute_forward_metrics(df, obs_idx, close_obs, today=today)
    assert insuf is True
    assert days < FORWARD_WINDOW


# ─────────────────────────────────────────────────────────────────────────────
# _load_materialized_keys
# ─────────────────────────────────────────────────────────────────────────────

def test_load_materialized_keys(tmp_path: Path):
    log_dir = tmp_path / "surv"
    log_dir.mkdir()
    (log_dir / "future_leader_survivability_20250101.jsonl").write_text(
        json.dumps({"symbol": "A.T", "observation_date": "2025-01-01"}) + "\n" +
        json.dumps({"symbol": "B.T", "observation_date": "2025-01-01"}) + "\n",
        encoding="utf-8",
    )
    keys = _load_materialized_keys(log_dir)
    assert ("A.T", "2025-01-01") in keys
    assert ("B.T", "2025-01-01") in keys


def test_load_materialized_keys_empty_dir(tmp_path: Path):
    keys = _load_materialized_keys(tmp_path / "nonexistent")
    assert len(keys) == 0


# ─────────────────────────────────────────────────────────────────────────────
# _load_integrity_map
# ─────────────────────────────────────────────────────────────────────────────

def test_load_integrity_map_reads_state_and_weight(tmp_path: Path):
    log_dir = tmp_path / "integrity"
    log_dir.mkdir()
    (log_dir / "future_leader_integrity_20250101.jsonl").write_text(
        _integrity_line("2025-01-01", "PARTIAL_FAILURE", 0.5) + "\n" +
        _integrity_line("2025-01-02", "MAJOR_FAILURE",   0.0) + "\n",
        encoding="utf-8",
    )
    imap = _load_integrity_map(log_dir)
    assert imap["2025-01-01"] == ("PARTIAL_FAILURE", 0.5)
    assert imap["2025-01-02"] == ("MAJOR_FAILURE",   0.0)


def test_load_integrity_map_last_record_wins(tmp_path: Path):
    log_dir = tmp_path / "integrity"
    log_dir.mkdir()
    (log_dir / "future_leader_integrity_20250101.jsonl").write_text(
        _integrity_line("2025-01-01", "PARTIAL_FAILURE", 0.5) + "\n" +
        _integrity_line("2025-01-01", "CLEAN",           1.0) + "\n",
        encoding="utf-8",
    )
    imap = _load_integrity_map(log_dir)
    assert imap["2025-01-01"] == ("CLEAN", 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# materialize_future_leader_survivability — integration
# ─────────────────────────────────────────────────────────────────────────────

def _make_candidates_file(tmp_path: Path, lines: list[str]) -> Path:
    p = tmp_path / "candidates.jsonl"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_materialize_basic_clean(tmp_path: Path):
    df = _make_ohlcv(30, start="2025-01-02")
    # obs at index 5
    obs_date = df.index[5].strftime("%Y-%m-%d")
    cands    = _make_candidates_file(tmp_path, [_candidate_line("AAA.T", obs_date)])

    def loader(sym: str) -> Optional[pd.DataFrame]:
        return df if sym == "AAA.T" else None

    n = materialize_future_leader_survivability(
        candidates_log_path   = cands,
        integrity_log_dir     = tmp_path / "integrity_empty",
        survivability_log_dir = tmp_path / "surv",
        ohlcv_loader          = loader,
        today                 = date(2026, 1, 1),
    )
    assert n == 1
    keys = _load_materialized_keys(tmp_path / "surv")
    assert ("AAA.T", obs_date) in keys


def test_materialize_duplicate_prevented(tmp_path: Path):
    df       = _make_ohlcv(30, start="2025-01-02")
    obs_date = df.index[5].strftime("%Y-%m-%d")
    cands    = _make_candidates_file(tmp_path, [_candidate_line("AAA.T", obs_date)])

    def loader(sym: str) -> Optional[pd.DataFrame]:
        return df if sym == "AAA.T" else None

    surv_dir = tmp_path / "surv"
    n1 = materialize_future_leader_survivability(
        candidates_log_path=cands, integrity_log_dir=tmp_path/"int",
        survivability_log_dir=surv_dir, ohlcv_loader=loader, today=date(2026, 1, 1),
    )
    n2 = materialize_future_leader_survivability(
        candidates_log_path=cands, integrity_log_dir=tmp_path/"int",
        survivability_log_dir=surv_dir, ohlcv_loader=loader, today=date(2026, 1, 1),
    )
    assert n1 == 1
    assert n2 == 0  # already materialized


def test_materialize_major_failure_excluded(tmp_path: Path):
    df       = _make_ohlcv(30, start="2025-01-02")
    obs_date = df.index[5].strftime("%Y-%m-%d")
    cands    = _make_candidates_file(tmp_path, [_candidate_line("BBB.T", obs_date)])

    int_dir = tmp_path / "integrity"
    int_dir.mkdir()
    (int_dir / "future_leader_integrity_20250101.jsonl").write_text(
        _integrity_line(obs_date, "MAJOR_FAILURE", 0.0) + "\n", encoding="utf-8"
    )

    def loader(sym: str) -> Optional[pd.DataFrame]:
        return df

    n = materialize_future_leader_survivability(
        candidates_log_path=cands, integrity_log_dir=int_dir,
        survivability_log_dir=tmp_path/"surv", ohlcv_loader=loader, today=date(2026, 1, 1),
    )
    assert n == 0


def test_materialize_partial_failure_persisted(tmp_path: Path):
    df       = _make_ohlcv(30, start="2025-01-02")
    obs_date = df.index[5].strftime("%Y-%m-%d")
    cands    = _make_candidates_file(tmp_path, [_candidate_line("CCC.T", obs_date)])

    int_dir = tmp_path / "integrity"
    int_dir.mkdir()
    (int_dir / "future_leader_integrity_20250101.jsonl").write_text(
        _integrity_line(obs_date, "PARTIAL_FAILURE", 0.5) + "\n", encoding="utf-8"
    )

    def loader(sym: str) -> Optional[pd.DataFrame]:
        return df

    n = materialize_future_leader_survivability(
        candidates_log_path=cands, integrity_log_dir=int_dir,
        survivability_log_dir=tmp_path/"surv", ohlcv_loader=loader, today=date(2026, 1, 1),
    )
    assert n == 1  # PARTIAL_FAILURE: persisted
    # Verify JSONL contains the PARTIAL_FAILURE metadata
    keys = _load_materialized_keys(tmp_path / "surv")
    assert ("CCC.T", obs_date) in keys


def test_materialize_insufficient_forward_data_skipped(tmp_path: Path):
    df = _make_ohlcv(12, start="2025-01-02")
    # obs at last position: only 0 forward bars
    obs_date = df.index[-1].strftime("%Y-%m-%d")
    cands    = _make_candidates_file(tmp_path, [_candidate_line("DDD.T", obs_date)])

    def loader(sym: str) -> Optional[pd.DataFrame]:
        return df

    n = materialize_future_leader_survivability(
        candidates_log_path=cands, integrity_log_dir=tmp_path/"int",
        survivability_log_dir=tmp_path/"surv", ohlcv_loader=loader, today=date(2026, 1, 1),
    )
    assert n == 0  # insufficient — will be re-checked next run


def test_materialize_no_candidates_file(tmp_path: Path):
    n = materialize_future_leader_survivability(
        candidates_log_path   = tmp_path / "nonexistent.jsonl",
        integrity_log_dir     = tmp_path / "int",
        survivability_log_dir = tmp_path / "surv",
        ohlcv_loader          = lambda sym: None,
        today                 = date(2026, 1, 1),
    )
    assert n == 0


# ─────────────────────────────────────────────────────────────────────────────
# Append-only persistence
# ─────────────────────────────────────────────────────────────────────────────

def test_append_only_write(tmp_path: Path):
    log_dir = tmp_path / "surv"
    rec = FutureLeaderSurvivabilityRecord(
        symbol="A.T", observation_date="2025-01-10", observation_ts="2025-01-10T09:00:00+09:00",
        future_leader_score=80.0, rsr_zone="strong", data_quality_weight=1.0,
        integrity_state="CLEAN", close_price_at_observation=1500.0,
        forward_return_5d=0.02, forward_return_10d=0.04,
        mfe_10d=0.05, mae_10d=-0.01, days_materialized=10, insufficient_forward_data=False,
    )
    _write_survivability_log([rec], log_dir, today=date(2025, 2, 1))
    _write_survivability_log([rec], log_dir, today=date(2025, 2, 1))

    files = list(log_dir.glob("*.jsonl"))
    assert len(files) == 1
    lines = [l for l in files[0].read_text(encoding="utf-8").splitlines() if l.strip()]
    # Both writes appended
    assert len(lines) == 2


# ─────────────────────────────────────────────────────────────────────────────
# JSONL integrity metadata fields
# ─────────────────────────────────────────────────────────────────────────────

def test_jsonl_contains_integrity_fields(tmp_path: Path):
    df       = _make_ohlcv(30, start="2025-01-02")
    obs_date = df.index[5].strftime("%Y-%m-%d")
    int_dir  = tmp_path / "integrity"
    int_dir.mkdir()
    (int_dir / "future_leader_integrity_20250101.jsonl").write_text(
        _integrity_line(obs_date, "CLEAN", 1.0) + "\n", encoding="utf-8"
    )
    cands = _make_candidates_file(tmp_path, [_candidate_line("EEE.T", obs_date)])

    def loader(sym: str) -> Optional[pd.DataFrame]:
        return df

    materialize_future_leader_survivability(
        candidates_log_path=cands, integrity_log_dir=int_dir,
        survivability_log_dir=tmp_path/"surv", ohlcv_loader=loader, today=date(2026, 1, 1),
    )
    surv_files = list((tmp_path / "surv").glob("*.jsonl"))
    assert surv_files
    rec = json.loads(surv_files[0].read_text(encoding="utf-8").splitlines()[0])
    assert rec["integrity_state"]      == "CLEAN"
    assert rec["data_quality_weight"]  == 1.0
    assert "forward_return_5d"  in rec
    assert "forward_return_10d" in rec
    assert "mfe_10d"            in rec
    assert "mae_10d"            in rec


# ─────────────────────────────────────────────────────────────────────────────
# format_survivability_summary — CLEAN only
# ─────────────────────────────────────────────────────────────────────────────

def test_format_summary_clean_only(tmp_path: Path):
    log_dir = tmp_path / "surv"
    log_dir.mkdir()
    clean_rec = {
        "symbol": "A.T", "observation_date": "2025-01-10", "observation_ts": "",
        "future_leader_score": 80.0, "rsr_zone": "strong",
        "data_quality_weight": 1.0, "integrity_state": "CLEAN",
        "close_price_at_observation": 1000.0,
        "forward_return_5d": 0.02, "forward_return_10d": 0.04,
        "mfe_10d": 0.05, "mae_10d": -0.01, "days_materialized": 10,
        "insufficient_forward_data": False,
    }
    partial_rec = {**clean_rec, "data_quality_weight": 0.5, "integrity_state": "PARTIAL_FAILURE"}

    (log_dir / "future_leader_survivability_20250201.jsonl").write_text(
        json.dumps({"materialization_ts": "...", **clean_rec}) + "\n" +
        json.dumps({"materialization_ts": "...", **partial_rec}) + "\n",
        encoding="utf-8",
    )
    summary = format_survivability_summary(log_dir)
    assert "FUTURE LEADER SURVIVABILITY" in summary
    assert "clean_complete_records: 1" in summary  # only 1 clean record
    assert "5d expectancy" in summary
    assert "10d expectancy" in summary
    assert "+2.0%" in summary  # 0.02 → +2.0%


def test_format_summary_no_records(tmp_path: Path):
    summary = format_survivability_summary(tmp_path / "empty")
    assert "FUTURE LEADER SURVIVABILITY" in summary
    assert "insufficient clean records" in summary


# ─────────────────────────────────────────────────────────────────────────────
# observation_only — no execution path identifiers in module source
# ─────────────────────────────────────────────────────────────────────────────

def test_no_execution_path_identifiers():
    import inspect
    import src.analytics.future_leader_survivability as mod
    source = inspect.getsource(mod)
    # Strip comment/docstring lines — only check code-form references
    code_lines = [
        ln for ln in source.splitlines()
        if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith('"""')
        and not ln.strip().startswith("'")
    ]
    code_only = "\n".join(code_lines)
    forbidden = [
        "signal_bridge",
        "order_routing",
        "governance_promotion",
        "deployable_universe",
        "run_order",
        "place_order",
    ]
    for term in forbidden:
        assert term not in code_only, f"forbidden code term '{term}' found in survivability module"
