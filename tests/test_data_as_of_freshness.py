"""
tests/test_data_as_of_freshness.py

RCA 2026-07-08: data_as_of が T-2 になる不具合の回帰テスト。

原因1: SignalBridge._download_data() の銘柄別データ取得ループが
       「当日バッチ(yfinance)取得より前にキャッシュ(最大5日有効)を優先」
       しており、当日既に取得済みの新鮮なbatchデータを毎回捨てていた
       (docstring自体は「1.バッチ 2.個別リトライ 3.キャッシュ」の優先順を明記)。
原因2: data_as_of が next(iter(universe_raw)) という辞書挿入順の
       先頭銘柄1件だけを見て決定されており、鮮度の実態と無関係だった。
"""
from __future__ import annotations
import sys
sys.stdout.reconfigure(encoding="utf-8")

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd
import pytest

from src.kabusapi.signal_bridge import SignalBridge, _compute_data_as_of

JST = timezone(timedelta(hours=9))


def _make_ohlcv(last_date: str, n: int = 260) -> pd.DataFrame:
    idx = pd.bdate_range(end=last_date, periods=n)
    return pd.DataFrame(
        {
            "Open":   100.0,
            "High":   101.0,
            "Low":     99.0,
            "Close":  100.0,
            "Volume": 1_000_000,
        },
        index=idx,
    )


# ─────────────────────────────────────────────────────────────────────
# _compute_data_as_of: 最古日を採用する（辞書挿入順に依存しない）
# ─────────────────────────────────────────────────────────────────────

def test_compute_data_as_of_empty():
    assert _compute_data_as_of({}) == str(datetime.now(JST).date())


def test_compute_data_as_of_takes_oldest_not_first_key():
    universe_raw = {
        "FRESH.T": {"df": _make_ohlcv("2026-07-07")},
        "STALE.T": {"df": _make_ohlcv("2026-07-03")},
    }
    # 辞書の先頭は FRESH.T だが、最古(=最も鮮度が低い) STALE.T の日付を採用すること
    assert _compute_data_as_of(universe_raw) == "2026-07-03"


def test_compute_data_as_of_ignores_empty_df():
    universe_raw = {
        "EMPTY.T": {"df": pd.DataFrame()},
        "OK.T":    {"df": _make_ohlcv("2026-07-07")},
    }
    assert _compute_data_as_of(universe_raw) == "2026-07-07"


# ─────────────────────────────────────────────────────────────────────
# _download_data: 当日バッチ取得が成功した銘柄はキャッシュより優先されること
# ─────────────────────────────────────────────────────────────────────

def _make_bridge() -> SignalBridge:
    return SignalBridge(
        universe_tickers={"TEST.T": "電機精密"},
        fujiko_params={
            "min_sepa": 6, "min_rsr": 75.0, "rsr_exit": 70.0,
            "mom_period": 21, "turtle_entry": 20, "turtle_exit": 55,
            "use_turtle_entry": True,
        },
        capital=3_000_000,
        max_positions=3,
        min_hold_days=3,
        emergency_exit_pct=-0.08,
        live=False,
    )


def test_fresh_batch_data_preferred_over_stale_cache():
    """
    キャッシュに2日前の古いデータがあっても、当日のバッチ取得(yfinance)が
    成功していればバッチ側の新しいデータが使われること
    (旧実装は _load_from_cache を先に参照し fresh batch を捨てていた)。
    """
    sb = _make_bridge()
    fresh_df   = _make_ohlcv("2026-07-07")   # 当日バッチ取得で得られる新しいデータ
    stale_df   = _make_ohlcv("2026-07-03")   # ローカルキャッシュに残っている古いデータ
    bench_df   = _make_ohlcv("2026-07-07")

    raw_multi = pd.concat({"TEST.T": fresh_df, sb.benchmark_ticker: bench_df}, axis=1)

    with patch.object(sb, "_load_from_snapshot", return_value=pd.DataFrame()), \
         patch.object(sb, "_load_from_cache", return_value=stale_df) as mock_cache, \
         patch.object(sb, "_save_to_cache") as mock_save, \
         patch("src.kabusapi.signal_bridge.yf.download", return_value=raw_multi):
        universe_raw, _bench = sb._download_data()

    assert universe_raw["TEST.T"]["df"].index[-1].date().isoformat() == "2026-07-07"
    # 新鮮なバッチデータがキャッシュへ保存されること（次回以降のフォールバック用）
    assert mock_save.called
    # バッチが成功した銘柄について、古いキャッシュ値がそのまま採用されていないこと
    assert universe_raw["TEST.T"]["df"]["Close"].iloc[-1] == fresh_df["Close"].iloc[-1]


def test_cache_used_only_when_batch_fails():
    """バッチ取得が失敗した銘柄のみキャッシュへフォールバックすること"""
    sb = _make_bridge()
    stale_df = _make_ohlcv("2026-07-03")
    bench_df = _make_ohlcv("2026-07-07")
    # TEST.T をバッチ結果(raw)に含めない = バッチ取得失敗を模す
    raw_multi = pd.concat({sb.benchmark_ticker: bench_df}, axis=1)

    with patch.object(sb, "_load_from_snapshot", return_value=pd.DataFrame()), \
         patch.object(sb, "_load_from_cache", return_value=stale_df), \
         patch.object(sb, "_retry_single_fetch", return_value=pd.DataFrame()), \
         patch("src.kabusapi.signal_bridge.yf.download", return_value=raw_multi):
        universe_raw, _bench = sb._download_data()

    assert universe_raw["TEST.T"]["df"].index[-1].date().isoformat() == "2026-07-03"
