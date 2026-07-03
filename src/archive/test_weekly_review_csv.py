"""
scripts/test_weekly_review_csv.py
weekly_review_csv.py の単体テスト

実行:
  cd C:/ai-trading
  python src/scripts/test_weekly_review_csv.py
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

from scripts.weekly_review_csv import (
    extract_rows,
    dedup_rows,
    load_signal_files,
    COLUMNS,
)

JST = timezone(timedelta(hours=9))
PASS = "[PASS]"
FAIL = "[FAIL]"


# ─────────────────────────────────────────────────────────────────────────────
# テストデータファクトリ
# ─────────────────────────────────────────────────────────────────────────────

def make_signal(
    date:       str   = "2026-04-01",
    mode:       str   = "DRY_RUN",
    api_ok:     bool  = False,
    cb_state:   str   = "NORMAL",
    signals:    list  = None,
    send_results: list = None,
) -> dict:
    """テスト用シグナル辞書を生成する。"""
    return {
        "generated_at": f"{date}T09:00:00+0900",
        "mode": mode,
        "data_as_of": date,
        "portfolio_summary": {
            "cb_state": cb_state,
            "positions_api": {"ok": api_ok},
            "wallet_api": {"ok": api_ok},
        },
        "signals": signals or [],
        "orders": [],
        "send_results": send_results or [],
        "_file_date": date,
        "_filename": f"signal_{date.replace('-','')}_090000.json",
    }


# ─────────────────────────────────────────────────────────────────────────────
# テストケース
# ─────────────────────────────────────────────────────────────────────────────

def test_extract_rows_basic():
    """signal=1/0/-1 が正しく列に変換されること。"""
    data = make_signal(
        date="2026-04-01",
        signals=[
            {"symbol": "8035.T", "signal": 1},
            {"symbol": "9104.T", "signal": 0},
            {"symbol": "6920.T", "signal": -1},
        ],
    )
    rows = extract_rows(data)
    assert len(rows) == 3, f"行数 {len(rows)} != 3"
    assert rows[0]["signal"] == "1"
    assert rows[1]["signal"] == "0"
    assert rows[2]["signal"] == "-1"
    # DRY_RUN → order_sent は常に False
    assert all(r["order_sent"] == "False" for r in rows)
    print(f"  {PASS} test_extract_rows_basic")


def test_order_sent_live():
    """LIVE モードかつ send_results に含まれる銘柄は order_sent=True になること。"""
    data = make_signal(
        date="2026-04-01",
        mode="LIVE",
        api_ok=True,
        signals=[
            {"symbol": "8035.T", "signal": 1},
            {"symbol": "9104.T", "signal": 0},
        ],
        send_results=[
            {"symbol": "8035.T", "side": "BUY", "qty": 100, "success": True},
        ],
    )
    rows = extract_rows(data)
    sent = {r["symbol"]: r["order_sent"] for r in rows}
    assert sent["8035.T"] == "True",  f"8035.T order_sent={sent['8035.T']}"
    assert sent["9104.T"] == "False", f"9104.T order_sent={sent['9104.T']}"
    print(f"  {PASS} test_order_sent_live")


def test_api_ok_column():
    """api_ok が True/False/unknown の各パターンで正しく出力されること。"""
    # True
    d_true = make_signal(api_ok=True, signals=[{"symbol": "X", "signal": 0}])
    assert extract_rows(d_true)[0]["api_ok"] == "True"

    # False
    d_false = make_signal(api_ok=False, signals=[{"symbol": "X", "signal": 0}])
    assert extract_rows(d_false)[0]["api_ok"] == "False"

    # api_ok なし → "unknown"
    d_none = make_signal(signals=[{"symbol": "X", "signal": 0}])
    d_none["portfolio_summary"].pop("positions_api", None)
    d_none["portfolio_summary"].pop("wallet_api", None)
    assert extract_rows(d_none)[0]["api_ok"] == "unknown"

    print(f"  {PASS} test_api_ok_column")


def test_cb_state_column():
    """cb_state が正しくセットされること。"""
    for state in ("NORMAL", "CB_ACTIVE", "RECOVERY"):
        data = make_signal(cb_state=state, signals=[{"symbol": "X", "signal": 0}])
        rows = extract_rows(data)
        assert rows[0]["cb_state"] == state, f"cb_state mismatch: {rows[0]['cb_state']}"
    print(f"  {PASS} test_cb_state_column")


def test_dedup_rows():
    """同一日・銘柄の重複行が最後の値に集約されること。"""
    rows = [
        {"date": "2026-04-01", "symbol": "8035.T", "signal": "0",
         "order_sent": "False", "api_ok": "False", "cb_state": "NORMAL"},
        {"date": "2026-04-01", "symbol": "8035.T", "signal": "1",
         "order_sent": "True",  "api_ok": "True",  "cb_state": "NORMAL"},
        {"date": "2026-04-02", "symbol": "8035.T", "signal": "0",
         "order_sent": "False", "api_ok": "False", "cb_state": "NORMAL"},
    ]
    result = dedup_rows(rows)
    assert len(result) == 2, f"dedup後 {len(result)} 行 (期待値 2)"
    # 後勝ちで signal=1 が残る
    apr1 = next(r for r in result if r["date"] == "2026-04-01")
    assert apr1["signal"] == "1", f"後勝ちが機能していない: {apr1['signal']}"
    print(f"  {PASS} test_dedup_rows")


def test_output_columns():
    """出力行のキーが COLUMNS と一致すること。"""
    data = make_signal(signals=[{"symbol": "8035.T", "signal": 1}])
    rows = extract_rows(data)
    assert len(rows) == 1
    assert set(rows[0].keys()) == set(COLUMNS), f"キー不一致: {set(rows[0].keys())}"
    print(f"  {PASS} test_output_columns")


def test_load_signal_files_date_filter():
    """date_from / date_to によるフィルタリングが機能すること。"""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # 2件作成: 2026-03-31 と 2026-04-10
        for date_str in ("20260331", "20260410"):
            sig = {
                "generated_at": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}T09:00:00+0900",
                "mode": "DRY_RUN",
                "portfolio_summary": {},
                "signals": [],
                "orders": [],
            }
            (tmpdir / f"signal_{date_str}_090000.json").write_text(
                json.dumps(sig), encoding="utf-8"
            )

        # 2026-03-01 ～ 2026-04-04 → 1件のみ
        result = load_signal_files(tmpdir, "2026-03-01", "2026-04-04")
        assert len(result) == 1, f"フィルタ後 {len(result)} 件 (期待値 1)"
        assert result[0]["_file_date"] == "2026-03-31"
    print(f"  {PASS} test_load_signal_files_date_filter")


# ─────────────────────────────────────────────────────────────────────────────
# 実行
# ─────────────────────────────────────────────────────────────────────────────

def run_all() -> int:
    tests = [
        test_extract_rows_basic,
        test_order_sent_live,
        test_api_ok_column,
        test_cb_state_column,
        test_dedup_rows,
        test_output_columns,
        test_load_signal_files_date_filter,
    ]
    print("=" * 50)
    print("  weekly_review_csv 単体テスト")
    print("=" * 50)
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as exc:
            print(f"  {FAIL} {t.__name__}: {exc}")
            failed += 1
    print("=" * 50)
    if failed == 0:
        print(f"  全 {len(tests)} テスト PASS")
    else:
        print(f"  {failed}/{len(tests)} テスト FAIL")
    return failed


if __name__ == "__main__":
    sys.exit(run_all())
