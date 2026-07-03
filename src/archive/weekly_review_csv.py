"""
scripts/weekly_review_csv.py
週次シグナルレビュー CSV 自動生成

【出力列】
  date        : シグナル生成日（JST）
  symbol      : 銘柄コード
  signal      : -1=SELL  0=HOLD  1=BUY
  order_sent  : True=実際に発注 / False=DRY_RUN または発注対象外
  api_ok      : kabuステーション API 接続成否（True/False/unknown）
  cb_state    : NORMAL / CB_ACTIVE / RECOVERY

【使い方】
  # 直近7日分をまとめて生成
  cd C:/ai-trading
  python -m scripts.weekly_review_csv

  # 期間指定
  python -m scripts.weekly_review_csv --from 2026-03-28 --to 2026-04-04

  # 出力先指定
  python -m scripts.weekly_review_csv --output reports/review_week.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_HERE = Path(__file__).resolve().parent.parent   # C:/ai-trading/src
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent))            # C:/ai-trading

from src.paths import SIGNALS_DIR

JST = timezone(timedelta(hours=9))

# ── 出力ヘッダー ──────────────────────────────────────────────────────────
COLUMNS = ["date", "symbol", "signal", "order_sent", "api_ok", "cb_state"]


# ─────────────────────────────────────────────────────────────────────────────
# パーサ
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="週次レビュー CSV 生成")
    p.add_argument(
        "--from", dest="date_from",
        default=None,
        help="集計開始日 YYYY-MM-DD（デフォルト: 7日前）",
    )
    p.add_argument(
        "--to", dest="date_to",
        default=None,
        help="集計終了日 YYYY-MM-DD（デフォルト: 今日）",
    )
    p.add_argument(
        "--signal-dir",
        default=str(SIGNALS_DIR),
        help=f"シグナル JSON ディレクトリ（デフォルト: {SIGNALS_DIR}）",
    )
    p.add_argument(
        "--output", "-o",
        default=None,
        help="出力 CSV パス（デフォルト: stdout）",
    )
    p.add_argument(
        "--no-dedup",
        action="store_true",
        help="同一日・銘柄の重複行を除去しない（デフォルト: 除去する）",
    )
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# シグナルファイルのロード
# ─────────────────────────────────────────────────────────────────────────────

def load_signal_files(
    signal_dir: Path,
    date_from:  str,
    date_to:    str,
) -> list[dict]:
    """
    指定期間内のシグナル JSON ファイルを読み込んで返す。

    Args:
        signal_dir: シグナルファイルのディレクトリ
        date_from:  "YYYY-MM-DD"
        date_to:    "YYYY-MM-DD"

    Returns:
        list of parsed JSON dicts（ファイルタイムスタンプ順）
    """
    files = sorted(signal_dir.glob("signal_*.json"))
    results = []

    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"  [WARN] {f.name}: 読み込み失敗 → {exc}", file=sys.stderr)
            continue

        # generated_at から日付を取得（例: "2026-03-31T20:42:04+0900"）
        raw_ts = data.get("generated_at", "")
        try:
            dt = datetime.fromisoformat(raw_ts)
            file_date = dt.date().isoformat()   # "YYYY-MM-DD"
        except Exception:
            # フォールバック: ファイル名から日付取得（signal_YYYYMMDD_HHMMSS.json）
            stem = f.stem  # "signal_20260331_204206"
            parts = stem.split("_")
            if len(parts) >= 2 and len(parts[1]) == 8:
                raw_d = parts[1]
                file_date = f"{raw_d[:4]}-{raw_d[4:6]}-{raw_d[6:8]}"
            else:
                continue

        if file_date < date_from or file_date > date_to:
            continue

        data["_file_date"] = file_date
        data["_filename"]  = f.name
        results.append(data)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# 行生成
# ─────────────────────────────────────────────────────────────────────────────

def extract_rows(data: dict) -> list[dict]:
    """
    1つのシグナルファイルから CSV 行リストを生成する。

    Returns:
        list of dicts matching COLUMNS
    """
    file_date  = data["_file_date"]
    mode       = data.get("mode", "UNKNOWN")          # "DRY_RUN" / "LIVE"
    ps         = data.get("portfolio_summary", {})

    # api_ok: positions_api.ok → wallet_api.ok → unknown
    positions_api = ps.get("positions_api", {})
    wallet_api    = ps.get("wallet_api", {})
    if isinstance(positions_api, dict) and "ok" in positions_api:
        api_ok = str(positions_api["ok"])
    elif isinstance(wallet_api, dict) and "ok" in wallet_api:
        api_ok = str(wallet_api["ok"])
    else:
        api_ok = "unknown"

    cb_state = ps.get("cb_state") or data.get("cb_state") or "NORMAL"

    # send_results: 実際に送信した銘柄セット
    sent_symbols: dict[str, bool] = {}   # symbol → success
    for r in data.get("send_results", []):
        sym = r.get("symbol", "")
        if sym:
            sent_symbols[sym] = r.get("success", False)

    rows = []
    for sig in data.get("signals", []):
        symbol     = sig.get("symbol", "")
        signal_int = sig.get("signal", 0)

        # order_sent: LIVE かつ send_results に存在する場合のみ True
        if mode == "LIVE" and symbol in sent_symbols:
            order_sent = "True"
        else:
            order_sent = "False"

        rows.append({
            "date":       file_date,
            "symbol":     symbol,
            "signal":     str(signal_int),
            "order_sent": order_sent,
            "api_ok":     api_ok,
            "cb_state":   cb_state,
        })

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# 重複除去
# ─────────────────────────────────────────────────────────────────────────────

def dedup_rows(rows: list[dict]) -> list[dict]:
    """
    同一日・同一銘柄のエントリーを1行に集約する。
    複数ファイルに同一銘柄が含まれる場合は最後のファイルの値を採用する
    （ファイルはタイムスタンプ順にソート済みのため「最新」を優先）。
    """
    seen: dict[tuple, dict] = {}
    for row in rows:
        key = (row["date"], row["symbol"])
        seen[key] = row   # 同一キーは後勝ち
    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# メイン
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    args    = parse_args()
    today   = datetime.now(JST).date()
    d_from  = args.date_from or (today - timedelta(days=7)).isoformat()
    d_to    = args.date_to   or today.isoformat()

    signal_dir = Path(args.signal_dir)
    if not signal_dir.exists():
        print(f"[ERROR] シグナルディレクトリが見つかりません: {signal_dir}", file=sys.stderr)
        return 1

    print(f"集計期間  : {d_from} ～ {d_to}", file=sys.stderr)
    print(f"シグナル  : {signal_dir}", file=sys.stderr)

    # ファイル読み込み
    signal_files = load_signal_files(signal_dir, d_from, d_to)
    print(f"対象ファイル: {len(signal_files)} 件", file=sys.stderr)

    # 行生成
    all_rows: list[dict] = []
    for sf in signal_files:
        all_rows.extend(extract_rows(sf))

    # 重複除去
    if not args.no_dedup:
        all_rows = dedup_rows(all_rows)

    # ソート（date → symbol）
    all_rows.sort(key=lambda r: (r["date"], r["symbol"]))

    print(f"出力行数  : {len(all_rows)} 行", file=sys.stderr)

    # 書き出し
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"保存先    : {out_path}", file=sys.stderr)
    else:
        writer = csv.DictWriter(sys.stdout, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    return 0


if __name__ == "__main__":
    sys.exit(main())
