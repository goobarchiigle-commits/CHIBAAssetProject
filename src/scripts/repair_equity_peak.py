"""
src/scripts/repair_equity_peak.py
logs/equity_snapshots.jsonl から equity_peak を再構築するオフライン修復スクリプト
（EQUITY_PEAK_HARDENING, 2026-07-03）。

【使い方】
  # レポートのみ（既定、書き込みなし）
  python src/scripts/repair_equity_peak.py

  # 実際に portfolio_state.json へ反映
  python src/scripts/repair_equity_peak.py --apply

  # 確認プロンプトを省略
  python src/scripts/repair_equity_peak.py --apply --force

【前提】
  - equity_peak は本来 _update_cb_state() (src/kabusapi/signal_bridge.py) 経由でのみ
    更新される。本スクリプトは state ファイル破損時の手動復旧専用であり、
    通常運用のフローには組み込まれない。
  - デフォルトは dry-run。peak 修復は DD/CB 判定の基盤を書き換える高リスク操作のため、
    sync_positions.py のような即時書込みは行わない。
"""

import sys
import json
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))   # project root
sys.stdout.reconfigure(encoding="utf-8")

import os
import logging

from src.paths import RUNTIME_DIR, LOGS_DIR
from src.portfolio.state_store import load_portfolio_state, save_portfolio_state
from src.portfolio.equity import rebuild_equity_peak, append_peak_audit

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("repair_equity_peak")

DEFAULT_STATE_FILE    = RUNTIME_DIR / "portfolio_state.json"
DEFAULT_SNAPSHOT_FILE = LOGS_DIR / "equity_snapshots.jsonl"
_RUN_ID = datetime.now(JST).strftime("%Y%m%d_%H%M%S")


def _load_snapshot_equities(snapshot_file: Path) -> list[tuple[str, float]]:
    """(timestamp, equity) のリストを時系列順で返す。壊れた行は無視する。"""
    if not snapshot_file.exists():
        raise FileNotFoundError(f"snapshot file が存在しない: {snapshot_file}")
    out: list[tuple[str, float]] = []
    for raw in snapshot_file.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            d  = json.loads(raw)
            eq = float(d.get("equity", 0))
            ts = str(d.get("timestamp", ""))
            if eq > 0:
                out.append((ts, eq))
        except Exception:
            continue
    return out


def _compute_new_peak(
    entries: list[tuple[str, float]], last_equity: float, method: str,
    n_entries: int, snapshot_file: Path,
) -> float:
    if method == "current":
        # Study96 EquityPeak SSOT Root Cause Audit (2026-07-18): 過去履歴を一切
        # 信用せず、直近のsnapshot（=直近runのbroker equity）のみを新しい
        # equity_peakとする「本日をDay0」リセット専用。median/maxのような
        # 過去の汚染された高値の影響を一切受けない。
        return last_equity
    if method == "max":
        return max(eq for _, eq in entries) if entries else last_equity
    # method == "median" — 既存 rebuild_equity_peak() のロジックを再利用
    return rebuild_equity_peak(
        current_equity=last_equity, n_entries=n_entries, snapshot_file=snapshot_file,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="logs/equity_snapshots.jsonl から equity_peak を再構築する（既定: dry-run）"
    )
    parser.add_argument("--state-file",    type=Path, default=DEFAULT_STATE_FILE)
    parser.add_argument("--snapshot-file", type=Path, default=DEFAULT_SNAPSHOT_FILE)
    parser.add_argument("--n-entries", type=int, default=0, help="0 = 全件使用（既定）")
    parser.add_argument("--method", choices=["median", "max", "current"], default="median")
    parser.add_argument("--apply", action="store_true", help="実際に書き込む（既定は dry-run）")
    parser.add_argument("--force", action="store_true", help="--apply 時の確認プロンプトを省略")
    parser.add_argument("--json", action="store_true", help="機械可読なJSONサマリも出力する")
    args = parser.parse_args()

    try:
        entries = _load_snapshot_equities(args.snapshot_file)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    if not entries:
        logger.error(f"snapshot file にequity>0のエントリが1件もない: {args.snapshot_file}")
        return 1

    last_ts, last_equity = entries[-1]
    n_used = args.n_entries if args.n_entries > 0 else len(entries)
    new_peak = _compute_new_peak(entries, last_equity, args.method, n_used, args.snapshot_file)

    try:
        state, vr = load_portfolio_state(args.state_file)
    except Exception as exc:
        logger.error(f"state ファイル読み込み失敗: {exc}")
        return 1

    old_peak = float(state.get("equity_peak", 0))
    delta     = new_peak - old_peak
    delta_pct = (delta / old_peak * 100) if old_peak > 0 else 0.0

    print("=" * 60)
    print("equity_peak 再構築レポート")
    print("=" * 60)
    print(f"  snapshot file : {args.snapshot_file}")
    print(f"  対象エントリ数 : {n_used} / 全{len(entries)}件")
    print(f"  期間          : {entries[max(0, len(entries) - n_used)][0]} 〜 {last_ts}")
    print(f"  方式          : {args.method}")
    print(f"  現在の peak   : ¥{old_peak:,.0f}")
    print(f"  再構築後 peak : ¥{new_peak:,.0f}")
    print(f"  差分          : ¥{delta:,.0f} ({delta_pct:+.1f}%)")
    print(f"  最新 equity   : ¥{last_equity:,.0f} ({last_ts})")
    print("=" * 60)

    if args.json:
        print(json.dumps({
            "old_peak": round(old_peak, 0), "new_peak": round(new_peak, 0),
            "delta": round(delta, 0), "delta_pct": round(delta_pct, 2),
            "n_entries": n_used, "method": args.method, "applied": False,
        }, ensure_ascii=False))

    if not args.apply:
        print("\nDry run — 書き込みは行われていません。反映するには --apply を付けて再実行してください。")
        return 0

    if not args.force:
        ans = input(
            f"\nportfolio_state.json の equity_peak を "
            f"¥{old_peak:,.0f} → ¥{new_peak:,.0f} に上書きしますか？ [y/N] > "
        ).strip().lower()
        if ans != "y":
            print("中止しました。")
            return 2

    _had_candidate = state.get("candidate_peak") is not None
    state["equity_peak"]    = round(new_peak, 0)
    state["candidate_peak"] = None
    save_portfolio_state(state, path=args.state_file, data_source="repair_equity_peak")

    append_peak_audit(
        action="APPLIED", old_peak=old_peak, new_peak=new_peak, current_equity=last_equity,
        broker_equity=None, caller="repair_equity_peak.main", reason="manual_repair",
        diag=f"method={args.method} n_entries={n_used} snapshot_file={args.snapshot_file}",
        trading_date=datetime.now(JST).strftime("%Y-%m-%d"), mode="manual",
        pid=os.getpid(), run_id=_RUN_ID,
    )

    if _had_candidate:
        logger.info("candidate_peak をクリアしました（手動修復により無効化）。")
    logger.info(f"equity_peak を ¥{old_peak:,.0f} → ¥{new_peak:,.0f} に反映しました。")

    if args.json:
        print(json.dumps({
            "old_peak": round(old_peak, 0), "new_peak": round(new_peak, 0),
            "delta": round(delta, 0), "delta_pct": round(delta_pct, 2),
            "n_entries": n_used, "method": args.method, "applied": True,
        }, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
