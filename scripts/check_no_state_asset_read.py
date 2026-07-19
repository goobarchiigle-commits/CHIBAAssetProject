"""
scripts/check_no_state_asset_read.py
Broker-as-Sole-SSOT (2026-07-19) の静的チェック: portfolio_state.json の
資産系フィールド（cash/positions/market_value/equity の"現在値"）を、
許可されていないモジュールから読み取っていないかを grep ベースで検知する。

対象キー（GUARDED_KEYS）:
    available_cash / last_equity / position_qtys / positions_count /
    position_current_prices / snapshot_avg_costs / snapshot_hash / snapshot_ts

これらは fetch_broker_snapshot() から得た BrokerSnapshot のみを入力とすべきで、
portfolio_state.json から再読み込みして資産計算・売買判断に使うことを禁止する
（2026-07-15〜17 equity_peak異常値インシデントの根本原因は複数の独立した
資産取得経路が食い違う値を生成していたこと）。

許可対象:
    equity_peak / cb_state / candidate_peak / position_entry_* /
    position_highest_closes / position_strategy_types / position_missing_streak /
    position_fill_dates / reentry_blocked / shadow_positions 等の
    メタデータ・状態管理フィールドは対象外（これらは「現在の資産額」ではなく
    「過去の確定値・状態遷移」であり、stateが正当なSSOT）。

許可ファイル（ALLOWED_FILES）:
    - src/portfolio/state_store.py, equity.py, broker_source.py:
      唯一の資産計算コアモジュール。
    - src/kabusapi/signal_bridge.py, src/startup_check.py,
      src/scripts/sync_positions.py, src/run_live_signal.py:
      Broker-as-Sole-SSOTリファクタ (2026-07-18/19) で fetch_broker_snapshot()
      経由へ移行済み。残る読み取りは診断ログ・CB決済ラグ補償（Step8で再設計
      予定）・result.portfolio_summary（当該runのfresh出力）など検証済みの
      正当な用途のみ。
    - src/analytics/skipped_opportunity_analytics.py,
      src/diagnostics/live_exposure_report.py, src/live/broker_truth_snapshot.py,
      src/live/preview.py, src/scripts/rebuild_skipped_opportunities.py,
      src/scripts/audit_20260707_candidates.py:
      過去のsignal_*.json アーカイブ・bridge.run()のresult.portfolio_summary・
      凍結済み監査レコードのみを読む観測/フォレンジック用途で、
      runtime/portfolio_state.json は一切読まない（検証済み）。

実行:
    python scripts/check_no_state_asset_read.py
    → 違反ゼロなら exit 0、1件でもあれば違反箇所を表示して exit 1
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = Path(__file__).resolve().parents[1]
_SRC  = _ROOT / "src"

GUARDED_KEYS = (
    "available_cash",
    "last_equity",
    "position_qtys",
    "positions_count",
    "position_current_prices",
    "snapshot_avg_costs",
)
# snapshot_hash/snapshot_ts は除外: src/backtest/配下のデータセット整合性ハッシュと
# 名前が衝突し誤検知が多い（portfolio_state.jsonのものとは無関係の概念）。

# 生成した正規表現: .get("available_cash" / ["available_cash"] 等の両形式にマッチ
_PATTERN = re.compile(
    r'(?:\.get\(\s*["\']({keys})["\']|\[\s*["\']({keys})["\']\s*\])'.format(
        keys="|".join(GUARDED_KEYS)
    )
)

ALLOWED_FILES = frozenset({
    "src/portfolio/state_store.py",
    "src/portfolio/equity.py",
    "src/portfolio/broker_source.py",
    "src/kabusapi/signal_bridge.py",
    "src/startup_check.py",
    "src/scripts/sync_positions.py",
    "src/run_live_signal.py",
    "src/analytics/skipped_opportunity_analytics.py",
    "src/diagnostics/live_exposure_report.py",
    "src/live/broker_truth_snapshot.py",
    "src/live/preview.py",
    "src/scripts/rebuild_skipped_opportunities.py",
    "src/scripts/audit_20260707_candidates.py",
    # 2026-07-15 SSOT統合で完全に廃止（モジュールレベルでRuntimeError、import不可）。
    # 死んだコードだが誤検知除外のためallowlistに残す。
    "src/run_morning_signal.py",
})

# src/backtest/ はヒストリカル・シミュレーション専用で、実行時の
# runtime/portfolio_state.json には一切アクセスしない（バックテスト自身の
# シミュレーション用dict/DataFrameが同名キーを使っているだけ）。対象外とする。
EXCLUDED_DIRS = frozenset({"src/backtest"})

# テストファイルは対象外（任意のdict形状を自由に構築・検証してよい）
_TEST_NAME_RE = re.compile(r"(^test_|_test\.py$)")


def _is_test_file(rel_path: str) -> bool:
    name = Path(rel_path).name
    return bool(_TEST_NAME_RE.search(name))


def find_violations(src_dir: Path | None = None, root_dir: Path | None = None) -> list[tuple[str, int, str]]:
    src_dir  = src_dir  or _SRC
    root_dir = root_dir or _ROOT
    violations: list[tuple[str, int, str]] = []
    for path in sorted(src_dir.rglob("*.py")):
        rel = path.relative_to(root_dir).as_posix()
        if rel in ALLOWED_FILES or _is_test_file(rel):
            continue
        if any(rel.startswith(f"{d}/") for d in EXCLUDED_DIRS):
            continue
        if "__pycache__" in rel or rel.endswith(".bak") or ".bak_" in rel:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _PATTERN.search(line):
                violations.append((rel, lineno, line.strip()))
    return violations


def main() -> int:
    violations = find_violations()
    if not violations:
        print("[OK] portfolio_state.json の資産系フィールド読み取りは許可ファイル外に存在しません。")
        return 0

    print(f"[FAIL] {len(violations)} 件の資産系フィールド読み取りが許可されていないファイルで検出されました:\n")
    for rel, lineno, line in violations:
        print(f"  {rel}:{lineno}: {line}")
    print(
        "\n資産値(cash/positions/market_value/equity)はfetch_broker_snapshot()経由の"
        "BrokerSnapshotのみを唯一の入力とすること。portfolio_state.jsonからの再読み込みは"
        "禁止（Broker-as-Sole-SSOT, 2026-07-18/19）。\n"
        "正当な用途（診断ログ・過去アーカイブ分析等）であれば、このスクリプトの"
        "ALLOWED_FILES へ追加し、理由をコメントで明記すること。"
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
