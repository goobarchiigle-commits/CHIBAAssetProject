"""
tests/analytics/test_metrics_schema_integrity.py

2026-07-08 Analytics層フィールド名統一の回帰テスト。

背景: analyze_opportunity_capture()（前回修正済み）以外にも
analyze_capital_efficiency() と _generate_charts() が
signals_blocked_rsr/signals_blocked_breakout という
logs/diagnostics/metrics.jsonl に実在しないキーを参照しており、
常に0扱いになっていた（total_rsr_blocked_30d, capital_fragmentation_score,
シナリオ推定値, シグナルフローチャートが機能不全）。

このテストは:
  1. 修正後の analyze_capital_efficiency() が正しいキーから非ゼロ値を
     計算できることを検証する。
  2. ソースコード全体に signals_blocked_rsr/signals_blocked_breakout の
     "読み取り" が残っていないことを静的に保証する（将来の再発防止）。
  3. Opportunity Capture(Section1) と Capital Efficiency(Section4) が
     同一の30日ウィンドウで同一のRSRブロック件数を報告する
     （2つの独立した集計経路が整合すること）。
"""
import re
from datetime import date, timedelta
from pathlib import Path
from typing import List

import pytest

from src.analytics.weekly_market_intelligence import (
    analyze_capital_efficiency,
    analyze_opportunity_capture,
)

_ROOT = Path(__file__).resolve().parents[2]
WEEK_END = date(2026, 6, 29)
WEEK_START = WEEK_END - timedelta(days=6)


def _date_range(start: date, end: date) -> List[str]:
    out = []
    cur = start
    while cur <= end:
        out.append(cur.isoformat())
        cur += timedelta(days=1)
    return out


def _make_real_schema_metric(date_str: str, blocked_by_rsr=5, blocked_by_breakout=2,
                              candidate_count=1, cash_ratio=0.6, exposure=0.4,
                              positions=1, raw_buy_count=8) -> dict:
    """logs/diagnostics/metrics.jsonl の実キー名のみを使う（signal_bridge.py
    の書き込みコードで確認済みのキー — 旧キー名は一切含めない）。"""
    return {
        "date": date_str,
        "run_at": f"{date_str}T09:00:00+0900",
        "cash_ratio": cash_ratio,
        "exposure": exposure,
        "positions": positions,
        "candidate_count": candidate_count,
        "blocked_by_rsr": blocked_by_rsr,
        "blocked_by_breakout": blocked_by_breakout,
        "raw_buy_count": raw_buy_count,
        "universe_size": 42,
    }


class TestCapitalEfficiencyUsesRealSchema:
    def test_total_rsr_blocked_nonzero_with_real_schema(self):
        metrics = [
            _make_real_schema_metric(d, blocked_by_rsr=10)
            for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END)
        ]
        result = analyze_capital_efficiency(metrics, WEEK_END)
        assert result.total_rsr_blocked_30d == 300  # 10 * 30日
        assert result.total_rsr_blocked_30d > 0

    def test_fragmentation_score_nonzero_when_blocked(self):
        metrics = [
            _make_real_schema_metric(d, blocked_by_rsr=20, cash_ratio=0.8)
            for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END)
        ]
        result = analyze_capital_efficiency(metrics, WEEK_END)
        assert result.capital_fragmentation_score > 0

    def test_scenario_gains_nonzero_when_blocked(self):
        metrics = [
            _make_real_schema_metric(d, blocked_by_rsr=15, blocked_by_breakout=5)
            for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END)
        ]
        result = analyze_capital_efficiency(metrics, WEEK_END)
        assert result.scenario_50pct_capture_gain > 0
        assert result.scenario_100pct_capture_gain > result.scenario_50pct_capture_gain


class TestOpportunityCaptureAndCapitalEfficiencyAgree:
    """Section1とSection4が同一ソースから同一のブロック件数を報告すること。"""

    def test_rsr_blocked_totals_match_across_sections(self):
        metrics = [
            _make_real_schema_metric(d, blocked_by_rsr=7, blocked_by_breakout=3)
            for d in _date_range(WEEK_END - timedelta(days=29), WEEK_END)
        ]
        opp = analyze_opportunity_capture(metrics, [], WEEK_START, WEEK_END)
        cap_eff = analyze_capital_efficiency(metrics, WEEK_END)

        assert opp.weekly_rsr_blocked == cap_eff.total_rsr_blocked_30d
        assert opp.weekly_breakout_blocked == cap_eff.total_breakout_blocked_30d


class TestNoStaleFieldNamesRemainInSourceTree:
    """
    静的監査: logs/diagnostics/metrics.jsonl を読む実コードに
    signals_blocked_rsr / signals_blocked_breakout / buy_candidates
    （2026-06-XX以前の誤ったキー名）の "読み取り" が残っていないことを保証する。
    コメント（説明文）内の言及は許容する — .get("...")/["..."] のような
    実際のdictアクセス構文のみを検査対象にする。
    """

    STALE_KEYS = ("signals_blocked_rsr", "signals_blocked_breakout")

    # metrics.jsonl を読む可能性のあるAnalytics層ファイル（Phase1調査で特定済み）
    AUDITED_FILES = (
        "src/analytics/weekly_market_intelligence.py",
        "src/diagnostics/signal_funnel_60d.py",
        "src/scripts/daily_monitor.py",
    )

    def _find_dict_access_lines(self, path: Path, key: str) -> List[str]:
        pattern = re.compile(
            r'\.get\(\s*["\']' + re.escape(key) + r'["\']|\[\s*["\']' + re.escape(key) + r'["\']\s*\]'
        )
        hits = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if pattern.search(line):
                hits.append(line.strip())
        return hits

    def test_no_stale_key_reads_in_audited_files(self):
        violations = {}
        for rel in self.AUDITED_FILES:
            path = _ROOT / rel
            if not path.exists():
                continue
            for key in self.STALE_KEYS:
                hits = self._find_dict_access_lines(path, key)
                if hits:
                    violations[f"{rel}:{key}"] = hits
        assert not violations, f"stale field-name reads found: {violations}"

    def test_metrics_jsonl_writer_schema_contains_correct_keys(self):
        """signal_bridge.py（書き込み側）に blocked_by_rsr/blocked_by_breakout が
        実在することを確認する（読み取り側との突合の基準点）。"""
        sb_path = _ROOT / "src" / "kabusapi" / "signal_bridge.py"
        src = sb_path.read_text(encoding="utf-8")
        assert '"blocked_by_rsr":' in src
        assert '"blocked_by_breakout":' in src
        # 旧キー名は書き込み側にも存在しないこと（読み書き双方でschema一致）
        assert '"signals_blocked_rsr":' not in src
        assert '"signals_blocked_breakout":' not in src


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
