"""
tests/analytics/test_build_opportunity_records.py

EVS完全性修正（2026-07-08）の回帰テスト。
対象: build_opportunity_records() — run_live_signal.py / run_morning_signal.py
両方から呼ばれる唯一の正しいEVSレコード生成経路。
"""
import pytest

from src.analytics.executed_vs_skipped_expectancy import (
    build_opportunity_records,
    SKIP_POSITION_FULL,
    SKIP_CAPITAL_CONSTRAINT,
    SKIP_SECTOR_EXPOSURE,
    SKIP_RANKING_CUTOFF,
    SKIP_UNKNOWN,
)


def _sig(symbol, rsr=80.0, rank=1, holding=False, signal=1):
    return {
        "symbol": symbol, "signal": signal, "currently_holding": holding,
        "rsr": rsr, "rsr_rank": rank, "sector": "テスト",
    }


class TestExecutedDetection:
    """executed は send_results の実約定成功のみを見る（result.ordersではない）。"""

    def test_live_mode_executed_true_on_real_success(self):
        signals = [_sig("7203.T")]
        send_results = [{"symbol": "7203.T", "side": "BUY", "success": True}]
        recs = build_opportunity_records(
            signals=signals, stage_audit=[], send_results=send_results,
            run_id="run1", run_timestamp="2026-06-23T08:41:18+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert len(recs) == 1
        assert recs[0].executed is True
        assert recs[0].skip_reason is None
        assert recs[0].final_stage == "ORDER_SENT"

    def test_live_mode_executed_false_when_send_failed(self):
        signals = [_sig("7203.T")]
        send_results = [{"symbol": "7203.T", "side": "BUY", "success": False}]
        recs = build_opportunity_records(
            signals=signals, stage_audit=[], send_results=send_results,
            run_id="run1", run_timestamp="2026-06-23T08:41:18+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].executed is False

    def test_dry_mode_never_executed_even_if_symbol_would_have_been_bought(self):
        """DRYは絶対に約定しない — result.ordersに候補があってもexecuted=Falseで固定。"""
        signals = [_sig("7203.T")]
        recs = build_opportunity_records(
            signals=signals, stage_audit=[{"symbol": "7203.T", "stage": "ORDER_BUILT", "passed": True}],
            send_results=[],  # DRYは常に空
            run_id="run1", run_timestamp="2026-06-23T05:41:00+0900", mode="DRY",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].executed is False
        assert recs[0].mode == "DRY"


class TestStageAttribution:
    """final_stage/skip_reasonはstage_auditの「最初にFAILしたstage」から決まる。"""

    def test_capacity_fail_maps_to_position_full(self):
        signals = [_sig("8035.T", rank=2)]
        stage_audit = [{"symbol": "8035.T", "stage": "CAPACITY", "passed": False, "reason": "position_full"}]
        recs = build_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="r", run_timestamp="2026-06-29T08:44:16+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].final_stage == "CAPACITY"
        assert recs[0].skip_reason == SKIP_POSITION_FULL

    def test_capital_fail_maps_to_capital_constraint(self):
        signals = [_sig("8035.T")]
        stage_audit = [
            {"symbol": "8035.T", "stage": "CAPACITY", "passed": True},
            {"symbol": "8035.T", "stage": "CAPITAL", "passed": False, "reason": "alloc_cap_exceeded"},
        ]
        recs = build_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="r", run_timestamp="2026-06-29T08:44:16+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].final_stage == "CAPITAL"
        assert recs[0].skip_reason == SKIP_CAPITAL_CONSTRAINT

    def test_first_fail_wins_not_last(self):
        """CAPACITYで既に落ちていればCAPITALは評価されていない — 最初のFAILのみ採用。"""
        signals = [_sig("8035.T")]
        stage_audit = [
            {"symbol": "8035.T", "stage": "CAPACITY", "passed": False, "reason": "position_full"},
        ]
        recs = build_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="r", run_timestamp="2026-06-29T08:44:16+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].final_stage == "CAPACITY"

    def test_ranking_cutoff(self):
        signals = [_sig("6645.T", rank=14)]
        stage_audit = [{"symbol": "6645.T", "stage": "RANKING", "passed": False, "reason": "below_top_k_cutoff"}]
        recs = build_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="r", run_timestamp="2026-06-29T08:44:16+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].skip_reason == SKIP_RANKING_CUTOFF

    def test_sector_concentration(self):
        signals = [_sig("2802.T")]
        stage_audit = [{"symbol": "2802.T", "stage": "SECTOR_CONCENTRATION", "passed": False, "reason": "x"}]
        recs = build_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="r", run_timestamp="2026-06-29T08:44:16+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].skip_reason == SKIP_SECTOR_EXPOSURE

    def test_no_stage_audit_entry_is_untracked_not_crash(self):
        signals = [_sig("9999.T")]
        recs = build_opportunity_records(
            signals=signals, stage_audit=[], send_results=[],
            run_id="r", run_timestamp="2026-06-29T08:44:16+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].final_stage == "UNTRACKED"
        assert recs[0].skip_reason == SKIP_UNKNOWN


class TestRunIdDisambiguation:
    """同一(date,symbol,executed,skip_reason)でもrun_idが違えばopportunity_idも違う
    （2026-06-23 2802.T: 08:41 skip / 08:44 別runの再発防止）。"""

    def test_different_run_id_yields_different_opportunity_id(self):
        signals = [_sig("2802.T")]
        stage_audit = [{"symbol": "2802.T", "stage": "CAPACITY", "passed": False, "reason": "position_full"}]
        recs1 = build_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="run_A", run_timestamp="2026-06-23T05:41:37+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        recs2 = build_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=[],
            run_id="run_B", run_timestamp="2026-06-23T08:41:16+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs1[0].opportunity_id != recs2[0].opportunity_id
        assert recs1[0].run_id == "run_A"
        assert recs2[0].run_id == "run_B"

    def test_2026_06_23_scenario_2802_correctly_executed_true_in_later_run(self):
        """再現: 08:41のrunではskip、後続runで実際に約定 → LIVEレコードは正しくexecuted=True。"""
        signals = [_sig("2802.T", rsr=82.5, rank=9)]
        stage_audit = [{"symbol": "2802.T", "stage": "ORDER_BUILT", "passed": True}]
        send_results = [{"symbol": "2802.T", "side": "BUY", "success": True}]
        recs = build_opportunity_records(
            signals=signals, stage_audit=stage_audit, send_results=send_results,
            run_id="run_executed", run_timestamp="2026-06-23T08:41:18+0900", mode="LIVE",
            source_script="run_live_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].executed is True, "実際に約定したrunのレコードはexecuted=Trueでなければならない"


class TestSourceScriptTagging:
    def test_source_script_recorded(self):
        signals = [_sig("7203.T")]
        recs = build_opportunity_records(
            signals=signals, stage_audit=[], send_results=[],
            run_id="r", run_timestamp="2026-06-23T05:00:00+0900", mode="DRY",
            source_script="run_morning_signal.py", capital_available_pct=0.5,
            portfolio_heat=0.0, market_regime="unknown", max_positions=3,
        )
        assert recs[0].source_script == "run_morning_signal.py"


class TestBackwardCompatibility:
    def test_v1_record_from_dict_still_loads(self):
        from src.analytics.executed_vs_skipped_expectancy import TradeOpportunityRecord
        v1_dict = {
            "opportunity_id": "abc", "schema_version": 1, "eval_date": "2026-06-23",
            "symbol": "2802.T", "executed": False, "skip_reason": "slot_full",
            "capital_available_pct": 0.5, "portfolio_heat": 0.0, "slot_utilization": 0.667,
            "sector": "食品", "market_regime": "unknown", "atr_pct": 0.02, "rs_rank": 14,
            "entry_score": 0.786, "liquidity_score": 0.5, "position_lifecycle_available": False,
        }
        rec = TradeOpportunityRecord.from_dict(v1_dict)
        assert rec.run_id == ""
        assert rec.mode == "unknown"
        assert rec.symbol == "2802.T"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
