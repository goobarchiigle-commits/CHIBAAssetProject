"""
tests/test_experiment_performance.py — Experiment Performance Tracker (Phase 6J-Plus)
"""
import json
import sys
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analytics.experiment_performance import (
    SCHEMA_VERSION,
    GROUP_CONTROL, GROUP_EXPERIMENT,
    ExperimentOutcome,
    record_experiment_outcome,
    load_performance_records,
    compute_performance_summary,
    format_performance_report,
    _safe_float, _safe_int,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_outcome(
    experiment_id="EXP_A",
    date="2026-05-30",
    symbol="1234.T",
    group=GROUP_CONTROL,
    trades=5,
    win_rate=0.6,
    expectancy=0.03,
    avg_return=0.025,
    max_drawdown=-0.05,
    pnl=150.0,
) -> ExperimentOutcome:
    return ExperimentOutcome(
        experiment_id=experiment_id,
        date=date,
        symbol=symbol,
        group=group,
        trades=trades,
        win_rate=win_rate,
        expectancy=expectancy,
        avg_return=avg_return,
        max_drawdown=max_drawdown,
        pnl=pnl,
    )


def _make_perf_dict(group=GROUP_CONTROL, pnl=100.0, win_rate=0.6, expectancy=0.02,
                    avg_return=0.02, max_drawdown=-0.05, trades=5,
                    experiment_id="EXP_A") -> dict:
    return {
        "experiment_id": experiment_id,
        "group": group,
        "pnl": pnl,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "avg_return": avg_return,
        "max_drawdown": max_drawdown,
        "trades": trades,
    }


# ── TestConstants ──────────────────────────────────────────────────────────────

class TestConstants:
    def test_schema_version(self):
        assert SCHEMA_VERSION == "v1"

    def test_group_control(self):
        assert GROUP_CONTROL == "CONTROL"

    def test_group_experiment(self):
        assert GROUP_EXPERIMENT == "EXPERIMENT"


# ── TestExperimentOutcome ──────────────────────────────────────────────────────

class TestExperimentOutcome:
    def test_creation(self):
        o = _make_outcome()
        assert o.experiment_id == "EXP_A"
        assert o.symbol == "1234.T"
        assert o.group == GROUP_CONTROL

    def test_defaults_schema_version(self):
        o = _make_outcome()
        assert o.schema_version == SCHEMA_VERSION

    def test_metrics_stored(self):
        o = _make_outcome(trades=10, win_rate=0.7, pnl=500.0)
        assert o.trades == 10
        assert o.win_rate == pytest.approx(0.7)
        assert o.pnl == pytest.approx(500.0)

    def test_max_drawdown_stored(self):
        o = _make_outcome(max_drawdown=-0.12)
        assert o.max_drawdown == pytest.approx(-0.12)


# ── TestSafeHelpers ────────────────────────────────────────────────────────────

class TestSafeHelpers:
    def test_safe_float_normal(self):
        assert _safe_float(1.5) == 1.5

    def test_safe_float_none(self):
        assert _safe_float(None) == 0.0

    def test_safe_float_string(self):
        assert _safe_float("bad", 99.0) == 99.0

    def test_safe_int_normal(self):
        assert _safe_int(5) == 5

    def test_safe_int_none(self):
        assert _safe_int(None) == 0


# ── TestRecordExperimentOutcome ────────────────────────────────────────────────

class TestRecordExperimentOutcome:
    def test_creates_file(self, tmp_path):
        p = tmp_path / "perf" / "perf.jsonl"
        record_experiment_outcome(_make_outcome(), p)
        assert p.exists()

    def test_appends_record(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(), p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1

    def test_record_fields_written(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(symbol="5678.T", pnl=200.0), p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["symbol"] == "5678.T"
        assert data["pnl"] == pytest.approx(200.0)

    def test_experiment_id_written(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(experiment_id="EXP_TEST"), p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["experiment_id"] == "EXP_TEST"

    def test_group_written(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(group=GROUP_EXPERIMENT), p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["group"] == GROUP_EXPERIMENT

    def test_schema_version_written(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(), p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["schema_version"] == SCHEMA_VERSION

    def test_multiple_appends(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(symbol="A"), p)
        record_experiment_outcome(_make_outcome(symbol="B"), p)
        lines = [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 2

    def test_all_metric_fields_written(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        o = _make_outcome(trades=7, win_rate=0.71, expectancy=0.04,
                          avg_return=0.03, max_drawdown=-0.08, pnl=300.0)
        record_experiment_outcome(o, p)
        data = json.loads(p.read_text(encoding="utf-8").strip())
        assert data["trades"] == 7
        assert data["win_rate"] == pytest.approx(0.71)
        assert data["max_drawdown"] == pytest.approx(-0.08)


# ── TestLoadPerformanceRecords ─────────────────────────────────────────────────

class TestLoadPerformanceRecords:
    def test_missing_file_returns_empty(self, tmp_path):
        p = tmp_path / "missing.jsonl"
        assert load_performance_records(p) == []

    def test_loads_all_records(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(experiment_id="EXP_A"), p)
        record_experiment_outcome(_make_outcome(experiment_id="EXP_B"), p)
        result = load_performance_records(p)
        assert len(result) == 2

    def test_filter_by_experiment_id(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(experiment_id="EXP_A"), p)
        record_experiment_outcome(_make_outcome(experiment_id="EXP_B"), p)
        result = load_performance_records(p, experiment_id="EXP_A")
        assert len(result) == 1
        assert result[0]["experiment_id"] == "EXP_A"

    def test_no_filter_returns_all(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(experiment_id="EXP_A"), p)
        record_experiment_outcome(_make_outcome(experiment_id="EXP_A"), p)
        result = load_performance_records(p, experiment_id=None)
        assert len(result) == 2

    def test_malformed_lines_skipped(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        p.write_text('{"experiment_id": "A", "group": "CONTROL"}\nnot-json\n',
                     encoding="utf-8")
        result = load_performance_records(p)
        assert len(result) == 1

    def test_blank_lines_skipped(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        p.write_text('{"experiment_id": "A"}\n\n\n', encoding="utf-8")
        result = load_performance_records(p)
        assert len(result) == 1

    def test_filter_returns_empty_when_no_match(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(experiment_id="EXP_A"), p)
        result = load_performance_records(p, experiment_id="EXP_MISSING")
        assert result == []


# ── TestComputePerformanceSummary ──────────────────────────────────────────────

class TestComputePerformanceSummary:
    def test_empty_records_all_zeros(self):
        result = compute_performance_summary([])
        assert result["trades"] == 0
        assert result["win_rate"] == 0.0
        assert result["pnl"] == 0.0

    def test_total_trades_summed(self):
        records = [_make_perf_dict(trades=5), _make_perf_dict(trades=3)]
        result = compute_performance_summary(records)
        assert result["trades"] == 8

    def test_pnl_summed(self):
        records = [_make_perf_dict(pnl=100.0), _make_perf_dict(pnl=200.0)]
        result = compute_performance_summary(records)
        assert result["pnl"] == pytest.approx(300.0)

    def test_avg_return_averaged(self):
        records = [_make_perf_dict(avg_return=0.02), _make_perf_dict(avg_return=0.04)]
        result = compute_performance_summary(records)
        assert result["avg_return"] == pytest.approx(0.03)

    def test_max_drawdown_worst_case(self):
        records = [_make_perf_dict(max_drawdown=-0.05), _make_perf_dict(max_drawdown=-0.15)]
        result = compute_performance_summary(records)
        assert result["max_drawdown"] == pytest.approx(-0.15)

    def test_filter_by_group_control(self):
        records = [
            _make_perf_dict(group=GROUP_CONTROL, pnl=100.0),
            _make_perf_dict(group=GROUP_EXPERIMENT, pnl=200.0),
        ]
        result = compute_performance_summary(records, group=GROUP_CONTROL)
        assert result["pnl"] == pytest.approx(100.0)

    def test_filter_by_group_experiment(self):
        records = [
            _make_perf_dict(group=GROUP_CONTROL, pnl=100.0),
            _make_perf_dict(group=GROUP_EXPERIMENT, pnl=200.0),
        ]
        result = compute_performance_summary(records, group=GROUP_EXPERIMENT)
        assert result["pnl"] == pytest.approx(200.0)

    def test_no_group_filter_all_included(self):
        records = [
            _make_perf_dict(group=GROUP_CONTROL, pnl=100.0),
            _make_perf_dict(group=GROUP_EXPERIMENT, pnl=200.0),
        ]
        result = compute_performance_summary(records, group=None)
        assert result["pnl"] == pytest.approx(300.0)

    def test_win_rate_averaged(self):
        records = [_make_perf_dict(win_rate=0.4), _make_perf_dict(win_rate=0.8)]
        result = compute_performance_summary(records)
        assert result["win_rate"] == pytest.approx(0.6)

    def test_group_filter_no_match_returns_zeros(self):
        records = [_make_perf_dict(group=GROUP_CONTROL)]
        result = compute_performance_summary(records, group=GROUP_EXPERIMENT)
        assert result["trades"] == 0
        assert result["pnl"] == 0.0


# ── TestFormatPerformanceReport ────────────────────────────────────────────────

class TestFormatPerformanceReport:
    def test_empty_records_returns_empty(self):
        assert format_performance_report("EXP_A", []) == ""

    def test_report_contains_header(self):
        records = [_make_perf_dict(group=GROUP_CONTROL, trades=5, pnl=100.0)]
        rpt = format_performance_report("EXP_A", records)
        assert "EXPERIMENT PERFORMANCE" in rpt

    def test_report_contains_experiment_id(self):
        records = [_make_perf_dict(group=GROUP_CONTROL, trades=5, pnl=100.0)]
        rpt = format_performance_report("EXP_UNIQUE", records)
        assert "EXP_UNIQUE" in rpt

    def test_report_contains_control_and_experiment_labels(self):
        records = [
            _make_perf_dict(group=GROUP_CONTROL, pnl=100.0, trades=5),
            _make_perf_dict(group=GROUP_EXPERIMENT, pnl=150.0, trades=5),
        ]
        rpt = format_performance_report("EXP_A", records)
        assert "CONTROL" in rpt
        assert "EXPERIMENT" in rpt

    def test_report_contains_metric_names(self):
        records = [_make_perf_dict(group=GROUP_CONTROL, trades=5, pnl=100.0)]
        rpt = format_performance_report("EXP_A", records)
        assert "pnl" in rpt or "trades" in rpt

    def test_separator_lines_present(self):
        records = [_make_perf_dict(group=GROUP_CONTROL, trades=5, pnl=100.0)]
        rpt = format_performance_report("EXP_A", records)
        assert "═" in rpt


# ── TestIntegration ────────────────────────────────────────────────────────────

class TestIntegration:
    def test_record_then_load_then_summarize(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        for i in range(5):
            record_experiment_outcome(_make_outcome(
                experiment_id="EXP_A", group=GROUP_CONTROL, pnl=100.0 * i, trades=1
            ), p)
        for i in range(3):
            record_experiment_outcome(_make_outcome(
                experiment_id="EXP_A", group=GROUP_EXPERIMENT, pnl=150.0 * i, trades=1
            ), p)

        records = load_performance_records(p, experiment_id="EXP_A")
        assert len(records) == 8

        ctrl_summary = compute_performance_summary(records, group=GROUP_CONTROL)
        exp_summary  = compute_performance_summary(records, group=GROUP_EXPERIMENT)
        assert ctrl_summary["trades"] == 5
        assert exp_summary["trades"] == 3

    def test_format_report_end_to_end(self, tmp_path):
        p = tmp_path / "perf.jsonl"
        record_experiment_outcome(_make_outcome(group=GROUP_CONTROL, trades=10, pnl=500.0), p)
        record_experiment_outcome(_make_outcome(group=GROUP_EXPERIMENT, trades=8, pnl=600.0), p)
        records = load_performance_records(p)
        rpt = format_performance_report("EXP_A", records)
        assert "CONTROL" in rpt
        assert "EXPERIMENT" in rpt

    def test_no_execution_imports(self):
        import src.analytics.experiment_performance as m
        src_text = Path(m.__file__).read_text(encoding="utf-8")
        assert "signal_bridge" not in src_text
        assert "place_order" not in src_text
        assert "kabu" not in src_text.lower()
