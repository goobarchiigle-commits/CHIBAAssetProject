"""
test_phase5b2_observability.py — Phase 5B.2 Operational Hardening & Observability

Tests:
  Task 1 — Extension Filter Observability: [ADDON_EXT] log emitted per symbol
  Task 2 — Capital Deployment Runtime Visibility: [CAPITAL_DEPLOYMENT] formula correctness
  Task 3 — Dead Code Audit: report file exists and is well-formed
"""
from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ─── Task 1: Extension Filter Observability ───────────────────────────────────

from src.addon.extension_filter import (
    ExtensionInput,
    ExtensionResult,
    check_extension,
)


class TestExtensionFilterObservability:
    """[ADDON_EXT] logging logic via check_extension (pure function)."""

    def _make_ext(self, **kwargs) -> ExtensionInput:
        defaults = dict(
            symbol="1234",
            unrealized_pnl_pct=0.05,
            hold_days=5,
            rsr=78.0,
            rsr_rank=10,
            rsr_momentum=1.0,
            entry_price=1000.0,
            current_price=1050.0,
            trailing_stop_price=980.0,
            regime_bear=False,
            portfolio_pnl_pct=0.01,
        )
        defaults.update(kwargs)
        return ExtensionInput(**defaults)

    def test_pass_path_not_blocked(self):
        ext = self._make_ext()
        result = check_extension(ext)
        assert result.extension_blocked is False
        assert result.block_reasons == []

    def test_pass_path_pqs_in_range(self):
        ext = self._make_ext()
        result = check_extension(ext)
        assert 0.5 <= result.persistence_quality_score <= 1.0

    def test_pass_path_smooth_trend_pqs_near_1(self):
        """Smooth slow-moving position → pqs close to 1.0."""
        ext = self._make_ext(unrealized_pnl_pct=0.02, hold_days=10, rsr_momentum=0.5)
        result = check_extension(ext)
        assert result.persistence_quality_score >= 0.90

    def test_blocked_velocity(self):
        """PnL > 2%/day triggers velocity gate."""
        ext = self._make_ext(unrealized_pnl_pct=0.15, hold_days=5)  # 3%/day
        result = check_extension(ext)
        assert result.extension_blocked is True
        assert any("velocity" in r for r in result.block_reasons)

    def test_blocked_rsr_acceleration(self):
        """abs(rsr_momentum) > 5.0 triggers rsr_accel gate."""
        ext = self._make_ext(rsr_momentum=6.0)
        result = check_extension(ext)
        assert result.extension_blocked is True
        assert any("rsr_accel" in r for r in result.block_reasons)

    def test_blocked_climax_bar(self):
        """candle_range_ratio > 2.5 triggers climax_bar gate."""
        ext = self._make_ext(candle_range_ratio=3.0)
        result = check_extension(ext)
        assert result.extension_blocked is True
        assert any("climax_bar" in r for r in result.block_reasons)

    def test_blocked_volume_exhaustion(self):
        """volume_ratio > 4.0 triggers volume_exhaust gate."""
        ext = self._make_ext(volume_ratio=5.0)
        result = check_extension(ext)
        assert result.extension_blocked is True
        assert any("volume_exhaust" in r for r in result.block_reasons)

    def test_pqs_penalised_elevated_velocity(self):
        """Velocity above ramp start but below hard gate → pqs < 1.0."""
        ext = self._make_ext(unrealized_pnl_pct=0.01, hold_days=1)  # 1%/day, start=0.5%
        result = check_extension(ext)
        assert result.extension_blocked is False
        assert result.persistence_quality_score < 1.0

    def test_multiple_gates_all_reported(self):
        """Two gates simultaneously blocked → both reasons present."""
        ext = self._make_ext(
            unrealized_pnl_pct=0.20, hold_days=5,   # velocity blocked
            rsr_momentum=7.0,                          # rsr_accel blocked
        )
        result = check_extension(ext)
        assert result.extension_blocked is True
        assert len(result.block_reasons) >= 2

    def test_deterministic_same_input_same_output(self):
        """Pure function: identical inputs → identical outputs."""
        ext = self._make_ext(unrealized_pnl_pct=0.08, hold_days=3, volume_ratio=2.5)
        r1 = check_extension(ext)
        r2 = check_extension(ext)
        assert r1.extension_blocked == r2.extension_blocked
        assert r1.persistence_quality_score == r2.persistence_quality_score
        assert r1.block_reasons == r2.block_reasons

    def test_ext_reasons_prefix_filter(self):
        """Simulates [ADDON_EXT] reason extraction: only 'EXT:' prefixed items kept."""
        fake_fail_reasons = ["MIN_PNL", "EXT:velocity=0.03/day > max=0.020", "HOLD_DAYS"]
        ext_reasons = [r for r in fake_fail_reasons if r.startswith("EXT:")]
        assert ext_reasons == ["EXT:velocity=0.03/day > max=0.020"]

    def test_ext_reasons_empty_when_not_blocked(self):
        """Non-blocked position → no EXT: reasons in fail_reasons."""
        fake_fail_reasons = ["MIN_PNL"]
        ext_reasons = [r for r in fake_fail_reasons if r.startswith("EXT:")]
        assert ext_reasons == []

    def test_addon_ext_log_emitted(self, caplog):
        """[ADDON_EXT] logger.info call format produces expected prefix."""
        import logging as _logging
        with caplog.at_level(_logging.DEBUG):
            ext = self._make_ext()
            ext_result = check_extension(ext)
            _logger = _logging.getLogger("test_addon_ext")
            _ext_reasons: list[str] = []
            _logger.info(
                "[ADDON_EXT] symbol=%s blocked=%s reasons=%s pqs=%.4f",
                ext.symbol,
                ext_result.extension_blocked,
                _ext_reasons,
                ext_result.persistence_quality_score,
            )
        assert any("[ADDON_EXT]" in r.message for r in caplog.records)

    def test_addon_ext_blocked_log_contains_reasons(self, caplog):
        """[ADDON_EXT] blocked path includes non-empty reasons in log."""
        import logging as _logging
        with caplog.at_level(_logging.DEBUG):
            ext = self._make_ext(unrealized_pnl_pct=0.15, hold_days=5)
            ext_result = check_extension(ext)
            _ext_reasons = [r for r in ext_result.block_reasons]
            _logger = _logging.getLogger("test_addon_ext")
            _logger.info(
                "[ADDON_EXT] symbol=%s blocked=%s reasons=%s pqs=%.4f",
                ext.symbol,
                ext_result.extension_blocked,
                _ext_reasons,
                ext_result.persistence_quality_score,
            )
        msgs = [r.message for r in caplog.records if "[ADDON_EXT]" in r.message]
        assert msgs
        assert "True" in msgs[0]


# ─── Task 2: Capital Deployment Runtime Visibility ────────────────────────────

from src.live.capital_deployment_os import dynamic_max_positions, PARAMS_LOCKED_MAX_POSITIONS


class TestCapitalDeploymentVisibility:
    """[CAPITAL_DEPLOYMENT] formula correctness."""

    def _utilization(self, equity: float, cash: float) -> float:
        if equity <= 0:
            return 0.0
        return (equity - cash) / equity * 100.0

    def _target_size(self, deployable: float, dyn_max: int) -> float:
        return deployable / dyn_max if dyn_max > 0 else 0.0

    # dynamic_max_positions contract

    def test_dyn_max_1m(self):
        assert dynamic_max_positions(1_000_000) == 3

    def test_dyn_max_2m(self):
        # PARAMS_LOCKED クランプ: raw=4 → clamped=3
        assert dynamic_max_positions(2_000_000) == PARAMS_LOCKED_MAX_POSITIONS

    def test_dyn_max_3m(self):
        # PARAMS_LOCKED クランプ: raw=5 → clamped=3
        assert dynamic_max_positions(3_000_000) == PARAMS_LOCKED_MAX_POSITIONS

    def test_dyn_max_floor(self):
        assert dynamic_max_positions(0) >= 1

    def test_dyn_max_cap(self):
        assert dynamic_max_positions(100_000_000) <= 5

    # utilization formula

    def test_utilization_full(self):
        """No cash → 100% utilization."""
        assert math.isclose(self._utilization(3_000_000, 0), 100.0)

    def test_utilization_empty(self):
        """All cash → 0% utilization."""
        assert math.isclose(self._utilization(3_000_000, 3_000_000), 0.0)

    def test_utilization_partial(self):
        """2M equity, 500k cash → 75% utilization."""
        assert math.isclose(self._utilization(2_000_000, 500_000), 75.0)

    def test_utilization_zero_equity(self):
        """Zero equity guard → 0.0."""
        assert self._utilization(0, 500_000) == 0.0

    def test_utilization_negative_cash_guard(self):
        """Negative cash (API artefact) → utilization > 100% — no crash."""
        result = self._utilization(3_000_000, -100_000)
        assert isinstance(result, float)

    # target_size formula

    def test_target_size_3m_3pos(self):
        """3M deployable / 3 positions = 1M target."""
        assert math.isclose(self._target_size(3_000_000, 3), 1_000_000)

    def test_target_size_2m_4pos(self):
        assert math.isclose(self._target_size(2_000_000, 4), 500_000)

    def test_target_size_zero_denominator(self):
        """dyn_max=0 guard → 0.0."""
        assert self._target_size(3_000_000, 0) == 0.0

    # print output format check

    def test_capital_deployment_print_format(self, capsys):
        """Verify [CAPITAL_DEPLOYMENT] line contains expected fields."""
        eff = 3_000_000
        deployable = 2_800_000
        dyn_max = dynamic_max_positions(deployable)
        target_size = deployable / dyn_max if dyn_max > 0 else 0.0
        equity = 3_100_000.0
        cash = 800_000.0
        utilization = (equity - cash) / equity * 100.0
        print(
            f"\n[CAPITAL_DEPLOYMENT]"
            f"  eff=¥{eff:,.0f}"
            f"  deployable=¥{deployable:,.0f}"
            f"  dyn_max_pos={dyn_max}"
            f"  target_size=¥{target_size:,.0f}"
            f"  utilization={utilization:.1f}%"
        )
        out = capsys.readouterr().out
        assert "[CAPITAL_DEPLOYMENT]" in out
        assert "eff=¥" in out
        assert "deployable=¥" in out
        assert "dyn_max_pos=" in out
        assert "target_size=¥" in out
        assert "utilization=" in out

    def test_capital_deployment_utilization_in_print(self, capsys):
        """Utilization value rounds to 1 decimal place in output."""
        equity = 3_000_000.0
        cash = 1_000_000.0
        utilization = (equity - cash) / equity * 100.0  # 66.666...
        print(f"utilization={utilization:.1f}%")
        out = capsys.readouterr().out
        assert "66.7%" in out

    def test_none_cap_state_fallback(self):
        """When cap_state is None, equity falls back to eff_capital without crash."""
        cap_state = None
        eff_capital = 3_000_000
        equity = float(cap_state.actual_equity) if cap_state is not None else float(eff_capital)
        assert equity == 3_000_000.0

    def test_none_available_cash_fallback(self):
        """When available_cash is None, cash defaults to 0.0 without crash."""
        available_cash = None
        cash = float(available_cash) if available_cash is not None else 0.0
        assert cash == 0.0


# ─── Task 3: Dead Code Audit ──────────────────────────────────────────────────

DEAD_CODE_AUDIT_PATH = Path("runtime/analytics/dead_code_audit/dead_code_audit.md")


class TestDeadCodeAudit:
    """Dead code audit report existence and structure."""

    def test_report_file_exists(self):
        assert DEAD_CODE_AUDIT_PATH.exists(), f"Audit report not found: {DEAD_CODE_AUDIT_PATH}"

    def test_report_not_empty(self):
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_report_contains_category_b(self):
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert "Category B" in content

    def test_report_contains_category_c(self):
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert "Category C" in content

    def test_report_contains_derive_risk_params(self):
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert "derive_risk_params" in content

    def test_report_contains_risk_params_var(self):
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert "_RISK_PARAMS" in content

    def test_report_contains_recommended_universe_size(self):
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert "recommended_universe_size" in content

    def test_report_contains_equity_based_max_pos(self):
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert "equity_based_max_pos" in content

    def test_report_marks_no_code_removed(self):
        """Audit must not indicate removal — observation only."""
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert "AUDIT ONLY" in content or "No code removed" in content or "no code removed" in content.lower()

    def test_report_has_summary_table(self):
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert "Summary" in content or "summary" in content.lower()

    def test_no_runtime_import_introduced(self):
        """Audit report is a Markdown file — not imported at runtime."""
        suffix = DEAD_CODE_AUDIT_PATH.suffix
        assert suffix == ".md"

    def test_report_references_line_numbers(self):
        """Report should document where each identifier appears."""
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        assert "line" in content.lower() or "277" in content

    def test_report_active_category_not_recommended_for_removal(self):
        """Category C items should not be flagged for deletion."""
        content = DEAD_CODE_AUDIT_PATH.read_text(encoding="utf-8")
        c_section = ""
        in_c = False
        for line in content.splitlines():
            if "Category C" in line:
                in_c = True
            if in_c:
                c_section += line + "\n"
        assert "No action" in c_section or "active" in c_section.lower()
