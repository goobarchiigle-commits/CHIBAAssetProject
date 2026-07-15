"""Tests for src/backtest/study76_clenow_benchmark_wf.py (Study76 Clenow engine).

方針: 実データ・実Universe・実バックテストは行わない（Study75未完了のため禁止）。
      合成データのみでエンジンの機械としての正しさ（ロジック）を検証する。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest import study76_clenow_benchmark_wf as s76


def _make_price_df(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    close = pd.Series(close, index=index)
    open_ = close.shift(1).fillna(close.iloc[0])
    return pd.DataFrame({
        "Open": open_, "High": close * 1.01, "Low": close * 0.99, "Close": close,
        "Volume": 100_000,
    })


class TestCalcClenowScore:
    def test_smooth_uptrend_positive_score(self):
        closes = 100.0 * np.exp(np.linspace(0, 0.3, 90))  # 滑らかな上昇
        score = s76.calc_clenow_score(closes)
        assert score > 0

    def test_smooth_downtrend_negative_score(self):
        closes = 100.0 * np.exp(np.linspace(0, -0.3, 90))
        score = s76.calc_clenow_score(closes)
        assert score < 0

    def test_flat_series_near_zero_score(self):
        closes = np.full(90, 100.0)
        score = s76.calc_clenow_score(closes)
        assert abs(score) < 1e-6

    def test_short_series_returns_nan(self):
        score = s76.calc_clenow_score(np.array([100.0]))
        assert np.isnan(score)

    def test_negative_price_returns_nan(self):
        closes = np.concatenate([np.full(10, 100.0), np.array([-5.0])])
        score = s76.calc_clenow_score(closes)
        assert np.isnan(score)

    def test_smoother_uptrend_scores_higher_than_noisy(self):
        smooth = 100.0 * np.exp(np.linspace(0, 0.2, 90))
        rng = np.random.default_rng(42)
        noisy = smooth * (1 + rng.normal(0, 0.05, size=90))
        assert s76.calc_clenow_score(smooth) > s76.calc_clenow_score(noisy)


class TestUniverseHelpers:
    def test_active_universe_picks_latest_applicable_key(self):
        monthly = {
            "2020-01-01": ["A.T", "B.T"],
            "2020-03-01": ["A.T", "C.T"],
        }
        result = s76.active_universe_for_date(monthly, pd.Timestamp("2020-02-15"))
        assert result == ["A.T", "B.T"]

        result2 = s76.active_universe_for_date(monthly, pd.Timestamp("2020-04-01"))
        assert result2 == ["A.T", "C.T"]

    def test_active_universe_before_any_key_returns_empty(self):
        monthly = {"2020-03-01": ["A.T"]}
        result = s76.active_universe_for_date(monthly, pd.Timestamp("2020-01-01"))
        assert result == []

    def test_build_daily_active_matrix_reflects_monthly_transitions(self):
        idx = pd.bdate_range("2020-01-01", periods=80)
        monthly = {"2020-01-01": ["A.T"], "2020-03-01": ["B.T"]}
        mat = s76.build_daily_active_matrix(monthly, ["A.T", "B.T"], idx)
        early = idx[5]
        late = idx[-1]
        assert mat.loc[early, "A.T"] == 1
        assert mat.loc[early, "B.T"] == 0
        assert mat.loc[late, "B.T"] == 1

    def test_weekly_rebalance_dates_one_per_calendar_week(self):
        idx = pd.bdate_range("2020-01-01", periods=30)
        dates = s76.weekly_rebalance_dates(idx)
        # ISO週ごとに1つ、かつ元のインデックス内の日付のみ
        assert len(dates) == len(set(pd.Timestamp(d).isocalendar()[:2] for d in dates))
        assert all(d in idx for d in dates)


class TestTurtleBreakoutMask:
    def test_breakout_flagged_after_new_high(self):
        idx = pd.bdate_range("2020-01-01", periods=30)
        # 横ばい100 → 25日目に単発急伸(150)。shift(1)済みの10日ローリング高値は
        # 急伸直後の数日間は横ばい期(~101)のままのため、ジャンプ日にブレイクアウトが立つ。
        closes = np.concatenate([np.full(25, 100.0), np.full(5, 150.0)])
        panel = {"A.T": _make_price_df(idx, closes)}
        mask = s76.compute_turtle_breakout_mask(panel, ["A.T"], lookback=10)
        assert bool(mask["A.T"].iloc[25]) is True


class TestDecisionLogic:
    @pytest.mark.parametrize("delta,expected", [
        (-1.0, "PASS"), (-2.0, "PASS"), (-2.01, "INCONCLUSIVE"),
        (-3.99, "INCONCLUSIVE"), (-4.0, "INCONCLUSIVE"), (-4.01, "FAIL"), (-5.0, "FAIL"), (5.0, "PASS"),
    ])
    def test_decide_arm_boundaries(self, delta, expected):
        assert s76.decide_arm(delta) == expected

    def test_decide_overall_one_pass_is_overall_pass(self):
        decision, winner = s76.decide_overall({"TURTLE_ON": "PASS", "TURTLE_OFF": "FAIL"})
        assert decision == "PASS"
        assert winner == "TURTLE_ON"

    def test_decide_overall_both_fail_is_overall_fail(self):
        decision, winner = s76.decide_overall({"TURTLE_ON": "FAIL", "TURTLE_OFF": "FAIL"})
        assert decision == "FAIL"
        assert winner is None

    def test_decide_overall_mixed_no_pass_is_inconclusive(self):
        decision, winner = s76.decide_overall({"TURTLE_ON": "INCONCLUSIVE", "TURTLE_OFF": "FAIL"})
        assert decision == "INCONCLUSIVE"
        assert winner is None


class TestRunClenowEngineSmoke:
    """合成データでエンジン全体（週次回転・ランク脱落Exit・NAVマーキング）が例外なく動くことを確認。"""

    def _build_synthetic(self, n_days: int = 260):
        idx = pd.bdate_range("2020-01-01", periods=n_days)
        rng = np.random.default_rng(7)

        # A: 一貫した強い上昇 → 常に上位ランク
        a = 100.0 * np.exp(np.linspace(0, 0.6, n_days))
        # B: 前半上昇・後半下落 → 途中でランク脱落するはず
        b = np.concatenate([
            100.0 * np.exp(np.linspace(0, 0.4, n_days // 2)),
            100.0 * np.exp(np.linspace(0.4, 0.0, n_days - n_days // 2)),
        ])
        # C: 前半横ばい・後半急伸 → 途中からランクイン
        c = np.concatenate([
            np.full(n_days // 2, 100.0) * (1 + rng.normal(0, 0.01, n_days // 2)),
            100.0 * np.exp(np.linspace(0, 0.5, n_days - n_days // 2)),
        ])
        # D: 一貫した下落
        d = 100.0 * np.exp(np.linspace(0, -0.3, n_days))

        panel = {
            "A.T": _make_price_df(idx, a), "B.T": _make_price_df(idx, b),
            "C.T": _make_price_df(idx, c), "D.T": _make_price_df(idx, d),
        }
        monthly_universe = {idx[0].strftime("%Y-%m-%d"): ["A.T", "B.T", "C.T", "D.T"]}
        topix = pd.Series(100.0 * np.exp(np.linspace(0, 0.2, n_days)), index=idx)  # 一貫した強気
        return panel, monthly_universe, topix, idx

    def test_engine_runs_and_produces_valid_equity_curve(self):
        panel, monthly_universe, topix, idx = self._build_synthetic()
        result = s76.run_clenow_engine(
            panel, monthly_universe, topix, turtle_arm=False,
            start=idx[0].strftime("%Y-%m-%d"), end=idx[-1].strftime("%Y-%m-%d"),
            capital=3_000_000, max_positions=2, rank_lookback=30, turtle_lookback=10,
        )
        assert len(result["equity_curve"]) == len(idx)
        assert all(v > 0 for v in result["equity_curve"])
        assert len(result["trades"]) > 0  # 少なくとも初回エントリーは発生する

    def test_engine_with_turtle_arm_runs_without_error(self):
        panel, monthly_universe, topix, idx = self._build_synthetic()
        result = s76.run_clenow_engine(
            panel, monthly_universe, topix, turtle_arm=True,
            start=idx[0].strftime("%Y-%m-%d"), end=idx[-1].strftime("%Y-%m-%d"),
            capital=3_000_000, max_positions=2, rank_lookback=30, turtle_lookback=10,
        )
        assert len(result["equity_curve"]) == len(idx)

    def test_metrics_extraction_matches_capital_allocation_abc(self):
        panel, monthly_universe, topix, idx = self._build_synthetic()
        result = s76.run_clenow_engine(
            panel, monthly_universe, topix, turtle_arm=False,
            start=idx[0].strftime("%Y-%m-%d"), end=idx[-1].strftime("%Y-%m-%d"),
            capital=3_000_000, max_positions=2, rank_lookback=30, turtle_lookback=10,
        )
        metrics = s76.extract_engine_metrics(result, 3_000_000)
        assert "cagr" in metrics and "sharpe" in metrics and "max_dd" in metrics
        assert isinstance(metrics["data_gap_count"], int)


class TestUniverseFileLoading:
    def test_missing_file_raises_filenotfound(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            s76.load_study75_rule_universe(tmp_path / "does_not_exist.json")

    def test_valid_file_loads_monthly_universe(self, tmp_path):
        import json
        path = tmp_path / "study75.json"
        path.write_text(json.dumps({"monthly_universe": {"2020-01-01": ["A.T"]}}), encoding="utf-8")
        result = s76.load_study75_rule_universe(path)
        assert result == {"2020-01-01": ["A.T"]}

    def test_empty_monthly_universe_raises_valueerror(self, tmp_path):
        import json
        path = tmp_path / "study75_empty.json"
        path.write_text(json.dumps({"monthly_universe": {}}), encoding="utf-8")
        with pytest.raises(ValueError):
            s76.load_study75_rule_universe(path)
