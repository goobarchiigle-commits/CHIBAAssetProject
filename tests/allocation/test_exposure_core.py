"""Tests for src/allocation/exposure_core.py"""
import math
import pytest
from src.allocation.exposure_core import (
    ExposureState,
    PairCorrelation,
    ThemeCluster,
    ExposureReport,
    _pearson,
    _pair_key,
    _effective_n_from_corr,
    _concentration_risk_label,
    _rsr_quartile,
    compute_pair_correlations,
    compute_theme_clusters,
    compute_exposure_report,
    update_exposure_state,
    load_exposure_state,
    save_exposure_state,
    CORR_WINDOW,
    MIN_OBSERVATIONS_FOR_CORR,
    COLLAPSE_THRESHOLD,
    COLLAPSE_STREAK_REQUIRED,
)


# ── _pearson ───────────────────────────────────────────────────────────────────

def test_pearson_perfect_positive():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [2.0, 4.0, 6.0, 8.0, 10.0]
    assert _pearson(xs, ys) == pytest.approx(1.0, abs=1e-4)


def test_pearson_perfect_negative():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [5.0, 4.0, 3.0, 2.0, 1.0]
    assert _pearson(xs, ys) == pytest.approx(-1.0, abs=1e-4)


def test_pearson_zero_correlation():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    ys = [3.0, 3.0, 3.0, 3.0, 3.0]  # constant → undefined variance
    assert _pearson(xs, ys) is None


def test_pearson_insufficient_data():
    xs = [1.0, 2.0]
    ys = [1.0, 2.0]
    assert _pearson(xs, ys) is None  # < MIN_OBSERVATIONS_FOR_CORR=5


def test_pearson_bounded():
    import random
    random.seed(42)
    xs = [random.gauss(0, 1) for _ in range(20)]
    ys = [random.gauss(0, 1) for _ in range(20)]
    r = _pearson(xs, ys)
    if r is not None:
        assert -1.0 <= r <= 1.0


# ── _pair_key ──────────────────────────────────────────────────────────────────

def test_pair_key_canonical():
    assert _pair_key("8035", "7203") == _pair_key("7203", "8035")


def test_pair_key_format():
    key = _pair_key("A", "B")
    assert "|" in key
    assert "A" in key and "B" in key


# ── _effective_n_from_corr ─────────────────────────────────────────────────────

def test_effective_n_no_correlation():
    assert _effective_n_from_corr(3, 0.0) == pytest.approx(3.0, abs=1e-4)


def test_effective_n_full_correlation():
    assert _effective_n_from_corr(3, 1.0) == pytest.approx(1.0, abs=1e-4)


def test_effective_n_half_correlation():
    n = 4
    rho = 0.5
    expected = n / (1 + (n - 1) * rho)
    assert _effective_n_from_corr(n, rho) == pytest.approx(expected, abs=1e-4)


def test_effective_n_single_position():
    assert _effective_n_from_corr(1, 0.8) == pytest.approx(1.0, abs=1e-4)


def test_effective_n_zero_positions():
    assert _effective_n_from_corr(0, 0.5) == 0.0


# ── _rsr_quartile ──────────────────────────────────────────────────────────────

def test_rsr_quartile_top():
    assert _rsr_quartile(1, 42) == 1
    assert _rsr_quartile(10, 42) == 1


def test_rsr_quartile_bottom():
    assert _rsr_quartile(42, 42) == 4
    assert _rsr_quartile(35, 42) == 4


def test_rsr_quartile_middle():
    assert _rsr_quartile(22, 42) == 3  # 22/42 ≈ 0.52


# ── compute_pair_correlations ──────────────────────────────────────────────────

def test_pair_correlations_insufficient_data():
    state = ExposureState(
        return_buffers={"A": [0.01, 0.02], "B": [0.01, 0.02]},
        collapse_streak={},
    )
    pairs = compute_pair_correlations(state, ["A", "B"])
    assert len(pairs) == 1
    assert pairs[0].valid is False


def test_pair_correlations_with_data():
    returns = list(range(1, 21))  # 20 observations
    state = ExposureState(
        return_buffers={
            "A": [float(x) for x in returns],
            "B": [float(x * 2) for x in returns],
        },
        collapse_streak={},
    )
    pairs = compute_pair_correlations(state, ["A", "B"])
    assert len(pairs) == 1
    assert pairs[0].valid is True
    assert pairs[0].correlation == pytest.approx(1.0, abs=1e-4)


def test_pair_correlations_three_symbols():
    state = ExposureState(
        return_buffers={
            "A": [float(i) for i in range(20)],
            "B": [float(i) for i in range(20)],
            "C": [float(i) for i in range(20)],
        },
        collapse_streak={},
    )
    pairs = compute_pair_correlations(state, ["A", "B", "C"])
    assert len(pairs) == 3  # 3 choose 2


# ── compute_theme_clusters ─────────────────────────────────────────────────────

def test_theme_clusters_same_sector():
    sector_map = {"8035": "電気機器", "6857": "電気機器", "7203": "輸送用機器"}
    rsr_rank_map = {"8035": 1, "6857": 2, "7203": 5}
    clusters = compute_theme_clusters(
        ["8035", "6857", "7203"], sector_map, rsr_rank_map, n_universe=42
    )
    sector_clusters = [c for c in clusters if c.cluster_type == "sector"]
    assert len(sector_clusters) == 1
    assert set(sector_clusters[0].symbols) == {"8035", "6857"}


def test_theme_clusters_same_rsr_quartile():
    sector_map = {"8035": "電気機器", "6857": "化学"}
    rsr_rank_map = {"8035": 1, "6857": 2}  # both Q1
    clusters = compute_theme_clusters(
        ["8035", "6857"], sector_map, rsr_rank_map, n_universe=42
    )
    rsr_clusters = [c for c in clusters if c.cluster_type == "rsr_quartile"]
    assert len(rsr_clusters) == 1
    assert rsr_clusters[0].theme_key == "rsr_q:1"


def test_theme_clusters_no_overlap():
    sector_map = {"A": "sec1", "B": "sec2", "C": "sec3"}
    rsr_rank_map = {"A": 1, "B": 15, "C": 30}  # all different quartiles
    clusters = compute_theme_clusters(
        ["A", "B", "C"], sector_map, rsr_rank_map, n_universe=42
    )
    # No clusters since each symbol is unique sector + unique quartile
    assert all(c.overlap_count >= 2 for c in clusters)  # only clusters with ≥2 members listed


# ── compute_exposure_report ────────────────────────────────────────────────────

def test_exposure_report_empty():
    state = ExposureState()
    report = compute_exposure_report(state, [], {}, {})
    assert report.nominal_positions == 0
    assert report.effective_independent_n == 0.0
    assert report.collapse_alert is False


def test_exposure_report_single_position():
    state = ExposureState(return_buffers={"8035": [0.01] * 20})
    report = compute_exposure_report(state, ["8035"], {"8035": "電気機器"}, {"8035": 1})
    assert report.nominal_positions == 1
    assert report.effective_independent_n == pytest.approx(1.0, abs=1e-4)
    assert report.concentration_risk in ("LOW", "MEDIUM", "HIGH")


def test_exposure_report_perfectly_correlated():
    returns = [float(i) for i in range(1, 21)]
    state = ExposureState(
        return_buffers={"A": returns, "B": returns},  # identical → corr=1.0
        collapse_streak={_pair_key("A", "B"): 10},   # streak above required
    )
    sector_map = {"A": "s1", "B": "s2"}
    rsr_rank_map = {"A": 1, "B": 2}
    report = compute_exposure_report(state, ["A", "B"], sector_map, rsr_rank_map)
    assert report.effective_independent_n < 2.0
    assert report.collapse_alert is True
    assert report.concentration_risk in ("MEDIUM", "HIGH")


def test_exposure_report_no_collapse_insufficient_streak():
    returns = [float(i) for i in range(1, 21)]
    state = ExposureState(
        return_buffers={"A": returns, "B": returns},
        collapse_streak={_pair_key("A", "B"): 2},    # below COLLAPSE_STREAK_REQUIRED=5
    )
    report = compute_exposure_report(state, ["A", "B"], {"A": "s1", "B": "s2"}, {"A": 1, "B": 2})
    assert report.collapse_alert is False  # streak not reached


# ── update_exposure_state ──────────────────────────────────────────────────────

def test_update_exposure_state_appends_returns():
    state = ExposureState()
    state2 = update_exposure_state(state, {"8035": 0.01, "7203": 0.02})
    assert len(state2.return_buffers["8035"]) == 1
    assert state2.return_buffers["8035"][0] == pytest.approx(0.01)


def test_update_exposure_state_bounded():
    state = ExposureState()
    for _ in range(30):
        state = update_exposure_state(state, {"A": 0.01})
    assert len(state.return_buffers["A"]) <= CORR_WINDOW + 5


def test_update_exposure_state_deterministic():
    state = ExposureState()
    daily = {"A": 0.01, "B": 0.02}
    s1 = update_exposure_state(state, daily)
    s2 = update_exposure_state(state, daily)
    assert s1.return_buffers == s2.return_buffers


def test_update_exposure_state_updates_collapse_streak():
    # Build up identical returns for high correlation
    returns_a = [float(i) * 0.01 for i in range(1, 21)]
    returns_b = [float(i) * 0.01 for i in range(1, 21)]
    state = ExposureState(
        return_buffers={"A": returns_a, "B": returns_b},
        collapse_streak={},
    )
    # Same return for both → corr=1.0 → streak increments
    state2 = update_exposure_state(state, {"A": 0.05, "B": 0.05})
    key = _pair_key("A", "B")
    assert state2.collapse_streak.get(key, 0) >= 1


# ── persistence ───────────────────────────────────────────────────────────────

def test_save_load_exposure_state(tmp_path):
    p = tmp_path / "exposure.json"
    state = ExposureState(
        return_buffers={"A": [0.01, 0.02, 0.03]},
        collapse_streak={"A|B": 3},
    )
    save_exposure_state(state, p)
    loaded = load_exposure_state(p)
    assert loaded.return_buffers["A"] == [0.01, 0.02, 0.03]
    assert loaded.collapse_streak["A|B"] == 3


def test_load_exposure_state_missing(tmp_path):
    p = tmp_path / "nonexistent.json"
    state = load_exposure_state(p)
    assert state.return_buffers == {}
