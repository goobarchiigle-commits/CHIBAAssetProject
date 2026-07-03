"""Tests for src/allocation/stability_metrics.py"""
import json
import pytest
from src.allocation.stability_metrics import (
    RankSnapshot,
    AllocatorStabilityObservation,
    StabilityWindow,
    _compute_rank_turnover,
    _std,
    _classify_stability,
    update_stability_window,
    compute_stability_observation,
    append_stability_observation,
    load_stability_window,
    save_stability_window,
    load_recent_stability,
    TURNOVER_ALERT_THRESHOLD,
    CONCENTRATION_ALERT_THRESHOLD,
    AGGRESSION_ALERT_THRESHOLD,
)


# ── _std ───────────────────────────────────────────────────────────────────────

def test_std_constant():
    assert _std([5.0, 5.0, 5.0]) == pytest.approx(0.0)


def test_std_single():
    assert _std([5.0]) == pytest.approx(0.0)


def test_std_known():
    # std([1, 2, 3, 4, 5]) ≈ 1.5811
    result = _std([1.0, 2.0, 3.0, 4.0, 5.0])
    assert result == pytest.approx(1.5811, abs=1e-3)


# ── _compute_rank_turnover ─────────────────────────────────────────────────────

def test_rank_turnover_identical():
    prev = ["A", "B", "C"]
    curr = ["A", "B", "C"]
    assert _compute_rank_turnover(prev, curr) == pytest.approx(0.0)


def test_rank_turnover_fully_changed():
    prev = ["A", "B", "C"]
    curr = ["D", "E", "F"]
    assert _compute_rank_turnover(prev, curr) == pytest.approx(1.0)


def test_rank_turnover_partial():
    prev = ["A", "B", "C"]
    curr = ["A", "C", "B"]  # 2 changed
    t = _compute_rank_turnover(prev, curr)
    assert t == pytest.approx(2 / 3, abs=1e-4)


def test_rank_turnover_empty():
    assert _compute_rank_turnover([], []) == pytest.approx(0.0)


def test_rank_turnover_different_lengths():
    prev = ["A", "B"]
    curr = ["A", "B", "C"]
    t = _compute_rank_turnover(prev, curr)
    # 1/3 changed (C added at position 2 which was empty)
    assert 0.0 <= t <= 1.0


# ── _classify_stability ────────────────────────────────────────────────────────

def test_classify_stable():
    state = _classify_stability(0.0, 0.0, 0.0)
    assert state == "STABLE"


def test_classify_unstable_two_alerts():
    state = _classify_stability(
        TURNOVER_ALERT_THRESHOLD,
        CONCENTRATION_ALERT_THRESHOLD,
        0.0,
    )
    assert state == "UNSTABLE"


def test_classify_critical():
    state = _classify_stability(
        TURNOVER_ALERT_THRESHOLD,
        CONCENTRATION_ALERT_THRESHOLD * 3,
        AGGRESSION_ALERT_THRESHOLD * 3,
    )
    assert state == "CRITICAL"


# ── update_stability_window ────────────────────────────────────────────────────

def test_update_stability_window_appends():
    window = StabilityWindow()
    snap = RankSnapshot(trading_day="2026-05-16", ranked_symbols=["A", "B"])
    w2 = update_stability_window(window, snap, sector_concentration=0.40, aggression_value=0.60)
    assert len(w2.rank_snapshots) == 1
    assert len(w2.concentration_history) == 1
    assert len(w2.aggression_history) == 1


def test_update_stability_window_bounded():
    window = StabilityWindow()
    for i in range(15):
        snap = RankSnapshot(trading_day=f"2026-05-{i+1:02d}", ranked_symbols=["A", "B"])
        window = update_stability_window(window, snap, 0.4, 0.6, max_history=10)
    assert len(window.rank_snapshots) <= 10


def test_update_bounds_concentration():
    window = StabilityWindow()
    snap = RankSnapshot(trading_day="D", ranked_symbols=[])
    w2 = update_stability_window(window, snap, sector_concentration=1.5, aggression_value=0.5)
    assert w2.concentration_history[-1] <= 1.0


def test_update_bounds_aggression():
    window = StabilityWindow()
    snap = RankSnapshot(trading_day="D", ranked_symbols=[])
    w2 = update_stability_window(window, snap, sector_concentration=0.5, aggression_value=-0.5)
    assert w2.aggression_history[-1] >= 0.0


# ── compute_stability_observation ─────────────────────────────────────────────

def test_stability_observation_no_history():
    window = StabilityWindow()
    obs = compute_stability_observation("T", "D", window)
    assert obs.stability_state == "STABLE"
    assert obs.rank_turnover == pytest.approx(0.0)


def test_stability_observation_detects_turnover():
    window = StabilityWindow()
    snap1 = RankSnapshot(trading_day="D1", ranked_symbols=["A", "B", "C"])
    snap2 = RankSnapshot(trading_day="D2", ranked_symbols=["D", "E", "F"])
    window = update_stability_window(window, snap1, 0.3, 0.5)
    window = update_stability_window(window, snap2, 0.3, 0.5)
    obs = compute_stability_observation("T", "D2", window)
    assert obs.rank_turnover == pytest.approx(1.0)


def test_stability_observation_detects_concentration_oscillation():
    window = StabilityWindow()
    for conc in [0.1, 0.9, 0.1, 0.9, 0.1, 0.9]:
        snap = RankSnapshot(trading_day="D", ranked_symbols=["A"])
        window = update_stability_window(window, snap, conc, 0.5)
    obs = compute_stability_observation("T", "D", window)
    assert obs.concentration_oscillation >= CONCENTRATION_ALERT_THRESHOLD


def test_stability_observation_deterministic():
    window = StabilityWindow()
    snap = RankSnapshot(trading_day="D", ranked_symbols=["A", "B"])
    window = update_stability_window(window, snap, 0.4, 0.6)
    window = update_stability_window(window, snap, 0.4, 0.6)
    obs1 = compute_stability_observation("T", "D", window)
    obs2 = compute_stability_observation("T", "D", window)
    assert obs1.rank_turnover == obs2.rank_turnover
    assert obs1.stability_state == obs2.stability_state


# ── persistence ───────────────────────────────────────────────────────────────

def test_append_and_load_stability(tmp_path):
    p = tmp_path / "stability.jsonl"
    obs = AllocatorStabilityObservation(
        timestamp="T", trading_day="D",
        rank_turnover=0.33, concentration_oscillation=0.05,
        aggressiveness_volatility=0.10, stability_state="STABLE",
        n_observations_used=5,
    )
    append_stability_observation(obs, p)
    loaded = load_recent_stability(p, n=10)
    assert len(loaded) == 1
    assert loaded[0].stability_state == "STABLE"


def test_save_load_window(tmp_path):
    p = tmp_path / "window.json"
    window = StabilityWindow(
        concentration_history=[0.3, 0.4, 0.5],
        aggression_history=[0.6, 0.7, 0.8],
    )
    save_stability_window(window, p)
    loaded = load_stability_window(p)
    assert loaded.concentration_history == [0.3, 0.4, 0.5]


def test_load_missing_window(tmp_path):
    p = tmp_path / "nonexistent.json"
    w = load_stability_window(p)
    assert w.rank_snapshots == []


def test_stability_append_only(tmp_path):
    p = tmp_path / "stability.jsonl"
    for i in range(5):
        obs = AllocatorStabilityObservation(
            timestamp="T", trading_day=f"D{i}",
            rank_turnover=0.0, concentration_oscillation=0.0,
            aggressiveness_volatility=0.0, stability_state="STABLE",
            n_observations_used=1,
        )
        append_stability_observation(obs, p)
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5
