"""Tests for src/allocation/alpha_lifecycle.py"""
import pytest
from src.allocation.alpha_lifecycle import (
    AlphaRecord,
    AlphaLifecycleState,
    build_lifecycle_state,
    build_lifecycle_map,
    append_alpha_record,
    load_alpha_records,
    save_lifecycle_state,
    load_lifecycle_state,
    _compute_edge_persistence,
    _compute_regime_win_rates,
    LIFECYCLE_WINDOW,
    DECAY_THRESHOLD,
    MIN_TRADES_FOR_SIGNAL,
)


def _make_record(symbol="8035", date="2026-05-10", gross=20.0, net=15.0, regime="BULL", days=5):
    return AlphaRecord(
        symbol=symbol,
        trade_date=date,
        return_bps=gross,
        post_cost_bps=net,
        regime=regime,
        holding_days=days,
    )


def _win_records(n, symbol="8035"):
    return [_make_record(symbol, f"2026-05-{i:02d}", gross=20.0, net=15.0) for i in range(1, n + 1)]


def _loss_records(n, symbol="8035"):
    return [_make_record(symbol, f"2026-05-{i:02d}", gross=-10.0, net=-15.0) for i in range(1, n + 1)]


# ── _compute_edge_persistence ──────────────────────────────────────────────────

def test_persistence_all_wins():
    returns = [15.0] * 10
    p = _compute_edge_persistence(returns)
    assert p > 0.5


def test_persistence_all_losses():
    returns = [-10.0] * 10
    p = _compute_edge_persistence(returns)
    assert p == pytest.approx(0.0)


def test_persistence_insufficient_data():
    p = _compute_edge_persistence([15.0, 15.0])  # < MIN_TRADES_FOR_SIGNAL
    assert p == pytest.approx(0.5)


def test_persistence_bounded():
    for r_list in [[10.0] * 10, [-5.0] * 10, [10.0, -5.0] * 5]:
        p = _compute_edge_persistence(r_list)
        assert 0.0 <= p <= 1.0


# ── _compute_regime_win_rates ──────────────────────────────────────────────────

def test_regime_win_rates_all_bull_wins():
    records = [_make_record(regime="BULL", net=15.0)] * 5
    rates = _compute_regime_win_rates(records)
    assert rates["BULL"] == pytest.approx(1.0)


def test_regime_win_rates_mixed():
    records = (
        [_make_record(regime="BULL", net=15.0)] * 3
        + [_make_record(regime="BEAR", net=-10.0)] * 2
    )
    rates = _compute_regime_win_rates(records)
    assert rates["BULL"] == pytest.approx(1.0)
    assert rates["BEAR"] == pytest.approx(0.0)


# ── build_lifecycle_state ──────────────────────────────────────────────────────

def test_lifecycle_empty_records():
    state = build_lifecycle_state("8035", [])
    assert state.n_trades == 0
    assert state.edge_persistence == pytest.approx(0.5)
    assert state.decay_flag is False


def test_lifecycle_all_wins():
    records = _win_records(8)
    state = build_lifecycle_state("8035", records)
    assert state.n_trades == 8
    assert state.n_wins == 8
    assert state.rolling_win_rate == pytest.approx(1.0)
    assert state.decay_flag is False


def test_lifecycle_all_losses():
    records = _loss_records(5)
    state = build_lifecycle_state("8035", records)
    assert state.rolling_win_rate == pytest.approx(0.0)
    assert state.decay_flag is True  # 0.0 < DECAY_THRESHOLD=0.35


def test_lifecycle_decay_flag_threshold():
    # win_rate at DECAY_THRESHOLD should set decay_flag True
    n = LIFECYCLE_WINDOW
    n_wins = int(n * DECAY_THRESHOLD) - 1  # just below threshold
    n_losses = n - n_wins
    records = _win_records(n_wins) + _loss_records(n_losses)
    state = build_lifecycle_state("8035", records)
    assert state.decay_flag is True


def test_lifecycle_post_cost_retention_positive():
    records = [_make_record(gross=20.0, net=15.0)] * 5
    state = build_lifecycle_state("8035", records)
    assert 0.0 < state.post_cost_retention <= 1.0
    assert state.post_cost_retention == pytest.approx(0.75, abs=0.01)


def test_lifecycle_post_cost_retention_negative_gross():
    records = [_make_record(gross=-5.0, net=-8.0)] * 5
    state = build_lifecycle_state("8035", records)
    assert state.post_cost_retention == pytest.approx(0.0)


def test_lifecycle_deterministic():
    records = _win_records(7) + _loss_records(3)
    s1 = build_lifecycle_state("X", records)
    s2 = build_lifecycle_state("X", records)
    assert s1.rolling_win_rate == s2.rolling_win_rate
    assert s1.edge_persistence == s2.edge_persistence


def test_lifecycle_last_trade_date():
    records = [
        _make_record(date="2026-05-10"),
        _make_record(date="2026-05-15"),
    ]
    state = build_lifecycle_state("8035", records)
    assert state.last_trade_date == "2026-05-15"


def test_lifecycle_regime_win_rates():
    records = (
        [_make_record(regime="BULL", gross=20.0, net=15.0)] * 4
        + [_make_record(regime="BEAR", gross=-5.0, net=-10.0)] * 2
    )
    state = build_lifecycle_state("8035", records)
    assert state.regime_win_rates.get("BULL", 0) > state.regime_win_rates.get("BEAR", 1)


# ── persistence ───────────────────────────────────────────────────────────────

def test_append_and_load_records(tmp_path):
    p = tmp_path / "8035.jsonl"
    r1 = _make_record(date="2026-05-01")
    r2 = _make_record(date="2026-05-02")
    append_alpha_record(r1, p)
    append_alpha_record(r2, p)
    loaded = load_alpha_records(p)
    assert len(loaded) == 2
    assert loaded[0].trade_date == "2026-05-01"


def test_load_missing_records(tmp_path):
    p = tmp_path / "nonexistent.jsonl"
    assert load_alpha_records(p) == []


def test_save_load_lifecycle_state(tmp_path):
    p = tmp_path / "8035_lifecycle.json"
    records = _win_records(5)
    state = build_lifecycle_state("8035", records)
    save_lifecycle_state(state, p)
    loaded = load_lifecycle_state(p, "8035")
    assert loaded.rolling_win_rate == state.rolling_win_rate
    assert loaded.n_trades == state.n_trades
    assert loaded.decay_flag == state.decay_flag


def test_lifecycle_append_only(tmp_path):
    p = tmp_path / "8035.jsonl"
    for i in range(5):
        append_alpha_record(_make_record(date=f"2026-05-{i+1:02d}"), p)
    lines = p.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 5


# ── build_lifecycle_map ────────────────────────────────────────────────────────

def test_build_lifecycle_map(tmp_path):
    lifecycle_dir = tmp_path / "lifecycle"
    lifecycle_dir.mkdir()
    for sym in ["8035", "7203"]:
        p = lifecycle_dir / f"{sym}.jsonl"
        for i in range(5):
            append_alpha_record(_make_record(sym, date=f"2026-05-{i+1:02d}"), p)
    result = build_lifecycle_map(lifecycle_dir, ["8035", "7203"])
    assert "8035" in result
    assert "7203" in result
    assert result["8035"].n_trades == 5


def test_build_lifecycle_map_missing_symbol(tmp_path):
    lifecycle_dir = tmp_path / "lifecycle"
    lifecycle_dir.mkdir()
    result = build_lifecycle_map(lifecycle_dir, ["9999"])
    assert "9999" in result
    assert result["9999"].n_trades == 0
