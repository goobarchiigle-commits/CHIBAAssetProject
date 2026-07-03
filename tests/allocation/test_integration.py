"""
Integration tests for Phase 5A: Economic Exposure & Capital Efficiency Core

Tests the full pipeline:
  ExposureState → ExposureReport → rank_candidates → AllocationDecision → telemetry
"""
import json
import pytest

from src.allocation import (
    ExposureState,
    update_exposure_state,
    compute_exposure_report,
    CandidateSignal,
    rank_candidates,
    AllocationDecision,
    build_allocation_decision,
    append_allocation_decision,
    load_allocation_decisions,
    AlphaRecord,
    AlphaLifecycleState,
    build_lifecycle_state,
    append_alpha_record,
    build_lifecycle_map,
    RankSnapshot,
    StabilityWindow,
    update_stability_window,
    compute_stability_observation,
    append_stability_observation,
    load_recent_stability,
    AllocatorFactor,
    ComplexityBudget,
    admit_factor,
    can_admit_factor,
    load_complexity_budget,
    save_complexity_budget,
)
from src.allocation.efficiency_ranker import compute_efficiency_score


# ── helpers ────────────────────────────────────────────────────────────────────

def _build_state_with_returns():
    """Build an ExposureState with 20 days of correlated/uncorrelated returns.
    Uses varying returns so Pearson correlation is well-defined.
    8035 and 6857 are strongly correlated (same direction, similar magnitude).
    7203 is anti-correlated (moves opposite).
    """
    import math
    state = ExposureState()
    for i in range(1, 21):
        # Semiconductor: both move with market momentum (correlated)
        ret_8035 = math.sin(i * 0.5) * 0.02
        ret_6857 = math.sin(i * 0.5) * 0.019 + 0.001  # very similar to 8035
        ret_7203 = math.cos(i * 0.5) * 0.015           # different phase = low corr
        state = update_exposure_state(state, {
            "8035": ret_8035, "6857": ret_6857, "7203": ret_7203,
        })
    return state


def _make_candidates():
    return [
        CandidateSignal("8035", rsr_rank=85.0, post_cost_alpha_bps=22.0,
                        avg_daily_volume_yen=2e9, sector="電気機器", estimated_symbol_dd=0.12),
        CandidateSignal("6857", rsr_rank=80.0, post_cost_alpha_bps=18.0,
                        avg_daily_volume_yen=1e9, sector="電気機器", estimated_symbol_dd=0.15),
        CandidateSignal("7203", rsr_rank=75.0, post_cost_alpha_bps=15.0,
                        avg_daily_volume_yen=5e9, sector="輸送用機器", estimated_symbol_dd=0.08),
    ]


# ── full pipeline ──────────────────────────────────────────────────────────────

def test_full_pipeline_produces_ranked_scores():
    state = _build_state_with_returns()
    candidates = _make_candidates()
    ranked = rank_candidates(candidates, ["7203"], state)
    assert len(ranked) == 3
    assert ranked[0].rank == 1
    assert ranked[-1].rank == len(ranked)


def test_semiconductor_cluster_penalized():
    """8035 + 6857 (same sector, correlated) → efficiency penalty vs 7203."""
    state = _build_state_with_returns()
    # 8035 competing with existing 6857 in portfolio
    cand_8035 = CandidateSignal("8035", rsr_rank=85.0, post_cost_alpha_bps=22.0,
                                avg_daily_volume_yen=2e9, sector="電気機器", estimated_symbol_dd=0.12)
    cand_7203 = CandidateSignal("7203", rsr_rank=75.0, post_cost_alpha_bps=15.0,
                                avg_daily_volume_yen=5e9, sector="輸送用機器", estimated_symbol_dd=0.08)
    s_8035 = compute_efficiency_score(cand_8035, ["6857"], state)
    s_7203 = compute_efficiency_score(cand_7203, ["6857"], state)
    # 8035 has higher alpha but is correlated with 6857;
    # with enough data, 7203 may win on efficiency
    # At minimum, 8035 should have a correlation penalty
    assert s_8035.correlation_with_portfolio > s_7203.correlation_with_portfolio


def test_exposure_report_reflects_corr():
    state = _build_state_with_returns()
    sector_map = {"8035": "電気機器", "6857": "電気機器", "7203": "輸送用機器"}
    rsr_rank_map = {"8035": 1, "6857": 3, "7203": 8}
    report = compute_exposure_report(state, ["8035", "6857", "7203"], sector_map, rsr_rank_map)
    assert report.nominal_positions == 3
    assert report.effective_independent_n <= 3.0
    # Semiconductor cluster should appear
    theme_keys = [c["theme_key"] for c in report.theme_clusters]
    assert any("電気機器" in k for k in theme_keys)


def test_allocation_decisions_persisted(tmp_path):
    state = _build_state_with_returns()
    candidates = _make_candidates()
    ranked = rank_candidates(candidates, [], state)
    p = tmp_path / "decisions.jsonl"
    for score in ranked:
        decision = build_allocation_decision(
            timestamp="2026-05-16T09:00:00",
            trading_day="2026-05-16",
            symbol=score.symbol,
            decision="SELECTED" if score.rank == 1 else "CANDIDATE",
            rank=score.rank,
            efficiency_score=score,
            edge_persistence=0.70,
            regime_confidence=0.65,
            post_cost_alpha_bps=score.expected_geometric_return * 100,
            aggressiveness_state="OPPORTUNISTIC",
            concentration_state="LOW",
            effective_independent_n=2.5,
        )
        append_allocation_decision(decision, p)
    loaded = load_allocation_decisions(p)
    assert len(loaded) == 3
    assert all(d.schema_version == 1 for d in loaded)


def test_alpha_lifecycle_reflects_history(tmp_path):
    lifecycle_dir = tmp_path / "lifecycle"
    lifecycle_dir.mkdir()

    # Symbol with good history
    for i in range(8):
        r = AlphaRecord("8035", f"2026-05-{i+1:02d}", return_bps=20.0,
                        post_cost_bps=15.0, regime="BULL", holding_days=5)
        append_alpha_record(r, lifecycle_dir / "8035.jsonl")

    # Symbol with poor history
    for i in range(5):
        r = AlphaRecord("9999", f"2026-05-{i+1:02d}", return_bps=-5.0,
                        post_cost_bps=-10.0, regime="BULL", holding_days=2)
        append_alpha_record(r, lifecycle_dir / "9999.jsonl")

    lifecycle = build_lifecycle_map(lifecycle_dir, ["8035", "9999"])
    assert lifecycle["8035"].decay_flag is False
    assert lifecycle["8035"].rolling_win_rate == pytest.approx(1.0)
    assert lifecycle["9999"].decay_flag is True
    assert lifecycle["9999"].rolling_win_rate == pytest.approx(0.0)


def test_stability_tracks_allocator_turnover(tmp_path):
    window = StabilityWindow()
    stability_file = tmp_path / "stability.jsonl"

    # Stable ranking
    for day in range(5):
        snap = RankSnapshot(trading_day=f"2026-05-{day+1:02d}",
                            ranked_symbols=["A", "B", "C"])
        window = update_stability_window(window, snap, 0.35, 0.55)
    obs = compute_stability_observation("T", "D", window)
    append_stability_observation(obs, stability_file)
    assert obs.rank_turnover == pytest.approx(0.0)
    assert obs.stability_state == "STABLE"

    # Flip ranking
    snap = RankSnapshot(trading_day="D6", ranked_symbols=["C", "A", "B"])
    window = update_stability_window(window, snap, 0.35, 0.55)
    obs2 = compute_stability_observation("T2", "D6", window)
    append_stability_observation(obs2, stability_file)
    assert obs2.rank_turnover > 0.0

    loaded = load_recent_stability(stability_file, n=10)
    assert len(loaded) == 2


def test_complexity_budget_enforces_discipline(tmp_path):
    audit_path = tmp_path / "audit.jsonl"
    budget_path = tmp_path / "budget.json"
    budget = ComplexityBudget()

    # Admit a valid factor
    f1 = AllocatorFactor(
        factor_id="liquidity_penalty",
        description="Liquidity-adjusted capital penalty",
        admitted_date="2026-05-16",
        out_of_sample_confirmed=True,
        post_cost_confirmed=True,
        marginal_cagr_bps=25.0,
        replay_deterministic=True,
        explainability_preserved=True,
    )
    budget, ok, _ = admit_factor(budget, f1, "2026-05-16T09:00", audit_path)
    assert ok is True
    assert budget.active_factor_count == 1

    # Reject an unconfirmed factor
    f2 = AllocatorFactor(
        factor_id="ml_cluster",
        description="ML latent cluster",
        admitted_date="2026-05-16",
        out_of_sample_confirmed=False,  # not confirmed
        post_cost_confirmed=True,
        marginal_cagr_bps=30.0,
        replay_deterministic=True,
        explainability_preserved=False,
    )
    _, ok2, reason2 = admit_factor(budget, f2, "2026-05-16T09:01", audit_path)
    assert ok2 is False

    save_complexity_budget(budget, budget_path)
    loaded = load_complexity_budget(budget_path)
    assert loaded.active_factor_count == 1


def test_replay_determinism_across_runs():
    """Same inputs → identical allocation rankings on two separate runs."""
    state = _build_state_with_returns()
    candidates = _make_candidates()

    ranked1 = rank_candidates(candidates, ["7203"], state)
    ranked2 = rank_candidates(candidates, ["7203"], state)

    symbols1 = [s.symbol for s in ranked1]
    symbols2 = [s.symbol for s in ranked2]
    scores1 = [s.capital_efficiency_score for s in ranked1]
    scores2 = [s.capital_efficiency_score for s in ranked2]

    assert symbols1 == symbols2
    assert scores1 == scores2


def test_paths_module_has_allocation_constants():
    """Verify paths.py exports Phase 5A constants."""
    from src.paths import (
        ALLOCATION_DIR,
        EXPOSURE_STATE_FILE,
        ALLOCATION_DECISIONS_FILE,
        ALPHA_LIFECYCLE_DIR,
        ALLOCATOR_STABILITY_FILE,
        STABILITY_WINDOW_FILE,
        COMPLEXITY_BUDGET_FILE,
        COMPLEXITY_AUDIT_FILE,
    )
    assert ALLOCATION_DIR.name == "allocation"
    assert EXPOSURE_STATE_FILE.suffix == ".json"
    assert ALLOCATION_DECISIONS_FILE.suffix == ".jsonl"
    assert ALPHA_LIFECYCLE_DIR.name == "alpha_lifecycle"
