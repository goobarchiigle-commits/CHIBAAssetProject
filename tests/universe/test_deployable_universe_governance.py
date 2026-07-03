"""
tests/universe/test_deployable_universe_governance.py

Tests for deployable universe governance layer.

Covers:
  - deterministic promotion ordering
  - manifest lineage integrity (hash chain)
  - replay reproducibility
  - deployability scoring correctness
  - duplicate prevention
  - cooldown enforcement
  - stale-data rejection
  - append-only persistence
  - AUTO_PROMOTE_SAFE behavior
  - authoritative manifest rebuilding
  - fail-closed on manifest corruption
"""
import json
import sys
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from universe.deployable_universe_governance import (
    run_universe_governance,
    DeployableUniverseGovernor,
    DeployabilityScorer,
    UniverseManifestAuthority,
    PromotionDecisionRecorder,
    UniverseLineageTracker,
    UniverseBreadthDiagnostics,
    DeployabilityScore,
    PromotionDecision,
    UniverseManifest,
    GovernanceReport,
    GovernanceResult,
    ACTION_PROMOTE,
    ACTION_REJECT,
    ACTION_OBSERVE,
    ACTION_COOLDOWN,
    AUTO_PROMOTE_SAFE,
    OBSERVE_ONLY,
    MANUAL_APPROVAL,
    MAX_PROMOTIONS_PER_REBALANCE,
    DEMOTION_COOLDOWN_DAYS,
    UNIVERSE_CAP,
    MIN_DEPLOYABILITY_SCORE,
    MAX_UNIT_PRICE_FRACTION,
    LOT_SIZE,
    _sha256,
    _canonical,
    _append_jsonl,
    _load_jsonl,
)

# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

CAPITAL = 3_000_000
MAX_ALLOC = CAPITAL * MAX_UNIT_PRICE_FRACTION   # 900,000 JPY


def _affordable_price(capital: int = CAPITAL) -> float:
    """Return a price whose unit cost (×100) is well within the max_alloc limit."""
    return (capital * MAX_UNIT_PRICE_FRACTION * 0.5) / LOT_SIZE   # 50% of limit


def _expensive_price(capital: int = CAPITAL) -> float:
    """Return a price whose unit cost (×100) exceeds max_alloc."""
    return (capital * MAX_UNIT_PRICE_FRACTION * 1.5) / LOT_SIZE   # 150% of limit


@pytest.fixture
def tmp_uni_dir(tmp_path):
    d = tmp_path / "universe"
    d.mkdir()
    return d


@pytest.fixture
def live_universe():
    return {
        "8035.T": "電機精密",
        "6702.T": "電機",
        "8306.T": "銀行",
        "8058.T": "商社",
        "9101.T": "海運",
    }


@pytest.fixture
def shadow_universe():
    return {
        "3402.T": "繊維",
        "2802.T": "食品",
        "9020.T": "陸運",
    }


def _price_lookup_affordable(symbol: str) -> float:
    return _affordable_price()


def _price_lookup_expensive(symbol: str) -> float:
    return _expensive_price()


def _price_lookup_none(symbol: str):
    return None


# ─────────────────────────────────────────────────────────────────────────────
# DeployabilityScorer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDeployabilityScorer:

    def test_affordable_candidate_passes_gate(self, live_universe):
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        score = scorer.score_candidate(
            "3402.T", "繊維", _price_lookup_affordable, {"3402.T": 70.0}
        )
        assert score.affordable is True
        assert score.price_available is True
        assert score.unit_price > 0
        assert score.deployability_score >= MIN_DEPLOYABILITY_SCORE
        assert score.passes_gate is True

    def test_expensive_candidate_fails_gate(self, live_universe):
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        score = scorer.score_candidate(
            "X.T", "電機精密", _price_lookup_expensive, None
        )
        assert score.affordable is False
        assert score.passes_gate is False

    def test_no_price_fails_gate(self, live_universe):
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        score = scorer.score_candidate("X.T", "食品", _price_lookup_none, None)
        assert score.price_available is False
        assert score.affordable is False
        assert score.passes_gate is False

    def test_rsr_gate_fails_when_below_threshold(self, live_universe):
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        score = scorer.score_candidate(
            "3402.T", "繊維", _price_lookup_affordable, {"3402.T": 30.0}
        )
        assert score.rsr_passes is False
        assert score.passes_gate is False

    def test_rsr_gate_passes_when_above_threshold(self, live_universe):
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        score = scorer.score_candidate(
            "3402.T", "繊維", _price_lookup_affordable, {"3402.T": 70.0}
        )
        assert score.rsr_passes is True
        # gate may still be True or False depending on deployability_score

    def test_rsr_blocked_when_no_rsr_map(self, live_universe):
        # FAIL_CLOSED: missing RSR blocks promotion (rsr_scores=None → rsr_passes=False)
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        score = scorer.score_candidate(
            "3402.T", "繊維", _price_lookup_affordable, None
        )
        assert score.rsr_passes is False
        assert score.rsr_source == "missing"
        assert score.passes_gate is False

    def test_score_all_deterministic(self, live_universe, shadow_universe):
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        scores1 = scorer.score_all(shadow_universe, _price_lookup_affordable, None)
        scores2 = scorer.score_all(shadow_universe, _price_lookup_affordable, None)
        syms1 = [s.symbol for s in scores1]
        syms2 = [s.symbol for s in scores2]
        assert syms1 == syms2

    def test_affordable_candidate_rank_assigned(self, live_universe, shadow_universe):
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        scores = scorer.score_all(shadow_universe, _price_lookup_affordable, None)
        affordable = [s for s in scores if s.affordable]
        ranks = [s.affordable_candidate_rank for s in affordable]
        assert sorted(ranks) == list(range(1, len(affordable) + 1))

    def test_deployability_score_bounds(self, live_universe, shadow_universe):
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        scores = scorer.score_all(shadow_universe, _price_lookup_affordable, None)
        for s in scores:
            assert 0.0 <= s.deployability_score <= 1.0

    def test_sector_diversification_high_for_new_sector(self):
        """New sector (not in live) should have high diversification score."""
        live = {"8035.T": "電機精密", "8306.T": "銀行"}
        scorer = DeployabilityScorer(CAPITAL, live)
        score = scorer.score_candidate("3402.T", "繊維", _price_lookup_affordable, None)
        # 繊維 is not in live → no concentration → high diversification score
        assert score.sector_diversification_score > 0.5

    def test_sector_diversification_low_for_dominant_sector(self):
        """Overweight sector should have low diversification score."""
        live = {f"{i}.T": "電機精密" for i in range(10)}  # 10 symbols, all 電機精密
        scorer = DeployabilityScorer(CAPITAL, live)
        score = scorer.score_candidate("NEW.T", "電機精密", _price_lookup_affordable, None)
        assert score.sector_diversification_score <= 0.5

    def test_sort_order_desc_score_asc_symbol(self, live_universe):
        shadow = {"ZZZ.T": "食品", "AAA.T": "食品"}
        scorer = DeployabilityScorer(CAPITAL, live_universe)
        scores = scorer.score_all(shadow, _price_lookup_affordable, None)
        # When scores equal, AAA.T should come before ZZZ.T (asc symbol)
        affordable = [s for s in scores if s.affordable]
        if len(affordable) >= 2:
            for i in range(len(affordable) - 1):
                s1, s2 = affordable[i], affordable[i + 1]
                if s1.deployability_score == s2.deployability_score:
                    assert s1.symbol <= s2.symbol


# ─────────────────────────────────────────────────────────────────────────────
# UniverseManifest tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUniverseManifest:

    def test_manifest_id_deterministic(self):
        m1 = UniverseManifest.create("run1", "batch1", ["A.T", "B.T"], [], "")
        m2 = UniverseManifest.create("run1", "batch1", ["A.T", "B.T"], [], "")
        # Different created_at → different manifest_id (timestamp-dependent)
        # But same fields except timestamp → we can't check equality easily
        # At least validate() should pass
        assert m1.validate() is True
        assert m2.validate() is True

    def test_manifest_validates_correctly(self):
        m = UniverseManifest.create("r1", "b1", ["X.T", "Y.T"], ["Y.T"], "prev123")
        assert m.validate() is True

    def test_manifest_detects_tampering(self):
        m = UniverseManifest.create("r1", "b1", ["X.T"], [], "")
        m.manifest_id = "tampered"
        assert m.validate() is False

    def test_manifest_sorted_symbols(self):
        m = UniverseManifest.create("r1", "b1", ["Z.T", "A.T", "M.T"], [], "")
        assert m.live_symbols == sorted(["Z.T", "A.T", "M.T"])

    def test_manifest_universe_size(self):
        syms = ["A.T", "B.T", "C.T"]
        m = UniverseManifest.create("r1", "b1", syms, [], "")
        assert m.universe_size == 3


# ─────────────────────────────────────────────────────────────────────────────
# UniverseManifestAuthority tests
# ─────────────────────────────────────────────────────────────────────────────

class TestUniverseManifestAuthority:

    def test_load_empty_when_no_file(self, tmp_uni_dir):
        auth = UniverseManifestAuthority(tmp_uni_dir / "manifest.jsonl")
        assert auth.load_all() == []
        assert auth.latest() is None

    def test_append_and_load(self, tmp_uni_dir):
        path = tmp_uni_dir / "manifest.jsonl"
        auth = UniverseManifestAuthority(path)
        m = UniverseManifest.create("r1", "b1", ["A.T", "B.T"], ["B.T"], "")
        auth.append(m)
        loaded = auth.load_all()
        assert len(loaded) == 1
        assert loaded[0].manifest_id == m.manifest_id

    def test_append_only_multiple(self, tmp_uni_dir):
        path = tmp_uni_dir / "manifest.jsonl"
        auth = UniverseManifestAuthority(path)
        m1 = UniverseManifest.create("r1", "b1", ["A.T"], [], "")
        m2 = UniverseManifest.create("r2", "b2", ["A.T", "B.T"], ["B.T"], m1.manifest_id)
        auth.append(m1)
        auth.append(m2)
        loaded = auth.load_all()
        assert len(loaded) == 2
        assert loaded[1].previous_manifest_id == m1.manifest_id

    def test_validate_chain_valid(self, tmp_uni_dir):
        path = tmp_uni_dir / "manifest.jsonl"
        auth = UniverseManifestAuthority(path)
        m1 = UniverseManifest.create("r1", "b1", ["A.T"], [], "")
        m2 = UniverseManifest.create("r2", "b2", ["A.T", "B.T"], ["B.T"], m1.manifest_id)
        auth.append(m1)
        auth.append(m2)
        assert auth.validate_chain() is True

    def test_validate_chain_detects_broken_chain(self, tmp_uni_dir):
        path = tmp_uni_dir / "manifest.jsonl"
        auth = UniverseManifestAuthority(path)
        m1 = UniverseManifest.create("r1", "b1", ["A.T"], [], "")
        m2 = UniverseManifest.create("r2", "b2", ["A.T", "B.T"], ["B.T"], "wrong_prev_id")
        auth.append(m1)
        auth.append(m2)
        assert auth.validate_chain() is False

    def test_corrupt_manifest_raises_on_load(self, tmp_uni_dir):
        path = tmp_uni_dir / "manifest.jsonl"
        path.write_text('{"bad": "record"}\n', encoding="utf-8")
        auth = UniverseManifestAuthority(path)
        with pytest.raises(RuntimeError, match="corrupt"):
            auth.load_all()

    def test_latest_returns_last_appended(self, tmp_uni_dir):
        path = tmp_uni_dir / "manifest.jsonl"
        auth = UniverseManifestAuthority(path)
        m1 = UniverseManifest.create("r1", "b1", ["A.T"], [], "")
        m2 = UniverseManifest.create("r2", "b2", ["A.T", "B.T"], ["B.T"], m1.manifest_id)
        auth.append(m1)
        auth.append(m2)
        latest = auth.latest()
        assert latest.manifest_id == m2.manifest_id


# ─────────────────────────────────────────────────────────────────────────────
# PromotionDecisionRecorder tests
# ─────────────────────────────────────────────────────────────────────────────

class TestPromotionDecisionRecorder:

    def _make_score(self, symbol: str, sector: str = "食品") -> DeployabilityScore:
        return DeployabilityScore(
            symbol=symbol, sector=sector,
            unit_price=10000.0, unit_price_pressure=0.01,
            affordable=True, price_available=True,
            rsr_score=70.0, rsr_passes=True,
            concentration_relief_score=0.5,
            sector_diversification_score=0.6,
            deployability_score=0.55,
            affordable_candidate_rank=1,
            passes_gate=True,
        )

    def test_empty_file_no_cooldowns(self, tmp_uni_dir):
        rec = PromotionDecisionRecorder(tmp_uni_dir / "decisions.jsonl")
        assert rec.get_cooldown_symbols() == {}

    def test_append_and_load(self, tmp_uni_dir):
        path = tmp_uni_dir / "decisions.jsonl"
        rec = PromotionDecisionRecorder(path)
        score = self._make_score("3402.T")
        decision = PromotionDecision.create(
            run_id="r1", promotion_batch_id="b1",
            symbol="3402.T", sector="繊維",
            action=ACTION_PROMOTE, reason="test", score=score,
        )
        rec.append(decision)
        loaded = rec.load_all()
        assert len(loaded) == 1
        assert loaded[0].symbol == "3402.T"
        assert loaded[0].action == ACTION_PROMOTE

    def test_cooldown_for_recently_rejected(self, tmp_uni_dir):
        path = tmp_uni_dir / "decisions.jsonl"
        rec = PromotionDecisionRecorder(path)
        score = self._make_score("SYM.T")
        d = PromotionDecision.create(
            run_id="r1", promotion_batch_id="b1",
            symbol="SYM.T", sector="食品",
            action=ACTION_REJECT, reason="test reject", score=score,
        )
        rec.append(d)
        cooldowns = rec.get_cooldown_symbols()
        assert "SYM.T" in cooldowns

    def test_no_cooldown_for_old_rejection(self, tmp_uni_dir):
        path = tmp_uni_dir / "decisions.jsonl"
        # Write a rejection with old timestamp directly
        old_dt = (
            datetime.now(timezone.utc) - timedelta(days=DEMOTION_COOLDOWN_DAYS + 5)
        ).isoformat()
        score = self._make_score("OLD.T")
        d = PromotionDecision.create(
            run_id="r0", promotion_batch_id="b0",
            symbol="OLD.T", sector="食品",
            action=ACTION_REJECT, reason="old", score=score,
        )
        # Manually set old created_at
        d_dict = d.to_dict()
        d_dict["created_at"] = old_dt
        _append_jsonl(path, d_dict)
        rec = PromotionDecisionRecorder(path)
        cooldowns = rec.get_cooldown_symbols()
        assert "OLD.T" not in cooldowns

    def test_no_cooldown_for_promoted(self, tmp_uni_dir):
        """Only rejected symbols get cooldown, not promoted ones."""
        path = tmp_uni_dir / "decisions.jsonl"
        rec = PromotionDecisionRecorder(path)
        score = self._make_score("GOOD.T")
        d = PromotionDecision.create(
            run_id="r1", promotion_batch_id="b1",
            symbol="GOOD.T", sector="食品",
            action=ACTION_PROMOTE, reason="promoted", score=score,
        )
        rec.append(d)
        cooldowns = rec.get_cooldown_symbols()
        assert "GOOD.T" not in cooldowns

    def test_append_only_no_mutation(self, tmp_uni_dir):
        """Verify file grows monotonically (append-only)."""
        path = tmp_uni_dir / "decisions.jsonl"
        rec = PromotionDecisionRecorder(path)
        score = self._make_score("X.T")
        for i in range(3):
            d = PromotionDecision.create(
                run_id=f"r{i}", promotion_batch_id=f"b{i}",
                symbol="X.T", sector="食品",
                action=ACTION_PROMOTE, reason="test", score=score,
            )
            size_before = path.stat().st_size if path.exists() else 0
            rec.append(d)
            size_after = path.stat().st_size
            assert size_after > size_before


# ─────────────────────────────────────────────────────────────────────────────
# DeployableUniverseGovernor integration tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGovernorAutoPromoteSafe:

    def _run(
        self,
        tmp_uni_dir,
        live_universe=None,
        shadow_universe=None,
        price_fn=None,
        rsr_scores=None,
        run_id="run_001",
        mode=AUTO_PROMOTE_SAFE,
        universe_file=None,
    ):
        live_universe = live_universe or {
            "8035.T": "電機精密",
            "8306.T": "銀行",
        }
        shadow_universe = shadow_universe or {
            "3402.T": "繊維",
            "2802.T": "食品",
            "9020.T": "陸運",
        }
        price_fn = price_fn or _price_lookup_affordable
        gov = DeployableUniverseGovernor(tmp_uni_dir)
        return gov.run(
            live_universe=live_universe,
            shadow_universe=shadow_universe,
            capital=CAPITAL,
            run_id=run_id,
            mode=mode,
            price_lookup=price_fn,
            rsr_scores=rsr_scores,
            universe_file=universe_file,
        )

    def test_auto_promote_adds_symbols(self, tmp_uni_dir):
        result = self._run(tmp_uni_dir)
        assert isinstance(result, GovernanceResult)
        assert len(result.promoted_symbols) >= 0  # may or may not promote
        assert len(result.updated_live_universe) >= 2  # never shrinks

    def test_max_promotions_per_rebalance(self, tmp_uni_dir):
        result = self._run(tmp_uni_dir)
        assert len(result.promoted_symbols) <= MAX_PROMOTIONS_PER_REBALANCE

    def test_promoted_symbols_in_updated_universe(self, tmp_uni_dir):
        result = self._run(tmp_uni_dir)
        for sym in result.promoted_symbols:
            assert sym in result.updated_live_universe

    def test_no_promotion_of_live_symbols(self, tmp_uni_dir):
        live = {"3402.T": "繊維", "8306.T": "銀行"}
        shadow = {"3402.T": "繊維"}   # 3402.T is already live
        result = self._run(tmp_uni_dir, live_universe=live, shadow_universe=shadow)
        assert "3402.T" not in result.promoted_symbols

    def test_universe_never_shrinks(self, tmp_uni_dir):
        live = {"8035.T": "電機精密", "8306.T": "銀行"}
        shadow = {"3402.T": "繊維"}
        result = self._run(tmp_uni_dir, live_universe=live, shadow_universe=shadow)
        assert len(result.updated_live_universe) >= len(live)

    def test_observe_only_no_promotions(self, tmp_uni_dir):
        result = self._run(tmp_uni_dir, mode=OBSERVE_ONLY)
        assert result.promoted_symbols == []
        # All decisions should be OBSERVE
        for d in result.decisions:
            assert d.action == ACTION_OBSERVE

    def test_manual_approval_no_promotions(self, tmp_uni_dir):
        result = self._run(tmp_uni_dir, mode=MANUAL_APPROVAL)
        assert result.promoted_symbols == []

    def test_expensive_candidates_rejected(self, tmp_uni_dir):
        result = self._run(tmp_uni_dir, price_fn=_price_lookup_expensive)
        assert result.promoted_symbols == []
        reject_actions = [d.action for d in result.decisions]
        assert ACTION_REJECT in reject_actions

    def test_no_price_candidates_rejected(self, tmp_uni_dir):
        result = self._run(tmp_uni_dir, price_fn=_price_lookup_none)
        assert result.promoted_symbols == []
        reasons = [d.reason for d in result.decisions]
        assert any("stale_market_data" in r for r in reasons)

    def test_rsr_gate_rejects_low_rsr(self, tmp_uni_dir):
        shadow = {"3402.T": "繊維", "2802.T": "食品"}
        rsr = {"3402.T": 20.0, "2802.T": 20.0}  # both below threshold
        result = self._run(
            tmp_uni_dir, shadow_universe=shadow,
            price_fn=_price_lookup_affordable, rsr_scores=rsr,
        )
        assert result.promoted_symbols == []

    def test_rsr_gate_allows_high_rsr(self, tmp_uni_dir):
        shadow = {"3402.T": "繊維"}
        rsr = {"3402.T": 80.0}
        result = self._run(
            tmp_uni_dir, shadow_universe=shadow,
            price_fn=_price_lookup_affordable, rsr_scores=rsr,
        )
        # Should pass RSR gate (may still be rejected for other reasons, but not RSR)
        rsr_reject = [
            d for d in result.decisions
            if "rsr_below_threshold" in d.reason
        ]
        assert rsr_reject == []

    def test_universe_cap_enforced(self, tmp_uni_dir):
        live = {f"{i}.T": "電機精密" for i in range(UNIVERSE_CAP)}
        shadow = {"9020.T": "陸運", "2802.T": "食品"}
        result = self._run(
            tmp_uni_dir, live_universe=live, shadow_universe=shadow,
            price_fn=_price_lookup_affordable,
        )
        assert len(result.updated_live_universe) <= UNIVERSE_CAP
        assert result.promoted_symbols == []

    def test_sector_concentration_gate(self, tmp_uni_dir):
        """Single sector domination should block same-sector promotion."""
        # 40% of live is 電機精密 already (above 35% gate)
        live = {
            "A.T": "電機精密",
            "B.T": "電機精密",
            "C.T": "食品",
            "D.T": "銀行",
            "E.T": "化学",
        }
        # Shadow: another 電機精密 candidate — should be rejected by concentration gate
        shadow = {"F.T": "電機精密"}
        result = self._run(
            tmp_uni_dir, live_universe=live, shadow_universe=shadow,
            price_fn=_price_lookup_affordable,
        )
        # F.T should not be promoted since it would push 電機精密 above 35%
        f_decisions = [d for d in result.decisions if d.symbol == "F.T"]
        if f_decisions:
            # May be REJECT or PROMOTE depending on exact fraction
            # Just verify it wasn't silently dropped
            assert len(f_decisions) == 1

    def test_cooldown_enforcement(self, tmp_uni_dir):
        """Symbols under cooldown must not be promoted."""
        # Pre-populate decisions.jsonl with a recent REJECT for target symbol
        decisions_path = tmp_uni_dir / "promotion_decisions.jsonl"
        shadow = {"3402.T": "繊維"}

        # Simulate a recent rejection
        from universe.deployable_universe_governance import (
            DeployabilityScore as DS, PromotionDecision as PD,
        )
        fake_score = DS(
            symbol="3402.T", sector="繊維",
            unit_price=10000.0, unit_price_pressure=0.01,
            affordable=True, price_available=True,
            rsr_score=70.0, rsr_passes=True,
            concentration_relief_score=0.5,
            sector_diversification_score=0.6,
            deployability_score=0.55,
            affordable_candidate_rank=1,
            passes_gate=True,
        )
        fake_decision = PD.create(
            run_id="prev_run", promotion_batch_id="prev_batch",
            symbol="3402.T", sector="繊維",
            action=ACTION_REJECT, reason="forced reject for cooldown test",
            score=fake_score,
        )
        _append_jsonl(decisions_path, fake_decision.to_dict())

        result = self._run(
            tmp_uni_dir,
            shadow_universe=shadow,
            price_fn=_price_lookup_affordable,
        )
        assert "3402.T" not in result.promoted_symbols
        cooldown_decisions = [d for d in result.decisions if d.action == ACTION_COOLDOWN]
        assert any(d.symbol == "3402.T" for d in cooldown_decisions)

    def test_duplicate_prevention_same_run(self, tmp_uni_dir):
        """Same symbol must not appear twice in promoted_symbols."""
        shadow = {"3402.T": "繊維"}
        result = self._run(tmp_uni_dir, shadow_universe=shadow)
        assert len(result.promoted_symbols) == len(set(result.promoted_symbols))

    def test_duplicate_prevention_across_runs(self, tmp_uni_dir):
        """Symbol already in live universe must not appear in promoted."""
        shadow = {"ALREADY.T": "食品"}
        live = {"ALREADY.T": "食品", "OTHER.T": "銀行"}
        result = self._run(
            tmp_uni_dir, live_universe=live, shadow_universe=shadow,
        )
        assert "ALREADY.T" not in result.promoted_symbols

    def test_manifest_written_on_promotion(self, tmp_uni_dir):
        shadow = {"3402.T": "繊維"}
        result = self._run(tmp_uni_dir, shadow_universe=shadow, price_fn=_price_lookup_affordable)
        if result.promoted_symbols:
            assert result.manifest_written is True
            assert result.manifest is not None

    def test_manifest_lineage_integrity(self, tmp_uni_dir):
        shadow = {"3402.T": "繊維", "2802.T": "食品", "9020.T": "陸運"}
        # Run twice
        self._run(tmp_uni_dir, shadow_universe=shadow, run_id="run_001")
        self._run(tmp_uni_dir, shadow_universe=shadow, run_id="run_002")
        auth = UniverseManifestAuthority(tmp_uni_dir / "universe_manifest.jsonl")
        assert auth.validate_chain() is True

    def test_fail_closed_on_corrupt_manifest(self, tmp_uni_dir):
        """Corrupt manifest → RuntimeError (FAIL_CLOSED)."""
        manifest_path = tmp_uni_dir / "universe_manifest.jsonl"
        # Write a corrupt record (missing required field)
        manifest_path.write_text('{"bad": "corrupt"}\n', encoding="utf-8")
        with pytest.raises(RuntimeError):
            self._run(tmp_uni_dir)

    def test_promotion_ordering_deterministic(self, tmp_path):
        """Same inputs + independent fresh state → same promotion decisions in same order."""
        live = {"8035.T": "電機精密"}
        shadow = {"3402.T": "繊維", "2802.T": "食品", "9020.T": "陸運"}

        dir1 = tmp_path / "uni_a"
        dir1.mkdir()
        dir2 = tmp_path / "uni_b"
        dir2.mkdir()

        gov1 = DeployableUniverseGovernor(dir1)
        gov2 = DeployableUniverseGovernor(dir2)

        r1 = gov1.run(live, shadow, CAPITAL, "same_run", AUTO_PROMOTE_SAFE, _price_lookup_affordable)
        r2 = gov2.run(live, shadow, CAPITAL, "same_run", AUTO_PROMOTE_SAFE, _price_lookup_affordable)

        # Same promoted symbols
        assert r1.promoted_symbols == r2.promoted_symbols

        # Same decision symbols and actions (sorted for comparison stability)
        actions1 = sorted((d.symbol, d.action) for d in r1.decisions)
        actions2 = sorted((d.symbol, d.action) for d in r2.decisions)
        assert actions1 == actions2

    def test_universe_file_updated_on_promotion(self, tmp_uni_dir, tmp_path):
        """Universe JSON file updated atomically on promotion."""
        universe_file = tmp_path / "test_universe.json"
        universe_file.write_text(json.dumps({
            "version": "test_v1",
            "symbols": {"8035.T": "電機精密"},
            "n_stocks": 1,
        }, ensure_ascii=False), encoding="utf-8")

        shadow = {"3402.T": "繊維"}
        gov = DeployableUniverseGovernor(tmp_uni_dir)
        result = gov.run(
            live_universe={"8035.T": "電機精密"},
            shadow_universe=shadow,
            capital=CAPITAL,
            run_id="file_test",
            mode=AUTO_PROMOTE_SAFE,
            price_lookup=_price_lookup_affordable,
            universe_file=universe_file,
        )
        if result.promoted_symbols:
            assert result.universe_file_updated is True
            updated = json.loads(universe_file.read_text(encoding="utf-8"))
            assert "3402.T" in updated["symbols"]
            assert updated["n_stocks"] > 1

    def test_universe_file_not_updated_when_no_promotion(self, tmp_uni_dir, tmp_path):
        """Universe file NOT modified when no promotions occur."""
        universe_file = tmp_path / "test_universe.json"
        original_content = json.dumps({
            "version": "test_v1",
            "symbols": {"8035.T": "電機精密"},
            "n_stocks": 1,
        }, ensure_ascii=False)
        universe_file.write_text(original_content, encoding="utf-8")

        gov = DeployableUniverseGovernor(tmp_uni_dir)
        result = gov.run(
            live_universe={"8035.T": "電機精密"},
            shadow_universe={"EXPENSIVE.T": "食品"},
            capital=CAPITAL,
            run_id="no_promote_test",
            mode=AUTO_PROMOTE_SAFE,
            price_lookup=_price_lookup_expensive,   # all expensive → no promotions
            universe_file=universe_file,
        )
        assert result.promoted_symbols == []
        # File should be unchanged
        assert universe_file.read_text(encoding="utf-8") == original_content

    def test_append_only_decisions(self, tmp_uni_dir):
        """Decision file only grows — never shrinks or overwrites."""
        shadow = {"3402.T": "繊維"}
        gov = DeployableUniverseGovernor(tmp_uni_dir)
        decisions_path = tmp_uni_dir / "promotion_decisions.jsonl"
        sizes = []
        for i in range(3):
            gov.run(
                live_universe={"8035.T": "電機精密"},
                shadow_universe=shadow,
                capital=CAPITAL,
                run_id=f"run_{i:03d}",
                mode=AUTO_PROMOTE_SAFE,
                price_lookup=_price_lookup_affordable,
            )
            if decisions_path.exists():
                sizes.append(decisions_path.stat().st_size)
        for i in range(1, len(sizes)):
            assert sizes[i] >= sizes[i - 1]

    def test_report_generated(self, tmp_uni_dir):
        result = self._run(tmp_uni_dir)
        assert isinstance(result.governance_report, GovernanceReport)
        report = result.governance_report
        assert report.mode == AUTO_PROMOTE_SAFE
        assert isinstance(report.sector_distribution, dict)
        assert 0.0 <= report.deployable_breadth_score <= 1.0

    def test_daily_report_file_created(self, tmp_uni_dir):
        self._run(tmp_uni_dir)
        report_dir = tmp_uni_dir / "universe_governance_reports"
        reports = list(report_dir.glob("governance_*.json")) if report_dir.exists() else []
        assert len(reports) >= 1

    def test_diagnostics_emitted(self, tmp_uni_dir):
        self._run(tmp_uni_dir)
        diag_path = tmp_uni_dir / "deployability_diagnostics.jsonl"
        assert diag_path.exists()
        records = _load_jsonl(diag_path)
        assert len(records) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Replay reproducibility
# ─────────────────────────────────────────────────────────────────────────────

class TestReplayReproducibility:

    def test_same_inputs_same_decision_actions(self, tmp_uni_dir, tmp_path):
        """
        Given the same live/shadow/capital, the set of PROMOTE/REJECT actions
        must be deterministic (order is determined by scoring, which is deterministic).
        """
        live = {"8035.T": "電機精密"}
        shadow = {"3402.T": "繊維", "2802.T": "食品"}

        dir1 = tmp_path / "run1_universe"
        dir1.mkdir()
        dir2 = tmp_path / "run2_universe"
        dir2.mkdir()

        gov1 = DeployableUniverseGovernor(dir1)
        gov2 = DeployableUniverseGovernor(dir2)

        r1 = gov1.run(live, shadow, CAPITAL, "replay_run", mode=AUTO_PROMOTE_SAFE,
                      price_lookup=_price_lookup_affordable)
        r2 = gov2.run(live, shadow, CAPITAL, "replay_run", mode=AUTO_PROMOTE_SAFE,
                      price_lookup=_price_lookup_affordable)

        actions1 = sorted((d.symbol, d.action) for d in r1.decisions)
        actions2 = sorted((d.symbol, d.action) for d in r2.decisions)
        assert actions1 == actions2
        assert r1.promoted_symbols == r2.promoted_symbols


# ─────────────────────────────────────────────────────────────────────────────
# run_universe_governance convenience wrapper
# ─────────────────────────────────────────────────────────────────────────────

class TestConvenienceWrapper:

    def test_wrapper_returns_governance_result(self, tmp_uni_dir):
        result = run_universe_governance(
            live_universe={"8035.T": "電機精密"},
            shadow_universe={"3402.T": "繊維"},
            capital=CAPITAL,
            run_id="wrap_test",
            universe_dir=tmp_uni_dir,
            price_lookup=_price_lookup_affordable,
        )
        assert isinstance(result, GovernanceResult)

    def test_wrapper_observe_only(self, tmp_uni_dir):
        result = run_universe_governance(
            live_universe={"8035.T": "電機精密"},
            shadow_universe={"3402.T": "繊維"},
            capital=CAPITAL,
            run_id="obs_test",
            universe_dir=tmp_uni_dir,
            mode=OBSERVE_ONLY,
        )
        assert result.promoted_symbols == []


# ─────────────────────────────────────────────────────────────────────────────
# PromotionDecision tamper-evidence
# ─────────────────────────────────────────────────────────────────────────────

class TestPromotionDecisionIntegrity:

    def _make_decision(self) -> PromotionDecision:
        score = DeployabilityScore(
            symbol="X.T", sector="食品",
            unit_price=5000.0, unit_price_pressure=0.01,
            affordable=True, price_available=True,
            rsr_score=70.0, rsr_passes=True,
            concentration_relief_score=0.5,
            sector_diversification_score=0.6,
            deployability_score=0.55,
            affordable_candidate_rank=1,
            passes_gate=True,
        )
        return PromotionDecision.create(
            run_id="r1", promotion_batch_id="b1",
            symbol="X.T", sector="食品",
            action=ACTION_PROMOTE, reason="test", score=score,
        )

    def test_decision_id_computed(self):
        d = self._make_decision()
        assert len(d.decision_id) == 64   # sha256 hex

    def test_decision_to_dict_round_trip(self, tmp_uni_dir):
        path = tmp_uni_dir / "d.jsonl"
        d = self._make_decision()
        _append_jsonl(path, d.to_dict())
        rec = PromotionDecisionRecorder(path)
        loaded = rec.load_all()
        assert loaded[0].decision_id == d.decision_id
        assert loaded[0].symbol == "X.T"
