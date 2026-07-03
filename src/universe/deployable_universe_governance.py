"""
deployable_universe_governance.py — Deployable universe governance layer

Converts the existing shadow/live universe observation layer into a fully integrated
deployable-universe governance system. Does NOT redesign universe logic — only adds
integration, governance, manifest updates, and observability.

Deployment modes:
  OBSERVE_ONLY       — score and log, no promotions
  AUTO_PROMOTE_SAFE  — automatic promotion with safety constraints (default)
  MANUAL_APPROVAL    — log candidates only, require external approval

AUTO_PROMOTE_SAFE constraints:
  - max 2 promotions per rebalance
  - cooldown after demotion (DEMOTION_COOLDOWN_DAYS)
  - append-only manifests
  - deterministic ordering (score desc → symbol asc as tiebreaker)
  - no silent removals
  - no mutable history
  - replay-compatible lineage

Fail policy:
  - promotion failures → FAIL_OPEN (log, skip, continue execution)
  - manifest corruption → FAIL_CLOSED (raise RuntimeError)
  - replay inconsistency → FAIL_CLOSED (raise RuntimeError)
  - stale market data → reject promotion (FAIL_OPEN for execution)
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Deployment modes ──────────────────────────────────────────────────────────
OBSERVE_ONLY      = "OBSERVE_ONLY"
AUTO_PROMOTE_SAFE = "AUTO_PROMOTE_SAFE"
MANUAL_APPROVAL   = "MANUAL_APPROVAL"
DEFAULT_MODE      = AUTO_PROMOTE_SAFE

# ── AUTO_PROMOTE_SAFE hard constraints ────────────────────────────────────────
MAX_PROMOTIONS_PER_REBALANCE = 2
DEMOTION_COOLDOWN_DAYS       = 30
UNIVERSE_CAP                 = 60   # hard ceiling on live universe size

# ── Deployability thresholds ──────────────────────────────────────────────────
MIN_DEPLOYABILITY_SCORE  = 0.40   # composite gate
LOT_SIZE                 = 100    # 東証標準単元株数
MAX_UNIT_PRICE_FRACTION  = 0.30   # unit cost ≤ capital × 30%
MIN_RSR_FOR_SHADOW       = 60.0   # shadow RSR gate (if RSR available)
MAX_SECTOR_FRACTION      = 0.35   # single-sector fraction of live universe

# ── Action constants ──────────────────────────────────────────────────────────
ACTION_PROMOTE  = "PROMOTE"
ACTION_OBSERVE  = "OBSERVE"
ACTION_REJECT   = "REJECT"
ACTION_COOLDOWN = "COOLDOWN"

# ── Phase 2: EV-based replacement gate ───────────────────────────────────────
# Imported from universe_ev_engine; duplicated here for zero circular-import risk.
ROTATION_COST_EV: float = 0.05     # EV penalty for any swap (commission+slippage proxy)
PROMOTION_EV_THRESHOLD: float = 0.05  # net_advantage must clear this to justify promotion
EXPLORATION_SLOTS_PER_BATCH: int = 1  # symbols with high uncertainty bypass EV gate once/batch

# ── Schema version ───────────────────────────────────────────────────────────
SCHEMA_VERSION = 1


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical(d: dict) -> str:
    return json.dumps(d, sort_keys=True, ensure_ascii=False)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _load_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception as exc:
            logger.warning("[GOVERNANCE] JSONL parse error %s: %s", path.name, exc)
    return out


def _write_atomic(path: Path, content: str) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DeployabilityScore:
    """Deployability assessment for a single shadow candidate."""
    symbol: str
    sector: str
    unit_price: float                    # last known price × LOT_SIZE
    unit_price_pressure: float           # unit_price / (capital × MAX_UNIT_PRICE_FRACTION)
    affordable: bool                     # unit_price ≤ capital × MAX_UNIT_PRICE_FRACTION
    price_available: bool                # False if no price data
    rsr_score: float                     # RSR percentile (0 if unavailable)
    rsr_passes: bool                     # rsr_score ≥ MIN_RSR_FOR_SHADOW; False when missing (FAIL_CLOSED)
    concentration_relief_score: float    # 0.0–1.0: higher = reduces sector concentration
    sector_diversification_score: float  # 0.0–1.0: higher = adds new sector breadth
    deployability_score: float           # composite 0.0–1.0
    affordable_candidate_rank: int       # rank among affordable candidates (1=best)
    passes_gate: bool                    # all gates pass
    rsr_source: str = ""                 # "snapshot" | "mtf" | "missing"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PromotionDecision:
    """Immutable record of a promotion/rejection decision."""
    decision_id: str          # sha256 of canonical fields (tamper-evident)
    run_id: str
    promotion_batch_id: str
    symbol: str
    sector: str
    action: str               # ACTION_PROMOTE | ACTION_OBSERVE | ACTION_REJECT | ACTION_COOLDOWN
    reason: str
    deployability_score: float
    unit_price: float
    unit_price_pressure: float
    concentration_relief_score: float
    sector_diversification_score: float
    rsr_score: float
    created_at: str
    schema_version: int = SCHEMA_VERSION
    rsr_source: str = ""                 # "snapshot" | "mtf" | "missing"

    @staticmethod
    def create(
        run_id: str,
        promotion_batch_id: str,
        symbol: str,
        sector: str,
        action: str,
        reason: str,
        score: "DeployabilityScore",
    ) -> "PromotionDecision":
        created_at = _now_utc()
        fields = {
            "action": action,
            "created_at": created_at,
            "deployability_score": round(score.deployability_score, 6),
            "promotion_batch_id": promotion_batch_id,
            "reason": reason,
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "sector": sector,
            "symbol": symbol,
            "unit_price": round(score.unit_price, 2),
            "unit_price_pressure": round(score.unit_price_pressure, 4),
            "concentration_relief_score": round(score.concentration_relief_score, 4),
            "sector_diversification_score": round(score.sector_diversification_score, 4),
            "rsr_score": round(score.rsr_score, 2),
        }
        decision_id = _sha256(_canonical(fields))
        return PromotionDecision(
            decision_id=decision_id,
            run_id=run_id,
            promotion_batch_id=promotion_batch_id,
            symbol=symbol,
            sector=sector,
            action=action,
            reason=reason,
            deployability_score=round(score.deployability_score, 6),
            unit_price=round(score.unit_price, 2),
            unit_price_pressure=round(score.unit_price_pressure, 4),
            concentration_relief_score=round(score.concentration_relief_score, 4),
            sector_diversification_score=round(score.sector_diversification_score, 4),
            rsr_score=round(score.rsr_score, 2),
            created_at=created_at,
            rsr_source=score.rsr_source,
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class UniverseManifest:
    """Immutable snapshot of the live universe at a point in time."""
    manifest_id: str          # sha256 of canonical fields
    run_id: str
    promotion_batch_id: str
    live_symbols: List[str]   # sorted list (deterministic)
    promoted_symbols: List[str]
    universe_size: int
    previous_manifest_id: str  # hash chain
    created_at: str
    schema_version: int = SCHEMA_VERSION

    @staticmethod
    def create(
        run_id: str,
        promotion_batch_id: str,
        live_symbols: List[str],
        promoted_symbols: List[str],
        previous_manifest_id: str = "",
    ) -> "UniverseManifest":
        created_at = _now_utc()
        syms_sorted = sorted(live_symbols)
        fields = {
            "created_at": created_at,
            "live_symbols": syms_sorted,
            "previous_manifest_id": previous_manifest_id,
            "promoted_symbols": sorted(promoted_symbols),
            "promotion_batch_id": promotion_batch_id,
            "run_id": run_id,
            "schema_version": SCHEMA_VERSION,
            "universe_size": len(syms_sorted),
        }
        manifest_id = _sha256(_canonical(fields))
        return UniverseManifest(
            manifest_id=manifest_id,
            run_id=run_id,
            promotion_batch_id=promotion_batch_id,
            live_symbols=syms_sorted,
            promoted_symbols=sorted(promoted_symbols),
            universe_size=len(syms_sorted),
            previous_manifest_id=previous_manifest_id,
            created_at=created_at,
        )

    def validate(self) -> bool:
        """Recompute manifest_id and verify tamper-evidence."""
        fields = {
            "created_at": self.created_at,
            "live_symbols": sorted(self.live_symbols),
            "previous_manifest_id": self.previous_manifest_id,
            "promoted_symbols": sorted(self.promoted_symbols),
            "promotion_batch_id": self.promotion_batch_id,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "universe_size": self.universe_size,
        }
        return _sha256(_canonical(fields)) == self.manifest_id

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GovernanceReport:
    """Daily governance summary."""
    run_id: str
    promotion_batch_id: str
    mode: str
    promoted: List[str]
    rejected: List[str]
    on_cooldown: List[str]
    universe_size_before: int
    universe_size_after: int
    deployable_breadth_score: float      # fraction of affordable shadow candidates
    affordable_opportunity_coverage: float  # affordable shadow / (live + affordable shadow)
    high_price_exclusion_count: int      # shadow candidates excluded by price
    sector_distribution: Dict[str, int]  # sector → count in updated live universe
    idle_cash_pressure: float            # proxy: 1 - (affordable_count / max_promotions)
    concentration_relief_contribution: float  # avg concentration_relief_score of promoted
    created_at: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class GovernanceResult:
    """Result of a governance run."""
    updated_live_universe: Dict[str, str]   # {symbol: sector} — may include promotions
    promoted_symbols: List[str]
    governance_report: GovernanceReport
    manifest: Optional[UniverseManifest]
    decisions: List[PromotionDecision]
    manifest_written: bool
    universe_file_updated: bool


# ─────────────────────────────────────────────────────────────────────────────
# DeployabilityScorer
# ─────────────────────────────────────────────────────────────────────────────

class DeployabilityScorer:
    """
    Scores shadow candidates for deployability under current capital constraints.

    Scoring factors:
      - affordability (unit price ≤ capital × MAX_UNIT_PRICE_FRACTION)
      - RSR persistence (rsr_score ≥ MIN_RSR_FOR_SHADOW if available)
      - concentration relief (does this symbol reduce sector concentration?)
      - sector diversification (is sector underrepresented in live universe?)
    """

    def __init__(
        self,
        capital: int,
        live_universe: Dict[str, str],
    ) -> None:
        self._capital = capital
        self._max_unit_cost = capital * MAX_UNIT_PRICE_FRACTION
        self._live_sector_counts = self._count_sectors(live_universe)
        self._live_size = len(live_universe)

    @staticmethod
    def _count_sectors(universe: Dict[str, str]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for sec in universe.values():
            counts[sec] = counts.get(sec, 0) + 1
        return counts

    def score_candidate(
        self,
        symbol: str,
        sector: str,
        price_lookup: Optional[Callable[[str], Optional[float]]],
        rsr_scores: Optional[Dict[str, float]],
    ) -> DeployabilityScore:
        # ── price / affordability ─────────────────────────────────────────────
        price = None
        if price_lookup is not None:
            try:
                price = price_lookup(symbol)
            except Exception:
                price = None

        price_available = price is not None and price > 0
        unit_price = float(price * LOT_SIZE) if price_available else 0.0
        unit_price_pressure = (
            unit_price / self._max_unit_cost if self._max_unit_cost > 0 else 999.0
        )
        affordable = price_available and unit_price <= self._max_unit_cost

        # ── RSR persistence (FAIL_CLOSED: missing RSR blocks promotion) ─────────
        rsr_score = 0.0
        rsr_source = "missing"
        if rsr_scores is not None:
            rsr_score = float(rsr_scores.get(symbol, 0.0))
            rsr_source = ""  # caller sets actual source via score_all() rsr_source param
        rsr_passes = rsr_scores is not None and rsr_score >= MIN_RSR_FOR_SHADOW

        # ── sector diversification score ──────────────────────────────────────
        live_count_in_sector = self._live_sector_counts.get(sector, 0)
        if self._live_size > 0:
            sector_fraction = live_count_in_sector / self._live_size
            # High sector fraction → low diversification benefit
            sector_diversification_score = max(0.0, 1.0 - sector_fraction / MAX_SECTOR_FRACTION)
        else:
            sector_diversification_score = 1.0

        # ── concentration relief score ─────────────────────────────────────────
        # Relief = inverse of current sector concentration.
        # Adding a symbol in an underweight sector → high relief.
        max_sector_count = max(self._live_sector_counts.values()) if self._live_sector_counts else 1
        concentration_relief_score = (
            1.0 - live_count_in_sector / max_sector_count
            if max_sector_count > 0
            else 0.5
        )

        # ── composite deployability score ─────────────────────────────────────
        # Weights: affordability=0 (binary gate), RSR=0.30, sector_div=0.35, conc_relief=0.35
        # Price pressure bonus: lower pressure → higher score
        price_efficiency = max(0.0, 1.0 - unit_price_pressure) if price_available else 0.0
        rsr_component = min(1.0, rsr_score / 100.0) if rsr_scores is not None else 0.5

        deployability_score = (
            0.30 * rsr_component
            + 0.35 * sector_diversification_score
            + 0.25 * concentration_relief_score
            + 0.10 * price_efficiency
        )
        deployability_score = round(max(0.0, min(1.0, deployability_score)), 6)

        passes_gate = (
            affordable
            and rsr_passes
            and deployability_score >= MIN_DEPLOYABILITY_SCORE
        )

        return DeployabilityScore(
            symbol=symbol,
            sector=sector,
            unit_price=unit_price,
            unit_price_pressure=round(unit_price_pressure, 4),
            affordable=affordable,
            price_available=price_available,
            rsr_score=rsr_score,
            rsr_passes=rsr_passes,
            concentration_relief_score=round(concentration_relief_score, 4),
            sector_diversification_score=round(sector_diversification_score, 4),
            deployability_score=deployability_score,
            affordable_candidate_rank=0,   # set after sorting
            passes_gate=passes_gate,
            rsr_source=rsr_source,
        )

    def score_all(
        self,
        shadow_universe: Dict[str, str],
        price_lookup: Optional[Callable[[str], Optional[float]]],
        rsr_scores: Optional[Dict[str, float]],
        rsr_source: str = "",
    ) -> List[DeployabilityScore]:
        """Score all shadow candidates. Returns list sorted by rank (desc score → asc symbol)."""
        scores = []
        for sym in sorted(shadow_universe.keys()):   # deterministic input order
            sec = shadow_universe[sym]
            s = self.score_candidate(sym, sec, price_lookup, rsr_scores)
            if rsr_source and s.rsr_source != "missing":
                s.rsr_source = rsr_source
            scores.append(s)

        # Rank affordable candidates (desc deployability_score, asc symbol as tiebreaker)
        affordable = sorted(
            [s for s in scores if s.affordable],
            key=lambda s: (-s.deployability_score, s.symbol),
        )
        for rank, s in enumerate(affordable, start=1):
            s.affordable_candidate_rank = rank

        return scores


# ─────────────────────────────────────────────────────────────────────────────
# UniverseManifestAuthority
# ─────────────────────────────────────────────────────────────────────────────

class UniverseManifestAuthority:
    """
    Reads and writes the append-only universe manifest JSONL.
    Validates hash chain on load.
    Raises RuntimeError on corruption (FAIL_CLOSED).
    """

    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path

    def load_all(self) -> List[UniverseManifest]:
        records = _load_jsonl(self._path)
        manifests = []
        for r in records:
            try:
                m = UniverseManifest(
                    manifest_id=r["manifest_id"],
                    run_id=r["run_id"],
                    promotion_batch_id=r.get("promotion_batch_id", ""),
                    live_symbols=r["live_symbols"],
                    promoted_symbols=r.get("promoted_symbols", []),
                    universe_size=int(r["universe_size"]),
                    previous_manifest_id=r.get("previous_manifest_id", ""),
                    created_at=r["created_at"],
                    schema_version=int(r.get("schema_version", SCHEMA_VERSION)),
                )
                manifests.append(m)
            except Exception as exc:
                raise RuntimeError(
                    f"[MANIFEST] corrupt record in {self._path}: {exc}"
                ) from exc
        return manifests

    def latest(self) -> Optional[UniverseManifest]:
        manifests = self.load_all()
        return manifests[-1] if manifests else None

    def validate_chain(self) -> bool:
        """Validate hash chain integrity. Returns True if valid."""
        manifests = self.load_all()
        for i, m in enumerate(manifests):
            if not m.validate():
                logger.error("[MANIFEST] tampered manifest_id at index %d: %s", i, m.manifest_id)
                return False
            if i > 0 and m.previous_manifest_id != manifests[i - 1].manifest_id:
                logger.error(
                    "[MANIFEST] broken hash chain at index %d: expected %s got %s",
                    i, manifests[i - 1].manifest_id, m.previous_manifest_id,
                )
                return False
        return True

    def append(self, manifest: UniverseManifest) -> None:
        _append_jsonl(self._path, manifest.to_dict())
        logger.info(
            "[MANIFEST] appended manifest_id=%s promoted=%s size=%d",
            manifest.manifest_id[:12],
            manifest.promoted_symbols,
            manifest.universe_size,
        )


# ─────────────────────────────────────────────────────────────────────────────
# PromotionDecisionRecorder
# ─────────────────────────────────────────────────────────────────────────────

class PromotionDecisionRecorder:
    """
    Reads cooldown state from JSONL and records new decisions.
    Cooldown: if a symbol was demoted, it cannot be re-promoted for DEMOTION_COOLDOWN_DAYS.
    """

    def __init__(self, decisions_path: Path) -> None:
        self._path = decisions_path

    def load_all(self) -> List[PromotionDecision]:
        records = _load_jsonl(self._path)
        out = []
        for r in records:
            try:
                out.append(PromotionDecision(
                    decision_id=r.get("decision_id", ""),
                    run_id=r.get("run_id", ""),
                    promotion_batch_id=r.get("promotion_batch_id", ""),
                    symbol=r["symbol"],
                    sector=r.get("sector", ""),
                    action=r["action"],
                    reason=r.get("reason", ""),
                    deployability_score=float(r.get("deployability_score", 0.0)),
                    unit_price=float(r.get("unit_price", 0.0)),
                    unit_price_pressure=float(r.get("unit_price_pressure", 0.0)),
                    concentration_relief_score=float(r.get("concentration_relief_score", 0.0)),
                    sector_diversification_score=float(r.get("sector_diversification_score", 0.0)),
                    rsr_score=float(r.get("rsr_score", 0.0)),
                    created_at=r.get("created_at", ""),
                    schema_version=int(r.get("schema_version", SCHEMA_VERSION)),
                    rsr_source=r.get("rsr_source", ""),
                ))
            except Exception as exc:
                logger.warning("[DECISIONS] parse error: %s", exc)
        return out

    def get_cooldown_symbols(self) -> Dict[str, datetime]:
        """
        Returns {symbol: demotion_date} for symbols still within DEMOTION_COOLDOWN_DAYS.
        A 'demotion' here means: the symbol was previously in live universe and then
        its last governance action was REJECT (after being promoted before).
        Practical use: if a symbol was promoted then rejected, apply cooldown.
        """
        decisions = self.load_all()
        # Track last decision per symbol
        last_decision: Dict[str, PromotionDecision] = {}
        for d in decisions:
            last_decision[d.symbol] = d

        cooldown: Dict[str, datetime] = {}
        cutoff_dt = datetime.now(timezone.utc) - timedelta(days=DEMOTION_COOLDOWN_DAYS)

        for sym, d in last_decision.items():
            if d.action != ACTION_REJECT:
                continue
            try:
                demotion_dt = datetime.fromisoformat(d.created_at.replace("Z", "+00:00"))
                if demotion_dt >= cutoff_dt:
                    cooldown[sym] = demotion_dt
            except Exception:
                pass
        return cooldown

    def append(self, decision: PromotionDecision) -> None:
        _append_jsonl(self._path, decision.to_dict())

    def append_batch(self, decisions: List[PromotionDecision]) -> None:
        for d in decisions:
            self.append(d)


# ─────────────────────────────────────────────────────────────────────────────
# UniverseLineageTracker
# ─────────────────────────────────────────────────────────────────────────────

class UniverseLineageTracker:
    """Append-only lineage record of universe state transitions."""

    def __init__(self, lineage_path: Path) -> None:
        self._path = lineage_path

    def record(
        self,
        run_id: str,
        batch_id: str,
        action: str,
        live_before: List[str],
        live_after: List[str],
        promoted: List[str],
        manifest_id: str,
    ) -> None:
        entry = {
            "run_id": run_id,
            "batch_id": batch_id,
            "action": action,
            "live_before_size": len(live_before),
            "live_after_size": len(live_after),
            "promoted": sorted(promoted),
            "manifest_id": manifest_id,
            "lineage_hash": _sha256(_canonical({
                "live_after": sorted(live_after),
                "manifest_id": manifest_id,
                "run_id": run_id,
            })),
            "created_at": _now_utc(),
            "schema_version": SCHEMA_VERSION,
        }
        _append_jsonl(self._path, entry)


# ─────────────────────────────────────────────────────────────────────────────
# UniverseBreadthDiagnostics
# ─────────────────────────────────────────────────────────────────────────────

class UniverseBreadthDiagnostics:
    """Generates daily universe breadth diagnostics."""

    def __init__(self, diagnostics_path: Path) -> None:
        self._path = diagnostics_path

    def emit(
        self,
        run_id: str,
        live_universe: Dict[str, str],
        shadow_universe: Dict[str, str],
        scores: List[DeployabilityScore],
        capital: int,
        promoted: List[str],
        report: GovernanceReport,
    ) -> None:
        live_sectors = {}
        for sec in live_universe.values():
            live_sectors[sec] = live_sectors.get(sec, 0) + 1

        affordable_shadow = [s for s in scores if s.affordable and s.symbol in shadow_universe]
        high_price_excluded = [s for s in scores if not s.affordable and s.price_available]
        no_price = [s for s in scores if not s.price_available]

        # idle_cash_attributable_to_universe_scarcity:
        # proxy = how many available slots remain due to unaffordable/unavailable universe
        max_alloc = capital * MAX_UNIT_PRICE_FRACTION
        entry = {
            "run_id": run_id,
            "created_at": _now_utc(),
            "live_universe_size": len(live_universe),
            "shadow_universe_size": len(shadow_universe),
            "affordable_shadow_count": len(affordable_shadow),
            "high_price_exclusion_count": len(high_price_excluded),
            "no_price_data_count": len(no_price),
            "promoted_count": len(promoted),
            "promoted_symbols": sorted(promoted),
            "deployable_breadth_score": report.deployable_breadth_score,
            "affordable_opportunity_coverage": report.affordable_opportunity_coverage,
            "idle_cash_pressure": report.idle_cash_pressure,
            "concentration_relief_contribution": report.concentration_relief_contribution,
            "live_sector_distribution": live_sectors,
            "max_unit_allocation": max_alloc,
            "capital": capital,
            "schema_version": SCHEMA_VERSION,
        }
        _append_jsonl(self._path, entry)
        logger.info(
            "[DIAGNOSTICS] breadth=%.2f affordable_shadow=%d high_price_excl=%d promoted=%d",
            report.deployable_breadth_score,
            len(affordable_shadow),
            len(high_price_excluded),
            len(promoted),
        )


# ─────────────────────────────────────────────────────────────────────────────
# DeployableUniverseGovernor
# ─────────────────────────────────────────────────────────────────────────────

class DeployableUniverseGovernor:
    """
    Main orchestrator for deployable universe governance.

    Wires together: scorer, manifest authority, decision recorder,
    lineage tracker, diagnostics.

    Fail policy (see module docstring):
      - Promotion failures: FAIL_OPEN (log, skip)
      - Manifest corruption: FAIL_CLOSED (RuntimeError propagates)
      - Stale market data: reject promotion candidate (symbol skipped)
    """

    def __init__(
        self,
        universe_dir: Path,
        manifest_path: Optional[Path] = None,
        decisions_path: Optional[Path] = None,
        lineage_path: Optional[Path] = None,
        diagnostics_path: Optional[Path] = None,
        report_dir: Optional[Path] = None,
    ) -> None:
        self._universe_dir = universe_dir
        self._manifest_path  = manifest_path  or (universe_dir / "universe_manifest.jsonl")
        self._decisions_path = decisions_path or (universe_dir / "promotion_decisions.jsonl")
        self._lineage_path   = lineage_path   or (universe_dir / "universe_lineage.jsonl")
        self._diag_path      = diagnostics_path or (universe_dir / "deployability_diagnostics.jsonl")
        self._report_dir     = report_dir or (universe_dir / "universe_governance_reports")

        self._manifest_auth = UniverseManifestAuthority(self._manifest_path)
        self._decision_rec  = PromotionDecisionRecorder(self._decisions_path)
        self._lineage       = UniverseLineageTracker(self._lineage_path)
        self._diagnostics   = UniverseBreadthDiagnostics(self._diag_path)

    # ── Public API ────────────────────────────────────────────────────────────

    def run(
        self,
        live_universe: Dict[str, str],
        shadow_universe: Dict[str, str],
        capital: int,
        run_id: str,
        mode: str = DEFAULT_MODE,
        price_lookup: Optional[Callable[[str], Optional[float]]] = None,
        rsr_scores: Optional[Dict[str, float]] = None,
        universe_file: Optional[Path] = None,
        candidate_ev_scores: Optional[Dict[str, float]] = None,
        live_ev_floor: Optional[float] = None,
        rsr_source: str = "",
    ) -> GovernanceResult:
        """
        Run universe governance evaluation.

        Args:
            live_universe:  current live universe {symbol: sector}
            shadow_universe: shadow candidates {symbol: sector}
            capital:        total portfolio capital (JPY)
            run_id:         execution run ID
            mode:           OBSERVE_ONLY | AUTO_PROMOTE_SAFE | MANUAL_APPROVAL
            price_lookup:   callable(symbol) → float|None; uses local cache prices
            rsr_scores:     optional {symbol: rsr_score}; if None, RSR gate is skipped
            universe_file:  if provided and promotions occur, update this JSON file

        Returns:
            GovernanceResult with updated universe + governance artifacts
        """
        batch_id = _sha256(f"{run_id}:universe_governance")[:16]
        live_before = sorted(live_universe.keys())

        # ── Validate manifest integrity (FAIL_CLOSED) ─────────────────────────
        if self._manifest_path.exists():
            if not self._manifest_auth.validate_chain():
                raise RuntimeError(
                    "[GOVERNANCE] universe manifest hash chain invalid — "
                    "possible corruption. Halting governance to preserve replay integrity."
                )

        # ── Score all shadow candidates ────────────────────────────────────────
        scorer = DeployabilityScorer(capital, live_universe)
        scores = scorer.score_all(shadow_universe, price_lookup, rsr_scores, rsr_source=rsr_source)

        # ── Load cooldowns ─────────────────────────────────────────────────────
        cooldown_symbols = self._decision_rec.get_cooldown_symbols()

        # ── Apply promotion policy ─────────────────────────────────────────────
        decisions: List[PromotionDecision] = []
        promoted_symbols: List[str] = []
        updated_live = dict(live_universe)   # mutable copy

        if mode == OBSERVE_ONLY or mode == MANUAL_APPROVAL:
            # Score and record but do not promote
            for s in scores:
                action = ACTION_OBSERVE
                reason = f"mode={mode}: observation only"
                decisions.append(PromotionDecision.create(
                    run_id=run_id, promotion_batch_id=batch_id,
                    symbol=s.symbol, sector=s.sector,
                    action=action, reason=reason, score=s,
                ))
        else:
            # AUTO_PROMOTE_SAFE
            decisions, promoted_symbols, updated_live = self._apply_auto_promote_safe(
                scores=scores,
                cooldown_symbols=cooldown_symbols,
                live_universe=live_universe,
                capital=capital,
                run_id=run_id,
                batch_id=batch_id,
                updated_live=updated_live,
                candidate_ev_scores=candidate_ev_scores,
                live_ev_floor=live_ev_floor,
            )

        # ── Compute governance report ──────────────────────────────────────────
        report = self._build_report(
            run_id=run_id,
            batch_id=batch_id,
            mode=mode,
            scores=scores,
            decisions=decisions,
            promoted=promoted_symbols,
            shadow_universe=shadow_universe,
            live_before=live_universe,
            live_after=updated_live,
            capital=capital,
        )

        # ── Write decisions (append-only, FAIL_OPEN) ──────────────────────────
        try:
            self._decision_rec.append_batch(decisions)
        except Exception as exc:
            logger.warning("[GOVERNANCE] decision record write failed (%s) — continuing", exc)

        # ── Write manifest + lineage if promotions occurred ────────────────────
        manifest: Optional[UniverseManifest] = None
        manifest_written = False
        universe_file_updated = False

        if promoted_symbols or not self._manifest_path.exists():
            try:
                prev = self._manifest_auth.latest()
                prev_id = prev.manifest_id if prev else ""
                manifest = UniverseManifest.create(
                    run_id=run_id,
                    promotion_batch_id=batch_id,
                    live_symbols=sorted(updated_live.keys()),
                    promoted_symbols=promoted_symbols,
                    previous_manifest_id=prev_id,
                )
                self._manifest_auth.append(manifest)
                manifest_written = True
            except RuntimeError:
                raise   # FAIL_CLOSED: corrupt manifest
            except Exception as exc:
                logger.warning("[GOVERNANCE] manifest write failed (%s) — continuing", exc)

        if manifest is not None:
            try:
                self._lineage.record(
                    run_id=run_id,
                    batch_id=batch_id,
                    action="promotion" if promoted_symbols else "no_change",
                    live_before=live_before,
                    live_after=sorted(updated_live.keys()),
                    promoted=promoted_symbols,
                    manifest_id=manifest.manifest_id,
                )
            except Exception as exc:
                logger.warning("[GOVERNANCE] lineage write failed (%s) — continuing", exc)

        # ── Update universe JSON file if promotions happened ──────────────────
        if promoted_symbols and universe_file is not None:
            try:
                universe_file_updated = self._update_universe_file(
                    universe_file=universe_file,
                    updated_live=updated_live,
                    promoted=promoted_symbols,
                    manifest_id=manifest.manifest_id if manifest else "",
                )
            except Exception as exc:
                logger.warning("[GOVERNANCE] universe file update failed (%s) — continuing", exc)

        # ── Emit diagnostics (FAIL_OPEN) ───────────────────────────────────────
        try:
            self._diagnostics.emit(
                run_id=run_id,
                live_universe=updated_live,
                shadow_universe=shadow_universe,
                scores=scores,
                capital=capital,
                promoted=promoted_symbols,
                report=report,
            )
        except Exception as exc:
            logger.warning("[GOVERNANCE] diagnostics emit failed (%s) — continuing", exc)

        # ── Write daily governance report ──────────────────────────────────────
        try:
            self._write_daily_report(report, scores)
        except Exception as exc:
            logger.warning("[GOVERNANCE] daily report write failed (%s) — continuing", exc)

        logger.info(
            "[GOVERNANCE] mode=%s promoted=%d universe=%d→%d batch=%s",
            mode,
            len(promoted_symbols),
            len(live_universe),
            len(updated_live),
            batch_id,
        )

        return GovernanceResult(
            updated_live_universe=updated_live,
            promoted_symbols=promoted_symbols,
            governance_report=report,
            manifest=manifest,
            decisions=decisions,
            manifest_written=manifest_written,
            universe_file_updated=universe_file_updated,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_auto_promote_safe(
        self,
        scores: List[DeployabilityScore],
        cooldown_symbols: Dict[str, datetime],
        live_universe: Dict[str, str],
        capital: int,
        run_id: str,
        batch_id: str,
        updated_live: Dict[str, str],
        candidate_ev_scores: Optional[Dict[str, float]] = None,
        live_ev_floor: Optional[float] = None,
        exploration_slots: int = EXPLORATION_SLOTS_PER_BATCH,
    ) -> Tuple[List[PromotionDecision], List[str], Dict[str, str]]:
        decisions: List[PromotionDecision] = []
        promoted: List[str] = []
        current_live = dict(updated_live)

        # Already in live → skip
        already_live = set(live_universe.keys())

        # Sector gate: no single sector > MAX_SECTOR_FRACTION of universe
        sector_counts: Dict[str, int] = {}
        for sec in current_live.values():
            sector_counts[sec] = sector_counts.get(sec, 0) + 1
        live_size_for_gate = len(current_live)

        # Sort candidates: desc deployability_score, asc symbol (deterministic)
        qualified: List[DeployabilityScore] = []
        for s in sorted(scores, key=lambda x: (-x.deployability_score, x.symbol)):
            if s.symbol in already_live:
                continue
            if s.symbol in cooldown_symbols:
                decisions.append(PromotionDecision.create(
                    run_id=run_id, promotion_batch_id=batch_id,
                    symbol=s.symbol, sector=s.sector,
                    action=ACTION_COOLDOWN,
                    reason=f"demotion cooldown until {cooldown_symbols[s.symbol].date().isoformat()}",
                    score=s,
                ))
                continue
            if not s.passes_gate:
                reason = self._rejection_reason(s)
                decisions.append(PromotionDecision.create(
                    run_id=run_id, promotion_batch_id=batch_id,
                    symbol=s.symbol, sector=s.sector,
                    action=ACTION_REJECT, reason=reason, score=s,
                ))
                continue
            qualified.append(s)

        # Universe cap check
        cap_remaining = UNIVERSE_CAP - live_size_for_gate

        # Promote top-N qualified (max MAX_PROMOTIONS_PER_REBALANCE, respecting cap + sector)
        promotions_this_batch = 0
        explorations_used = 0   # Phase 2: track exploration budget consumption
        for s in qualified:
            if promotions_this_batch >= MAX_PROMOTIONS_PER_REBALANCE:
                decisions.append(PromotionDecision.create(
                    run_id=run_id, promotion_batch_id=batch_id,
                    symbol=s.symbol, sector=s.sector,
                    action=ACTION_OBSERVE,
                    reason=f"batch_limit={MAX_PROMOTIONS_PER_REBALANCE} reached",
                    score=s,
                ))
                continue
            if cap_remaining <= 0:
                decisions.append(PromotionDecision.create(
                    run_id=run_id, promotion_batch_id=batch_id,
                    symbol=s.symbol, sector=s.sector,
                    action=ACTION_REJECT,
                    reason=f"universe_cap={UNIVERSE_CAP} reached",
                    score=s,
                ))
                continue
            # Sector concentration gate
            current_sector_count = sector_counts.get(s.sector, 0)
            projected_fraction = (
                (current_sector_count + 1) / (live_size_for_gate + 1)
                if live_size_for_gate >= 0 else 0.0
            )
            if projected_fraction > MAX_SECTOR_FRACTION:
                decisions.append(PromotionDecision.create(
                    run_id=run_id, promotion_batch_id=batch_id,
                    symbol=s.symbol, sector=s.sector,
                    action=ACTION_REJECT,
                    reason=(
                        f"sector_concentration={projected_fraction:.2f} > "
                        f"max={MAX_SECTOR_FRACTION} ({s.sector})"
                    ),
                    score=s,
                ))
                continue

            # ── Phase 2: replacement_delta EV gate ────────────────────────────
            # Only promote if candidate EV > weakest-live EV + rotation_cost.
            # Exploration budget: high-uncertainty candidates bypass this gate once/batch.
            _ev_gate_reason: Optional[str] = None
            if candidate_ev_scores is not None and live_ev_floor is not None:
                cand_ev = candidate_ev_scores.get(s.symbol)
                if cand_ev is not None:
                    net_adv = cand_ev - live_ev_floor - ROTATION_COST_EV
                    if net_adv < PROMOTION_EV_THRESHOLD:
                        # Check exploration budget (high-uncertainty symbols may bypass)
                        cand_n = candidate_ev_scores.get(f"__n_{s.symbol}", 20)
                        is_explore = (
                            int(cand_n) < 3           # high uncertainty
                            and explorations_used < exploration_slots
                        )
                        if is_explore:
                            explorations_used += 1
                            _ev_gate_reason = f"exploration net_ev={net_adv:.3f}"
                        else:
                            decisions.append(PromotionDecision.create(
                                run_id=run_id, promotion_batch_id=batch_id,
                                symbol=s.symbol, sector=s.sector,
                                action=ACTION_OBSERVE,
                                reason=(
                                    f"ev_gate: net_advantage={net_adv:.3f} "
                                    f"< threshold={PROMOTION_EV_THRESHOLD} "
                                    f"(candidate_ev={cand_ev:.3f} live_floor={live_ev_floor:.3f})"
                                ),
                                score=s,
                            ))
                            continue

            # ── Promote ────────────────────────────────────────────────────────
            _reason = (
                f"deployability_score={s.deployability_score:.3f} "
                f"rsr={s.rsr_score:.1f} sector={s.sector}"
            )
            if _ev_gate_reason:
                _reason += f" [{_ev_gate_reason}]"
            decisions.append(PromotionDecision.create(
                run_id=run_id, promotion_batch_id=batch_id,
                symbol=s.symbol, sector=s.sector,
                action=ACTION_PROMOTE,
                reason=_reason,
                score=s,
            ))
            current_live[s.symbol] = s.sector
            promoted.append(s.symbol)
            sector_counts[s.sector] = sector_counts.get(s.sector, 0) + 1
            live_size_for_gate += 1
            cap_remaining -= 1
            promotions_this_batch += 1

        return decisions, promoted, current_live

    @staticmethod
    def _rejection_reason(s: DeployabilityScore) -> str:
        if not s.price_available:
            return "stale_market_data: no price available"
        if not s.affordable:
            return (
                f"unit_price_too_high: ¥{s.unit_price:,.0f} "
                f"(pressure={s.unit_price_pressure:.2f})"
            )
        if not s.rsr_passes:
            return f"rsr_below_threshold: {s.rsr_score:.1f} < {MIN_RSR_FOR_SHADOW}"
        return f"deployability_score_too_low: {s.deployability_score:.3f} < {MIN_DEPLOYABILITY_SCORE}"

    @staticmethod
    def _build_report(
        run_id: str,
        batch_id: str,
        mode: str,
        scores: List[DeployabilityScore],
        decisions: List[PromotionDecision],
        promoted: List[str],
        shadow_universe: Dict[str, str],
        live_before: Dict[str, str],
        live_after: Dict[str, str],
        capital: int,
    ) -> GovernanceReport:
        rejected = [d.symbol for d in decisions if d.action == ACTION_REJECT]
        on_cooldown = [d.symbol for d in decisions if d.action == ACTION_COOLDOWN]

        affordable_shadow = [s for s in scores if s.affordable and s.symbol in shadow_universe]
        high_price_excl = [s for s in scores if not s.affordable and s.price_available]

        total_candidates = len(affordable_shadow) + len(live_after)
        affordable_coverage = (
            len(affordable_shadow) / total_candidates if total_candidates > 0 else 0.0
        )
        deployable_breadth = (
            len(affordable_shadow) / len(shadow_universe) if shadow_universe else 0.0
        )

        # idle_cash_pressure: fraction of max_promotions not covered by affordable candidates
        idle_pressure = max(
            0.0,
            1.0 - len(affordable_shadow) / MAX_PROMOTIONS_PER_REBALANCE,
        ) if MAX_PROMOTIONS_PER_REBALANCE > 0 else 0.0

        # concentration relief contribution of promoted
        promoted_scores = [s for s in scores if s.symbol in promoted]
        conc_relief = (
            sum(s.concentration_relief_score for s in promoted_scores) / len(promoted_scores)
            if promoted_scores else 0.0
        )

        sector_dist: Dict[str, int] = {}
        for sec in live_after.values():
            sector_dist[sec] = sector_dist.get(sec, 0) + 1

        return GovernanceReport(
            run_id=run_id,
            promotion_batch_id=batch_id,
            mode=mode,
            promoted=sorted(promoted),
            rejected=sorted(rejected),
            on_cooldown=sorted(on_cooldown),
            universe_size_before=len(live_before),
            universe_size_after=len(live_after),
            deployable_breadth_score=round(deployable_breadth, 4),
            affordable_opportunity_coverage=round(affordable_coverage, 4),
            high_price_exclusion_count=len(high_price_excl),
            sector_distribution=dict(sorted(sector_dist.items())),
            idle_cash_pressure=round(idle_pressure, 4),
            concentration_relief_contribution=round(conc_relief, 4),
            created_at=_now_utc(),
        )

    def _update_universe_file(
        self,
        universe_file: Path,
        updated_live: Dict[str, str],
        promoted: List[str],
        manifest_id: str,
    ) -> bool:
        """Atomically update universe JSON file. Returns True on success."""
        try:
            existing: dict = {}
            if universe_file.exists():
                existing = json.loads(universe_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("[GOVERNANCE] could not read universe file: %s", exc)
            existing = {}

        # Bump governance version suffix while preserving base version
        base_version = existing.get("version", "rsr42_v1")
        # Strip old _gov suffix if present
        if "_gov" in base_version:
            base_version = base_version.split("_gov")[0]
        new_version = f"{base_version}_gov{len(promoted)}"

        updated: dict = {
            **existing,
            "version": new_version,
            "symbols": {sym: sec for sym, sec in sorted(updated_live.items())},
            "n_stocks": len(updated_live),
            "last_governance_update": _now_utc(),
            "last_governance_manifest_id": manifest_id[:16],
            "last_promoted_symbols": sorted(promoted),
        }
        _write_atomic(universe_file, json.dumps(updated, ensure_ascii=False, indent=2))
        logger.info(
            "[GOVERNANCE] universe file updated: %s (+%d symbols, total=%d)",
            universe_file.name, len(promoted), len(updated_live),
        )
        return True

    def _write_daily_report(
        self,
        report: GovernanceReport,
        scores: List[DeployabilityScore],
    ) -> None:
        self._report_dir.mkdir(parents=True, exist_ok=True)
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        report_path = self._report_dir / f"governance_{date_str}_{report.run_id}.json"
        content = {
            "report": report.to_dict(),
            "candidate_scores": [s.to_dict() for s in scores],
        }
        report_path.write_text(
            json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_universe_governance(
    live_universe: Dict[str, str],
    shadow_universe: Dict[str, str],
    capital: int,
    run_id: str,
    universe_dir: Path,
    universe_file: Optional[Path] = None,
    price_lookup: Optional[Callable[[str], Optional[float]]] = None,
    rsr_scores: Optional[Dict[str, float]] = None,
    mode: str = DEFAULT_MODE,
    candidate_ev_scores: Optional[Dict[str, float]] = None,
    live_ev_floor: Optional[float] = None,
    rsr_source: str = "",
) -> GovernanceResult:
    """
    Convenience wrapper for DeployableUniverseGovernor.run().

    Args:
        live_universe:  current live universe {symbol: sector}
        shadow_universe: shadow candidates {symbol: sector}
        capital:        total portfolio capital (JPY)
        run_id:         execution run ID (e.g. "20260518_090000")
        universe_dir:   path to runtime/universe/ directory
        universe_file:  if given and promotions happen, update this JSON file atomically
        price_lookup:   callable(symbol) → float|None
        rsr_scores:     optional {symbol: rsr_score}
        mode:           OBSERVE_ONLY | AUTO_PROMOTE_SAFE | MANUAL_APPROVAL

    Returns:
        GovernanceResult
    """
    governor = DeployableUniverseGovernor(universe_dir=universe_dir)
    return governor.run(
        live_universe=live_universe,
        shadow_universe=shadow_universe,
        capital=capital,
        run_id=run_id,
        mode=mode,
        price_lookup=price_lookup,
        rsr_scores=rsr_scores,
        universe_file=universe_file,
        candidate_ev_scores=candidate_ev_scores,
        live_ev_floor=live_ev_floor,
        rsr_source=rsr_source,
    )
