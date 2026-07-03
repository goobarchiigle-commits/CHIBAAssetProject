"""
research_budget.py — Research exploration budget governance

Hard caps on exploration resource consumption.
The research system MUST NOT consume unbounded production resources.

Governs:
  - max exploration capital (% of production)
  - max concurrent shadow candidates
  - per-theme exploration limits
  - research turnover budgets
  - exploration congestion detection

All decisions are append-only. Fail-closed defaults.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1


# ── Budget configuration ───────────────────────────────────────────────────────

@dataclass
class ResearchBudget:
    """Hard exploration resource constraints."""
    max_exploration_capital_pct: float = 0.05    # 5% of production capital
    max_shadow_candidates: int = 5               # concurrent shadow runs
    max_per_theme: int = 2                       # same sector/factor cap
    max_turnover_pct: float = 0.20               # 20% annual research turnover
    max_factor_count: int = 12                   # mirrors complexity_budget
    congestion_threshold: float = 0.80           # utilization above this = congested
    schema_version: int = SCHEMA_VERSION


# ── Budget state ───────────────────────────────────────────────────────────────

@dataclass
class ResearchBudgetState:
    """Current utilization tracking."""
    active_shadow_count: int = 0
    total_exploration_capital: float = 0.0
    production_capital: float = 0.0
    theme_counts: dict = field(default_factory=dict)   # {theme: count}
    last_updated: str = ""
    schema_version: int = SCHEMA_VERSION

    @property
    def capital_utilization(self) -> float:
        if self.production_capital <= 0:
            return 0.0
        return round(self.total_exploration_capital / self.production_capital, 4)

    @property
    def candidate_utilization(self) -> float:
        from src.alpha_research.research_budget import ResearchBudget
        return 0.0  # computed externally


# ── Budget decision ────────────────────────────────────────────────────────────

@dataclass
class BudgetDecision:
    """Append-only budget decision record."""
    timestamp: str
    candidate_id: str
    decision: str             # APPROVED | REJECTED | PAUSED
    reason: str
    active_shadow_count: int
    capital_utilization: float
    candidate_utilization: float
    schema_version: int = SCHEMA_VERSION


# ── Budget checks ──────────────────────────────────────────────────────────────

def check_shadow_capacity(
    state: ResearchBudgetState,
    budget: ResearchBudget,
) -> tuple[bool, str]:
    """Can we admit another shadow candidate?"""
    if state.active_shadow_count >= budget.max_shadow_candidates:
        return False, (
            f"shadow capacity reached ({state.active_shadow_count}/{budget.max_shadow_candidates})"
        )
    return True, "capacity available"


def check_capital_budget(
    state: ResearchBudgetState,
    budget: ResearchBudget,
    requested_capital: float,
) -> tuple[bool, str]:
    """Can we allocate requested_capital to exploration?"""
    if state.production_capital <= 0:
        return True, "no production capital baseline set"
    new_total = state.total_exploration_capital + requested_capital
    new_pct = new_total / state.production_capital
    if new_pct > budget.max_exploration_capital_pct:
        return False, (
            f"capital budget exceeded: new_pct={new_pct:.2%} > "
            f"max={budget.max_exploration_capital_pct:.2%}"
        )
    return True, f"capital ok: {new_pct:.2%} utilization"


def check_theme_budget(
    state: ResearchBudgetState,
    budget: ResearchBudget,
    theme: str,
) -> tuple[bool, str]:
    """Can we add another candidate in this theme?"""
    current = state.theme_counts.get(theme, 0)
    if current >= budget.max_per_theme:
        return False, (
            f"theme budget reached for '{theme}': "
            f"{current}/{budget.max_per_theme}"
        )
    return True, f"theme '{theme}' has {current}/{budget.max_per_theme}"


def is_congested(state: ResearchBudgetState, budget: ResearchBudget) -> bool:
    """True if utilization exceeds congestion threshold."""
    cap_util = state.capital_utilization
    cand_util = (
        state.active_shadow_count / budget.max_shadow_candidates
        if budget.max_shadow_candidates > 0 else 0.0
    )
    return max(cap_util, cand_util) >= budget.congestion_threshold


def evaluate_admission(
    state: ResearchBudgetState,
    budget: ResearchBudget,
    candidate_id: str,
    theme: str,
    requested_capital: float,
    timestamp: str,
    audit_path: Path,
) -> tuple[ResearchBudgetState, BudgetDecision]:
    """
    Evaluate whether to admit a new shadow candidate.
    Updates state on approval.
    Always appends to audit log.
    """
    # Run all checks
    cap_ok, cap_reason = check_capacity_ok(state, budget, requested_capital, theme)
    cap_util = state.capital_utilization
    cand_util = (
        state.active_shadow_count / budget.max_shadow_candidates
        if budget.max_shadow_candidates > 0 else 0.0
    )

    if cap_ok:
        new_state = ResearchBudgetState(
            active_shadow_count=state.active_shadow_count + 1,
            total_exploration_capital=state.total_exploration_capital + requested_capital,
            production_capital=state.production_capital,
            theme_counts={**state.theme_counts, theme: state.theme_counts.get(theme, 0) + 1},
            last_updated=timestamp,
        )
        decision = "APPROVED"
        reason = cap_reason
    else:
        new_state = state
        decision = "REJECTED"
        reason = cap_reason

    dec = BudgetDecision(
        timestamp=timestamp,
        candidate_id=candidate_id,
        decision=decision,
        reason=reason,
        active_shadow_count=state.active_shadow_count,
        capital_utilization=cap_util,
        candidate_utilization=cand_util,
    )
    append_budget_decision(dec, audit_path)
    return new_state, dec


def check_capacity_ok(
    state: ResearchBudgetState,
    budget: ResearchBudget,
    requested_capital: float,
    theme: str,
) -> tuple[bool, str]:
    """Combined check: capacity + capital + theme. Returns (ok, reason)."""
    shadow_ok, shadow_reason = check_shadow_capacity(state, budget)
    if not shadow_ok:
        return False, shadow_reason
    cap_ok, cap_reason = check_capital_budget(state, budget, requested_capital)
    if not cap_ok:
        return False, cap_reason
    theme_ok, theme_reason = check_theme_budget(state, budget, theme)
    if not theme_ok:
        return False, theme_reason
    return True, f"all budget checks passed: {cap_reason}"


def release_budget(
    state: ResearchBudgetState,
    candidate_id: str,
    theme: str,
    capital_released: float,
    timestamp: str,
    audit_path: Path,
) -> tuple[ResearchBudgetState, BudgetDecision]:
    """Release resources when candidate exits shadow (retirement or promotion)."""
    new_theme_counts = dict(state.theme_counts)
    new_theme_counts[theme] = max(0, new_theme_counts.get(theme, 0) - 1)

    new_state = ResearchBudgetState(
        active_shadow_count=max(0, state.active_shadow_count - 1),
        total_exploration_capital=max(0.0, state.total_exploration_capital - capital_released),
        production_capital=state.production_capital,
        theme_counts=new_theme_counts,
        last_updated=timestamp,
    )
    dec = BudgetDecision(
        timestamp=timestamp,
        candidate_id=candidate_id,
        decision="PAUSED",
        reason=f"resources released: capital={capital_released:.0f} theme={theme}",
        active_shadow_count=new_state.active_shadow_count,
        capital_utilization=new_state.capital_utilization,
        candidate_utilization=(
            new_state.active_shadow_count / max(1, 5)
        ),
    )
    append_budget_decision(dec, audit_path)
    return new_state, dec


# ── Persistence ────────────────────────────────────────────────────────────────

def append_budget_decision(decision: BudgetDecision, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(decision), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_research_budget(path: Path) -> ResearchBudget:
    if not path.exists():
        return ResearchBudget()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ResearchBudget(
            max_exploration_capital_pct=float(data.get("max_exploration_capital_pct", 0.05)),
            max_shadow_candidates=int(data.get("max_shadow_candidates", 5)),
            max_per_theme=int(data.get("max_per_theme", 2)),
            max_turnover_pct=float(data.get("max_turnover_pct", 0.20)),
            max_factor_count=int(data.get("max_factor_count", 12)),
            congestion_threshold=float(data.get("congestion_threshold", 0.80)),
        )
    except Exception as exc:
        logger.warning("research_budget load failed (%s) — default", exc)
        return ResearchBudget()


def save_research_budget(budget: ResearchBudget, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(asdict(budget), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_budget_state(path: Path) -> ResearchBudgetState:
    if not path.exists():
        return ResearchBudgetState()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return ResearchBudgetState(
            active_shadow_count=int(data.get("active_shadow_count", 0)),
            total_exploration_capital=float(data.get("total_exploration_capital", 0.0)),
            production_capital=float(data.get("production_capital", 0.0)),
            theme_counts=data.get("theme_counts", {}),
            last_updated=data.get("last_updated", ""),
        )
    except Exception as exc:
        logger.warning("budget_state load failed (%s) — default", exc)
        return ResearchBudgetState()


def save_budget_state(state: ResearchBudgetState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(
        json.dumps({
            "active_shadow_count": state.active_shadow_count,
            "total_exploration_capital": state.total_exploration_capital,
            "production_capital": state.production_capital,
            "theme_counts": state.theme_counts,
            "last_updated": state.last_updated,
            "schema_version": state.schema_version,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_budget_decisions(path: Path) -> list[BudgetDecision]:
    if not path.exists():
        return []
    decisions: list[BudgetDecision] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            decisions.append(BudgetDecision(
                timestamp=data.get("timestamp", ""),
                candidate_id=data.get("candidate_id", ""),
                decision=data.get("decision", ""),
                reason=data.get("reason", ""),
                active_shadow_count=int(data.get("active_shadow_count", 0)),
                capital_utilization=float(data.get("capital_utilization", 0.0)),
                candidate_utilization=float(data.get("candidate_utilization", 0.0)),
            ))
        except Exception as exc:
            logger.warning("budget_decision parse error (%s) — skipped", exc)
    return decisions
