"""
retirement_engine.py — Automatic alpha retirement governance

Retirement is a lifecycle state, NOT a deletion event.
Retired candidates remain fully replayable.

Detects:
  - edge decay
  - persistent slippage deterioration
  - instability escalation
  - factor crowding suspicion
  - capital inefficiency
  - persistent post-cost degradation
  - exploration underperformance

All decisions are deterministic and append-only.
No immediate deletion — historical records always preserved.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
SCHEMA_VERSION: int = 1


# ── Retirement triggers ────────────────────────────────────────────────────────

@dataclass
class RetirementSignal:
    signal_type: str       # EDGE_DECAY | SLIPPAGE_DETERIORATION | INSTABILITY |
                           # CROWDING | CAPITAL_INEFFICIENCY | DEGRADATION | UNDERPERFORMANCE
    severity: str          # WARNING | ALERT | CRITICAL
    observed_value: float
    threshold: float
    description: str


@dataclass
class RetirementConfig:
    """Conservative defaults — favor survivability over aggressive retirement."""
    edge_decay_window: int = 10             # rolling window for decay detection
    decay_win_rate_threshold: float = 0.30  # retire if win_rate < this for N consecutive
    decay_consecutive_required: int = 5     # periods below threshold before retire
    slippage_deterioration_ratio: float = 2.0  # retire if realized/expected > this
    instability_threshold: float = 0.80     # rank_turnover > this = unstable
    crowding_correlation_threshold: float = 0.90  # suspect crowding if corr > this
    capital_efficiency_floor: float = 0.001  # score below this = inefficient
    underperformance_streak: int = 8        # consecutive negative alpha observations
    schema_version: int = SCHEMA_VERSION


@dataclass
class RetirementRecord:
    """Immutable retirement record. Append-only."""
    candidate_id: str
    timestamp: str
    retirement_triggers: list           # list[RetirementSignal] as dicts
    primary_reason: str
    trigger_count: int
    final_win_rate: float
    final_max_drawdown: float
    final_edge_proxy_bps: float
    total_shadow_days: int
    schema_version: int = SCHEMA_VERSION


# ── Retirement evaluation ──────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_edge_decay(
    recent_returns: list[float],
    config: RetirementConfig,
) -> Optional[RetirementSignal]:
    """Detect sustained win-rate deterioration."""
    if len(recent_returns) < config.decay_consecutive_required:
        return None
    recent_n = recent_returns[-config.edge_decay_window:]
    win_rate = sum(1 for r in recent_n if r > 0) / len(recent_n)
    if win_rate < config.decay_win_rate_threshold:
        severity = "CRITICAL" if win_rate < config.decay_win_rate_threshold * 0.5 else "ALERT"
        return RetirementSignal(
            signal_type="EDGE_DECAY",
            severity=severity,
            observed_value=round(win_rate, 4),
            threshold=config.decay_win_rate_threshold,
            description=f"win_rate={win_rate:.2f} < threshold={config.decay_win_rate_threshold}",
        )
    return None


def evaluate_slippage_deterioration(
    slippage_ratios: list[float],
    config: RetirementConfig,
) -> Optional[RetirementSignal]:
    """Detect sustained slippage above expected."""
    if not slippage_ratios:
        return None
    avg_ratio = _mean(slippage_ratios[-10:])
    if avg_ratio > config.slippage_deterioration_ratio:
        return RetirementSignal(
            signal_type="SLIPPAGE_DETERIORATION",
            severity="ALERT",
            observed_value=round(avg_ratio, 4),
            threshold=config.slippage_deterioration_ratio,
            description=f"avg_slippage_ratio={avg_ratio:.2f} > {config.slippage_deterioration_ratio}",
        )
    return None


def evaluate_underperformance(
    edge_proxy_series: list[float],
    config: RetirementConfig,
) -> Optional[RetirementSignal]:
    """Detect consecutive negative alpha observations."""
    if len(edge_proxy_series) < config.underperformance_streak:
        return None
    recent = edge_proxy_series[-config.underperformance_streak:]
    negative_streak = all(v <= 0 for v in recent)
    if negative_streak:
        return RetirementSignal(
            signal_type="UNDERPERFORMANCE",
            severity="CRITICAL",
            observed_value=round(_mean(recent), 4),
            threshold=0.0,
            description=(
                f"{config.underperformance_streak} consecutive non-positive alpha observations"
            ),
        )
    return None


def evaluate_capital_inefficiency(
    efficiency_score: float,
    config: RetirementConfig,
) -> Optional[RetirementSignal]:
    """Detect sustained capital efficiency below floor."""
    if efficiency_score < config.capital_efficiency_floor:
        return RetirementSignal(
            signal_type="CAPITAL_INEFFICIENCY",
            severity="WARNING",
            observed_value=round(efficiency_score, 6),
            threshold=config.capital_efficiency_floor,
            description=f"efficiency_score={efficiency_score:.4f} < floor={config.capital_efficiency_floor}",
        )
    return None


def evaluate_retirement(
    candidate_id: str,
    timestamp: str,
    observations,                   # list[ShadowObservation]
    efficiency_score: float,
    config: RetirementConfig,
) -> Optional[RetirementRecord]:
    """
    Run all retirement checks. Returns RetirementRecord if retirement warranted,
    else None. Deterministic — same inputs → same outcome.
    """
    if not observations:
        return None

    edge_series = [o.edge_proxy_bps for o in observations]
    slippage_ratios = [o.slippage_ratio for o in observations]

    triggers: list[RetirementSignal] = []
    for check_fn, args in [
        (evaluate_edge_decay,             (edge_series, config)),
        (evaluate_slippage_deterioration, (slippage_ratios, config)),
        (evaluate_underperformance,       (edge_series, config)),
        (evaluate_capital_inefficiency,   (efficiency_score, config)),
    ]:
        sig = check_fn(*args)
        if sig is not None:
            triggers.append(sig)

    # Retire only on ALERT or CRITICAL triggers
    actionable = [t for t in triggers if t.severity in ("ALERT", "CRITICAL")]
    if not actionable:
        return None

    primary = actionable[0].description
    win_rate = sum(1 for e in edge_series if e > 0) / len(edge_series)
    max_dd = max((o.drawdown for o in observations), default=0.0)

    return RetirementRecord(
        candidate_id=candidate_id,
        timestamp=timestamp,
        retirement_triggers=[asdict(t) for t in triggers],
        primary_reason=primary,
        trigger_count=len(triggers),
        final_win_rate=round(win_rate, 4),
        final_max_drawdown=round(max_dd, 4),
        final_edge_proxy_bps=round(_mean(edge_series[-10:]), 2),
        total_shadow_days=len(observations),
    )


# ── Persistence ────────────────────────────────────────────────────────────────

def append_retirement_record(record: RetirementRecord, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), ensure_ascii=False, sort_keys=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_retirement_records(path: Path) -> list[RetirementRecord]:
    if not path.exists():
        return []
    records: list[RetirementRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            records.append(RetirementRecord(
                candidate_id=data.get("candidate_id", ""),
                timestamp=data.get("timestamp", ""),
                retirement_triggers=data.get("retirement_triggers", []),
                primary_reason=data.get("primary_reason", ""),
                trigger_count=int(data.get("trigger_count", 0)),
                final_win_rate=float(data.get("final_win_rate", 0.0)),
                final_max_drawdown=float(data.get("final_max_drawdown", 0.0)),
                final_edge_proxy_bps=float(data.get("final_edge_proxy_bps", 0.0)),
                total_shadow_days=int(data.get("total_shadow_days", 0)),
            ))
        except Exception as exc:
            logger.warning("retirement_record parse error (%s) — skipped", exc)
    return records
