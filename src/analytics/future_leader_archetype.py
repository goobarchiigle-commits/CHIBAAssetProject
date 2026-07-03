"""
src/analytics/future_leader_archetype.py — Future Leader Failure Archetype Layer

Purpose:
  Extend failure clustering from "what happened" (primary_failure_reason) to
  "how it collapsed" (failure_archetype). Observes institutional drift collapse
  structures. Research telemetry only — never connected to execution path.

INVARIANTS (変更禁止):
  - observation_only=True: no execution path connection
  - rule-based only: no ML, no complex scoring
  - single transition_pair per record: primary_failure_reason→failure_archetype
  - append-only JSONL: (symbol, observation_date) once written, never rewritten
  - CLEAN-only aggregation: data_quality_weight >= 1.0
  - PARTIAL_FAILURE: persisted to JSONL, excluded from aggregation
  - lookahead-safe: operates on already-materialized failure records only

禁止:
  - allocator / broker / order / signal_bridge / deployable_universe imports
  - ML clustering / sequence modeling / Markov chains
  - execution path connection / predictive_entry_enabled mutation
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))

_QUALITY_CLEAN_MIN: float = 1.0
_MIN_SAMPLE:        int   = 3


# ─────────────────────────────────────────────────────────────────────────────
# Failure archetype enum — minimal set
# ─────────────────────────────────────────────────────────────────────────────

class FailureArchetype(str, Enum):
    instant_rejection  = "instant_rejection"   # weak entry, collapsed ≤ 3d
    slow_bleed         = "slow_bleed"          # gradual price erosion after breakout
    volatility_trap    = "volatility_trap"     # high MFE + deep MAE: sharp spike then reversal
    liquidity_fade     = "liquidity_fade"      # volume dries up ≥ 5d, price stagnates
    institutional_exit = "institutional_exit"  # RSR reversal: institutional selling detected
    regime_mismatch    = "regime_mismatch"     # macro regime drove the failure
    unresolved         = "unresolved"          # no rule matched or no failure detected


# ─────────────────────────────────────────────────────────────────────────────
# Classification — pure rule-based, no ML
# ─────────────────────────────────────────────────────────────────────────────

def classify_failure_archetype(
    primary_failure_reason: Optional[str],
    survival_days:          Optional[int],
    mfe_10d:                Optional[float],
    mae_10d:                Optional[float],
    regime_tag:             str,
    rsr_delta:              Optional[float],
) -> FailureArchetype:
    """
    Map (primary_failure_reason, forward metrics) → FailureArchetype.

    Rules (applied in order, first match wins):
      market_selloff                                    → regime_mismatch
      rsr_reversal                                      → institutional_exit
      volume_faded   AND survival_days >= 5             → liquidity_fade
      gap_fill       AND mfe_10d > 5% AND mae_10d < -5% → volatility_trap
      weak_followthrough AND survival_days <= 3         → instant_rejection
      breakout_failed_3d                                → slow_bleed
      else                                              → unresolved

    Pure function: no I/O, no external state.
    observation_only invariant preserved.
    """
    r = primary_failure_reason

    if r == "market_selloff":
        return FailureArchetype.regime_mismatch

    if r == "rsr_reversal":
        return FailureArchetype.institutional_exit

    if r == "volume_faded":
        if survival_days is not None and survival_days >= 5:
            return FailureArchetype.liquidity_fade

    if r == "gap_fill":
        if (mfe_10d is not None and mfe_10d > 0.05
                and mae_10d is not None and mae_10d < -0.05):
            return FailureArchetype.volatility_trap

    if r == "weak_followthrough":
        if survival_days is not None and survival_days <= 3:
            return FailureArchetype.instant_rejection

    if r == "breakout_failed_3d":
        return FailureArchetype.slow_bleed

    return FailureArchetype.unresolved


def build_transition_pair(
    primary_failure_reason: Optional[str],
    archetype:              FailureArchetype,
) -> Optional[str]:
    """
    Single transition pair for telemetry.
    Format: "primary_failure_reason→archetype_value"
    Returns None when primary_failure_reason is absent.
    """
    if not primary_failure_reason:
        return None
    return f"{primary_failure_reason}→{archetype.value}"


# ─────────────────────────────────────────────────────────────────────────────
# Record dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ArchetypeRecord:
    """
    Archetype classification for one Future Leader failure observation.
    observation_only=True: invariant, never connected to execution path.
    """
    symbol:                  str
    observation_date:        str
    primary_failure_reason:  Optional[str]
    failure_archetype:       str
    transition_pair:         Optional[str]
    survival_days:           Optional[int]
    mfe_10d:                 Optional[float]
    mae_10d:                 Optional[float]
    regime_tag:              str
    integrity_state:         str
    data_quality_weight:     float
    observation_only:        bool = True   # invariant: always True

    def to_dict(self) -> dict:
        return {
            "symbol":                 self.symbol,
            "observation_date":       self.observation_date,
            "primary_failure_reason": self.primary_failure_reason,
            "failure_archetype":      self.failure_archetype,
            "transition_pair":        self.transition_pair,
            "survival_days":          self.survival_days,
            "mfe_10d":                self.mfe_10d,
            "mae_10d":                self.mae_10d,
            "regime_tag":             self.regime_tag,
            "integrity_state":        self.integrity_state,
            "data_quality_weight":    self.data_quality_weight,
            "observation_only":       self.observation_only,
        }


@dataclass
class ArchetypeSummary:
    """Aggregated metrics per archetype. CLEAN-only."""
    archetype:       str
    count:           int
    median_survival: Optional[float]
    median_mfe:      Optional[float]
    median_mae:      Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_jsonl(log_dir: Path, pattern: str) -> List[dict]:
    records: List[dict] = []
    if not log_dir.exists():
        return records
    for p in sorted(log_dir.glob(pattern)):
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass
        except Exception:
            pass
    return records


def _load_failure_records(failure_log_dir: Path) -> List[dict]:
    return _load_jsonl(failure_log_dir, "future_leader_failure_*.jsonl")


def _load_archetype_keys(archetype_log_dir: Path) -> Set[Tuple[str, str]]:
    """Load already-classified (symbol, observation_date) pairs."""
    keys: Set[Tuple[str, str]] = set()
    for rec in _load_jsonl(archetype_log_dir, "future_leader_archetype_*.jsonl"):
        sym = rec.get("symbol", "")
        dt  = rec.get("observation_date", "")
        if sym and dt:
            keys.add((sym, dt))
    return keys


def _load_regime_lookup(regime_log_dir: Path) -> Dict[str, str]:
    """Load {observation_date: regime_tag}. Last occurrence per date wins."""
    result: Dict[str, str] = {}
    for rec in _load_jsonl(regime_log_dir, "future_leader_regime_*.jsonl"):
        dt  = rec.get("observation_date", "")
        tag = rec.get("regime_tag", "")
        if dt and tag:
            result[dt] = tag
    return result


def _write_archetype_log(
    records:          List[ArchetypeRecord],
    archetype_log_dir: Path,
    today:            Optional[date] = None,
) -> None:
    """Append new archetype records to today's JSONL. Fail-open."""
    if not records:
        return
    if today is None:
        today = datetime.now(JST).date()

    today_str = today.strftime("%Y%m%d")
    log_path  = archetype_log_dir / f"future_leader_archetype_{today_str}.jsonl"

    try:
        archetype_log_dir.mkdir(parents=True, exist_ok=True)
        ts    = datetime.now(JST).isoformat()
        lines = []
        for r in records:
            d = {"materialization_ts": ts, **r.to_dict()}
            lines.append(json.dumps(d, ensure_ascii=False, default=str))
        with log_path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        logger.info("[FL_ARCH] wrote %d records → %s", len(records), log_path)
    except Exception as e:
        logger.warning("[FL_ARCH] log write failed (fail-open): %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Core materialization pipeline
# ─────────────────────────────────────────────────────────────────────────────

def materialize_archetypes(
    failure_log_dir:   Path,
    regime_log_dir:    Path,
    archetype_log_dir: Path,
    today:             Optional[date] = None,
) -> int:
    """
    Classify failure archetypes for unprocessed failure records.

    observation_only=True: research telemetry only.
    append-only: (symbol, observation_date) never reprocessed.
    PARTIAL_FAILURE records persisted; excluded from aggregation.

    Returns count of newly classified records.
    """
    if today is None:
        today = datetime.now(JST).date()

    all_failure = _load_failure_records(failure_log_dir)
    if not all_failure:
        logger.info("[FL_ARCH] no failure records — nothing to classify")
        return 0

    # Dedup failure records by (symbol, observation_date) — first occurrence wins
    unique_fail: Dict[Tuple[str, str], dict] = {}
    for rec in all_failure:
        sym = rec.get("symbol", "")
        dt  = rec.get("observation_date", "")
        if sym and dt:
            key = (sym, dt)
            if key not in unique_fail:
                unique_fail[key] = rec

    existing_keys = _load_archetype_keys(archetype_log_dir)
    pending = {k: v for k, v in unique_fail.items() if k not in existing_keys}

    if not pending:
        logger.debug("[FL_ARCH] %d observations already classified", len(existing_keys))
        return 0

    logger.info("[FL_ARCH] %d pending observations to classify", len(pending))

    regime_lookup = _load_regime_lookup(regime_log_dir)
    new_records: List[ArchetypeRecord] = []

    for (sym, obs_date_str), fail_rec in sorted(pending.items()):
        dq_weight = float(fail_rec.get("data_quality_weight", 1.0))

        primary_reason  = fail_rec.get("primary_failure_reason") or fail_rec.get("failure_reason")
        failure_day_off = fail_rec.get("failure_day_offset")
        survival_days   = int(failure_day_off) if failure_day_off is not None else None

        mfe_10d_raw = fail_rec.get("mfe_10d")
        mae_10d_raw = fail_rec.get("mae_10d")
        mfe_10d = float(mfe_10d_raw) if mfe_10d_raw is not None else None
        mae_10d = float(mae_10d_raw) if mae_10d_raw is not None else None

        regime_tag    = regime_lookup.get(obs_date_str, "unclassified")
        integrity_state = str(fail_rec.get("integrity_state", "CLEAN"))

        archetype = classify_failure_archetype(
            primary_failure_reason=primary_reason,
            survival_days=survival_days,
            mfe_10d=mfe_10d,
            mae_10d=mae_10d,
            regime_tag=regime_tag,
            rsr_delta=None,
        )

        transition_pair = build_transition_pair(primary_reason, archetype)

        new_records.append(ArchetypeRecord(
            symbol=sym,
            observation_date=obs_date_str,
            primary_failure_reason=primary_reason,
            failure_archetype=archetype.value,
            transition_pair=transition_pair,
            survival_days=survival_days,
            mfe_10d=mfe_10d,
            mae_10d=mae_10d,
            regime_tag=regime_tag,
            integrity_state=integrity_state,
            data_quality_weight=dq_weight,
            observation_only=True,
        ))

    if new_records:
        _write_archetype_log(new_records, archetype_log_dir, today=today)

    logger.info("[FL_ARCH] classified=%d", len(new_records))
    return len(new_records)


# ─────────────────────────────────────────────────────────────────────────────
# Analytics — pure functions
# ─────────────────────────────────────────────────────────────────────────────

def _filter_clean(records: List[dict]) -> List[dict]:
    """CLEAN-only: data_quality_weight >= 1.0. PARTIAL_FAILURE excluded."""
    return [r for r in records if float(r.get("data_quality_weight", 0.0)) >= _QUALITY_CLEAN_MIN]


def _opt_median(vals: List[float]) -> Optional[float]:
    if len(vals) < _MIN_SAMPLE:
        return None
    return round(float(np.median(vals)), 6)


def compute_archetype_summary(archetype_log_dir: Path) -> List[ArchetypeSummary]:
    """
    Compute per-archetype analytics from CLEAN archetype records.

    CLEAN filter: data_quality_weight >= 1.0.
    PARTIAL_FAILURE excluded from aggregation.

    Returns one ArchetypeSummary per FailureArchetype value (count may be 0).
    Pure except for loading JSONL.
    """
    all_records  = _load_jsonl(archetype_log_dir, "future_leader_archetype_*.jsonl")
    clean        = _filter_clean(all_records)

    groups: Dict[str, List[dict]] = {a.value: [] for a in FailureArchetype}
    for r in clean:
        arch = r.get("failure_archetype", FailureArchetype.unresolved.value)
        if arch in groups:
            groups[arch].append(r)
        else:
            groups.setdefault(arch, []).append(r)

    summaries: List[ArchetypeSummary] = []
    for archetype in FailureArchetype:
        recs = groups.get(archetype.value, [])
        survival = [float(r["survival_days"]) for r in recs if r.get("survival_days") is not None]
        mfe_vals = [float(r["mfe_10d"])  for r in recs if r.get("mfe_10d")  is not None]
        mae_vals = [float(r["mae_10d"])  for r in recs if r.get("mae_10d")  is not None]

        summaries.append(ArchetypeSummary(
            archetype=archetype.value,
            count=len(recs),
            median_survival=_opt_median(survival),
            median_mfe=_opt_median(mfe_vals),
            median_mae=_opt_median(mae_vals),
        ))

    return summaries


# ─────────────────────────────────────────────────────────────────────────────
# Report formatter — pure function
# ─────────────────────────────────────────────────────────────────────────────

def _pct(v: Optional[float]) -> str:
    return f"{v * 100:+.1f}%" if v is not None else "N/A"


def format_archetype_section(summaries: List[ArchetypeSummary]) -> str:
    """
    Render FAILURE ARCHETYPES section. Pure function: no I/O.

    Example output:
      === FAILURE ARCHETYPES ===
      instant_rejection:
        count: 22
        median survival: 2d
        median MAE: -4.1%
    """
    lines = ["=== FAILURE ARCHETYPES ==="]

    any_data = any(s.count > 0 for s in summaries)
    if not any_data:
        lines.append("(no archetype records)")
        return "\n".join(lines)

    for s in summaries:
        if s.count == 0:
            continue
        lines.append("")
        lines.append(f"{s.archetype}:")
        lines.append(f"  count: {s.count}")
        if s.median_survival is not None:
            lines.append(f"  median survival: {s.median_survival:.1f}d")
        if s.median_mfe is not None:
            lines.append(f"  median MFE: {_pct(s.median_mfe)}")
        if s.median_mae is not None:
            lines.append(f"  median MAE: {_pct(s.median_mae)}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# run_live_signal.py integration hook
# ─────────────────────────────────────────────────────────────────────────────

def run_future_leader_archetype_hook(
    failure_log_dir:   Path,
    regime_log_dir:    Path,
    archetype_log_dir: Path,
    reports_dir:       Path,
    today:             Optional[date] = None,
) -> None:
    """
    Integration hook for run_live_signal.py. Fail-open.

    Positioned after failure clustering hook.
    1. Materializes archetype classifications for unprocessed failure records.
    2. Computes archetype summary (CLEAN-only).
    3. Appends FAILURE ARCHETYPES section to today's future_leader_report.

    observation_only=True: no execution path connection.
    Failure: warn and return — execution path untouched.
    """
    if today is None:
        today = datetime.now(JST).date()

    today_str   = today.strftime("%Y%m%d")
    report_path = reports_dir / f"future_leader_report_{today_str}.md"

    try:
        count = materialize_archetypes(
            failure_log_dir=failure_log_dir,
            regime_log_dir=regime_log_dir,
            archetype_log_dir=archetype_log_dir,
            today=today,
        )
        logger.info("[FL_ARCH] hook: classified %d new archetype records", count)

        summaries = compute_archetype_summary(archetype_log_dir)
        section   = format_archetype_section(summaries)

        reports_dir.mkdir(parents=True, exist_ok=True)
        mode = "a" if report_path.exists() else "w"
        with report_path.open(mode, encoding="utf-8") as f:
            f.write("\n\n" + section + "\n")

        logger.info("[FL_ARCH] section written → %s", report_path)

    except Exception as e:
        logger.warning("[FL_ARCH] hook failed (fail-open): %s", e)
