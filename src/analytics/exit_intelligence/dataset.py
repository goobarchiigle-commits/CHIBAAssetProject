"""
src/analytics/exit_intelligence/dataset.py — Research Dataset Layer

ExitResearchDatasetBuilder: builds deterministic, replayable, schema-versioned
research dataset from all observation/analytics records.

Append-only JSONL. Hash-verifiable. Corruption-detectable.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .models import (
    ExitObservationRecord,
    ExitResearchRecord,
    ExitVelocityScore,
    PortfolioExitPressureSnapshot,
)
from .observation import ExitPathSnapshot

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))
SCHEMA_VERSION = 1


def _now_jst() -> str:
    return datetime.now(JST).isoformat()


def _sha256(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _atomic_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            if path.exists():
                fh.write(path.read_text(encoding="utf-8"))
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


class ExitResearchDatasetBuilder:
    """
    Build deterministic research dataset by joining:
      - ExitObservationRecord (base)
      - ExitVelocityScore at exit
      - ExitPathSnapshot summary (deterioration state)
      - PortfolioExitPressureSnapshot (context)
      - counterfactual returns (if available)

    Each record is append-only. Schema versioned. Hash-verifiable.
    """

    def build_record(
        self,
        obs: ExitObservationRecord,
        velocity_score: Optional[ExitVelocityScore] = None,
        path_snapshots: Optional[List[ExitPathSnapshot]] = None,
        portfolio_pressure: Optional[PortfolioExitPressureSnapshot] = None,
        post_exit_return_5d: Optional[float] = None,
        post_exit_return_10d: Optional[float] = None,
        exit_was_premature: Optional[bool] = None,
    ) -> ExitResearchRecord:
        # Summarize deterioration from path snapshots
        peak_decay_velocity = 0.0
        momentum_velocity = 0.0
        if path_snapshots:
            sorted_snaps = sorted(path_snapshots, key=lambda s: s.day_number)
            # Peak decay: distance from peak at last snapshot / days
            last = sorted_snaps[-1]
            days = max(last.day_number, 1)
            peak_decay_velocity = last.distance_from_peak_pct / days

            # Momentum velocity: slope of momentum
            moms = [s.momentum for s in sorted_snaps[-5:]]
            if len(moms) >= 2:
                n = len(moms)
                x_mean = (n - 1) / 2.0
                y_mean = sum(moms) / n
                num = sum((i - x_mean) * (moms[i] - y_mean) for i in range(n))
                den = sum((i - x_mean) ** 2 for i in range(n))
                momentum_velocity = num / den if den > 0 else 0.0

        # Velocity score fields
        evs_total = obs.exit_velocity_score
        evs_band = "unknown"
        structural_failure_score = 0.0
        convexity_retention_score = 0.0
        persistence_score = 0.0
        if velocity_score is not None:
            evs_total = velocity_score.total_score
            evs_band = velocity_score.action_band

        # Portfolio pressure fields
        concurrent_positions = 1
        portfolio_heat = 0.0
        if portfolio_pressure is not None:
            concurrent_positions = portfolio_pressure.concurrent_positions
            portfolio_heat = portfolio_pressure.portfolio_heat

        return ExitResearchRecord(
            research_id=ExitResearchRecord.make_research_id(obs.obs_id),
            obs_id=obs.obs_id,
            symbol=obs.symbol,
            entry_date=obs.entry_date,
            exit_date=obs.exit_date,
            holding_days=obs.holding_days,
            realized_return=obs.realized_return,
            exit_velocity_score=evs_total,
            action_band=evs_band,
            peak_decay_velocity=peak_decay_velocity,
            structural_failure_score=structural_failure_score,
            momentum_velocity=momentum_velocity,
            convexity_retention_score=convexity_retention_score,
            persistence_score=persistence_score,
            regime=obs.regime_at_exit,
            breadth_score=obs.breadth_score_at_exit,
            rs_rank=obs.rs_rank_at_exit,
            atr_pct=obs.atr_pct_at_exit,
            concurrent_positions=concurrent_positions,
            portfolio_heat=portfolio_heat,
            post_exit_return_5d=post_exit_return_5d,
            post_exit_return_10d=post_exit_return_10d,
            exit_was_premature=exit_was_premature,
            schema_version=SCHEMA_VERSION,
        )

    def compute_integrity_hash(self, records: List[ExitResearchRecord]) -> str:
        """SHA-256 of sorted, canonical JSON of all records. For corruption detection."""
        lines = [
            json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False)
            for r in sorted(records, key=lambda r: (r.exit_date, r.obs_id))
        ]
        combined = "\n".join(lines)
        return _sha256(combined)


class ExitResearchDatasetStore:
    """Append-only JSONL store for ExitResearchRecord with integrity hash."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._hash_path = path.with_suffix(".hash")

    def append(self, record: ExitResearchRecord) -> None:
        _atomic_append(self._path, record.to_dict())
        # Recompute hash
        all_records = self.load_all()
        builder = ExitResearchDatasetBuilder()
        new_hash = builder.compute_integrity_hash(all_records)
        _atomic_write(self._hash_path, new_hash + "\n")
        logger.debug("[DATASET] appended research_id=%s symbol=%s", record.research_id, record.symbol)

    def load_all(self) -> List[ExitResearchRecord]:
        if not self._path.exists():
            return []
        records = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(ExitResearchRecord(**{
                    k: d.get(k) for k in ExitResearchRecord.__dataclass_fields__
                }))
            except Exception as exc:
                logger.warning("[DATASET] parse error: %s", exc)
        return records

    def verify_integrity(self) -> bool:
        """Return True if stored hash matches recomputed hash."""
        if not self._hash_path.exists():
            return True  # no hash yet — pass
        stored_hash = self._hash_path.read_text(encoding="utf-8").strip()
        records = self.load_all()
        if not records:
            return True
        builder = ExitResearchDatasetBuilder()
        computed = builder.compute_integrity_hash(records)
        return computed == stored_hash

    def snapshot_stats(self) -> Dict:
        records = self.load_all()
        if not records:
            return {"total_records": 0}
        returns = [r.realized_return for r in records]
        n = len(returns)
        mean_ret = sum(returns) / n
        premature_count = sum(1 for r in records if r.exit_was_premature is True)
        return {
            "total_records": n,
            "mean_realized_return": round(mean_ret, 6),
            "positive_return_rate": round(sum(1 for r in returns if r > 0) / n, 4),
            "premature_exit_rate": round(premature_count / n, 4),
            "integrity_ok": self.verify_integrity(),
        }
