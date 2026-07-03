"""PaperLiveComparator — measures divergence between simulated/paper and live execution."""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

MODE_SIMULATED = "SIMULATED"
MODE_PAPER = "PAPER"
MODE_LIVE = "LIVE"

ALL_MODES = (MODE_SIMULATED, MODE_PAPER, MODE_LIVE)


@dataclass(frozen=True)
class ComparisonRecord:
    record_id: str          # "cmp_" + SHA256[:12]
    session_id: str
    symbol: str
    mode_a: str             # reference mode
    mode_b: str             # comparison mode
    slippage_a_bps: float
    slippage_b_bps: float
    fill_ratio_a: float
    fill_ratio_b: float
    latency_a_ns: int
    latency_b_ns: int
    execution_divergence_score: float   # normalized [0, 1]
    realism_degradation_score: float    # how much live degrades vs. research
    recorded_at: str

    def to_dict(self) -> dict:
        return {
            "record_id": self.record_id,
            "session_id": self.session_id,
            "symbol": self.symbol,
            "mode_a": self.mode_a,
            "mode_b": self.mode_b,
            "slippage_a_bps": self.slippage_a_bps,
            "slippage_b_bps": self.slippage_b_bps,
            "fill_ratio_a": self.fill_ratio_a,
            "fill_ratio_b": self.fill_ratio_b,
            "latency_a_ns": self.latency_a_ns,
            "latency_b_ns": self.latency_b_ns,
            "execution_divergence_score": self.execution_divergence_score,
            "realism_degradation_score": self.realism_degradation_score,
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ComparisonRecord":
        return cls(**d)


def _record_id(session_id: str, symbol: str, mode_a: str, mode_b: str, recorded_at: str) -> str:
    raw = json.dumps(
        {"session_id": session_id, "symbol": symbol, "mode_a": mode_a,
         "mode_b": mode_b, "recorded_at": recorded_at},
        sort_keys=True, separators=(",", ":")
    )
    return "cmp_" + hashlib.sha256(raw.encode()).hexdigest()[:12]


def _divergence_score(
    slippage_a: float, slippage_b: float,
    fill_ratio_a: float, fill_ratio_b: float,
    latency_a_ns: int, latency_b_ns: int,
) -> float:
    slip_diff = abs(slippage_b - slippage_a) / max(abs(slippage_a), 1.0)
    fill_diff = abs(fill_ratio_b - fill_ratio_a)
    lat_diff = abs(latency_b_ns - latency_a_ns) / max(latency_a_ns, 1)
    raw = (slip_diff + fill_diff + lat_diff) / 3.0
    return min(1.0, raw)


def _realism_degradation(
    slippage_a: float, slippage_b: float,
    fill_ratio_a: float, fill_ratio_b: float,
) -> float:
    slip_deg = max(0.0, slippage_b - slippage_a) / max(abs(slippage_a), 1.0)
    fill_deg = max(0.0, fill_ratio_a - fill_ratio_b)
    return min(1.0, (slip_deg + fill_deg) / 2.0)


class PaperLiveComparator:
    """Compares execution quality across modes; measures realism degradation."""

    def __init__(self, divergence_alert_threshold: float = 0.30):
        if not (0.0 < divergence_alert_threshold <= 1.0):
            raise ValueError("divergence_alert_threshold must be in (0, 1]")
        self._alert_threshold = divergence_alert_threshold
        self._records: List[ComparisonRecord] = []

    def compare(
        self,
        session_id: str,
        symbol: str,
        mode_a: str,
        mode_b: str,
        slippage_a_bps: float,
        slippage_b_bps: float,
        fill_ratio_a: float,
        fill_ratio_b: float,
        latency_a_ns: int,
        latency_b_ns: int,
        recorded_at: str,
    ) -> ComparisonRecord:
        for m in (mode_a, mode_b):
            if m not in ALL_MODES:
                raise ValueError(f"Unknown mode: {m!r}")
        record_id = _record_id(session_id, symbol, mode_a, mode_b, recorded_at)
        div_score = _divergence_score(slippage_a_bps, slippage_b_bps, fill_ratio_a, fill_ratio_b,
                                      latency_a_ns, latency_b_ns)
        real_score = _realism_degradation(slippage_a_bps, slippage_b_bps, fill_ratio_a, fill_ratio_b)
        rec = ComparisonRecord(
            record_id=record_id,
            session_id=session_id,
            symbol=symbol,
            mode_a=mode_a,
            mode_b=mode_b,
            slippage_a_bps=slippage_a_bps,
            slippage_b_bps=slippage_b_bps,
            fill_ratio_a=fill_ratio_a,
            fill_ratio_b=fill_ratio_b,
            latency_a_ns=latency_a_ns,
            latency_b_ns=latency_b_ns,
            execution_divergence_score=div_score,
            realism_degradation_score=real_score,
            recorded_at=recorded_at,
        )
        self._records.append(rec)
        return rec

    def mean_divergence_score(self) -> float:
        if not self._records:
            return 0.0
        return sum(r.execution_divergence_score for r in self._records) / len(self._records)

    def mean_realism_degradation(self) -> float:
        if not self._records:
            return 0.0
        return sum(r.realism_degradation_score for r in self._records) / len(self._records)

    def has_significant_divergence(self) -> bool:
        return self.mean_divergence_score() >= self._alert_threshold

    def all_records(self) -> List[ComparisonRecord]:
        return sorted(self._records, key=lambda r: r.recorded_at)

    def record_count(self) -> int:
        return len(self._records)

    def to_summary(self) -> dict:
        return {
            "record_count": len(self._records),
            "mean_divergence_score": self.mean_divergence_score(),
            "mean_realism_degradation": self.mean_realism_degradation(),
            "has_significant_divergence": self.has_significant_divergence(),
            "alert_threshold": self._alert_threshold,
        }
