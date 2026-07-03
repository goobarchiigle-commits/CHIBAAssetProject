"""FoldContract — immutable description of a walk-forward data split."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date  # exclusive

    def days(self) -> int:
        return (self.end - self.start).days

    def to_dict(self) -> dict:
        return {"start": self.start.isoformat(), "end": self.end.isoformat()}

    @classmethod
    def from_dict(cls, d: dict) -> DateRange:
        return cls(
            start=date.fromisoformat(d["start"]),
            end=date.fromisoformat(d["end"]),
        )

    def __contains__(self, d: date) -> bool:
        return self.start <= d < self.end


@dataclass(frozen=True)
class FoldContract:
    fold_id: int
    train_range: DateRange
    validation_range: DateRange
    test_range: DateRange
    embargo_range: DateRange
    regime_distribution: Dict[str, float]  # regime_label -> fraction of test period

    def to_dict(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train_range": self.train_range.to_dict(),
            "validation_range": self.validation_range.to_dict(),
            "test_range": self.test_range.to_dict(),
            "embargo_range": self.embargo_range.to_dict(),
            "regime_distribution": dict(self.regime_distribution),
        }

    @classmethod
    def from_dict(cls, d: dict) -> FoldContract:
        return cls(
            fold_id=int(d["fold_id"]),
            train_range=DateRange.from_dict(d["train_range"]),
            validation_range=DateRange.from_dict(d["validation_range"]),
            test_range=DateRange.from_dict(d["test_range"]),
            embargo_range=DateRange.from_dict(d["embargo_range"]),
            regime_distribution=dict(d.get("regime_distribution", {})),
        )

    def with_regime_distribution(self, dist: Dict[str, float]) -> FoldContract:
        """Return new fold with regime distribution filled in."""
        return FoldContract(
            fold_id=self.fold_id,
            train_range=self.train_range,
            validation_range=self.validation_range,
            test_range=self.test_range,
            embargo_range=self.embargo_range,
            regime_distribution=dist,
        )

    def validate(self) -> None:
        """Raise if ranges overlap or violate ordering constraints."""
        if self.train_range.end > self.embargo_range.start:
            raise ValueError(f"fold {self.fold_id}: train overlaps embargo")
        if self.embargo_range.end > self.test_range.start:
            raise ValueError(f"fold {self.fold_id}: embargo overlaps test")
        if self.validation_range.days() > 0:
            if self.train_range.end > self.validation_range.start:
                raise ValueError(f"fold {self.fold_id}: train overlaps validation")
