"""Deterministic walk-forward fold generator.

Supports rolling, anchored, and expanding windows with purge and embargo.
All fold lists are deterministic: same inputs → same output, always.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import List

from src.research.contracts.fold import DateRange, FoldContract


class WalkForwardEngine:
    """Generate deterministic fold sequences for walk-forward validation."""

    def __init__(self, universe_start: date, universe_end: date, seed: int) -> None:
        if universe_end <= universe_start:
            raise ValueError("universe_end must be after universe_start")
        self.universe_start = universe_start
        self.universe_end = universe_end
        self.seed = seed

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def rolling_folds(
        self,
        train_days: int,
        val_days: int,
        test_days: int,
        embargo_days: int,
        step_days: int,
    ) -> List[FoldContract]:
        """Fixed-size train window slides forward by step_days."""
        _validate_positive(train_days=train_days, val_days=val_days,
                           test_days=test_days, embargo_days=embargo_days,
                           step_days=step_days)
        folds: List[FoldContract] = []
        cursor = self.universe_start
        fold_id = 0
        while True:
            train_end = cursor + timedelta(days=train_days)
            embargo_end = train_end + timedelta(days=embargo_days)
            val_end = embargo_end + timedelta(days=val_days)
            test_end = val_end + timedelta(days=test_days)
            if test_end > self.universe_end:
                break
            fold = FoldContract(
                fold_id=fold_id,
                train_range=DateRange(cursor, train_end),
                validation_range=DateRange(embargo_end, val_end),
                test_range=DateRange(val_end, test_end),
                embargo_range=DateRange(train_end, embargo_end),
                regime_distribution={},
            )
            fold.validate()
            folds.append(fold)
            cursor += timedelta(days=step_days)
            fold_id += 1
        return folds

    def anchored_folds(
        self,
        initial_train_days: int,
        val_days: int,
        test_days: int,
        embargo_days: int,
        step_days: int,
    ) -> List[FoldContract]:
        """Train always starts at universe_start; extends by step_days each fold."""
        _validate_positive(initial_train_days=initial_train_days, val_days=val_days,
                           test_days=test_days, embargo_days=embargo_days,
                           step_days=step_days)
        folds: List[FoldContract] = []
        extension = 0
        fold_id = 0
        while True:
            train_end = self.universe_start + timedelta(days=initial_train_days + extension)
            embargo_end = train_end + timedelta(days=embargo_days)
            val_end = embargo_end + timedelta(days=val_days)
            test_end = val_end + timedelta(days=test_days)
            if test_end > self.universe_end:
                break
            fold = FoldContract(
                fold_id=fold_id,
                train_range=DateRange(self.universe_start, train_end),
                validation_range=DateRange(embargo_end, val_end),
                test_range=DateRange(val_end, test_end),
                embargo_range=DateRange(train_end, embargo_end),
                regime_distribution={},
            )
            fold.validate()
            folds.append(fold)
            extension += step_days
            fold_id += 1
        return folds

    def expanding_folds(
        self,
        initial_train_days: int,
        test_days: int,
        embargo_days: int,
        step_days: int,
    ) -> List[FoldContract]:
        """Train expands from universe_start; no separate validation window."""
        _validate_positive(initial_train_days=initial_train_days, test_days=test_days,
                           embargo_days=embargo_days, step_days=step_days)
        folds: List[FoldContract] = []
        extension = 0
        fold_id = 0
        while True:
            train_end = self.universe_start + timedelta(days=initial_train_days + extension)
            embargo_end = train_end + timedelta(days=embargo_days)
            test_end = embargo_end + timedelta(days=test_days)
            if test_end > self.universe_end:
                break
            fold = FoldContract(
                fold_id=fold_id,
                train_range=DateRange(self.universe_start, train_end),
                validation_range=DateRange(embargo_end, embargo_end),  # empty
                test_range=DateRange(embargo_end, test_end),
                embargo_range=DateRange(train_end, embargo_end),
                regime_distribution={},
            )
            fold.validate()
            folds.append(fold)
            extension += step_days
            fold_id += 1
        return folds


def _validate_positive(**kwargs: int) -> None:
    for name, val in kwargs.items():
        if val < 0:
            raise ValueError(f"{name} must be >= 0, got {val}")
