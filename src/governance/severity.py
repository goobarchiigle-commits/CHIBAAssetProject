"""
src/governance/severity.py
CI finding severity classification.

Levels:
  CRITICAL — fail immediately, block all execution
  HIGH     — fail gate, require fix before merge
  MEDIUM   — warning in gate, tracked
  LOW      — informational

Critical triggers (any match → CRITICAL):
  - hash mismatch / integrity failure
  - determinism break
  - production/research contamination (state separation)
  - mutable artifact (overwrite attempt)
  - stale OOS beyond hard limit
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    def __lt__(self, other: "Severity") -> bool:
        order = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other: "Severity") -> bool:
        return self == other or self < other

    def __gt__(self, other: "Severity") -> bool:
        return not self <= other

    def __ge__(self, other: "Severity") -> bool:
        return self == other or self > other


@dataclass
class Finding:
    severity: Severity
    category: str
    message: str
    source: str = ""

    def to_dict(self) -> dict:
        return {
            "severity": self.severity.value,
            "category": self.category,
            "message": self.message,
            "source": self.source,
        }

    def is_critical(self) -> bool:
        return self.severity == Severity.CRITICAL

    def __str__(self) -> str:
        return f"[{self.severity.value}][{self.category}] {self.message}"


# ── Category → default severity ───────────────────────────────────────────────

_CATEGORY_SEVERITY: dict[str, Severity] = {
    # Always CRITICAL
    "OOS":       Severity.CRITICAL,
    "STATE":     Severity.CRITICAL,
    "REPRO":     Severity.CRITICAL,
    "ARTIFACT":  Severity.CRITICAL,
    # HIGH
    "CONSISTENCY": Severity.HIGH,
    "IMPORTS":     Severity.HIGH,
    "HYGIENE":     Severity.HIGH,
    "PROMOTE":     Severity.HIGH,
    # MEDIUM
    "PERF":      Severity.MEDIUM,
    "FLAKY":     Severity.MEDIUM,
    "QUARANTINE": Severity.MEDIUM,
    # LOW
    "LINEAGE":   Severity.LOW,
    "INFO":      Severity.LOW,
}

# Keywords in message that upgrade severity to CRITICAL
_CRITICAL_KEYWORDS = frozenset({
    "hash mismatch",
    "integrity failure",
    "determinism",
    "contamination",
    "mutable artifact",
    "mutable",
    "stale oos",
    "production/research",
    "seed is null",
    "null hash",
})


def classify_finding(category: str, message: str) -> Severity:
    """
    Determine Severity for a finding based on category and message content.

    Critical keyword match always wins, regardless of category.
    """
    base = _CATEGORY_SEVERITY.get(category.upper(), Severity.MEDIUM)
    msg_lower = message.lower()
    if any(kw in msg_lower for kw in _CRITICAL_KEYWORDS):
        return Severity.CRITICAL
    return base


def make_finding(category: str, message: str, source: str = "") -> Finding:
    """Convenience constructor with auto-classified severity."""
    sev = classify_finding(category, message)
    return Finding(severity=sev, category=category, message=message, source=source)


def findings_from_gate_output(
    errors: list[str],
    warnings: list[str],
) -> list[Finding]:
    """
    Convert governance_gate (errors, warnings) tuples to typed Finding list.

    Parses category from gate prefix: "[OOS] message" → category=OOS.
    """
    findings: list[Finding] = []

    for msg in errors:
        cat, body = _split_prefix(msg)
        findings.append(make_finding(cat, body))

    for msg in warnings:
        cat, body = _split_prefix(msg)
        f = make_finding(cat, body)
        # Warnings from gate are never CRITICAL (the gate already decided)
        if f.severity == Severity.CRITICAL:
            f = Finding(Severity.HIGH, f.category, f.message, f.source)
        findings.append(f)

    return findings


def _split_prefix(msg: str) -> tuple[str, str]:
    """Extract [CATEGORY] prefix from gate messages, e.g. '[OOS] foo' → ('OOS', 'foo')."""
    if msg.startswith("[") and "]" in msg:
        bracket_end = msg.index("]")
        cat = msg[1:bracket_end].strip()
        body = msg[bracket_end + 1:].strip()
        return cat, body
    return "GENERAL", msg


def max_severity(findings: list[Finding]) -> Optional[Severity]:
    """Return highest severity in a list, or None if empty."""
    if not findings:
        return None
    return max(f.severity for f in findings)


def count_by_severity(findings: list[Finding]) -> dict[str, int]:
    """Return {severity_name: count} dict."""
    counts: dict[str, int] = {s.value: 0 for s in Severity}
    for f in findings:
        counts[f.severity.value] += 1
    return counts
