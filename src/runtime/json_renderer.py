"""
src/runtime/json_renderer.py
Pure structural renderer: GovernanceSummary → JSON string.

ZERO business logic.
ZERO arithmetic.
ZERO conditional evaluation.

Receives a fully-evaluated GovernanceSummary (with anomalies and score
already attached) and renders it to a stable JSON string.
Field ordering is deterministic (sort_keys=True).
"""
from __future__ import annotations

import dataclasses
import json
from typing import Any

from src.runtime.runtime_summary import GovernanceSummary


class JSONRenderer:
    """
    Stateless JSON renderer.

    render() is pure: same GovernanceSummary → same JSON string.
    indent=2 for human readability; sort_keys for replay stability.
    """

    def render(self, summary: GovernanceSummary) -> str:
        d = _to_dict(summary)
        return json.dumps(d, indent=2, sort_keys=True, default=_json_default)

    def render_dict(self, summary: GovernanceSummary) -> dict:
        return _to_dict(summary)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — structural only
# ─────────────────────────────────────────────────────────────────────────────

def _to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            k: _to_dict(v)
            for k, v in dataclasses.asdict(obj).items()
        }
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return _to_dict(obj)
    return str(obj)
