"""
research/quarantine/__init__.py
IMPORT GUARD — this package cannot be imported.

Quarantined experiments are moved here automatically when they fail validation.
Production code must never import from this directory.
"""
raise ImportError(
    "research/quarantine/ is a restricted zone. "
    "Code in quarantine has failed validation and cannot be used in production. "
    "To rehabilitate: fix the experiment and promote via tools/promote_strategy.py."
)
