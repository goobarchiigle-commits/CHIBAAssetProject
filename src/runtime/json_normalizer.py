"""
src/runtime/json_normalizer.py — lightweight numpy scalar → Python native cast.

Only converts numpy scalars that json.dumps cannot handle:
  np.bool_  → bool
  np.integer subclasses (int8/16/32/64 etc.) → int
  np.floating subclasses (float16/32/64 etc.) → float
  other np.generic → str fallback

No recursion, no deep traversal, no heavy serializers.
Call normalize_json_scalar() only at JSONL append boundaries.
"""
from __future__ import annotations
from typing import Any


def normalize_json_scalar(value: Any) -> Any:
    """
    Convert a numpy scalar to its Python native equivalent.
    Python native types pass through unchanged.
    """
    typ = type(value)
    type_name = typ.__name__

    # Fast path: most common Python native types
    if typ in (bool, int, float, str, type(None)):
        return value

    # numpy detection by module prefix (avoids importing numpy at module load)
    module = getattr(typ, "__module__", "") or ""
    if not module.startswith("numpy"):
        return value

    if type_name == "bool_":
        return bool(value)
    if issubclass(typ, int):         # np.integer hierarchy
        return int(value)
    if issubclass(typ, float):       # np.floating hierarchy
        return float(value)
    # np.generic catch-all (complexes, voids, etc.) — stringify
    return str(value)


def normalize_for_json(d: dict) -> dict:
    """
    Shallow-normalize all values in a flat dict.
    Only one level deep — use for dataclass.asdict() output on JSONL append.
    """
    return {k: normalize_json_scalar(v) for k, v in d.items()}
