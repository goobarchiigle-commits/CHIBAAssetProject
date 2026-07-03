"""
tools/fingerprint_dataset.py
Dataset fingerprinting for reproducibility.

Computes:
  - sha256 of raw file bytes
  - row_count
  - schema_hash (column names + dtypes)
  - datetime_range (first/last index value if DatetimeIndex)
  - nan_profile (per-column NaN count)

Stores results in state/dataset_registry.json (append/update by path).

Usage:
    python tools/fingerprint_dataset.py path/to/data.csv
    python tools/fingerprint_dataset.py path/to/data.parquet --name my_dataset
    python tools/fingerprint_dataset.py --list
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT      = Path(__file__).resolve().parent.parent
STATE_DIR      = REPO_ROOT / "state"
REGISTRY_FILE  = STATE_DIR / "dataset_registry.json"

JST = timezone(timedelta(hours=9))

logging.basicConfig(
    level=logging.INFO,
    format="[FINGERPRINT] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(REPO_ROOT))


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _schema_hash(df) -> str:
    """Hash column names + dtype names."""
    schema_str = json.dumps(
        {col: str(dtype) for col, dtype in df.dtypes.items()},
        sort_keys=True,
    )
    return hashlib.sha256(schema_str.encode()).hexdigest()[:16]


def _nan_profile(df) -> dict[str, int]:
    return {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().any()}


def _datetime_range(df) -> dict[str, str] | None:
    try:
        import pandas as pd
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 0:
            return {
                "start": str(df.index.min()),
                "end":   str(df.index.max()),
            }
    except Exception:
        pass
    return None


def fingerprint(
    data_path: Path,
    name: str | None = None,
) -> dict:
    """
    Compute fingerprint for a dataset file (CSV, Parquet, JSON).

    Returns fingerprint dict. Does NOT write to registry.

    Raises:
        FileNotFoundError: if data_path does not exist.
        ImportError:       if pandas is not available.
        ValueError:        if file format is unsupported.
    """
    import pandas as pd

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    sha = _file_sha256(data_path)

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    elif suffix in (".parquet", ".pq"):
        df = pd.read_parquet(data_path)
    elif suffix == ".json":
        df = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    try:
        rel_path = str(data_path.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        rel_path = str(data_path).replace("\\", "/")

    fp = {
        "name":           name or data_path.stem,
        "path":           rel_path,
        "sha256":         sha,
        "row_count":      int(len(df)),
        "column_count":   int(len(df.columns)),
        "schema_hash":    _schema_hash(df),
        "columns":        list(df.columns),
        "nan_profile":    _nan_profile(df),
        "datetime_range": _datetime_range(df),
        "file_size_bytes": int(data_path.stat().st_size),
        "created_at":     datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    return fp


def load_registry() -> dict:
    if not REGISTRY_FILE.exists():
        return {"_meta": {"schema_version": "1.0"}, "datasets": {}}
    try:
        data = json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
        if "datasets" not in data:
            data["datasets"] = {}
        return data
    except Exception as exc:
        logger.warning("Could not load dataset registry: %s", exc)
        return {"_meta": {"schema_version": "1.0"}, "datasets": {}}


def save_registry(data: dict) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(REGISTRY_FILE)


def register_dataset(
    data_path: Path,
    name: str | None = None,
) -> dict:
    """
    Fingerprint data_path and upsert into state/dataset_registry.json.

    Returns the fingerprint dict.
    """
    fp   = fingerprint(data_path, name=name)
    data = load_registry()
    key  = fp["name"]
    if key in data["datasets"]:
        logger.info("Updating existing fingerprint: %s", key)
    else:
        logger.info("Registering new dataset: %s", key)
    data["datasets"][key] = fp
    save_registry(data)
    return fp


def get_fingerprint(name: str) -> dict | None:
    """Return fingerprint dict for named dataset, or None."""
    return load_registry()["datasets"].get(name)


def verify_dataset(data_path: Path, name: str) -> bool:
    """
    Recompute sha256 and compare against registered fingerprint.

    Returns True if match, False if mismatch or not registered.
    """
    registry = load_registry()
    entry    = registry["datasets"].get(name)
    if entry is None:
        logger.warning("[FINGERPRINT] '%s' not in registry — cannot verify", name)
        return False
    expected = entry.get("sha256", "")
    actual   = _file_sha256(data_path)
    if expected != actual:
        logger.error(
            "[FINGERPRINT] Hash mismatch for '%s': expected=%s actual=%s",
            name, expected[:16], actual[:16],
        )
        return False
    logger.info("[FINGERPRINT] Verified: %s (sha256=%s)", name, actual[:16])
    return True


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Fingerprint a dataset and register in state/dataset_registry.json"
    )
    parser.add_argument("path", nargs="?", default="", help="Path to dataset file")
    parser.add_argument("--name",   default="", help="Dataset name (default: file stem)")
    parser.add_argument("--list",   action="store_true", help="List all registered datasets")
    parser.add_argument("--verify", default="", metavar="NAME",
                        help="Verify dataset at PATH matches registered NAME")
    args = parser.parse_args()

    if args.list:
        data = load_registry()
        for k, v in data["datasets"].items():
            print(f"  {k}: rows={v.get('row_count')} sha256={v.get('sha256','')[:16]}")
        return

    if not args.path:
        parser.error("path argument required unless --list")

    target = REPO_ROOT / args.path.strip()

    if args.verify:
        ok = verify_dataset(target, args.verify)
        sys.exit(0 if ok else 1)

    fp = register_dataset(target, name=args.name or None)
    logger.info("Fingerprint: %s", json.dumps(fp, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    _cli()
