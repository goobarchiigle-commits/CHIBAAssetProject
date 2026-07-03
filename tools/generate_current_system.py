"""
tools/generate_current_system.py
Generates docs/CURRENT_SYSTEM.md — the authoritative live system snapshot.

Reads from:
  state/strategy_registry.json   ← active strategies and OOS results
  src/configs/strategy.yaml      ← production config
  src/research_state.md          ← current phase / known issues (first section)
  runtime/portfolio_state.json   ← current capital (if exists)

Writes:
  docs/CURRENT_SYSTEM.md

IMPORTANT:
  This document is the ONLY thing Claude should read to understand the current system.
  It is always regenerated from canonical sources — never edit manually.

Usage:
    python tools/generate_current_system.py
    python tools/generate_current_system.py --dry-run   # print to stdout only
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO_ROOT    = Path(__file__).resolve().parent.parent
STATE_DIR    = REPO_ROOT / "state"
DOCS_DIR     = REPO_ROOT / "docs"
RUNTIME_DIR  = REPO_ROOT / "runtime"
CONFIGS_DIR  = REPO_ROOT / "src" / "configs"

REGISTRY_FILE      = STATE_DIR / "strategy_registry.json"
STRATEGY_YAML      = CONFIGS_DIR / "strategy.yaml"
RESEARCH_STATE     = REPO_ROOT / "src" / "research_state.md"
PORTFOLIO_STATE    = RUNTIME_DIR / "portfolio_state.json"
OUTPUT_FILE        = DOCS_DIR / "CURRENT_SYSTEM.md"

JST = timezone(timedelta(hours=9))


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_yaml_raw(path: Path) -> str:
    if not path.exists():
        return "(strategy.yaml not found)"
    return path.read_text(encoding="utf-8")


def _load_research_state_header(path: Path, max_lines: int = 35) -> str:
    if not path.exists():
        return "(research_state.md not found)"
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(lines[:max_lines])


def _load_portfolio(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def generate(dry_run: bool = False) -> str:
    now = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")

    registry_raw = _load_json(REGISTRY_FILE)
    strategies   = {k: v for k, v in registry_raw.items() if not k.startswith("_")}
    active       = {k: v for k, v in strategies.items() if v.get("status") == "active"}
    experimental = {k: v for k, v in strategies.items() if v.get("status") == "experimental"}

    yaml_raw   = _load_yaml_raw(STRATEGY_YAML)
    rs_header  = _load_research_state_header(RESEARCH_STATE)
    portfolio  = _load_portfolio(PORTFOLIO_STATE)

    # ── Active strategies block ──────────────────────────────────────────────
    strat_lines: list[str] = []
    for name, entry in active.items():
        strat_lines.append(f"### `{name}`")
        strat_lines.append(f"- **Status**: {entry.get('status', 'unknown')}")
        strat_lines.append(f"- **Source file**: `{entry.get('source_file', 'N/A')}`")
        strat_lines.append(f"- **Config**: `{entry.get('active_config', 'N/A')}`")
        strat_lines.append(f"- **OOS period**: {entry.get('oos_period', 'N/A')}")
        strat_lines.append(f"- **OOS Sharpe**: {entry.get('oos_sharpe', 'N/A')}")
        strat_lines.append(f"- **OOS MaxDD**: {entry.get('oos_max_dd', 'N/A')}")
        strat_lines.append(f"- **OOS CAGR**: {entry.get('oos_cagr', 'N/A')}")
        strat_lines.append(f"- **WF result**: {entry.get('wf_result', 'N/A')}")
        strat_lines.append(f"- **Last validated**: {entry.get('last_validated_at', 'N/A')}")
        eps = entry.get("live_entrypoints", [])
        if eps:
            strat_lines.append(f"- **Live entrypoints**: {', '.join(f'`{e}`' for e in eps)}")
        notes = entry.get("notes", "")
        if notes:
            strat_lines.append(f"- **Notes**: {notes}")
        strat_lines.append("")

    # ── Experimental strategies block ────────────────────────────────────────
    exp_lines: list[str] = []
    for name, entry in experimental.items():
        exp_lines.append(f"### `{name}` _(experimental — NOT in production)_")
        exp_lines.append(f"- **Source file**: `{entry.get('source_file', 'N/A')}`")
        exp_lines.append(f"- **Notes**: {entry.get('notes', 'No notes')}")
        exp_lines.append("")

    # ── Capital snapshot ─────────────────────────────────────────────────────
    if portfolio:
        cap_lines = ["```json", json.dumps(portfolio, ensure_ascii=False, indent=2), "```"]
    else:
        cap_lines = [
            "_runtime/portfolio_state.json not found._",
            "_Run morning sync before reading capital figures._",
        ]

    # ── Execution protections from strategy.yaml ─────────────────────────────
    # Extract key risk_controls and live_execution sections
    protections: list[str] = []
    in_risk     = False
    in_live     = False
    for line in yaml_raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("risk_controls:"):
            in_risk = True
            in_live = False
        elif stripped.startswith("live_execution:"):
            in_live = True
            in_risk = False
        elif stripped.startswith("fraction:"):
            in_risk = False
            in_live = False
        elif not stripped.startswith(" ") and not stripped.startswith("#") and stripped:
            if in_risk or in_live:
                in_risk = False
                in_live = False
        if (in_risk or in_live) and stripped and not stripped.startswith("#"):
            protections.append(f"  {line.rstrip()}")

    doc = f"""# CURRENT_SYSTEM.md — CHIBAAssetProject Live System Snapshot
> **Generated**: {now}
> **IMPORTANT**: This file is auto-generated by `tools/generate_current_system.py`.
> Do NOT edit manually. Regenerate after any registry or config change.

---

## 1. Active Strategies

{chr(10).join(strat_lines) if strat_lines else "_No active strategies registered._"}

---

## 2. Experimental / Research Strategies

{chr(10).join(exp_lines) if exp_lines else "_None._"}

---

## 3. Current Capital Snapshot

{chr(10).join(cap_lines)}

---

## 4. Production Config (`src/configs/strategy.yaml`)

<details>
<summary>Full strategy.yaml (click to expand)</summary>

```yaml
{yaml_raw}
```

</details>

---

## 5. Enabled Protections & Feature Flags

Extracted from `strategy.yaml`:

```yaml
{chr(10).join(protections) if protections else "  (could not extract — read strategy.yaml directly)"}
```

---

## 6. Execution Pipeline

| Step | Module | Description |
|------|--------|-------------|
| 1 | `src/run_morning_signal.py` | Morning entry: API check, dry-run, signal summary |
| 2 | `src/execution/adaptive_alloc.py` | Adaptive allocation with sector/symbol caps |
| 3 | `src/execution/replay_guard.py` | Storm protection: auction block, cooldown, throttle |
| 4 | `src/execution/deferred_retry.py` | Deferred entry retry (regime drift invalidation) |
| 5 | `src/kabusapi/signal_bridge.py` | kabuステーション REST API order submission |
| 6 | `src/execution/reconcile.py` | Post-fill reconciliation |
| 7 | `src/live/position_sync.py` | Broker position truth enforcement |

---

## 7. Production Entrypoints

| Script | Purpose |
|--------|---------|
| `src/run_morning_signal.py` | Daily morning routine (dry-run + live) |
| `src/run_live_signal.py` | Live signal execution |

```
# Dry run (required before live)
python src/run_morning_signal.py

# Live execution (LIVE_MODE=true in .env)
python src/run_live_signal.py --live --yes
```

---

## 8. Research State (current phase header)

From `src/research_state.md`:

```
{rs_header}
```

---

## 9. Registry Source

Full registry: `state/strategy_registry.json`
Validate: `python tools/consistency_check.py`
Audit: `python tools/repo_audit.py`
"""
    return doc


def _cli() -> None:
    parser = argparse.ArgumentParser(
        description="Generate docs/CURRENT_SYSTEM.md from canonical sources."
    )
    parser.add_argument("--dry-run", action="store_true", help="Print to stdout, do not write")
    args = parser.parse_args()

    doc = generate(dry_run=args.dry_run)

    if args.dry_run:
        print(doc)
    else:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(doc, encoding="utf-8")
        print(f"[GENERATE_CURRENT_SYSTEM] Written: {OUTPUT_FILE}")


if __name__ == "__main__":
    _cli()
