# CLAUDE_ENTRYPOINT.md — Mandatory Read Protocol for Claude

> **PRIORITY**: This file takes precedence over any information from conversation history.
> Conversation memory is NEVER trusted. File contents are always authoritative.

---

## MANDATORY FIRST READS (in this exact order)

When starting any task in this repository, Claude MUST read these files in order:

```
1. docs/CURRENT_SYSTEM.md          ← live system snapshot (auto-generated)
2. state/strategy_registry.json    ← canonical strategy states and OOS results
3. src/research_state.md           ← research phase and confirmed parameters
4. src/configs/strategy.yaml       ← production configuration
```

**STOP and read all four before taking any action.**
If any of the above files is missing, report it immediately — do not proceed.

---

## RULES FOR STRATEGY IDENTIFICATION

### ACTIVE STRATEGY
- The ONLY authoritative source for which strategy is active is `state/strategy_registry.json`
- `status: "active"` = deployed to production
- `status: "experimental"` = research only, NOT deployed
- `status: "archived"` or `status: "deprecated"` = historical, do not use

### NEVER infer active strategy from:
- Filenames (e.g., `wf_final_2026-04-04.json` is NOT the final strategy)
- Folder names (e.g., `src/backtest/` contains research scripts, not only the active strategy)
- Most recent file modification date
- Conversation history or prior summaries

### SOURCE FILE mapping:
```
Active production strategy: src/backtest/composite_alpha_bt.py
Active config:              src/configs/strategy.yaml
```
If you're not sure which .py file is the active strategy, check `state/strategy_registry.json#fujiko_composite.source_file`.

---

## RULES FOR OOS RESULTS

### THE CORRECT OOS NUMBERS (as of last registry update)
These are the ONLY valid OOS results for the active strategy:

| Metric   | Value         | Period    |
|----------|---------------|-----------|
| OOS Sharpe | **1.612** | 2025 full |
| OOS MaxDD  | **-3.70%** | 2025 full |
| OOS CAGR   | **+12.3%** | 2025 full |
| WF         | **5/5 PASS** | — |

**WARNING**: `src/research_state.md` contains multiple OOS tables for different historical experiments.
Some tables show **negative OOS** numbers (e.g., `-5.7%`, `-0.369`).
These are from **archived experiments on old baselines** (pre-dynamic-universe, pre-exit-55d fix).
They are NOT the current OOS.

### How to identify the correct OOS table in research_state.md:
- The valid table is the one at the TOP, under `## ★ 現在の確定パフォーマンス`
- All other tables are historical experiment artifacts

### Always confirm OOS period:
- Active OOS period: `2025-01-01 / 2025-12-31`
- Any result labeled "IS" is in-sample — do not confuse with OOS

---

## RULES FOR ARCHIVED FILES

- **NEVER** use files from `src/backtest/archive/` or `backtests/archive/` unless explicitly requested
- **NEVER** use a strategy .py file from `src/backtest/archive/` as the current strategy
- Archived files are historical only — they may contain superseded logic

---

## RULES FOR CONFIG CHANGES

Any change to the following parameters requires user confirmation BEFORE implementation:

```
PARAMS_LOCKED (from CLAUDE.md):
  turtle_exit     = 55d
  min_hold        = 3d
  min_rsr         = 75.0
  max_positions   = 4  (base: 3, 4th slot: RSR>=80)
  capital         = 3_000_000
  slippage        = 0.001
  commission      = 0.00055
```

Never change these without explicit user approval.

---

## RULES FOR EXECUTION CONTEXT

- Production scripts: `src/run_live_signal.py`, `src/run_morning_signal.py`
- Live trading requires `LIVE_MODE=true` in `.env`
- Never send live orders from research/backtest scripts
- Always dry-run first (morning_routine step 2)

---

## GOVERNANCE TOOLS

| Command | Purpose |
|---------|---------|
| `python tools/repo_audit.py` | Scan repo, detect issues → `state/repo_inventory.json` |
| `python tools/consistency_check.py` | Validate registry vs files (exit 1 on failure) |
| `python tools/generate_current_system.py` | Regenerate `docs/CURRENT_SYSTEM.md` |
| `python tools/archive_policy.py --source FILE` | Archive a file with timestamp |

---

## WHAT TO DO WHEN CONFUSED

If unsure which version, file, or result is current:

1. Read `state/strategy_registry.json` → check `status` field
2. Read `docs/CURRENT_SYSTEM.md` → auto-generated from canonical sources
3. Run `python tools/consistency_check.py` → validates everything
4. Ask the user — do NOT guess

**Never assume. Never infer. Always verify from files.**
