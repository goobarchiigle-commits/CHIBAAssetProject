"""
rollout_phase.py — Production Rollout Phase Switcher

Usage:
  python tools/rollout_phase.py --phase 1    # Addon only
  python tools/rollout_phase.py --phase 2    # Addon + VOL_ADJ
  python tools/rollout_phase.py --phase 3    # All three features
  python tools/rollout_phase.py --rollback   # Disable all (= phase 0)
  python tools/rollout_phase.py --status     # Show current state

Phase definitions:
  Phase 1: eq_scale_addon=ON, vol_adj=OFF, atr_extension=OFF
  Phase 2: eq_scale_addon=ON, vol_adj=ON,  atr_extension=OFF
  Phase 3: eq_scale_addon=ON, vol_adj=ON,  atr_extension=ON
  Phase 0: all OFF (rollback)
"""
from __future__ import annotations
import argparse, json, sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

STRATEGY_YAML = Path(__file__).resolve().parents[1] / "src" / "configs" / "strategy.yaml"
ROLLOUT_LOG   = Path(__file__).resolve().parents[1] / "logs" / "rollout_phase.jsonl"

PHASE_CONFIGS = {
    0: {"eq_scale_addon": False, "vol_adj": False, "atr_extension": False},
    1: {"eq_scale_addon": True,  "vol_adj": False, "atr_extension": False},
    2: {"eq_scale_addon": True,  "vol_adj": True,  "atr_extension": False},
    3: {"eq_scale_addon": True,  "vol_adj": True,  "atr_extension": True},
}

PHASE_NAMES = {0: "ROLLBACK", 1: "Phase1_Addon", 2: "Phase2_Addon+VOL", 3: "Phase3_All"}


def _read_yaml_raw(path: Path) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.readlines()


def _write_yaml_raw(path: Path, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _set_feature_enabled(lines: list[str], feature: str, value: bool) -> list[str]:
    """
    Set `enabled: true/false` under a specific feature block.
    Matches the first `enabled:` line after `<feature>:` in the research_candidates block.
    """
    val_str   = "true" if value else "false"
    in_rc     = False
    in_feat   = False
    done      = False
    out       = []

    for line in lines:
        stripped = line.strip()

        if "research_candidates:" in stripped:
            in_rc = True
        elif in_rc and not stripped.startswith("#") and stripped and not stripped.startswith("research_candidates"):
            # sub-block detection: check if line matches `  <feature>:`
            if f"{feature}:" in stripped and stripped.startswith(feature + ":"):
                in_feat = True
            elif in_feat and stripped.endswith(":") and not stripped.startswith("enabled"):
                # new sub-block inside research_candidates → left the feature block
                in_feat = False

        if in_rc and in_feat and stripped.startswith("enabled:") and not done:
            # Replace this line preserving indentation
            indent = len(line) - len(line.lstrip())
            line   = " " * indent + f"enabled: {val_str}\n"
            done   = True

        out.append(line)

    if not done:
        raise RuntimeError(f"Could not find `enabled:` under `{feature}:` in {path}")
    return out


def get_current_state(lines: list[str]) -> dict[str, bool]:
    """Parse current enabled state for all three features."""
    state: dict[str, bool] = {}
    features = ["atr_extension", "vol_adj", "eq_scale_addon"]
    in_rc    = False
    cur_feat = None

    for line in lines:
        stripped = line.strip()
        if "research_candidates:" in stripped:
            in_rc = True
            continue
        if not in_rc:
            continue

        for feat in features:
            if stripped == f"{feat}:":
                cur_feat = feat
                break

        if cur_feat and stripped.startswith("enabled:"):
            val             = stripped.split(":", 1)[1].strip().split()[0].lower()
            state[cur_feat] = val == "true"
            cur_feat        = None

    return state


def apply_phase(phase: int, dry_run: bool = False) -> None:
    cfg = PHASE_CONFIGS[phase]
    lines = _read_yaml_raw(STRATEGY_YAML)

    before = get_current_state(lines)
    for feat, enabled in cfg.items():
        lines = _set_feature_enabled(lines, feat, enabled)
    after = get_current_state(lines)

    changed = {k for k in cfg if before.get(k) != after.get(k)}

    if dry_run:
        print(f"[DRY RUN] Phase {phase} ({PHASE_NAMES[phase]}):")
        for feat, val in cfg.items():
            marker = "→ CHANGED" if feat in changed else "  (no change)"
            print(f"  {feat}: {before.get(feat)} → {val}  {marker}")
        return

    _write_yaml_raw(STRATEGY_YAML, lines)

    # Log
    ROLLOUT_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts":       datetime.now().isoformat(),
        "phase":    phase,
        "name":     PHASE_NAMES[phase],
        "before":   before,
        "after":    after,
        "changed":  list(changed),
    }
    with open(ROLLOUT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Phase {phase} ({PHASE_NAMES[phase]}) applied:")
    for feat, val in after.items():
        marker = " ← CHANGED" if feat in changed else ""
        icon   = "ON " if val else "OFF"
        print(f"  {feat:<20} {icon}{marker}")
    print(f"\nLogged: {ROLLOUT_LOG}")
    print("Next: run_live_signal.py will pick up new config on next execution.")


def show_status() -> None:
    lines = _read_yaml_raw(STRATEGY_YAML)
    state = get_current_state(lines)

    # Determine current phase
    for ph, cfg in PHASE_CONFIGS.items():
        if all(state.get(k) == v for k, v in cfg.items()):
            cur_phase = ph
            break
    else:
        cur_phase = -1

    print(f"Current rollout state (strategy.yaml):")
    for feat, val in state.items():
        print(f"  {feat:<20} {'ON' if val else 'OFF'}")
    label = PHASE_NAMES.get(cur_phase, "CUSTOM")
    print(f"\nPhase: {cur_phase} ({label})")

    # Last log entry
    if ROLLOUT_LOG.exists():
        with open(ROLLOUT_LOG, encoding="utf-8") as f:
            entries = [json.loads(l) for l in f if l.strip()]
        if entries:
            last = entries[-1]
            print(f"Last change: {last['ts']}  → {last['name']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--phase", type=int, choices=[0, 1, 2, 3])
    grp.add_argument("--rollback", action="store_true")
    grp.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.rollback:
        apply_phase(0, dry_run=args.dry_run)
    else:
        apply_phase(args.phase, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
