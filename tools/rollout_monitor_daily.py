"""
rollout_monitor_daily.py — Daily Production Rollout Monitor

Reads runtime state files and today's decision logs.
Emits: feature activation counts, portfolio utilization,
       rollback-trigger flags, per-position lifecycle.

Usage:
  python tools/rollout_monitor_daily.py
  python tools/rollout_monitor_daily.py --date 2026-06-28
  python tools/rollout_monitor_daily.py --json   # machine-readable output
"""
from __future__ import annotations
import argparse, json, sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from src.paths import (
    RUNTIME_DIR, EQ_SCALE_ADDON_STATE_FILE,
    ATR_EXT_DEFER_STATE_FILE, VOL_ADJ_STATE_FILE,
    ADDON_DECISIONS_FILE, EXIT_POLICY_DECISIONS_FILE,
    EXECUTION_TRUTH_FILE,
)

PORTFOLIO_STATE  = RUNTIME_DIR / "portfolio_state.json"
STRATEGY_YAML    = Path(__file__).resolve().parents[1] / "src" / "configs" / "strategy.yaml"
DAILY_LOG_FILE   = RUNTIME_DIR / "rollout_daily_monitor.jsonl"
CAPITAL_STATE    = RUNTIME_DIR / "capital_state.json"


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"_error": str(e)}


def _load_jsonl(path: Path, date_str: str | None = None) -> list[dict]:
    if not path.exists():
        return []
    records = []
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if date_str:
                    ts = rec.get("ts") or rec.get("timestamp") or rec.get("date") or ""
                    if not str(ts).startswith(date_str):
                        continue
                records.append(rec)
    except Exception:
        pass
    return records


def _feature_enabled(feature: str) -> bool:
    try:
        lines = STRATEGY_YAML.read_text(encoding="utf-8").splitlines()
        in_rc   = False
        in_feat = False
        for line in lines:
            s = line.strip()
            if "research_candidates:" in s:
                in_rc = True
                continue
            if not in_rc:
                continue
            if s == f"{feature}:":
                in_feat = True
                continue
            if in_feat and s.startswith("enabled:"):
                return s.split(":", 1)[1].strip().split()[0].lower() == "true"
            if in_feat and s.endswith(":") and not s.startswith("enabled"):
                in_feat = False
    except Exception:
        pass
    return False


def _pnl_str(v: float | None) -> str:
    if v is None:
        return "—"
    sign = "+" if v >= 0 else ""
    return f"{sign}¥{v:,.0f}"


def build_report(target_date: str) -> dict:
    report: dict = {
        "date":          target_date,
        "ts":            datetime.now().isoformat(),
        "features":      {},
        "portfolio":     {},
        "addon":         {},
        "vol_adj":       {},
        "atr_extension": {},
        "rollback_flags": [],
        "violations":    [],
    }

    # ── Feature enable state ────────────────────────────────────────────────
    for feat in ("eq_scale_addon", "vol_adj", "atr_extension"):
        report["features"][feat] = _feature_enabled(feat)

    # ── Portfolio state ─────────────────────────────────────────────────────
    ps = _load_json(PORTFOLIO_STATE) or {}
    if isinstance(ps, dict) and "_error" not in ps:
        syms       = set(ps.get("position_entry_dates", {}).keys())
        prices     = ps.get("position_current_prices", {})
        entry_px   = ps.get("position_entry_prices",  {})
        entry_dates= ps.get("position_entry_dates",   {})
        entry_atrs = ps.get("position_entry_atrs",    {})
        qtys       = ps.get("position_qtys",          {})
        high_close = ps.get("position_highest_closes",{})
        unreal_pnl = ps.get("position_unrealized_pnl",{})
        unreal_pct = ps.get("position_unrealized_pct",{})
        last_equity= ps.get("last_equity",  0.0) or 0.0
        cash       = ps.get("available_cash", 0.0) or 0.0
        equity_peak= ps.get("equity_peak",  0.0) or 0.0

        dd = (last_equity - equity_peak) / max(1, equity_peak) * 100 if equity_peak else 0.0
        exposure   = (last_equity - cash) / max(1, last_equity) * 100 if last_equity else 0.0

        positions = []
        for sym in syms:
            cur  = prices.get(sym)
            epx  = entry_px.get(sym)
            atr  = entry_atrs.get(sym)
            qty  = qtys.get(sym, 0)
            hcl  = high_close.get(sym)
            upnl = unreal_pnl.get(sym)
            upct = unreal_pct.get(sym)
            edt  = entry_dates.get(sym)
            value= (cur or 0) * qty
            weight = value / max(1, last_equity) * 100

            # max_single_weight check (25% × 1.5 = 37.5%)
            if weight > 37.5:
                report["violations"].append(
                    f"MAX_WEIGHT_VIOLATION: {sym} weight={weight:.1f}% > 37.5%"
                )

            positions.append({
                "symbol":         sym,
                "qty":            qty,
                "entry_px":       epx,
                "cur_px":         cur,
                "entry_atr":      atr,
                "highest_close":  hcl,
                "entry_date":     edt,
                "unrealized_pnl": upnl,
                "unrealized_pct": upct,
                "value":          value,
                "weight_pct":     round(weight, 1),
            })

        report["portfolio"] = {
            "n_positions":   len(syms),
            "last_equity":   last_equity,
            "available_cash": cash,
            "equity_peak":   equity_peak,
            "drawdown_pct":  round(dd, 2),
            "exposure_pct":  round(exposure, 1),
            "positions":     positions,
        }

        if dd <= -15.0:
            report["violations"].append(
                f"DD_LIMIT_BREACH: drawdown={dd:.1f}% ≤ -15.0% → BUY_STOP"
            )
    else:
        report["portfolio"]["_error"] = str(ps)

    # ── Addon state ─────────────────────────────────────────────────────────
    addon_state = _load_json(EQ_SCALE_ADDON_STATE_FILE) or {}
    addon_decisions_today = _load_jsonl(ADDON_DECISIONS_FILE, target_date)

    # Duplicate detection: same (symbol, entry_date) with "EXECUTED" twice
    seen_exec: dict[tuple, int] = {}
    addon_executed = []
    addon_rejected = []

    for rec in addon_decisions_today:
        decision = rec.get("decision", "")
        sym      = rec.get("symbol", "")
        edt      = rec.get("entry_date", "")
        key      = (sym, edt)
        if decision == "EXECUTE":
            addon_executed.append(rec)
            seen_exec[key] = seen_exec.get(key, 0) + 1
            if seen_exec[key] > 1:
                report["rollback_flags"].append(
                    f"DUPLICATE_ADDON: {sym} entry_date={edt} executed {seen_exec[key]}x"
                )
        else:
            addon_rejected.append(rec)

    # State corruption check
    if isinstance(addon_state, dict) and "_error" in addon_state:
        report["rollback_flags"].append(f"STATE_CORRUPTION: eq_scale_addon_state.json — {addon_state['_error']}")

    # Stale entry check: state keys not in current positions
    ps_syms = {p["symbol"] for p in report.get("portfolio", {}).get("positions", [])}
    stale   = [k for k in (addon_state or {}) if k not in ps_syms]
    # (stale entries are purged at next run — not a flag, just info)

    report["addon"] = {
        "enabled":             report["features"]["eq_scale_addon"],
        "state_entries":       len(addon_state) if isinstance(addon_state, dict) else 0,
        "stale_entries":       stale,
        "today_executed":      len(addon_executed),
        "today_rejected":      len(addon_rejected),
        "today_rejected_reasons": list({r.get("reason", "UNKNOWN") for r in addon_rejected}),
        "duplicate_detected":  any("DUPLICATE_ADDON" in f for f in report["rollback_flags"]),
        "addon_details":       addon_executed[:10],
    }

    # ── VOL_ADJ state ────────────────────────────────────────────────────────
    vol_state = _load_json(VOL_ADJ_STATE_FILE) or {}
    if isinstance(vol_state, dict) and "_error" in vol_state:
        report["rollback_flags"].append(f"STATE_CORRUPTION: vol_adj_state.json — {vol_state['_error']}")

    eff_max  = vol_state.get("effective_max_positions", 3) if isinstance(vol_state, dict) else 3
    n_pos    = report.get("portfolio", {}).get("n_positions", 0)
    util_pct = round(n_pos / max(1, eff_max) * 100, 1)

    # Position cap violation: held > effective_max
    if n_pos > eff_max:
        report["violations"].append(
            f"POSITION_CAP_VIOLATION: n_held={n_pos} > effective_max={eff_max}"
        )

    report["vol_adj"] = {
        "enabled":              report["features"]["vol_adj"],
        "effective_max_pos":    eff_max,
        "n_positions":          n_pos,
        "slot_utilization_pct": util_pct,
        "computed_at":          vol_state.get("computed_at", "—") if isinstance(vol_state, dict) else "—",
        "topix_vol_20d":        vol_state.get("topix_vol_20d", None) if isinstance(vol_state, dict) else None,
    }

    # ── ATR Extension state ───────────────────────────────────────────────────
    atr_state = _load_json(ATR_EXT_DEFER_STATE_FILE) or {}
    if isinstance(atr_state, dict) and "_error" in atr_state:
        report["rollback_flags"].append(f"STATE_CORRUPTION: atr_ext_defer_state.json — {atr_state['_error']}")

    exit_dec_today = _load_jsonl(EXIT_POLICY_DECISIONS_FILE, target_date)
    ext_today = [r for r in exit_dec_today if r.get("decision") == "DEFER"]

    # Deferred entries
    deferred_syms = list(atr_state.keys()) if isinstance(atr_state, dict) else []
    deferred_details = []
    for sym, info in (atr_state if isinstance(atr_state, dict) else {}).items():
        days_held = None
        if info.get("first_defer_date"):
            try:
                d0 = datetime.strptime(info["first_defer_date"][:10], "%Y-%m-%d").date()
                days_held = (date.today() - d0).days
            except Exception:
                pass
        deferred_details.append({
            "symbol":       sym,
            "first_defer":  info.get("first_defer_date"),
            "expire_date":  info.get("defer_expires"),
            "days_deferred": days_held,
        })

    report["atr_extension"] = {
        "enabled":            report["features"]["atr_extension"],
        "currently_deferred": deferred_syms,
        "deferred_count":     len(deferred_syms),
        "today_new_deferrals": len(ext_today),
        "deferred_details":  deferred_details,
    }

    # ── Execution truth (signals generated / orders executed) ─────────────────
    exec_today = _load_jsonl(EXECUTION_TRUTH_FILE, target_date)
    buys  = [r for r in exec_today if r.get("side") == "BUY"]
    sells = [r for r in exec_today if r.get("side") == "SELL"]
    report["execution"] = {
        "signals_executed_today": len(exec_today),
        "buys":  len(buys),
        "sells": len(sells),
        "records": exec_today[:10],
    }

    # ── Rollback flag: unexpected order ───────────────────────────────────────
    # BUY order for symbol already held → check if it was an addon (expected) or not
    ps_syms_set = {p["symbol"] for p in report.get("portfolio", {}).get("positions", [])}
    for rec in buys:
        sym = rec.get("symbol", "")
        if sym in ps_syms_set:
            # Check if it was an addon order
            addon_syms = {r.get("symbol") for r in addon_executed}
            if sym not in addon_syms:
                report["rollback_flags"].append(
                    f"UNEXPECTED_BUY: {sym} already held and not an addon order"
                )

    return report


def print_report(r: dict) -> None:
    print("=" * 68)
    print(f"Production Rollout Monitor — {r['date']}")
    print("=" * 68)

    # Feature state
    feat = r["features"]
    print(f"\n  Feature State:")
    for k, v in feat.items():
        print(f"    {k:<20} {'ON' if v else 'OFF'}")

    # Portfolio
    pf = r["portfolio"]
    if "n_positions" in pf:
        print(f"\n  Portfolio:")
        print(f"    Equity:      ¥{pf['last_equity']:>12,.0f}  Peak: ¥{pf['equity_peak']:>12,.0f}")
        print(f"    Cash:        ¥{pf['available_cash']:>12,.0f}")
        print(f"    DrawDown:    {pf['drawdown_pct']:>+8.2f}%")
        print(f"    Exposure:    {pf['exposure_pct']:>8.1f}%")
        print(f"    Positions:   {pf['n_positions']}")
        print(f"\n    Position Lifecycle:")
        hdr = f"    {'Symbol':<10} {'Qty':>6} {'EntryPx':>9} {'CurPx':>8} {'U_PnL':>12} {'Wt%':>5} {'EntryDate'}"
        print(hdr)
        print("    " + "─" * (len(hdr) - 4))
        for p in pf["positions"]:
            upnl  = p["unrealized_pnl"]
            upct  = p["unrealized_pct"] or 0
            upnl_s= _pnl_str(upnl)
            print(
                f"    {p['symbol']:<10} {p['qty']:>6} {p['entry_px'] or 0:>9.0f}"
                f" {p['cur_px'] or 0:>8.0f} {upnl_s:>12} {p['weight_pct']:>4.1f}%"
                f" {p['entry_date'] or '—'}"
            )
            if p["entry_atr"] and p["highest_close"] and p["cur_px"]:
                thr = p["highest_close"] - p["entry_atr"]
                dist = p["cur_px"] - thr
                print(f"      ATR_threshold={thr:.0f}  dist_to_threshold={dist:+.0f}")

    # VOL_ADJ
    va = r["vol_adj"]
    print(f"\n  VOL_ADJ:")
    print(f"    Effective max_positions: {va['effective_max_pos']}  "
          f"(TOPIX 20d vol: {va.get('topix_vol_20d') or '—'})")
    print(f"    Slot utilization:        {va['n_positions']}/{va['effective_max_pos']}"
          f"  ({va['slot_utilization_pct']:.0f}%)")
    print(f"    Computed at:             {va['computed_at']}")

    # Addon
    ad = r["addon"]
    print(f"\n  Addon (EQ_SCALE):")
    print(f"    Today executed:  {ad['today_executed']}")
    print(f"    Today rejected:  {ad['today_rejected']}")
    if ad["today_rejected_reasons"]:
        print(f"    Reject reasons:  {', '.join(ad['today_rejected_reasons'])}")
    print(f"    State entries:   {ad['state_entries']}")
    if ad["stale_entries"]:
        print(f"    Stale entries:   {ad['stale_entries']}")
    print(f"    Duplicate flag:  {'⚠ YES' if ad['duplicate_detected'] else 'clean'}")

    # ATR Extension
    ae = r["atr_extension"]
    print(f"\n  ATR Extension:")
    print(f"    Currently deferred: {ae['deferred_count']} positions")
    print(f"    New deferrals today: {ae['today_new_deferrals']}")
    if ae["deferred_details"]:
        for d in ae["deferred_details"]:
            print(f"    {d['symbol']}: deferred {d['days_deferred']}d  "
                  f"since={d['first_defer']}  expires={d['expire_date']}")

    # Execution
    ex = r["execution"]
    print(f"\n  Execution:")
    print(f"    Signals executed today: {ex['signals_executed_today']}  "
          f"(BUY={ex['buys']} SELL={ex['sells']})")

    # Violations
    viol = r["violations"]
    flags = r["rollback_flags"]
    print(f"\n  " + "─" * 64)
    if not viol and not flags:
        print(f"  Rollback Triggers: NONE — clean run")
    else:
        if viol:
            print(f"  ⛔ VIOLATIONS ({len(viol)}):")
            for v in viol:
                print(f"    {v}")
        if flags:
            print(f"  ⚠ ROLLBACK FLAGS ({len(flags)}):")
            for fl in flags:
                print(f"    {fl}")
            print(f"\n  Rollback: python tools/rollout_phase.py --rollback")
    print("=" * 68)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date",   default=date.today().strftime("%Y-%m-%d"))
    parser.add_argument("--json",   action="store_true")
    args = parser.parse_args()

    r = build_report(args.date)

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    else:
        print_report(r)

    # Persist to log
    DAILY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DAILY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
