"""
rollout_monitor_weekly.py — Weekly Production Rollout Monitor

Aggregates 7-day window from daily_monitor logs and execution_truth records.
Reports: realized pnl, unrealized pnl, turnover, holding days,
         feature activation counts, performance vs baseline.

Usage:
  python tools/rollout_monitor_weekly.py
  python tools/rollout_monitor_weekly.py --weeks 2   # look back 2 weeks
  python tools/rollout_monitor_weekly.py --json
"""
from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout.reconfigure(encoding="utf-8")

from src.paths import (
    RUNTIME_DIR, ADDON_DECISIONS_FILE,
    EXIT_POLICY_DECISIONS_FILE, EXECUTION_TRUTH_FILE,
)

PORTFOLIO_STATE   = RUNTIME_DIR / "portfolio_state.json"
CAPITAL_STATE     = RUNTIME_DIR / "capital_state.json"
DAILY_MONITOR_LOG = RUNTIME_DIR / "rollout_daily_monitor.jsonl"
WEEKLY_LOG_FILE   = RUNTIME_DIR / "rollout_weekly_monitor.jsonl"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_jsonl_window(path: Path, since: date) -> list[dict]:
    if not path.exists():
        return []
    since_str = since.isoformat()
    out = []
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
                ts = str(rec.get("ts") or rec.get("timestamp") or rec.get("date") or "")
                if ts[:10] >= since_str:
                    out.append(rec)
    except Exception:
        pass
    return out


def _load_daily_logs(since: date) -> list[dict]:
    return _load_jsonl_window(DAILY_MONITOR_LOG, since)


def _pnl_str(v: float | None) -> str:
    if v is None:
        return "      —"
    sign = "+" if v >= 0 else ""
    return f"{sign}¥{v:>10,.0f}"


def build_weekly_report(since: date, until: date) -> dict:
    report = {
        "period_start": since.isoformat(),
        "period_end":   until.isoformat(),
        "ts":           datetime.now().isoformat(),
        "realized_pnl": {},
        "unrealized":   {},
        "addon":        {},
        "atr_ext":      {},
        "vol_adj":      {},
        "turnover":     {},
        "holding_days": {},
        "violations":   [],
        "rollback_flags": [],
    }

    # ── Realized PnL from execution truth ────────────────────────────────────
    exec_records = _load_jsonl_window(EXECUTION_TRUTH_FILE, since)
    sells = [r for r in exec_records if r.get("side") == "SELL"]
    buys  = [r for r in exec_records if r.get("side") == "BUY"]

    realized_total   = 0.0
    realized_by_day  = defaultdict(float)
    holds_days       = []
    turnover_total   = 0.0

    for s in sells:
        pnl   = float(s.get("pnl", 0) or 0)
        ts    = str(s.get("ts") or s.get("timestamp") or "")[:10]
        price = float(s.get("price", 0) or 0)
        qty   = int(s.get("qty", 0) or 0)
        realized_total += pnl
        realized_by_day[ts] += pnl
        if price and qty:
            turnover_total += price * qty

        # Holding days
        entry_ts = str(s.get("entry_ts") or s.get("entry_date") or "")[:10]
        exit_ts  = ts
        if entry_ts and exit_ts and entry_ts <= exit_ts:
            try:
                d0 = datetime.strptime(entry_ts, "%Y-%m-%d").date()
                d1 = datetime.strptime(exit_ts,  "%Y-%m-%d").date()
                holds_days.append((d1 - d0).days)
            except Exception:
                pass

    for b in buys:
        price = float(b.get("price", 0) or 0)
        qty   = int(b.get("qty", 0) or 0)
        if price and qty:
            turnover_total += price * qty

    n_wins  = sum(1 for s in sells if (s.get("pnl") or 0) > 0)
    n_loss  = sum(1 for s in sells if (s.get("pnl") or 0) < 0)
    win_rate= round(n_wins / max(1, len(sells)) * 100, 1) if sells else 0.0

    report["realized_pnl"] = {
        "total":       round(realized_total, 0),
        "trade_count": len(sells),
        "wins":        n_wins,
        "losses":      n_loss,
        "win_rate_pct": win_rate,
        "by_day":      dict(sorted(realized_by_day.items())),
    }

    # ── Unrealized PnL from current portfolio state ───────────────────────────
    ps = _load_json(PORTFOLIO_STATE)
    unreal_total = 0.0
    if ps and isinstance(ps, dict):
        up = ps.get("position_unrealized_pnl", {})
        unreal_total = sum(float(v or 0) for v in up.values())
        cur_prices   = ps.get("position_current_prices", {})
        entry_dates  = ps.get("position_entry_dates", {})
        entry_prices = ps.get("position_entry_prices", {})
        qtys         = ps.get("position_qtys", {})
        last_equity  = float(ps.get("last_equity", 0) or 0)

        positions_now = []
        for sym in set(ps.get("position_qtys", {}).keys()):
            pnl_v = float(up.get(sym, 0) or 0)
            positions_now.append({
                "symbol":    sym,
                "qty":       qtys.get(sym, 0),
                "cur_px":    cur_prices.get(sym),
                "entry_px":  entry_prices.get(sym),
                "entry_date": entry_dates.get(sym),
                "unreal_pnl": pnl_v,
            })

        report["unrealized"] = {
            "total":       round(unreal_total, 0),
            "last_equity": last_equity,
            "positions":   positions_now,
        }

    # ── Addon weekly stats ────────────────────────────────────────────────────
    addon_recs = _load_jsonl_window(ADDON_DECISIONS_FILE, since)
    addon_exec = [r for r in addon_recs if r.get("decision") == "EXECUTE"]
    addon_rej  = [r for r in addon_recs if r.get("decision") != "EXECUTE"]

    addon_pnl   = 0.0
    addon_wins  = 0
    addon_exits = [r for r in sells if "addon" in str(r.get("tags", "")).lower()
                   or "addon" in str(r.get("reason", "")).lower()]
    for t in addon_exits:
        pnl = float(t.get("pnl", 0) or 0)
        addon_pnl += pnl
        if pnl > 0:
            addon_wins += 1

    rej_reasons: dict[str, int] = defaultdict(int)
    for r in addon_rej:
        rej_reasons[r.get("reason", "UNKNOWN")] += 1

    report["addon"] = {
        "executed":      len(addon_exec),
        "rejected":      len(addon_rej),
        "reject_reasons": dict(rej_reasons),
        "addon_pnl":     round(addon_pnl, 0),
        "addon_wins":    addon_wins,
        "addon_win_rate_pct": round(addon_wins / max(1, len(addon_exits)) * 100, 1),
    }

    # ── ATR Extension weekly stats ────────────────────────────────────────────
    exit_dec = _load_jsonl_window(EXIT_POLICY_DECISIONS_FILE, since)
    ext_defers= [r for r in exit_dec if r.get("decision") == "DEFER"]
    ext_forced= [r for r in exit_dec if r.get("decision") == "FORCE_EXIT"]

    # Pnl from exits that were deferred (identify by tags if available)
    ext_sells   = [r for r in sells if "atr_ext" in str(r.get("tags", "")).lower()
                   or "SELL[多層RSR]" in str(r.get("original_reason", "")).lower()]
    ext_pnl     = sum(float(r.get("pnl", 0) or 0) for r in ext_sells)

    # Average added holding days from defer records
    added_days = []
    for r in ext_defers:
        diff = r.get("days_added") or r.get("defer_days")
        if diff is not None:
            added_days.append(int(diff))

    report["atr_ext"] = {
        "new_deferrals":    len(ext_defers),
        "forced_exits":     len(ext_forced),
        "ext_pnl":          round(ext_pnl, 0),
        "avg_days_added":   round(sum(added_days) / max(1, len(added_days)), 1) if added_days else None,
    }

    # ── VOL_ADJ weekly stats ─────────────────────────────────────────────────
    daily_logs = _load_daily_logs(since)
    va_days    = [d for d in daily_logs if d.get("vol_adj", {}).get("effective_max_pos") == 4]

    report["vol_adj"] = {
        "days_at_max4":    len(va_days),
        "days_total":      len(daily_logs),
        "calm_pct":        round(len(va_days) / max(1, len(daily_logs)) * 100, 1),
    }

    # ── Turnover ─────────────────────────────────────────────────────────────
    ps_equity = float((ps or {}).get("last_equity", 1) or 1)
    report["turnover"] = {
        "gross_turnover":     round(turnover_total, 0),
        "turnover_vs_equity": round(turnover_total / max(1, ps_equity) * 100, 1),
        "buy_count":          len(buys),
        "sell_count":         len(sells),
    }

    # ── Holding days ─────────────────────────────────────────────────────────
    report["holding_days"] = {
        "closed_positions":    len(holds_days),
        "avg_hold_days":       round(sum(holds_days) / max(1, len(holds_days)), 1) if holds_days else None,
        "min_hold_days":       min(holds_days) if holds_days else None,
        "max_hold_days":       max(holds_days) if holds_days else None,
        "distribution":        {
            "<=5d":  sum(1 for d in holds_days if d <= 5),
            "6-20d": sum(1 for d in holds_days if 6 <= d <= 20),
            "21-55d":sum(1 for d in holds_days if 21 <= d <= 55),
            ">55d":  sum(1 for d in holds_days if d > 55),
        },
    }

    # ── Violations from daily log ─────────────────────────────────────────────
    all_violations = []
    all_flags      = []
    for dl in daily_logs:
        all_violations.extend(dl.get("violations",    []))
        all_flags.extend(dl.get("rollback_flags", []))
    report["violations"]    = list(set(all_violations))
    report["rollback_flags"]= list(set(all_flags))

    return report


def print_weekly_report(r: dict) -> None:
    print("=" * 68)
    print(f"Weekly Production Rollout Report")
    print(f"Period: {r['period_start']} → {r['period_end']}")
    print("=" * 68)

    # Realized PnL
    rp = r["realized_pnl"]
    print(f"\n  Realized PnL:")
    print(f"    Total:         {_pnl_str(rp['total'])}")
    print(f"    Trades closed: {rp['trade_count']}  "
          f"(W={rp['wins']} L={rp['losses']} WR={rp['win_rate_pct']:.1f}%)")
    if rp["by_day"]:
        print(f"    Daily breakdown:")
        for d, pnl in sorted(rp["by_day"].items()):
            sign = "+" if pnl >= 0 else ""
            print(f"      {d}  {sign}¥{pnl:>10,.0f}")

    # Unrealized
    up = r["unrealized"]
    if "total" in up:
        print(f"\n  Unrealized PnL:  {_pnl_str(up['total'])}")
        print(f"  Current Equity:  ¥{up.get('last_equity', 0):>12,.0f}")
        if up.get("positions"):
            print(f"  Open Positions:")
            for p in up["positions"]:
                ud = p["entry_date"] or "—"
                print(f"    {p['symbol']:<10} qty={p['qty']:>5}  "
                      f"unreal={_pnl_str(p['unreal_pnl'])}  since={ud}")

    # Addon
    ad = r["addon"]
    print(f"\n  Addon (EQ_SCALE):")
    print(f"    Executed:      {ad['executed']}")
    print(f"    Rejected:      {ad['rejected']}")
    if ad["reject_reasons"]:
        for reason, cnt in ad["reject_reasons"].items():
            print(f"      {reason}: {cnt}")
    print(f"    Addon PnL:     {_pnl_str(ad['addon_pnl'])}")
    print(f"    Addon WinRate: {ad['addon_win_rate_pct']:.1f}%")

    # ATR Extension
    ae = r["atr_ext"]
    print(f"\n  ATR Extension:")
    print(f"    New deferrals: {ae['new_deferrals']}")
    print(f"    Forced exits:  {ae['forced_exits']}")
    if ae["avg_days_added"] is not None:
        print(f"    Avg days added: {ae['avg_days_added']:.1f}d")
    print(f"    Extension PnL: {_pnl_str(ae['ext_pnl'])}")

    # VOL_ADJ
    va = r["vol_adj"]
    print(f"\n  VOL_ADJ:")
    print(f"    Days at max_pos=4:  {va['days_at_max4']}/{va['days_total']}  ({va['calm_pct']:.0f}% calm)")

    # Turnover
    tur = r["turnover"]
    print(f"\n  Turnover:")
    print(f"    Gross:         ¥{tur['gross_turnover']:>12,.0f}  ({tur['turnover_vs_equity']:.1f}% of equity)")
    print(f"    Transactions:  BUY={tur['buy_count']}  SELL={tur['sell_count']}")

    # Holding days
    hd = r["holding_days"]
    avg = hd["avg_hold_days"]
    print(f"\n  Holding Days:")
    print(f"    Closed positions: {hd['closed_positions']}")
    print(f"    Average: {avg:.1f}d" if avg else f"    Average: —")
    if hd["distribution"]:
        d = hd["distribution"]
        print(f"    Distribution:  ≤5d={d['<=5d']}  6-20d={d['6-20d']}  21-55d={d['21-55d']}  >55d={d['>55d']}")

    # Violations / flags
    viol  = r["violations"]
    flags = r["rollback_flags"]
    print(f"\n  " + "─" * 64)
    if not viol and not flags:
        print(f"  Rollback Triggers: NONE — clean week")
    else:
        if viol:
            print(f"  ⛔ VIOLATIONS DETECTED ({len(viol)}):")
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
    parser.add_argument("--weeks", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    until = date.today()
    since = until - timedelta(weeks=args.weeks)
    r     = build_weekly_report(since, until)

    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    else:
        print_weekly_report(r)

    # Persist
    WEEKLY_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(WEEKLY_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()
