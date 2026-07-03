"""
post_sync_sizing_audit.py — Post-Sync Capital Deployment Audit

BEFORE : bridge.capital = 3,000,000 (config固定)
AFTER  : bridge.capital = actual_equity × 0.90 (実資産同期後・定常状態)

全57銘柄について BEFORE/AFTER のサイジングを比較。
"""
from __future__ import annotations
import sys, json, csv
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

ROOT      = Path(__file__).resolve().parents[1]
OHLCV_DIR = ROOT / "cache" / "ohlcv"

# ── 定数 ─────────────────────────────────────────────────────────────────────
CAPITAL_CONFIG    = 3_000_000      # BEFORE
CASH_BUFFER       = 0.10
MAX_SINGLE_WEIGHT = 0.25
RISK_PCT          = 0.0125
LOT               = 100
LEADER_RSR        = 85.0
LEADER_SLOT       = 0.35           # リーダー銘柄の slot weight

ATR_WINDOW        = 20
FOCUS_SYMS        = ["6762.T", "5301.T", "7182.T", "8750.T"]

# ── データロード ──────────────────────────────────────────────────────────────
def load_universe() -> dict[str, str]:
    from src.paths import LIVE_UNIVERSE_FILE
    d = json.loads(Path(LIVE_UNIVERSE_FILE).read_text(encoding="utf-8"))
    return d.get("symbols", d)

def latest_price_atr(sym: str) -> tuple[float | None, float | None]:
    p = OHLCV_DIR / f"{sym}.parquet"
    if not p.exists():
        return None, None
    try:
        df = pd.read_parquet(p)
        cl = df["Close"].dropna()
        if cl.empty:
            return None, None
        price = float(cl.iloc[-1])
        hi = df["High"].dropna(); lo = df["Low"].dropna()
        tr = pd.concat([
            hi - lo,
            (hi - cl.shift()).abs(),
            (lo - cl.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_s = tr.rolling(ATR_WINDOW).mean().dropna()
        atr = float(atr_s.iloc[-1]) if not atr_s.empty else None
        return price, atr
    except Exception:
        return None, None

def get_rsr(sym: str) -> float | None:
    """直近 DRY ランの RSR スコア (top20 のみ)。"""
    try:
        p = ROOT / "logs" / "diagnostics" / "rsr_distribution.jsonl"
        lines = p.read_text(encoding="utf-8").strip().split("\n")
        last  = json.loads(lines[-1])
        d     = {item["symbol"]: item["rsr"] for item in last.get("top20", [])}
        return d.get(sym)
    except Exception:
        return None

# ── 資本階層計算 ──────────────────────────────────────────────────────────────
def capital_chain(actual_equity: float) -> dict:
    eff    = actual_equity                          # 定常状態: rate-limit 収束後
    dep    = eff * (1 - CASH_BUFFER)               # deployable (10% buffer)
    risk_adj = dep                                  # scalars=1.0
    return {
        "actual_equity"      : actual_equity,
        "effective_capital"  : eff,
        "deployable_capital" : dep,
        "risk_adjusted"      : risk_adj,
    }

# ── サイジング計算 ─────────────────────────────────────────────────────────────
def compute_sizing(bridge_cap: float, price: float, atr: float | None,
                   rsr: float | None) -> dict:
    is_leader    = (rsr is not None and rsr >= LEADER_RSR)
    slot_weight  = LEADER_SLOT if is_leader else MAX_SINGLE_WEIGHT
    pos_budget   = bridge_cap * slot_weight
    risk_per_trade = bridge_cap * RISK_PCT
    lot_cost     = price * LOT

    # qty_cap
    qty_cap = int(pos_budget // lot_cost) * LOT

    # qty_risk
    if atr and atr > 0:
        qty_raw  = int(risk_per_trade / atr)
        qty_risk = (qty_raw // LOT) * LOT
        qty_risk = max(LOT, qty_risk)
    else:
        qty_risk = None

    # final
    if qty_cap <= 0:
        qty_final   = 0
        binding     = "price_too_high"
    elif qty_risk is None:
        qty_final   = qty_cap
        binding     = "atr_unavailable"
    else:
        qty_final   = min(qty_risk, qty_cap)
        if   qty_risk < qty_cap: binding = "qty_risk"
        elif qty_cap  < qty_risk: binding = "qty_cap"
        else:                    binding = "equal"

    est_val = qty_final * price

    return {
        "pos_budget"   : pos_budget,
        "qty_cap"      : qty_cap,
        "qty_risk"     : qty_risk if qty_risk else 0,
        "qty_final"    : qty_final,
        "est_value"    : est_val,
        "binding"      : binding,
        "is_leader"    : is_leader,
        "slot_weight"  : slot_weight,
    }

# ── メイン ────────────────────────────────────────────────────────────────────
def main():
    # ── 1. 実資産取得 ────────────────────────────────────────────────────────
    ps    = json.loads((ROOT / "runtime" / "portfolio_state.json").read_text(encoding="utf-8"))
    cash  = float(ps["available_cash"])
    held  = ps.get("position_qtys", {})
    mktval = 0.0
    for sym, qty in held.items():
        pr, _ = latest_price_atr(sym)
        if pr:
            mktval += qty * pr

    ACTUAL_EQUITY = cash + mktval
    chain_before  = capital_chain(float(CAPITAL_CONFIG))
    chain_after   = capital_chain(ACTUAL_EQUITY)

    BRIDGE_BEFORE = chain_before["risk_adjusted"]
    BRIDGE_AFTER  = chain_after["risk_adjusted"]

    SEP = "=" * 76
    print(SEP)
    print("  POST-SYNC CAPITAL DEPLOYMENT AUDIT")
    print(SEP)
    print(f"  BEFORE: config capital    = ¥{CAPITAL_CONFIG:>12,.0f}")
    print(f"  AFTER : actual_equity     = ¥{ACTUAL_EQUITY:>12,.0f}  (cash + position mktval)")
    print()
    print(f"  {'レイヤ':<28}  {'BEFORE':>14}  {'AFTER':>14}  {'Δ':>14}")
    print("  " + "-" * 72)
    layers = [
        ("actual_equity",       float(CAPITAL_CONFIG), ACTUAL_EQUITY),
        ("effective_capital",   chain_before["effective_capital"], chain_after["effective_capital"]),
        ("deployable_capital",  chain_before["deployable_capital"], chain_after["deployable_capital"]),
        ("risk_adjusted(bridge)", BRIDGE_BEFORE, BRIDGE_AFTER),
        ("position_budget(25%)", BRIDGE_BEFORE*MAX_SINGLE_WEIGHT, BRIDGE_AFTER*MAX_SINGLE_WEIGHT),
        ("risk_per_trade(1%)",  BRIDGE_BEFORE*RISK_PCT, BRIDGE_AFTER*RISK_PCT),
    ]
    for name, bv, av in layers:
        delta = av - bv
        print(f"  {name:<28}  ¥{bv:>13,.0f}  ¥{av:>13,.0f}  ¥{delta:>+13,.0f}")

    # ── 2. 全57銘柄サイジング ────────────────────────────────────────────────
    universe = load_universe()
    rsr_map  = {item["symbol"]: item["rsr"]
                for item in json.loads(
                    (ROOT / "logs" / "diagnostics" / "rsr_distribution.jsonl")
                    .read_text(encoding="utf-8").strip().split("\n")[-1]
                ).get("top20", [])}

    rows = []
    for sym, sector in sorted(universe.items()):
        price, atr = latest_price_atr(sym)
        rsr        = rsr_map.get(sym)
        s_b = compute_sizing(BRIDGE_BEFORE, price, atr, rsr) if price else None
        s_a = compute_sizing(BRIDGE_AFTER,  price, atr, rsr) if price else None
        row = {
            "symbol"                : sym,
            "sector"                : sector,
            "price"                 : f"{price:.1f}" if price else "",
            "ATR20"                 : f"{atr:.1f}"   if atr   else "",
            "RSR"                   : f"{rsr:.1f}"   if rsr   else "",
            "actual_equity"         : f"{ACTUAL_EQUITY:.0f}",
            "effective_capital"     : f"{chain_after['effective_capital']:.0f}",
            "deployable_capital"    : f"{chain_after['deployable_capital']:.0f}",
            # BEFORE
            "B_position_budget"     : f"{s_b['pos_budget']:.0f}"  if s_b else "",
            "B_qty_cap"             : str(s_b["qty_cap"])          if s_b else "",
            "B_qty_risk"            : str(s_b["qty_risk"])         if s_b else "",
            "B_qty_final"           : str(s_b["qty_final"])        if s_b else "",
            "B_est_value"           : f"{s_b['est_value']:.0f}"   if s_b else "",
            "B_binding"             : s_b["binding"]               if s_b else "no_price",
            # AFTER
            "A_position_budget"     : f"{s_a['pos_budget']:.0f}"  if s_a else "",
            "A_qty_cap"             : str(s_a["qty_cap"])          if s_a else "",
            "A_qty_risk"            : str(s_a["qty_risk"])         if s_a else "",
            "A_qty_final"           : str(s_a["qty_final"])        if s_a else "",
            "A_est_value"           : f"{s_a['est_value']:.0f}"   if s_a else "",
            "A_binding"             : s_a["binding"]               if s_a else "no_price",
            # DELTA
            "delta_qty"             : str((s_a["qty_final"] - s_b["qty_final"]) if (s_a and s_b) else ""),
            "delta_value"           : f"{(s_a['est_value'] - s_b['est_value']):.0f}" if (s_a and s_b) else "",
        }
        rows.append((sym, sector, price, atr, rsr, s_b, s_a))

        # Print console line
        bq = s_b["qty_final"] if s_b else 0
        aq = s_a["qty_final"] if s_a else 0
        dq = aq - bq
        flag = f" Δ{dq:+d}" if dq != 0 else ""
        ab  = s_a["binding"] if s_a else "no_price"

    # ── 3. 追加分析1: 重点銘柄比較 ──────────────────────────────────────────
    print()
    print(SEP)
    print("  追加分析1: 重点銘柄 BEFORE vs AFTER")
    print(SEP)
    print(f"  {'Symbol':<10} {'Price':>8} {'ATR20':>7} {'RSR':>6}"
          f"  {'B_qfin':>7} {'A_qfin':>7} {'ΔQty':>7}"
          f"  {'B_value':>10} {'A_value':>10} {'ΔValue':>11}"
          f"  {'B_bind':<12} {'A_bind'}")
    print("  " + "-" * 102)
    for sym in FOCUS_SYMS + [s for s in ["6506.T", "6981.T"] if s in held]:
        match = [(sym2, sec, p, a, r, sb, sa)
                 for sym2, sec, p, a, r, sb, sa in rows if sym2 == sym]
        if not match:
            print(f"  {sym:<10} not in universe")
            continue
        _, sec, p, a, r, sb, sa = match[0]
        bq = sb["qty_final"] if sb else 0
        aq = sa["qty_final"] if sa else 0
        bv = sb["est_value"] if sb else 0
        av = sa["est_value"] if sa else 0
        bb = sb["binding"]   if sb else "no_price"
        ab_ = sa["binding"]  if sa else "no_price"
        rsr_s = f"{r:.1f}" if r else "-"
        p_s   = f"{p:,.1f}" if p else "-"
        a_s   = f"{a:,.1f}" if a else "-"
        print(
            f"  {sym:<10} {p_s:>8} {a_s:>7} {rsr_s:>6}"
            f"  {bq:>7} {aq:>7} {aq-bq:>+7}"
            f"  {bv:>10,.0f} {av:>10,.0f} {av-bv:>+11,.0f}"
            f"  {bb:<12} {ab_}"
        )

    # ── 4. 追加分析2: 全57銘柄集計 ──────────────────────────────────────────
    qty_up    = sum(1 for *_, sb, sa in rows if sb and sa and sa["qty_final"] > sb["qty_final"])
    qty_same  = sum(1 for *_, sb, sa in rows if sb and sa and sa["qty_final"] == sb["qty_final"])
    qty_down  = sum(1 for *_, sb, sa in rows if sb and sa and sa["qty_final"] < sb["qty_final"])
    no_price  = sum(1 for *_, sb, sa in rows if not sb or not sa)
    pt_high_b = sum(1 for *_, sb, _  in rows if sb and sb["binding"] == "price_too_high")
    pt_high_a = sum(1 for *_, _,  sa in rows if sa and sa["binding"] == "price_too_high")
    bc_risk_b = sum(1 for *_, sb, _  in rows if sb and sb["binding"] == "qty_risk")
    bc_cap_b  = sum(1 for *_, sb, _  in rows if sb and sb["binding"] == "qty_cap")
    bc_risk_a = sum(1 for *_, _,  sa in rows if sa and sa["binding"] == "qty_risk")
    bc_cap_a  = sum(1 for *_, _,  sa in rows if sa and sa["binding"] == "qty_cap")
    n_priced  = sum(1 for *_, sb, sa in rows if sb and sa)

    print()
    print(SEP)
    print("  追加分析2: 全57銘柄 BEFORE vs AFTER 集計")
    print(SEP)
    print(f"  {'指標':<30}  {'BEFORE':>10}  {'AFTER':>10}")
    print("  " + "-" * 54)
    stats = [
        ("qty増加銘柄",    "-",      str(qty_up)),
        ("qty不変銘柄",    "-",      str(qty_same)),
        ("qty減少銘柄",    "-",      str(qty_down)),
        ("price_too_high", str(pt_high_b),  str(pt_high_a)),
        ("qty_risk binding", f"{bc_risk_b}/{n_priced}({bc_risk_b/max(n_priced,1):.0%})",
                             f"{bc_risk_a}/{n_priced}({bc_risk_a/max(n_priced,1):.0%})"),
        ("qty_cap  binding", f"{bc_cap_b}/{n_priced}({bc_cap_b/max(n_priced,1):.0%})",
                             f"{bc_cap_a}/{n_priced}({bc_cap_a/max(n_priced,1):.0%})"),
    ]
    for name, bv, av in stats:
        print(f"  {name:<30}  {bv:>10}  {av:>10}")

    # ── 5. 追加分析3: 投資率改善効果 ─────────────────────────────────────────
    print()
    print(SEP)
    print("  追加分析3: 投資率改善効果")
    print(SEP)
    for label, bridge_cap, chain in [("BEFORE", BRIDGE_BEFORE, chain_before),
                                      ("AFTER",  BRIDGE_AFTER,  chain_after)]:
        valid = [(p, s_a if label == "AFTER" else s_b)
                 for _, _, p, _, _, s_b, s_a in rows
                 if p and (s_a if label == "AFTER" else s_b)]
        bindings = [sv["binding"] for _, sv in valid if sv]
        avg_qty  = np.mean([sv["qty_final"] for _, sv in valid if sv]) if valid else 0
        avg_pos_val = np.mean([sv["est_value"] for _, sv in valid if sv and sv["qty_final"] > 0]) if valid else 0
        pos_bud  = bridge_cap * MAX_SINGLE_WEIGHT
        theo_max = min(chain["deployable_capital"],
                       3 * pos_bud)   # 3 slots × per-position budget
        util_cont = theo_max / chain["actual_equity"]
        # lot-rounded max (assume avg price across deployable set)
        prices_ok = [p for _, _, p, _, _, s_b, s_a in rows
                     if p and (s_a if label == "AFTER" else s_b)
                     and (s_a if label == "AFTER" else s_b)["qty_final"] > 0]
        avg_price  = np.mean(prices_ok) if prices_ok else 5000
        lots_per_p = int(pos_bud // (avg_price * LOT))
        max_invest_lot = 3 * lots_per_p * avg_price * LOT
        util_lot   = max_invest_lot / chain["actual_equity"]

        print(f"  [{label}] bridge.capital = ¥{bridge_cap:,.0f}")
        print(f"    position_budget       = ¥{pos_bud:>12,.0f}  ({MAX_SINGLE_WEIGHT:.0%})")
        print(f"    avg position value    = ¥{avg_pos_val:>12,.0f}  (qty>0 銘柄平均)")
        print(f"    theoretical_max_invest= ¥{theo_max:>12,.0f}  (連続値 / lot丸め前)")
        print(f"    expected_utilization  =  {util_cont:>11.2%}  (連続値)")
        print(f"    lot-rounded max_invest= ¥{max_invest_lot:>12,.0f}  (avg_price=¥{avg_price:,.0f})")
        print(f"    lot-rounded utilization= {util_lot:>10.2%}")
        print()

    # ── 6. 追加分析4: max_positions 別 理論投資率 ─────────────────────────────
    print(SEP)
    print("  追加分析4: max_positions 別 理論投資率 (AFTER 実資産ベース)")
    print(SEP)
    pos_bud_a  = BRIDGE_AFTER * MAX_SINGLE_WEIGHT
    dep_a      = chain_after["deployable_capital"]
    act_a      = chain_after["actual_equity"]
    valid_rows = [(p, s_a) for _, _, p, _, _, s_b, s_a in rows
                  if p and s_a and s_a["qty_cap"] > 0]
    avg_pos_budget_used = np.mean([min(p * s_a["qty_final"], pos_bud_a)
                                   for p, s_a in valid_rows if s_a["qty_final"] > 0]) if valid_rows else pos_bud_a

    print(f"  {'max_pos':>8}  {'expected_util':>14}  {'idle_capital':>13}  {'add_capacity':>14}  {'lot_at_avg_price':>16}")
    print("  " + "-" * 74)
    for mp in [3, 4, 5]:
        theo   = min(dep_a, mp * pos_bud_a)
        util   = theo / act_a
        idle   = act_a - theo - act_a * CASH_BUFFER   # rough idle after deployment
        add_c  = theo - min(dep_a, 3 * pos_bud_a)     # vs current 3-slot max
        avg_p  = np.mean([p for p, _ in valid_rows]) if valid_rows else 5000
        lots_p = int(pos_bud_a // (avg_p * LOT))
        inv_lot = mp * lots_p * avg_p * LOT
        util_l = inv_lot / act_a
        print(
            f"  {mp:>8}  {util:>14.2%}  ¥{idle:>12,.0f}  ¥{add_c:>13,.0f}  "
            f"{mp}×{lots_p}lots=¥{inv_lot:>9,.0f}({util_l:.2%})"
        )

    # ── 7. 最終ボトルネック特定 ───────────────────────────────────────────────
    print()
    print(SEP)
    print("  最終ボトルネック分析")
    print(SEP)
    # 現在の状態を定量化
    # 実際投資中 (held positions)
    held_val = sum(
        s_a["est_value"]
        for sym, _, p, _, _, s_b, s_a in rows
        if sym in held and s_a and s_a["qty_final"] > 0
    )
    # 空きスロット
    n_held      = len(held)
    open_slots  = 3 - n_held
    slot_idle   = open_slots * pos_bud_a
    # 10% buffer ロック
    cash_locked = act_a * CASH_BUFFER
    # weight 制約 (33.3% = equal weight 上限 − 25% = weight cap による追加圧縮)
    weight_loss = (1/3 - MAX_SINGLE_WEIGHT) * dep_a * 3 if 1/3 > MAX_SINGLE_WEIGHT else 0
    # risk 制約 (qty_risk binding 率 × 追加 lost capital)
    risk_binding_loss = sum(
        max(0, s_a["qty_cap"] - s_a["qty_final"]) * p
        for _, _, p, _, _, _, s_a in rows
        if s_a and s_a["binding"] == "qty_risk" and p
    ) / max(n_priced, 1) * 3  # 3 slots

    btl_scores = {
        "1.資本不足"      : max(0.0, (CAPITAL_CONFIG - ACTUAL_EQUITY) / CAPITAL_CONFIG),
        "2.ポジション数不足": slot_idle / act_a,
        "3.weight制約"    : weight_loss / act_a if weight_loss > 0 else 0.01,
        "4.risk制約"      : risk_binding_loss / act_a,
    }
    worst = max(btl_scores, key=btl_scores.get)

    print(f"  bridge.capital (AFTER)  = ¥{BRIDGE_AFTER:>12,.0f}")
    print(f"  現在の投資額            = ¥{held_val:>12,.0f}  (保有 {n_held} ポジション)")
    print(f"  投資率                  =  {held_val/act_a:>11.2%}")
    print()
    print(f"  ボトルネック定量評価 (AFTER ベース):")
    print(f"  {'制約':<20}  {'アイドル額':>14}  {'投資率圧迫':>12}  {'優先度':>8}")
    print("  " + "-" * 62)
    for name, score in sorted(btl_scores.items(), key=lambda x: -x[1]):
        idle_est = score * act_a
        mark = " ← 最大" if name == worst else ""
        print(f"  {name:<20}  ¥{idle_est:>13,.0f}  {score:>12.2%}{mark}")

    print()
    print(f"  【最大ボトルネック】: {worst}")
    btl_desc = {
        "1.資本不足"      : "actual_equity が config より少ない。実資産同期で解決済み。",
        "2.ポジション数不足": "空きスロットに候補がない（CB_ACTIVE）またはスロット < 3。\n"
                              "    CB解除 (2026-06-30) で {:.0f}%pt 投資率回復見込み。".format(slot_idle/act_a*100),
        "3.weight制約"    : "max_single_weight=25% が qty_cap を制約。33%+ への引き上げ要WF検証。",
        "4.risk制約"      : "risk_pct=1%/ATR が qty_risk を制約。1.25% で改善可能（WF PASS率60%）。",
    }
    print(f"  対策: {btl_desc.get(worst, '')}")

    # ── CSV 出力 ──────────────────────────────────────────────────────────────
    out_path = ROOT / "logs" / "post_sync_sizing_audit.csv"
    out_path.parent.mkdir(exist_ok=True)
    fields = [
        "symbol", "sector", "price", "ATR20", "RSR",
        "actual_equity", "effective_capital", "deployable_capital",
        "B_position_budget", "B_qty_cap", "B_qty_risk", "B_qty_final",
        "B_est_value", "B_binding",
        "A_position_budget", "A_qty_cap", "A_qty_risk", "A_qty_final",
        "A_est_value", "A_binding",
        "delta_qty", "delta_value",
    ]
    csv_rows = []
    for sym, sec, p, a, r, s_b, s_a in rows:
        csv_rows.append({
            "symbol": sym, "sector": sec,
            "price" : f"{p:.1f}" if p else "",
            "ATR20" : f"{a:.1f}" if a else "",
            "RSR"   : f"{r:.1f}" if r else "",
            "actual_equity"      : f"{ACTUAL_EQUITY:.0f}",
            "effective_capital"  : f"{chain_after['effective_capital']:.0f}",
            "deployable_capital" : f"{chain_after['deployable_capital']:.0f}",
            "B_position_budget"  : f"{s_b['pos_budget']:.0f}"  if s_b else "",
            "B_qty_cap"          : str(s_b["qty_cap"])          if s_b else "",
            "B_qty_risk"         : str(s_b["qty_risk"])         if s_b else "",
            "B_qty_final"        : str(s_b["qty_final"])        if s_b else "",
            "B_est_value"        : f"{s_b['est_value']:.0f}"   if s_b else "",
            "B_binding"          : s_b["binding"]               if s_b else "no_price",
            "A_position_budget"  : f"{s_a['pos_budget']:.0f}"  if s_a else "",
            "A_qty_cap"          : str(s_a["qty_cap"])          if s_a else "",
            "A_qty_risk"         : str(s_a["qty_risk"])         if s_a else "",
            "A_qty_final"        : str(s_a["qty_final"])        if s_a else "",
            "A_est_value"        : f"{s_a['est_value']:.0f}"   if s_a else "",
            "A_binding"          : s_a["binding"]               if s_a else "no_price",
            "delta_qty"  : str(s_a["qty_final"] - s_b["qty_final"]) if (s_a and s_b) else "",
            "delta_value": f"{s_a['est_value'] - s_b['est_value']:.0f}" if (s_a and s_b) else "",
        })
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(csv_rows)
    print(f"\n  CSV: {out_path}  ({len(csv_rows)} rows)")


if __name__ == "__main__":
    main()
