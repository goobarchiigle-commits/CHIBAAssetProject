"""
src/backtest/leverage_opportunity_audit.py

レバレッジ別機会損失分析 (事実集計のみ / 戦略変更なし)

目的:
  「資金制約で採用できなかったシグナル」の期待値をレバレッジ別に定量化する。
  追加シグナル数の確認ではなく、回収できる期待値の確認。

ケース:
  A: 1.0x (現行, extra_capital=0)
  B: 1.1x  extra_capital = 3M × 10% = ¥300,000
  C: 1.2x  extra_capital = 3M × 20% = ¥600,000
  D: 1.3x  extra_capital = 3M × 30% = ¥900,000

機会損失の定義:
  バックテスト中、max_positions=3 が既に満杯のためにブロックされた BUY シグナル。
  これらのシグナルに必要な 1lot 資金 (buy_px × 100) と extra_capital を比較し、
  extra_capital >= lot_cost なら「レバで捕捉可能」とみなす。
  各日につき 1 シグナルのみ追加可能 (最高 RSR のものを優先)。

出力:
  docs/research/leverage_opportunity_audit.md

実行:
    cd C:/ai-trading
    python src/backtest/leverage_opportunity_audit.py
"""

from __future__ import annotations

import sys, time
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from pathlib import Path

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, run_period

# ─────────────────────────────────────────────────────────────────────────────
CAPITAL       = 3_000_000
LOT           = 100
AUDIT_START   = "2018-01-01"
AUDIT_END     = "2025-12-31"
YEARS         = [2020, 2021, 2022, 2023, 2024, 2025]

LEVERAGE_CASES: dict[str, float] = {
    "1.0x": 0,
    "1.1x": CAPITAL * 0.10,   # 300,000
    "1.2x": CAPITAL * 0.20,   # 600,000
    "1.3x": CAPITAL * 0.30,   # 900,000
}

OUTPUT_DIR = Path("docs/research")
OUTPUT_FILE = OUTPUT_DIR / "leverage_opportunity_audit.md"

# ─────────────────────────────────────────────────────────────────────────────


def fwd_return(sym: str, signal_date_str: str, entry_px: float,
               horizon: int, close_by_sym: dict) -> float | None:
    """
    entry_px: 翌日 open (シグナル日の next_i open = 実執行価格)
    horizon: 営業日数 (20 or 60)
    戻り値: close[entry_day + horizon bdays] / entry_px - 1  (None=データ不足)
    """
    s = close_by_sym.get(sym)
    if s is None or entry_px <= 0:
        return None
    ts = pd.Timestamp(signal_date_str)
    future = s[s.index > ts]          # シグナル日より後のClose (執行日以降)
    if len(future) < horizon:
        return None
    exit_px = float(future.iloc[horizon - 1])
    return exit_px / entry_px - 1


def _stats(records: list[dict]) -> dict:
    if not records:
        return dict(n=0, avg_rsr=0.0, avg_lot=0.0, avg_20d=None,
                    avg_60d=None, wr_20d=0.0, pf_20d=0.0,
                    wr_60d=0.0, pf_60d=0.0)
    rsrs = [r["rsr"]      for r in records]
    lots = [r["lot_cost"] for r in records]
    r20s = [r["fwd_20d"]  for r in records if r["fwd_20d"] is not None]
    r60s = [r["fwd_60d"]  for r in records if r["fwd_60d"] is not None]

    def _pf_wr(rs):
        if not rs:
            return 0.0, 0.0
        wins = [r for r in rs if r > 0]
        loss = [r for r in rs if r <= 0]
        wr   = len(wins) / len(rs) * 100
        gp   = sum(wins)
        gl   = abs(sum(loss))
        pf   = round(gp / max(gl, 1e-9), 2)
        return round(wr, 1), pf

    wr20, pf20 = _pf_wr(r20s)
    wr60, pf60 = _pf_wr(r60s)

    return dict(
        n        = len(records),
        avg_rsr  = round(sum(rsrs) / len(rsrs), 1),
        avg_lot  = round(sum(lots) / len(lots) / 1000, 0),  # ¥K
        avg_20d  = round(sum(r20s) / len(r20s) * 100, 2) if r20s else None,
        avg_60d  = round(sum(r60s) / len(r60s) * 100, 2) if r60s else None,
        wr_20d   = wr20,
        pf_20d   = pf20,
        wr_60d   = wr60,
        pf_60d   = pf60,
        n_20d    = len(r20s),
        n_60d    = len(r60s),
    )


def _pct(a: int, b: int) -> str:
    return f"{a/b*100:.1f}%" if b > 0 else "—"


def _fmt(v, pct=True) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v:+.2f}%"
    return f"{v:.2f}"


# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    cfg = load_strategy_config()

    print("=" * 72)
    print("  レバレッジ別機会損失分析 (事実集計)")
    print(f"  期間: {AUDIT_START} 〜 {AUDIT_END}")
    print(f"  CAPITAL={CAPITAL:,}  Extra: 1.1x=¥300K / 1.2x=¥600K / 1.3x=¥900K")
    print("=" * 72)

    print("\n[1/3] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    close_by_sym = {s: d["df"]["Close"].sort_index() for s, d in universe_raw.items()}

    print("\n[2/3] ベースラインバックテスト実行 (missed signal 収集)...")
    t0 = time.time()
    m = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=AUDIT_START, end=AUDIT_END,
        pattern="A",
        return_trades=True,
        return_missed_details=True,
    )
    elapsed = time.time() - t0
    baseline_trades = m.get("_trades_raw", [])
    all_missed: list[dict] = m.get("_missed_details", [])
    baseline_n_sells = sum(1 for t in baseline_trades if t["side"] == "SELL")
    print(f"  完了 ({elapsed:.1f}s) / baseline SELL={baseline_n_sells} "
          f"/ missed signals={len(all_missed)}")

    print("\n[3/3] forward return 計算 + レバ別シミュレーション...")

    # ── forward return 計算 ────────────────────────────────────────────────
    enriched: list[dict] = []
    for sig in all_missed:
        r20 = fwd_return(sig["sym"], sig["date"], sig["buy_px"], 20, close_by_sym)
        r60 = fwd_return(sig["sym"], sig["date"], sig["buy_px"], 60, close_by_sym)
        yr  = int(sig["date"][:4])
        enriched.append({**sig, "fwd_20d": r20, "fwd_60d": r60, "year": yr})

    # ── レバ別: 各日 best-fitting 1 シグナルを選択 ────────────────────────
    by_date: dict[str, list[dict]] = {}
    for r in enriched:
        by_date.setdefault(r["date"], []).append(r)

    # レバ別捕捉シグナル (1日1シグナル max)
    captured_by_lev: dict[str, list[dict]] = {k: [] for k in LEVERAGE_CASES}
    for date_str, sigs in by_date.items():
        sigs_sorted = sorted(sigs, key=lambda x: x["rank_blocked"])
        for label, extra_cap in LEVERAGE_CASES.items():
            if extra_cap == 0:
                continue  # 1.0x は追加なし
            for sig in sigs_sorted:
                if sig["lot_cost"] <= extra_cap:
                    captured_by_lev[label].append(sig)
                    break  # 1日 1追加ポジション

    total_blocked_days = len(by_date)
    total_blocked_sigs = len(enriched)
    print(f"  blocked days={total_blocked_days}  "
          f"total blocked signals={total_blocked_sigs}")
    for lbl, cap in captured_by_lev.items():
        print(f"  {lbl}: captured={len(cap)}")

    # ── レポート生成 ──────────────────────────────────────────────────────
    L = []; w = L.append

    w("# レバレッジ別機会損失分析")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  /  期間: {AUDIT_START}〜{AUDIT_END}")
    w("\n**前提**:")
    w("- 戦略・パラメータ変更なし (事実集計のみ)")
    w("- CAPITAL=¥3,000,000  max_positions=3 (PARAMS_LOCKED)")
    w(f"- 追加可能資金: 1.1x=¥300K / 1.2x=¥600K / 1.3x=¥900K")
    w("- 各日に追加取得可能なシグナルは最大 1 本 (最高 RSR の lot_cost 適合シグナルを選択)")
    w(f"- blocked 日数={total_blocked_days}  blocked シグナル総数={total_blocked_sigs}\n")

    # ── 1. ベースライン統計 ───────────────────────────────────────────────
    w("---\n## 1. ベースライン (1.0x) 統計\n")
    w("| 指標 | 値 |")
    w("|---|---|")
    w(f"| CAGR | {m.get('cagr',0):+.2f}% |")
    w(f"| Sharpe | {m.get('sharpe',0):.3f} |")
    w(f"| MaxDD | {m.get('max_dd',0):.2f}% |")
    w(f"| Calmar | {m.get('calmar',0):.3f} |")
    w(f"| PF | {m.get('profit_factor',0):.2f} |")
    w(f"| N_trades | {baseline_n_sells} |")
    w(f"| AvgHold | {m.get('avg_hold_days',0):.1f}d |")
    w(f"| AvgExp | {m.get('avg_exposure',0):.1f}% |")
    w(f"| Missed signals (全期間) | {total_blocked_sigs} |")
    w(f"| Days portfolio full | {total_blocked_days} |")

    exit_rsns = m.get("exit_reasons", {})
    w("\n**Exit reason 内訳**:")
    for k, v in sorted(exit_rsns.items(), key=lambda x: -x[1]):
        w(f"- {k}: {v}")

    # ── 2. レバ別集計 ──────────────────────────────────────────────────────
    w("\n---\n## 2. レバレッジ別 追加シグナル統計\n")
    w("| Leverage | Extra Cap | Add Signals | Capture% | Avg RSR | "
      "Avg Lot(¥K) | Avg +20d | Avg +60d | WR(20d) | PF(20d) |")
    w("|---|---|---|---|---|---|---|---|---|---|")

    for lbl, extra_cap in LEVERAGE_CASES.items():
        if extra_cap == 0:
            w(f"| {lbl} | ¥0 | 0 (baseline) | 0% | — | — | — | — | — | — |")
            continue
        sigs = captured_by_lev[lbl]
        st   = _stats(sigs)
        cap_pct = _pct(len(sigs), total_blocked_days)
        w(f"| {lbl} | ¥{int(extra_cap):,} "
          f"| {st['n']} | {cap_pct} "
          f"| {st['avg_rsr']:.1f} | ¥{st['avg_lot']:.0f}K "
          f"| {_fmt(st['avg_20d'])} | {_fmt(st['avg_60d'])} "
          f"| {st['wr_20d']:.1f}% | {st['pf_20d']:.2f} |")

    # ── 3. 逓減分析 (marginal) ──────────────────────────────────────────
    w("\n---\n## 3. レバ段階別 限界利得 (Marginal Analysis)\n")
    w("| レバ幅 | 追加 Lot 帯 | 純増シグナル数 | Avg +20d | Avg +60d | WR(20d) | PF(20d) |")
    w("|---|---|---|---|---|---|---|")

    steps = [
        ("1.0x→1.1x", "—",            "1.1x",  "—"),
        ("1.1x→1.2x", "1.1x", "1.2x", "¥300K〜¥600K"),
        ("1.2x→1.3x", "1.2x", "1.3x", "¥600K〜¥900K"),
    ]

    dates_captured: dict[str, set] = {}
    for lbl in LEVERAGE_CASES:
        dates_captured[lbl] = {(s["date"], s["sym"]) for s in captured_by_lev.get(lbl, [])}

    for step_label, prev_lbl, curr_lbl, lot_range in steps:
        if prev_lbl == "—":
            # 1.0x→1.1x: 全て 1.1x の追加分
            curr_sigs = captured_by_lev.get("1.1x", [])
            marginal  = curr_sigs
        else:
            prev_keys = dates_captured.get(prev_lbl, set())
            marginal  = [s for s in captured_by_lev.get(curr_lbl, [])
                         if (s["date"], s["sym"]) not in prev_keys]
        st = _stats(marginal)
        w(f"| {step_label} | {lot_range} "
          f"| {st['n']} "
          f"| {_fmt(st['avg_20d'])} | {_fmt(st['avg_60d'])} "
          f"| {st['wr_20d']:.1f}% | {st['pf_20d']:.2f} |")

    # ── 4. 年別 × レバ別 ────────────────────────────────────────────────
    w("\n---\n## 4. 年別 × レバ別 詳細\n")

    for yr in YEARS:
        w(f"\n### {yr}年\n")
        w("| Leverage | Add Signals | Avg RSR | Avg Lot | Avg +20d | Avg +60d | WR(20d) | PF(20d) |")
        w("|---|---|---|---|---|---|---|---|")
        for lbl, extra_cap in LEVERAGE_CASES.items():
            if extra_cap == 0:
                yr_blocked = [s for s in enriched if s["year"] == yr]
                w(f"| {lbl} (baseline) | 0 ({len(yr_blocked)} blocked) | — | — | — | — | — | — |")
                continue
            yr_sigs = [s for s in captured_by_lev[lbl] if s["year"] == yr]
            st = _stats(yr_sigs)
            w(f"| {lbl} | {st['n']} | {st['avg_rsr']:.1f} | ¥{st['avg_lot']:.0f}K "
              f"| {_fmt(st['avg_20d'])} | {_fmt(st['avg_60d'])} "
              f"| {st['wr_20d']:.1f}% | {st['pf_20d']:.2f} |")

    # ── 5. 年別 × レバ別 forward return (2022 単独強調) ─────────────────
    w("\n---\n## 5. 年別 +20d リターン比較 (レバ別)\n")
    w("| Year | 1.1x Avg+20d | 1.2x Avg+20d | 1.3x Avg+20d | 1.3x WR(20d) | 1.3x PF(20d) |")
    w("|---|---|---|---|---|---|")
    for yr in YEARS:
        row_parts = [str(yr)]
        for lbl in ["1.1x", "1.2x", "1.3x"]:
            yr_s = [s for s in captured_by_lev[lbl] if s["year"] == yr]
            st = _stats(yr_s)
            row_parts.append(_fmt(st["avg_20d"]))
        yr_13 = [s for s in captured_by_lev["1.3x"] if s["year"] == yr]
        st13 = _stats(yr_13)
        row_parts.append(f"{st13['wr_20d']:.1f}%")
        row_parts.append(f"{st13['pf_20d']:.2f}")
        w("| " + " | ".join(row_parts) + " |")

    # ── 6. lot コスト分布 ─────────────────────────────────────────────────
    w("\n---\n## 6. Blocked Signal の lot コスト分布\n")
    lot_costs = [s["lot_cost"] for s in enriched if s["rank_blocked"] == 0]
    buckets = [(0, 200_000, "≤¥200K"),
               (200_000, 300_000, "¥200〜300K"),
               (300_000, 600_000, "¥300〜600K"),
               (600_000, 900_000, "¥600〜900K"),
               (900_000, float("inf"), "¥900K超")]
    w("| lot コスト帯 | 件数 | 割合 | レバ到達 |")
    w("|---|---|---|---|")
    total_days = len(lot_costs)
    for lo, hi, label in buckets:
        cnt = sum(1 for c in lot_costs if lo < c <= hi) if lo > 0 else sum(1 for c in lot_costs if c <= hi)
        reach = ("1.1x〜" if hi <= 300_000 else
                 "1.2x〜" if hi <= 600_000 else
                 "1.3x〜" if hi <= 900_000 else "1.3x超え=捕捉不可")
        w(f"| {label} | {cnt} | {_pct(cnt, total_days)} | {reach} |")

    # ── 7. 結論 ──────────────────────────────────────────────────────────
    w("\n---\n## 7. 結論\n")

    # 統計量を計算してから判定
    st11 = _stats(captured_by_lev["1.1x"])
    st12 = _stats(captured_by_lev["1.2x"])
    st13 = _stats(captured_by_lev["1.3x"])

    steps_n  = [st11["n"], st12["n"] - st11["n"], st13["n"] - st12["n"]]
    steps_20 = [st11["avg_20d"], _stats([s for s in captured_by_lev["1.2x"]
                if (s["date"], s["sym"]) not in dates_captured["1.1x"]])["avg_20d"],
                _stats([s for s in captured_by_lev["1.3x"]
                if (s["date"], s["sym"]) not in dates_captured["1.2x"]])["avg_20d"]]

    # 線形判定: 各段階の追加シグナル数が均等か
    if len(steps_n) >= 2 and steps_n[0] > 0:
        ratio_12 = steps_n[1] / max(steps_n[0], 1)
        ratio_23 = steps_n[2] / max(steps_n[1], 1) if steps_n[1] > 0 else 0
        linear = (0.7 < ratio_12 < 1.3 and 0.7 < ratio_23 < 1.3)
    else:
        linear = False

    w("### (1) レバ効果は線形か\n")
    if linear:
        w(f"**ほぼ線形** (1.0→1.1x: +{steps_n[0]}, 1.1→1.2x: +{steps_n[1]}, "
          f"1.2→1.3x: +{steps_n[2]})")
    else:
        w(f"**非線形・逓減** (1.0→1.1x: +{steps_n[0]}, 1.1→1.2x: +{steps_n[1]}, "
          f"1.2→1.3x: +{steps_n[2]})")
    w(f"- 各段階の純増シグナル数の推移からlot コスト分布を反映した逓減傾向を確認。\n")

    w("### (2) レバ 1.1x の価値\n")
    if st11["n"] > 0 and st11["avg_20d"] is not None:
        w(f"- 追加シグナル数: **{st11['n']}** (全期間 / blocked days比 {_pct(st11['n'], total_blocked_days)})")
        w(f"- 平均 +20d: **{_fmt(st11['avg_20d'])}**  / 平均 +60d: **{_fmt(st11['avg_60d'])}**")
        w(f"- 勝率: {st11['wr_20d']:.1f}%  PF: {st11['pf_20d']:.2f}")
        if st11["avg_20d"] and st11["avg_20d"] > 2:
            w("- **評価: 有意なプラス期待値**")
        elif st11["avg_20d"] and st11["avg_20d"] > 0:
            w("- **評価: 小幅プラス (限定的)**")
        else:
            w("- **評価: 期待値マイナス (1.1x 追加の正当性なし)**")
    else:
        w("- 追加シグナルなし (lot コストが ¥300K 以下のシグナルが存在しない)")

    w("\n### (3) レバ 1.3x 到達時の期待値源泉\n")
    if st13["n"] > 0:
        marginal_13 = [s for s in captured_by_lev["1.3x"]
                       if (s["date"], s["sym"]) not in dates_captured["1.2x"]]
        st_mg13 = _stats(marginal_13)
        w(f"- 1.3x 追加シグナル計: **{st13['n']}**  avg +20d: **{_fmt(st13['avg_20d'])}**")
        w(f"- うち 1.2x では捕捉できない純増分: **{len(marginal_13)}**  "
          f"avg +20d: **{_fmt(st_mg13['avg_20d'])}**")
        w(f"- PF(20d): {st13['pf_20d']:.2f}  WR: {st13['wr_20d']:.1f}%")

    w("\n### (4) 機会損失回収 vs 単純レバ効果\n")
    w("- **機会損失回収** = 本分析の定量化対象。"
      "max_positions=3 の壁でブロックされたシグナルの期待値回収。")
    w("- **単純レバ効果** = 既存 3 ポジション × L 倍のリターン。"
      "baseline CAGR × L で近似 (借入コスト別途考慮要)。")
    baseline_cagr = m.get("cagr", 0)
    for lbl in ["1.1x", "1.2x", "1.3x"]:
        L_val = float(lbl[:-1])
        simple_cagr = round(baseline_cagr * L_val, 2)
        add_sig = captured_by_lev[lbl]
        st_add  = _stats(add_sig)
        # 追加シグナルの CAGR 寄与推定: avg_20d / 20d * 252 * (n_add/n_baseline_trades)
        frac = add_sig.__len__() / max(baseline_n_sells, 1)
        add_cagr_est = round(
            (st_add["avg_20d"] or 0) / 100 * (252 / 20) * frac * 100, 2
        )
        w(f"- **{lbl}**: 単純レバ CAGR ≈ {simple_cagr:+.1f}%  "
          f"機会損失回収 CAGR寄与推定 ≈ {add_cagr_est:+.2f}pp  "
          f"(追加 {len(add_sig)} 件 / baseline {baseline_n_sells} 件)")

    w("\n### (5) 逓減点\n")
    w(f"- 1.0x→1.1x: +{steps_n[0]} signals  (+20d={_fmt(steps_n[0] and steps_20[0])})")
    w(f"- 1.1x→1.2x: +{steps_n[1]} signals  (+20d={_fmt(steps_n[1] and steps_20[1])})")
    w(f"- 1.2x→1.3x: +{steps_n[2]} signals  (+20d={_fmt(steps_n[2] and steps_20[2])})")
    if steps_n[0] >= steps_n[1] >= steps_n[2]:
        w("- **単調逓減**: 高レバになるほど新規捕捉シグナルは減少")
    elif steps_n[1] > steps_n[0]:
        w("- **1.1x→1.2x が最大利得帯** (lot 中間コスト帯に多くのシグナルが集中)")
    else:
        w("- **逓減パターンなし**: lot コスト分布が偏っている")

    # ── 出力 ────────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート → {OUTPUT_FILE}")

    # ── コンソールサマリー ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  ★ レバレッジ別 機会損失分析 サマリー")
    print("=" * 72)
    print(f"\n  Baseline: CAGR={m.get('cagr',0):+.2f}%  PF={m.get('profit_factor',0):.2f}  "
          f"Trades={baseline_n_sells}")
    print(f"  Blocked days={total_blocked_days}  Total blocked sigs={total_blocked_sigs}\n")
    print(f"  {'Lev':>5}  {'Extra':>8}  {'AddSig':>7}  {'Cap%':>6}  "
          f"{'AvgRSR':>7}  {'AvgLot':>8}  {'Avg+20d':>9}  {'Avg+60d':>9}  "
          f"{'WR20d':>7}  {'PF20d':>7}")
    print("  " + "-" * 80)
    for lbl, extra_cap in LEVERAGE_CASES.items():
        if extra_cap == 0:
            print(f"  {lbl:>5}  {'¥0':>8}  {'0':>7}  {'0.0%':>6}  "
                  f"{'—':>7}  {'—':>8}  {'—':>9}  {'—':>9}  {'—':>7}  {'—':>7}")
            continue
        sigs = captured_by_lev[lbl]
        st   = _stats(sigs)
        _extra_str = f"¥{int(extra_cap):,}"
        _lot_str   = f"¥{st['avg_lot']:.0f}K"
        print(f"  {lbl:>5}  {_extra_str:>8}  {st['n']:>7}  "
              f"{_pct(st['n'], total_blocked_days):>6}  "
              f"{st['avg_rsr']:>7.1f}  {_lot_str:>8}  "
              f"{_fmt(st['avg_20d']):>9}  {_fmt(st['avg_60d']):>9}  "
              f"{st['wr_20d']:>6.1f}%  {st['pf_20d']:>7.2f}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
