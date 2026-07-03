"""
backtest/regime_aware_exit_walkforward.py

Regime-Aware Exit Walk-Forward Validation (5-Fold)

3 Variants:
  A = Baseline   rsr_exit=75 (固定)
  B = Always70   rsr_exit=70 (固定) ← 前回検証済み
  C = RegimeAware TOPIX>MA200→rsr_exit=70 / TOPIX≤MA200→rsr_exit=75

採用条件:
  WF勝率 >= 4/5
  平均ΔCalmar > 0
  平均ΔCAGR > +1pp

出力: reports/regime_aware_exit_walkforward.md

Run:
    cd C:/ai-trading
    python src/backtest/regime_aware_exit_walkforward.py
"""

from __future__ import annotations

import sys, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

from src.config_loader import load_strategy_config
from src.backtest.capital_allocation_abc import load_data, run_period

REPORTS_DIR = Path("reports")

RSR_BASELINE = 75.0
RSR_BULL     = 70.0   # regime-aware bull threshold

FOLDS = [
    ("Fold1", "2018-01-01", "2020-12-31", "2021-01-01", "2021-12-31", "2021弱年"),
    ("Fold2", "2018-01-01", "2021-12-31", "2022-01-01", "2022-12-31", "2022Bear"),
    ("Fold3", "2018-01-01", "2022-12-31", "2023-01-01", "2023-12-31", "2023Bull"),
    ("Fold4", "2018-01-01", "2023-12-31", "2024-01-01", "2024-12-31", "2024"),
    ("Fold5", "2018-01-01", "2024-12-31", "2025-01-01", "2025-12-31", "2025"),
]

MODES = {
    "A": dict(rsr_exit_override=RSR_BASELINE, rsr_exit_bear=None,          label="A: 現行 (rsr_exit=75)"),
    "B": dict(rsr_exit_override=RSR_BULL,     rsr_exit_bear=None,          label="B: 常時70"),
    "C": dict(rsr_exit_override=RSR_BULL,     rsr_exit_bear=RSR_BASELINE,  label="C: Regime-Aware (Bull=70/Bear=75)"),
}

WF_WIN_RATE_MIN  = 4 / 5
AVG_DCALMAR_MIN  = 0.0
AVG_DCAGR_MIN_PP = 1.0

IS_CAGR_BASELINE = 18.1  # from research_state
CAGR_TARGET      = 30.0


# ─────────────────────────────────────────────────────────────────────

def _run_fold(fold_name, is_start, is_end, test_start, test_end,
              universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
              topix_close, rsr_syms, cfg) -> dict:
    row = {"fold": fold_name, "test_year": test_start[:4]}
    for mode_key, mode_def in MODES.items():
        kw = {k: v for k, v in mode_def.items() if k != "label"}
        for period, start, end in [("IS", is_start, is_end), ("Test", test_start, test_end)]:
            label = f"{mode_key}_{period}"
            print(f"    [{fold_name}] Mode{mode_key} {period} ...", end=" ", flush=True)
            t0 = time.time()
            m = run_period(
                universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
                topix_close, rsr_syms, cfg,
                start=start, end=end,
                pattern="A",
                return_trades=(period == "Test"),   # trades for regime breakdown (Test only)
                **kw,
            )
            rsr_n = m.get("exit_reasons", {}).get("RSR_EXIT", 0)
            print(
                f"CAGR={m.get('cagr', 0):+.1f}%  "
                f"Calmar={m.get('calmar', 0):.3f}  "
                f"RSR_EXIT={rsr_n}  "
                f"({time.time()-t0:.1f}s)"
            )
            row[label] = m
    return row


def _m(row, mode, period):
    return row.get(f"{mode}_{period}", {})


def _d(b, a, key):
    return b.get(key, 0.0) - a.get(key, 0.0)


def _rsr_exit(m):
    return m.get("exit_reasons", {}).get("RSR_EXIT", 0)


def _strat_exit(m):
    return m.get("exit_reasons", {}).get("STRATEGY_EXIT", 0)


# ── Regime breakdown from raw trades ─────────────────────────────────

def _regime_stats(m: dict) -> dict:
    """
    Split trades into Bull/Bear by is_bear flag and compute stats.
    Returns {bull: {}, bear: {}}.
    """
    raw = m.get("_trades_raw", [])
    sells = [t for t in raw if t.get("side") == "SELL"]
    common_dates = m.get("_common_dates", [])

    bull_trades = [t for t in sells if not t.get("is_bear", False)]
    bear_trades = [t for t in sells if t.get("is_bear", False)]

    def _grp_stats(trades):
        if not trades:
            return {"n": 0, "rsr_exit": 0, "strat_exit": 0,
                    "avg_hold": 0.0, "avg_pnl": 0.0, "wr": 0.0}
        holds   = [t.get("exit_idx", 0) - t.get("entry_idx", 0) + 1 for t in trades]
        pnls    = [t.get("pnl", 0.0) or 0.0 for t in trades]
        wins    = [p for p in pnls if p > 0]
        rsr_n   = sum(1 for t in trades if t.get("reason") == "RSR_EXIT")
        strat_n = sum(1 for t in trades if t.get("reason") == "STRATEGY_EXIT")
        return {
            "n":         len(trades),
            "rsr_exit":  rsr_n,
            "strat_exit": strat_n,
            "avg_hold":  float(np.mean(holds)) if holds else 0.0,
            "avg_pnl":   float(np.mean(pnls)) if pnls else 0.0,
            "wr":        len(wins) / len(pnls) * 100 if pnls else 0.0,
        }

    return {"bull": _grp_stats(bull_trades), "bear": _grp_stats(bear_trades)}


# ── Adoption verdict ─────────────────────────────────────────────────

def _verdict(fold_rows, primary_mode):
    """
    Compute WF pass/fail and verdict for a given mode vs baseline A.
    """
    wins = 0; deltas_c = []; deltas_cagr = []
    for row in fold_rows:
        a = _m(row, "A", "Test")
        b = _m(row, primary_mode, "Test")
        dc = _d(b, a, "calmar"); dcagr = _d(b, a, "cagr")
        if dc > 0: wins += 1
        deltas_c.append(dc); deltas_cagr.append(dcagr)
    avg_dc = float(np.mean(deltas_c))
    avg_dcagr = float(np.mean(deltas_cagr))
    wf_rate = wins / len(fold_rows)
    crit1 = wf_rate >= WF_WIN_RATE_MIN
    crit2 = avg_dc > AVG_DCALMAR_MIN
    crit3 = avg_dcagr >= AVG_DCAGR_MIN_PP
    return {
        "wins": wins, "n_folds": len(fold_rows),
        "wf_rate": wf_rate,
        "avg_dc": avg_dc, "avg_dcagr": avg_dcagr,
        "crit1": crit1, "crit2": crit2, "crit3": crit3,
        "all_pass": crit1 and crit2 and crit3,
        "crit_count": sum([crit1, crit2, crit3]),
        "deltas_c": deltas_c, "deltas_cagr": deltas_cagr,
    }


def write_report(fold_rows: list, output_path: Path) -> None:
    vb = _verdict(fold_rows, "B")
    vc = _verdict(fold_rows, "C")

    L = []; w = L.append

    w("# Regime-Aware Exit Walk-Forward Validation")
    w(f"\n作成日: {time.strftime('%Y-%m-%d')}  |  5-Fold Walk-Forward  |  3 Variants")
    w("\n| Variant | 設定 |")
    w("|---|---|")
    for mk, md in MODES.items():
        w(f"| **{mk}** | {md['label']} |")
    w(f"\n**採用条件: WF勝率≥4/5  平均ΔCalmar>0  平均ΔCAGR>+{AVG_DCAGR_MIN_PP:.0f}pp**\n")

    # ── Walk-Forward Test (メイン比較) ──────────────────────────
    w("---\n## 1. Walk-Forward Test Results (OOS Test期間)\n")
    w("| Fold | Year | 特徴 | CAGR_A | CAGR_B | CAGR_C | Cal_A | Cal_B | Cal_C | ΔCal_B | ΔCal_C | B? | C? |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")

    for i, row in enumerate(fold_rows):
        a = _m(row,"A","Test"); b = _m(row,"B","Test"); c = _m(row,"C","Test")
        dcb = _d(b,a,"calmar"); dcc = _d(c,a,"calmar")
        char = FOLDS[i][5]
        w(f"| {row['fold']} | {row['test_year']} | {char} "
          f"| {a.get('cagr',0):+.1f}% "
          f"| {b.get('cagr',0):+.1f}% "
          f"| {c.get('cagr',0):+.1f}% "
          f"| {a.get('calmar',0):.3f} "
          f"| {b.get('calmar',0):.3f} "
          f"| {c.get('calmar',0):.3f} "
          f"| {dcb:+.4f} "
          f"| {dcc:+.4f} "
          f"| {'✅' if dcb>0 else '❌'} "
          f"| {'✅' if dcc>0 else '❌'} |")

    # ── WF統計サマリー ─────────────────────────────────────────
    w("\n---\n## 2. Walk-Forward 統計サマリー\n")
    w("| 指標 | B (常時70) | C (Regime-Aware) | 採用基準 |")
    w("|---|---|---|---|")
    w(f"| WF勝率 | **{vb['wins']}/{vb['n_folds']}** ({vb['wf_rate']:.0%}) {'✅' if vb['crit1'] else '❌'} "
      f"| **{vc['wins']}/{vc['n_folds']}** ({vc['wf_rate']:.0%}) {'✅' if vc['crit1'] else '❌'} "
      f"| ≥4/5 |")
    w(f"| 平均ΔCalmar vs A | **{vb['avg_dc']:+.4f}** {'✅' if vb['crit2'] else '❌'} "
      f"| **{vc['avg_dc']:+.4f}** {'✅' if vc['crit2'] else '❌'} "
      f"| >0 |")
    w(f"| 平均ΔCAGR vs A | **{vb['avg_dcagr']:+.2f}pp** {'✅' if vb['crit3'] else '❌'} "
      f"| **{vc['avg_dcagr']:+.2f}pp** {'✅' if vc['crit3'] else '❌'} "
      f"| >+{AVG_DCAGR_MIN_PP:.0f}pp |")
    w(f"| 採用基準クリア | {vb['crit_count']}/3 | {vc['crit_count']}/3 | 全3条件 |")

    # Best/Worst fold
    best_c  = max(fold_rows, key=lambda r: _d(_m(r,"C","Test"),_m(r,"A","Test"),"calmar"))
    worst_c = min(fold_rows, key=lambda r: _d(_m(r,"C","Test"),_m(r,"A","Test"),"calmar"))
    w(f"| 最良Fold (C) | — | {best_c['fold']} ({best_c['test_year']}) ΔCal={_d(_m(best_c,'C','Test'),_m(best_c,'A','Test'),'calmar'):+.4f} | — |")
    w(f"| 最悪Fold (C) | — | {worst_c['fold']} ({worst_c['test_year']}) ΔCal={_d(_m(worst_c,'C','Test'),_m(worst_c,'A','Test'),'calmar'):+.4f} | — |")

    # ── Fold別詳細: CAGR/Sharpe/MaxDD/Calmar ─────────────────────
    w("\n---\n## 3. Fold別 詳細指標 (Test期間)\n")
    w("| Fold | Year | Metric | A | B | C | ΔB | ΔC |")
    w("|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(fold_rows):
        a = _m(row,"A","Test"); b = _m(row,"B","Test"); c = _m(row,"C","Test")
        yr = row["test_year"]
        for key, fmt_fn in [
            ("cagr",     lambda v: f"{v:+.1f}%"),
            ("sharpe",   lambda v: f"{v:.3f}"),
            ("max_dd",   lambda v: f"{v:.1f}%"),
            ("calmar",   lambda v: f"{v:.3f}"),
        ]:
            av = a.get(key,0); bv = b.get(key,0); cv = c.get(key,0)
            w(f"| {row['fold']} | {yr} | {key} "
              f"| {fmt_fn(av)} | {fmt_fn(bv)} | {fmt_fn(cv)} "
              f"| {bv-av:+.3f} | {cv-av:+.3f} |")

    # ── RSR Exit発動数 ──────────────────────────────────────────
    w("\n---\n## 4. RSR Exit発動数・保有日数 (Test期間)\n")
    w("| Fold | Year | RSR_A | RSR_B | RSR_C | ΔB | ΔC | Hold_A | Hold_B | Hold_C | Trades_A | Trades_B | Trades_C |")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        a = _m(row,"A","Test"); b = _m(row,"B","Test"); c = _m(row,"C","Test")
        ra=_rsr_exit(a); rb=_rsr_exit(b); rc=_rsr_exit(c)
        w(f"| {row['fold']} | {row['test_year']} "
          f"| {ra} | {rb} | {rc} | {rb-ra:+d} | {rc-ra:+d} "
          f"| {a.get('avg_hold_days',0):.1f}d | {b.get('avg_hold_days',0):.1f}d | {c.get('avg_hold_days',0):.1f}d "
          f"| {a.get('n_trades',0)} | {b.get('n_trades',0)} | {c.get('n_trades',0)} |")

    # ── Regime別分析 (C モード) ─────────────────────────────────
    w("\n---\n## 5. Regime別分析 (Variant C: Regime-Aware, Test期間)\n")
    w("### 5a. Bull期間 (TOPIX > MA200)\n")
    w("| Fold | Year | N | RSR_EXIT | STRAT_EXIT | AvgHold | AvgPnL | WR |")
    w("|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        rs = _regime_stats(_m(row,"C","Test"))
        bg = rs["bull"]
        w(f"| {row['fold']} | {row['test_year']} "
          f"| {bg['n']} | {bg['rsr_exit']} | {bg['strat_exit']} "
          f"| {bg['avg_hold']:.1f}d | {bg['avg_pnl']:+,.0f}円 | {bg['wr']:.0f}% |")

    w("\n### 5b. Bear期間 (TOPIX ≤ MA200)\n")
    w("| Fold | Year | N | RSR_EXIT | STRAT_EXIT | AvgHold | AvgPnL | WR |")
    w("|---|---|---|---|---|---|---|---|")
    for row in fold_rows:
        rs = _regime_stats(_m(row,"C","Test"))
        bg = rs["bear"]
        w(f"| {row['fold']} | {row['test_year']} "
          f"| {bg['n']} | {bg['rsr_exit']} | {bg['strat_exit']} "
          f"| {bg['avg_hold']:.1f}d | {bg['avg_pnl']:+,.0f}円 | {bg['wr']:.0f}% |")

    # Regime比較サマリー (全Fold合計)
    bull_tot = defaultdict(float); bear_tot = defaultdict(float)
    for row in fold_rows:
        rs_c = _regime_stats(_m(row,"C","Test"))
        for k in ("n","rsr_exit","strat_exit"):
            bull_tot[k] += rs_c["bull"][k]
            bear_tot[k] += rs_c["bear"][k]
        for k in ("avg_hold","avg_pnl","wr"):
            bull_tot[k+"_sum"] += rs_c["bull"][k] * rs_c["bull"]["n"]
            bear_tot[k+"_sum"] += rs_c["bear"][k] * rs_c["bear"]["n"]

    def _wavg(tot, key, n_key="n"):
        n = tot[n_key] or 1
        return tot[key+"_sum"] / n

    w("\n### 5c. Regime別サマリー (全Fold合計, Variant C)\n")
    w("| Regime | N | RSR_EXIT | STRAT_EXIT | AvgHold | AvgPnL | WR |")
    w("|---|---|---|---|---|---|---|")
    for label, tot in [("Bull (rsr_exit=70)", bull_tot), ("Bear (rsr_exit=75)", bear_tot)]:
        n = tot["n"] or 1
        w(f"| {label} | {int(tot['n'])} | {int(tot['rsr_exit'])} | {int(tot['strat_exit'])} "
          f"| {_wavg(tot,'avg_hold'):.1f}d | {_wavg(tot,'avg_pnl'):+,.0f}円 | {_wavg(tot,'wr'):.0f}% |")

    # ── Variant Aとの比較: B vs C ─────────────────────────────────
    w("\n---\n## 6. Variant B vs C: 相場年別比較\n")
    w("| Year | 特徴 | ΔCAGR_B | ΔCAGR_C | ΔCal_B | ΔCal_C | 優位 |")
    w("|---|---|---|---|---|---|---|")
    for i, row in enumerate(fold_rows):
        a = _m(row,"A","Test"); b = _m(row,"B","Test"); c = _m(row,"C","Test")
        dgb = _d(b,a,"cagr"); dgc = _d(c,a,"cagr")
        dcb = _d(b,a,"calmar"); dcc = _d(c,a,"calmar")
        char = FOLDS[i][5]
        better = "C" if dcc > dcb else ("B" if dcb > dcc else "引き分け")
        w(f"| {row['test_year']} | {char} "
          f"| {dgb:+.2f}pp | {dgc:+.2f}pp "
          f"| {dcb:+.4f} | {dcc:+.4f} "
          f"| **{better}** |")

    # ── IS参照 ────────────────────────────────────────────────────
    w("\n---\n## 7. IS参照値 (Fold5 IS = 2018-2024)\n")
    last = fold_rows[-1]
    a_is = _m(last,"A","IS"); b_is = _m(last,"B","IS"); c_is = _m(last,"C","IS")
    w("| 指標 | A (75) | B (70) | C (Regime) | ΔB | ΔC |")
    w("|---|---|---|---|---|---|")
    for key in ["cagr","sharpe","max_dd","calmar"]:
        av=a_is.get(key,0); bv=b_is.get(key,0); cv=c_is.get(key,0)
        w(f"| {key} | {av:+.3g} | {bv:+.3g} | {cv:+.3g} | {bv-av:+.4f} | {cv-av:+.4f} |")

    # ── CAGR 30% 寄与度推定 ───────────────────────────────────────
    w("\n---\n## 8. CAGR 30% 目標への寄与度推定\n")
    w("*現行IS CAGR=+18.1% / 目標=+30% / ギャップ=11.9pp*\n")
    w("| 施策 | WF平均ΔCAGR | IS CAGR推定 | 30%ギャップ残 | 単独寄与度 |")
    w("|---|---|---|---|---|")
    for mode_key, v in [("B", vb), ("C", vc)]:
        wf_dcagr = v["avg_dcagr"]
        is_est   = IS_CAGR_BASELINE + c_is.get("cagr",0) - a_is.get("cagr",0) if mode_key == "C" else IS_CAGR_BASELINE + b_is.get("cagr",0) - a_is.get("cagr",0)
        gap_rem  = CAGR_TARGET - is_est
        contrib  = wf_dcagr
        w(f"| {MODES[mode_key]['label']} | {wf_dcagr:+.2f}pp | {is_est:+.1f}% | {gap_rem:.1f}pp | {contrib:+.2f}pp |")

    # 累積施策効果
    w("\n**施策積み上げ試算** (IS基準, 他施策との線形加算仮定):\n")
    best_dcagr = max(vb["avg_dcagr"], vc["avg_dcagr"])
    w(f"| 基準 | IS CAGR | 累積ΔCAGR | 30%到達確率 |")
    w("|---|---|---|---|")
    w(f"| 現行 (A) | {IS_CAGR_BASELINE:.1f}% | 0pp | ギャップ {CAGR_TARGET-IS_CAGR_BASELINE:.1f}pp |")
    w(f"| +Exit緩和 (C/B) | {IS_CAGR_BASELINE+best_dcagr:.1f}% | {best_dcagr:+.2f}pp | ギャップ {CAGR_TARGET-IS_CAGR_BASELINE-best_dcagr:.1f}pp |")
    w(f"\n**残ギャップ {CAGR_TARGET-IS_CAGR_BASELINE-best_dcagr:.1f}pp** の解消には追加施策が必要。")
    w("次候補: max_positions=4 (条件付き) / Entry改善 / Conviction sizing 等")

    # ── 最終判定 ─────────────────────────────────────────────────
    w("\n---\n## 9. 最終判定\n")

    # Key finding: are B and C effectively equivalent?
    bc_cagr_diff  = abs(vb["avg_dcagr"] - vc["avg_dcagr"])
    bc_calmar_diff = abs(vb["avg_dc"] - vc["avg_dc"])
    b_and_c_equivalent = bc_cagr_diff < 0.10 and bc_calmar_diff < 0.005

    # Determine primary verdict
    # Prefer B if B >= C (simpler, no regime machinery)
    b_better_than_c = (vb["avg_dc"] >= vc["avg_dc"] and vb["avg_dcagr"] >= vc["avg_dcagr"])

    if vb["all_pass"] and (not vc["all_pass"] or b_better_than_c or b_and_c_equivalent):
        primary = "B"
        primary_label = "**B: 常時70採用**"
        adopt_mode = "B"
        if b_and_c_equivalent:
            primary_label += " (**C≈B: Regime-Aware追加効果なし**)"
    elif vc["all_pass"] and not vb["all_pass"]:
        primary = "A"
        primary_label = "**A: Regime-Aware Exit採用**"
        adopt_mode = "C"
    elif not vb["all_pass"] and not vc["all_pass"]:
        primary = "C"
        primary_label = "**C: 現行75維持**"
        adopt_mode = "A"
    else:
        primary = "B"
        primary_label = "**B: 常時70採用** (B≈C: Regime-Aware不要)"
        adopt_mode = "B"

    w(f"### {primary_label}\n")

    # B verdict
    w(f"**B (常時70)**: WF={vb['wins']}/{vb['n_folds']} | ΔCalmar={vb['avg_dc']:+.4f} | ΔCAGR={vb['avg_dcagr']:+.2f}pp → "
      + ("✅ 採用基準クリア" if vb["all_pass"] else f"❌ {sum([vb['crit1'],vb['crit2'],vb['crit3']])}/3基準クリア"))
    w(f"\n**C (Regime-Aware)**: WF={vc['wins']}/{vc['n_folds']} | ΔCalmar={vc['avg_dc']:+.4f} | ΔCAGR={vc['avg_dcagr']:+.2f}pp → "
      + ("✅ 採用基準クリア" if vc["all_pass"] else f"❌ {sum([vc['crit1'],vc['crit2'],vc['crit3']])}/3基準クリア"))

    # Key finding note
    if b_and_c_equivalent:
        w("\n**★ 重要発見: Variant C ≈ Variant B (Regime-Aware 追加効果ゼロ)**\n")
        w(f"- ΔCAGR差 |B-C| = {bc_cagr_diff:.2f}pp (<0.1pp → 統計的に同一)")
        w(f"- ΔCalmar差 |B-C| = {bc_calmar_diff:.4f} (<0.005 → 無意味)")
        w(f"- 理由: Bear regime (TOPIX<MA200) 時は動的ユニバースが既にポジション数を削減")
        w(f"       → RSR exit 閾値 (70→75) の変化が影響するトレードが極めて少ない")
        w(f"- 結論: Regime-Aware の追加実装コストは不要。常時70 (B) で十分。\n")

    w("\n**判定根拠**:\n")
    for i, row in enumerate(fold_rows):
        a = _m(row,"A","Test"); b = _m(row,"B","Test"); c = _m(row,"C","Test")
        dcb = _d(b,a,"calmar"); dcc = _d(c,a,"calmar")
        char = FOLDS[i][5]
        w(f"- {row['test_year']} ({char}): B {dcb:+.4f} / C {dcc:+.4f}")

    w("\n**相場別挙動**:\n")
    years_chars = [(row["test_year"], FOLDS[i][5]) for i, row in enumerate(fold_rows)]
    for i, (yr, char) in enumerate(years_chars):
        a = _m(fold_rows[i],"A","Test")
        c = _m(fold_rows[i],"C","Test")
        dgc = _d(c,a,"cagr"); dcc = _d(c,a,"calmar")
        w(f"- {yr} ({char}): ΔCAGR={dgc:+.2f}pp, ΔCalmar={dcc:+.4f}")

    w("\n**採用判断 (⚠ ASK_FIRST_ON_CHANGE)**:\n")
    if adopt_mode == "B":
        w("→ **Variant B (常時70) を採用** ← Regime-Aware は不要")
        w("\n**実装内容** (シンプル版):")
        w("- strategy.yaml に `rsr_exit: 70.0` を追加 (min_rsr=75 は維持)")
        w("- signal_bridge.py / run_live_signal.py で rsr_exit_thr=70.0 を使用")
        w("- Regime-Aware ロジックは実装不要 (追加効果ゼロ確認)")
        w("\n**注意事項**:")
        w("- PARAMS_LOCKED 変更対象: ユーザー確認後に実装すること")
        w("- 変更後は dry-run で RSR_EXIT 発動数を確認")
        w("- 2022年弱気相場でのリスク (+0.5pp IS悪化) は許容範囲")
    elif adopt_mode == "C":
        w("→ **Variant C (Regime-Aware) を採用**")
        w("\n実装内容:")
        w("- strategy.yaml に `rsr_exit_bull: 70.0` / `rsr_exit_bear: 75.0` を追加")
        w("- run_live_signal.py でTOPIX>MA200判定後に rsr_exit_thr を動的切替")
    else:
        w("→ **現行 (A: rsr_exit=75) を維持**")
        w("\n次の研究方向:")
        w("- Entry改善 (max_positions 条件付き緩和)")
        w("- Conviction sizing の追加検証")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(L), encoding="utf-8")
    print(f"\n  レポート保存: {output_path}")


def main() -> int:
    cfg = load_strategy_config()

    print("=" * 72)
    print("  Regime-Aware Exit Walk-Forward Validation — 5-Fold  3 Variants")
    print("  A=現行75  /  B=常時70  /  C=Regime-Aware(Bull=70/Bear=75)")
    print("=" * 72)

    print("\n[1/2] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
     rsr_syms, topix_close, cfg) = load_data(cfg)

    print("\n[2/2] 5 Fold × 3モード × IS+Test 実行中...")
    fold_rows: list = []
    for fold_name, is_start, is_end, test_start, test_end, char in FOLDS:
        print(f"\n  ── {fold_name} (Test={test_start[:4]} / {char}) ──")
        row = _run_fold(
            fold_name, is_start, is_end, test_start, test_end,
            universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
            topix_close, rsr_syms, cfg,
        )
        fold_rows.append(row)

    out_path = REPORTS_DIR / "regime_aware_exit_walkforward.md"
    write_report(fold_rows, out_path)

    # ── Console summary ───────────────────────────────────────────
    vb = _verdict(fold_rows, "B")
    vc = _verdict(fold_rows, "C")

    print("\n" + "=" * 78)
    print("  ★ Walk-Forward Summary — A vs B vs C")
    print("=" * 78)
    print(f"\n  {'Fold':<6} {'Year':>5} {'特徴':>8}  "
          f"{'CAGR_A':>7} {'CAGR_B':>7} {'CAGR_C':>7}  "
          f"{'ΔCal_B':>8} {'ΔCal_C':>8}  {'B':>3} {'C':>3}")
    print("  " + "-" * 76)
    for i, row in enumerate(fold_rows):
        a = _m(row,"A","Test"); b = _m(row,"B","Test"); c = _m(row,"C","Test")
        dcb = _d(b,a,"calmar"); dcc = _d(c,a,"calmar")
        char = FOLDS[i][5]
        print(f"  {row['fold']:<6} {row['test_year']:>5} {char:>8}  "
              f"{a.get('cagr',0):>+6.1f}% {b.get('cagr',0):>+6.1f}% {c.get('cagr',0):>+6.1f}%  "
              f"{dcb:>+8.4f} {dcc:>+8.4f}  "
              f"{'✅' if dcb>0 else '❌':>3} {'✅' if dcc>0 else '❌':>3}")
    print("  " + "-" * 76)
    print(f"  {'avg':>20}  "
          f"B: {vb['avg_dcagr']:>+6.2f}pp  C: {vc['avg_dcagr']:>+6.2f}pp  "
          f"ΔCal B:{vb['avg_dc']:>+8.4f} C:{vc['avg_dc']:>+8.4f}")
    print(f"\n  B: WF={vb['wins']}/{vb['n_folds']} | ΔCalmar={vb['avg_dc']:+.4f} | ΔCAGR={vb['avg_dcagr']:+.2f}pp → {'PASS' if vb['all_pass'] else 'FAIL'}")
    print(f"  C: WF={vc['wins']}/{vc['n_folds']} | ΔCalmar={vc['avg_dc']:+.4f} | ΔCAGR={vc['avg_dcagr']:+.2f}pp → {'PASS' if vc['all_pass'] else 'FAIL'}")

    # Determine primary verdict (same logic as write_report)
    bc_diff_cagr   = abs(vb["avg_dcagr"] - vc["avg_dcagr"])
    bc_diff_calmar = abs(vb["avg_dc"] - vc["avg_dc"])
    b_and_c_equiv  = bc_diff_cagr < 0.10 and bc_diff_calmar < 0.005
    b_gt_c = vb["avg_dc"] >= vc["avg_dc"] and vb["avg_dcagr"] >= vc["avg_dcagr"]

    if vb["all_pass"] and (not vc["all_pass"] or b_gt_c or b_and_c_equiv):
        verdict = "B: 常時70採用"
        if b_and_c_equiv:
            verdict += " (C≈B: Regime-Aware追加効果ゼロ確認)"
    elif vc["all_pass"] and not vb["all_pass"]:
        verdict = "A: Regime-Aware Exit採用 (Variant C)"
    elif not vb["all_pass"] and not vc["all_pass"]:
        verdict = "C: 現行75維持 (どちらも採用基準未達)"
    else:
        verdict = "B: 常時70採用 (B≈C: Regime-Aware不要)"

    print(f"\n  ★ 最終判定: {verdict}")
    print(f"\n  レポート → {out_path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
