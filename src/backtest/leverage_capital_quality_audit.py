"""
leverage_capital_quality_audit.py
==================================
目的: 「資金不足による高品質銘柄スキップ」の定量化

前提:
- max_positions=3 固定 (PARAMS_LOCKED)
- スロット(空き枠)はあるが、alloc < lot_cost で qty=0 になりスキップした銘柄を対象
- 「ポジション数増加効果」は評価しない
- 事実集計のみ。戦略・パラメータ変更なし。

出力: docs/research/leverage_capital_quality_audit.md
"""

import sys
import os
import math
from pathlib import Path
from datetime import date as Date

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

# ─── 定数 ────────────────────────────────────────────────────────────────
CAPITAL   = 3_000_000
LOT       = 100
PERIOD_START = "2018-01-01"
PERIOD_END   = "2025-12-31"
LEVERAGE_CASES = {
    "1.0x": 1.0,
    "1.1x": 1.1,
    "1.2x": 1.2,
    "1.3x": 1.3,
}
FWD_HORIZONS = [20, 60]  # 営業日

OUT_FILE = ROOT / "docs" / "research" / "leverage_capital_quality_audit.md"

# ─── データロード ──────────────────────────────────────────────────────────
from src.config_loader import load_strategy_config
from backtest.capital_allocation_abc import load_data, run_period

def _load():
    cfg = load_strategy_config()
    return load_data(cfg)

# ─── Forward Return ────────────────────────────────────────────────────────
def _fwd_return(sym: str, signal_date_str: str, entry_px: float,
                horizon: int, close_by_sym: dict) -> "float | None":
    """
    entry_px: signal翌営業日のopen (取得価格)
    horizon:  signal翌営業日から数えた営業日後のclose
    Returns: close[n] / entry_px - 1 、データ不足→None
    """
    if sym not in close_by_sym or entry_px <= 0:
        return None
    s = close_by_sym[sym]
    try:
        sig_dt = pd.Timestamp(signal_date_str)
        pos = s.index.searchsorted(sig_dt, side="right")  # 翌営業日以降
        target = pos + horizon - 1
        if target >= len(s):
            return None
        cl = float(s.iloc[target])
        return cl / entry_px - 1 if cl > 0 else None
    except Exception:
        return None

# ─── 統計集計 ──────────────────────────────────────────────────────────────
def _stats(records: list, key20: str = "fwd20", key60: str = "fwd60") -> dict:
    v20 = [r[key20] for r in records if r.get(key20) is not None]
    v60 = [r[key60] for r in records if r.get(key60) is not None]
    if not v20:
        return {"n": len(records), "avg20": None, "avg60": None,
                "wr20": None, "wr60": None, "pf20": None, "pf60": None,
                "avg_rsr": None, "avg_lot": None}

    def _pf(vals):
        wins  = [v for v in vals if v > 0]
        loss  = [abs(v) for v in vals if v < 0]
        if not loss:
            return float("inf")
        return sum(wins) / sum(loss) if wins else 0.0

    avg_rsr = float(np.mean([r["rsr"] for r in records]))
    avg_lot = float(np.mean([r["lot_cost"] for r in records])) / 1000.0

    return {
        "n":       len(records),
        "avg_rsr": round(avg_rsr, 1),
        "avg_lot": round(avg_lot, 0),
        "avg20":   round(float(np.mean(v20)) * 100, 2),
        "avg60":   round(float(np.mean(v60)) * 100, 2) if v60 else None,
        "wr20":    round(sum(1 for v in v20 if v > 0) / len(v20) * 100, 1),
        "wr60":    round(sum(1 for v in v60 if v > 0) / len(v60) * 100, 1) if v60 else None,
        "pf20":    round(_pf(v20), 2),
        "pf60":    round(_pf(v60), 2) if v60 else None,
    }

def _fmt(v):
    if v is None:
        return "—"
    return f"{v:+.2f}%"

def _fmtf(v):
    if v is None:
        return "—"
    return f"{v:.2f}"

# ─── メイン ──────────────────────────────────────────────────────────────
def main() -> int:
    print("=" * 72)
    print("  資金不足スキップ分析 — 銘柄品質向上効果 (事実集計)")
    print(f"  期間: {PERIOD_START} 〜 {PERIOD_END}")
    print(f"  CAPITAL=¥{CAPITAL:,}  max_positions=3 (固定)")
    print("=" * 72)

    # 1. データロード
    print("\n[1/3] データロード中...")
    (universe_raw, rsr_df, alpha_df, sym_active_df,
     regime_df, rsr_syms, topix_close, cfg) = _load()

    close_by_sym: dict = {
        sym: universe_raw[sym]["df"]["Close"]
        for sym in universe_raw
    }
    open_by_sym: dict = {
        sym: universe_raw[sym]["df"]["Open"]
        for sym in universe_raw
    }

    # 2. ベースラインBT: 資金不足スキップ収集
    print("\n[2/3] ベースラインバックテスト実行 (cash_skipped 収集)...")
    res = run_period(
        universe_raw, rsr_df, alpha_df, sym_active_df, regime_df,
        topix_close, rsr_syms, cfg,
        start=PERIOD_START, end=PERIOD_END,
        pattern="A",
        max_new_per_day=2,
        return_cash_skipped=True,
        verbose=True,
    )
    skipped_raw = res.get("_cash_skipped", [])
    print(f"  完了 / CAGR={res['cagr']:+.2f}%  Trades={res['n_trades']}")
    print(f"  cash_skipped total={len(skipped_raw)}")

    # 3. Forward return 計算
    print("\n[3/3] forward return 計算 + レバ別救済シミュレーション...")

    # open_by_sym を使って entry_px を取得（翌営業日open）
    enriched: list = []
    for r in skipped_raw:
        sym        = r["sym"]
        sig_date   = r["date"]
        buy_px     = r["buy_px"]  # 翌営業日open（バックテスト内で記録済み）
        lot_cost   = r["lot_cost"]
        alloc      = r["alloc"]

        f20 = _fwd_return(sym, sig_date, buy_px, 20, close_by_sym)
        f60 = _fwd_return(sym, sig_date, buy_px, 60, close_by_sym)

        enriched.append({
            **r,
            "fwd20": f20,
            "fwd60": f60,
        })

    # ─── 集計1: 全資金不足スキップ概要 ────────────────────────────────
    total_skipped = len(enriched)
    rank0 = [r for r in enriched if r["rank"] == 0]   # 1位候補スキップ
    rank1 = [r for r in enriched if r["rank"] == 1]   # 2位候補スキップ
    rank2 = [r for r in enriched if r["rank"] == 2]   # 3位候補スキップ
    rank_top3 = [r for r in enriched if r["rank"] <= 2]

    st_all   = _stats(enriched)
    st_rank0 = _stats(rank0)
    st_rank1 = _stats(rank1)
    st_rank2 = _stats(rank2)
    st_top3  = _stats(rank_top3)

    # ─── 集計2: レバ別救済分析 ────────────────────────────────────────
    # 救済条件: alloc * L >= lot_cost
    lev_rescued: dict = {}
    for lbl, L in LEVERAGE_CASES.items():
        rescued = [r for r in enriched if r["alloc"] * L >= r["lot_cost"]]
        lev_rescued[lbl] = rescued

    # 救済されない（1.3xでも不足）: alloc * 1.3 < lot_cost
    not_rescued_any = [r for r in enriched
                       if r["alloc"] * 1.3 < r["lot_cost"]]

    # レバ段階の純増（新たに救済される分）
    prev_set: set = set()
    lev_marginal: dict = {}
    for lbl, L in LEVERAGE_CASES.items():
        cur_set = set(
            (r["date"], r["sym"], r["rank"]) for r in lev_rescued[lbl]
        )
        new_rescued = [r for r in lev_rescued[lbl]
                       if (r["date"], r["sym"], r["rank"]) not in prev_set]
        lev_marginal[lbl] = new_rescued
        prev_set = cur_set

    # ─── 年別×レバ別集計 ──────────────────────────────────────────────
    years = list(range(2020, 2026))
    year_lev_stats: dict = {}
    for yr in years:
        year_lev_stats[yr] = {}
        for lbl, L in LEVERAGE_CASES.items():
            yr_recs = [r for r in lev_rescued[lbl] if r["date"].startswith(str(yr))]
            year_lev_stats[yr][lbl] = _stats(yr_recs)

    # ─── レポート生成 ─────────────────────────────────────────────────
    lines: list = []
    A = lines.append

    A("# 資金不足スキップ分析 — レバレッジによる銘柄品質向上効果")
    A("")
    A(f"作成日: {Date.today().isoformat()}  /  期間: {PERIOD_START}〜{PERIOD_END}")
    A("")
    A("**前提**:")
    A("- max_positions=3 固定 (PARAMS_LOCKED) — ポジション数増加効果は非評価")
    A("- 評価対象: スロット(空き枠)があるにもかかわらず資金不足で買えなかった銘柄")
    A("- 「資金不足」= qty=0 (alloc < lot_cost) で Pattern A がスキップ処理した件")
    A("- 各倍率: 1.1x=¥330万 / 1.2x=¥360万 / 1.3x=¥390万 相当の資金力")
    A("")
    A("---")

    # ── Section 1: 全体サマリー ──────────────────────────────────────
    A("## 1. 資金不足スキップ全体サマリー")
    A("")
    A(f"- **ベースライン**: CAGR={res['cagr']:+.2f}%  PF={res['profit_factor']:.2f}  "
      f"Trades={res['n_trades']}")
    A(f"- **総スキップ件数**: {total_skipped}")
    A(f"  - rank1（1位候補）: {len(rank0)} 件")
    A(f"  - rank2（2位候補）: {len(rank1)} 件")
    A(f"  - rank3（3位候補）: {len(rank2)} 件")
    A(f"  - rank4以降: {total_skipped - len(rank_top3)} 件")
    A("")
    A("| グループ | N | AvgRSR | AvgLot(¥K) | Avg+20d | Avg+60d | WR20 | WR60 | PF20 | PF60 |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for lbl2, st2 in [
        ("全スキップ", st_all),
        ("rank1(1位)", st_rank0),
        ("rank2(2位)", st_rank1),
        ("rank3(3位)", st_rank2),
        ("top3合計",   st_top3),
    ]:
        A(f"| {lbl2} | {st2['n']} | {st2['avg_rsr'] or '—'} | "
          f"{st2['avg_lot'] or '—'} | {_fmt(st2['avg20'])} | {_fmt(st2['avg60'])} | "
          f"{st2['wr20'] or '—'}% | {st2['wr60'] or '—'}% | "
          f"{_fmtf(st2['pf20'])} | {_fmtf(st2['pf60'])} |")
    A("")
    A("---")

    # ── Section 2: レバ別救済統計 ────────────────────────────────────
    A("## 2. レバ別救済統計")
    A("")
    A("救済条件: `alloc × L ≥ lot_cost` — レバがあれば1ロット取得可能")
    A("")
    A("| Leverage | 救済件数 | 救済率 | AvgRSR | AvgLot(¥K) | Avg+20d | Avg+60d | WR20 | WR60 | PF20 | PF60 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|")
    for lbl, L in LEVERAGE_CASES.items():
        rescued = lev_rescued[lbl]
        st = _stats(rescued)
        rate = f"{len(rescued)/max(1,total_skipped)*100:.1f}%" if total_skipped else "—"
        A(f"| {lbl} | {len(rescued)} | {rate} | {st['avg_rsr'] or '—'} | "
          f"{st['avg_lot'] or '—'} | {_fmt(st['avg20'])} | {_fmt(st['avg60'])} | "
          f"{st['wr20'] or '—'}% | {st['wr60'] or '—'}% | "
          f"{_fmtf(st['pf20'])} | {_fmtf(st['pf60'])} |")
    A("")
    A(f"※ 1.3xでも救済不可: {len(not_rescued_any)} 件"
      f" (avg lot_cost={int(np.mean([r['lot_cost'] for r in not_rescued_any])):,}円"
      if not_rescued_any else
      f"※ 1.3xでも救済不可: {len(not_rescued_any)} 件")
    A("")
    A("---")

    # ── Section 3: レバ段階別 限界利得 ──────────────────────────────
    A("## 3. レバ段階別 限界救済 (Marginal Analysis)")
    A("")
    A("| レバ幅 | 純増救済数 | AvgRSR | AvgLot(¥K) | Avg+20d | Avg+60d | WR20 | PF20 | PF60 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for lbl, L in LEVERAGE_CASES.items():
        mg = lev_marginal[lbl]
        if not mg:
            A(f"| 〜{lbl} | 0 | — | — | — | — | — | — | — |")
            continue
        st = _stats(mg)
        A(f"| 〜{lbl} | {len(mg)} | {st['avg_rsr'] or '—'} | "
          f"{st['avg_lot'] or '—'} | {_fmt(st['avg20'])} | {_fmt(st['avg60'])} | "
          f"{st['wr20'] or '—'}% | {_fmtf(st['pf20'])} | {_fmtf(st['pf60'])} |")
    A("")
    A("---")

    # ── Section 4: 年別 × レバ別詳細 ────────────────────────────────
    A("## 4. 年別 × レバ別詳細")
    A("")
    for yr in years:
        A(f"### {yr}年")
        A("")
        A("| Leverage | 救済N | Avg+20d | Avg+60d | WR20 | WR60 | PF20 | PF60 |")
        A("|---|---|---|---|---|---|---|---|")
        for lbl in LEVERAGE_CASES:
            st = year_lev_stats[yr][lbl]
            A(f"| {lbl} | {st['n']} | {_fmt(st['avg20'])} | {_fmt(st['avg60'])} | "
              f"{st['wr20'] or '—'}% | {st['wr60'] or '—'}% | "
              f"{_fmtf(st['pf20'])} | {_fmtf(st['pf60'])} |")
        A("")
    A("---")

    # ── Section 5: rank1スキップ詳細（最重要） ───────────────────────
    A("## 5. rank1（最高位）スキップ銘柄の詳細")
    A("")
    A("rank=0 は「その日のBUY候補1位」がスキップされたケース。")
    A("これが多いほど、戦略は「最高品質銘柄を資金不足で諦めている」ことになる。")
    A("")
    if rank0:
        st0 = st_rank0
        A(f"- 件数: **{st0['n']}** (全スキップの{len(rank0)/max(1,total_skipped)*100:.1f}%)")
        A(f"- 平均RSR: {st0['avg_rsr']}")
        A(f"- 平均lot_cost: ¥{int(st0['avg_lot']*1000):,}")
        A(f"- Avg+20d: **{_fmt(st0['avg20'])}**  WR20: {st0['wr20']}%  PF20: **{_fmtf(st0['pf20'])}**")
        A(f"- Avg+60d: **{_fmt(st0['avg60'])}**  WR60: {st0['wr60']}%  PF60: **{_fmtf(st0['pf60'])}**")
        A("")
        # レバ別救済
        A("#### rank1スキップ × レバ別救済")
        A("")
        A("| Leverage | 救済N | 救済率 | Avg+20d | WR20 | PF20 | PF60 |")
        A("|---|---|---|---|---|---|---|")
        for lbl, L in LEVERAGE_CASES.items():
            rank0_rescued = [r for r in rank0 if r["alloc"] * L >= r["lot_cost"]]
            st = _stats(rank0_rescued)
            rate = f"{len(rank0_rescued)/max(1,len(rank0))*100:.1f}%"
            A(f"| {lbl} | {len(rank0_rescued)} | {rate} | "
              f"{_fmt(st['avg20'])} | {st['wr20'] or '—'}% | "
              f"{_fmtf(st['pf20'])} | {_fmtf(st['pf60'])} |")
    else:
        A("rank1スキップなし (資金不足で1位候補を見送ったケースは発生していない)")
    A("")
    A("---")

    # ── Section 6: lot_cost分布 ──────────────────────────────────────
    A("## 6. スキップ銘柄のlot_cost分布")
    A("")
    lot_costs = [r["lot_cost"] for r in enriched]
    thresholds = [
        (0,       300_000, "≤¥300K",    "1.1x必要"),
        (300_001, 600_000, "¥300-600K", "1.2x必要"),
        (600_001, 900_000, "¥600-900K", "1.3x必要"),
        (900_001, 9_999_999_999, ">¥900K", "1.3x超え(救済不可)"),
    ]
    A("| lot_cost帯 | 件数 | 割合 | レバ要件 |")
    A("|---|---|---|---|")
    for lo, hi, label, req in thresholds:
        cnt = sum(1 for c in lot_costs if lo <= c <= hi)
        pct = cnt / max(1, total_skipped) * 100
        A(f"| {label} | {cnt} | {pct:.1f}% | {req} |")
    A("")
    A("---")

    # ── Section 7: 結論 ──────────────────────────────────────────────
    A("## 7. 結論")
    A("")

    # 判定
    top3_pf20  = st_top3.get("pf20")
    top3_wr20  = st_top3.get("wr20")
    top3_avg60 = st_top3.get("avg60")
    rank0_pf20 = st_rank0.get("pf20")
    rank0_pf60 = st_rank0.get("pf60")

    def _judge(pf20, wr20, avg60):
        if pf20 is None:
            return "データ不足"
        if pf20 > 1.2 and (wr20 or 0) > 50 and (avg60 or 0) > 0:
            return "POSITIVE — レバ導入価値あり (PF>1.2 ∧ WR>50% ∧ +60d>0)"
        if pf20 > 1.0:
            return "WEAK_POSITIVE — PF>1.0 だが採用基準(PF>1.2 ∧ WR>50%)未達"
        return "NEGATIVE — 期待値マイナス。レバによる品質向上効果なし"

    A("### (1) 資金不足スキップの頻度")
    A(f"- 全スキップ: **{total_skipped}件** (期間2018-2025)")
    A(f"- うちrank1(最高位)スキップ: **{len(rank0)}件**")
    if total_skipped == 0:
        A("- **スキップゼロ**: 現物3Mで資金不足は発生していない → レバ不要")
    else:
        A(f"- スキップが{total_skipped}件存在 → 資金制約が一定頻度で発動している")

    A("")
    A("### (2) スキップ銘柄の期待値")
    verdict_top3  = _judge(top3_pf20,  top3_wr20, top3_avg60)
    verdict_rank0 = _judge(rank0_pf20, st_rank0.get("wr20"), st_rank0.get("avg60"))
    A(f"- top3スキップ (rank0-2): **{verdict_top3}**")
    A(f"  - PF20={_fmtf(top3_pf20)}  WR20={top3_wr20 or '—'}%  Avg+60d={_fmt(top3_avg60)}")
    A(f"- rank1(最高位)スキップ: **{verdict_rank0}**")
    A(f"  - PF20={_fmtf(rank0_pf20)}  PF60={_fmtf(rank0_pf60)}")

    A("")
    A("### (3) レバ1.1xの価値")
    r11 = lev_rescued["1.1x"]
    st11 = _stats(r11)
    verdict11 = _judge(st11.get("pf20"), st11.get("wr20"), st11.get("avg60"))
    A(f"- 救済件数: {len(r11)} / 全スキップ{total_skipped}件の{len(r11)/max(1,total_skipped)*100:.1f}%")
    A(f"- **{verdict11}**")
    A(f"  - PF20={_fmtf(st11['pf20'])}  WR20={st11['wr20'] or '—'}%  Avg+60d={_fmt(st11['avg60'])}")

    A("")
    A("### (4) レバ1.3xの価値")
    r13 = lev_rescued["1.3x"]
    st13 = _stats(r13)
    verdict13 = _judge(st13.get("pf20"), st13.get("wr20"), st13.get("avg60"))
    A(f"- 救済件数: {len(r13)} / 全スキップ{total_skipped}件の{len(r13)/max(1,total_skipped)*100:.1f}%")
    A(f"- **{verdict13}**")
    A(f"  - PF20={_fmtf(st13['pf20'])}  WR20={st13['wr20'] or '—'}%  Avg+60d={_fmt(st13['avg60'])}")

    A("")
    A("### (5) 総合判断")
    any_positive = any(
        (st.get("pf20") or 0) > 1.2
        for lbl, st in [
            ("rank0", st_rank0), ("top3", st_top3),
            ("1.1x", _stats(lev_rescued["1.1x"])),
            ("1.3x", _stats(lev_rescued["1.3x"])),
        ]
    )
    if total_skipped == 0:
        A("**資金不足スキップ = 0件。現物3Mでは資金制約は発生していない。**")
        A("レバによる銘柄品質向上効果は存在しない（機会そのものが存在しないため）。")
        A("C1 レバ1.3x の価値は「既存ポジション×1.3の単純レバ効果」のみ。")
    elif any_positive:
        A("**スキップ銘柄の期待値がプラス (PF>1.2) → レバによる品質向上効果が存在する。**")
        A("C1 レバ導入に「機会損失回収＋品質向上」の両面的な価値がある可能性。")
        A("ただし借入コスト・2022 Bear時のDD拡大リスクを含めた総合評価が必要。")
    else:
        A("**スキップ銘柄の期待値は採用基準(PF>1.2 ∧ WR>50%)を満たさない。**")
        A("レバによる銘柄品質向上効果は限定的またはゼロ。")
        A("C1 レバ1.3x の価値は「既存ポジション×1.3の単純レバ効果」のみで評価すること。")

    report_text = "\n".join(lines)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(report_text, encoding="utf-8")
    print(f"\n  レポート → {OUT_FILE.relative_to(ROOT)}")

    # コンソールサマリー
    print("\n" + "=" * 72)
    print("  ★ 資金不足スキップ分析 サマリー")
    print("=" * 72)
    print(f"  総スキップ={total_skipped}  rank1={len(rank0)}  rank2={len(rank1)}  rank3={len(rank2)}")
    print("")
    print(f"  {'グループ':>12}  {'N':>5}  {'AvgRSR':>7}  {'Avg+20d':>9}  "
          f"{'WR20':>6}  {'PF20':>6}  {'Avg+60d':>9}  {'PF60':>6}")
    print("  " + "-" * 70)
    for grp, st in [("全体", st_all), ("rank1", st_rank0),
                    ("rank2", st_rank1), ("top3合計", st_top3)]:
        rsr_str = f"{st['avg_rsr']:.1f}" if st['avg_rsr'] else "—"
        print(f"  {grp:>12}  {st['n']:>5}  {rsr_str:>7}  "
              f"{_fmt(st['avg20']):>9}  {(str(st['wr20'])+'%') if st['wr20'] else '—':>6}  "
              f"{_fmtf(st['pf20']):>6}  {_fmt(st['avg60']):>9}  {_fmtf(st['pf60']):>6}")
    print("")
    print(f"  {'Leverage':>8}  {'救済N':>6}  {'Avg+20d':>9}  {'WR20':>6}  {'PF20':>6}  {'PF60':>6}")
    print("  " + "-" * 50)
    for lbl, L in LEVERAGE_CASES.items():
        rescued = lev_rescued[lbl]
        st = _stats(rescued)
        print(f"  {lbl:>8}  {len(rescued):>6}  {_fmt(st['avg20']):>9}  "
              f"{(str(st['wr20'])+'%') if st['wr20'] else '—':>6}  "
              f"{_fmtf(st['pf20']):>6}  {_fmtf(st['pf60']):>6}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
