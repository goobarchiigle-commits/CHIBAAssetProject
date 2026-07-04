"""
study74b_diagnostics_2026-07-04.py
Study74確定後の統合分析 — Part2(Study78 DD Attribution拡張+Worst10 DD Episode) +
                        Part3(Study74B: 候補不足構造分析・Study75とは別名称)

ユーザー指示（2026-07-04）: 「新規BTは必要最小限にする」「既存Research Assetsを最大限再利用する」。
本スクリプトは**唯一の新規BT**（FULL 2018-2025・CURRENT・M1適用後）で、これまで永続化していなかった
以下の既存エンジン計装（Study41/42/45/53で元々実装済み・新規改修ゼロ）を取得する:
  equity_curve / drawdown_curve（Worst10 DD Episode検出に必須・従来JSON化していなかった）
  _skip_detail / _rejected_by_lot_detail / _missed_cands（候補品質・見送り理由ランキング用）
  q_idle_days / q1-q3 idle attribution（cash滞留原因用）
  days_at_max_positions / cap_saturation_rate_pct（max_positions到達率用）

このJSON1本の抽出結果を使って Part2・Part3 双方を構築する（BTはこの1回のみ）。

⚠ 命名注意: 本Part3の内容はロードマップ既存の「Study75(J-Quants survivorship-free universe)」
とは別物。番号衝突を避けるため本ファイル・成果物では「Study74B」と呼称する。
"""
from __future__ import annotations

import json
import sys
import warnings
from datetime import date
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from src.backtest.snapshot_archaeology_202606 import build_common_dataset
import src.backtest.composite_alpha_bt as cab
from src.backtest.study78_ror_mc_sensitivity import (
    build_trade_ledger, get_active, PROD_CFG, CAPITAL, MIN_HOLD, FULL_START, FULL_END,
)

TODAY_STR = date.today().strftime("%Y-%m-%d")
BT_DIR = ROOT / "backtests"


def run_full(ds, sym_active_df):
    return cab.run_scenario(
        scenario="BASELINE",
        universe_raw=ds["universe_raw"], rsr_df=ds["rsr_df"], alpha_df=None,
        regime_df=ds["regime_df"], trade_syms=ds["trade_syms"], rsr_syms=ds["rsr_syms"],
        cfg=ds["base_cfg"], start=FULL_START, end=FULL_END, verbose=False,
        tech_matrices=ds["tech_matrices"], breadth_series=ds["breadth_series"],
        capital=CAPITAL, min_hold=MIN_HOLD, topix_close=ds["topix_close"],
        market_shock_mode="composite", rsr_exit_threshold=PROD_CFG["rsr_exit"],
        sym_active_df=sym_active_df,
        enable_simple_rsr_exit=True, enable_atr_trailing_prod=True,
        enable_multilayer_rsr=True, enable_atr_risk_sizing=False, enable_mtf_filter=False,
        sizing_mode="existing", exit_policy="A", addon_policy="D",
        addon_size_frac=0.25, addon_atr_mult=1.0,
    )


# ======================================================================
# Part2: Drawdown Episode検出 + 拡張Attribution
# ======================================================================

def detect_dd_episodes(dd: pd.Series, threshold: float = -0.03) -> list[dict]:
    """dd<threshold の連続区間を1エピソードとして検出。谷が最も深い順にソート。"""
    in_episode = False
    episodes = []
    start_i = None
    for i in range(len(dd)):
        v = dd.iloc[i]
        if v <= threshold and not in_episode:
            in_episode = True
            start_i = i
        elif v > threshold * 0.2 and in_episode:  # ほぼ0近くまで回復したら終了（ノイズでの分断防止に閾値20%を許容）
            in_episode = False
            episodes.append((start_i, i))
            start_i = None
    if in_episode:
        episodes.append((start_i, len(dd) - 1))

    out = []
    for s, e in episodes:
        window = dd.iloc[s:e + 1]
        trough_rel = int(window.values.argmin())
        trough_i = s + trough_rel
        depth = float(dd.iloc[trough_i])
        # 回復判定: trough以降 dd>=-0.001 になった最初の日
        recovered_i = None
        for j in range(trough_i, len(dd)):
            if dd.iloc[j] >= -0.001:
                recovered_i = j
                break
        out.append({
            "start_idx": s, "trough_idx": trough_i, "end_idx": recovered_i,
            "start_date": str(dd.index[s].date()), "trough_date": str(dd.index[trough_i].date()),
            "end_date": str(dd.index[recovered_i].date()) if recovered_i is not None else None,
            "depth_pct": round(depth * 100, 2),
            "duration_days": (recovered_i - s) if recovered_i is not None else (len(dd) - 1 - s),
            "recovered": recovered_i is not None,
        })
    out.sort(key=lambda x: x["depth_pct"])  # 最も深い(最も負)順
    return out


def annotate_episode_trades(ledger: list[dict], episode: dict) -> dict:
    start_d = pd.Timestamp(episode["start_date"])
    end_d = pd.Timestamp(episode["end_date"]) if episode["end_date"] else pd.Timestamp("2099-01-01")
    window_trades = [t for t in ledger if t["exit_date"] and start_d <= pd.Timestamp(t["exit_date"]) <= end_d]
    total_neg = sum(t["pnl"] for t in window_trades if t["pnl"] < 0) or -1.0
    annotated = []
    for t in window_trades:
        if t["pnl"] >= 0:
            continue
        annotated.append({
            "symbol": t["symbol"], "entry_date": t["entry_date"], "exit_date": t["exit_date"],
            "holding_days": t["holding_days"], "pnl": t["pnl"], "r_multiple": t["r_multiple"],
            "dd_contribution_pct": round(t["pnl"] / total_neg * 100, 2),
            "entry_type": t["entry_type"], "entry_atr_pct": t["entry_atr_pct"], "entry_rsr": t["entry_rsr"],
            "exit_policy": t["exit_policy"], "addon_received": t["addon_received"], "exit_reason": t["exit_reason"],
        })
    annotated.sort(key=lambda x: x["pnl"])
    return {
        **episode,
        "contributing_trade_count": len(window_trades),
        "loss_trade_count": len(annotated),
        "contributing_trades": annotated,
    }


# ======================================================================
# Part3 (Study74B): 候補不足構造分析
# ======================================================================

def candidate_shortage_analysis(raw: dict, ledger: list[dict], ds: dict) -> dict:
    skip_detail = raw.get("_skip_detail", [])
    rejected_lot = raw.get("_rejected_by_lot_detail", [])
    missed_cap = raw.get("_missed_cands", [])
    avg_candidates = raw.get("avg_candidates", 0)
    n_days = len(raw["equity_curve"])

    # 見送り理由ランキング
    reason_counts = {"LOT_REJECT": len(rejected_lot), "CAP_MISS": len(missed_cap)}
    for s in skip_detail:
        r = s.get("reason", "unknown")
        reason_counts[r] = reason_counts.get(r, 0) + 1
    reason_ranking = sorted(reason_counts.items(), key=lambda kv: -kv[1])

    # 候補品質: 見送られた候補 vs 実際に採用された候補のRSR/alpha分布
    def stats(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return {}
        return {"mean": round(float(np.mean(vals)), 2), "median": round(float(np.median(vals)), 2),
                "p25": round(float(np.percentile(vals, 25)), 2), "p75": round(float(np.percentile(vals, 75)), 2)}

    missed_rsr = [c.get("rsr") for c in missed_cap]
    missed_alpha = [c.get("alpha") for c in missed_cap]
    missed_rank = [c.get("rank") for c in missed_cap]
    traded_rsr = [t.get("entry_rsr") for t in ledger]

    # rank別ヒストグラム（候補同士競合: 上位ランクなのに見送られた件数）
    rank_hist = {}
    for r in missed_rank:
        if r is None: continue
        bucket = "rank0(最上位)" if r == 0 else ("rank1-2" if r <= 2 else "rank3+")
        rank_hist[bucket] = rank_hist.get(bucket, 0) + 1

    # 市場局面別（regime_df risk_off）: missed_cap日付の局面内訳
    regime_df = ds.get("regime_df")
    regime_breakdown = {"risk_off": 0, "normal": 0, "unknown": 0}
    if regime_df is not None:
        for c in missed_cap:
            d = pd.Timestamp(c["date"]) if c.get("date") else None
            if d is not None and d in regime_df.index:
                regime_breakdown["risk_off" if bool(regime_df.loc[d, "risk_off"]) else "normal"] += 1
            else:
                regime_breakdown["unknown"] += 1

    # 年別・月別 missed_cap 件数
    def by_period(items, fmt_len):
        agg = {}
        for c in items:
            d = c.get("date")
            if not d: continue
            agg[d[:fmt_len]] = agg.get(d[:fmt_len], 0) + 1
        return dict(sorted(agg.items()))

    missed_by_year = by_period(missed_cap, 4)
    missed_by_month = by_period(missed_cap, 7)

    # セクター別（missed_candsにsectorフィールドなし。trade_symsからsector引当）
    trade_syms = ds.get("trade_syms", {})
    missed_sector = {}
    for c in missed_cap:
        sec = trade_syms.get(c.get("symbol"), "不明")
        missed_sector[sec] = missed_sector.get(sec, 0) + 1

    # cash滞留原因（Study45 Q1-Q3既存計装の再掲）
    cash_idle_cause = {
        "q_idle_days": raw.get("q_idle_days"),
        "q1_idle_when_winner_pct": raw.get("q1_idle_when_winner_pct"),
        "q2_idle_days_with_winner_pct": raw.get("q2_idle_days_with_winner_pct"),
        "q3_deployable_idle_cash_avg_pct": raw.get("q3_deployable_idle_cash_avg_pct"),
    }

    return {
        "avg_candidates_per_day": avg_candidates,
        "candidate_shortage_rate_pct": round(max(0.0, (3 - avg_candidates) / 3 * 100), 1),
        "reason_ranking": reason_ranking,
        "max_positions_saturation": {
            "days_at_max_positions": raw.get("days_at_max_positions"),
            "cap_saturation_rate_pct": raw.get("cap_saturation_rate_pct"),
            "total_days": n_days,
        },
        "candidate_quality": {
            "missed_candidates_rsr": stats(missed_rsr), "missed_candidates_alpha": stats(missed_alpha),
            "traded_entries_rsr": stats(traded_rsr),
        },
        "candidate_competition_rank_histogram": rank_hist,
        "missed_by_regime": regime_breakdown,
        "missed_by_year": missed_by_year,
        "missed_by_month_sample": dict(list(missed_by_month.items())[:12]),
        "missed_by_sector": dict(sorted(missed_sector.items(), key=lambda kv: -kv[1])),
        "cash_idle_cause": cash_idle_cause,
        "opportunity_loss_reference": "Study53(2026-06-28): 候補ゼロ日66.4%・機会損失¥19k/年(参照値・本Studyでの再計測ではない)",
    }


def main():
    print("=" * 80)
    print("  Study74B診断 — Part2(DD Episode拡張) + Part3(候補不足構造分析)")
    print(f"  Date: {TODAY_STR}  (唯一の新規BT: FULL 2018-2025 CURRENT M1適用後)")
    print("=" * 80)

    ds = build_common_dataset(FULL_END)
    all_syms = list(ds["trade_syms"].keys())
    act = get_active(ds, all_syms, FULL_START, FULL_END)

    print("\n[RUN] Production FULL run（唯一の新規BT）...")
    raw = run_full(ds, act)
    print(f"  CAGR={raw['cagr']:+.2f}%  MaxDD={raw['max_dd']:.2f}%  Trades={raw['n_trades']}")

    ledger = build_trade_ledger(raw)
    print(f"  台帳: {len(ledger)}トレード（Study78 trade_dataset.jsonと同一手法で再構築）")

    # ── Part2: DD Episode検出 ────────────────────────────────────────────
    print("\n[Part2] Drawdown Episode検出...")
    dd_curve = raw["drawdown_curve"]
    episodes = detect_dd_episodes(dd_curve, threshold=-0.03)
    print(f"  検出エピソード数: {len(episodes)}（閾値-3%）")
    worst10 = episodes[:10]
    worst10_annotated = [annotate_episode_trades(ledger, ep) for ep in worst10]
    for i, ep in enumerate(worst10_annotated, 1):
        print(f"  #{i}: {ep['start_date']}~{ep['trough_date']}~{ep['end_date']}  depth={ep['depth_pct']}%  "
              f"duration={ep['duration_days']}d  loss_trades={ep['loss_trade_count']}")

    # ── Part3: 候補不足構造分析 ───────────────────────────────────────────
    print("\n[Part3] 候補不足構造分析（Study74B）...")
    shortage = candidate_shortage_analysis(raw, ledger, ds)
    print(f"  平均候補数/日={shortage['avg_candidates_per_day']}  候補不足率={shortage['candidate_shortage_rate_pct']}%")
    print(f"  見送り理由ランキング: {shortage['reason_ranking']}")
    print(f"  max_positions到達率: {shortage['max_positions_saturation']['cap_saturation_rate_pct']}%")

    # ── 保存 ─────────────────────────────────────────────────────────────
    dd_out = {
        "date": TODAY_STR, "source": "study74b_diagnostics (唯一の新規BT: FULL 2018-2025 CURRENT)",
        "detection_threshold_pct": -3.0, "total_episodes_detected": len(episodes),
        "worst10_drawdown_episodes": worst10_annotated,
    }
    dd_path = BT_DIR / f"study78_worst10_dd_episodes_{TODAY_STR}.json"
    with open(dd_path, "w", encoding="utf-8") as f:
        json.dump(dd_out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[OUTPUT] {dd_path}")

    shortage_out = {"date": TODAY_STR, "study": "Study74B_candidate_shortage_structural_analysis",
                     "note": "roadmap既存Study75(J-Quants survivorship-free)とは別物。番号衝突回避のためStudy74Bと呼称。",
                     "analysis": shortage}
    shortage_path = BT_DIR / f"study74b_candidate_shortage_{TODAY_STR}.json"
    with open(shortage_path, "w", encoding="utf-8") as f:
        json.dump(shortage_out, f, ensure_ascii=False, indent=2, default=str)
    print(f"[OUTPUT] {shortage_path}")


if __name__ == "__main__":
    main()
