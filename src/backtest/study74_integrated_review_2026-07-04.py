"""
study74_integrated_review_2026-07-04.py
Study74確定後の統合分析 — Part1: 制約分解の"改善可能/構造限界"分類 + Capacity Curve可視化

新規BTなし。既存JSON（study74_capital_scaling_2026-07-04.json /
study74b_candidate_shortage_2026-07-04.json）のみを再集計・可視化する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "MS Gothic"
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[2]
BT_DIR = ROOT / "backtests"
TODAY_STR = "2026-07-04"

d74 = json.load(open(BT_DIR / f"study74_capital_scaling_{TODAY_STR}.json", encoding="utf-8"))
d74b = json.load(open(BT_DIR / f"study74b_candidate_shortage_{TODAY_STR}.json", encoding="utf-8"))

CAPS = ["3000000", "10000000", "20000000", "30000000"]
CAP_LABELS = ["3M", "10M", "20M", "30M"]

matrix = d74["capital_config_matrix"]
capacity = d74["part_b_capacity_analysis"]
waterfall = d74["part_a_constraint_waterfall"]

# ======================================================================
# 制約分類テーブル
# ======================================================================
print("=" * 80)
print("  制約分類: 改善可能 vs 構造限界")
print("=" * 80)

classification = []

# lot丸め
lot_deltas = [waterfall[c]["delta_cagr"]["lot"] for c in CAPS]
classification.append({
    "constraint": "lot丸め", "category": "改善可能（資本で解消済み）",
    "evidence": f"delta_cagr(3M→30M)={lot_deltas} / lot_shortage_rate: "
                f"{[capacity[c]['lot_shortage_rate_pct'] for c in CAPS]}",
    "verdict": "¥20M以降で完全解消。資本拡大で自然に改善する唯一の制約。",
})

# max_positions
maxpos_deltas = [waterfall[c]["delta_cagr"]["max_positions"] for c in CAPS]
missed_counts = [matrix[c]["CURRENT"]["is"]["missed_by_cap_count"] for c in CAPS]
classification.append({
    "constraint": "max_positions=3", "category": "構造限界（PARAMS_LOCKED・恒久閉鎖#11）",
    "evidence": f"delta_cagr(3M→30M)={maxpos_deltas}（¥20M以降は解除するとCAGR悪化） / "
                f"missed_by_cap_count(3M→30M)={missed_counts}（資本を上げても減らない）",
    "verdict": "資本非依存の構造限界。緩和は変更禁止パラメータのため対象外。",
})

# symbol_cap
symcap_deltas = [waterfall[c]["delta_cagr"]["symbol_cap"] for c in CAPS]
classification.append({
    "constraint": "symbol_cap(0.40)", "category": "非該当（そもそも非拘束）",
    "evidence": f"delta_cagr(3M→30M)={symcap_deltas}（全資本水準で0.00pp）",
    "verdict": "現行設計で十分な余裕があり、資本規模によらず一切効いていない。",
})

# candidate不足
shortage = d74b["analysis"]
classification.append({
    "constraint": "candidate不足", "category": "構造限界（資本では解決しない）",
    "evidence": f"平均候補数/日={shortage['avg_candidates_per_day']}（候補不足率{shortage['candidate_shortage_rate_pct']}%） / "
                f"見送り理由ランキング1位=CAP_MISS({shortage['reason_ranking'][0][1]}件、圧倒的多数)",
    "verdict": "資本を増やしても候補の絶対数は増えない。Study74Bで見送り理由の9割近くがCAP_MISS(スロット競合)と確認 — "
               "真因はシグナル生成側（ユニバース・エントリー条件）にあり、Study75/76/77/81の領域。",
})

# cash滞留
idle_ratios = [capacity[c]["cash_idle_ratio_pct"] for c in CAPS]
classification.append({
    "constraint": "cash滞留", "category": "構造限界の帰結（独立制約ではない）",
    "evidence": f"現金滞留率(3M→30M)={idle_ratios}（87.6-90.9%でほぼ一定） / "
                f"q1_idle_when_winner_pct（勝ち候補があるのに滞留）={shortage['cash_idle_cause']['q1_idle_when_winner_pct']}%",
    "verdict": "candidate不足・max_positions=3の結果として生じる従属指標。独立に改善する対象ではない。",
})

# entry頻度
n_trades = [matrix[c]["CURRENT"]["is"]["n_trades"] for c in CAPS]
classification.append({
    "constraint": "entry頻度", "category": "構造限界（資本に対して不変）",
    "evidence": f"IS期間 n_trades(3M→30M)={n_trades}（263→264→257→258、資本を10倍にしてもほぼ一定）",
    "verdict": "取引頻度はシグナル生成条件(RSR/モメンタム)が支配し資本には反応しない。",
})

for c in classification:
    print(f"\n■ {c['constraint']} — 【{c['category']}】")
    print(f"  根拠: {c['evidence']}")
    print(f"  結論: {c['verdict']}")

# ======================================================================
# Capacity Curve（PNG）
# ======================================================================
print("\n[CHART] Capacity Curve 生成中...")

caps_m = [3, 10, 20, 30]
cagr_vals = [matrix[c]["CURRENT"]["is"]["cagr"] for c in CAPS]
inv_ratio = [capacity[c]["avg_investment_ratio_pct"] for c in CAPS]
cash_ratio = [capacity[c]["cash_idle_ratio_pct"] for c in CAPS]
shortage_rate = [max(0.0, (3 - matrix[c]["CURRENT"]["is"]["avg_candidates"]) / 3 * 100) for c in CAPS]
fill_rate = [capacity[c]["position_fill_rate_pct"] for c in CAPS]

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel("資本（百万円）")
ax1.set_ylabel("CAGR（%）", color="tab:red")
l1 = ax1.plot(caps_m, cagr_vals, "o-", color="tab:red", label="CAGR(IS)", linewidth=2)
ax1.tick_params(axis="y", labelcolor="tab:red")
ax1.set_ylim(0, max(cagr_vals) * 1.5)

ax2 = ax1.twinx()
ax2.set_ylabel("% (投資率/現金滞留率/候補不足率/Position充足率)")
l2 = ax2.plot(caps_m, inv_ratio, "s--", color="tab:blue", label="平均投資率")
l3 = ax2.plot(caps_m, cash_ratio, "^--", color="tab:gray", label="現金滞留率")
l4 = ax2.plot(caps_m, shortage_rate, "d--", color="tab:orange", label="候補不足率")
l5 = ax2.plot(caps_m, fill_rate, "v--", color="tab:green", label="Position充足率")
ax2.set_ylim(0, 100)

lines = l1 + l2 + l3 + l4 + l5
labels = [ln.get_label() for ln in lines]
ax1.legend(lines, labels, loc="center right")
plt.title("Study74 Capacity Curve（資本×CAGR・投資率・現金滞留・候補不足・Position充足）")
plt.xticks(caps_m, CAP_LABELS)
plt.tight_layout()

chart_path = BT_DIR / f"study74_capacity_curve_{TODAY_STR}.png"
plt.savefig(chart_path, dpi=150, bbox_inches="tight")
print(f"  [OUTPUT] {chart_path}")

# ======================================================================
# 保存
# ======================================================================
output = {
    "date": TODAY_STR, "study": "Study74_integrated_review_part1",
    "constraint_classification": classification,
    "capacity_curve_data": {
        "capital_million": caps_m, "cagr_is": cagr_vals, "avg_investment_ratio_pct": inv_ratio,
        "cash_idle_ratio_pct": cash_ratio, "candidate_shortage_rate_pct": shortage_rate,
        "position_fill_rate_pct": fill_rate,
    },
    "chart_file": str(chart_path.name),
}
out_path = BT_DIR / f"study74_integrated_review_{TODAY_STR}.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)
print(f"[OUTPUT] {out_path}")
