# Study15 Lot-Constrained Capacity Attribution Audit

作成日: 2026-06-21  |  監査のみ / 変更禁止 / live注文禁止

**Strategy**: Study9 Case B (entry/exit/signal完全固定)
**Purpose**: Study14 NO_GO (Lot Feasibility skip=31.5%) 原因分離

---
## Executive Summary

| 項目 | 値 |
|---|---|
| **root_cause** | **MIXED** |
| **decision** | **GO_LIMITED** |
| minimum_efficient_capital | ¥1,500,000 |
| expected_fillable | 100.0% |
| expected_alpha (vs VirtualFrac) | 44.4% (Case C) |
| capital_elasticity | +0.0234 pp / ¥100k |
| alpha_elasticity | +0.0033 Calmar / ¥100k |
| next_phase | Study16 Order Authority Gate (min_capital=¥1,500,000) |

---
## Case A: Baseline (cap=¥400k, LOT=100)

| 指標 | 値 | vs A |
|---|---|---|
| sl_cash_init | ¥400,000 | — |
| fillable_rate | 68.5% | — |
| forced_skip_rate | 31.5% | — |
| n_price_reject | 0 | — |
| avg_buy_price | ¥3,008 | — |
| avg_skip_price | ¥15,941 | — |
| avg_cagr (fold avg) | +40.68% | +40.68pp |
| avg_calmar | 1.392 | +1.392 |
| P95_DD_yen | ▲¥140,027 | — |
| fill_loss_avg | 14.6% | — |
| n_trades | 36 | — |
| win_rate | 41.7% | — |

**skip 上位銘柄 (lot_conc_top5=100.0%)**:
  - 8035.T
  - 4021.T
  - 6594.T
  - 6645.T
  - 6146.T

---
## Case B: Virtual Fractional (cap=¥400k, lot=1株)

| 指標 | 値 | vs A |
|---|---|---|
| sl_cash_init | ¥400,000 | — |
| fillable_rate | 100.0% | +31.5pp |
| forced_skip_rate | 0.0% | — |
| n_price_reject | 0 | — |
| avg_buy_price | ¥4,417 | — |
| avg_skip_price | ¥0 | — |
| avg_cagr (fold avg) | +46.95% | +6.27pp |
| avg_calmar | 1.496 | +0.104 |
| P95_DD_yen | ▲¥185,608 | — |
| fill_loss_avg | 0.1% | — |
| n_trades | 35 | — |
| win_rate | 37.1% | — |

**Δfillable A→B (lot制約除去効果): +31.5pp**
→ 単元制約が支配的 (LOT問題)

---
## Case C: Price Cap (cap=¥400k, LOT=100, 最小単元≤¥400k)

| 指標 | 値 | vs A |
|---|---|---|
| sl_cash_init | ¥400,000 | — |
| fillable_rate | 100.0% | +31.5pp |
| forced_skip_rate | 0.0% | — |
| n_price_reject | 50 | — |
| avg_buy_price | ¥2,244 | — |
| avg_skip_price | ¥0 | — |
| avg_cagr (fold avg) | +16.72% | -23.96pp |
| avg_calmar | 0.664 | -0.728 |
| P95_DD_yen | ▲¥182,823 | — |
| fill_loss_avg | 14.6% | — |
| n_trades | 33 | — |
| win_rate | 45.5% | — |

**Calmar比 C/B: 44.4%  (採用基準: ≥95%)**
→ 高単価銘柄が alpha に必須 (除外でCalmar低下)
**Δfillable A→C: +31.5pp** (price_rejects=50件)

---
## Case D: Capital Sweep (LOT=100)

| cap(¥) | fillable | Δfill | calmar | ΔCAGR | P95_yen | elasticity |
|---|---|---|---|---|---|---|
| ¥400,000 | 68.5% | +0.0pp | 1.392 | +0.00pp | ▲¥140,027 | — |
| ¥600,000 | 87.5% | +19.0pp | 1.694 | +4.14pp | ▲¥270,076 | +9.490 |
| ¥800,000 | 89.7% | +2.2pp | 1.358 | -7.21pp | ▲¥333,044 | +1.120 |
| ¥1,000,000 | 89.7% | +0.0pp | 1.307 | +1.59pp | ▲¥432,055 | +0.000 |
| ¥1,200,000 | 89.7% | +0.0pp | 1.439 | +4.31pp | ▲¥540,151 | +0.000 |
| ¥1,500,000 | 100.0% ✅| +10.3pp | 1.449 | -1.32pp | ▲¥685,726 | +3.420 |
| ¥1,800,000 | 100.0% ✅| +0.0pp | 1.428 | +1.10pp | ▲¥811,581 | +0.000 |

**avg_capital_elasticity: +0.0234 pp / ¥100k**
**min_efficient_capital: ¥1,500,000** (fillable≥90%を達成する最小資本)

---
## Case E: Capital Sweep + Price Cap

| cap(¥) | fillable | Δfill vs D | calmar | notes |
|---|---|---|---|---|
| ¥400,000 | 100.0% ✅| +31.5pp | 0.664 | price_cap isolates capital-only effect |
| ¥600,000 | 100.0% ✅| +12.5pp | 0.646 |  |
| ¥800,000 | 100.0% ✅| +10.3pp | 0.747 |  |
| ¥1,000,000 | 100.0% ✅| +10.3pp | 0.681 |  |
| ¥1,200,000 | 100.0% ✅| +10.3pp | 0.677 |  |
| ¥1,500,000 | 100.0% ✅| +0.0pp | 0.671 |  |
| ¥1,800,000 | 100.0% ✅| +0.0pp | 0.647 |  |

**avg Δ(E-D)=+10.7pp**: 価格制約によりfillable上昇 → 高単価排除が有効
→ 高単価銘柄が単元スキップを引き起こしている (LOT + PRICE混合)

---
## Counterfactual Access (スキップトレードの仮想収益)

n_skipped=17件  avg_virtual_ret=+2.95%  median=+0.28%  win_rate=58.8%  avg_hold=27.7d

**lost_alpha_from_skip** = +2.95% / +40.68% = 0.073
→ スキップされたトレードはプラス収益 — Lot制約が alpha を喪失させている

**スキップ上位5件 (仮想リターン降順):**

| symbol   | skip_date   |   entry_price |   virtual_ret_pct |   hold_days | exit_reason   |
|:---------|:------------|--------------:|------------------:|------------:|:--------------|
| 6594.T   | 2020-08-13  |          4537 |             35.77 |         136 | RSR_EXIT      |
| 6594.T   | 2020-08-18  |          4550 |             35.38 |         133 | RSR_EXIT      |
| 8035.T   | 2020-11-24  |         11350 |              6.7  |          22 | RSR_EXIT      |
| 8035.T   | 2024-05-16  |         36150 |              3.65 |           3 | RSR_EXIT      |
| 4021.T   | 2020-11-19  |          6260 |              2.88 |           3 | RSR_EXIT      |

---
## 原因分離サマリ

| 要因 | Δfillable | 重要度 | 結論 |
|---|---|---|---|
| LOT制約 (lot_size 100→1) | +31.5pp | 高 | 単元サイズが制限要因 |
| 高単価偏り (price_cap追加) | +10.7pp | 中 | 高単価銘柄が主因 |
| 資本不足 (cap 400k→min_eff) | +31.5pp | 高 | 資本増加で解決可能 |

**root_cause確定: MIXED**

---
## 最終判定

| 項目 | 値 |
|---|---|
| root_cause | **MIXED** |
| minimum_efficient_capital | ¥1,500,000 |
| capital_confidence | HIGH |
| expected_fillable@min | 100.0% |
| expected_alpha@min | 1.449 Calmar |
| expected_DD@min | ▲¥685,726 P95 |
| decision | **GO_LIMITED** |
| next_phase | Study16 Order Authority Gate (min_capital=¥1,500,000) |
| blocked_by | — |