# Study15B Access Elasticity Confirmation

作成日: 2026-06-21  |  監査のみ / 変更禁止

**Strategy**: Study9 Case B (固定)  **Reference**: Virtual Fractional Calmar=1.496

---
## Executive Summary

| 項目 | 値 |
|---|---|
| **root_cause** | **ACCESS** |
| **efficient_capital** | ¥1,500,000 |
| plateau_region | ¥1,200,000+ |
| capital_margin | ¥300,000 |
| recommend_live_capital | ¥1,800,000 |
| **decision** | **GO** |
| next_phase | Study16 Order Authority Gate |

---
## 参照: Virtual Fractional (lot=1, cap=¥400k)

| 指標 | 値 |
|---|---|
| fillable | 100.0% |
| calmar (avg fold) | 1.496 |
| CAGR | +46.95% |
| trade_count | 35 |

---
## Cases A–F: Capital Elasticity

| Case | cap | fillable | skip | calmar | CAGR | access_rec | cap_eff | P95_yen | 基準 |
|---|---|---|---|---|---|---|---|---|---|
| A| ¥1000k| 89.7% ❌| 10.3% ❌| 1.307 ❌| +39.20%| 87.4% ❌| 1.307| ▲¥432,055| — |
| B| ¥1200k| 89.7% ❌| 10.3% ❌| 1.439 ✅| +43.51%| 96.2% ✅| 1.199| ▲¥540,151| — |
| C| ¥1350k| 100.0% ✅| 0.0% ✅| 1.344 ❌| +40.72%| 89.8% ❌| 0.995| ▲¥598,268| — |
| D| ¥1500k| 100.0% ✅| 0.0% ✅| 1.449 ✅| +42.19%| 96.9% ✅| 0.966| ▲¥685,726| 🎯 |
| E| ¥1800k| 100.0% ✅| 0.0% ✅| 1.428 ✅| +43.29%| 95.5% ✅| 0.794| ▲¥811,581| 🎯 |
| F| ¥2200k| 100.0% ✅| 0.0% ✅| 1.502 ✅| +45.39%| 100.4% ✅| 0.683| ▲¥1,009,158| 🎯 |

> 採用基準: fillable≥95% / skip≤5% / access_rec≥90% / calmar≥1.427(=95%×best)

---
## 弾力性・停止判定

| Case | cap | Δfillable | ΔCalmar | marginal_access | marginal_alpha | 停止 |
|---|---|---|---|---|---|---|
| A| ¥1000k| -10.3pp| -0.189| —| —|  |
| B| ¥1200k| +0.0pp| +0.132| 0.0000| 0.0000|  |
| C| ¥1350k| +10.3pp| -0.095| 0.6840| -0.9259|  |
| D| ¥1500k| +0.0pp| +0.105| 0.0000| 0.0000|  |
| E| ¥1800k| +0.0pp| -0.021| 0.0000| 0.0000| 🛑 STOP |
| F| ¥2200k| +0.0pp| +0.074| 0.0000| 0.0000| 🛑 STOP |

---
## Lot/Price Binding Rate

| Case | cap | n_skip | lot_binding | price_binding | top skip 銘柄 |
|---|---|---|---|---|---|
| A| ¥1000k| 4| 100%| 0%| 8035.T |
| B| ¥1200k| 4| 100%| 0%| 8035.T |
| C| ¥1350k| 0| 0%| 0%|  |
| D| ¥1500k| 0| 0%| 0%|  |
| E| ¥1800k| 0| 0%| 0%|  |
| F| ¥2200k| 0| 0%| 0%|  |

---
## Missed Alpha (Counterfactual)

| Case | cap | n_missed | avg_virt_ret | win_rate | lot_bound_avg | price_bound_avg |
|---|---|---|---|---|---|---|
| A| ¥1000k| 4| +1.08%| 75.0%| +1.08%| +0.00% |
| B| ¥1200k| 4| +1.08%| 75.0%| +1.08%| +0.00% |
| C| ¥1350k| 0| +0.00%| 0.0%| +0.00%| +0.00% |
| D| ¥1500k| 0| +0.00%| 0.0%| +0.00%| +0.00% |
| E| ¥1800k| 0| +0.00%| 0.0%| +0.00%| +0.00% |
| F| ¥2200k| 0| +0.00%| 0.0%| +0.00%| +0.00% |

---
## Top10 Missed Trades (Case A ¥1.0M)

| symbol   | skip_date   |   entry_price |   virt_ret_pct |   hold_days | exit_reason   |   min_lot_cost | lot_binding   |
|:---------|:------------|--------------:|---------------:|------------:|:--------------|---------------:|:--------------|
| 8035.T   | 2024-05-16  |         36150 |           3.65 |           3 | RSR_EXIT      |      3.615e+06 | True          |
| 8035.T   | 2024-05-15  |         36600 |           2.38 |           4 | RSR_EXIT      |      3.66e+06  | True          |
| 8035.T   | 2024-05-17  |         35800 |           0.28 |           3 | RSR_EXIT      |      3.58e+06  | True          |
| 8035.T   | 2024-05-20  |         36640 |          -1.97 |           3 | RSR_EXIT      |      3.664e+06 | True          |

---
## Access Loss Breakdown

| Case | access_recovery | access_loss | lot_driven_loss | capital_gap_yen |
|---|---|---|---|---|
| A| 87.4%| 12.6%| 12.6%| ¥500,000 |
| B| 96.2%| 3.8%| 3.8%| ¥300,000 |
| C| 89.8%| 10.2%| 0.0%| ¥150,000 |
| D| 96.9%| 3.1%| 0.0%| ¥0 |
| E| 95.5%| 4.5%| 0.0%| ¥0 |
| F| 100.4%| -0.4%| -0.0%| ¥0 |

---
## Capital Frontier Summary

| capital | fillable | Calmar | cap_efficiency | access_recovery | 評価 |
|---|---|---|---|---|---|
| ¥1,000,000| 89.7%| 1.307| 1.3068| 87.4%| ⚠ fillable不足 |
| ¥1,200,000| 89.7%| 1.439| 1.1988| 96.2%| ⚠ fillable不足 |
| ¥1,350,000| 100.0%| 1.344| 0.9954| 89.8%| ⚠ calmar不足 |
| ¥1,500,000| 100.0%| 1.449| 0.9659| 96.9%| ✅ 採用基準達成 |
| ¥1,800,000| 100.0%| 1.428| 0.7936| 95.5%| ✅ 採用基準達成 |
| ¥2,200,000| 100.0%| 1.502| 0.6829| 100.4%| ✅ 採用基準達成 |

---
## Plateau Analysis

**plateau_region: ¥1,200,000+**

平坦化点: marginal_access < 0.02 (Δfillable/Δ¥M)
この水準以上の資本追加は fillable 改善が飽和 → calmar への追加効果も限界的

---
## 最終判定

| 項目 | 値 |
|---|---|
| root_cause | **ACCESS** |
| efficient_capital | ¥1,500,000 |
| plateau_region | ¥1,200,000+ |
| capital_margin | ¥300,000 (=20% buffer) |
| recommend_live_capital | **¥1,800,000** |
| virtual_frac_calmar (ref) | 1.496 |
| best_realized_calmar | 1.502 |
| access_recovery@best | 100.4% |
| decision | **GO** |
| next_phase | Study16 Order Authority Gate |