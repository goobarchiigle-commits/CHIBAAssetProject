# Entry Velocity WF Confirmation

作成日: 2026-06-20  |  研究専用 / 実装変更禁止

仮説: 高速度エントリーが低品質 trade を混入させ DD と CAP 制約を生んでいた

採用条件: WF>=4/5 AND ΔCAGR≥+0.3pp AND ΔDD≤+1.5pp
         AND selection_gain≥1.10 AND bad_trade_removed≥1.5×opportunity_loss

CAUTION: winner_removed_rate>30%

**recommend_promotion: REJECT**  |  **recommend_live_shadow: FAIL**  |  best_filter: **C**

---
## 1. Filter Quality

| 指標 | 値 | 解釈 |
|---|---|---|
| filter_precision | 0.8421 | removed trades のうち損失の割合 |
| filter_recall    | 0.7619  | 全損失 trades のうち捕捉した割合 |
| winner_removed_rate | 85.00% | top20利益 trade のうち除外率 |
| recommend_promotion | **REJECT** | PROMOTE基準 |
| recommend_live_shadow | **FAIL** | CAUTION考慮後 |

⚠ CAUTION: winner_removed_rate=85.0% > 30% → 高パフォーマンス trade を除外している可能性


---
## 2. WF サマリ (Case A vs B vs C)

| Case | WF | ΔCAGR | ΔDD | ΔCalmar | avg_PF_X | sel_gain | BTR | OPL | PROMOTE |
|---|---|---|---|---|---|---|---|---|---|
| B (GLOBAL_FILTER slope≤5) | 1/5 | +4.05pp | +2.97pp | +0.796 | 2.267 | 7.0959 | 0.6663 | 0.3337 | REJECT |
| C (SOFT_PENALTY λ=2.0) | 0/5 | +9.96pp | +2.58pp | +0.504 | 1.936 | 14.2473 | 0.6833 | 0.3167 | REJECT |

---
## 3. Fold 詳細 (Case B: GLOBAL_FILTER)

| Fold | Regime | CAGR_A | CAGR_B | ΔCAGR | ΔDD | sel_gain | BTR | n_removed | blocked_ret20 | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold1 | Bull (+13.3%) | +32.10% | -2.34% | -34.44pp | +3.87pp | 0.4629 | 0.9706 | 34 | -15.34% | ❌ |
| Fold2 | Bear (-4.7%) | -6.33% | +4.00% | +10.33pp | -1.80pp | 7.2648 | 0.7778 | 9 | -14.68% | ✅ |
| Fold3 | Bull (+29.8%) | +2.08% | +29.65% | +27.57pp | +6.23pp | 0.9697 | 0.7500 | 4 | -3.21% | ❌ |
| Fold4 | Bull (+19.7%) | -6.03% | +5.80% | +11.83pp | +5.85pp | 26.5622 | 0.3333 | 3 | +4.82% | ❌ |
| Fold5 | Bull (+26.8%) | +3.57% | +8.51% | +4.94pp | +0.68pp | 0.2201 | 0.5000 | 2 | +0.28% | ❌ |

---
## 4. Fold 詳細 (Case C: SOFT_PENALTY λ=2.0)

| Fold | Regime | CAGR_A | CAGR_C | ΔCAGR | ΔDD | sel_gain | BTR | n_removed | pass |
|---|---|---|---|---|---|---|---|---|---|
| Fold1 | Bull (+13.3%) | +32.10% | +32.40% | +0.30pp | -0.85pp | 1.0063 | 1.0000 | 2 | ❌ |
| Fold2 | Bear (-4.7%) | -6.33% | -5.43% | +0.90pp | -0.03pp | 1.0224 | 1.0000 | 3 | ❌ |
| Fold3 | Bull (+29.8%) | +2.08% | +7.34% | +5.26pp | +5.50pp | 0.6034 | 0.7500 | 4 | ❌ |
| Fold4 | Bull (+19.7%) | -6.03% | +34.01% | +40.04pp | +3.66pp | 68.4892 | 0.3333 | 3 | ❌ |
| Fold5 | Bull (+26.8%) | +3.57% | +6.86% | +3.29pp | +4.62pp | 0.1154 | 0.3333 | 3 | ❌ |

---
## 5. Attribution 詳細

> blocked_ret20: 除外された entry の fwd20 return (損失なら仮説支持)

| Case | blocked_ret20 | accepted_ret20 | bad_trade_removed | opportunity_loss | BTR/OPL ratio |
|---|---|---|---|---|---|
| B | -5.63% | -3.68% | 0.6663 | 0.3337 | 2.00 |
| C | -5.48% | -10.17% | 0.6833 | 0.3167 | 2.16 |

→ 弱支持: accepted vs blocked gap=+1.95pp (軽微)


---
## 6. Case D: MONITOR_ONLY 仮想監査

| 指標 | 値 |
|---|---|
| would_block count | 22 |
| would_keep count  | 54 |
| block_rate        | 28.9% |
| blocked slope min/mean/max | 7.1 / 17.5 / 36.6 |
| 仮想除外シンボル top5 | 6857.T(3), 6762.T(3), 9101.T(2), 9104.T(2), 6506.T(2) |

---
## 7. Final Recommendation

| 指標 | 値 |
|---|---|
| filter_precision | 0.8421 (84.2%) |
| filter_recall    | 0.7619 (76.2%) |
| best_filter      | **C** |
| recommend_promotion | **REJECT** |
| recommend_live_shadow | **FAIL** |
| CAUTION | winner_removed_rate=85.0% > 30% |

### REJECT

バインディング制約: WF=1/5 < 4 / ΔDD=+2.97pp > +1.5pp

保守推奨: slope≤5 フィルターを採用済み研究 (standalone) で維持
production 統合は再評価後