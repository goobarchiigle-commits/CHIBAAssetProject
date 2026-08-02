# Study14 Standalone Pre-Live Audit

作成日: 2026-06-21  |  監査のみ / 実装変更禁止 / live注文禁止

**Strategy**: Study9 Case B  RSR[92,95) d90≤5 slope≤5 exit<90 hold≥3  MaxPos=1
**Capital**: fixed_cap=¥400,000 (直接sleeve)  **Governance**: annual_rebalance(¥750k target)

---
## Executive Summary

| 項目 | 値 |
|---|---|
| **最終判定** | **NO_GO** |
| blocked_by | Lot Feasibility (forced_skip_rate=31.5%) |
| rollback_rule | alpha_real_30d<80% または fill_rate<90% |
| authority_required | true |
| next_phase | 最大寄与要因=Lot Feasibility → capital再検討 |

---
## Step1 Exit Attribution

counterfactual_hold: min(40d, p90_hold=47d) = **40d**

| 指標 | 値 | 判定閾値 | 判定 |
|---|---|---|---|
| median_exit_efficiency | 80.0% | ≥70% | ✅ |
| P10_exit_efficiency | -44.5% | ≥40% | ❌ IMMEDIATE_REVIEW |
| P90_exit_regret | 16.56pp | ≤15pp | ❌ |
| profit_left_avg | 12.69pp | — | — |
| tail_capture | 27.8% | — | — |
| avg_regret_per_day | 0.317pp/d | — | — |
| late_exit_penalty | -15.42pp | — | — |
| n_trades | 36 | — | — |

**Step1 判定: **FAIL** ❌**

→ Step2 Stop Audit を実行

---
## Step2 Stop Audit (条件付き)

Step1 FAIL により実施。

| 指標 | 値 |
|---|---|
| n_rsr_exits | 35 |
| n_rsr_loss | 20 |
| rsr_loss_rate | 57.1% |
| stop_cost_avg_yen | ¥-31,784 |
| stop_saved_pct_est | 35.0% |
| n_mkt_exits | 1 |

**結論: NO_CHANGE**

stop調整では exit品質の根本解決にならない。EXIT変更は禁止。

---
## Step3 Standalone DD Audit

capital: ¥400,000  bootstrap N=5,000 (5-fold resample)

| 指標 | 値 | 判定閾値 | 判定 |
|---|---|---|---|
| ruin_probability | 0.00% | <1.0% | ✅ |
| P95_DD_yen | ▲¥140,027 | ≤¥80,000 | ❌ |
| P99_DD_yen | ▲¥140,027 | ≤¥140,000 | ❌ |
| longest_underwater | 361d | ≤~90d | ❌ |
| capital_recovery_days | 166d | ≤60d | ❌ |
| ulcer_index | 17.464% | — | — |
| max_dd_realised | -35.23% | — | — |
| max_dd_yen | ▲¥461,000 | — | — |
| fold_cagr_avg | +40.68% | — | — |
| fold_dd_avg | -28.04% | — | — |

**Step3 判定: **FAIL** ❌**

---
## Step4 Lot Feasibility

sl_cash=¥400,000  LOT=100株  alloc=sl_cash×0.95

| 指標 | 値 | 判定閾値 | 判定 |
|---|---|---|---|
| fillable_rate | 68.5% | ≥90% | ❌ |
| median_idle | 11.9% | ≤20% | ✅ |
| forced_skip_rate | 31.5% | ≤10% | ❌ |
| n_buy_executed | 37 | — | — |
| n_forced_skip | 17 | — | — |
| avg_buy_price | ¥3,008 | — | — |
| avg_skip_min_lot_yen | ¥1,594,110 | — | — |
| cap_util_pct | 81.0% | — | — |
| eff_cap_util | 60.3% | — | — |

**Step4 判定: **FAIL** ❌**

⚠ 最小有効capital推奨: ~¥1,678,010 (avg_skip_min_lot÷0.95)
現行capital=¥400,000では多数の銘柄がLot制約でスキップされる

---
## Step5 Exit Execution Audit

注: Study12ベース推定値。live注文禁止。signal変更禁止。

| 指標 | BASE推定 | WORST推定 | 確認ライン |
|---|---|---|---|
| cancel_rate | 0.0% | 0.0% | — (成行・取消なし) |
| partial_fill_rate | 0.8% | 2.5% | 追跡要 |
| avg_slippage_bp (追加) | 5bp | 13bp | — |
| broker_reject_pct | 0.0% | 0.0% | ≤0.5% |
| signal_latency | 900s | 900s | — |
| capacity_pct_ADV | 0.0331% | — | — (問題なし) |
| mkt_impact_bp_est | 1.655bp | — | 無視可能 |
| avg_order_yen | ¥661,808 | — | — |

**確認のみ / 変更禁止**: kabu REST API 成行 寄り付き. 約定追跡=不要(成行). signal変更禁止.

---
## 最終判定

| ステップ | 判定 | 備考 |
|---|---|---|
| Step1 Exit Attribution | ❌ FAIL | median_eff=80.0% P90_regret=16.6pp |
| Step2 Stop Audit | — | NO_CHANGE |
| Step3 DD Audit | ❌ FAIL | P95=▲¥140,027 P99=▲¥140,027 |
| Step4 Lot Feasibility | ❌ FAIL | fillable=68.5% skip=31.5% |
| Step5 Execution Audit | ✅ 確認完了 | cancel=0% fill≥97.5% slip≤13bp |

### 最終判定: **NO_GO**
- blocked_by: Lot Feasibility (forced_skip_rate=31.5%)
- rollback_rule: alpha_realization_rolling_30d < 80% → alloc=0%即時縮小
- authority_required: **true** → Study15 Order Authority Gate
- next_phase: 最大寄与要因=Lot Feasibility → capital再検討
