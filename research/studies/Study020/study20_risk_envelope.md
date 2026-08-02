# Study20 Limited Live Risk Envelope

作成日: 2026-06-21  |  リスク検証のみ / 収益最適化禁止

**Strategy**: Study9 Case B (固定)  **Capital**: ¥1,800,000  **max_notional**: ¥1,200,000  **Allocation**: 10%  **Authority**: LIMITED_LIVE

**Period**: 2020-08-12 → 2020-09-24  (30 trading days, 1 complete trades)

---
## Executive Summary

| 指標 | 値 | 閾値 | 判定 |
|---|---|---|---|
| **risk_conformance_score** | **50.0/100** | — | — |
| max_trade_loss_R | -1.9360R | < 2.5R | ✅ |
| P50_trade_loss_R | -1.9360R | — | — |
| P95_trade_loss_R | -1.9360R | — | — |
| R_budget_violation_rate | 100.0% | ≤10% | ❌ |
| rolling_DD | ¥33,435 (1.95%) | ≤¥140,000 | ✅ |
| capital_activation_ratio | 8.64% | ≥30% | ⚠️ |
| exit_efficiency_avg | 100.0% | — | — |
| gap_loss_R_total | 0.0000R | — | — |
| overnight_loss_R_total | 0.0000R | — | — |
| n_complete_trades | 1 | ≥3 (有意) | ⚠️ |
| n_winning / n_losing | 0 / 1 | — | — |

**production_decision: GO_HOLD**

---
## Risk Attribution (Loss Source)

| loss_source | 件数 | 説明 |
|---|---|---|
| signal_failure | 1 (100%) | RSRシグナルが予測失敗（モメンタム崩壊） |
| market_gap | 0 (0%) | ギャップダウン優位（overnight gap > 2%） |
| execution_delay | 0 (0%) | 約定遅延によるスリッページ超過 |
| lot_constraint | 0 (0%) | 最小ロット制約によるサイズ縮小 |
| normal_variance | 0 (0%) | 通常分散範囲内の損失（< 1R） |

---
## Trade Log

| # | 銘柄 | Entry | Exit | 保有日 | Entry価格 | Exit価格 | PnL | R | violation | loss_source |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4021.T | 2020-08-12 | 2020-08-18 | 4d | ¥5,606 | ¥5,395 | ¥-43,410 | -1.9360R | ⚠️ YES | signal_failure |

### Exit Efficiency & Gap Metrics

| # | 銘柄 | MFE% | MAE% | exit_eff% | gap_loss_R | max_gap% |
|---|---|---|---|---|---|---|
| 1 | 4021.T | +0.00% | -15.66% | 100.0% | 0.0000R | 0.00% |

---
## Infrastructure Monitoring

| 指標 | 値 | 判定 |
|---|---|---|
| cash_truth_gap | 0 | ✅ |
| position_truth_gap | 0 | ✅ |
| reconciliation_error | 0 | ✅ |
| restart_restore_ms | 32256ms | ✅ |
| broker_disconnect_count | 0 | — |
| unexpected_position | 0 | ✅ |
| stop_condition_fired | — なし | ✅ |

---
## Promotion Criteria

| 条件 | 要件 | 判定 |
|---|---|---|
| R_budget_violation_rate_le10pct | ≤ 10% | ❌ FAIL |
| rolling_DD_le_study15b_p95 | ≤ ¥140,000 | ✅ PASS |
| cash_truth_gap_zero | = 0 | ✅ PASS |
| position_truth_gap_zero | = 0 | ✅ PASS |
| reconciliation_error_zero | = 0 | ✅ PASS |
| unexpected_position_zero | = 0 | ✅ PASS |

**昇格基準: 5/6 PASS**

---
## 最終判定

| 指標 | 値 |
|---|---|
| risk_conformance_score | **50.0/100** |
| max_trade_loss_R | -1.9360R |
| R_budget_violation_rate | 100.0% |
| rolling_DD | ¥33,435 |
| loss_source_breakdown | {'signal_failure': 1, 'market_gap': 0, 'execution_delay': 0, 'lot_constraint': 0, 'normal_variance': 0} |
| **production_decision** | **GO_HOLD** |
