# Study12 Deployment Readiness Gate

作成日: 2026-06-20  |  研究専用 / 実装変更禁止

Signal: Study9 Case B  RSR[92,95) d90∈[1,5] slope≤5 exit<90  MaxPos=1  Rebalance=annual(750k)
Broker: kabuステーション REST API localhost:18080  注文種別: 成行 寄り付き  ADV推定: 20億円

判定閾値: alpha_realization≥85.0% | fill_rate≥95.0% | exec_tax≥0.9 | reject≤0.5%

---
## 1. Execution シナリオ定義

| Scenario | entry_slip_bp | exit_slip_bp | fill_ratio | reject_per_N | partial_fill% | broker_reject% |
|---|---|---|---|---|---|---|
| **ideal** | 0bp | 0bp | 100.0% | — | 0.0% | 0.0% |
| **base** | 3bp | 2bp | 99.3% | 1/500 | 0.8% | 0.2% |
| **worst** | 8bp | 5bp | 97.5% | 1/200 | 2.0% | 0.5% |

注: backtest COST_ONE_WAY = 15.5bp 片道 (既計上)。上記は追加摩擦のみ。
signal_delay_sec = 900s (シグナル生成→寄り成行提出→約定まで)

---
## 2. 注文執行品質 Audit

| Metric | ideal | base | worst | 採用ライン |
|---|---|---|---|---|
| fill_rate % | 100.00 | 100.00 | 100.00 | ≥95% |
| broker_reject % | 0.00 | 0.00 | 0.00 | ≤0.5% |
| partial_fill % | 0.00 | 2.50 | 2.50 | — |
| avg_slippage_bp (追加) | 0.00 | 5.00 | 13.00 | — |
| lot_round_loss % | 28.34 | 27.95 | 27.74 | — |
| avg_order_size ¥ | 831497.00 | 822278.00 | 822689.00 | — |
| capacity_pct (ADV%) | 0.04 | 0.04 | 0.04 | — |
| cash_sync_error | 0 | 0 | 0 | ≈0 |
| signal_delay_sec | 0 (backtest) | 900s | 900s | — |
| n_orders (全期) | 40 | 40 | 40 | — |
| n_rejected | 0 | 0 | 0 | — |

---
## 3. Standalone 性能比較 (ideal vs exec simulation)

| Fold | Regime | CAGR_ideal | DD_ideal | Calmar_ideal | CAGR_base | DD_base | Calmar_base | CAGR_worst | DD_worst | Calmar_worst |
|---|---|---|---|---|---|---|---|---|---|---|
| Fold1 2021 | Bull (+13.3%) | +25.70% | -40.81% | 0.630 | +25.58% | -40.85% | 0.626 | +25.38% | -40.91% | 0.620 |
| Fold2 2022 | Bear (-4.7%) | +22.68% | -27.49% | 0.825 | +22.73% | -27.71% | 0.820 | +22.01% | -28.07% | 0.784 |
| Fold3 2023 | Bull (+29.8%) | +20.87% | -16.49% | 1.265 | +22.35% | -16.46% | 1.358 | +21.54% | -16.75% | 1.286 |
| Fold4 2024 | Bull (+19.7%) | +33.26% | -27.08% | 1.228 | +32.86% | -27.24% | 1.206 | +32.23% | -27.50% | 1.172 |
| Fold5 2025 | Bull (+26.8%) | +84.45% | -26.02% | 3.245 | +84.56% | -26.15% | 3.233 | +84.75% | -26.37% | 3.214 |
| **avg** | — | **+37.39%** | **-27.58%** | **1.439** | **+37.62%** | **-27.68%** | **1.449** | **+37.18%** | **-27.92%** | **1.415** |

---
## 4. Alpha 実現率 (live_like_CAGR / ideal_CAGR)

> 採用ライン ≥ 85%

| Fold | Regime | ideal_CAGR | base_αReal | worst_αReal |
|---|---|---|---|---|
| Fold1 2021 | Bull (+13.3%) | +25.70% | 99.5% ✅ | 98.8% ✅ |
| Fold2 2022 | Bear (-4.7%) | +22.68% | 100.2% ✅ | 97.0% ✅ |
| Fold3 2023 | Bull (+29.8%) | +20.87% | 107.1% ✅ | 103.2% ✅ |
| Fold4 2024 | Bull (+19.7%) | +33.26% | 98.8% ✅ | 96.9% ✅ |
| Fold5 2025 | Bull (+26.8%) | +84.45% | 100.1% ✅ | 100.4% ✅ |
| **avg** | — | **+37.39%** | **100.6% ✅** | **99.4% ✅** |

---
## 5. Execution Tax (Calmar_after / Calmar_before)

> 採用ライン ≥ 0.90

| Fold | Regime | ideal_Calmar | base_exTax | worst_exTax |
|---|---|---|---|---|
| Fold1 2021 | Bull (+13.3%) | 0.630 | 0.994 ✅ | 0.984 ✅ |
| Fold2 2022 | Bear (-4.7%) | 0.825 | 0.994 ✅ | 0.950 ✅ |
| Fold3 2023 | Bull (+29.8%) | 1.265 | 1.074 ✅ | 1.017 ✅ |
| Fold4 2024 | Bull (+19.7%) | 1.228 | 0.982 ✅ | 0.954 ✅ |
| Fold5 2025 | Bull (+26.8%) | 3.245 | 0.996 ✅ | 0.990 ✅ |
| **avg** | — | **1.439** | **1.007 ✅** | **0.984 ✅** |

---
## 6. Combined 性能 (production + governed standalone, exec sim)

| Fold | Regime | prod_CAGR | prod_DD | prod_Cal | C_CAGR_ideal | C_DD_ideal | C_Cal_ideal | C_CAGR_base | C_DD_base | C_Cal_base | C_CAGR_worst | C_DD_worst | C_Cal_worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fold1 2021 | Bull (+13.3%) | +32.10% | -20.11% | 1.596 | +30.89% | -19.07% | 1.620 | +30.87% | -19.08% | 1.618 | +30.83% | -19.09% | 1.615 |
| Fold2 2022 | Bear (-4.7%) | -6.33% | -13.83% | -0.458 | -1.45% | -11.52% | -0.126 | -1.44% | -11.52% | -0.125 | -1.56% | -11.52% | -0.136 |
| Fold3 2023 | Bull (+29.8%) | +2.08% | -2.88% | 0.724 | +5.68% | -3.84% | 1.478 | +5.79% | -3.88% | 1.495 | +5.65% | -3.88% | 1.456 |
| Fold4 2024 | Bull (+19.7%) | -6.03% | -10.37% | -0.582 | +1.16% | -12.73% | 0.091 | +1.09% | -12.75% | 0.085 | +0.97% | -12.78% | 0.076 |
| Fold5 2025 | Bull (+26.8%) | +3.57% | -1.95% | 1.831 | +22.97% | -7.08% | 3.246 | +22.95% | -7.10% | 3.234 | +22.93% | -7.13% | 3.215 |
| **avg** | — | **+5.08%** | **-9.83%** | **0.622** | **+11.85%** | **-10.85%** | **1.262** | **+11.85%** | **-10.87%** | **1.261** | **+11.76%** | **-10.88%** | **1.245** |

---
## 7. PnL Drag 分析

| Metric | base | worst |
|---|---|---|
| execution_pnl_drag (¥) | ¥1,623 | ¥24,000 |
| drag as % of ideal_pnl | 0.17% | 2.58% |
| drag per trade | ¥41 | ¥600 |
| idle_capital (avg fold%) | 14.0% | 14.0% |

注: backtest既計上 COST_ONE_WAY=15.5bp 片道 は ideal に含む。
drag = 追加摩擦 (extra_slip + partial_fill_loss + reject_loss) のみ。

---
## 8. Capacity 分析

- 平均注文サイズ: ¥831,497
- ADV推定 (RSR42 large/mid cap): ¥20億円
- Capacity utilization: 0.0416% of ADV
- Market impact (Almgren-Chriss概算 @ 0.0416% ADV): < 0.1bp
- 判定: **Capacity 問題なし** — 成行 寄り付き で市場影響は無視可能

---
## 9. Deployment Gate 判定

| 条件 | BASE | WORST | ライン | 結果 |
|---|---|---|---|---|
| alpha_realization % | 100.60 | 99.44 | 85.0 | ✅ |
| fill_rate % | 100.00 | 100.00 | 95.0 | ✅ |
| execution_tax | 1.01 | 0.98 | 0.9 | ✅ |
| broker_reject % | 0.00 | 0.00 | 0.5 | ✅ |
| cash_sync_error | 0 | 0 | ≈0 | ✅ |

---
## 10. 最終判定

| 項目 | 値 |
|---|---|
| **go_live** | **GO** |
| **initial_alloc** | **10%** |
| alpha_realization_base | 100.6% |
| alpha_realization_worst | 99.4% |
| execution_tax_base | 1.007 |
| execution_tax_worst | 0.984 |
| ideal_standalone_calmar | 1.439 |
| base_standalone_calmar | 1.449 |
| worst_standalone_calmar | 1.415 |
| **deploy_recommendation** | 全シナリオ(BASE/WORST)で採用基準クリア。kabuステーション 成行 寄り付き でのライブ実装を推奨。初期配分 10% (Study11 CONDITIONAL Near-Pass に準拠)。 |

### rollback_trigger 一覧

| トリガー | 閾値 | アクション |
|---|---|---|
| alpha_realization_rolling_3m | < 70% | 3ヶ月ローリングで実現αが70%を下回った場合 → alloc=0%に即時縮小・原因調査 |
| fill_rate_monthly | < 90% | 月次約定率90%未満 (API障害疑い) → alloc=0%に即時縮小・原因調査 |
| execution_tax_monthly | < 0.80 | 月次Calmar比率0.80未満 → alloc=0%に即時縮小・原因調査 |
| n_missed_trades_monthly | > 2 | 月2件超のシグナル見逃し → alloc=0%に即時縮小・原因調査 |
| consecutive_reject | > 3 | 3連続ブローカー拒否 → alloc=0%に即時縮小・原因調査 |
| cash_sync_error_detected | > 0 | キャッシュ残高不一致検出 → alloc=0%に即時縮小・原因調査 |