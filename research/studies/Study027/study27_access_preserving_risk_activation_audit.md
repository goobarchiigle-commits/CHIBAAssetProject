# Study27 Access-Preserving Risk Activation Audit

作成日: 2026-06-24  |  リスク発火条件のみ変更（risk_activation_only）/ Entry・Exit・Signal・RSR閾値・days_cross90・slope・Execution・Authority・Capital・Production変更禁止 / サイズ変更ではなく発火タイミングだけでの改善余地判定

**Strategy**: Study9 Case B (FROZEN)  **Entry/Exit/Signal/Execution/Authority**: 現行（無変更）  **Capital**: ¥1,800,000  **max_pos**: 1

**観測ウィンドウ**: 2018-01-01 → 2025-12-31（Study24/25と同期）

**最重要原則**: 本研究は「勝ち方を変える研究」ではない。既存alphaを維持したまま、Accessを守りながら、Risk Activationだけで改善余地が残るかを検証する。

---
## Case別結果

| Case | 説明 | trade_count | CAGR | Calmar | MAR | maxDD | alpha_retention | lot_access |
|---|---|---|---|---|---|---|---|---|
| A | Baseline (現行) | 35 | +41.75% | 0.926 | 0.926 | -45.09% | 100.0% | 100.0% |
| B | Volatility-aware Shrink (Study25 Case E再現) | 34 | +40.82% | 1.014 | 1.014 | -40.26% | 98.8% | 89.7% |
| C | Volatility-aware Shrink + Activation Floor (最小1lot保証) | 35 | +40.93% | 1.017 | 1.017 | -40.26% | 100.0% | 100.0% |
| D | Drawdown Throttle (サイズ不変、DD上昇時のEntry間隔延長) | 32 | +38.94% | 0.937 | 0.937 | -41.57% | 102.3% | 100.0% |
| E | Heat Budget (サイズ不変、リスク過熱時のEntry抑制) | 31 | +15.56% | 0.339 | 0.339 | -45.86% | -0.1% | 84.2% |
| F | Combined Activation (D∨E、サイズ縮小禁止) | 28 | +8.34% | 0.182 | 0.182 | -45.86% | 10.1% | 93.5% |

| Case | activation_ratio | capital_activation_ratio | capital_efficiency | slot_utilization | cash_idle_ratio |
|---|---|---|---|---|---|
| A | 100.0% | 80.2% | 1.00x | 87.5% | 19.8% |
| B | 100.0% | 73.8% | 1.06x | 87.1% | 26.2% |
| C | 100.0% | 75.0% | 1.05x | 87.5% | 25.0% |
| D | 47.8% | 68.3% | 1.10x | 73.6% | 31.7% |
| E | 44.7% | 38.3% | 0.78x | 47.7% | 61.7% |
| F | 33.3% | 35.3% | 0.45x | 46.6% | 64.7% |

| Case | P95_trade_loss | P95_R | recovery_days | recovery_speed |
|---|---|---|---|---|
| A | ¥-444,252 | -4.196R | 12.9d | 0.0145/d |
| B | ¥-394,229 | -4.286R | 12.7d | 0.0151/d |
| C | ¥-386,434 | -4.196R | 12.6d | 0.0150/d |
| D | ¥-247,450 | -3.763R | 20.2d | 0.0121/d |
| E | ¥-139,859 | -4.332R | 39.6d | 0.0213/d |
| F | ¥-140,168 | -4.392R | 79.7d | 0.0098/d |

### 追加監査

| Case | access_loss_ratio | activation_loss_ratio | opportunity_loss(R) | tail_capture_change | signal_preservation | decision_complexity | risk_activation_frequency(/yr) | trigger_effectiveness |
|---|---|---|---|---|---|---|---|---|
| A | 0.0% | 0.0% | +0.0000R | +0.0% | 100.0% | 0 | 0.00 | +0.00000 |
| B | 10.3% | 0.0% | +0.0000R | +0.0% | 97.1% | 1 | 0.00 | +0.08800 |
| C | 0.0% | 0.0% | +0.0000R | +0.0% | 100.0% | 2 | 0.00 | +0.09100 |
| D | 0.0% | 52.2% | +0.1332R | -33.3% | 51.4% | 1 | 6.68 | +0.00031 |
| E | 15.8% | 55.3% | +80.3592R | -66.7% | 54.3% | 1 | 8.73 | -0.01249 |
| F | 6.5% | 66.7% | +69.9967R | -66.7% | 37.1% | 2 | 11.51 | -0.01200 |

注（lot_access / access_loss_ratio）: admission（D/E/Fのdd_throttle・heat_budget）を通過した日のうち、サイズが縮小され1単元未満に切り下げられてentryが事実上消失した割合（Study25既知のlot feasibility問題）。Case D/E/Fはサイズ不変なので構造上 lot_access≈100%となる。

注（activation_ratio / activation_loss_ratio）: trigger_daysのうち、admissionそのものが拒否された日の割合。B/Cは一度もadmissionを拒否しない設計（サイズのみ変更）のため activation_ratio=100%。

---
## 採用条件判定

基準: alpha_retention≥98% AND lot_access≥95% AND DD改善≥2.0pp AND Calmar改善≥0.10

| Case | calmar_delta | dd_delta | alpha_retention | lot_access | meets_adoption |
|---|---|---|---|---|---|
| B | +0.0880 | +4.83pp | 98.8% | 89.7% | ❌ no |
| C | +0.0910 | +4.83pp | 100.0% | 100.0% | ❌ no |
| D | +0.0110 | +3.52pp | 102.3% | 100.0% | ❌ no |
| E | -0.5870 | -0.77pp | -0.1% | 84.2% | ❌ no |
| F | -0.7440 | -0.77pp | 10.1% | 93.5% | ❌ no |

---
## 最終出力

| 指標 | 値 |
|---|---|
| **best_case** | C |
| best_case_label | Volatility-aware Shrink + Activation Floor (最小1lot保証) |
| **root_constraint** | EFFECT_SIZE_INSUFFICIENT (sizing-side lever dominates pure-timing levers: best sizing Δcalmar=+0.0910 (C) vs best pure-timing Δcalmar=+0.0110 (D) -- risk_activation TIMING alone captures less DD-reduction headroom than size modulation, and even the best sizing lever falls short of the adoption bar (0.10)). Secondary finding: pure-timing Heat Budget (E/F) is actively destructive (E alpha_retention=-0.1%, F=10.1%) -- portfolio realized-vol heat budget blocks the strategy's own alpha-bearing momentum continuation; equity-curve volatility IS the alpha signature here, not an independent risk signal that can be gated separately. |
| geometry_interaction | Case B/C(sizing) vs D/E/F(activation-only) lot_access comparison: B=89.7% / C=100.0% / D=100.0% / E=84.2% / F=93.5%. サイズ縮小を含むCaseのみ lot_access低下→Study25と同様のLOTカスケード再現 |
| access_preservation | C: lot_access=100.0%, alpha_retention=100.0% |
| risk_activation_effect | C: dd_reduction=+4.83pp via 0dd_blocks+0heat_blocks (0.00/yr), calmar_delta=+0.0910 |
| calmar_delta (best) | +0.0910 |
| dd_delta (best) | +4.83pp |
| **recommend_policy** | NO_CHANGE_RECOMMENDED (risk_activation_only軸も新たな改善余地なし) |
| **research_status** | **EXHAUSTED** |

制約: 本研究は改善案の実装・Entry/Exit/Signal/Production変更を一切行わない（改善余地の存在判定のみ）。
