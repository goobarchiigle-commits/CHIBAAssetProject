# Study25 Portfolio Geometry Audit

作成日: 2026-06-23  |  ポートフォリオ構造監査のみ（research audit）/ Entry・Exit・Signal・RSR閾値・days_cross90・slope・Execution・Authority変更禁止 / 改善実装禁止（改善余地の存在判定のみ）

**Strategy**: Study9 Case B (FROZEN)  **Entry/Exit/Signal/Execution/Authority**: 現行（無変更）  **Capital**: ¥1,800,000  **max_pos**: 1

**観測ウィンドウ**: 2018-01-01 → 2025-12-31（Study24と同一）

**目的**: 既存alphaを変更せず、資本配置とリスク配分（Portfolio Geometry）だけで CAGR維持 + MaxDD削減 + Calmar改善の余地が残るかを判定する。利益最大化は目的外。

---
## Case別結果

| Case | 説明 | trade_count | CAGR | Calmar | MAR | maxDD | alpha_retention | capital_activation_ratio |
|---|---|---|---|---|---|---|---|---|
| A | Baseline (現行) | 35 | +41.75% | 0.926 | 0.926 | -45.09% | 100.0% | 80.2% |
| B | Capital Reserve (常時20%キャッシュ保持) | 34 | +33.40% | 0.936 | 0.936 | -35.68% | 98.8% | 62.4% |
| C | Exposure Decay (rolling DD上昇時のみ size↓) | 18 | +2.89% | 0.082 | 0.082 | -35.26% | 58.0% | 17.4% |
| D | Drawdown-aware Admission (DD>10%でnew_entry blocked) | 9 | -2.60% | -0.064 | -0.064 | -40.45% | -8.4% | 6.9% |
| E | Volatility-aware Sizing (新規Entryのみ risk_unit=base×f(vol)) | 34 | +40.82% | 1.014 | 1.014 | -40.26% | 98.8% | 73.8% |
| F | Combined (B+C+D+E) | 10 | +6.80% | 0.240 | 0.240 | -28.35% | 88.0% | 16.0% |

| Case | slot_utilization | cash_idle_ratio | exposure_utilization | recovery_days | recovery_speed | risk_adjusted_return |
|---|---|---|---|---|---|---|
| A | 87.5% | 19.8% | 91.7% | 12.9d | 0.0145/d | 0.871 |
| B | 87.1% | 37.6% | 71.6% | 12.9d | 0.0121/d | 0.840 |
| C | 40.4% | 82.6% | 43.1% | 19.2d | 0.0154/d | 0.268 |
| D | 8.2% | 93.1% | 84.4% | 0.0d | 0.0000/d | 0.173 |
| E | 87.1% | 26.2% | 84.7% | 12.7d | 0.0151/d | 0.891 |
| F | 38.3% | 84.0% | 41.7% | 14.7d | 0.0047/d | 0.375 |

### 追加監査

| Case | DD_reduction | return_preservation | capital_efficiency | activation_loss | opportunity_loss(R) | tail_capture_change | decision_complexity | lot_infeasible_days |
|---|---|---|---|---|---|---|---|---|
| A | +0.00pp | 1.00x | 1.00x | +0.00pp | +0.0000R | +0.0% | 0 | 0 |
| B | +9.41pp | 0.80x | 1.03x | +17.80pp | +0.0000R | +0.0% | 1 | 4 |
| C | +9.83pp | 0.07x | 0.32x | +62.80pp | +0.0000R | -66.7% | 2 | 102 |
| D | +4.64pp | -0.06x | -0.72x | +73.30pp | +90.5763R | -100.0% | 1 | 0 |
| E | +4.83pp | 0.98x | 1.06x | +6.40pp | +0.0000R | +0.0% | 3 | 4 |
| F | +16.74pp | 0.16x | 0.82x | +64.20pp | +10.1824R | -33.3% | 7 | 0 |

注（lot_infeasible_days）: geometryによる縮小サイズが1単元(LOT)未満に切り下げられ、entryが事実上スキップされた日数。Case C/D/Fでtrade_countが大きく減少しているのは、サイズ縮小が高価格銘柄で1単元購入不能になる「lot feasibility」問題（Study14既知の論点）と連動して発生しており、単純なdrawdown抑制効果だけでなく執行制約も同時に効いている。

注: alpha_retentionはR単位（1株あたりリターン比率）であり、ポジションサイズに非依存。従って Case B/C/E（資金量のみ変更・entryをスキップしない限り）はalpha_retention≈100%が構造的に当然となる — CAGR/Calmar/maxDDの変化は全てdollar-weightedなcapital_efficiency経由で発生しており、シグナル自体の変化ではない。alpha_retentionが95%を下回るのはentryが実際にスキップされた場合（Case Dのadmission block、またはCase C/Fのdecay=0到達時）のみ。

---
## 採用条件判定

基準: alpha_retention≥95% AND Calmar > baseline+0.15 AND maxDD ≤ baseline-1.0pp

| Case | calmar_delta | dd_delta | meets_adoption |
|---|---|---|---|
| B | +0.0100 | +9.41pp | ❌ no |
| C | -0.8440 | +9.83pp | ❌ no |
| D | -0.9900 | +4.64pp | ❌ no |
| E | +0.0880 | +4.83pp | ❌ no |
| F | -0.6860 | +16.74pp | ❌ no |

---
## 最終出力

| 指標 | 値 |
|---|---|
| **best_case** | E |
| geometry_effect | Volatility-aware Sizing (新規Entryのみ risk_unit=base×f(vol)) |
| calmar_delta (best) | +0.0880 |
| dd_delta (best) | +4.83pp |
| **research_decision** | **PORTFOLIO_GEOMETRY_EXHAUSTED** |
| recommend_change | NO_CHANGE_RECOMMENDED |
| final_research_state | CLOSE_GEOMETRY_RESEARCH |

制約: 本研究は改善案の実装・Entry/Exit/Signal/Production変更を一切行わない（改善余地の存在判定のみ）。
