# Study28 Portfolio Capital Allocation Audit

作成日: 2026-06-24  |  保有中ポートフォリオ内の資本配分のみ変更 / Entry・Exit・Signal・Execution・Universe・Position Count・Production変更禁止 / Sizing研究ではない（同時保有時の配分比のみ）

**基盤**: 本番PARAMS_LOCKED（Capital=¥3,000,000, max_positions=3）  **Entry/Exit/Signal**: Study9 Case B系列と同一（RSR[92,95) d90≤5 slope5≤5 / exit RSR<90）

**観測ウィンドウ**: 2018-01-01 → 2025-12-31

**重要**: 本研究はSizing研究ではない。Entry/Exit/Signal/Executionは完全固定。変更対象は「同時保有中の複数ポジション間での資本配分比」のみ。

---
## Case別結果

| Case | 説明 | trade_count | CAGR | Calmar | maxDD | alpha_retention | lot_access |
|---|---|---|---|---|---|---|---|
| A | Current Allocation (現行実装, 固定1/MAX_POS, 再配分なし) | 88 | +18.75% | 0.767 | -24.43% | 100.0% | 94.7% |
| B | RSR Weighted Allocation (weight=RSR/ΣRSR) | 88 | +19.28% | 0.797 | -24.19% | 101.1% | 94.7% |
| C | Volatility Quality Allocation (weight=(1/ATRpct)/Σ(1/ATRpct)) | 87 | +16.98% | 0.684 | -24.84% | 78.1% | 93.7% |
| D | RSR x Vol Quality (weight=(RSR/ATRpct)正規化) | 87 | +16.99% | 0.684 | -24.85% | 78.3% | 93.7% |
| E | Winner Concentration (Top1=40%/Top2=30%/Top3=20%, 残り現金) | 88 | +18.93% | 0.773 | -24.49% | 96.9% | 94.7% |
| F | Dynamic Composite (softmax(z(RSR)+z(1/ATRpct))) | 88 | +17.98% | 0.656 | -27.43% | 52.0% | 94.7% |
| G | Perfect Foresight Allocation (oracle, 理論上限のみ) | 88 | +18.83% | 0.836 | -22.52% | 97.0% | 94.7% |

| Case | calmar_delta | dd_delta | capital_activation | allocation_entropy | concentration_risk |
|---|---|---|---|---|---|
| A | +0.0000 | +0.00pp | 38.8% | 0.852 | 43.5% |
| B | +0.0300 | +0.24pp | 38.0% | 0.852 | 44.4% |
| C | -0.0830 | -0.41pp | 37.2% | 0.795 | 55.1% |
| D | -0.0830 | -0.42pp | 37.1% | 0.794 | 55.4% |
| E | +0.0060 | -0.06pp | 37.8% | 0.831 | 40.0% |
| F | -0.1110 | -3.00pp | 36.8% | 0.458 | 75.1% |
| G | +0.0690 | +1.91pp | 36.0% | 0.574 | 70.7% |

### Capital Efficiency

| Case | winner_capture | loser_capital_ratio | realloc_gain(¥) | realloc_loss(¥) | n_rebalance_events | n_simultaneous_days |
|---|---|---|---|---|---|---|
| A | 100.0% | 45.5% | ¥0 | ¥0 | 0 | 829 |
| B | 96.5% | 46.1% | ¥363,804 | ¥0 | 102 | 829 |
| C | 79.9% | 48.2% | ¥231,206 | ¥10,408 | 101 | 829 |
| D | 79.9% | 48.2% | ¥231,206 | ¥10,408 | 101 | 829 |
| E | 88.6% | 46.5% | ¥480,470 | ¥3,463 | 102 | 829 |
| F | 73.3% | 46.0% | ¥618,727 | ¥70,024 | 102 | 829 |
| G | 88.3% | 46.4% | ¥369,645 | ¥17,412 | 102 | 829 |

### 追加分析（利益上位20%/損失下位20%への資本配分比率）

| Case | top20%profit_alloc_share | bottom20%loss_alloc_share | 資本再配置利益増加額 | 資本再配置損失増加額 |
|---|---|---|---|---|
| A | 30.6% | 20.3% | ¥0 | ¥0 |
| B | 29.9% | 20.8% | ¥363,804 | ¥0 |
| C | 27.8% | 21.7% | ¥231,206 | ¥10,408 |
| D | 27.9% | 21.7% | ¥231,206 | ¥10,408 |
| E | 30.4% | 20.7% | ¥480,470 | ¥3,463 |
| F | 34.3% | 23.9% | ¥618,727 | ¥70,024 |
| G | 35.0% | 22.8% | ¥369,645 | ¥17,412 |

注: 「資本再配置利益/損失増加額」は、同時保有中の配分変更（リバランス）に伴う部分売却が確定した実現益・実現損の合計（実現P&Lのみ。最終Exitの損益とは別に集計）。

---
## 採用条件判定

基準: alpha_retention≥98% AND lot_access≥95% AND Calmar改善≥+0.10 AND DD改善≥2.0pp

| Case | calmar_delta | dd_delta | alpha_retention | lot_access | meets_adoption |
|---|---|---|---|---|---|
| B | +0.0300 | +0.24pp | 101.1% | 94.7% | ❌ no |
| C | -0.0830 | -0.41pp | 78.1% | 93.7% | ❌ no |
| D | -0.0830 | -0.42pp | 78.3% | 93.7% | ❌ no |
| E | +0.0060 | -0.06pp | 96.9% | 94.7% | ❌ no |
| F | -0.1110 | -3.00pp | 52.0% | 94.7% | ❌ no |
| G | +0.0690 | +1.91pp | 97.0% | 94.7% | ❌ no |

---
## 最終出力

| 指標 | 値 |
|---|---|
| **research_status** | **EXHAUSTED** |
| **best_case** | B |
| **best_allocation_policy** | RSR Weighted Allocation (weight=RSR/ΣRSR) |
| calmar_delta (best, B-F) | +0.0300 |
| dd_delta (best, B-F) | +0.24pp |
| **allocation_theoretical_ceiling** (Case G calmar_delta) | +0.0690 |
| portfolio_alpha_leverage (best B-F calmar_delta / ceiling) | +0.435 |
| **recommend_change** | PORTFOLIO_ALLOCATION_RESEARCH_END_CANDIDATE -- 次段階としてLIMITED_LIVE + Exit Monitoringへの移行を提言。STRONGLY SUPPORTED: even Case G (perfect-foresight oracle, theoretical ceiling) only reaches calmar_delta=+0.0690<0.10 -- no real-time-executable allocation policy can ever exceed this, so intra-portfolio capital allocation has NO exploitable headroom at all. |

制約: 本研究は改善案の実装・Entry/Exit/Signal/Execution/Universe/Production変更を一切行わない（配分余地の存在判定のみ）。
