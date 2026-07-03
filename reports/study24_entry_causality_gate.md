# Study24 Entry Causality Gate

作成日: 2026-06-23  |  因果判定のみ（research-closure audit）/ Entry改善提案禁止 / Entry実装変更禁止 / Sizing・Capital変更禁止 / signal_bridge.py変更禁止

**Strategy**: Study9 Case B (FROZEN)  **Entry/Exit/Authority/Execution/Sizing**: 現行（無変更）  **Capital**: ¥1,800,000

**観測ウィンドウ**: 2018-01-01 → 2025-12-31

**目的**: Study23で発見された `days_cross90<=2` が「介入可能な改善要因」か「既存alphaの説明変数」かを判定する。

---
## Case A〜F 比較表

| Case | 説明 | trade_count | CAGR | Calmar | MaxDD | PF | win_rate |
|---|---|---|---|---|---|---|---|
| A | Baseline | 35 | +41.75% | 0.926 | -45.09% | 1.531 | 37.1% |
| B | Counterfactual: skip days_cross90<=2 | 29 | +26.82% | 0.721 | -37.19% | 1.461 | 37.9% |
| C | Counterfactual: +1d delay on days_cross90<=2 | 33 | +14.61% | 0.389 | -37.60% | 1.562 | 42.4% |
| D | Counterfactual: same-day substitute (days_cross90>=3) | 29 | +26.82% | 0.721 | -37.19% | 1.461 | 37.9% |
| E | Counterfactual: substitute + lot/notional constraint | 29 | +26.82% | 0.721 | -37.19% | 1.461 | 37.9% |
| F | Oracle Removal: Study22 FALSE_BREAKOUT/HIGH_VOL_ENTRY excluded | 28 | +52.18% | 1.200 | -43.49% | 1.905 | 53.6% |

| Case | alpha_retention | loss_removed | winner_removed | causal_precision | counterfactual_profit_loss | intervention_efficiency | INTERVENTION_VALID? |
|---|---|---|---|---|---|---|---|
| A | — | — | — | — | — | — | — |
| B | 80.0% | +10.4% | +20.1% | 59.1% | +26.2% | 0.52x | ❌ no |
| C | 86.6% | +10.9% | +13.5% | 59.1% | +15.1% | 0.81x | ❌ no |
| D | 80.0% | +10.4% | +20.1% | 59.1% | +26.2% | 0.52x | ❌ no |
| E | 80.0% | +10.4% | +20.1% | 59.1% | +26.2% | 0.52x | ❌ no |
| F | 100.6% | +29.0% | -0.6% | 100.0% | -19.4% | -50.93x | ❌ no |

| Case | execution_feasibility | access_preservation | timing_shift_days | signal_substitution_rate | tail_preservation | capital_utilization | portfolio_turnover_delta | hold_overlap |
|---|---|---|---|---|---|---|---|---|
| A | 100.0% | 1.00x | 0.0d | 0.0% | 100.0% | 87.5% | +0.0% | 0 |
| B | 0.0% | 0.92x | 0.0d | 0.0% | 33.3% | 80.5% | -17.1% | 0 |
| C | 36.7% | 0.88x | 1.0d | 0.0% | 33.3% | 77.0% | -5.7% | 0 |
| D | 0.0% | 0.92x | 0.0d | 0.0% | 33.3% | 80.5% | -17.1% | 0 |
| E | 0.0% | 0.92x | 0.0d | 0.0% | 33.3% | 80.5% | -17.1% | 0 |
| F | 100.0% | 0.99x | 0.0d | 0.0% | 66.7% | 86.6% | -20.0% | 0 |

`entry_delay_cost`（Case Cのみ意味を持つ）: -2.5831R
注: entry_delay_costはportfolio-level簡易proxy（全Case C成立トレード平均R − 全trigger平均R）。個別トレードのbaseline/delayed厳密マッチングではない点に注意。

注: Case Fの`intervention_efficiency=-50.93x`はwinner_removed≈0%（-0.6%）による分母近傍ゼロの数学的退化であり、解釈不能。Case Fはalpha_retention≥100%かつloss_removed>0%という形で直接評価すべきで、intervention_efficiency比率はCase Fの判定根拠には使用していない（INTERVENTION_VALID判定はCase B〜Eのみに適用、ALPHA_COMPONENT判定はCalmar/CAGR直接比較で行っている）。

注: Case D/Eの`signal_substitution_rate=0.0%`は、全39トリガー日においてdays_cross90>=3を満たす同日代替候補が一度も存在しなかったことを意味する。D/Eの介入メカニズムは本データセットでは実質的に一度も発火せず、結果はCase Bと完全一致した。単一スロット・狭ユニバース構成では「容量を逃さず別銘柄に振り替える」という設計目的自体が成立しにくいことを示す一発見。

---
## 判定基準

**INTERVENTION_VALID**（Case B〜Eのいずれかが該当）:
causal_precision≥75% AND alpha_retention≥90% AND counterfactual_profit_loss≤10% AND trade_count比≥80% AND intervention_efficiency≥3.0x

**ALPHA_COMPONENT**: 上記不成立 かつ Case Fのみ有意改善（説明可能だが介入不可）

**EXPLANATION_ONLY**: 上記以外（説明変数だが改善要因とは言えない）

---
## 最終出力

| 指標 | 値 |
|---|---|
| **research_decision** | **ALPHA_COMPONENT** |
| causal_driver | Study22 FALSE_BREAKOUT/HIGH_VOL_ENTRY classification (oracle-only, no implementable real-time proxy found in B-E) |
| intervention_efficiency (reference=F) | -50.93x |
| alpha_retention | 100.6% |
| loss_removed | +29.0% |
| winner_removed | -0.6% |
| expected_calmar_change | +0.274 (0.926 → 1.200) |
| expected_dd_change | +1.60pp (-45.09% → -43.49%) |
| recommend_entry_change | NO_CHANGE_RECOMMENDED |
| research_state | CLOSE_ENTRY_RESEARCH |

---
## 最終回答

1. **days_cross90<=2 は介入可能な改善要因か？** → NO（介入を試みた全Caseで net改善基準を満たさない）
2. **days_cross90<=2 は既存alphaの説明変数か？** → YES
3. **Entry研究を継続すべきか終了すべきか？** → 終了（このリードでは追加のEntry研究価値は確認できない）
4. **Walk-Forwardへ進む価値があるか？** → NO — Walk-Forwardに進む前にcausal gateを通過していない

制約: 新規Entry/Exitルール・Sizing/Capital変更・Productionコード変更は本研究では一切提案しない。
