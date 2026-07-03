# Dedicated Alpha Sleeve — Exit Persistence WF  (Study 8C)

作成日: 2026-06-14  |  研究専用 / 実装変更禁止

固定ENTRY: RSR[92,95) d90≤5, cap=equity×25%, max_pos=1

採用条件: WF≥4/5, ΔCAGR>+0.3pp, ΔDD≤+1.5pp, trigger≤8.0/yr, alpha_retention≥85%

**採用Case**: 0件  |  最終判定: **🔬 KEEP RESEARCH**

---
## 1. Executive Summary

- ベースライン (Case A): sl_CAGR=37.0%  trig=9.5/yr  WF=0/5
- 検証軸: EXIT 条件変更による trig/yr 削減 + alpha_retention ≥85% 維持
- 採用: **0件**

---
## 2. WF サマリ (7 Exit Case)

| Case | EXIT条件 | WF | sl_CAGR | ΔCAGR | ΔDD | trig/yr | α_ret | bounce | false | 採用 |
|---|---|---|---|---|---|---|---|---|---|---|
| A | RSR<90 即exit | 0/5 | 37.0% | +9.70pp | +1.79pp | 9.5 | 100% | 33% | 38% | ❌ |
| B | RSR<90 2日連続 | 1/5 | 33.2% | +8.86pp | +0.96pp | 7.6 | 90% | 29% | 33% | ❌ |
| C | RSR<90 3日連続 | 2/5 | 39.2% | +10.86pp | +2.46pp | 3.7 | 106% | 10% | 51% | ❌ |
| D | RSR<89 即exit | 0/5 | 37.0% | +9.70pp | +1.79pp | 9.5 | 100% | 33% | 38% | ❌ |
| E | RSR<89 2日連続 | 1/5 | 33.2% | +8.86pp | +0.96pp | 7.6 | 90% | 29% | 33% | ❌ |
| F | RSR<90 AND hold≥5d | 1/5 | 32.1% | +4.98pp | +1.81pp | 6.8 | 87% | 29% | 33% | ❌ |
| G | RSR<90 AND hold≥3d | 1/5 | 24.9% | +4.31pp | +1.72pp | 6.8 | 67% | 28% | 38% | ❌ |

---
## 3. Trigger Elasticity & Calmar per Trigger Removed

> Δtrig = trig_A − trig_X (正 = trigger 削減)  ΔCalmar/Δtrig = Calmar 改善 per trigger 削除

| Case | trig/yr | Δtrig vs A | Δavg_hold | elast. | ΔCalmar | Cal/trig | 評価 |
|---|---|---|---|---|---|---|---|
| A | 9.5 | +0.0 | +0.0d | N/A | +0.000 | N/A | 同等/悪化 |
| B | 7.6 | +1.9 | -7.9d | hold不変 | +0.131 | +0.0689 | ✅ |
| C | 3.7 | +5.8 | +31.3d | 0.19 trig/d | +0.084 | +0.0145 | ✅ |
| D | 9.5 | +0.0 | +0.0d | N/A | +0.000 | N/A | 同等/悪化 |
| E | 7.6 | +1.9 | -7.9d | hold不変 | +0.131 | +0.0689 | ✅ |
| F | 6.8 | +2.7 | -20.6d | hold不変 | -0.067 | -0.0248 | ❌ -0.0248/trig |
| G | 6.8 | +2.7 | -14.7d | hold不変 | -0.242 | -0.0896 | ❌ -0.0896/trig |

---
## 4. Exit Diagnostics

| Case | avg_hold | med_hold | idle% | flip/d | bounce% | false% | reason (top2) |
|---|---|---|---|---|---|---|---|
| A | 40.6d | 27.3d | 6.6% | 0.222 | 33% | 38% | RSR<90:98%  MARKET_SHOCK:2% |
| B | 32.7d | 13.1d | 7.9% | 0.273 | 29% | 33% | RSR<90_2D:97%  MARKET_SHOCK:3% |
| C | 71.9d | 57.2d | 7.0% | 0.114 | 10% | 51% | RSR<90_3D:89%  MARKET_SHOCK:11% |
| D | 40.6d | 27.3d | 6.6% | 0.222 | 33% | 38% | RSR<89:98%  MARKET_SHOCK:2% |
| E | 32.7d | 13.1d | 7.9% | 0.273 | 29% | 33% | RSR<89_2D:97%  MARKET_SHOCK:3% |
| F | 20.0d | 16.5d | 34.6% | 0.122 | 29% | 33% | RSR<90_HOLD5:97%  MARKET_SHOCK:3% |
| G | 25.9d | 19.3d | 33.6% | 0.147 | 28% | 38% | RSR<90_HOLD3:97%  MARKET_SHOCK:3% |

---
## 5. Alpha Retention (vs Case A baseline)

> 基準 sl_CAGR (Case A, avg 5fold) = **37.0%**  採用基準 ≥85%

| Case | sl_CAGR | α_retention | 採用基準 |
|---|---|---|---|
| A | 37.0% | 100% | ✅ |
| B | 33.2% | 90% | ✅ |
| C | 39.2% | 106% | ✅ |
| D | 37.0% | 100% | ✅ |
| E | 33.2% | 90% | ✅ |
| F | 32.1% | 87% | ✅ |
| G | 24.9% | 67% | ❌ |

---
## 6. Fold 詳細

| Case | Fold | Regime | sl_CAGR | ΔCAGR | ΔDD | trig/yr | hold | bounce | false | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| A | Fold1 | Bull (+13.3%) | +109.7% | +17.06pp | +4.27pp | 2.1 | 69.0d | 0% | 50% | ❌ |
| A | Fold2 | Bear (-4.7%) | +27.6% | +11.12pp | -0.26pp | 13.4 | 60.0d | 46% | 31% | ❌ |
| A | Fold3 | Bull (+29.8%) | +1.7% | -0.15pp | +1.63pp | 16.4 | 17.1d | 62% | 38% | ❌ |
| A | Fold4 | Bull (+19.7%) | +11.7% | +7.03pp | -0.59pp | 11.3 | 13.9d | 55% | 46% | ❌ |
| A | Fold5 | Bull (+26.8%) | +34.3% | +13.46pp | +3.88pp | 4.1 | 42.8d | 0% | 25% | ❌ |
| B | Fold1 | Bull (+13.3%) | +60.9% | +6.32pp | +0.04pp | 3.1 | 51.7d | 0% | 33% | ✅ |
| B | Fold2 | Bear (-4.7%) | +41.8% | +12.79pp | -0.93pp | 13.4 | 29.8d | 46% | 23% | ❌ |
| B | Fold3 | Bull (+29.8%) | +14.7% | +4.47pp | +0.66pp | 9.2 | 24.3d | 22% | 22% | ❌ |
| B | Fold4 | Bull (+19.7%) | +22.0% | +10.55pp | -0.90pp | 8.2 | 21.0d | 75% | 62% | ❌ |
| B | Fold5 | Bull (+26.8%) | +26.8% | +10.17pp | +5.93pp | 4.1 | 36.8d | 0% | 25% | ❌ |
| C | Fold1 | Bull (+13.3%) | +107.4% | +16.54pp | +4.35pp | 1.0 | 135.0d | 0% | 100% | ❌ |
| C | Fold2 | Bear (-4.7%) | +26.6% | +10.71pp | +0.25pp | 5.2 | 81.6d | 0% | 40% | ✅ |
| C | Fold3 | Bull (+29.8%) | +16.3% | +5.56pp | +5.48pp | 6.1 | 34.3d | 17% | 50% | ❌ |
| C | Fold4 | Bull (+19.7%) | +18.8% | +10.34pp | -1.65pp | 3.1 | 57.7d | 33% | 33% | ✅ |
| C | Fold5 | Bull (+26.8%) | +27.1% | +11.15pp | +3.86pp | 3.1 | 51.0d | 0% | 33% | ❌ |
| D | Fold1 | Bull (+13.3%) | +109.7% | +17.06pp | +4.27pp | 2.1 | 69.0d | 0% | 50% | ❌ |
| D | Fold2 | Bear (-4.7%) | +27.6% | +11.12pp | -0.26pp | 13.4 | 60.0d | 46% | 31% | ❌ |
| D | Fold3 | Bull (+29.8%) | +1.7% | -0.15pp | +1.63pp | 16.4 | 17.1d | 62% | 38% | ❌ |
| D | Fold4 | Bull (+19.7%) | +11.7% | +7.03pp | -0.59pp | 11.3 | 13.9d | 55% | 46% | ❌ |
| D | Fold5 | Bull (+26.8%) | +34.3% | +13.46pp | +3.88pp | 4.1 | 42.8d | 0% | 25% | ❌ |
| E | Fold1 | Bull (+13.3%) | +60.9% | +6.32pp | +0.04pp | 3.1 | 51.7d | 0% | 33% | ✅ |
| E | Fold2 | Bear (-4.7%) | +41.8% | +12.79pp | -0.93pp | 13.4 | 29.8d | 46% | 23% | ❌ |
| E | Fold3 | Bull (+29.8%) | +14.7% | +4.47pp | +0.66pp | 9.2 | 24.3d | 22% | 22% | ❌ |
| E | Fold4 | Bull (+19.7%) | +22.0% | +10.55pp | -0.90pp | 8.2 | 21.0d | 75% | 62% | ❌ |
| E | Fold5 | Bull (+26.8%) | +26.8% | +10.17pp | +5.93pp | 4.1 | 36.8d | 0% | 25% | ❌ |
| F | Fold1 | Bull (+13.3%) | -8.3% | -7.38pp | -1.95pp | 5.1 | 11.0d | 0% | 20% | ❌ |
| F | Fold2 | Bear (-4.7%) | +8.2% | +1.95pp | -2.70pp | 6.2 | 15.2d | 33% | 33% | ✅ |
| F | Fold3 | Bull (+29.8%) | +45.1% | +6.30pp | +3.78pp | 8.2 | 26.6d | 25% | 25% | ❌ |
| F | Fold4 | Bull (+19.7%) | +5.4% | +2.16pp | +4.64pp | 10.3 | 19.2d | 60% | 60% | ❌ |
| F | Fold5 | Bull (+26.8%) | +110.2% | +21.89pp | +5.30pp | 4.1 | 28.0d | 25% | 25% | ❌ |
| G | Fold1 | Bull (+13.3%) | +2.7% | -6.44pp | -1.92pp | 5.1 | 36.0d | 0% | 40% | ❌ |
| G | Fold2 | Bear (-4.7%) | +7.9% | +2.54pp | -2.90pp | 6.2 | 15.0d | 33% | 33% | ✅ |
| G | Fold3 | Bull (+29.8%) | +36.5% | +6.68pp | +4.86pp | 10.2 | 22.0d | 30% | 30% | ❌ |
| G | Fold4 | Bull (+19.7%) | -9.4% | -0.82pp | +2.64pp | 8.2 | 12.2d | 75% | 62% | ❌ |
| G | Fold5 | Bull (+26.8%) | +86.6% | +19.58pp | +5.92pp | 4.1 | 44.5d | 0% | 25% | ❌ |

---
## 7. Structural Analysis

**Case A (baseline) bounce_rate=33%  false_exit_rate=38%  flip/day=0.222**

- bounce_rate: 即exit後10日以内にRSR≥92に復帰する割合 → 高いほど早期 exit が多い
- false_exit_rate: exit後20日maxリターン>10% → 高いほど持続 alpha を切っている
- flip/day: 保有中のRSR90クロス頻度 → 高いほどRSRがbounce帯で振動

最低 trigger Case: **C** (RSR<90 3日連続) trig=3.7/yr

**trigger/yr 感度まとめ:**
  A: RSR<90 即exit                        trig=  9.5/yr  hold= 40.6d  α_ret=100%
  B: RSR<90 2日連続                         trig=  7.6/yr  hold= 32.7d  α_ret=90%
  C: RSR<90 3日連続                         trig=  3.7/yr  hold= 71.9d  α_ret=106%
  D: RSR<89 即exit                        trig=  9.5/yr  hold= 40.6d  α_ret=100%
  E: RSR<89 2日連続                         trig=  7.6/yr  hold= 32.7d  α_ret=90%
  F: RSR<90 AND hold≥5d                  trig=  6.8/yr  hold= 20.0d  α_ret=87%
  G: RSR<90 AND hold≥3d                  trig=  6.8/yr  hold= 25.9d  α_ret=67%

---
## 8. Failure Analysis


**A** (RSR<90 即exit): REJECT — WF=0/5 / ΔDD=+1.79pp / trig=9.5/yr

**B** (RSR<90 2日連続): REJECT — WF=1/5

**C** (RSR<90 3日連続): REJECT — WF=2/5 / ΔDD=+2.46pp

**D** (RSR<89 即exit): REJECT — WF=0/5 / ΔDD=+1.79pp / trig=9.5/yr

**E** (RSR<89 2日連続): REJECT — WF=1/5

**F** (RSR<90 AND hold≥5d): REJECT — WF=1/5 / ΔDD=+1.81pp

**G** (RSR<90 AND hold≥3d): REJECT — WF=1/5 / ΔDD=+1.72pp / α_ret=67%<85%

---
## 9. Final Recommendation

## `🔬 KEEP RESEARCH`

全7 Case 採用基準未達。最良: **Case C** (RSR<90 3日連続)

- sl_CAGR=39.2%  ΔCAGR=+10.86pp  trig=3.7/yr  WF=2/5

**バインディング制約**: WF=2/5 / ΔDD=+2.46pp

**構造的発見:**
- Case A (即exit): trig=9.5/yr
- 最低 trig: 3.7/yr — EXIT 持続化でも 8/yr 未達
- bounce_rate=33%: 約33% の exit が10日以内に RSR≥92 復帰
  → RSR<90 threshold 自体が市場ノイズに対して敏感すぎる可能性

次ステップ候補:
- Study 8D: EXIT = ATR-based stop (threshold-free, noise-robust)
- Study 8E: capital 25%→15% で ΔDD 削減 + current best EXIT 組み合わせ
- Study 8F: RSR 70 exit (regime-aware 既採用済み, Study 7 知見) を sleeve に適用