# Entry Velocity Tail Preservation WF

作成日: 2026-06-20  |  研究専用 / 実装変更禁止

仮説: slope>5 entry のサイズ・期間・利食い制御によりリスク削減しつつ tail を保全できる

採用条件: WF>=4/5 AND ΔCAGR≥+0.3pp AND ΔDD≤+1.5pp AND tail_capture≥90%

---
## 1. Final Output

| Case | tail_efficiency | keep_or_drop |
|---|---|---|
| B: slope>5 weight×0.5 | 0.1400 | **DROP** |
| C: slope>5 weight×0.3 | 0.1400 | **DROP** |
| D: slope>5 max_hold=1 | 0.0600 | **DROP** |
| E: slope>5 slot2のみ | 0.0800 | **DROP** |
| F: slope>5 profit_lock+8% | 0.3600 | **DROP** |

---
## 2. WF サマリ

| Case | WF | ΔCAGR | ΔDD | ΔCalmar | PF_X | win_X | tail_capture | ADOPT |
|---|---|---|---|---|---|---|---|---|
| B (slope>5 weight×0.5) | 1/5 | +5.32pp | +1.78pp | +0.264 | 1.647 | 53.3% | 70.0% | REJECT |
| C (slope>5 weight×0.3) | 1/5 | -4.33pp | +0.33pp | -0.560 | 11080.669 | 49.0% | 70.0% | REJECT |
| D (slope>5 max_hold=1) | 1/5 | +0.54pp | +3.54pp | +0.672 | 2.201 | 47.4% | 30.0% | REJECT |
| E (slope>5 slot2のみ) | 1/5 | -4.91pp | +3.27pp | -0.626 | 38220.524 | 50.4% | 40.0% | REJECT |
| F (slope>5 profit_lock+8%) | 2/5 | +4.70pp | +1.31pp | +0.896 | 3.551 | 54.8% | 90.0% | REJECT |

---
## 3. Tail 保全指標

| Case | tail_capture | winner_retention | avg_loss_s5 | avg_win_s5 | PF_s5 | top10_share |
|---|---|---|---|---|---|---|
| B (slope>5 weight×0.5) | ⚠70.0% | 45.1% | ¥-16,398 | ¥+44,973 | 2.981 | 45.2% |
| C (slope>5 weight×0.3) | ⚠70.0% | 25.3% | ¥-15,519 | ¥+26,307 | 1.695 | 71.7% |
| D (slope>5 max_hold=1) | ⚠30.0% | 10.7% | ¥-18,484 | ¥+26,443 | 0.969 | 55.0% |
| E (slope>5 slot2のみ) | ⚠40.0% | 46.9% | ¥-62,028 | ¥+75,239 | 0.992 | 68.5% |
| F (slope>5 profit_lock+8%) | 90.0% | 106.5% | ¥-30,731 | ¥+86,859 | 3.180 | 49.8% |

> **A baseline**: slope>5 winners=10, losers=12, PF_s5=1.926, avg_win=¥+89,542, avg_loss=¥-38,750


---
## 4. Fold 詳細 (Case F: slope>5 profit_lock+8%)

| Fold | Regime | CAGR_A | CAGR_X | ΔCAGR | ΔDD | n_A | n_X | pass |
|---|---|---|---|---|---|---|---|---|
| Fold1 | Bull (+13.3%) | +32.10% | +31.40% | -0.70pp | -0.38pp | 44 | 44 | ❌ |
| Fold2 | Bear (-4.7%) | -6.33% | -6.22% | +0.11pp | -0.23pp | 9 | 9 | ❌ |
| Fold3 | Bull (+29.8%) | +2.08% | +3.96% | +1.88pp | +1.30pp | 7 | 11 | ✅ |
| Fold4 | Bull (+19.7%) | -6.03% | +4.79% | +10.82pp | +4.93pp | 3 | 21 | ❌ |
| Fold5 | Bull (+26.8%) | +3.57% | +14.95% | +11.38pp | +0.93pp | 2 | 24 | ✅ |

---
## 5. 全 Case × Fold マトリクス (ΔCAGR)

| Fold | B | C | D | E | F |
|---|---|---|---|---|---|
| Fold1 (2021) | -18.09pp ❌ | -17.15pp ❌ | -22.82pp ❌ | -26.44pp ❌ | -0.70pp ❌ |
| Fold2 (2022) | +11.25pp ✅ | -2.81pp ❌ | -0.10pp ❌ | +6.02pp ✅ | +0.11pp ❌ |
| Fold3 (2023) | +7.08pp ❌ | -7.71pp ❌ | +5.92pp ❌ | -5.75pp ❌ | +1.88pp ✅ |
| Fold4 (2024) | +23.82pp ❌ | +7.96pp ✅ | +9.03pp ❌ | +0.31pp ❌ | +10.82pp ❌ |
| Fold5 (2025) | +2.54pp ❌ | -1.96pp ❌ | +10.68pp ✅ | +1.29pp ❌ | +11.38pp ✅ |

---
## 6. 全 Case × Fold マトリクス (ΔDD)

| Fold | B | C | D | E | F |
|---|---|---|---|---|---|
| Fold1 (2021) | -1.96pp ✅ | -2.94pp ✅ | -0.91pp ✅ | +1.60pp ❌ | -0.38pp ✅ |
| Fold2 (2022) | -4.68pp ✅ | +2.01pp ❌ | +2.53pp ❌ | -0.38pp ✅ | -0.23pp ✅ |
| Fold3 (2023) | +5.19pp ❌ | +4.30pp ❌ | +7.83pp ❌ | +3.82pp ❌ | +1.30pp ✅ |
| Fold4 (2024) | +4.78pp ❌ | -2.69pp ✅ | +7.60pp ❌ | +5.71pp ❌ | +4.93pp ❌ |
| Fold5 (2025) | +5.57pp ❌ | +0.96pp ✅ | +0.66pp ✅ | +5.58pp ❌ | +0.93pp ✅ |

---
## 7. Case 分析

### B/C: サイズ削減 (weight×0.5 / ×0.3)
- B: ΔCAGR=+5.32pp  ΔDD=+1.78pp  tail_capture=70.0%
- C: ΔCAGR=-4.33pp  ΔDD=+0.33pp  tail_capture=70.0%
> サイズ削減は slope>5 ポジションのスロットを占有しつつ投資額を減らす。 tail_capture は count ベースで高い (同じ取引を実行) が PnL 規模は削減される。

### D: max_hold=1 (slope>5)
- ΔCAGR=+0.54pp  ΔDD=+3.54pp  tail_capture=30.0%
- avg_win_s5=¥+26,443  avg_loss_s5=¥-18,484  PF_s5=0.969
> 短期保有により損失拡大を抑制するが、multi-week winner を早期退出で逃す可能性あり。

### E: slope>5 最大1ポジション
- ΔCAGR=-4.91pp  ΔDD=+3.27pp  tail_capture=40.0%
- n_s5_trades_X=20  (A baseline: 22 total)
> slope>5 ポジション数を制限。Bull 相場で複数 slope>5 候補が並ぶ場合の集中リスクを削減。

### F: profit_lock +8% (slope>5)
- ΔCAGR=+4.70pp  ΔDD=+1.31pp  tail_capture=90.0%
- avg_win_s5=¥+86,859  PF_s5=3.180
> +8%到達で即利食い。slope>5 ポジションの益確保でリターン安定化を狙うが、 大型 winner のキャップが PnL を押し下げる可能性あり。

---
## 8. 最終判定

| Case | WF | ΔCAGR | ΔDD | tail_capture | tail_efficiency | keep_or_drop |
|---|---|---|---|---|---|---|
| **B** slope>5 weight×0.5 | 1/5 | +5.32pp | +1.78pp | 70.0% | 0.1400 | **DROP** |
| **C** slope>5 weight×0.3 | 1/5 | -4.33pp | +0.33pp | 70.0% | 0.1400 | **DROP** |
| **D** slope>5 max_hold=1 | 1/5 | +0.54pp | +3.54pp | 30.0% | 0.0600 | **DROP** |
| **E** slope>5 slot2のみ | 1/5 | -4.91pp | +3.27pp | 40.0% | 0.0800 | **DROP** |
| **F** slope>5 profit_lock+8% | 2/5 | +4.70pp | +1.31pp | 90.0% | 0.3600 | **DROP** |

### 全 Case REJECT

slope>5 の allocation 制御は production portfolio では機能しない。
根本原因: Bull 相場での slope>5 = alpha source → 任意の制御が 2021 Fold を破壊。

推奨: slope>5 を production でそのまま保持。
standalone/研究文脈のみ slope≤5 フィルターを維持 (WF=4/5 確認済み)。
次研究候補: regime-conditional control (Bear/Sideways のみ slope>5 制御)。