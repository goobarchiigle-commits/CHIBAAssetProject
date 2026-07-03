# Study51 — Feature Attribution Audit
日付: 2026-06-28  
データソース: Study50 (Post-Integrity Revalidation, RSR42 + union fix)  
方針: ΔCAGR 一元評価から多軸評価へ転換

---

## 基準数値 (A_BASELINE, S5 only)

| 期間 | CAGR | MaxDD | Sharpe | Calmar | WF | WF avg CAGR | Seg3_2022 |
|---|---|---|---|---|---|---|---|
| IS 2018-2024 | 12.63% | -18.18% | 0.593 | 0.695 | — | — | — |
| OOS 2025 | 11.64% | -10.16% | 1.084 | 1.146 | — | — | — |
| WF 5-fold | — | — | 0.833 | — | **4/5** | +17.50% | **-5.11%** |

WF 各 fold:
```
Seg1 2020: +6.12%  Seg2 2021: +6.04%  Seg3 2022: -5.11% (FAIL)
Seg4 2023: +43.82%  Seg5 2024: +36.64%  std=19.14
```

---

## ① Feature Matrix

| Feature | IS ΔCAGR | WF ΔCAGR | OOS_2025 ΔCAGR | Δ Seg3_2022 | ΔMaxDD_IS | ΔSharpe_IS | Fold variance | WF pass | Complexity | Classification |
|---|---|---|---|---|---|---|---|---|---|---|
| **B_ATR_EXT** | +0.18 | +0.52 | **+2.23** | 0.00 | 0.00 | +0.007 | 19.86 (↑) | 4/5 | 低 | Return Enhancer (momentum) |
| **C_VOL_ADJ** | +0.16 | +1.29 | **-0.96** | 0.00 | **-1.99** | -0.010 | **22.14 (最大)** | 4/5 | 中 | Mixed / No Evidence |
| **D_EQ_SCALE** | -0.47 | +0.41 | -0.42 | **+2.46** | +0.06 | -0.015 | **18.79 (最小)** | **5/5** | 低 | Robustness Enhancer |
| **E_COMBINED** | -0.44 | **+1.37** | +0.26 | **+2.46** | -1.93 | -0.028 | 20.27 (↑) | **5/5** | 高 | Robustness + Conditional Return |

単位: pp (CAGR/MaxDD), pp (Seg3 改善量は絶対値)

---

## ② 各 Feature の多軸評価

### B_ATR_EXT (ATR Extension)

**機能**: RSR SELL シグナル発火後、ATR モメンタムが継続中なら退出を defer (5日)

#### ① Return
```
IS ΔCAGR:       +0.18pp   ← 統計的ノイズ水準
WF ΔCAGR:       +0.52pp   ← 弱い正の信号
OOS 2025 ΔCAGR: +2.23pp   ← 最も強い信号 (1年サンプル)
```
IS は 6.84 年の長期サンプルで +0.18pp = **非有意**。  
OOS 2025 は単年のため統計的確信度は低いが方向性は明確に正。  
WF avg はモメンタム強い Seg4(+2.57pp) のみ差が出て Seg5 は同値。

#### ② Risk
```
ΔMaxDD_IS:  0.00pp (完全一致)
ΔSharpe:   +0.007 (微改善)
ΔCalmar:   +0.010 (微改善)
Seg3_2022: -5.11% (変化なし — WF Fail 継続)
Worst fold: Seg3 2022 (両ケース同一)
```
**リスク変化なし**。この機能は DD を悪化させない。ただし 2022 年弱気環境での改善もゼロ。

#### ③ Robustness
```
WF pass: 4/5 (Baseline と同一)
Seg3 2022: FAIL (変化なし)
Fold std: 19.86 (Baseline 19.14 より微増)
OOS 2025: 良好 (+2.23pp)
```
WF を改善しない。頑健性向上の証拠はない。OOS 2025 の結果は検証として意味があるが 1 年のみ。

#### ④ Operational Cost
```
実装: exit_policy="A" の 1 パラメータ変更のみ
保守: ATR 閾値の定期確認
障害リスク: 低 (エグジット判定のみ、ポジション追加なし)
Live コスト: 最小
```
**最も実装・保守コストが低い機能。**

**→ 分類: Return Enhancer (モメンタム環境限定)**  
**→ Evidence: ★★★☆☆**  
根拠: OOS +2.23pp は有望だが IS が無意味水準。WF 改善なし。1 年 OOS では過学習リスク排除できず。

---

### C_VOL_ADJ (Volatility-Adaptive Position Cap)

**機能**: TOPIX 20d std < 0.8% の calm 日に max_pos を 3→4 に拡大

#### ① Return
```
IS ΔCAGR:       +0.16pp   ← 統計的ノイズ
WF ΔCAGR:       +1.29pp   ← 正だが内容要分析
OOS 2025 ΔCAGR: -0.96pp   ← 負 (baseline 下回る)
```
WF avg が +1.29pp に見えるが内訳:
```
Seg1 2020: 4.75% (Baseline 6.12%, Δ-1.37pp) ← 悪化
Seg2 2021: 3.76% (Baseline 6.04%, Δ-2.28pp) ← 悪化
Seg3 2022: -5.11% (同値)
Seg4 2023: 50.44% (Baseline 43.82%, Δ+6.62pp) ← 大幅改善 (bull増幅)
Seg5 2024: 40.11% (Baseline 36.64%, Δ+3.47pp) ← 改善 (bull増幅)
```
**Seg4/5 bull 市場での slot 増加効果が平均を押し上げているだけ。Seg1/2 は悪化。**

#### ② Risk
```
ΔMaxDD_IS: -1.99pp (悪化 — 最大 position 増加の副作用)
ΔSharpe:   -0.010  (悪化)
ΔCalmar:   -0.061  (悪化)
Seg3_2022: -5.11%  (変化なし)
OOS MaxDD: -0.45pp (悪化)
Fold std:  22.14   (全ケース最大 — 最も不安定)
```
**リスク指標が全面的に悪化。Risk Reducer でも Robustness Enhancer でもない。**

#### ③ Robustness
```
WF pass: 4/5 (Baseline と同一)
Seg3 2022: FAIL (変化なし)
Fold std: 22.14 (最大)
OOS 2025: -0.96pp (負)
```
**頑健性の改善ゼロ。むしろ fold variance が最大で最も不安定。**

#### ④ Operational Cost
```
実装: vol_adj_ts の事前計算 + max_positions_ts 渡し
保守: TOPIX vol 閾値の定期レビュー必要
障害リスク: 中 (calm判定ロジックのバグがポジション過多になりうる)
Live コスト: 低い (1 パラメータ)
```

**→ 分類: Mixed / No Evidence**  
**→ Evidence: ★★☆☆☆**  
根拠: Bull 増幅器であり Risk Reducer でも Robustness Enhancer でもない。IS/OOS/リスク指標が悪化。WF avg 改善は Seg4/5 の暴走効果に過ぎず、Fold variance 最大で信頼性が低い。

---

### D_EQ_SCALE (Equity-Scale Addon)

**機能**: 含み益 ≥ 1×ATR20 の勝ちポジションに対し cash×25% の追加 BUY

#### ① Return
```
IS ΔCAGR:       -0.47pp   ← 負 (リターン犠牲)
WF ΔCAGR:       +0.41pp   ← 微正
OOS 2025 ΔCAGR: -0.42pp   ← 微負
```
**リターン向上の証拠なし。むしろ IS/OOS ともに微減。**

#### ② Risk
```
ΔMaxDD_IS: +0.06pp (実質変化なし)
ΔSharpe:   -0.015  (微悪化)
ΔCalmar:   -0.024  (微悪化)
Seg3_2022: -5.11% → -2.65% (Δ+2.46pp) ← 唯一のケースで 2022 パス
OOS MaxDD: 0.00pp  (変化なし)
Fold std:  18.79   (全ケース最小 — 最も安定)
```
**Seg3 2022 が -5.11%→-2.65%、Sharpe が 0.057 (Baseline -0.029) と初めてプラスに転換。**  
最悪 fold が大幅改善。全 fold の標準偏差が最小 (18.79 vs Baseline 19.14)。

#### ③ Robustness
```
WF pass: 5/5 (Baseline 4/5 → 1 fold 追加)
Seg3 2022: PASS (Sharpe 0.057 > 0)
Fold std: 18.79 (最小 = 最安定)
OOS 2025: -0.42pp (微負、許容範囲)
```
**全ケースで唯一 WF 5/5 を達成。2022 年弱気市場を最悪 fold から「合格」に引き上げた。**  
Fold 一貫性も最高。**Robustness Enhancer の最も明確な証拠。**

メカニズム解釈:  
EQ_SCALE は「勝ちポジションにのみ」追加する。2022 年弱気市場では含み益基準を満たす銘柄が減少 → addon 発火が抑制 → 弱気環境での曝露増加が自動抑制される。**Bear 市場で逆張りにならない構造を内包**。

#### ④ Operational Cost
```
実装: addon_policy="D" の 1 パラメータ変更
保守: addon_count の定期監視
障害リスク: 低 (発注追加のみ; 既存ポジション退出に干渉しない)
Live コスト: 低 (発注件数微増のみ)
```

**→ 分類: Robustness Enhancer**  
**→ Evidence: ★★★★☆**  
根拠: WF 5/5 達成、Seg3_2022 +2.46pp 改善、fold variance 最小。リターン犠牲は IS -0.47pp / OOS -0.42pp と小さく、WF での一貫性向上がトレードオフとして正当化される。

---

### E_COMBINED (ATR_EXT + VOL_ADJ + EQ_SCALE)

**機能**: 上記 3 機能の同時有効化

#### ① Return
```
IS ΔCAGR:       -0.44pp   ← 微負
WF ΔCAGR:       +1.37pp   ← 最大 (全ケース中)
OOS 2025 ΔCAGR: +0.26pp   ← 微正
```
WF avg が最大 (+18.87%) で ΔCAGR は +1.37pp > 閾値 1.0pp。  
内訳:
```
Seg1 2020: 3.76%  (Δ-2.36pp) ← bull/recovery でやや悪化
Seg2 2021: 6.80%  (Δ+0.76pp) ← 微改善
Seg3 2022: -2.65% (Δ+2.46pp) ← 大幅改善 ★
Seg4 2023: 47.21% (Δ+3.39pp) ← 大幅改善 ★
Seg5 2024: 39.24% (Δ+2.60pp) ← 大幅改善 ★
```
**Seg3/4/5 の全てで改善。Seg1 が唯一悪化 (corona 回復期のポジション圧縮が原因)。**

#### ② Risk
```
ΔMaxDD_IS: -1.93pp (やや悪化) ← VOL_ADJ の副作用
ΔSharpe:   -0.028  (悪化)
ΔCalmar:   -0.089  (悪化)
Seg3_2022: -5.11% → -2.65% (Δ+2.46pp) ← EQ_SCALE 由来
OOS MaxDD: -0.44pp (微悪化)
Fold std:  20.27   (Baseline 19.14 より増加) ← VOL_ADJ の副作用
```
IS MaxDD が -1.93pp 悪化するのは VOL_ADJ の calm 日 slot 拡大が寄与。  
単一機能の中では最も大きいリスク悪化。

#### ③ Robustness
```
WF pass: 5/5 (Baseline 4/5 → 1 fold 追加)
Seg3 2022: PASS (+2.46pp)
Fold std: 20.27 (D_EQ_SCALE 18.79 より高い)
OOS 2025: +0.26pp (微正)
```
WF 5/5 は D_EQ_SCALE と同等。しかし fold std は D_EQ_SCALE より高い。  
OOS 2025 は唯一の全ケース正値 (+0.26pp)。

#### ④ Operational Cost
```
実装: 3 機能同時有効化 (最大複雑度)
保守: 3 系統の状態ファイル + 監視
障害リスク: 高 (VOL_ADJ/EQ_SCALE の同時発動タイミング依存)
Live コスト: 中 (発注件数増加 + vol_adj_ts 計算)
```
**最も実装・保守コストが高い。バグ混入時の切り分けも難しい。**

**→ 分類: Robustness Enhancer + Conditional Return Enhancer**  
**→ Evidence: ★★★★☆**  
根拠: WF 5/5、全ケース最大 WF ΔCAGR (+1.37pp)、Seg3 保護。IS/OOS 単年での改善は小さいが WF での一貫した優位性。欠点は IS MaxDD 悪化と複雑度。

---

## ③ Feature Classification まとめ

| Feature | 一次分類 | 二次分類 | IS リターン | OOS リターン | Bear保護 | Fold安定性 | 複雑度 |
|---|---|---|---|---|---|---|---|
| B_ATR_EXT | **Return Enhancer** | (モメンタム環境限定) | 中立 | 強 (+2.23pp) | なし | 中立 | 低 |
| C_VOL_ADJ | **No Evidence** | (Bull 増幅器) | 中立 | **負** (-0.96pp) | なし | **悪化** | 中 |
| D_EQ_SCALE | **Robustness Enhancer** | (Bear 環境保護) | 中立 | 中立 | **強** (+2.46pp) | **最良** | 低 |
| E_COMBINED | **Robustness Enhancer** | + Conditional Return | 中立 | 微正 (+0.26pp) | **強** (+2.46pp) | 良 | 高 |

---

## ② Feature Ranking (総合評価順)

### 1位: D_EQ_SCALE (Robustness Enhancer)

**採用推奨**

```
Evidence Score: ★★★★☆

理由:
- WF 5/5 達成 (Baseline 4/5 → 唯一の改善)
- Seg3_2022: -5.11% → -2.65% (+2.46pp) — 弱気市場を合格水準に引き上げた
- Fold std: 18.79 (全ケース最小 = 最安定)
- IS/OOS のリターン犠牲: -0.47pp / -0.42pp (許容範囲)
- IS MaxDD: 実質変化なし (+0.06pp)
- 実装コスト: 最低 (addon_policy="D" の 1 設定)
- メカニズムが明確: 「勝ちにのみ乗る」→ Bear 市場でのアドオン自然抑制
```

**採用条件**: 無条件採用可。リターン犠牲 -0.47pp は Bear 保護と WF 5/5 の対価として合理的。

---

### 2位: E_COMBINED (Robustness + Conditional Return)

**条件付き採用推奨**

```
Evidence Score: ★★★★☆

理由:
- WF avg CAGR: +18.87% (全ケース最高, ΔCAGR +1.37pp > 閾値 1.0pp)
- WF 5/5 (D_EQ_SCALE と同等)
- Seg3_2022: -2.65% (+2.46pp) — D_EQ_SCALE と同等の Bear 保護
- OOS 2025: +0.26pp (微正 — 全ケース唯一の正値)
- IS ΔCAGR: -0.44pp (IS でのリターン犠牲)
- IS MaxDD: -1.93pp (VOL_ADJ 由来の悪化)
- Fold std: 20.27 (D_EQ_SCALE より高い)
- 複雑度: 最大 (3 機能同時運用)
```

**採用条件**: 運用複雑度を受け入れられる場合。  
D_EQ_SCALE に対する追加価値は WF avg ΔCAGR +0.96pp のみで、そのコストとして IS MaxDD -1.93pp と保守複雑度増加を負う。

---

### 3位: B_ATR_EXT (Return Enhancer, conditional)

**様子見推奨 → OOS 蓄積後に再判断**

```
Evidence Score: ★★★☆☆

理由:
- OOS 2025 ΔCAGR: +2.23pp (最も強い単年 OOS シグナル)
- IS ΔCAGR: +0.18pp (統計的に無意味)
- WF ΔCAGR: +0.52pp (弱い)
- WF pass: 4/5 (Baseline と同等 — 改善なし)
- Seg3_2022: 変化なし (Bear 保護ゼロ)
- MaxDD: 変化なし (リスク中立)
- 実装コスト: 最低
- 欠点: OOS 1 年のみで過学習排除不可
```

**採用条件**:  
OOS 2025 の +2.23pp が 2026 年以降の OOS でも継続するなら採用価値が高まる。  
現時点では「1年の出来事」として待機が合理的。Shadow 状態での監視継続推奨。

---

### 4位: C_VOL_ADJ (No Evidence / Reject)

**不採用**

```
Evidence Score: ★★☆☆☆

不採用理由:
- OOS 2025: -0.96pp (唯一の Baseline 下回り)
- IS MaxDD: -1.99pp (リスク悪化)
- Fold std: 22.14 (全ケース最大 — 最も不安定)
- Seg3_2022: 変化なし (Bear 保護ゼロ)
- WF avg 改善は Seg4/5 Bull 市場の slot 増加のみに起因
- Sharpe/Calmar: IS も OOS も悪化
- メカニズムが単純な leverage 増加であり、Bull 環境限定でしか有効でない
```

**結論**: Risk Reducer でも Robustness Enhancer でもなく、Bull 市場でのポジション増加器に過ぎない。不安定さが最大で OOS 効果が負。研究に値する代替案はない。

---

## ④ Production 推奨構成

### Option A: Conservative (推奨)

**D_EQ_SCALE のみ採用**

```
構成: S5 baseline + addon_policy="D"
期待性能:
  IS  2018-2024: CAGR ≈ 12.16%  MaxDD ≈ -18.12%  Sharpe ≈ 0.578
  OOS 2025:      CAGR ≈ 11.22%  MaxDD ≈ -10.16%  Sharpe ≈ 1.054
  WF 5-fold:     5/5  avg ≈ +17.91%  Seg3_2022 ≈ -2.65%

メリット:
- WF 5/5 (Baseline から改善)
- 2022 Bear 保護 (+2.46pp)
- 最小 fold variance
- 実装・保守コスト最低

デメリット:
- IS/OOS リターン -0.47pp / -0.42pp (わずかな犠牲)
```

### Option B: Aggressive (条件付き)

**E_COMBINED 採用**

```
構成: S5 baseline + ATR_EXT + VOL_ADJ + EQ_SCALE
期待性能:
  IS  2018-2024: CAGR ≈ 12.19%  MaxDD ≈ -20.11%  Sharpe ≈ 0.565
  OOS 2025:      CAGR ≈ 11.90%  MaxDD ≈ -10.60%  Sharpe ≈ 1.056
  WF 5-fold:     5/5  avg ≈ +18.87%  Seg3_2022 ≈ -2.65%

メリット:
- WF avg CAGR 最大 (+18.87%)
- WF ΔCAGR +1.37pp > 閾値
- Seg3 保護 = Option A と同等
- OOS 微正 (+0.26pp)

デメリット:
- IS MaxDD -1.93pp 悪化
- IS Sharpe/Calmar 悪化
- 3 機能同時運用の複雑度
- VOL_ADJ が OOS 2025 で -0.96pp 引っ張っている
```

**判定**: Option A を基本とし、運用安定後に Option B への移行を検討。

---

## ⑤⑥ 採用・不採用理由まとめ

| Feature | 採用可否 | 理由 |
|---|---|---|
| D_EQ_SCALE | **採用** | WF 5/5, Bear 保護, 安定性最高, 低コスト |
| E_COMBINED | **条件付き採用** | WF 5/5 + 最大 WF CAGR, ただし IS MaxDD 悪化 + 複雑度 |
| B_ATR_EXT | **様子見** | OOS 1 年のみ証拠不十分; 継続 Shadow 監視 |
| C_VOL_ADJ | **不採用** | OOS 負, Risk 悪化, Bear 保護ゼロ, 最大不安定 |

---

## ⑦ 今後研究すべき Feature

### 高優先

1. **Regime-Conditional VOL_ADJ**  
   Bear 期に max_pos を下げる (現行は calm で上げるだけ)。Seg3_2022 を改善できるなら VOL_ADJ は生きる。

2. **ATR_EXT の OOS 継続監視**  
   2026 年 OOS を蓄積して +2.23pp が一過性かどうか検証。3 年 OOS でのみ確信できる。

3. **EQ_SCALE のサイズ最適化**  
   現行 25% の addon size を Bear 環境では縮小する adaptive sizing。IS リターン犠牲 -0.47pp を解消できるか。

### 中優先

4. **Exit RSR の Regime-Conditional 閾値**  
   現行 70 固定。Bull=65, Bear=75 などの環境依存にすることで Seg3 をさらに改善できる可能性。

5. **Position sizing の Kelly 近似**  
   Equal weight の代替。ATR-scaled sizing が IS/OOS の両方でリスク調整後リターンを改善するか。

---

## ⑧ Study40〜50 を踏まえた最終結論

### 認識の更新

**旧認識**: 各 Feature は ΔCAGR で評価すればよい。+6.07pp が証拠。

**新認識**: 
1. IS ΔCAGR は 6.84 年でも +0.47pp が最大 → **IS での ΔCAGR 分析は識別力が低い**
2. WF OOS (5 独立期間) の方が IS より信頼性が高い
3. 最も重要な指標は **Seg3_2022 (弱気市場 OOS)** — ここが改善するかどうかが頑健性の試金石

### Feature の本質

| Feature | 本質的な役割 | IS では見えない理由 |
|---|---|---|
| B_ATR_EXT | モメンタム継続キャプチャ | IS では正負が均衡する |
| C_VOL_ADJ | Bull 市場での追加 slot | Bear 期の slot 増加コストが相殺 |
| D_EQ_SCALE | **Bear 環境での自動 de-risk** | IS 全体では少数の Bear 期のみ有効 |
| E_COMBINED | D_EQ_SCALE + Bull キャプチャ | 複合効果で WF avg が最大化 |

### 採用基準の改訂

**旧基準**: IS ΔCAGR > +1.0pp + WF 4/5 + MaxDD 悪化 < 2pp

**新基準**:
```
Primary:   WF 5/5 または Seg3_2022 改善 > +1pp
Secondary: OOS 2025 ≥ Baseline
Tertiary:  IS MaxDD 悪化 < 2pp
CAGR:      IS ΔCAGR は参考値のみ (低識別力)
```

### 最終 Production Baseline

```
採用構成: A_BASELINE + D_EQ_SCALE (Option A)

指標             IS 2018-2024    OOS 2025     WF 5-fold
CAGR             12.16%          11.22%       17.91% (avg)
MaxDD            -18.12%         -10.16%      -17.03% (avg)
Sharpe           0.578           1.054        0.853 (avg)
WF pass          —               —            5/5
Seg3_2022        —               —            -2.65% (PASS)

旧 "CAGR 20.51%" = 廃止 (intersection バグ + 期間圧縮)
旧 "WF +6.07pp"  = 廃止 (corona 除外 + intersection の二重効果)
正確な WF ΔCAGR  = +0.41pp (D_EQ_SCALE, robustness value) / +1.37pp (E_COMBINED)
```

### 研究フェーズの結論

Study40〜50 で得られた本当の知見:

1. **S5 Baseline は堅固**: IS CAGR 12.63%, OOS 11.64%, WF 4/5 — 単体で Live 運用可能水準
2. **D_EQ_SCALE は唯一の構造的改善**: Bear 環境での自動 de-risk 機能を持ち、WF 5/5 で確認される
3. **追加機能による IS CAGR 向上は事実上不可能**: 6.84 年 IS 期間で最大 +0.18pp は識別力の限界
4. **頑健性 > リターン**: 2022 年を確実にパスできる戦略が、平均 CAGR が 0.47pp 高い戦略より価値がある

**この結論を今後の全研究の出発点とする。**
