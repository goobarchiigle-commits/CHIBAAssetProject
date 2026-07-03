# Study52 — Production Candidate Optimization
日付: 2026-06-28  
目的: VOL_ADJ 完全除外のうえで ATR_EXT × EQ_SCALE の最終比較  
データ: Study50 キャッシュ (A/B/C) + 新規実行 (D_ATR_EQ)

---

## 全数値 — IS / OOS / WF

### IS 2018-2024

| Case | CAGR | MaxDD | Sharpe | Calmar | Trades | Exposure | ΔCAGR |
|---|---|---|---|---|---|---|---|
| A_BASELINE | +12.63% | -18.18% | 0.593 | 0.695 | 264 | 31.7% | — |
| B_ATR_EXT | +12.81% | -18.18% | 0.600 | 0.705 | 263 | 31.8% | **+0.18** |
| C_EQ_SCALE | +12.16% | -18.12% | 0.578 | 0.671 | 264 | 31.3% | -0.47 |
| **D_ATR_EQ** | **+12.37%** | **-18.12%** | **0.586** | **0.683** | **263** | **31.3%** | **-0.26** |

### OOS 2025

| Case | CAGR | MaxDD | Sharpe | Calmar | ΔCAGR |
|---|---|---|---|---|---|
| A_BASELINE | +11.64% | -10.16% | 1.084 | 1.146 | — |
| B_ATR_EXT | +13.87% | -10.15% | 1.225 | 1.367 | **+2.23** |
| C_EQ_SCALE | +11.22% | -10.16% | 1.054 | 1.104 | -0.42 |
| **D_ATR_EQ** | **+13.48%** | **-10.15%** | **1.199** | **1.329** | **+1.84** |

### WF 5-fold

| Case | WF | avg CAGR | avg Sharpe | avg MaxDD | Fold std | Seg3_2022 | ΔCAGR_WF |
|---|---|---|---|---|---|---|---|
| A_BASELINE | 4/5 | +17.50% | 0.833 | -16.81% | 19.14 | **-5.11%** | — |
| B_ATR_EXT | 4/5 | +18.02% | 0.858 | -16.70% | 19.86 | -5.11% | +0.52 |
| C_EQ_SCALE | **5/5** | +17.91% | 0.853 | -17.03% | **18.79** | **-2.65%** | +0.41 |
| **D_ATR_EQ** | **5/5** | **+18.37%** | **0.876** | -16.91% | 19.41 | **-2.65%** | **+0.87** |

### WF 各 Fold

| Fold | A_BASELINE | B_ATR_EXT | C_EQ_SCALE | D_ATR_EQ |
|---|---|---|---|---|
| OOS 2020 | +6.12% ✓ | +6.12% ✓ | +5.66% ✓ | +5.66% ✓ |
| OOS 2021 | +6.04% ✓ | +6.04% ✓ | +5.38% ✓ | +5.38% ✓ |
| **OOS 2022** | **-5.11% ✗** | **-5.11% ✗** | **-2.65% ✓** | **-2.65% ✓** |
| OOS 2023 | +43.82% ✓ | +46.39% ✓ | +42.59% ✓ | **+44.90% ✓** |
| OOS 2024 | +36.64% ✓ | +36.64% ✓ | +38.58% ✓ | +38.58% ✓ |

**OOS 2022 は C/D のみ PASS。ATR_EXT 単体 (B) は 2022 Bear に無効。**

---

## 2022 Bear 環境の詳細分析

| Case | Seg3 CAGR | Sharpe | MaxDD | PASS? |
|---|---|---|---|---|
| A_BASELINE | -5.11% | -0.029 | -20.93% | ✗ |
| B_ATR_EXT | -5.11% | -0.029 | -20.93% | ✗ (同値) |
| C_EQ_SCALE | **-2.65%** | **+0.057** | -20.93% | **✓** |
| D_ATR_EQ | **-2.65%** | **+0.057** | -20.93% | **✓** |

**結論: 2022 Bear への有効性は EQ_SCALE が担う。ATR_EXT は Bear 耐性に寄与しない。**  
D_ATR_EQ は EQ_SCALE の Bear 保護を完全継承。

---

## Feature Cost

| Case | State files | Params | Complexity | 主要リスク |
|---|---|---|---|---|
| A_BASELINE | 0 | 0 | なし | なし |
| B_ATR_EXT | 1 | 2 | 低 | 退出タイミングの defer |
| C_EQ_SCALE | 1 | 3 | 低 | 追加発注の重複防止 |
| **D_ATR_EQ** | **2** | **5** | **中** | **exit defer + 追加発注の組み合わせ** |

D_ATR_EQ = C_EQ_SCALE に ATR_EXT を追加した増分コスト。  
E_COMBINED (旧) = state files 3 / params 7 / VOL_ADJ 込みの最大複雑度。  
**D_ATR_EQ は E_COMBINED より低コストで優れた結果を出す。**

---

## Feature 役割判定

### B_ATR_EXT
- **Return Enhancer (OOS momentum)**
- OOS 2025: +2.23pp — モメンタムの継続を捕捉する確実な OOS シグナル
- WF では Seg4/5 Bull 環境でのみ効果 (Seg4 2023: 43.82% → 46.39%)
- Bear 耐性: ゼロ (Seg3 2022 不変)
- MaxDD: 変化なし
- **役割: OOS での勝ちトレードをより長く保持する機能**

### C_EQ_SCALE
- **Robustness Enhancer (Bear de-risk)**
- WF 5/5 達成 — Baseline の 4/5 から改善
- Seg3_2022: -5.11% → -2.65% (+2.46pp) — 弱気市場での含み益基準未達 → アドオン自然抑制
- IS/OOS リターン: 微負 (-0.47pp, -0.42pp)
- MaxDD: 変化なし
- **役割: Bear 環境での曝露増加を構造的に抑制する機能**

### D_ATR_EQ (S5 + ATR Extension + EQ_SCALE)
- **Return Enhancer + Robustness Enhancer**
- EQ_SCALE の Bear 保護を完全継承: WF 5/5、Seg3_2022 -2.65%
- ATR_EXT の OOS momentum 捕捉を加算: OOS 2025 +13.48% (+1.84pp vs Baseline)
- WF avg: 18.37% (C_EQ_SCALE 17.91% より +0.46pp 向上)
- MaxDD: -18.12% (Baseline とほぼ同値、E_COMBINED -20.11% より大幅に良好)
- **役割: Bear 保護 (EQ_SCALE) + OOS momentum 捕捉 (ATR_EXT) の両立**

---

## VOL_ADJ 除外の検証結果

Study50 E_COMBINED (ATR+VOL+EQ) と Study52 D_ATR_EQ (ATR+EQ) の比較:

| 指標 | E_COMBINED | D_ATR_EQ | 差 |
|---|---|---|---|
| IS CAGR | +12.19% | **+12.37%** | +0.18pp |
| IS MaxDD | -20.11% | **-18.12%** | **+1.99pp 改善** |
| IS Sharpe | 0.565 | **0.586** | +0.021 |
| OOS CAGR | +11.90% | **+13.48%** | **+1.58pp 改善** |
| OOS Sharpe | 1.056 | **1.199** | +0.143 |
| WF avg | +18.87% | +18.37% | -0.50pp |
| WF count | 5/5 | 5/5 | 同等 |
| Seg3_2022 | -2.65% | -2.65% | 同等 |
| State files | 3 | **2** | 保守コスト削減 |

**VOL_ADJ を除去することで IS MaxDD +1.99pp 改善、OOS +1.58pp 改善。WF と Bear 保護は同等。**  
VOL_ADJ は E_COMBINED の足を引っ張っていた。

---

## 採用判定 (Study51 新基準)

| Case | Primary (WF 5/5 or ΔSeg3>1pp) | ΔMaxDD_IS | OOS | 判定 | 役割 |
|---|---|---|---|---|---|
| B_ATR_EXT | ✗ (WF 4/5, Seg3 不変) | 0.00 | +2.23pp | **REJECT** | Return Enhancer のみ |
| C_EQ_SCALE | ✓ (WF 5/5, ΔSeg3=+2.46pp) | +0.06 | -0.42pp | **ADOPT** | Robustness Enhancer |
| **D_ATR_EQ** | **✓ (WF 5/5, ΔSeg3=+2.46pp)** | **+0.06** | **+1.84pp** | **ADOPT** | **Return + Robustness** |

**B_ATR_EXT 単体は REJECT**: WF が改善しない (4/5)、Bear 保護なし。OOS 2025 の +2.23pp は有望だが 1 年サンプルのみで Primary 基準を満たさない。ただし D に組み込まれた形では有効。

---

## Production Recommendation — 1 構成のみ

### FINAL: D_ATR_EQ (S5 + ATR Extension + EQ_SCALE)

```
構成:
  exit_policy       = "A"    (ATR Extension, defer=5d)
  max_positions_ts  = None   (VOL_ADJ 除外)
  addon_policy      = "D"    (EQ_SCALE, size_frac=0.25, atr_mult=1.0, max_per_pos=1)

期待成績:
  IS  2018-2024:  CAGR=12.37%  MaxDD=-18.12%  Sharpe=0.586  Calmar=0.683
  OOS 2025:       CAGR=13.48%  MaxDD=-10.15%  Sharpe=1.199  Calmar=1.329
  WF 5-fold:      5/5  avg_CAGR=+18.37%  Seg3_2022=-2.65%
```

### 採用理由

1. **WF 5/5 達成** — Baseline 4/5 から改善。2022 年弱気市場を合格水準に引き上げた。

2. **Bear 保護** — EQ_SCALE が Seg3_2022 を -5.11%→-2.65% (+2.46pp) に改善。弱気環境でアドオンが自然抑制されるメカニズム。

3. **OOS momentum 捕捉** — ATR_EXT が OOS 2025 を +13.48% (+1.84pp vs Baseline) に引き上げ。Seg4 2023 でも 42.59%→44.90% (+2.31pp)。

4. **MaxDD が優良** — E_COMBINED -20.11% に対し D_ATR_EQ -18.12% (1.99pp 改善)。VOL_ADJ 除外の最大のメリット。

5. **EQ_SCALE 単体 (C) より優位** — OOS +13.48% vs +11.22% (+2.26pp)、WF avg +18.37% vs +17.91% (+0.46pp)、IS CAGR +12.37% vs +12.16%。MaxDD は同等。

### 不採用理由

| Feature | 不採用理由 |
|---|---|
| VOL_ADJ | IS MaxDD 悪化 (-1.99pp)、OOS 2025 負 (-0.96pp)、Bear 保護ゼロ、Fold variance 最大。除外することで D が E より良くなることが実証された。 |
| B_ATR_EXT 単体 | WF 4/5 (Primary 基準未達)、Bear 耐性なし。D に組み込まれた形でのみ有効。 |
| E_COMBINED | D_ATR_EQ に比べ IS MaxDD -1.99pp 悪化、OOS -1.58pp 悪化、複雑度増加。VOL_ADJ が足を引いている。 |

---

## 今後研究すべき項目 (Study51 引継ぎ)

1. **ATR_EXT の OOS 継続監視** — 2026 年 OOS で +2.23pp シグナルが継続するか。3 年で統計的確認。
2. **EQ_SCALE size_frac 最適化** — 0.25 固定を Bear/Bull 環境で adaptive にできるか。IS リターン犠牲 (-0.47pp) を解消できる可能性。
3. **Regime-Conditional max_pos** — VOL_ADJ の代替として Bear 期に max_pos を 3→2 に下げる方向 (calm 期上げではなく Bear 期下げ)。

---

## 結論

```
Production Configuration (最終確定):
  D_ATR_EQ = S5 + ATR Extension + EQ_SCALE
  VOL_ADJ = 完全除外 (有害確認)

IS  2018-2024: CAGR=12.37%  MaxDD=-18.12%  Sharpe=0.586
OOS 2025:      CAGR=13.48%  MaxDD=-10.15%  Sharpe=1.199
WF:            5/5  avg=+18.37%  Seg3_2022=-2.65% (PASS)

旧 "CAGR 20.51% / WF +6.07pp" 廃止
正確な Production IS CAGR = 12.37% (D_ATR_EQ, 2018-2024, 6.84yr)
```
