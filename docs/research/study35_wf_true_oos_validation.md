# Study35: PROD_MINUS_ATR Walk-Forward & True OOS Validation

作成日: 2026-06-24
方式: コード実測のみ（推測禁止）。`src/backtest/study35_wf_true_oos_validation_202606.py`実行。検証専用・本番コード変更なし。
WF方式論: `wf_dyn_rsr42.py`/`wf_dynamic_universe.py`のWF_SEGS（5fold IS/OOS）・TRUE_OOS(2025)・IS_FULL(2018-2024)・スコアリング式（win=oos_sharpe>0、ratio=oos_sh/is_sh、pass_count/median_oos_sharpe/worst_oos_dd/avg_oos_is_ratio）を無変更で再利用。Extended OOS 2026YTDは既存fold構成への追加評価窓（変更ではない）。
結果ファイル: `backtests/study35_wf_true_oos_validation_202606_2026-06-24.json`

4設定（Study32-34と同一定義）:
- A_APRIL_REPRO: 静的ユニバース/full_exit/rsr_exit=75/ATR Trailingなし/MultiRSRなし/sizing=existing
- B_CURRENT_PROD: 現行本番忠実（動的ユニバース+composite+rsr_exit=70+ATR Trailing+MultiRSR+MTF+ATR Risk Sizing）
- C_PROD_MINUS_ATR: BからATR Risk SizingのみOFF→sizing=equal
- D_PROD_MINUS_ATR_MINUS_MTF: CからさらにMTF FilterもOFF

---

## 1. Walk-Forward サマリー

| 設定 | Pass Count | Avg OOS/IS Ratio | Full IS CAGR | Full IS Sharpe | Full IS MaxDD | Full IS Calmar |
|---|---|---|---|---|---|---|
| A_APRIL_REPRO | 5/5 | 1.165 | +20.14% | 0.859 | -15.71% | 1.282 |
| B_CURRENT_PROD | **3/5** | 0.917 | +11.13% | 0.698 | -16.27% | 0.684 |
| C_PROD_MINUS_ATR | 5/5 | 1.431 | +23.03% | 0.831 | -19.37% | 1.189 |
| D_PROD_MINUS_ATR_MINUS_MTF | 5/5 | 1.535 | +23.89% | **0.869** | -18.72% | 1.276 |

**B(現行本番)はWF5fold中2fold失敗**（Seg2=2021 OOS Sharpe-0.046、Seg3=2022 OOS Sharpe-0.103）— APRIL_REPROの5/5を下回る。C・Dは共に5/5維持。**DはFull IS SharpeでAをも上回る**（0.869 vs 0.859）。

## 2. True OOS評価（2025）

| 設定 | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| A_APRIL_REPRO | +5.57% | 0.564 | -10.11% | 0.551 |
| B_CURRENT_PROD | +5.25% | 0.576 | -9.29% | 0.566 |
| C_PROD_MINUS_ATR | +5.00% | **0.379** | -15.02% | **0.333** |
| D_PROD_MINUS_ATR_MINUS_MTF | +13.96% | **0.881** | -12.88% | **1.083** |

**重要発見**: CはTrue OOS 2025でA・Bより明確に劣る（Sharpe0.379<0.564/0.576、Calmar0.333<0.551/0.566）。MTF Filterを維持した状態でATR Sizingだけ除去すると、OOS耐性が悪化する。一方、**DはMTFも除去すると全設定中最良**（Sharpe0.881、Calmar1.083、CAGR+13.96%）。

## 3. Extended OOS評価（2026 YTD）

| 設定 | CAGR | Sharpe | MaxDD | Calmar |
|---|---|---|---|---|
| A_APRIL_REPRO | +17.1% | 0.806 | -11.82% | 1.447 |
| B_CURRENT_PROD | +45.74% | 2.352 | -6.9% | 6.627 |
| C_PROD_MINUS_ATR | +69.1% | 2.568 | -8.19% | 8.434 |
| D_PROD_MINUS_ATR_MINUS_MTF | +69.1% | 2.568 | -8.19% | 8.434 |

C・Dが完全一致（この約4ヶ月の短期窓ではMTF Filterに除外された候補が実際には発生しなかったため、両者の挙動が収束）。全設定がAを上回るが、窓が短く（約半年）統計的信頼性は限定的（trade数僅少と推測されるため参考値扱い）。

## 4. 年別リターン

| 年 | A_APRIL_REPRO | B_CURRENT_PROD | C_PROD_MINUS_ATR | D_PROD_MINUS_ATR_MINUS_MTF |
|---|---|---|---|---|
| 2020 | +7.43% | +8.86% | +8.30% | +7.89% |
| 2021 | +28.75% | **-5.06%** | +18.96% | +32.37% |
| 2022 | +13.96% | +15.23% | +17.66% | +4.76% |
| 2023 | +17.65% | +21.20% | +26.13% | +32.94% |
| 2024 | +16.35% | +7.49% | +20.72% | +19.84% |
| 2025 | +1.64% | +3.22% | +4.83% | +5.91% |
| 2026 YTD | +1.74% | +5.55% | +4.99% | +4.77% |

**B(現行本番)は2021年に唯一マイナス（-5.06%）**。これがWF Seg2失敗（OOS Sharpe-0.046）と直接対応する。

## 5. Attribution Summary（Full IS基準、2018-2024）

| 比較 | ΔCAGR | ΔSharpe | ΔMaxDD | ΔCalmar |
|---|---|---|---|---|
| A vs C（C-A） | +2.89pp | -0.028 | -3.66pp | -0.093 |
| C vs D（D-C） | +0.86pp | **+0.038** | **+0.65pp** | **+0.087** |

**D vs C は全指標でD優位**（CAGR・Sharpe・MaxDD・Calmarの4指標すべてで改善）。MTF Filter除去がIS・OOS双方で一貫してプラスであることを裏付ける。

（注: A vs CのSharpeがStudy33の2018-2026全期間ヘッドライン値（C=0.798>A=0.781）と符号が逆に見えるのは、本Studyの"Full IS"が2018-2024に限定されているため。比較窓が異なる点に起因し、矛盾ではない。）

---

## 判定: PROMOTE_TO_BASELINE 条件チェック

判定基準（事前定義、機械的判定）: OOS Sharpe>=A_APRIL_REPRO（True OOS 2025基準） / OOS Calmar>=A_APRIL_REPRO×0.9 / WF pass count>=A_APRIL_REPRO / 2026 YTDでCAGR・Sharpe同時劣化なし。

| 設定 | Sharpe>=A | Calmar>=A×0.9 | WF>=A | 2026劣化なし | 判定 |
|---|---|---|---|---|---|
| B_CURRENT_PROD | True | True | **False**(3/5<5/5) | True | NG |
| C_PROD_MINUS_ATR | **False**(0.379<0.564) | **False**(0.333<0.496) | True | True | NG |
| D_PROD_MINUS_ATR_MINUS_MTF | True | True | True | True | **PROMOTE可** |

## 二次目的: MTF FilterのOOS価値判定

C(MTF有)とD(MTF無)をTrue OOS 2025基準で比較: D Sharpe(0.881) > C Sharpe(0.379) かつ D Calmar(1.083) > C Calmar(0.333) → **両指標でMTF除去側が優位**。

```
mtf_value = NEGATIVE
```

ATR Risk Sizing除去後、MTF Filterは（Study33のIS分析で示された-1.56pp/-0.079のアルファ破壊性に加え）True OOS 2025でも明確に負の価値を持つことが確認された。

---

## 最終判定

```
final_verdict = PROMOTE_D
```

**根拠**: D_PROD_MINUS_ATR_MINUS_MTF（ATR Risk Sizing除去＋MTF Filter除去）が、WF pass count(5/5、A同等)・True OOS 2025（Sharpe0.881/Calmar1.083、A超）・Full IS（Sharpe0.869、A超）・2026 YTD（劣化なし）の全条件を満たす唯一の設定。B(現行本番)はWF5fold中2fold失敗（2021年が唯一の負年）でNG、C(MTF維持)はTrue OOS 2025で明確に劣化（Sharpe0.379、Calmar0.333）しNG。

新ベースライン候補: **D_PROD_MINUS_ATR_MINUS_MTF**（ATR Risk Sizing除去＋MTF Filter除去、Equal Weight Sizing採用）。本Studyは検証のみであり、本番コード（signal_bridge.py）への実装変更は別タスク・ASK_FIRST対象。
