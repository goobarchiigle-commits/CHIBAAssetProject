# Study33: Post-ATR Removal Validation

作成日: 2026-06-24
方式: コード実測のみ（推測禁止）。`src/backtest/post_atr_removal_validation_202606.py`実行。
期間: 2018-01-01〜2026-06-23
結果ファイル: `backtests/post_atr_removal_validation_202606_2026-06-24.json`
前提: Study32（`atr_sizing_decision_202606.md`）で`decision=REMOVE confidence=HIGH`確定済み。本Studyはその除去後の真のベースラインを確定する。

---

## ヘッドライン比較

| | A_APRIL_REPRO | B_CURRENT_PROD | C_PROD_MINUS_ATR |
|---|---|---|---|
| 説明 | 旧ベースライン（sizing_mode=existing） | 現行本番（ATR Risk Sizing含む全機能ON） | 現行本番からATR Sizingのみ除去（sizing_mode=equalに代替、Study32準拠） |
| CAGR | +16.1% | +10.4% | **+19.8%** |
| Sharpe | 0.781 | 0.713 | **0.798** |
| MaxDD | -15.7% | -16.3% | -19.4% |
| Calmar | **1.026** | 0.641 | 1.020 |
| PF | 2.277 | 2.020 | **2.420** |
| AvgExposure | 31.0% | 23.6% | 35.2% |
| Trades | 280 | 246 | 220 |

**C(PROD minus ATR)がCAGR・Sharpe・PFで最良**。CalmarはAとほぼ同値（1.020 vs 1.026、差0.006で実質タイ）— Cの方がMaxDDが大きい(-19.4% vs -15.7%)分、CAGRも大きく伸びている。**ATR Risk Sizing除去により、現行本番より明確に優れたベースラインが確定**（CAGR+9.4pp/Sharpe+0.085/PF+0.400 vs B）。

---

## 寄与度分解: APRIL_REPRO → PROD minus ATR

4要因（Universe/ATR Trailing/Multi-layer RSR/MTF Filter）を単独・累積で追加。sizing_modeはAPRIL_REPRO側の"existing"に固定し、4要因以外の変数を排除。

| Step | CAGR | ΔCAGR | Sharpe | ΔSharpe | MaxDD | Calmar | PF | Trades |
|---|---|---|---|---|---|---|---|---|
| 0_APRIL_REPRO | +16.1% | -- | 0.781 | -- | -15.7% | 1.026 | 2.277 | 280 |
| 1_+UNIVERSE | +16.9% | **+0.77pp** | 0.836 | **+0.055** | -18.2% | 0.925 | 2.524 | 251 |
| 2_+ATR_TRAILING | +17.2% | +0.30pp | 0.849 | +0.013 | -18.2% | 0.942 | 2.528 | 255 |
| 3_+MULTILAYER_RSR | +17.2% | **+0.00pp** | 0.849 | **+0.000** | -18.2% | 0.942 | 2.528 | 255 |
| 4_+MTF_FILTER | +15.6% | **-1.56pp** | 0.770 | **-0.079** | -18.5% | 0.844 | 2.293 | 242 |

**補正行（4要因の外、Study32のATR除去代替方式選定によるsizing切替）**:
| 切替 | ΔCAGR | ΔSharpe |
|---|---|---|
| sizing_mode existing→equal（4_+MTF_FILTER → C_PROD_MINUS_ATR） | **+4.12pp** | **+0.028** |

---

## 判定: 各機能の真贋

| 機能 | ΔCAGR | ΔSharpe | 判定 |
|---|---|---|---|
| **Universe**（動的ユニバース+ショックcomposite化+RSR Exit閾値70） | +0.77pp | +0.055 | **真のアルファ**。CAGR・Sharpe共に明確改善。既存ロック済み機能の効果を再確認。 |
| **ATR Trailing Exit** | +0.30pp | +0.013 | **真のアルファ（小）**。方向は一貫してプラスだが効果量は小さい。維持を支持するが優先度は低い。 |
| **Multi-layer RSR Exit** | +0.00pp | +0.000 | **純粋なノイズ（無効）**。CAGR・Sharpeとも完全にゼロ変化 — 既存のシンプルRSR Exit(閾値70)と完全に冗長で、単独では一度も追加的に発火していない。実装上は害もないが効果もない。 |
| **MTF Filter** | **-1.56pp** | **-0.079** | **アルファ破壊的**。明確にマイナス。`fujiko_production_baseline_202606.md`のresearch_priority_3（MTF除外率39%への懸念）を裏付ける実測結果。除外がBUY候補の質的劣化を伴わずに機会のみを削っている。 |

**結論**:
- Universe・ATR Trailingは真にアルファを追加する機能 → 維持を推奨。
- Multi-layer RSR Exitは単独追加効果ゼロのノイズ → 既存シンプルRSR Exitと完全に冗長、削除しても性能に影響しない（簡素化候補）。
- MTF Filterはアルファ破壊的 → 除去または条件緩和を検討すべき最有力候補（次点研究優先度として浮上）。
- ATR Risk Sizing除去 + sizing_mode=equal への切替（Study32準拠）が最大の単独改善要因（+4.12pp、4要因合計+ -0.49pp相当を大きく上回る）— **サイジング方式の選択が、Exit/Universe側の個別機能改善より支配的な影響を持つ**ことが定量的に再確認された。

## 新ベースライン（確定）

```
C_PROD_MINUS_ATR: CAGR=+19.8%  Sharpe=0.798  MaxDD=-19.4%  Calmar=1.020  PF=2.420  AvgExposure=35.2%  Trades=220
```

このベースラインを今後の研究（MTF Filter見直し、Multi-layer RSR削除検討等）の起点とする。WF検証は未実施（本Study対象外）。
