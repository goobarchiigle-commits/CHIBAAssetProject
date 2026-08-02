# Study110B — Future Winner Predictability Audit（実行結果）

**日付**: 2026-07-22
**性格**: 実行結果（Study110A panel再利用のみ・新規計算ゼロ・alpha提案なし・backtestなし）。
**入力**: `backtests/study110a_panel_enriched_2026-07-22.csv`（Study110A出力・再計算なし）
**スクリプト**: `src/backtest/study110b_future_winner_predictability_audit.py`
**出力**: `backtests/study110b_future_winner_predictability_audit_2026-07-22.json`

---

## 0. Decision rule（事前固定・結果を見た後の変更なし）

```
if p>=0.05 or lift<1.5: TERMINAL
else: PROCEED_TO_STUDY112
```

---

## 1. Tier1: Transition Entropy + Permutation Null(N=1000) + Economic Lift

| Horizon | 観測lift | null lift平均±std | p値(lift) | 観測entropy比 | null entropy比平均 | p値(entropy) | n_pairs | 判定 |
|---|---|---|---|---|---|---|---|---|
| 3M | 1.304 | 1.002±0.013 | **0.001** | 0.9893 | 0.999 | **0.001** | 53,750 | **TERMINAL**(lift<1.5) |
| 6M | 1.287 | 1.008±0.014 | **0.001** | 0.9894 | 0.998 | **0.001** | 47,826 | **TERMINAL**(lift<1.5) |
| 12M | 1.208 | 1.014±0.016 | **0.001** | 0.9890 | 0.997 | **0.001** | 36,003 | **TERMINAL**(lift<1.5) |

**核心的発見**: 全horizonで統計的有意性は極めて強い（p=0.001はpermutation test N=1000の分解能限界——1000回中1回もobservedを超える/下回るnull統計量が出なかった。観測lift(1.2-1.3)はnull分布平均(1.00-1.01)からσ換算で約15-20σ乖離）。**「完全にランダム」ではない——構造は確かに存在する**。しかし**経済的な大きさ(lift)がユーザー事前固定の閾値1.5を全horizonで下回る**——「統計的に本物・実務的に小さすぎる」という、最も判定が難しいパターンがそのまま出た。

**Overall verdict: TERMINAL**（決定ルールは`p>=0.05 OR lift<1.5`のOR条件——lift条件が3horizon全てで発動）。

---

## 2. Tier2診断（判定には不使用・参考情報のみ）

### Sector Persistence

liftは業種間で大きくばらつく（12M horizon: エネルギー資源0.44〜その他1.97）。一部業種（その他・銀行・鉄鋼非鉄・医薬品）はlift>1.5に達するが、**n=233〜379と小さく・permutation検定なし（Tier2は診断専用）——survival宣言の根拠にはならない**。H1(条件付き持続性)の芽はあるが、Tier1相当の厳密検定を経ていない。

### Regime Persistence（Bull/Bear）

| Horizon | Above200MA lift | Below200MA lift |
|---|---|---|
| 3M | 1.309 | 1.295 |
| 6M | 1.272 | 1.317 |
| 12M | 1.221 | 1.184 |

**regime間でほぼ差なし**——Study82E・Study95Eに続き4件目のregime-invariance確認。市場サイクルはこの現象を説明しない。

### State Persistence（非対称性・注目点）

| | 3M | 6M | 12M |
|---|---|---|---|
| Q10→Q10（勝者維持率） | 17.2% | 16.6% | 15.5% |
| **Q1→Q1（敗者維持率）** | **20.6%** | **19.7%** | **18.8%** |
| Q10→Q8+（上位3分位維持） | 37.7% | 36.2% | 35.4% |

**敗者(Q1)の方が勝者(Q10)より持続性が高い**（全horizonで一貫）。「未来の勝者」より「今のダメな銘柄がダメなまま」の方が予測しやすい、という非対称性——参考情報だがTier1判定には使用しない。

---

## 3. 結論

```
Overall: TERMINAL
根拠: 3/3 horizonでlift<1.5（p値は全て0.001で統計的有意性は充分）
```

ユーザー指定の仮説ツリーに対する回答:

```
H0: Future winners are mostly stochastic → 統計的には否定(構造は存在・p=0.001)
    しかし economically → 支持に近い(lift<1.5・実務的に活用困難)
H1: conditional persistence (sector/regime依存) → Regimeでは否定・Sectorでは示唆はあるが未検証
```

厳密には「完全にランダム」ではなく「統計的に検出可能だが実務的に弱すぎる持続性」——ユーザー事前固定の判定ルールはこのケースをTERMINAL側に倒す設計になっており、その通りの結果が出た。

---

## 4. 禁止事項遵守確認

alpha探索なし。MLなし。新factorなし。新規データ取得なし（Study110A panel再利用のみ）。backtestなし。

*作成: CLD (Fable 5)・2026-07-22。実行済み・Study110A panel再利用のみ・新規データ取得なし。*
