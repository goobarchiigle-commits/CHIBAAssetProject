# Production Strategy Comprehensive Audit — 2026-07-02

**エビデンス範囲**: Study33/34/50/52/70/71/72/73 結果JSON + `composite_alpha_bt.py` コード実測 + `strategy.yaml`。新規バックテスト実行なし（新規スクリプト=ASK_FIRST対象のため）。

## 結論（要約）

- **真の利益源泉**: ①Baseline構造（composite alpha ranking + Turtle + RSRシステム = IS 12.63%の土台）②ATR Extension（OOS +2.26pp、唯一のfeature級Critical）③Dynamic Universe（ΔCAGR +0.77pp/ΔSharpe +0.055、ただしΔDD -2.54ppの対価）
- **最大の構造リスク**: レジーム依存（利益の大半が2023-2024 Bull集中、WF Fold4+5でavg押上げ）とサバイバル/選択バイアス（CAGR 1〜3pp過大の既知問題、未解消）
- **再現性**: D_ATR_EQ本体=A（誤差0.0pp）。ただしF1(RSR70)採用根拠は旧エンジン固有で**INVALID確定**、Study52キャッシュ汚染がStudy70-72の誤判定を連鎖させた（Study73で訂正済み）
- **未決着**: CAND_B（rsr_exit 70→75）。リターン-1〜2.7pp vs 2022年+5.02pp/WF5/5/Bootstrap P(>0)=100%。**ダウンサイド重視なら採用推奨**

---

## Task 1: Edge Attribution Audit

### 要素別寄与（Study71 LOO + Study73 fresh run + Study33/34実測）

| 要素 | 実装 | 単独寄与（除去時Δ） | 出典 | 分類 |
|---|---|---|---|---|
| Universe Selection | dyn_rsr42_bear_rs0 + Bear 7セクター除外 | CAGR +0.77pp / Sharpe +0.055 / **DD -2.54pp（悪化）** | Study33/34 | **Critical** |
| Entry Filter | RSR≥75 + composite alpha (slope×r2)²×RSR | Baseline本体（分離LOB未実施） | Study50 baseline 12.63% | **Critical** |
| Entry Timing | entry_timing boost_weight=0.06 | IC全特徴量<0.065（微弱） | Study54 | Optional |
| Exit (RSR70) | rsr_exit=70 vs 75 | WF avg +1.35pp / **2022 -5.88pp** / OOS2025 -0.89pp | Study72 3C | Important（条件付き） |
| Exit (ATR Ext) | defer 5営業日 | **OOS +2.26pp** / IS +0.21pp / WF5/5 | Study71 marginal | **Critical** |
| Position Sizing | equal weight (cash/slots) | existing→equal: リターン改善/DD -0.86pp | Study33/34 | Important |
| Capital Allocation (EQ Scale) | addon 25% | IS +0.54pp / OOS +2.64pp / 2022 +2.95pp（Study73 fresh） | Study73 Phase1 | Important |
| Risk Control (Shock Exit) | composite mode | WF検証済・定量LOB未実施 | Study71 F6 score=0.7 | Important（**定量不明**） |
| Execution Logic | 翌日寄付+cost 0.155% | BT前提そのもの | コード実測 | **Critical** |

### 相互作用

- ATR_EXT × EQ_SCALE = IS/OOS +0.03pp → **ほぼ加法的**（Study71 phase5）
- Universe × Bear Filter = 密結合（分離不能、Study71 F4/F5）
- ⚠️ Study71のF3限界寄与(-0.44pp)はStudy52キャッシュ汚染由来。Study73 fresh runで**+0.54pp ISに訂正済み**。F3=KEEP。

### 壊してはいけないコード Top10

| # | 対象 | 根拠 |
|---|---|---|
| 1 | `composite_alpha_bt.py` alpha_df `shift(1)`（L1979） | 先読み防止の生命線。除去=全結果無効 |
| 2 | 同 翌日寄付執行 `open_mat[next_i]`（L1224/L1390） | 執行現実性の土台 |
| 3 | 同 COST_ONE_WAY=0.155%（L67-69） | コスト無しBTは全て幻 |
| 4 | 動的ユニバース選定ロジック（sym_active_mat） | 唯一の確証アルファ源泉 |
| 5 | `atr_extension`ロジック（exit_policy=A, defer5d） | OOS +2.26ppの源泉 |
| 6 | RSR計算（`rsr.py`、パーセンタイルランク） | 全Entry/Exitの基盤 |
| 7 | WF fold構造（rolling 2yr IS） | 採用判定の物差し。変更=過去判定と非互換 |
| 8 | Bear Universe Filter（7セクター除外） | F4と密結合、単独除去不可 |
| 9 | `eq_scale_addon`（Study73でKEEP確定） | 2022年-2.95pp防御 |
| 10 | shock_exit composite | Seg3改善寄与（定量は不明→検証要） |

---

## Task 2: Data Leakage Audit

| 項目 | 状態 | 根拠 | リスク |
|---|---|---|---|
| Lookahead | **対策済み** | alpha_df/atr20_med90/high200全て`shift(1)`、執行は`next_i` | Low |
| **Survivorship Bias** | **未解消・既知** | research_state既知問題表「CAGR 1〜3%過大評価の可能性」。yfinance現存銘柄のみ | **High** |
| **Selection Bias** | **未解消・既知** | RSR42ユニバースはin-sample screening選定（research_state既知問題表） | **High** |
| Close→Open整合 | OK | SELL/BUYとも翌日寄付（L1224/L1390）。research_state既知問題表の「SELL=当日終値」記述は**旧エンジンのstale記述** | Low |
| **Addon執行価格の不整合** | **発見** | `composite_alpha_bt.py` L1663-1666: addonは**翌日close**執行、コメント「新規BUYと同じ」は虚偽（新規BUYは翌日open）。ライブでは寄付執行のはず→BT/Live乖離 | **Medium** |
| **max_single_weight×1.5** | **発見** | L1684-1685: addon経路は`capital×0.25×1.5=37.5%`まで許容。CIRCUIT `max_single_weight=0.25`（変更禁止）と形式矛盾 | **Medium** |
| Split/Dividend調整 | 概ねOK | yfinance調整済価格（TOPIX: "Adj Close"優先 L154）。調整後価格で執行=軽微バイアス | Low |
| Delisting/Corporate Actions | 未対応 | Survivorshipと同根 | High（上記に包含） |
| **fraction.bull=0.0の選定方法** | **疑義** | strategy.yaml L51: 「OOS Sharpe 1.827→2.226で採用」=**OOS成績で選定した形跡**。run_multi_strategy用でfujiko本経路には非適用だが、適用時はOOS選択バイアス | Medium（経路限定） |
| Timezone/Holiday | OK | 実データ日付準拠 + SimpleTSECalendar | Low |
| Missing Data | OK | ffill/dropna明示（L313等） | Low |

**修正案**: Survivorship/Selectionは上場廃止銘柄込みユニバース再構築でしか解消不能。Addon執行価格は`open_mat[next_i]`へ統一しREG確認（PATCH、ASK_FIRST）。

---

## Task 3: Reproducibility Audit

| Feature | 採用時Δ | 現在Δ | 差分 | 原因分類 | 再現性 |
|---|---|---|---|---|---|
| D_ATR_EQ本体 | IS 12.37/OOS 13.48 | 同値（error 0.0pp） | 0 | — | **A** |
| F2 ATR_EXT | +1.84pp OOS | +2.26pp | +0.42 | 誤差内 | **A** |
| F1 RSR70 | +2.72pp WF | +1.35pp / OOS -0.89pp | **-1.37pp** | **Code Drift（エンジン差: multilayer RSR z-scoreが5pt差を代替）+ Fold Drift（expanding→rolling、2025→2020入替）** | **D（INVALID確定）** |
| F3 EQ_SCALE | +2.46pp Seg3 | +0.54pp IS/+2.64pp OOS（fresh） | 判定3転 | **Dataset Drift（Study52キャッシュのaddon_cnt=14混入）** | **C→A（Study73訂正後）** |
| F4/F5/F6 | WF5/5 | 統合済・fresh LOB未実施 | 不明 | — | **B** |

- **Production一致率**: overall_consistency = **0.62**（Study72）。7 Feature中 KEEP4 / INVALID1 / SHADOW1 / 訂正1
- **重大教訓**: Study52キャッシュ値の再利用がStudy70/71/72の3研究連鎖誤判定を生んだ。**キャッシュ由来値でのProduction判定は禁止すべき**

---

## Task 4: Strategy Robustness Audit

**実施済みエビデンス**:

| 手法 | 結果 | 出典 |
|---|---|---|
| WF 5-fold | CURRENT 4/5（2022 FAIL）/ CAND_B(RSR75) **5/5** | Study73 |
| Bootstrap | CURRENT P(>0)=98.8% CI[-]=+1.95% / CAND_B **P(>0)=100%** CI[-]=+2.85% | Study73 Phase3 |
| Fold std | CURRENT 19.41pp / CAND_B **14.78pp** | Study73 |
| Monte Carlo（トレード順シャッフル） | **未実施 → 不明** | — |
| Purged CV | **未実施 → 不明** | — |
| ±5/10/20% 感度sweep | **体系的には未実施 → 不明**（mom_period=21のみ感度確認済み） | strategy.yaml注記 |

**過剰最適化疑い箇所**:

1. `turtle_exit=55`: **IS Sharpe +12.6%を根拠に20→55変更**（strategy.yaml L12）。IS単独指標選定=OVERFIT_GUARD違反類型。ただし「Exit Alpha Audit」でデッドパラメータ判定→実害小だが根拠不健全
2. `fraction.bull=0.0`: OOS成績で選定
3. `quality_replacement.qs_weights`: IS 2018-2024 fitでLOCK（shadow限定なので現状無害）
4. `boost_weight=0.06`: 検証提示なし

**壊れやすい/頑健**: 感度sweep未実施のため正式Top10不能（不明）。エビデンス上、**壊れやすい**=rsr_exit境界（70↔75で2022年±5.9pp振れ）、Fold構造依存(F1)。**頑健**=ATR_EXT（エンジン・Fold変更を跨いで+2pp維持）、D_ATR_EQ再現性（error 0.0pp）、Bootstrap P(>0)≥98.8%。

**採用継続妥当性**: 現構成の維持は妥当。ただしF1はCAND_B移行判断待ち。

---

## Task 5: Regime Attribution Audit

年次=レジーム代理（正式10分類は**不明**、TOPIX系レジームタグはFL analyticsのみ）:

| 年 | レジーム | 年次リターン | MaxDD | WF fold CAGR | 判定 |
|---|---|---|---|---|---|
| 2023 | Strong Bull | **+32.77%** | -6.01% | **+44.90%** | **利益源泉#1** |
| 2024 | Bull | +15.68% | -8.61% | +38.58% | 利益源泉#2 |
| 2021 | Range/Bull | +16.10% | -18.12% | +5.38% | 中位（DD大） |
| 2022 | Bear | +11.61%(IS) | -9.65% | **-2.65%(OOS)** | **最弱#2**（IS/OOS乖離大） |
| 2020 | Crash→Recovery | +8.76% | -16.90% | +5.66% | 中位 |
| 2019 | Sideways | +0.96% | -18.02% | — | **最弱#1** |
| 2018 | Sideways/Bear | -3.61% | -12.68% | — | 最弱#1 |
| 2025 | OOS | +3.98%（暦年） | -3.72% | — | 低活動 |

- **構造**: 戦略はBullモメンタム捕捉機。2023+2024でWF avg 18.37%の大半を稼ぐ。Sideways（2018-19）はほぼゼロ〜負
- **IS/OOS乖離注意**: 2022はIS年次+11.61%だがWF OOSでは-2.65%
- **改善優先**: ①Bear 2022型（CAND_B採用で+5.02pp直接改善可能）②Sideways無収益（新規エッジ必要=難、Roadmap 2026-06-13「新エッジ不要(A判定)」のため優先度下）
- Rising/Falling Rate分類: **不明**（金利データ未統合）

---

## Task 6: Tail Risk Attribution Audit

**実施済み（Study34、DD帰属確定）**:

| 帰属先 | 寄与 | 割合 |
|---|---|---|
| 動的ユニバース | -2.54pp | **69.4%（主因）** |
| sizing equal化（exposure 31→35.2%） | -0.86pp | 23.5% |
| Exit変更 | -0.26pp | 7.1% |
| 集中化仮説 | — | **反証**（4指標全てで低集中） |

**重要発見（Study34）**: 最大DD期間15営業日中の確定損益は**プラス**。DDは未実現評価減が本体 → Stop Failureではなく**保有継続中の含み減**が構造。Gap/VIX/相関spike別のTop20ケース個票分析は**未実施→不明**。

**改善余地**: DD -18.12%の69%はアルファ源泉と表裏一体（Study34: リスク・リターントレードオフであり不具合ではない）→ 除去不能。**実効的な改善レバーはCAND_B（2022 DD構造+5pp）が唯一の実証済み候補**。

---

## Final Report

**1. 利益源泉ランキング**: ①Baseline構造(12.63%) ②ATR_EXT(+2.26pp OOS) ③Dynamic Universe(+0.77pp/Sharpe+0.055) ④EQ_SCALE(+0.54pp IS・防御価値) ⑤RSR70(Bull年+3〜9pp/Bear年-5.9pp条件付き)

**2. 主要リスクランキング**: ①レジーム集中(2023-24依存) ②Survivorship+Selection Bias(CAGR 1-3pp過大) ③F1のFold/エンジン依存性 ④DD構造(Universe起因69%・除去不能) ⑤キャッシュ汚染型の研究プロセスリスク

**3. 改善効果期待 Top**: ①CAND_B採用判断 ②Survivorship-free再検証 ③Addon執行価格統一 ④shock_exit定量LOB ⑤Monte Carlo/感度sweep ⑥Quality Replacement Phase9昇格判断 ⑦Top20 DDケース個票 ⑧turtle_exit根拠再検証 ⑨レジーム正式分類 ⑩entry_timing boost_weight検証

**4. 過剰最適化疑い**: turtle_exit=55（IS単独選定）/ fraction.bull=0.0（OOS選定）/ boost_weight=0.06（未検証）

**5. 信頼性が高い要素 Top**: ATR_EXT / D_ATR_EQ再現性(error 0.0pp) / shift(1)体制 / コストモデル / WF基盤 / Bootstrap P(>0)≥98.8% / Dynamic Universe / Bear Filter / integrity_check 25/25 / Study73の三重検証プロセス

**6. 維持/修正判断**: **維持**（Bootstrap P(>0)=98.8%、OOS 2025 +13.48%、再現性A）。修正すべきは戦略本体ではなく ①F1の決着 ②バイアス定量化 ③研究プロセス（キャッシュ禁止）

**7. ロードマップ**:
- **High**: CAND_B採用判断（ASK_FIRST）/ Survivorship監査
- **Medium**: Addon執行修正 / shock_exit LOB / MC+感度sweep / max_single_weight×1.5整合
- **Low**: レジーム正式分類 / entry_timing検証 / turtle_exit再根拠

**8. 追加研究テーマ（優先順）**: Study74=CAND_B最終移行WF → Study75=Survivorship-free universe再構築 → Study76=MC+Purged CV+感度sweep → Study77=Top20 DD個票 → Study78=shock_exit定量LOB

### Productionへの影響度総括

| 発見 | 影響度 |
|---|---|
| F1採用根拠INVALID（CAND_B未決着） | **High** |
| Survivorship/Selection Bias未解消 | **High** |
| Addon執行価格不整合（BT翌日close vs BUY翌日open） | Medium |
| max_single_weight×1.5 CIRCUIT形式矛盾 | Medium |
| キャッシュ汚染プロセスリスク | Medium（再発防止要） |
| turtle_exit/fraction選定根拠 | Low |
| Critical該当 | **なし**（即時停止要因ゼロ） |

---

## 次ステップ Top3

1. **CAND_B(rsr_exit=75)採用可否の決断** — 判断材料は出揃済み（Study73）。DD耐性・WF5/5・Bootstrap P(>0)=100%を重視し採用推奨。PARAMS_LOCKED隣接→ユーザー承認必須
2. **Addon執行価格のPATCH**（`close_mat[next_i]`→`open_mat[next_i]`+REG）— ASK_FIRST承認待ち
3. **Study75: Survivorship監査**スクリプト設計 — 新規作成ASK_FIRST

## CDX Task

```
TASK-1: composite_alpha_bt.py L1666 _addon_px を open_mat[next_i] に変更
        + コメント修正 + D_ATR_EQ IS/OOS/WF 再実行で差分報告（REG）
TASK-2: strategy.yaml rsr_exit 70.0→75.0（CAND_B採用決定時のみ）
        + run_live_signal.py / signal_bridge.py / live_equivalent.py の閾値定数同期確認
TASK-3: 研究プロトコルに「Production判定にキャッシュ値使用禁止・fresh run必須」を明文化
        （CLAUDE.md OVERFIT_GUARD へ1行追加）
```

## リスク/テスト観点

- TASK-1実行時: addon 14件の執行価格変化がIS/OOS CAGRに与える影響±0.3pp程度と推定（**要実測、推定のまま採用禁止**）
- TASK-2実行時: MORNING_ROUTINE dry-runで既存ポジションのRSR75跨ぎExit発火有無を必ず確認してからLIVE

**補足**: research_state.md既知問題表の「SELL=当日終値」記述が現エンジンと不一致（stale）。次回research_state更新時に訂正推奨。
