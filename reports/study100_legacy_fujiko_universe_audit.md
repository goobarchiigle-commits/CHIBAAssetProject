# Study100 — Legacy Fujiko + Dynamic Universe Audit（Phase1・2026-07-16）

目的: (1)Dynamic Universe Generator健全性監査 (2)selection alpha寄与度推定 (3)旧フジコ法真の期待値推定 (4)運用継続可否。
本書=Phase1(監査)+Promotion Logic Audit。**ゲート判定=FATAL(名簿形成層)につきPhase2 BTは非続行**（ユーザー指定プロトコル「FATALが出た場合のみUniverse修正を先に」準拠）。

## 前提: 「動的ユニバース」は4系統存在する

| 系統 | 実体 | 状態 |
|---|---|---|
| U-1 静的RSR42名簿 | `configs/rsr_universe_42.csv`（42銘柄固定） | **本番Live使用中** |
| U-2 名簿内動的選抜 | `dyn_rsr42_bear_rs0`（42内から月次Bull30/Bear20選抜） | 本番採用（2026-04-05） |
| U-3 Dynamic RSR42 v1 | 広域プールからtrailing 12M composite上位42を月次ローテ | 研究系・**不適格確定済み**（ユーザー確定・Study75E/76D/77） |
| U-4 Universe C | ルールベースPIT（約1,100銘柄・月次） | 研究基盤（Study75A） |

「本番の動的ユニバース」= U-1(プール) × U-2(内側選抜)の二層構造。

## Phase1 出力1 — Universe生成因子一覧

| factor_name | factor_definition | implemented_file | implemented_date |
|---|---|---|---|
| static_rsr42_membership | G27+15追加。G27=2018-2024バックテストでSharpe>0.3∧MaxDD<30%を通過した銘柄（形成窓=評価窓） | `configs/rsr_universe_42.csv` | 2026-03-22 |
| mom_63 | 63日モメンタム（Bullスコア0.40） | `src/strategy/universe.py` | 2026-04-05 |
| rsr | IBD式12M加重複合リターンのプール内パーセンタイル（Bull 0.35/Bear 0.30） | `backtest/rsr.py`・`src/data/make_dataset.py` | 初期 |
| log_vol | 出来高流動性（Bull 0.25/Bear 0.20） | `src/strategy/universe.py` | 2026-04-05 |
| rs_topix | 対TOPIX相対強度（Bear 0.50・rs>0フィルタ必須） | `src/strategy/universe.py` | 2026-04-05 |
| regime_gate | TOPIX MA200 + sustained_bear_days=40/60 | `src/strategy/universe.py` | 2026-04-05 |
| dynamic_v1_trailing_composite | trailing 12M composite上位42月次ローテ（研究系） | `src/backtest/study75_universe_generator.py`系 | 2026-07 |
| universe_c_rules | 流動性・規模等の機械的規則（PIT） | `src/backtest/study75_universe_generator.py` | 2026-07-04 |

## Phase1 出力2 — 因子分類

| 分類 | 該当因子 |
|---|---|
| A: Execution Filter | log_vol、universe_c_rules（ADV・価格帯・上場日数） |
| B: Alpha Factor | mom_63、rsr、rs_topix、regime_gate、dynamic_v1_trailing_composite |
| C: Future Leakage | **コード上は該当なし**（下記出力3参照） |
| D: Potential Survivorship / Hindsight | **static_rsr42_membership** — in-sample実現成績で選定＝「勝った銘柄だけ残る構造」そのもの |

## Phase1 出力3 — 参照時点監査

| 層 | 実装 | 参照時点 | 判定 |
|---|---|---|---|
| 月次選抜（U-2） | `build_sym_active_df`: 月T選択=月T-1末データ（`eval_dt=pos-1`・`hist=.loc[:eval_dt]`） | t-1 | SAFE |
| 日次シグナル | `composite_alpha_bt.py` L2042: `alpha_df.shift(1)` | t-1 | SAFE |
| 特徴量rolling | atr20_med90/high200等 `.shift(1)` | t-1 | SAFE |
| 執行価格 | 翌日寄付（2026-07 M1 PATCH・BT/Live parity） | t+1 open | SAFE |
| RSR計算 | `shift(63/126/189/252)`過去のみ | t以前 | SAFE |
| **名簿形成（U-1）** | 2018-2024実現成績スクリーニング→同窓で評価 | **形成時点から見て将来を含む全期間** | **FATAL**（研究評価文脈） |

## Promotion Logic Audit（追加タスク）

| # | 項目 | U-1 静的42名簿 | U-2 名簿内選抜 | U-3 Dynamic v1 | U-4 Universe C |
|---|---|---|---|---|---|
| 1 | promotion条件 | 2018-2024 BT成績（Sharpe>0.3∧MaxDD<30%）+裁量15追加 | 月次スコアTop30/20 | trailing 12M composite Top42 | 流動性規則のみ |
| 2 | 参照時点 | **形成窓=評価窓** | t-1 | t-1（月境界1ヶ月ラグ・mismatch=0検証済み・2026-07-13病理診断） | t-1 |
| 3 | future return使用 | **YES**（実現リターンで選定） | NO | NO | NO |
| 4 | 疑い箇所 | `rsr_universe_42.csv`形成プロセス（G27系譜）のみ。コード内リークなし | なし | 測定側にwarm-upアーティファクト（76D: RunB/RunB_fixed両方非クリーン） | なし |
| 5 | 判定 | **FATAL**（BT評価）/ QUESTIONABLE（forward凍結名簿として） | **SAFE** | SAFE（タイミング）・QUESTIONABLE（測定） | **SAFE** |

**ユーザー指摘への回答**: 「RSR上位をUniverse化→その中からRSR75を買う」構造（U-2層）は監査の結果リークなし・SAFE。問題は Observation→Promotion がU-1層で「2018-2024に勝った銘柄だけを2018-2024の評価対象に残す」構造になっている点に集中する。これは将来情報のコード的リークではなく、**名簿形成時のhindsight選定**。

## Selection alpha定量（CaseB−CaseC相当・Study75C E1既測値を引用）

CaseB(現行42) vs CaseC(無作為化ユニバース)の比較はStudy75C（2026-07-11・PIT twin bootstrap・fresh run）が同一設計で実施済み。再実行は重複のため省略（リミット節約・ユーザー方針）。

| 指標 | 値 |
|---|---|
| 選定バイアス（U0'−twin median） | **+12.26pp** |
| 複合バイアス | +11.30pp |
| 純粋survivorshipバイアス | ≈0（RSR42帯の上場廃止はTOB/MBO型で微増益要因） |
| 同帯PIT無作為ユニバース期待値（IS） | **-2.5%**（配当補正後≈-1〜0%） |
| OOSパーセンタイル退行 | 95%→70%（hindsight選定エッジの窓外減衰と整合） |

**Universe_alpha ≈ +11〜12pp（IS窓）・OOSで減衰中。**

## Selection alphaの実運用上の弊害（設問回答）

1. **期待値の過大評価**: 名目IS 12.22%/OOS 11.42%（2026-07 M3 fresh run）のうち約11ppが選定バイアス→補正後期待CAGR≈0〜5%。増資・拡大判断を誤る主因
2. **固定名簿の陳腐化**: エッジが減衰しても（OOS 95→70%ile退行が既に観測）名簿を入れ替える機構がない。入替には再びhindsight選定するしかないという自己矛盾
3. **リスク較正の誤り**: 想定MaxDD・RoRがバイアス込み楽観値でCIRCUIT閾値(-15%)の実効余裕を過大評価
4. **ベンチマーク汚染**: 新手法をRSR42上の名目値と比較すると+11ppの下駄と戦うことになり、真に優れた手法をREJECTする（比較不能）
5. **候補枯渇**: 42名固定×RSR75で候補/日0.2-0.3・exposure低迷→資本効率が構造的に上がらない

## ゲート判定と結論

- **Phase1判定: FATAL（U-1名簿形成層・hindsight選定・定量済み+11-12pp）**
- ユーザー指定プロトコルに従い**Phase2 BT（現行ユニバース上）は非続行**。CaseBの測定値は事前にバイアス込みと分かっており、コスト対情報価値ゼロ
- CaseB−CaseC定量はStudy75C E1で充足。**Phase2で未充足の唯一の未知=CaseA（中立ユニバース上の旧フジコ法）**

### 最終結論（タスク指定5項目・Phase1時点）

1. **旧フジコ法単体価値**: 選定アルファ未証明（補正後0〜5%）。ただしエンジン・Exit群・BT/Live parity・執行安全機構はA級資産（FUJIKO 2.0 Part1分類と整合）
2. **Dynamic Universe価値**: U-2内側選抜ロジックは健全（SAFE）だが乗っているプール（U-1）が汚染。U-3は不適格確定済み。U-4のみクリーン基盤
3. **組み合わせ価値**: 現行（U-1×フジコ法）の名目値は意思決定根拠として使用禁止。真の組み合わせ価値はPhase2'（下記）まで不明
4. **運用継続**: **条件付きYES** — 即時売買停止は不要（CIRCUIT/DD監視は選定バイアスと独立に機能・¥3M限定）。増資・拡大・成績を根拠とした判断は禁止（Study95時の整理を維持）
5. **研究優先順位**: ①Phase2'（修正版CaseA）②PEAD/TSMOM factor-level ③テーマ・牽引銘柄研究は凍結

## Phase2' 修正提案（ユーザー決裁待ち・ASK_FIRST）

Universe修正を先に行う場合の設計:

1. **CaseA' = Universe C流動性上位500（PIT・月次）**上で旧フジコ法完全固定を実行
   - TOPIX500現在スナップショットは不採用（survivorship持ち込み・Universe CはPIT担保済み+ADV20既存成果物あり）
   - 静的（月次全入替なし・年次固定等）にすれば76D warm-upアーティファクト非該当
2. **意味論警告（仮説4）**: min_rsr=75はプール相対のためpool=500では42時と別戦略化する。①そのまま実行（「移植した場合」の答え）②絶対スコア版（Clenow等）併走、の2本を対にして初めて解釈可能
3. 判定基準はタスク指定のまま: GREEN CAGR>8%∧Calmar>0.6 / YELLOW 5-8% / RED <5%
4. 工数 ~1-2週（jquants→FujikoStrategyブリッジはStudy76系で既存）

---
*生成: Study100 Phase1監査, 2026-07-16。新規BTなし（コード・文書監査+既測値引用のみ）。引用: Study75C E1（2026-07-11 fresh）・Study76D・Study77・2026-07-13病理診断・2026-03-22 RSRコンテキスト診断。*
