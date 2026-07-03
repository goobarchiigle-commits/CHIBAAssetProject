# Complete Execution Roadmap — CAGR30% / Calmar1.5 達成までの完全実行計画

**日付**: 2026-07-04
**性格**: 実行手順書（Sonnet引き継ぎ前提）。戦略判断は既存6レポートで確定済み — 本書は「誰が実行しても同じ結果になる」ことを目的とする。
**拘束エビデンス（読込必須・全て `C:/ai-trading/reports/`）**:
1. `production_comprehensive_audit_2026-07-02.md` — 修正対象の発見元
2. `cro_decision_memo_2026-07-03.md` — 意思決定
3. `design_philosophy_review_2026-07-03.md` — 設計思想の問題Top10
4. `final_architecture_review_2026-07-03.md` — 閉鎖12領域・FUJIKO-R・Study74-79定義
5. `alternative_architectures_5x_2026-07-03.md` — ARCH-A〜E定義
6. `final_research_roadmap_2026-07-04.md` — **正典**（Part1判定・統治原則・Study74-86の成功/失敗/Kill条件）

---

# Section 0: 引き継ぎプロトコル（後継モデルはここから読む）

## 0.1 起動手順（毎セッション必須・会話メモリ不信）

```
1. C:/ai-trading/CLAUDE.md            ← ルール（PARAMS_LOCKED / ASK_FIRST / GUARD）
2. src/research_state.md              ← 先頭セクション=最新状態
3. src/configs/strategy.yaml          ← 現行パラメータ
4. backtests/backtest_summary.json
5. reports/final_research_roadmap_2026-07-04.md  ← 研究プログラム正典
6. 本書                                ← 実行手順
```

## 0.2 絶対禁止（起案前チェックリスト — 1つでも該当したら即却下）

恒久閉鎖14項（根拠は正典§最終1）:
1. Exitマイクロ最適化 2. BigWinner検出・保護 3. 日足OHLCV新規ML 4. 幾何・配分@¥3M 5. Adaptive CAP/レジームsizing/MSW 6. Add-on拡張 7. Entryフィルター/タイミング 8. Conditional RSR 9. Lot cost緩和 10. 監査の監査 11. max_positions拡大 12. Entry core改変 13. 検証独立前のハイブリッド化 14. 目標を仮説に賭ける行為

加えて: PARAMS_LOCKED（turtle_exit=55/min_hold=3/min_rsr=75.0/max_positions=3/capital=3M/slippage=0.001/commission=0.00055）の変更、`shift(1)`・翌日寄付執行・COST_ONE_WAY=0.155%の除去は**全結果を無効化する**ため禁止。

## 0.3 ASK_FIRST（ユーザー承認なしに実行禁止）

新規スクリプト作成 / PARAMS_LOCKED隣接変更（rsr_exit含む） / strategy.yaml動作変更 / signal_bridge.py発注ロジック / git push / 口座・APIプラン契約 / 実弾投入・レバ / CLAUDE.md変更。
**承認の単位はタスク毎**。「前回OKだったから」は無効。

## 0.4 採用ゲート（全Study共通・機械判定・裁量禁止）

```
ADOPT ⇔ WF 5/5 ∧ 2022非悪化 ∧ ΔCAGR≥+1pp ∧ Bootstrap P(>0)≥95% ∧ コストストレス生存
```
1条件でも欠けたらREJECT。REJECTの再挑戦（条件緩和・期間変更・パラメータ追加）は禁止。
**fresh run必須**: Production判定にキャッシュ値・過去JSON流用禁止（Study52汚染事件: 3研究連鎖誤判定の原因）。

## 0.5 Windows実行規約

- 実行ディレクトリ: `C:/ai-trading` 固定。パス直書き禁止（`from paths import RESULTS_DIR`）。
- Pythonスクリプト冒頭必須: `sys.stdout.reconfigure(encoding='utf-8')`
- matplotlib日本語: `rcParams['font.family']='MS Gothic'`
- 「動きそう」提出禁止。実行して「動いた」を数値つきで提出。

## 0.6 Study実行の標準11手順（全新規Studyに適用）

```
①  research_state.md 先頭で現在状態を確認
②  起案書をユーザーに提示（目的/仮説/成功条件/失敗条件/Kill/期待情報価値 — 正典に定義済みならそれを引用）→ ASK_FIRST承認
③  スクリプト作成: src/backtest/studyXX_<name>.py（既存study73_*.pyの構造を踏襲）
④  エンジンは composite_alpha_bt.py をライブラリとして使用（shift(1)/翌日寄付/コストを継承）
⑤  fresh run（キャッシュ無効化を明示）
⑥  結果JSON → backtests/studyXX_<name>_YYYY-MM-DD.json
⑦  レポート → reports/studyXX_<name>.md（判定は0.4ゲートで機械的に）
⑧  research_state.md 先頭にセクション追記（最終更新日も変更）
⑨  docs/research/YYYY-MM-DD.md 日次ログ
⑩  メモリ index 更新
⑪  git commit -m "research update: YYYY-MM-DD"（push はASK_FIRST）
```

## 0.7 現在地（2026-07-04時点の確定数値）

| 項目 | 値 |
|---|---|
| Production (D_ATR_EQ) | IS 12.37% / OOS2025 13.48% / Full 11.35% / Calmar IS 0.683 |
| 素の実力（バイアス補正後） | 10-12% |
| 制約固定オラクル上界 | 16-18% → **30%∧1.5は制約内で理論的に矛盾**（正典Part1） |
| 実弾 | ¥3M・auカブコム・live 35トレード |
| 未解消High risk | Survivorship+Selection Bias（±1-3pp） |
| ⚠ 環境 | `C:/ai-trading` が **git未初期化**（Stage1 M5で復旧） |

---

# Section 1: 到達方程式（30%∧Calmar1.5への唯一の経路）

制約固定では矛盾（正典Part1）。到達は以下の**条件連鎖**のみ。各ゲートの白黒で目標を公式改定する — 目標を仮説に賭けない。

```
[Stage1-2] 軽微修正+短期改善     → 実力12-13%・parity確保・RoR確定（土台）
[Stage3]   Study74白(資本¥20-30M) → Core 18-22% / Calmar1.2   ← CP1: 目標改定
[Stage4]   Study76/77白(構造純化)  → Core +0〜3pp・複雑性1/10
           Study80白(MNスプレッドα≥8%)                        ← CP2: 30%/1.5に初の実証根拠
[Stage5]   Study86白(ショート執行) + Study81/82/83白(Satellite群)
[Stage6]   Study85: Core(18-22%) ⊕ MN(α8-12%×レバ2-2.5x・β≈0) ⊕ Satellites
           結合CAGR≥30% ∧ 結合MaxDD≤20% ∧ RoR<1%              ← CP4: 達成判定
```

- **なぜMNが必須か**: Long Onlyではレバが必ずDDを比例拡大しCalmar1.5を破る。β≈0のMNスプレッドだけがレバとCalmarの衝突を解消する（5案中唯一 — alternative_architectures_5x §総括）。
- **推定成功確率 <30%**（final_architecture_review §6）。フォールバック目標が常に公式: CP1白=18-22%/1.2、CP1黒=15-20%/1.2。
- **プログラムKill**: 連続2四半期新規採用ゼロ → 研究を月次メンテに縮退・運用フェーズ移行。

---

# Section 2: Stage 1 — 軽微修正タスク（2026-07 / 数日 / 全て実弾非影響 or REG付き）

## M1: Addon執行価格の統一PATCH【最優先・Audit発見Medium】

- **問題**: addonが翌日**close**執行、新規BUYは翌日**open**。BT/Live乖離。コメント「新規BUYと同じ」は虚偽。
- **ASK_FIRST**: 要（エンジン変更）。CRO Memoで採用決定済み — 実行承認のみ取得。
- **手順**:
  1. `src/backtest/composite_alpha_bt.py` で `_addon_px = float(close_mat[next_i, _aidx])` を検索（2026-07-04時点 L1666。**行番号は信用せずgrepで特定**）。
  2. `close_mat` → `open_mat` に変更。直上コメントの「新規BUYと同じ」記述を「翌日寄付執行（新規BUYと統一 2026-07 PATCH）」に修正。
  3. REG: D_ATR_EQ を IS 2018-2024 / OOS 2025 / WF5fold でfresh run。
  4. 差分報告: 推定|Δ|≤0.3pp。**実測せず採用禁止**。|ΔCAGR|>0.5pp または 2022悪化なら停止しユーザー報告。
- **完了条件**: REG差分レポート提出 + research_state.md追記。

## M2: max_single_weight ×1.5 バイパスの解消【CIRCUIT形式矛盾】

- **問題**: `_max_val = capital * cfg.portfolio.max_single_weight * 1.5`（同ファイル L1685付近、grep `_max_val` で特定）。addon経路のみ37.5%まで許容 = CIRCUIT `max_single_weight=0.25`（変更禁止）と矛盾。
- **ASK_FIRST**: 要。推奨案 = `* 1.5` を削除し0.25厳格化（「例外を持つ上限は規律の不在」— design_philosophy #7）。
- **手順**: M1と同一REGサイクルで同時実施（fresh run 1回で両方測る）。addon発火件数の変化（現14件）を報告。
- **完了条件**: REG差分 + CIRCUIT整合の確認文言。

## M3: 研究プロトコルの恒久化（CLAUDE.md 1行）

- **内容**: `# OVERFIT_GUARD` に `fresh_run_required=true ← Production判定にキャッシュ値使用禁止` を追加。
- **ASK_FIRST**: 要（CLAUDE.md変更）。コード変更なし・リスクゼロ。

## M4: research_state.md のstale記述訂正【承認不要・文書のみ】

- **内容**: 既知問題表の「SELL=当日終値」記述は旧エンジンのもの。現エンジンはSELL/BUYとも翌日寄付（Audit Task2で確認済み）。該当箇所をgrep `当日終値` で特定し訂正、訂正日を明記。

## M5: gitリポジトリ復旧【データ保全・最優先級】

- **問題**: `C:/ai-trading` に `.git` が存在しない。REQUIRE_COMMIT実行不能・73 Study分の成果物がバージョン管理外。
- **手順**:
  1. ユーザーに確認: リポジトリが別所にあるか、意図的削除か。
  2. 復旧承認後: `git init` → `.gitignore` に `data/` が含まれることを**コミット前に必ず確認**（data_gitignore=true・絶対コミット禁止）。`src/.env` も除外確認（auth_leak=forbid）。
  3. `git remote add origin git@github.com:goobarchiigle-commits/CHIBAAssetProject.git`
  4. 初回コミット。**pushはASK_FIRST**。
- **完了条件**: `git status` 正常 + data//.env が追跡外であることの証跡。

## M6: 死蔵設定の棚卸し【文書のみ・動作変更なし】

- **対象**（final_architecture_review DISCARD該当）: `entry_timing.boost_weight=0.06`（未検証）/ `fraction.bull=0.0`（OOS選定バイアス）/ `vol_adj`残置コード / `turtle_exit=55`（IS単独選定・デッドパラメータ）。
- **Stage1では変更しない**（各変更はWF再検証が必要=軽微でない）。strategy.yaml各行に `# ⚠ DISCARD候補 (design_philosophy_review 2026-07-03)` 注記を追加するのみ。実変更はStage4のStudy76/77結果が方向を決めてから。

---

# Section 3: Stage 2 — 短期売買手法改善（2026-07〜08 / 数週間）

## S1: CAND_B移行 — rsr_exit 70→75【研究済・決断のみ・最大の短期改善】

- **効果（Study73実測）**: 2022年 -2.65%→+2.37%（+5.02pp）/ WF 4/5→5/5 / Bootstrap P(>0)=100% / Fold std -4.63pp。代償: 平均リターン-1〜2.7pp。将来レバの前提=worst-year正転。
- **ASK_FIRST**: 要（PARAMS_LOCKED隣接）。
- **手順**:
  1. ユーザー承認取得（トレードオフを明示: 平均-2pp vs 2022+5pp・WF5/5）。
  2. `src/configs/strategy.yaml` の `rsr_exit: 70.0` → `75.0`（fujiko: セクション。grep `rsr_exit` で確認）。
  3. 同期確認: `run_live_signal.py` / `src/kabusapi/signal_bridge.py` / `live_equivalent.py` がハードコードせずconfigから読むことをgrep `rsr_exit` で確認。ハードコード発見時は同値に修正。
  4. **MORNING_ROUTINE dry-runで既存保有ポジションのRSR75跨ぎExit発火有無を必ず確認**（Audit リスク/テスト観点）。跨ぎ発火がある場合はユーザーに事前報告してからLIVE。
  5. 3営業日 shadow確認 → LIVE。
- **完了条件**: dry-run証跡 + 初回LIVE実行ログ + research_state.md追記。

## S2: Quality Replacement Phase9 判定【待機のみ・追加コストゼロ】

- **状態**: shadow稼働中・2026-08中旬に30営業日評価が自動発火（Study59 P9E）。
- **手順**: 発火後、shadow判定ログを読み、Study58A採用基準（WF5/5・Calmar+0.075・bt_err=0.0）との整合を確認。`quality_replacement.enabled: true` への変更は**ASK_FIRST必須**（strategy.yamlに明記済み）。
- **注意**: Stage4のStudy77が「ランク脱落回転」を検証する場合、QRは思想的に吸収される — 採用判断時はStudy77の進行状況を必ず参照。

## S3: Study78 — RoR + Monte Carlo + 感度sweep【新規BT不要・即実行可】

- **正典定義**: 成功=現行RoR<1% ∧ レバ1.3x時RoR<5% ∧ 感度曲線滑らか。失敗=レバ1.3x時RoR≥5%（→レバ経路放棄）。
- **ASK_FIRST**: 要（新規スクリプト `src/backtest/study78_ror_mc_sensitivity.py`）。
- **実装仕様**:
  1. **入力**: 既存309トレードのR-multiple分布。取得方法: composite_alpha_bt.py をD_ATR_EQ設定でfresh runし、トレード台帳（entry/exit/損益）をJSONダンプ（エンジンは変更せず、呼び出し側で `trades` リストを収集）。
  2. **RoR定義（事前固定・変更禁止）**: 5年ホライゾン・トレード順ブートストラップ10,000本で、①P(MaxDD>30%) ②P(最終資本<初期50%)。レバは損益をL倍して同一計算（L∈{1.0, 1.1, 1.2, 1.3}）。
  3. **MC**: 同ブートストラップからCAGR/MaxDD分布のCI[5%,95%]。
  4. **感度sweep（bounded・この表以外の探索禁止）**: mom_period{16,19,21,23,26} / atr_extension.atr_mult{0.8,0.9,1.0,1.1,1.2} / eq_scale.size_frac{0.20,0.225,0.25,0.275,0.30} / rsr_exit{72,73.5,75,76.5,78}。各1パラメータのみ動かしIS CAGR/Calmar曲線をプロット。**崖（±10%変化でΔCAGR>3pp）があれば頑健性フラグ**。
  5. **出力**: `backtests/study78_ror_2026-XX-XX.json` + `reports/study78_ror_mc_sensitivity.md`。
- **期待情報価値**: Study79（レバ）・Study86（MN）・Study81（小型）全ての前提数字。

---

# Section 4: Stage 3 — 中期: 目標の生死とデータ基盤（2026 Q3）

## L1: Study74 — 資本スケーリング清浄再検証【プログラム最優先】

- **正典定義**: 成功=¥20-30MでCAGR≥22%（fix後・コスト込・WF5/5）。失敗=<18% or DD%が資本比例悪化。終了=4資本点×2構成の全測定（追加スイープ禁止）。
- **ASK_FIRST**: 要（新規スクリプト `src/backtest/study74_capital_scaling_fresh.py`）。
- **実装仕様**:
  1. M1/M2 PATCH適用後のエンジンで実行（**汚染前Study42/43A/46のJSONは参照値としてのみ使用・判定に使用禁止**）。
  2. マトリクス: capital ∈ {3M, 10M, 20M, 30M} × config ∈ {D_ATR_EQ現行, CAND_B適用後}。
  3. 各セル: IS 2018-2024 / OOS 2025 / WF5fold / MaxDD / lot_skip率 / 1銘柄あたり平均約定額。
  4. lot制約の解消検証: Study25/44が示した「¥3Mでlot丸めが破壊」が¥20M+で消えるか（skip率<5%を目安に報告）。
  5. 出力: `backtests/study74_capital_scaling_YYYY-MM-DD.json` + レポート。
- **分岐（CP1・目標公式改定）**:
  - **白** → 目標=CAGR18-22%/Calmar1.2（¥20-30M前提）。入金計画起案（capital_scaling層実装済・1.5%/日ランプ・ASK_FIRST）。Study78のレバ判定と合流。
  - **黒** → 目標=CAGR15-20%/Calmar1.2（現資本）。研究縮小・運用フェーズ移行。Stage4以降は「30%再挑戦」ではなく「Satellite分散によるCalmar改善」として継続可否をユーザー決裁。

## L2: Study75 — Survivorship-free ルールベースユニバース（J-Quants）

- **正典定義**: 成功=バイアス実測≤-1.5pp ∧ 規則ユニバースBaseline CAGR≥9%。失敗=バイアス>-3pp（→全歴史判定の再審査。これ自体が重要情報）。
- **ASK_FIRST**: 要（J-Quants APIプラン契約 — 料金プラン表を提示しユーザー選定。上場廃止込み株価が取得できるプランであること）。
- **実装仕様**:
  1. データ層: `data/jquants/` に日次株価（上場廃止込み）・銘柄情報・売買代金を格納。**data/はgitignore必須（コミット絶対禁止）**。取得スクリプト `src/data/jquants_fetch.py`（ASK_FIRST）。
  2. 規則ユニバース定義（final_architecture_review 5.1）: TOPIX500構成 ∩ 20日平均売買代金≥¥300M ∩ 株価×100株≤有効資本×0.30。毎月第1営業日に再適用。銘柄数は規則の出力（固定禁止）。
  3. バイアス実測: 同一戦略（D_ATR_EQ）を「現行yfinance現存銘柄」vs「J-Quants廃止込み」で並走。ΔCAGR=バイアス幅。
  4. 出力: バイアス幅の確定値 + 規則ユニバースWF成績。
- **採用**: 規則ユニバースがWF5/5∧2022非悪化でRSR42同等以上 → ユニバース定義を規則版へ移行（ASK_FIRST）。
- **重要**: 本データ基盤はStage4-5の全Study（76/77/80/81/82/84）の前提。**1投資6用途**。

---

# Section 5: Stage 4 — 構造改革とβ除去の実在確認（2026 Q4〜2027 H1）

## L3: Study76 — Clenow純正ベンチマークWF（複雑性の対価定量化）

- **正典定義**: 成功=純正構成がD_ATR_EQ比ΔCAGR≥-2pp以内。失敗=<-4pp（→FUJIKO-R骨格放棄・現行維持確定）。
- **仕様**: Study75規則ユニバース上で、{週次回転・`(slope90d×R²)`ランクのみ・TOPIX>MA200のみ・回転Exit} の最簡構成 vs D_ATR_EQ。Turtleトリガー有無の2アーム。パラメータ<10個・全て事前固定。
- **勝敗の使い方**: 勝てば現行の多層（レジーム5機構・Exit7系統・boost群）を削る根拠。負ければ複雑性は正当化されStage1 M6のDISCARD候補は現状維持。

## L4: Study77 — Exit構造置換WF（回転 vs 災害ストップ+トレイル）

- **正典定義**: 成功=A(回転+災害ストップ) or B(災害ストップ+ATRトレイルのみ) がC(現行多層)比 ΔCAGR≥+1.5pp ∧ WF5/5 ∧ tail_capture≥80%。失敗=全アームC劣位（→**Exit領域を恒久閉鎖**）。
- **境界注意**: これは閉鎖領域#1/#2（Exit micro・BW予測）では**ない**。「予測せず構造で守る」置換であり未検証（Study61-69が閉じたのは予測経路のみ）。ただし**アーム追加・閾値スイープは即閉鎖領域入り** — 3アーム固定厳守。
- **順序**: Study76決着後のみ（勝者構成に対して実施）。

## L5: Study80 — ARCH-A モメンタム・スプレッド実測【純データ分析・実装不要】

- **正典定義**: 成功=コストストレス後（貸株料+逆日歩一律-2%/年）スプレッドα≥8% ∧ WF5fold ∧ 2022非悪化 ∧ 2020/3クラッシュDD<25%。失敗=α<5% or ショート側寄与<2pp or 貸借ユニバース<100銘柄。
- **仕様**: J-Quants貸借銘柄でモメンタムdecile構築 → 上位decileロング/下位decileショートの日次スプレッド系列。**ショート執行は実装しない**（データ分析のみ・研究コスト低）。パラメータ=decile数10・週次リバランスのみ（スイープ禁止）。
- **分岐（CP2）**: 白 → **30%/1.5再挑戦に初の実証根拠** → Study86起案。黒 → MN経路閉鎖、30%/1.5は最終棄却し、以降はCP1目標+Satellite分散のみ。

---

# Section 6: Stage 5 — 拡張（2027）

## L6: Study81 — ARCH-E 小型グロース・モメンタム

- 成功=スリッページ0.5%・値幅制限モデル込みCAGR≥20% ∧ WF5/5。失敗=<15% or MaxDD>40% or 執行不能日>10%。
- **前提**: Study75廃止込みデータ必須（なければ起案禁止 — 小型はsurvivorship影響が大型の数倍）。
- 採用時: 「30% vs DD30-50%」トレードオフをユーザー明示決裁 → Satellite 10-20%配分から。Calmar1.5とは両立しない経路であることを提案時に必ず明記。

## L7: Study79 — 資本投入+レバ1.3x設計【3条件成立時のみ起案】

- 起案条件: Study74白 ∧ Study78合格（レバ1.3x RoR<5%）∧ CAND_B移行済。1つでも欠ければ**起案自体禁止**。
- 成功=paper 30営業日 tracking error<1pp ∧ DD予算内 → 段階レバ投入（10%刻み・各段ASK_FIRST）。
- レバ上限はCalmar制約から逆算（DD≤20%が先・レバが後）。

## L8: Study82 — PEAD（決算ドリフト）

- **第一関門=発表日時精度監査**（分単位・場中/引後判別）。監査FAILで即終了（リーク排除不能な研究は無効）。
- 成功=イベント当たりコスト後CAR≥+1.5%(40d) ∧ 年間≥30件 ∧ WF5fold。保有日数・閾値は事前固定少数グリッドのみ。
- データ: J-Quants決算発表日時+財務 / TDnet適時開示。

## L9: Study83 — 指数TSMOM（先物レイヤー）【データ独立・随時並行可】

- 成功=40年データSharpe≥0.8 ∧ 現行Core相関<0.5 ∧ ロールコスト補正後正。固定グリッド{20,60,120d}のみ。
- 採用時: 先物口座開設（ASK_FIRST）→ 証拠金¥30-60万レイヤー。単独30%不可の合算部品。

## L10: Study84 — クロスマーケット・リードラグ killテスト【1-2週間厳守】

- 生存条件事前固定: IC>0.05 ∧ コスト後期待値>0。未達→**即殺・再訪禁止・延命禁止**（期間延長・条件緩和は全て違反）。

---

# Section 7: Stage 6 — 統合と最終判定（2028〜2029）

## L11: Study86 — MNショート執行設計【Study80白の場合のみ】

- 信用口座開設（ASK_FIRST）・建玉/追証管理・逆日歩実測。既存reconciliation/fail-closedランタイム流用。
- paper 3ヶ月 → BT/live parity<1pp → 10%配分から段階投入。

## L12: Study85 — ポートフォリオ統合（最終形）

- 成功=結合RoR<1% ∧ 結合Calmar≥1.2 ∧ 各スリーブ限界寄与>0（限界寄与≤0のスリーブは除外）。
- 構成: Core（現行 or Study76/77勝者・80-90%）+ 当選Satellite（MN/PEAD/TSMOM/小型/Study10 RSR90 corr=0.097再評価・計10-20%）。
- **CP4 最終判定**: 結合CAGR≥30% ∧ MaxDD≤20% ∧ RoR<1% が実測で成立 → 30%/Calmar1.5達成を宣言。不成立 → その時点の実測値で最終目標を確定し運用フェーズ（研究は月次decay監視のみ）。

---

# Section 8: 全体スケジュールと決定木

## タイムライン（依存順・並行可能なものは並記）

| 時期 | タスク | ゲート |
|---|---|---|
| 2026-07 | Stage1 M1-M6（数日）→ S1 CAND_B移行 | REG |
| 2026-07〜08 | S3 Study78（BT不要）/ S2 QR Phase9（8月中旬自動） | RoR |
| 2026 Q3 | L1 Study74 + L2 Study75（並行） | **CP1: 目標改定** |
| 2026 Q4 | L3 Study76 → L4 Study77 / L5 Study80（並行） | **CP2: MN実在** |
| 2027 H1 | L6 Study81 / L7 Study79（条件付き）/ L9 Study83（並行可） | CP3 |
| 2027 H2〜2028 | L8 Study82 / L10 Study84 kill / L11 Study86（80白時） | — |
| 2028〜2029 | L12 Study85 統合 | **CP4: 30%/1.5最終判定** |

## 決定木（要点のみ — 詳細は各Study分岐）

```
Study74 ─黒→ 目標15-20%/1.2で運用フェーズ（Stage4以降はCalmar改善目的に縮小）
   └白→ 入金→Core 18-22%/1.2
Study80 ─黒→ 30%/1.5最終棄却。CP1目標+Satellite分散が最終形
   └白→ Study86→Study85で30%/1.5判定
Study76 ─黒(<-4pp)→ FUJIKO-R骨格放棄・現行Production維持
Study77 ─黒→ Exit領域恒久閉鎖
どの時点でも: 連続2四半期採用ゼロ → 運用フェーズ縮退
```

## 各停止点の残余価値（この順序の理由）

74黒で止まっても: RoR・クリーンデータ・parity修正が運用資産として残る。80黒で止まっても: 規則ユニバース+複雑性判定+Exit判定でCoreが強化されている。**どの停止点でも「使えない中間生成物」が残らない順序**になっている — 逆順は全てStudy46→50→52→70-73型の連鎖汚染（未検証の前提の上に研究を積む）を再演する。

---

# Section 9: 後継モデルへの申し送り（最重要5点）

1. **このプロジェクトの最大の資産は戦略ではなく検証機構**（shift(1)/翌日寄付/コスト/WF/Bootstrap・再現性error 0.0pp）。どんな改善よりも検証機構の保全が優先。
2. **数字はfresh runのみ信用**。過去JSONは「当時の判定記録」であり現在の判定材料ではない。Study52キャッシュ汚染が3研究を連鎖誤判定させた。
3. **REJECTは資産**。73 Studyの大半はREJECTだが、それが閉鎖14項という「掘ってはいけない場所の地図」を作った。REJECTされた領域を「新しいアイデア」として再提案しないこと — 表現を変えた再訪が最頻の失敗モード。
4. **ユーザー承認は目標変更の唯一の経路**。CP1/CP2/CP4の目標改定は必ずユーザー決裁。エージェントが目標を勝手に上げ下げしない。
5. **迷ったら0.4ゲートで機械判定**。「惜しい」「もう少しで」は全て禁止語。WF4/5は0/5と同じREJECTである。

---

*作成: CRO/Chief Architect, 2026-07-04。新規バックテスト実行なし。本書の全タスクは実行前にASK_FIRST該当有無を0.3で確認すること。正典（final_research_roadmap_2026-07-04.md）と矛盾する場合は正典が優先。*
