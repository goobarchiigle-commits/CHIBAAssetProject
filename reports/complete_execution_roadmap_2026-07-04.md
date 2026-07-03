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
| ⚠ 環境 | **訂正(M5実行後)**: `.git`は実際には`src/`配下に既存(origin一致・最終コミット2026-04-07・3ヶ月分未コミット)だった。root直下へ統合済み(commit 8641863)。詳細→§2.1 |

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
- **手順**: M1と同一REGサイクルで同時実施（fresh run 1回で両方測る）。addon発火件数の変化を報告。~~（現14件）~~ **→訂正(2026-07-04)**: 14件はStudy52のA_BASELINE構成の値。D_ATR_EQの実測はIS 2018-2024=**5件**/OOS 2025=**16件**（§2.2前提事実3）。
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

## 2.1 Stage1 実行ログ（2026-07-04・Sonnet実行）— 進捗・決裁待ち・追加検討事項

### 進捗サマリ

| タスク | 状態 | 備考 |
|---|---|---|
| M1 (Addon執行価格PATCH) | **REG FAIL → ロールバック** | 下記参照 |
| M2 (×1.5バイパス撤廃) | **REG FAIL → ロールバック** | 下記参照 |
| M3 (CLAUDE.md恒久化) | ✅完了 | `# OVERFIT_GUARD`に`fresh_run_required=true`追加 |
| M4 (stale記述訂正) | ✅完了 | research_state.md SELL/BUY非対称記述を訂正 |
| M5 (git復旧) | ✅完了（想定と異なる実態） | 下記参照 |
| M6 (DISCARD注記) | ✅完了 | strategy.yaml 4箇所に注記追加、動作変更なし |

### M1/M2: REG FAIL の詳細 — **ユーザー決裁待ち**

`composite_alpha_bt.py`にM1(`close_mat`→`open_mat`)とM2(`×1.5`撤廃)を適用しfresh runでREG測定した結果:

- **M1単独**: OOS ΔCAGR=**-2.06pp**（roadmap想定「|Δ|≤0.3pp」を6倍超過・ゲート閾値0.5pp超過）、2022 CAGR **-2.65%→-2.95%（悪化）**。
- **M1+M2**: addon発火が**完全にゼロ化**（IS/OOS/WF全foldでaddon_cnt=0。CAND_A=EQ Scale完全除去の数値と完全一致）。
  - **根本原因**: `max_positions=3`均等配分の通常エントリーは目標ウェイト約33.3%（capital/3）であり、これは`max_single_weight=0.25`（CIRCUIT値）を**エントリー時点で既に超過**している。旧`×1.5`（37.5%上限）はCIRCUIT違反ではなく、addon機能が動作するために必須の実務的ヘッドルームだった（strategy.yamlの`symbol_cap=0.40`が0.333超に意図的に設定されているのも同じ理由）。
  - M2をdesign_philosophy_reviewの提言通り単純撤廃すると、Study45/52でADOPT済みのEQ Scale addon機能（Production構成D_ATR_EQの中核要素）を実質的に無効化する。roadmap自身のゲート「|ΔCAGR|>0.5pp または 2022悪化 → 停止」に抵触したため、コードは**PATCH前に完全ロールバック済み**（`composite_alpha_bt.py`は現在CURRENT/Study73と同一）。
  - 検証物: `src/backtest/reg_m1m2_addon_patch_2026-07-04.py` / `backtests/reg_m1m2_addon_patch_2026-07-04.json`

**決裁待ちの選択肢**（いずれか、または他案をユーザーが指示）:
1. **M1/M2を完全放棄**し現状維持（addon close執行・×1.5ヘッドルームのまま）。CRO Memoの「PATCH採用決定」を覆すことになる。
2. **max_single_weightの適用範囲を再設計**（例: 新規エントリーのみに0.25を厳格適用し、addon増分は別枠のcapとして再定義）→ 新規WF検証が必要（Stage1の「軽微」の範囲を超える）。
3. **M1のみ採用・M2は放棄**（execution price統一は今後のBT/Live parityのために価値があるが、M1単独でも-2.06pp/2022悪化のゲート抵触があるため、これも「軽微修正」の前提が崩れており単独でもユーザー判断が必要）。
4. 上記いずれも保留し、Stage2以降（S1 CAND_B等）を先行させる。

**→ 推奨案は§2.2に確定済み（= 選択肢3の変形: M1採用+M2は文書整合M2'に置換）。§2.2は追加実測に基づく実行仕様まで含む — 決裁は§2.2を読んで行うこと。**

### M5: 想定外の発見（環境認識の誤り）

roadmap執筆時点の前提「`C:/ai-trading`がgit未初期化」は不正確だった。実際には`.git`が`src/`配下に単独で存在し、GitHub origin（CHIBAAssetProject.git, branch=main）と一致・HEAD=origin/mainの状態だったが、最終コミットは2026-04-07で以降3ヶ月分（src/内の再編作業＋reports/backtests/docs/scripts/tools/tests等ルート直下の新規ディレクトリ全て）が未コミットのまま放置されていた。

ユーザー承認により`src/.git`を`C:/ai-trading/.git`へ移動し単一リポジトリへ統合（コミット`8641863`）。data/・.envの除外はルート/src/両方の`.gitignore`で再確認し、履歴上の混入もなし。**push未実施**（別途ASK_FIRST）。

### その他、今後のroadmap実行にあたり検討が必要な項目

1. **`max_single_weight=0.25`（CIRCUIT・変更禁止）と`max_positions=3`均等配分（33.3%）の構造的不整合**。この不整合はaddonだけでなく、**Study74（資本スケーリング¥20-30M検証）や将来のsizing関連研究すべてに影響する**可能性がある。CIRCUIT値自体は変更禁止だが、「何に対して0.25を適用するか」の解釈が現状コード内で一貫していない（新規エントリーには事実上適用されず、addonにのみ×1.5付きで適用）。Stage3 Study74着手前に、この解釈をRCAとして明文化しておくことを推奨。
2. **`addon_cnt`フィールドのバグ**: `study73_production_migration_audit.py`の`extract_metrics()`は`raw.get("addon_cnt")`を参照しているが、`composite_alpha_bt.py`の実際の返り値キーは`"addon_count"`（本REGで発覚）。このため**Study70-73で報告されてきたaddon件数（addon_cnt列）は全て常に0であり、実際のaddon発火回数を反映していない**。roadmap本文中「addon発火件数の変化（現14件）」（M2節）の根拠が何であったか要再確認。過去の addon 関連判定（Study45/52/70等）のCAGR/Sharpe等の主要指標自体は別ロジックで計算されており影響ないが、addon件数に基づく解釈（発火頻度・実務インパクト評価）は再検証が必要。
3. **CAND_B (S1, rsr_exit 70→75) は本ロールバックの影響を受けない**独立変更のため、M1/M2の決裁と切り離してStage2へ進行可能。ただしdry-run時の検証は引き続き必須。
4. **git統合コミット(8641863)は大規模リネームを含む**（1425ファイル変更、うち126件はrename検出）。push前に、GitHub側リモートに既存コミット(639163f以降)がないか（他端末からの追加pushがないか）を`git fetch`で確認することを推奨。

---

## 2.2 M1/M2 決裁推奨案 + 追加検討4項目の実行仕様（2026-07-04追記・CLD起案）

**目的**: 決裁と実行の間に検討事項を残さない。本節の各仕様は承認後そのまま実行可能なレベルまで具体化済み。

### 前提事実（§2.1以降に追加実測で確定した5点）

1. **live執行の実装確認**: `src/run_live_signal.py` のaddon発注は `order_type="MARKET_OPEN"`（翌日寄付成行・L5670付近）。**BTのclose執行がliveと乖離している側** — M1は「改悪」ではなく「BTの過大評価の是正」。
2. **ロールバック完全性のfresh run検証**: 復元後エンジンでIS/OOS/WF/2018-2019全再実行 → Study73基準値と**全指標Δ=0.00pp**（IS 12.37/OOS 13.48/WF avg 18.37・4/5/2022 -2.65）。証跡: `backtests/reg_m1m2_addon_patch_2026-07-04_rollback_verify.json`。
3. **真のaddon件数**（`addon_count`キー修正後の実測）: IS 2018-2024=**5件** / OOS 2025=**16件** / WF fold別=2/1/1/5/6件。M2節の「現14件」はStudy52のA_BASELINE構成の値であり誤引用。
4. **OOS 2025のaddon依存**: 1年で16件はIS 7年分(5件)の3倍超。M1の執行価格差の影響がOOSに集中した理由（IS -0.15pp vs OOS -2.06pp）はこの分布と整合。**OOS 2025の好成績(13.48%)はaddon経路への依存度が高い** — 今後のOOS解釈全般で注意。
5. **エントリー経路のweight cap実装**: `composite_alpha_bt.py` L73 `MAX_POS_WEIGHT=0.40`（alpha加重cap）。CIRCUITの`max_single_weight=0.25`はBTエンジン内では**addon経路にのみ**登場（×1.5=0.375）。エントリーには0.25は一切適用されていない。

### M1 推奨: **採用**（Production公式数字の再基準化とセット）

- **判断根拠**: (i) live=MARKET_OPENが実装事実（前提1）。BTをliveに合わせるのが検証機構の真実性（§9原則1: 検証機構の保全>どんな改善）。(ii) -2.06ppは損失ではなくバイアス除去 — 数字が下がるからREJECTするのは「都合の良い測定の選択」であり本プロジェクトが最も禁じる行為。(iii) ゲート「|Δ|>0.5pp→停止」は停止・報告条項であり自動REJECTではない（M1手順4の原文どおり）。
- **承認に必要な明示的合意**: 採用するとProduction公式数字が下がる — IS 12.37→12.22% / OOS 13.48→**11.42%** / WF avg 18.37→17.99% / 2022 -2.65→-2.95%（2026-07-04 REG実測値）。この「正直な数字」への引き下げを承認すること。
- **実行仕様（承認後・検討ゼロで実行）**:
  1. PATCH再適用: `close_mat[next_i, _aidx]` → `open_mat[next_i, _aidx]`（grep `_addon_px` で特定・コメントも「翌日寄付執行（新規BUYと統一 2026-07 PATCH）」へ）。
  2. fresh run で新基準値確定: IS/OOS/WF5fold/2018/2019単独 + Bootstrap（Study73 Phase3方式・N=500・seed=42）。スクリプトは既存 `reg_m1m2_addon_patch_2026-07-04.py` を流用（M2部分は触らない）。
  3. §0.7現在地表・research_state.md冒頭の公式数字を更新。旧値は「close執行（過大評価）時代の参照値」として併記保持 — 削除しない。
  4. **S1への波及**: Study73 Phase2のCAND_B数字は旧エンジン産。S1決裁の前に、パッチ後エンジンで CURRENT vs CAND_B を再測定（fresh run・同スクリプト改変1行 `rsr_exit=75.0`）。CAND_Bの採用根拠（2022正転・WF5/5）がパッチ後も成立するかを確認してからS1承認へ。
  5. `src/live/live_equivalent.py` / parity計算がaddon執行価格を参照していないか grep `addon` で確認。参照があれば同値修正。
  6. research_state.md追記 + commit（`research update: YYYY-MM-DD`）。
- **却下する場合**: BT/Live乖離が恒久残存し、以後の全Study（74/76/77含む）がaddon分の過大評価を含んだまま進む。却下時はその旨を本書とresearch_state.mdに「既知バイアス（OOS約+2pp）」として登録すること（未登録での続行は禁止）。

### M2 推奨: **コード変更は恒久放棄 → M2'（文書整合）に置換**

- **判断根拠**: ×1.5撤廃=EQ Scale addon機能死は実測で確定（§2.1）。Study73のCAND_A=KEEP判定（EQ Scale除去は有害）と直接矛盾するため、コード側を変える選択肢は存在しない。design_philosophy #7「例外を持つ上限は規律の不在」は、本件では「上限の定義が実装と乖離している」ことが真因であり、例外の除去ではなく定義の明文化が正しい処方。
- **M2' 実行仕様（2案・ユーザー選択、デフォルト推奨=案a）**:
  - **案a（推奨・CLAUDE.md不変更）**: research_state.mdに実効上限のRCAを1段落記録するのみ:「CIRCUIT `max_single_weight=0.25` の実装実態 — エントリー経路: alpha加重cap 0.40（`MAX_POS_WEIGHT`）・均等時約33.3%。addon経路: 0.25×1.5=0.375 hard cap。0.25が単独で効く経路は現エンジンに存在しない。addon経路の0.375は機能要件（Study45/52 ADOPT済みEQ Scaleの動作条件）」。→ 本§2.2の記載を転記するだけで完了。
  - **案b（CLAUDE.md注記・ASK_FIRST）**: CIRCUIT行に注記追加 `max_single_weight=0.25 ← 実効: entry cap=0.40(MAX_POS_WEIGHT)/addon hard cap=0.375。詳細roadmap§2.2`。CIRCUIT値自体は不変更。
- **構造的解消の委譲先**: Study76（Clenow純正・最簡構成）にはaddonが存在しないため、Study76が勝てば不整合ごと消滅する。Study77も同様。**weight体系の再設計を独立研究として起案することは禁止**（恒久閉鎖#4「幾何・配分@¥3M」に該当）。

### 追加検討4項目の実行仕様（確定）

- **①（max_single_weight不整合）**: M2'案aの文書化で完了扱い。加えてStudy74実装仕様（§4 L1）に出力列1つ追加: 各セル（capital×config）で「実効max単一銘柄ウェイト実測値（日次ピーク）」を報告。¥20-30Mではlot丸め解消により均等33.3%へ張り付くはずで、その確認自体がStudy74の副産物になる。追加研究・スイープは不要。
- **②（addon_cntバグ）**: 修正はREGスクリプトに実装済み（正キー`addon_count`）。`study73_production_migration_audit.py`本体は**修正しない**（再実行予定なし・出力JSONは「当時の判定記録」として凍結 — §9原則2）。ただしresearch_state.mdのStudy73セクションの推論「addon_cnt: Study52 B_ATR_EXT=14 vs Study73 CAND_A=0 がその証拠」はこのバグにより**証拠として無効**（Study73側の0はバグ産。なおCAND_Aはaddon_policy=NONEなので真値も0であり、F3=KEEPの結論自体はCAGR比較12.37 vs 11.83に基づくため不変）→ 訂正注記をresearch_state.mdに追記（実施済み 2026-07-04）。
- **③（S1の実行順序）**: 確定順序 = **M1決裁 → （採用なら）パッチ後エンジンでCAND_B再測定 → S1承認 → strategy.yaml変更 → 同期確認grep → dry-run（RSR75跨ぎExit確認）→ 3営業日shadow → LIVE**。M1却下ならStudy73数字のままS1へ直行可。**M1決裁を先に済ませることが必須ではないが強く推奨** — 逆順（S1先行→後からM1採用）だと再基準化が2回発生し、shadow期間中に判定基準が変わる。
- **④（push手順）**: `git fetch origin` → `git rev-list --left-right --count origin/main...main` で分岐確認 → 左側（リモート先行）が0なら `git push origin main`。左側>0なら**停止・ユーザー報告**（他端末push痕跡＝履歴統合の再検討要）。実行自体はASK_FIRST維持。初回push成功後は通常運用（コミット毎push可否は都度確認）。

### 決裁チェックリスト（ユーザーはこの3問に答えるだけでStage1が閉じる）

| # | 決裁事項 | 推奨 |
|---|---|---|
| D1 | M1採用（公式数字 OOS 13.48→11.42%への再基準化を含む）— YES/NO | **YES** |
| D2 | M2'方式 — 案a（research_state.md記録のみ）/ 案b（CLAUDE.md注記） | **案a** |
| D3 | git push実行（④手順） — YES/保留 | YES（fetch確認PASS前提） |

D1=YES → M1実行仕様1-6を実施後、S1（§3）へ。D1=NO → 既知バイアス登録後、S1へ直行。いずれでもStage1完了・Stage2開始可能。

---

# Section 3: Stage 2 — 短期売買手法改善（2026-07〜08 / 数週間）

## S1: CAND_B移行 — rsr_exit 70→75【研究済・決断のみ・最大の短期改善】

- **効果（Study73実測）**: 2022年 -2.65%→+2.37%（+5.02pp）/ WF 4/5→5/5 / Bootstrap P(>0)=100% / Fold std -4.63pp。代償: 平均リターン-1〜2.7pp。将来レバの前提=worst-year正転。
- **⚠ 前提改訂(2026-07-04 §2.2③)**: 上記Study73数字は旧（close執行）エンジン産。M1決裁を先に確定させること（強く推奨）。M1採用時は手順1の前に「パッチ後エンジンで CURRENT vs CAND_B 再測定」を挿入し、2022正転・WF5/5がパッチ後も成立することを確認してから承認へ。M1却下時はStudy73数字のまま手順1へ直行可。
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
  1. ~~M1/M2 PATCH適用後のエンジンで実行~~ **→改訂(2026-07-04 §2.2)**: M1採否確定後のエンジンで実行（M2はコード変更放棄が確定 — §2.2参照）。M1未決裁のままの起案は禁止。（**汚染前Study42/43A/46のJSONは参照値としてのみ使用・判定に使用禁止**）。
  2. マトリクス: capital ∈ {3M, 10M, 20M, 30M} × config ∈ {D_ATR_EQ現行, CAND_B適用後}。
  3. 各セル: IS 2018-2024 / OOS 2025 / WF5fold / MaxDD / lot_skip率 / 1銘柄あたり平均約定額 / **実効max単一銘柄ウェイト実測値（日次ピーク・§2.2①）**。
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
