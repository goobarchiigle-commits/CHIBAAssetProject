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
| **Production (D_ATR_EQ) 【2026-07-04 M1適用後・現行公式値】** | **IS 12.22% / OOS2025 11.42% / Full 11.22% / Calmar IS 0.671 / WF avg 17.99%(4/5)** |
| ~~Production (D_ATR_EQ) 参考値~~ | ~~IS 12.37% / OOS2025 13.48% / Full 11.35% / Calmar IS 0.683~~ **← Addon close執行時代の参考値（M1採用前・2026-07-02 Study73時点）。削除せず保持** |
| 素の実力（バイアス補正後） | 10-12%（M1適用によりOOSの過大評価-2.06pp分は解消済み。上記が「正直な数字」） |
| 制約固定オラクル上界 | 16-18% → **30%∧1.5は制約内で理論的に矛盾**（正典Part1） |
| 実弾 | ¥3M・auカブコム・live 35トレード |
| 未解消High risk | Survivorship+Selection Bias（±1-3pp） |
| ⚠ 環境 | **訂正(M5実行後)**: `.git`は実際には`src/`配下に既存(origin一致・最終コミット2026-04-07・3ヶ月分未コミット)だった。root直下へ統合済み(commit 8641863)。詳細→§2.1 |
| M1採用状態(2026-07-04) | **採用済み**（addon執行=翌日寄付。BT/Live parity目的・成績改善目的ではない）。詳細→§2.3 |
| **CP1判定(2026-07-04)** | **🔴 BLACK**（¥20-30MでCAGR≥22%∧WF5/5未達）。目標改定はユーザー決裁待ち。詳細→L1節・`reports/study74_final_review.md` |

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

### §2.3 M1採用 実行結果（2026-07-04・ユーザー決裁: D1=採用）

**決裁理由（ユーザー明示）**: 「BTをLiveへ合わせるため」の採用。成績改善目的ではない。Production基準値が低下してもそれを正式な基準値として受容する。M1-RCA（DIVERGED_PORTFOLIO=0件・カスケードなし）により、副次効果が限定的であることも確認済み。

**実施内容**:
1. `composite_alpha_bt.py` の `_addon_px` を `close_mat[next_i, _aidx]` → `open_mat[next_i, _aidx]` へ恒久変更（コメントも更新）。M2（×1.5バイパス）は変更なし（維持）。
2. fresh runでIS/OOS/FULL(2018-2025継続run)/WF5fold/Bootstrapを再測定。
3. `src/backtest/live_equivalent.py` をgrep確認 → addon関連コード無し（parity修正不要）。

**新公式Production基準値（D_ATR_EQ・CURRENT・M1適用後）**:

| 指標 | 旧値（close執行・参考値） | **新値（open執行・現行公式）** | Δ |
|---|---|---|---|
| IS CAGR (2018-2024) | 12.37% | **12.22%** | -0.15pp |
| OOS CAGR (2025) | 13.48% | **11.42%** | -2.06pp |
| FULL CAGR (2018-2025継続run) | 11.35% | **11.22%** | -0.13pp |
| WF avg CAGR | 18.37% | **17.99%** | -0.38pp |
| WF pass | 4/5 | **4/5**（変化なし） | — |
| 2022 CAGR | -2.65% | **-2.95%** | -0.30pp |
| IS Calmar | 0.683 | **0.671** | -0.012 |
| Bootstrap median(N=500,IS年) | — | **11.65%** | CI=[2.01%, 23.63%] P(>0)=0.984 |

検証物: `backtests/study_m1_production_update_2026-07-04.json` / `src/backtest/study_m1_production_update_2026-07-04.py`

**旧値は削除せず「Addon close執行時代の参考値」として本書§0.7・research_state.mdに併記保持済み。**

### CAND_B (rsr_exit=75) M1適用後 再測定結果 — Study73旧結果は参考値扱い

| 指標 | CURRENT(M1後) | CAND_B(M1後) | Δ |
|---|---|---|---|
| IS CAGR | 12.22% | 11.24% | -0.98pp |
| OOS CAGR | 11.42% | 8.73% | -2.69pp |
| FULL CAGR | 11.22% | 10.16% | -1.06pp |
| WF avg CAGR | 17.99% | 14.92% | -3.07pp |
| WF pass | 4/5 | **5/5** | +1 |
| 2022 CAGR | -2.95% | **+1.51%** | **+4.46pp** |
| Bootstrap P(>0) | 0.984 | 1.0 | +0.016 |

**採用ゲート判定（WF5/5 ∧ 2022改善）: PASS**。M1適用後もCAND_Bの採用根拠（2022正転・WF5/5化）は健在。ただし平均リターンの代償（IS/OOS/FULL/WF avg全てで-1〜3pp）はM1適用前（Study73: IS-1.01pp/OOS-2.69pp/WFavg-1.99pp）よりやや拡大（特にWF avg -1.99pp→-3.07pp）。**S1（CAND_B採用）は別途ユーザー承認が必要 — 本節は再測定のみで自動採用ではない**。

### M2': research_state.md記録内容（案a・実施済み）

CIRCUIT `max_single_weight=0.25` の実装実態を以下の通り明文化（research_state.mdに転記済み）:
- エントリー経路: alpha加重cap **0.40**（`MAX_POS_WEIGHT`、`composite_alpha_bt.py` L73）。均等配分時は実質約33.3%。
- addon経路: **0.375**（0.25×1.5）hard cap。
- CIRCUIT値0.25が単独（×1.5なし）で効く経路は現エンジンに存在しない。addon経路の0.375はStudy45/52 ADOPT済みEQ Scale機能の動作要件であり、これを0.25へ厳格化するとEQ Scale addonが完全停止する（§2.1実測済み）。
- コード変更なし。CLAUDE.md変更なし（案a確定）。

### Study77 申し送り事項（Stage1では変更しない）

M1-RCAで判明した設計特性: `exit_policy="A"`（ATR Extension）のRSR Exit defer判定は `_pnow=(close-entry_price)/entry_price`（entry_price=addon込みの加重平均取得価格）に依存する（`composite_alpha_bt.py` L1082-1088）。このため**ポジションの加重平均取得価格が変化するイベント（addon等）は、Exit deferタイミングに副次的に影響しうる**という設計上の結合が存在する。この結合自体の是非（entry_price依存をやめてETRシンプルな絶対%等に置き換えるべきか）はStage1のスコープ外。**Study77（Exit構造置換WF）の検討事項として記録** — Study77がexit_policy="A"を代替案と比較する際、この結合の有無・影響を評価軸に加えることを推奨。Stage1では一切変更しない。

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

| # | 決裁事項 | 推奨 | **ユーザー最終決裁(2026-07-04)** |
|---|---|---|---|
| D1 | M1採用（公式数字 OOS 13.48→11.42%への再基準化を含む）— YES/NO | YES | **採用・実施完了**（§2.3参照） |
| D2 | M2'方式 — 案a（research_state.md記録のみ）/ 案b（CLAUDE.md注記） | 案a | **案a採用・実施完了** |
| D3 | git push実行（④手順） — YES/保留 | YES（fetch確認PASS前提） | **保留（M1反映後に判断・継続保留中）** |

**M1-RCA完了 → D1確定の経緯**: OOS ΔCAGR=-2.06ppを分解した結果、DIVERGED_PORTFOLIO(カスケード)=0件・TIMING_SHIFTは42件中1件のみ（3197.T・ATR Extension defer判定の1日ずれ、因果特定済み）。95%基準は形式上不達だったが、副次効果が局所的・説明可能であることを確認した上でユーザーがD1=採用を決裁（「BTをLiveへ合わせるため」であり成績改善目的ではない、との理由）。D3(push)は引き続き保留 — 次回git操作の判断時に④手順（fetch確認）を実施。

---

## M1-RCA: OOS ΔCAGR=-2.06pp の要因分解【D1決裁の前提・ユーザー追加指示 2026-07-04】

- **目的**: M1（addon執行価格 close→open）がOOSでのみ-2.06ppとなった理由を、以下6軸に分解して説明する。
  1. Addon価格差だけの寄与（同一トリガー・同一日・同一銘柄で価格のみ異なる場合の直接コスト差）
  2. 約定順位（エントリー/エグジットの実行順序が変化していないか）
  3. 約定率（addonトリガーが発生したが資金/上限制約で不成立になった比率の変化）
  4. 保有日数（ポジション保有期間の分布変化）
  5. Exit連鎖（addonによるblended entry_price変化がExitトリガー・タイミングを変えていないか）
  6. 銘柄別寄与（どの銘柄がΔCAGRを支配しているか）
- **判定基準（ユーザー指定）**:
  - **価格差だけの寄与が95%以上** → M1は「価格の置き換え」に閉じている → **安心してM1採用可**（D1=YESへ）。
  - **Exit構造（保有日数・Exitトリガー・銘柄構成）まで変化** → Stage1の「軽微修正」の前提が崩れている → M1はStage1から切り離し、WF全期間での再検証（Stage2以降相当）が必要。
- **ASK_FIRST**: 要（既存エンジンの一時パッチ適用によるA/Bトレード台帳比較・観測専用・Production変更なし）。
- **手順**:
  1. OOS 2025においてBASELINE（close執行・現行）とPATCHED（open執行）を同一条件でfresh run、両方の全SELLトレード（symbol/entry_idx/exit_idx/entry_px/exit_px/qty/pnl/reason）とaddon明細（date/symbol/stage/qty/px）を取得。
  2. トレードを`(symbol, entry_idx)`でマッチングし3分類: **IDENTICAL_TRADE**（同一symbol/entry_idx/exit_idx/reason、価格差のみ）/ **TIMING_SHIFT**（同一symbol/entry_idxだが exit_idx or reason が相違 = Exitトリガーが変化）/ **DIVERGED_PORTFOLIO**（片方にのみ存在するsymbol/entry_idx組 = 保有銘柄構成そのものが変化）。
  3. 各カテゴリのΔPnL合計をΔCAGR相当に換算し、寄与率(%)を算出。IDENTICAL_TRADEの寄与率が判定基準の「価格差だけ」に対応。
  4. 保有日数: 全体平均 baseline vs patched、およびTIMING_SHIFT該当トレードの保有日数差分を個別報告。
  5. 銘柄別寄与: symbol別ΔPnLを降順ソートし上位5-10銘柄を報告。
  6. 約定率: addon_countのbaseline/patched差分を報告（既存`_addon_detail`から算出。トリガー発生も含めた真の約定率にはengine計装追加が必要な場合、追加要否をユーザーに確認してから実施）。
- **出力**: `backtests/study_m1rca_oos_decomposition_2026-07-04.json` + 本節への結果転記。

### 実行結果（2026-07-04・OOS 2025・n_trades=42）

| カテゴリ | 件数 | ΔPnL | 寄与率 |
|---|---|---|---|
| IDENTICAL_TRADE（価格差のみ） | 41/42 | -39,268円 | **71.1%** |
| TIMING_SHIFT（Exitトリガー変化） | 1/42 | -15,968円 | 28.9% |
| DIVERGED_PORTFOLIO（保有銘柄構成が変化） | 0/42 | 0円 | 0.0% |
| **合計** | 42 | **-55,236円**（ΔCAGR -2.06pp相当） | 100% |

- **保有日数**: 全体平均は不変（baseline 8.3d = patched 8.3d）。TIMING_SHIFT該当の1件のみ-1日。
- **銘柄別寄与**: 上位は9531.T(-23,567円)・3197.T(-20,132円)・8002.T(-10,501円)。3197.Tの-20,132円の大半（-15,968円）がTIMING_SHIFT側。
- **約定率（addon件数）**: baseline=16件 / patched=16件、**完全一致** — addonの発火可否自体はM1で変化していない（発火判定はトリガー条件のみに依存し価格には依存しないため妥当）。
- **DIVERGED_PORTFOLIO=0件**が最重要所見: どの実行順位でも「別銘柄が代わりに選ばれる」「別日に乗り換わる」というカスケード（cash timing経由の連鎖）は一切発生していない。M1の影響範囲は**addonを受けた当該ポジション自身のExitタイミングに限定**される。

**TIMING_SHIFTの因果機序（RCA）**: exit_policy="A"（ATR Extension）は `_pnow = (close_today - entry_price) / entry_price`（entry_price=addon込みの加重平均取得単価）を用いてRSR Exit延期(defer)を判定する（`composite_alpha_bt.py` L1082-1088）。addon価格差でblended entry_priceが変わる→同じ終値でも`_pnow`（含み益率）が変わる→delay判定の閾値越えタイミングが変わる、という単一の明確な経路でExitが1日ずれた（3197.T: entry_idx=230固定・exit_idx 234→233、価格3444.0→3410.0で退出）。**この経路以外にaddon価格がExitへ波及する回路は存在しない**（DIVERGED_PORTFOLIO=0件がその証拠）。

### 判定（ユーザー基準: 価格差だけの寄与≥95% → 安心してM1採用可）

**71.1% — 95%未達 → 形式上は`STRUCTURAL_CHANGE_DETECTED`。ただし内容は限定的**:
- 42トレード中「Exit構造が変化」したのは**1件のみ**（2.4%）。しかも変化の中身は「別のExit方式に切り替わった」のではなく、**同一Exit方式(ATR Extension)内でdefer可否の1日ずれ**という狭い現象。
- カスケード（他ポジション・他銘柄への波及）は0件で構造的に存在しない。
- よって「Exit構造まで変化」という悪いケースではなく、「価格差が、まれに（年1件）ATR Extension deferの1日判定を動かす」という**説明可能・再現性のある副次効果**という評価が妥当。
- **推奨**: この副次効果を「既知の挙動」としてresearch_state.mdに明記した上でD1=YESとすることは可能（§2.2の推奨変わらず）。ただし95%の基準を厳密に適用するなら形式的にはD1保留継続が妥当 — **最終判断はユーザーに委ねる**。

---

# Section 3: Stage 2 — 短期売買手法改善（2026-07〜08 / 数週間）

## S1: CAND_B移行 — rsr_exit 70→75【研究済・決断のみ・最大の短期改善】

- ~~**効果（Study73実測）**: 2022年 -2.65%→+2.37%（+5.02pp）/ WF 4/5→5/5 / Bootstrap P(>0)=100% / Fold std -4.63pp。代償: 平均リターン-1〜2.7pp。将来レバの前提=worst-year正転。~~ **← M1適用前(close執行)の参考値。削除せず保持。**
- **✅ M1適用後 再測定済み(2026-07-04・§2.3)**: 2022年 -2.95%→**+1.51%（+4.46pp）** / WF 4/5→**5/5** / Bootstrap P(>0)=0.984→1.0 / IS -0.98pp・OOS -2.69pp・WF avg -3.07pp（代償やや拡大）。**採用ゲート(WF5/5∧2022改善)=PASS**。検証物: `backtests/study_m1_production_update_2026-07-04.json`。
- **ASK_FIRST**: 要（PARAMS_LOCKED隣接）。**手順1のユーザー承認はまだ取得していない — 上記M1適用後の新しいトレードオフ数字で改めて提示・承認を得ること**。
- **手順**:
  1. ユーザー承認取得（トレードオフを明示: **M1適用後の数字** — 平均IS-0.98pp/OOS-2.69pp/WFavg-3.07pp vs 2022+4.46pp・WF5/5）。
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

### ✅ 実行完了（2026-07-04・ユーザー拡張指示によりPart4-8追加・Production研究基盤として構築）

**成功基準達成**: 現行RoR(L=1.0)=P(MaxDD>30%)=0.13%<1% ✓ / レバ1.3x RoR=1.02%<5% ✓。感度は4パラメータ中3つ頑健（atr_mult/eq_scale/rsr_exit）、**mom_period=21のみ崖検出+ピーク形状で非頑健（過学習疑い・Study76/77へ申し送り）**。

Part1-8全て実行完了。詳細レポート`reports/study78_ror_mc_sensitivity.md`・研究資産6JSON（`backtests/study78_*.json`）はStudy74/79/81/85/86が追加BTなしで再利用可能。research_state.md先頭に全結果転記済み。

**Study79への含意**: 起案3条件（Study74白∧Study78合格∧CAND_B移行済）のうちStudy78は合格側材料が揃った。残り2条件（Study74・S1/CAND_B）待ち。

---

# Section 4: Stage 3 — 中期: 目標の生死とデータ基盤（2026 Q3）

## L1: Study74 — 資本スケーリング清浄再検証【プログラム最優先】

- **正典定義**: 成功=¥20-30MでCAGR≥22%（fix後・コスト込・WF5/5）。失敗=<18% or DD%が資本比例悪化。終了=4資本点×2構成の全測定（追加スイープ禁止）。
- **ASK_FIRST**: 要（新規スクリプト `src/backtest/study74_capital_scaling_fresh.py`）。
- **実装仕様**:
  1. ~~M1/M2 PATCH適用後のエンジンで実行~~ **→確定(2026-07-04 §2.3)**: M1は採用・実施完了（addon執行=翌日寄付。M2はコード変更放棄・案a確定）。現行エンジンでそのまま実行可（追加の待機条件なし）。（**汚染前Study42/43A/46のJSONは参照値としてのみ使用・判定に使用禁止**）。
  2. マトリクス: capital ∈ {3M, 10M, 20M, 30M} × config ∈ {D_ATR_EQ現行, CAND_B適用後}。
  3. 各セル: IS 2018-2024 / OOS 2025 / WF5fold / MaxDD / lot_skip率 / 1銘柄あたり平均約定額 / **実効max単一銘柄ウェイト実測値（日次ピーク・§2.2①）**。
  4. lot制約の解消検証: Study25/44が示した「¥3Mでlot丸めが破壊」が¥20M+で消えるか（skip率<5%を目安に報告）。
  5. 出力: `backtests/study74_capital_scaling_YYYY-MM-DD.json` + レポート。

### 追加仕様（2026-07-04・ユーザー拡張指示 — 「20Mで伸びた」ではなく「なぜ伸びたか」を数字で説明する）

**Part A: 資本制約の分解（waterfall寄与分析）** — CURRENT(D_ATR_EQ)構成・IS 2018-2024のみ・各資本水準で実施。

- **手法**: 既存エンジンに元々備わる研究用レバー（新規改修不要・全て既存kwarg/cfg）を使い、制約を1つずつ解除した反実仮想runとbaselineの差分でCAGR寄与を測定。
  - lot丸め解除: `lot_size=1`（コード内コメントに「単元未満/フラクション研究用」と明記済みの既存パラメータ）
  - max_positions解除: `max_positions_override=10`（実質上限撤廃、候補プール42銘柄に対し十分な値）
  - symbol_cap解除: `cfg.risk_controls`を`dataclasses.replace`で`symbol_cap=1.0`に複製差し替え（frozen dataclass・破壊的変更なし）
- **測定**: 資本水準ごとに [baseline, lot解除, max_pos解除, symbol_cap解除, 3つ全解除] の5パターンをrun。各単独解除のΔCAGRを「その制約が押し下げているpp」として報告。3つ全解除との差（相互作用）も報告。
- **現金余力（cash slack）**: これは独立して解除できる「制約」ではなく他制約の結果指標のため、waterfallには含めず、既存のQ1-Q3 Idle Cash Attribution計装（Study45: `q1_idle_when_winner_pct`等）をそのまま流用し「勝ち候補があるのに現金が遊んでいる日数」として別掲する。
- **注意（恒久閉鎖14項#4との切り分け）**: 本分析は¥3M固定でのProduction変更提案ではなく、**Study74（資本スケーリング）の因果構造を数値で説明するための診断専用ツール**。反実仮想run結果はStudy74のCP1判定材料としてのみ使用し、¥3Mでの制約緩和をProduction変更として提案しない（提案した場合は即閉鎖領域入り）。

**Part B: Capacity分析** — 資本水準ごとに以下5指標を可視化（既存計装の再利用のみ・新規計装不要）。

| 指標 | 定義・既存ソース |
|---|---|
| スキップ率 | `rejected_by_lot_count` / (候補総数) — Study42既存計装 |
| 平均投資率 | `avg_exposure`（既存メトリクス） |
| 現金滞留率 | `avg_idle_cash_ratio_pct`（Study41既存計装） |
| lot不足率 | `rejected_by_lot_count` / (`rejected_by_lot_count` + `n_trades`) |
| Position充足率 | `avg_simultaneous_holdings` / `max_positions`（1-見逃し率、Study41既存計装） |

**期待成果**: 「3M→30MでCAGRが伸びる」という結果を、lot丸め・max_positions・symbol_cap・現金滞留の**内訳付き**で説明できるようにする。入金判断（¥10M/¥20M/¥30Mのどの水準で何がボトルネック解消の主因か）に直接使える形にする。

- **分岐（CP1・目標公式改定）**:
  - **白** → 目標=CAGR18-22%/Calmar1.2（¥20-30M前提）。入金計画起案（capital_scaling層実装済・1.5%/日ランプ・ASK_FIRST）。Study78のレバ判定と合流。
  - **黒** → 目標=CAGR15-20%/Calmar1.2（現資本）。研究縮小・運用フェーズ移行。Stage4以降は「30%再挑戦」ではなく「Satellite分散によるCalmar改善」として継続可否をユーザー決裁。

### ✅ 実行完了（2026-07-04）— **CP1判定 = 黒（失敗）・ユーザー決裁待ち**

**実測**: 4資本点(¥3M/10M/20M/30M)×2構成(CURRENT/CAND_B)の全8セルでIS/OOS/WF5fold/annual標準実施。IS CAGRは¥3M(12.22%)→¥20M(13.11%)で+0.89ppのみ、¥30Mで後退(12.84%)。WF5/5はどの資本水準でも未達成（唯一例外=CAND_B¥3Mの5/5だがIS CAGR最低）。正典基準「CAGR≥22%∧WF5/5」に対し明確に未達 → **黒判定**。

**ユーザー拡張指示による追加分析（Part A制約分解・Part B Capacity分析）**: 「20Mで伸びた」の主因はほぼlot丸め解消（+1.12pp止まり、¥20M以降完全解消）で説明可能。max_positions=3は資本を上げても一切緩和されず（むしろ¥20M以降は解除するとCAGR悪化）、資本規模によらない構造的天井として残存。symbol_cap(0.40)はどの資本水準でも非拘束。詳細→`reports/study74_capital_scaling.md`・`src/research_state.md`Study74節。

**⚠ 副次発見**: CAND_BのWF5/5達成は¥3M固有現象（資本を上げるとWF pass低下）。S1決裁はこの点を踏まえて再評価が必要。

**次アクション（ユーザー決裁待ち）**: CP1目標改定（黒→CAGR15-20%/Calmar1.2への変更）の可否。統治原則4によりユーザー明示決裁が必須 — 本書は判定材料の提示までとし、目標変更は宣言しない。

### 🔴 CP1判定: **BLACK**（確定・ユーザー決裁待ち）

正典ゲート「¥20-30MでCAGR≥22% ∧ WF5/5」に対し全8セル未達。**BLACK判定**（roadmap統治原則の分岐表記に合わせ英語表記で明示）。目標改定（→CAGR15-20%/Calmar1.2）はユーザー決裁待ちのまま据え置き、本書では宣言しない。

### 統合レビュー（Part1-3・2026-07-04・ユーザー拡張指示・新規BTは合計1回のみ）

Study74を失敗報告で終わらせず、以下3点を追加実施（詳細は各レポート参照）:

- **Part1（①制約分類・②Capacity Curve・③CP1=BLACKの論理整理）**: `reports/study74_final_review.md` + `backtests/study74_capacity_curve_2026-07-04.png` + `backtests/study74_integrated_review_2026-07-04.json`。**新規BTゼロ**（既存study74 JSONの再集計のみ）。
  - 制約6種を「🟢改善可能(lot丸め=資本で解消済み)」「🔴構造限界(max_positions/candidate不足/entry頻度=資本非依存)」「⚪非該当(symbol_cap=非拘束)」「🟡従属指標(cash滞留)」に分類。
  - 「資本拡大の理論上限=lot丸め解消分(+1.12pp)のみ」であり「目標(18-22%)とのギャップ(5-9pp)は資本経路だけでは埋まらない」ことを論理証明（QED形式でreport内に明記）。
- **Part2（Study78 Research Assets拡張・Worst10 Drawdown Episode）**: `reports/study78_ror_mc_sensitivity.md`のPart4に追記 + `backtests/study78_worst10_dd_episodes_2026-07-04.json`。単一最大DDだけでなく全DDエピソード(閾値-3%)を検出しワースト10件を寄与トレード全量付きで格納。**今後Study81-86は「この変更でWorstDDが改善したか」をこのJSONとの比較のみで判定可能**（新規BT不要）。
- **Part3（Study74B: 候補不足構造分析）**: `reports/study74b_candidate_shortage_design.md` + `backtests/study74b_candidate_shortage_2026-07-04.json`。**⚠命名注意: ロードマップ既存のL2 Study75(J-Quants)とは別物のため「Study74B」と呼称**（Study75の定義・優先順位は変更なし）。
  - 見送り理由の74%がCAP_MISS(スロット競合)。うち56%は「その日の最上位候補」の喪失（質の劣化ではなく機会そのものの喪失）。
  - 見送り候補と採用候補のRSR中央値は完全一致(81.0)— 「質が低いから弾かれた」わけではない。しかしmax_positions緩和はCAGR悪化（Part1）という**未解決の矛盾**をStudy81/85への申し送り事項として記録。
  - 候補不足の9割は通常相場（risk_off以外）で発生。idle-cash日の75%は真に候補が存在しない日。

**唯一の追加BT**: `src/backtest/study74b_diagnostics_2026-07-04.py`（FULL 2018-2025・CURRENT・M1適用後を1回のみ再実行し、既存計装`_skip_detail`/`_missed_cands`/equity・drawdown曲線を初めて永続化。Part2/Part3双方で共用）。Part1は新規BTゼロ。

## L1B: Study74B — 候補不足構造分析【完了・2026-07-04・番号衝突回避のため暫定命名】

上記Part3参照。ロードマップの正式Study番号ではなく、Study74の派生分析として実施・完了。将来的にこの内容を正式なStudy番号に組み込むかはユーザー判断とする（現時点では未割当のまま`reports/study74b_candidate_shortage_design.md`に格納）。

### ✅ Study74B-RCA 追加実施（2026-07-04・新規BTゼロ・ユーザー指示）— **CAP_MISS矛盾は部分解明・因果証明は未解決**

**目的**: 「見送り候補RSR中央値=採用候補RSR中央値=81.0（品質差なし）なのに、max_positions緩和はCAGR悪化する」という矛盾のRCA。新規BT厳禁のため既存Research Assets（trade_dataset/study74b/study74_capital_scaling）のみ使用。

- **解析1**: 採用トレード(309件)の完全プロファイルを価格データから直接算出（run_scenario不使用）— 勝率54.4%/PF=2.135/Expectancy+14,536円/MFE中央値+3.22%/MAE中央値-2.56%。RSR等価性を再確認。
- **解析2/3**: **未解決（構造的ブロック）**。見送り候補449件の個別(date,symbol)が未永続化のため日次ペアリング・Opportunity Costは実行不可能。年次集計のみ実施（2023年は見送り最多かつ採用側も最高収益 — 市場全体のモメンタムが両群に共通して効いている傍証）。
- **解析4**: **ポートフォリオ状態依存仮説**（候補品質ではなく到着タイミングが差を生む）を提示。rank0(最上位)候補が見送りの55.9%。ただし因果の完全証明はできず「仮説」止まり。
- **結論**: **未解決**。説明できた部分（品質差なし）と説明できなかった部分（それでもmax_positions緩和が悪化する因果メカニズム）を明確に分離。
- **Study81への申し送り**: `_missed_cands`個別レコードの永続化（新規BT1回）+ フォワードリターン追跡 + 同時保有相関の実測が必要。

詳細→`reports/study74b_rca.md` / `backtests/cap_miss_pairs.json` / `backtests/opportunity_cost.json` / `backtests/hidden_factor_analysis.json`

## L1C: Study80A — Observation Infrastructure & CAP_MISS Root Cause Foundation【完了・2026-07-04・新規BT1回のみ・Parity PASS】

**目的**: Study74B-RCAが未解決に終わった原因（見送り候補の個別レコード未永続化）を恒久的に解消する観測基盤の構築。改善研究ではない。

- **Parity**: PASS（CAGR=11.22%/Trades=309/Sharpe=0.564/MaxDD=-18.22%/Calmar=0.616、全て変更前と完全一致）。詳細→`reports/parity_report.md`。
- **エンジン変更**（観測専用）: 候補ログ4種に日次コンテキスト(cash_before_entry/used_slots/max_slots/selected_symbols/selected_scores/position_weights/momentum_63d_pct/sector/market_regime/skip_reason)を追加、新規`_selected_cands`リスト追加。制御フロー変更ゼロ。
- **成果物**（Study81が追加BTなしで再利用可能）: `trade_dataset_v2.json` / `missed_candidates_full.json`(見送り607件・個別記録) / `forward_return_dataset.json` / `opportunity_cost_dataset.json` / `correlation_dataset.json` / `study81_analysis_template.py`(統計解析7関数実装済み)。詳細スキーマ→`reports/observation_schema.md`。
- **⚠ 副産物（Study74B-RCA未解決事項への統計的裏付け）**: 同日に実際に競合していた3候補群の分散縮小率=**24.8%**（日をまたぐ無作為抽出の67.3%＝理論的独立水準と比べ大幅に低い）— 「見かけの分散が実質的な相関の高い集中になっている」という仮説を初めて定量的に裏付け。セクター集中度も偶然を有意に上回る(p=0.0)。rank0見送り率63.6%は継続。詳細→`reports/study80a_observation_infrastructure.md`。
- **申し送り**: 上記知見はStudy81での正式検証・報告対象（本Studyは基盤構築が主目的のため速報扱い）。

## L1D: Study81 — Cluster Diversification Hypothesis【完了・2026-07-04・追加BTゼロ】

**目的**: 「max_positions=3が最適なのではなく、4銘柄目は既存3銘柄と同じクラスターに属するため期待値が増えない」仮説の検証。改善案は提示せず説明のみ（指示通り）。

- **Cluster ID設計**: Production既存の`src/strategy/cluster.py`(CLUSTER_MAP_DEFAULT)を再利用したmacro_cluster × momentum/ATR/RSRのtercileによるfactor_clusterの組み合わせ。alpha_scoreはdegenerateのため除外。
- **核心検定（解析4）**: CAP_MISS候補を同cluster/別clusterで二分 → **同cluster群(n=366)forward_20=+3.46% > 別cluster群(n=79)=+1.71%**（仮説と逆方向、p=0.1443非有意）。
- **Hidden Factor（解析6）**: Cluster理論からの逸脱42件中35件(83%)が「同clusterなのに好成績」で解析4と同方向に一貫。
- **重要な留保**: macro_cluster(4分類)レベルの集中度はランダムと非有意(p=0.0661)。Study80Aのraw sector(13-14分類)では有意(p=0.0)だったこととの対比で、**クラスター粒度定義次第で結論が変わる**ことが判明。
- **結論**: **棄却（REJECT）**。ただしStudy80Aの「同日競合3候補群の分散縮小率24.8%」（リターンの大きさではなくリスク相関構造の知見）は本Studyのスコープ外であり否定も肯定もしていない（未解決のまま残置）。
- 詳細→`reports/study81.md`

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

### ✅ 実行計画準備完了（2026-07-04・新規BT/コード変更ゼロ・Research Assetsのみ）

Study75がJ-Quants契約待ちで着手不可のため、Study76を「Study75完了後に即実行できる状態」まで計画のみ先行整備。成果物: `reports/study76_execution_plan.md`（目的明文化・Production差分5項目・固定10条件・成功/失敗基準・Research Assets棚卸し）/ `reports/study76_dependency_matrix.md`（Study75/77依存関係・変数分離確認・ブロッキング状態）/ `reports/study76_checklist.md`（Phase0-5実行チェックリスト）。

**未確定のまま残した仕様確認事項（Study75完了後に最優先でASK_FIRST）**: 正典「Study75規則ユニバース上で純正構成 vs D_ATR_EQ」の一文が、比較対象のD_ATR_EQ自体もStudy75ユニバース上で再測定する前提を含むか未確定。現行D_ATR_EQ公式値（RSR42ベース）をそのまま比較対象に流用すると、ユニバース差とアーキテクチャ差が交絡し「複雑性の対価」の測定が汚染されるリスクがある。Study75完了直後の最初のアクションとして確認必須（詳細→`study76_execution_plan.md`§5）。

**Study77への影響整理**: Study77は「Study76勝者構成に対して」実施が正典定義のため、Study76失敗（現行維持確定）で終わった場合Study77は起案自体が成立しない。この場合の扱い（対象差し替え/研究終了）はStudy76結果確定後にユーザー決裁が必要（詳細→`study76_dependency_matrix.md`§3）。

### 🔒 Universe統制ポリシー確定（2026-07-04・ユーザー決裁・Study76/77/85全てに恒久適用）

Study75完了時点で、Survivorship-free Universeを新基準Universeと定義。以降のStudy76・Study77（将来のStudy85統合評価含む）で使用する**全比較対象（D_ATR_EQを含む）は必ずStudy75 Universe上でfresh run再測定した値のみ使用**。旧Universe（RSR42）値との比較は禁止 — Universe差とArchitecture差の交絡を排除するための恒久統制。Study75終了後、本ポリシーの適用をASK_FIRSTで確認してからStudy76へ進む（再検討ではなく実行着手ゲート）。`study76_execution_plan.md`/`study76_dependency_matrix.md`/`study76_checklist.md`は本決裁に合わせ訂正済み。旧RSR42ベースのD_ATR_EQ公式値は「旧Universe参考値」として凍結保持するのみで判定には不使用。

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

## 実行ログ追記（2026-07-09・Sonnet実行）— L2 Study75: J-Quants Data Lake Bootstrap 実装フェーズ完了

**背景**: J-Quants Standardプラン契約完了（ユーザー確認済み）。前回セッション（2026-07-04）は契約前の
"pre-implementation"（`src/jquants/` 骨格のみ・小規模ユニバース限定設計）だった。今回はユーザーから
「全上場銘柄（上場廃止含む）・2016年〜現在・年パーティションParquet」という本格スコープの指示を受け、
取得オーケストレーション層と保存レイアウトを再設計・実装した（`reports/study76_execution_plan.md` 等の
既存正典・Study76側コードは無改修）。

### 発見事項（認証情報の配線ミス）
- ルート直下 `C:\ai-trading\.env` の `JQUANTS_API_KEY` はどのコードからも読まれていなかった
  （アプリが実際に読むのは `src/.env` のみ・`src/paths.py` の `PROJECT_ROOT` 起点）。
- `src/.env` 側にも同キーはあったがコメントアウトされ無効だった。
- ユーザー確認によりこの値は J-Quantsダッシュボード発行の **refreshToken**（mailaddress/password不要）と判明。
  → `JQUANTS_REFRESH_TOKEN` を正式環境変数名として新設し、`auth.py`/`config.py` に直接注入経路を追加。

### 実装完了（コード・テストのみ。ネットワーク通信は未実施）
- `src/jquants/universe.py`（新規）: Universeのイベントソーシング復元（ADD/REMOVE・営業日粒度・
  `JPXCalendar`再利用・チェックポイント/再開対応）。`metadata/universe_events.parquet` が正本。
- `src/jquants/cache.py`（再設計）: 銘柄別ステージング方式（`cache/staging/{symbol}.parquet`）。
- `src/jquants/normalize.py` + `compaction.py`（新規）: raw/processedの責務分離
  （raw=列名リネームなし、processed=固定正規化スキーマ）。年パーティション `daily_bars_{year}.parquet` を
  ステージングから冪等に再構築。
- `src/jquants/study75_adapter.py`（新規）: Study76互換の `processed/{symbol}.parquet` 生成層。
- `src/jquants/catalog.py` + `manifest.py`（新規）: `metadata/catalog.json`（現在状態集約）・
  `metadata/manifest.json`（実行ごとの再現性レコード: git_commit・dataset_hash等）。
- `src/jquants/provider.py`: `topix_raw()`/`topix_df()` 追加（`/indices/topix`、実エンドポイントは未検証）。
- `src/scripts/jquants_sync.py`: `--full-market` / `--rebuild-universe` / `--materialize` / `--compact-only` 追加。
- `docs/implementation/jquants_execution_infrastructure.md`: 新アーキテクチャに合わせ全面改訂。

### テスト結果
`pytest tests/jquants/ -v` → **38/38 pass**（認証情報・ネットワークなしで全件確認）。
広域回帰確認: `pytest tests/` 全体で本タスク由来の regression なし（既存の13件のtest失敗は
`git stash` で本タスクの変更を除いた状態でも再現する無関係な事前存在の問題と確認済み・対応不要）。

### 未実施（ASK_FIRSTゲート待ち・次アクション）
①`src/.env` への `JQUANTS_REFRESH_TOKEN` 反映（ユーザーがエディタで直接編集） →
②初回API疎通smoke test（TOPIXエンドポイント名・コード桁数の実測確認） →
③Universeイベントログのフル復元実行（約2,600営業日分） →
④Full Download本番実行（推定4,000〜5,000銘柄）。いずれも本書§0.3/CLAUDE.md ASK_FIRSTに従い
個別に承認を得てから実行する。

---

## 実行ログ追記（2026-07-10・Sonnet実行）— L2 Study75: Full Download完了 + Universe復元完了 + Universe Design記述分析完了

**J-Quants API v2実態判明**: 契約後の実APIは事前ドキュメント（v1想定）と異なり、
`x-api-key`静的キー認証・略記フィールド名（O/H/L/C等）・エンドポイント`/v2/equities/bars/daily`
`/v2/equities/master`。ASK_FIRST②スモークテストで実測確認し互換レイヤー（`src/jquants/schema.py`）を
構築。Download Strategy Validationにより「1営業日1リクエストで全銘柄取得（Strategy C）」が
銘柄別リクエストより効率的（総リクエスト数少・再現性高）と判明し採用。

**ASK_FIRST③④ 実施完了（2026-07-10・ユーザー承認）**:
- Full Download（`--study75-download`）: 2016-07-11〜2026-07-09の全2,439営業日、
  **10,084,970行・5,376銘柄**（上場廃止944件含むsurvivorship-free）取得。
  バックグラウンド実行中に一度中断したが、チェックポイント設計により再開は1リクエストのみで完了。
- verify: 2,439日全件 status=ok（欠落・破損・行数不一致・ハッシュ不一致ゼロ）。
- catalog.json / manifest.json 生成。dataset_hash記録。
- Universe復元（`--rebuild-universe`・Option B・完全オフライン・API通信ゼロ）:
  6,326イベント（ADD 5,382 / REMOVE 944）、現在上場中4,438銘柄。

**実行中に発見・修正した実バグ**: オフラインUniverse復元のギャップ時break処理で末尾未flushバッファが
破棄される不具合を発見・修正（`git 4649743`・回帰テスト追加）。データ破損はなし
（dedupにより自己修復可能な設計だった）。

**Study75 Universe Design（記述分析のみ・バックテストなし）**: `reports/study75_universe_design.md`。
現行ユニバース統計（銘柄数/年・上場廃止数/年・流動性分布・ロットコスト分布）+
候補フィルター4案（A全銘柄/B ADV20/C ADV20+ロット/D TOPIX500近似）を評価。
全て point-in-time 実装であれば生存者バイアス・lookahead biasの新規混入なしと確認。
現行資本¥3MではC、将来¥20-30M想定ではDがキャパシティ面で有利という所見（決定はユーザー判断）。
検証物: `backtests/study75_universe_design_2026-07-10.json`。

**未実施**: `enrich_universe_reference_with_listed_info()`（TOPIX500真値化用メタデータ補完・
API通信約5,376件・別途ASK_FIRST）。Study75本体（月次規則ユニバース生成ロジック）。
Study75データセット上でのWFバックテスト（Study76が既に待機中・Study75完了後着手）。

---

---

## 実行ログ追記（2026-07-11・Sonnet実行）— L2 Study75A/75B 完了

- **Study75A**: PIT Universe Generator（`src/backtest/study75_universe_generator.py`）完成。
  Universe C（ADV20≥¥300M∧lot feasible∧上場60営業日以上・月次T-1スナップショット）120ヶ月分を
  `backtests/study75_rule_universe.json` へ出力（STUDY75_UNIVERSE_FILEスキーマ・Study76が直接消費可能）。
- **PIT audit**（`reports/study75_pit_audit.md`）: rebalance/snapshot分離=SAFE・IPO 60日=SAFE・
  ADV20=SAFE・**lot feasibility=PARTIAL**（2018-10単元統一以前は未検証・診断に記録）。
- **Study75B**（`reports/study75_survivorship_report.md`）: 4シナリオ比較。
  Diagnostic Aで公式値12.22/11.42を完全一致再現（wiring証明）後、J-Quants基盤へ再基準化。
  U0=+8.71% / U2=-8.25%（IS）→ **Delta_A=-16.96pp（交絡3種の注記付き・詳細は本体報告）**。
  廃止銘柄トレードの系統的損失（-¥1.30M vs 現存+¥0.35M）を実証 — 旧公式値の上方バイアス確認。
  Study74 BLACK維持（強化）。**Study76ブロック解除条件は3つとも充足**。
- 実行中に修正した実バグ: エンジンの上場廃止セマンティクス欠如への対応（ffill+alive-mask二重ビュー・
  エンジン本体は無改変）・276A0型直近上場銘柄のspan切り詰め・欠落17銘柄のmaterialize。

---

## 実行ログ追記（2026-07-11・Fable5実行）— Study75C バイアス分離（E1 PITブートストラップ）

Study75BのDelta_A=-16.96ppの解釈監査 → ランキング・セマンティクス汚染（5経路・
`reports/study75c_interpretation.md`§1-2）で無効と判定。E1（PITブートストラップ・K=20+ツイン20+
U0'アンカー・プールサイズ42固定でセマンティクス完全統制）により分解を実測:
**生存者バイアス=-0.87pp（統計的ゼロ・E2ゲート不成立→E2不要）/ 選定バイアス=+12.26pp
（RSR42=測定窓選定ユニバース・PIT分布の95パーセンタイル・PIT期待値median IS -2.48%）**。
Study74 BLACK維持（a fortiori）。実行中に発覚・修正: エンジンのセクターキャップが'不明'セクター
集約でE1初回実行を1ポジション戦略に退化させていた問題（擬似固有セクターで解消・初回結果は破棄）。
未決（ユーザー決裁）: Core期待値・CP1目標体系の再アンカー。次ステップ=Study76。

---

## 実行ログ追記（2026-07-13・Sonnet 5実行）— Study75D/E/F（暫定命名）+ FUJIKO-R2ロードマップ再構築

**★番号衝突の注記**: 本セッションはユーザー指示で「Study76」「Study76D」「Study77」の名称を
使用したが、canon（本ファイル）が予約する Study76=Clenow純正ベンチマークWF・Study77=Exit構造
置換WFとは**別内容**。`reports/fujiko_r2_research_roadmap.md`にて**Study75D/E/Fへの改名を提案・
ユーザー決裁待ち**。canon Study76/77は未実行のまま予約継続。以下、提案名称で記録する。

**Study75D（暫定・D_ATR_EQ Study75-Universe再ベースライン）**: D_ATR_EQ本体（Exit/リスク/breadth/
dyn_rsr42_bear_rs0）を無改変のまま、Study75AのUniverse C月次プールから各月T-1トレイリング
composite return上位42名を機械選抜する「Dynamic RSR42」に静的RSR42を置換して測定。
**Δ_dynamic = RunB − U0 = IS -25.17pp / OOS +62.27pp**（IS CAGR-16.46%・MaxDD-85.02%／
OOS CAGR+61.29%）。月次turnover平均44.57%・在籍月数中央値3.0ヶ月。成果物:
`reports/study76_datr_eq_universe_c_rebaseline.md`（ファイル名は当時のまま）。

**Study75E（暫定・contamination ablation）**: Study75D病理診断で確定した「FujikoStrategy内RSRの
ffill汚染（実測99%）」を0埋めで除去したRunB_fixedをfresh run。**修正すると成績が悪化**
（IS CAGR -24.98%・OOS CAGR+42.71%・Δ_bug IS-8.52pp/OOS-18.58pp）——事前仮説（汚染が偽の好調
シグナルを作っている）は反証された。0埋め自体が別の不連続アーティファクト（在籍開始時の
モメンタム急上昇）を生んでいる可能性が高く、**RunB・RunB_fixedいずれもクリーンな測定ではない**。
成果物: `reports/study76d_contamination_ablation.md`。

**Study75F（暫定・Dynamic42 Path Decomposition）**: 2025 OOS+61.29%を分解 →
**単一銘柄(23340)が総利益の79.04%・単月(8月)が81.63%**（判定=数銘柄集中・偶然性大）。
セクターローテーション仮説の相関は弱い負（corr=-0.252、n=96・根拠なし）。IS崩壊の主因は
position-level cap（sector/cluster cap）ではなく候補枯渇+breadth連動停止。
成果物: `reports/study77_dynamic42_path_decomposition.md`（ファイル名は当時のまま）。

**統合作業**: 上記3件+Study75C（E1監査）を統合し、静的RSR42前提を完全に外した次世代研究
ロードマップ`reports/fujiko_r2_research_roadmap.md`を新規作成。既存研究を
再利用可能/要再検証/廃棄候補へ分類（Part A）。Market→Sector→Stock階層モメンタム仮説を設計
（Part B）。Study87-94（純データ分析2本→診断1本→fresh run4本の順で情報価値/コスト比を
維持する依存構造）を新規起案。**旧正典Phase2-5（Study79/82-86）は本ロードマップ完了まで
一時凍結を提案**（RSR42基準の「素の実力10-12%」前提が崩れたため）。

**未決（ユーザー決裁待ち）**: (1) Study75D/E/Fへの正式改番 (2) Phase2-5凍結の是非 (3) Study87
以降の起案承認。次ステップ=Study87（Warm-up修正版ユニバース生成器）またはStudy88
（セクターモメンタム持続性、Study87と並行可）。

---

## 実行ログ追記（2026-07-14・Fable 5実行）— FUJIKO 2.0 Ground Truth Reconstruction（fujiko_r2 v2）完了記録

**性格**: 文書統合作業のみ（新規BT・コード変更・実弾変更なし）。
`reports/fujiko_r2_research_roadmap.md`を**v2へ全面改定済み（2026-07-13・Fable 5）**。
本メモはそのSAVE伝播（research_state.md / 本ファイル）の追記（2026-07-14）。
ユーザー指示「FUJIKO 2.0 Ground Truth Reconstruction」Part1-5に完全準拠:

- **Part1**: Study01-77を**Study単位**でA（高信頼再利用: 75系バイアス実測・インフラ検証・
  エンジンforensics・WF/統計手法）/ B（条件付き: 構造的発見のみ・絶対値再較正必須）/
  C（再検証必須: 全Production採用判定・D_ATR_EQ系譜・**strategy_review_2026-06-28**・
  M1後Official値）/ D（廃棄候補・凍結保持: Dynamic42 v1・Study75B Delta_A・
  **strategy_review_2026-04-13**・Study52キャッシュ旧数値）へ分類。
- **Part2**: 6仮説の事前確率評価——CS momentum有効=0.45 / Market→Sector→Stock階層=0.35 /
  セクター持続性=0.40 / **RSR定義無効=0.75**（プールサイズ従属セマンティクス＝移植不能）/
  Entry/Exitエッジ残存=0.30 / Dynamic universe必須=二分割（PIT必須6a≈0.95・
  月次ローテーション形態6b≈0.25）。全仮説に検証Studyを割当。
- **Part3/4**: Candidate A（Market→Stock Clenow）/ B（Market→Sector→Stock）/
  C（Sector ETF Rotation）/ D（Dynamic Top500）/ E（Hybrid）をエッジ源泉・失敗モード・
  必要データ・計算コスト・実装複雑性・期待頑健性の6軸で評価。
  **検証優先順位 A→D→B→C・Eは原則起案しない**（アーキテクチャ選択自体のselection bias防止）。
- **Part5**: Phase R0-R4の新roadmap。**Study95（CSモメンタムfactor-level ground truth・H0・
  最優先・即時着手可）/ Study96（Entry/Exit帰属分解・H5）/ Study97（Sector ETFデータ実現性・
  条件付き）を新設**。統治原則: factor-first・honestベースライン3基準（Universe Cランダム
  median / TOPIX B&H / バイアス補正後RSR42推定）・パーセンタイル型パラメータ新規採用禁止・
  現行実弾との分離（Study94まで実弾変更を派生させない）。プログラムレベルKill条件を定義
  （H0失敗→旧正典ARCH系PEAD/TSMOMへ転進提起）。

**RSRランク付けへの対処**: FUJIKO 2.0ランキングはプールサイズ非依存の絶対スコア
（Clenow slope×R²等）を第一候補とし、min_rsr等パーセンタイル型の新規採用を禁止（仮説4）。
**strategy_review両版**: 2026-04-13=D（凍結参考）・2026-06-28=C（ゼロベース再検証・
Study94完了後に新版起草）——以降の意思決定根拠としての使用禁止。

**未決（ユーザー決裁待ち5点）**: (1) Study75D/E/F改番 (2) Study95/96/97採番・起案承認
(3) canon Study76をPhase R2基準器として実施する承認 (4) 旧正典Phase2-5凍結の正式承認
（例外並行候補: Study80/83） (5) strategy_review両版の凍結参考値格下げの正式承認。
**次の一手=Study95**（純データ分析・BTエンジン不要・1日規模・依存はStudy75Aのみ）。

---

## 実行ログ追記（2026-07-14・Fable 5実行）— Study95 完了（H0・CS Momentum Factor-Level Ground Truth）

**性格**: 新規fresh run（純粋クロスセクション統計・フジコ法/RSR/percentile型トレーディング
パラメータ/BTエンジン一切不使用・タスク仕様の禁止事項を厳守）。成果物:
`reports/study95_cs_momentum_factor_level.md` / `backtests/study95_cs_momentum_factor_level.json` /
`reports/study95_decile_chart.png` / `src/backtest/study95_cs_momentum_factor_level.py`。
コミット未実施。

**データ**: Study75A Universe C（PIT月次・119ヶ月）・panel=108,895行（rebalance×code）・
価格ファイル欠落=0銘柄。2ファクター: 12-1モメンタム（P[t-21]/P[t-252]-1）・
Clenow slope90d×R²（canon Study76と同一定義）。1M/3M/6M/12M forward return・IC・
Q10-Q1スプレッド（Newey-West t + block bootstrap 95%CI）・monotonicity（Spearman）・
regime分解（TOPIX>MA200）・sector分解（TOPIX17 IC + sector-neutral demean版）・
容量診断（ADV20/turnover）・factor persistence（rank自己相関）を実装。

**判定（ユーザー指定基準を機械適用）**: **両ファクターともFAIL・Kill基準機械発動=True**。
- 12-1モメンタム: 12M年率spread=-1.83%（NW-t=-0.368）。ただし3M/6M/12M ICのt統計量が
  有意に負（-2.77/-3.46/-2.83）——「ゼロ」というより**弱い逆転**（Decile10=過去勝者が
  6M/12Mで明確に最下位）。Study61 FalseHero率67.8%等の既存知見とfactor-levelで整合。
- Clenow: 1M/3M/6M horizonで有意な正の単調性（Spearman ρ=0.818 p=0.004等）・正スプレッド
  （1M=+7.02%等）を示すが12Mで反転（-1.79%）し合格基準「複数期間で一貫」未達。
  Bear regimeで-7.64%（t=-2.681・有意）と顕著悪化。
- Sector-neutral・容量診断で交絡（sector bet・流動性アーティファクト）は否定。

**未決（ユーザー決裁）**: プログラムレベルKill条件（fujiko_r2_research_roadmap.md v2）が
形式上発動——「Candidate A-E全凍結・旧正典ARCH系(PEAD/TSMOM)へ転進提起」が既定の帰結。
ただしClenowの短期(1-6M)regime依存シグナルは完全なゼロではないため、
(a) Kill基準通り全面凍結してPEAD/TSMOM転進 (b) Clenow短期シグナルをregime-gated型で
限定継続検証、のいずれをとるかはユーザー判断が必要。**全凍結の実行（Study87以降着手停止）は
自動実行しない**（ユーザー承認後に着手）。

*作成: CRO/Chief Architect, 2026-07-04。新規バックテスト実行なし。本書の全タスクは実行前にASK_FIRST該当有無を0.3で確認すること。正典（final_research_roadmap_2026-07-04.md）と矛盾する場合は正典が優先。*
