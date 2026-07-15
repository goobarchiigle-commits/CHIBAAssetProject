# Study76 実行計画 — Clenow純正ベンチマークWF（複雑性の対価定量化）

**作成日**: 2026-07-04
**性格**: 実行計画のみ。本書自体はBacktest非実施・コード変更ゼロ・既存Research Assetsのみで作成。
**正典**: `reports/complete_execution_roadmap_2026-07-04.md` §L3（Section 4）/ `reports/final_architecture_review_2026-07-03.md` Study76節。矛盾時は正典優先。
**前提**: Study75（J-Quants survivorship-free universe）は契約待ちで未着手。本計画は「Study75完了後に即実行できる状態」を作ることが目的。

---

## 1. 検証目的（明文化）

現行Production（D_ATR_EQ）は5流派混血（Minervini+IBD+Turtle+Clenow+Carver）・Exit複数系統（RSR_EXIT/TURTLE_EXIT/ATR_TRAIL/ATR_TRAILING/RUNNER_TRAIL_EXIT/TIME_STOP/MARKET_SHOCK_EXIT/MARKET_SHOCK_PARTIAL/FIXED_PCT_TRAILING等・`composite_alpha_bt.py`実装ベース）・レジーム機構5重奏（dynamic_universe/bear_universe_filter/gross_exposure/shock_exit_mode/fraction）を持つ。この複雑性が実際にリターンを稼いでいるのか、それとも自由度の消費（過学習債務）に過ぎないのかは一度も直接比較されたことがない（design_philosophy_review_2026-07-03 §8/§9、final_architecture_review_2026-07-03 Task6）。

Study76は、単一パラダイム（Clenowモメンタム・ランクのみ・週次回転・単一レジームフィルター・ランク脱落回転Exit）の最簡構成を、Study75の survivorship-free 規則ユニバース上でD_ATR_EQと直接WF対決させ、「複雑性は+何ppを稼いだか」を1本の数字で回答する。

**問いは1つ**: 純正構成のΔCAGR（対D_ATR_EQ）はいくつか。
**使途は2つ**:
- 勝てば（ΔCAGR≥-2pp以内）→ 現行の多層構造（レジーム5機構・Exit複数系統・boost群）を削る根拠、FUJIKO-R骨格へのshadow移行検討に進む。
- 負ければ（ΔCAGR<-4pp）→ 複雑性は正当化される。FUJIKO-R骨格を放棄し現行維持を確定。Stage1 M6のDISCARD候補（turtle_exit/fraction.bull/entry_timing.boost_weight/vol_adj残置コード）は現状維持。

中間（-4pp〜-2pp）はグレーゾーンであり、正典に判定手順の明記なし — 発生時はユーザー決裁事項として報告する（機械判定ゲートの欠落を埋める形での裁量判断はしない）。

---

## 2. Production差分一覧（削除/固定/評価指標/成功基準/失敗基準）

### 2.1 何を削除するか（D_ATR_EQ → Study76純正構成）

| # | 現行要素 | 削除内容 | 根拠 |
|---|---|---|---|
| 1 | レジーム機構5重奏 | `dynamic_universe`（Bull/Bear別スコアリング・銘柄数可変）/ `bear_universe_filter`（セクター除外）/ `gross_exposure`（DD連動gross cap）/ `risk_controls.shock_exit_mode=composite`（Market Shock Exit）/ `fraction`（bull=0.0/bear=0.02の日次発注比率制御）を全廃止 → **TOPIX>MA200の単一フィルターのみ**に置換 | design_philosophy_review §8「Clenowは1本のMA200で済ませる仕事に5機構」 |
| 2 | Exit複数系統 | RSR_EXIT/ATR_EXTENSION(defer)/ATR_TRAILING/ATR_TRAIL/RUNNER_TRAIL_EXIT/TIME_STOP/TURTLE_EXIT/FIXED_PCT_TRAILING/MARKET_SHOCK_EXIT・PARTIAL/DEFERRED_EXITを全廃止 → **ランク脱落回転Exit（週次リバランスで上位N位から脱落したら機械的手仕舞い）のみ**に置換 | final_architecture_review Task6「Exit2種」/ design_philosophy_review §3「Exit哲学の一本化」 |
| 3 | Entry複合スコア | RSR42フィルター＋composite alpha（(slope×r2)²×RSR）＋`entry_timing.boost_weight=0.06`を全廃止 → **`(slope90d × R²)`ランクのみ**に置換 | final_architecture_review Task6「1エッジ」 |
| 4 | Addon/Sizing機構 | `eq_scale_addon`（含み益25%増し玉）/ `position_sizing`（Phase1観測）/ `adaptive_growth`（アグレッションEMA等）/ `quality_replacement`（shadow）を全廃止 → **均等ウェイト固定**（vol_target=0） | design_philosophy_review §5「資本と設計様式の不一致」への回答の一部 |
| 5 | Capital Scaling層 | `capital_scaling`（有効資本連動サイズ調整）を不使用 → 固定資本でのWF比較のみ | Study76はStudy74（資本スケーリング）と独立変数のため混在させない |

### 2.2 何を固定するか（両アーム・全期間で不変）

| 項目 | 値 | 備考 |
|---|---|---|
| ユニバース | Study75規則ユニバース（TOPIX500∩20日平均売買代金≥¥300M∩株価×100株≤有効資本×0.30、毎月第1営業日再適用） | RSR42は使用しない（selection bias混入のため） |
| リバランス頻度 | 週次 | 日次シグナルは使用しない |
| ランク基準 | `(slope90d × R²)` のみ | 複合スコア禁止 |
| レジームフィルター | TOPIX > MA200 の1本のみ | 5機構は使用しない |
| Exit | ランク脱落回転のみ | 個別ストップ・トレイル・時間切れ等は使用しない |
| ウェイト | 均等（vol_target=0相当） | サイジング機構は使用しない |
| コスト前提 | slippage=0.001 / commission=0.00055（PARAMS_LOCKED不変） | 変更禁止 |
| capital | ¥3,000,000（PARAMS_LOCKED不変） | Study74と変数分離 |
| パラメータ総数 | <10個・全て事前固定 | スイープ禁止（0.4ゲート「fresh run必須」と同じ精神。パラメータ探索は即閉鎖領域入り） |
| アーム数 | 2（Turtleブレイクアウト・トリガー有 / 無） | 3アーム目以降の追加は禁止 |

### 2.3 評価指標

CAGR（IS 2018-2024 / OOS 2025相当・Study75ユニバースの利用可能期間に準拠）、Calmar、Sharpe、MaxDD、WF 5fold pass数、2022年（Fold3相当）CAGR、Bootstrap P(>0)（Study73/78と同一N=500・seed=42手法を流用）、tail_capture（Study21基準・Study77への申し送り情報として参考記録のみ・Study76自体の合否には使わない）。

### 2.4 成功基準・失敗基準（正典固定・変更禁止）

| 判定 | 条件 | 帰結 |
|---|---|---|
| **成功** | 純正構成のΔCAGR（対D_ATR_EQ、同一期間・同一ユニバース）≥ **-2pp以内** | 現行多層を削る根拠。FUJIKO-R骨格をCoreエンジン置換候補としてshadow並走6ヶ月へ（成功∧WF5/5∧2022非悪化が全て揃った場合。roadmap Production採用条件） |
| **失敗** | ΔCAGR < **-4pp** | 複雑性は正当化。FUJIKO-R骨格放棄・現行Production維持確定。Study76以降（77）の起案は「勝者構成に対する」ものであるため、Study77も併せてスコープ消滅 |
| **未定義域** | -4pp ≤ ΔCAGR < -2pp | 正典に判定手順なし。裁量判断せずユーザー決裁を仰ぐ |

比較対象はD_ATR_EQ（CURRENT）。**【2026-07-04 ユーザー決裁確定】Study75完了後、D_ATR_EQを含む全比較対象はStudy75新Universe上でfresh run再測定した値のみを使用する。現行RSR42ベースの公式値（M1適用後: IS CAGR 12.22% / OOS CAGR 11.42% / FULL 11.22% / Calmar IS 0.671 / WF avg 17.99%・4/5 / 2022 -2.95%）は「旧Universe参考値」として保持するが、Study76の成功/失敗判定には一切使用しない（比較禁止）。** これによりUniverse差とArchitecture差の交絡を排除する。CAND_B（rsr_exit=75）はS1決裁未完了のため引き続き比較対象に含めない。

---

## 3. Study76実行に必要なResearch Assets（確認結果）

| Asset | 状態 | 用途 |
|---|---|---|
| Study75規則ユニバース + バイアス実測 | **未生成（J-Quants契約待ち）** | Study76の実行基盤そのもの。これなしでは新規BT不可 |
| D_ATR_EQ公式ベースライン数値（M1適用後・RSR42ベース） | ✅存在（`backtests/study_m1_production_update_2026-07-04.json` / roadmap §2.3） | **旧Universe参考値のまま凍結・Study76の比較対象には使用不可（2026-07-04ユーザー決裁）**。Study75完了後、同一エンジン・同一パラメータでStudy75 Universe上にfresh run再測定したものを比較対象とする（新規BT・Study76実行段階の一部） |
| `composite_alpha_bt.py`（既存エンジン） | ✅存在 | shift(1)/翌日寄付/コスト計算をライブラリとして継承。ただしStudy76の最簡構成は現行エンジンの大半の機構をOFFにする必要があり、新規スクリプト（既存study73_*.py構造踏襲）が必要 — ASK_FIRST対象（実行段階） |
| `src/backtest/archive/portfolio_clenow.py` / `clenow_momentum.py` | ✅存在（archive・参考実装） | Clenowモメンタムランク・slope×R²計算ロジックの実装参考。RSR42/yfinance現存銘柄ベースの旧実装のため**そのまま流用不可**（Study75ユニバース未対応・survivorship-free化前）。ロジック参照のみに使用 |
| Study78トレード分布（309件）・Worst10 DDエピソード | ✅存在（`backtests/study78_trade_dataset.json` / `study78_worst10_dd_episodes_2026-07-04.json`） | Study76結果の「WorstDD改善したか」の比較参照に流用可（新規BT不要、roadmap L1統合レビューPart2の方針を踏襲） |
| Study80A観測基盤（`missed_candidates_full.json`等） | ✅存在 | Study76は候補見送り分析を主目的としないため直接使用しないが、Exit回転設計時の副次参照として保持 |
| Study81クラスター分析 | ✅存在（REJECT確定） | Study76のユニバース設計に直接影響なし（クラスター抑制ロジックは既に反証済みのため導入しない） |
| J-Quants API認証情報（`src/.env` JQUANTS_MAIL_ADDRESS/JQUANTS_API_PASSWORD） | **未設定** | Study75完了の前提条件。Study76の直接の入力ではないが、依存元 |

**結論**: Study76固有の新規Research Assetは現時点で不足なし（比較対象値は既存・エンジンは既存・参考実装は既存）。唯一かつ決定的な欠落は**Study75規則ユニバースそのもの**であり、これは新規生成物（Study75完了待ち）。

---

## 4. Study77との依存関係

詳細は`study76_dependency_matrix.md`参照。要点:

- Study77（Exit構造置換WF）は**Study76の勝者構成に対して実施**（正典: 「順序: Study76決着後のみ（勝者構成に対して実施）」）。Study76が失敗（複雑性正当化・現行維持確定）で終わった場合、Study77の前提（「76勝者構成」）自体が消滅し、**Study77は起案自体が成立しなくなる**（別途ユーザー判断で「現行D_ATR_EQに対する3アームExit比較」に対象を差し替える再定義をするかは、Study76失敗確定後にユーザー決裁）。
- Study76が成功（純正構成採用）で終わった場合のみ、Study77は「純正構成のランク脱落回転Exit」に対する3アーム（A:回転+災害ストップ / B:災害ストップ+ATRトレイルのみ / C:現行多層）比較として正典どおり起案可能。
- したがってStudy77の実行計画は**Study76の結果（成功/失敗/グレー）が確定するまで具体化できない**。本タスクではStudy77側の計画書は作成しない（依存関係の整理のみ）。

---

## 5. Study75結果がStudy76へ与える影響の切り分け

詳細は`study76_dependency_matrix.md`参照。要点:

**Study75前でも着手できる部分**:
- 検証目的の明文化（本書§1・完了）
- Production差分一覧の確定（本書§2・完了。Study75の結果に依存しない — 純正構成の定義自体は普遍）
- 評価指標・成功/失敗基準の固定（本書§2.4・完了。正典既定のため変更なし）
- Research Assets棚卸し（本書§3・完了）
- Study77との依存関係整理（本書§4・完了）
- 比較対象ポリシーの確定（既存資産は「旧Universe参考値」に格下げ・完了）
- Clenow参考実装（archive）のロジック確認（コード変更はしないが読解は可能）
- 新規スクリプト設計仕様の文書化（ASK_FIRST承認後に着手可能な状態まで詰める。ただし本タスクではコード変更禁止のため、設計仕様書自体も次タスクで作成）

**Study75完了後でなければ実施できない部分**:
- 規則ユニバースの構成銘柄・期間データの取得（Study75の直接生成物）
- Study76の新規スクリプト実装（`src/backtest/study76_clenow_benchmark_wf.py`相当・ASK_FIRST）— ユニバースデータ形式が確定しないと入出力インターフェースが確定しない
- **D_ATR_EQ（比較対象）のStudy75新Universe上でのfresh run再測定**（新規BT）— Study76の純正構成と同一Universeで揃えるため必須。旧RSR42値との比較は禁止
- 純正構成側のfresh run実行（IS/OOS/WF5fold）
- 両者とも同一Universe・同一期間・同一コスト前提で測定した上でのΔCAGR算出

**【解決済み】Universe統制ポリシー（2026-07-04 ユーザー決裁確定・以降固定）**:
Study75完了時点で、Survivorship-free Universeを新しい基準Universeと定義し、以降のStudy76・Study77で使用する**全比較対象（D_ATR_EQを含む）は必ず同一Universe上で再測定した値のみを使用する**。旧Universe値（RSR42ベース）との比較は禁止。Universe差とArchitecture差の交絡を防ぐための恒久統制であり、Study76に限らずStudy77以降の全比較にも適用する。**Study75終了後、本ポリシーの適用をASK_FIRSTで確認してからStudy76の新規スクリプト実装・fresh run実行に進む**（確認内容: 「D_ATR_EQをStudy75 Universeで再測定する」ことの実行承認・再測定に使うfresh runスクリプトのASK_FIRST新規作成承認の2点）。

---

## 6. 未確定事項・次アクション

1. **J-Quants契約完了待ち**（ユーザー決裁待ち・進行中の別課題）。
2. Study75完了後、上記「Universe統制ポリシー」の適用をASK_FIRSTで確認（ポリシー自体は確定済み・確認は実行着手の儀礼的ゲートであり再検討ではない）。
3. 確認後、D_ATR_EQ再測定用スクリプトとStudy76純正構成スクリプトの新規作成についてASK_FIRST承認を取得してから実装着手（本タスクのスコープ外）。
4. 本計画に矛盾する正典改定があった場合は正典を優先し本書を訂正する。
