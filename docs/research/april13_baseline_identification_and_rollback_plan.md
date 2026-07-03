# Exact April-13 Trading Logic Identification & Proven-Only Baseline Rollback Plan

作成日: 2026-06-24
方式: コード調査のみ（推測禁止・パラメータ推定禁止・再構築の当て推量禁止）。**バックテスト未実行・コミット未実行・実装変更なし**（本タスク範囲）。

## 前提制約（最初に報告すべき事実）

```
git status → fatal: not a git repository (or any of the parent directories): .git
git log    → fatal: not a git repository
git rev-list --all → fatal: not a git repository
```

**`C:/ai-trading` はgitリポジトリではない**（`.git`ディレクトリが存在しない）。CLAUDE.mdは`remote=git@github.com:.../CHIBAAssetProject.git`と`branch=main`を記載しているが、実体としてのgit管理は行われていない。

**結論**: STEP1項目1「Exact commit(s)」は**原理的に提供不可能**。コミットハッシュは存在しない。以降の全項目は「現在のファイル内容＋ファイル内のコメント（変更日付付き）」からの調査結果であり、「2026-03-31時点のコード」を直接見ているわけではない。両者が異なる可能性がある箇所は明示する。

---

# STEP1: Locate Baseline

## 0. ソース特定（確実）

`backtests/min_hold_sensitivity_2026-03-31.json` の `hold3d.IS` を実測値と照合した結果、以下が完全一致した。

| | レポート/メモリ記載値 | JSON実測値（hold3d.IS, 2018-2024） |
|---|---|---|
| CAGR | 22.4% | **22.4%** |
| Sharpe | 1.582（記載は1.35との混同あり） | **1.582** |

**結論（確実）**: 2026-04-13レポートの数値は `src/backtest/min_hold_sensitivity.py` を `min_hold=3` で実行した **IS（2018-2024）** 区間の結果。OOS(2025)は同設定でCAGR+0.1%/Sharpe0.067と別途記録されている（レポートの主張する性能はIS区間限定）。

## 1. Exact commit(s)

**UNKNOWN — 提供不可能**。gitリポジトリが存在しないため、コミットハッシュは記録されていない。`min_hold_sensitivity.py`のdocstringとファイルの存在のみが手掛かり。

## 2. strategy.yaml used

**部分的にUNKNOWN**。現在の`src/configs/strategy.yaml`には変更日コメントが個別に付与されている項目があり、それらは2026-03-31以降に変更されたことが**明示的に記録**されている:

| パラメータ | 現在値 | コメント（変更日の直接証拠） | 2026-03-31時点の値 |
|---|---|---|---|
| `fujiko.turtle_exit` | 55 | 「exit感度テストで20→55に変更 **2026-03-31**（IS Sharpe +12.6%）」 | **55**（同日確定、`min_hold_sensitivity.py`のdocstring「turtle_exit=55（2026-03-31確定値）」と一致 — 確実） |
| `fujiko.rsr_exit` | 70.0 | 「WF5fold検証済 **2026-06-05**」 | **UNKNOWN**（現在値は本レポートより2ヶ月以上後に確定。2026-03-31時点の値は記録なし。後述の通りBASELINEシナリオでも`rsr_exit_threshold`は常時使用されるため、この未確定はSTEP1全体に影響する） |
| `portfolio.max_positions` | 3 | 「PARAMS_LOCKED確定値（**2026-03-31**）」 | **3**（同日確定 — 確実） |
| `portfolio.capital` | 3,000,000 | 「2026-03-30 実口座300万円と統一」 | **3,000,000**（前日確定 — 確実） |
| `dynamic_universe.enabled` | true | ファイル先頭「2026-04-05: 動的ユニバース採用」 | **記録なし/無関係**（後述: `min_hold_sensitivity.py`はこのセクションを読まないため無関係） |
| `fujiko.min_sepa`/`min_rsr`/`mom_period`/`turtle_entry`/`use_turtle_entry` | 6/75.0/21/20/true | コメントなし（変更日不明） | **UNKNOWN（変更履歴コメントなし、現在値が当時から不変かは未確認）** |

**重要な発見（コード実証）**: `src/backtest/composite_alpha_bt.py`の`run_scenario()`内で、**RSR Exitの単純閾値チェック（920行目）はシナリオ分岐なしに常時実行される**:
```python
# composite_alpha_bt.py:920
_simple_rsr_exit = rsr_val < rsr_exit_threshold
```
`rsr_exit_threshold`はNone時に`cfg.fujiko.rsr_exit`へフォールバックする（570行目）。`min_hold_sensitivity.py`はこの引数を渡していないため、**実行時点のstrategy.yamlの`rsr_exit`値がそのまま使われていた**。現在値(70.0)は2026-06-05確定と明記されているため、2026-03-31時点の値ではない可能性が高いが、当時の実値は記録が残っていないため**確定不可**。

## 3. Universe selection logic

**ファイル**: `src/backtest/min_hold_sensitivity.py:61-64`
```python
rsr_syms   = _bt._load_rsr_universe()
trade_syms = rsr_syms
all_syms = {**rsr_syms, **trade_syms}
```
`_load_rsr_universe()`（`composite_alpha_bt.py:132-138`）は`src/configs/rsr_universe_42.csv`を読み込み、**静的42銘柄（RSR42）**を返す。`run_one()`は`sym_active_df`を渡していない（`min_hold_sensitivity.py:99-112`）→ `run_scenario`内のデフォルト`sym_active_df=None`が使われる → **動的ユニバース・セクター/クラスター集中キャップは完全に未適用**（`_enable_conc_caps = sym_active_df is not None` が`False`になる箇所が複数存在）。

## 4. Entry logic

**ファイル**: `src/backtest/fujiko_strategy.py:61-106, 280-420`
SEPA条件数 ≥ `min_sepa`（cfg経由、現在値6）かつ RSR ≥ `min_rsr`（cfg経由、現在値75.0、変更禁止パラメータ）かつ（`use_turtle_entry=True`時）直近`turtle_entry`日高値（現在値20日）ブレイクアウト。`scenario="BASELINE"`のため`use_alpha_rank=False`（RSR降順ランキング、Composite Alphaランキング不使用）。

## 5. Exit logic

**ファイル**: `composite_alpha_bt.py:868-944`（BASELINEシナリオで実際に到達するブロックのみ）
1. Market Shock判定（後述）
2. （`enable_atr_trailing_prod`=False のため未到達）
3. Time Stop（`max_hold_limit`、`cfg.risk.max_hold_days`現在60日、コメントなし=変更履歴不明）
4. **RSR Exit**: `rsr_val < rsr_exit_threshold`（上記の通り当時値不明） かつ `hold_idx >= min_hold`（スイープ対象: 0/3/5/7/10）
5. （`use_phase_exit=False`のため未到達のBREAKEVEN_STOP/ATR_TRAIL）
6. 以降のFujikoStrategy固有exit（momentum/turtle、`exit_momentum_mask_full`/`exit_turtle_mask_full`）

**Multi-layer RSR Exit・ATR Trailing Exit（PROD_FAITHFUL専用機能）は2026-03-31当時に存在していたか不明** — これらの実装（`enable_multilayer_rsr`/`enable_atr_trailing_prod`パラメータ自体）が現在のコードに存在するが、`min_hold_sensitivity.py`は明示的にこれらを渡しておらずデフォルトFalseとなる。これらの機能がそもそも当時実装されていたかどうかはgit履歴なしに確認不可能だが、**呼び出し側で明示的に有効化されていない以上、実行結果には影響していない**（これは確実 — デフォルト値の存在の有無に関わらず、無効化されていたことは呼び出しコードから直接言える）。

## 6. Shock logic

**ファイル**: `composite_alpha_bt.py:509, 866-900`
`market_shock_mode`は渡されていない → デフォルト`"full_exit"`（509行目: `market_shock_mode: str = "full_exit"`）。TOPIX日次リターン ≤ `composite_market_thr`（デフォルト-5%）で**全ポジション強制全量決済**。現在の本番（`risk_controls.shock_exit_mode: composite`、2026-04-05 WF検証済み採用＝本レポートの2週間後）とは異なる、より単純な決済ロジック。

## 7. Position sizing logic

**ファイル**: `composite_alpha_bt.py:1217-1226`（`enable_atr_risk_sizing=False`時の`else`分岐）
`sizing_mode`は渡されていない → デフォルト`"existing"`:
```python
# composite_alpha_bt.py:1206-1208 (existing分岐)
n_remaining = sum(1 for _, _, _, s in buy_candidates if s not in positions)
effective_slots = min(open_slots, max(1, n_remaining))
alloc = (cash / effective_slots) * regime_step * cb_scale * ext_scale * sym_scale
```
**現金÷残り候補数で動的分配**（固定`capital/max_positions`の"equal"ではない）。ATR Risk Sizing（本番のリスクベース1/ATR方式）は不使用。

## 8. Capital allocation logic

**ファイル**: `min_hold_sensitivity.py:109` — `capital = _bt.CAPITAL`

**重大な発見（確実・実証済み）**: `_bt.CAPITAL`という属性は**現在の`composite_alpha_bt.py`に存在しない**（grep全文検索で一致なし、`CAPITAL`という裸の定数は定義されていない。唯一のマッチは`print(f"CAPITAL {cfg.portfolio.capital:,}")`という別関数内のf-string）。
**これは`min_hold_sensitivity.py`を現在の状態で実行すると`AttributeError: module 'backtest.composite_alpha_bt' has no attribute 'CAPITAL'`で即座に失敗することを意味する。**

この事実は、composite_alpha_bt.pyが2026-03-31以降に「CAPITALというモジュール定数を削除し、`cfg.portfolio.capital`を直接参照する設計に変更された」ことの**直接証拠**である。これにより、本レポートのSTEP1全体に以下の重要な留保が確定する: **「現在のcomposite_alpha_bt.pyのデフォルト値」は2026-03-31時点のデフォルト値と必ずしも一致しない。一致を主張できるのは、ファイル内に明示的な変更日コメントがある項目（turtle_exit, max_positions, capital）のみ。**

## 9. Active filters

**ファイル**: `composite_alpha_bt.py:517, 569-571`
`enable_filters`は渡されていない → デフォルト`False`（517行目）。`enable_filters=False`の場合`filter_frames = {}`（570行目）となり、`enable_volatility_filter`/`enable_atr_filter`/`enable_volume_filter`/`enable_market_filter`の個別デフォルト（いずれも`True`）は実質無効化される（filter_frame未生成のため適用対象なし）。**結論: エントリーフィルターは全て不使用。**

---

# STEP2: APRIL_13_BASELINE vs CURRENT_PRODUCTION 差分

CURRENT_PRODUCTIONは`signal_bridge.py`本番ロジック（Study32-35で確認済み、`composite_alpha_bt.py`内コメントで`signal_bridge.py`行番号を直接引用）。

| 項目 | APRIL_13_BASELINE | CURRENT_PRODUCTION | 分類 |
|---|---|---|---|
| Universe | 静的RSR42 | 動的(dyn_rsr42_bear_rs0、Bull Top30/Bear Top20+rs>0+セクター除外) | 追加機能 |
| Shock Exit | full_exit（全量決済） | composite（個別-8%のみ決済） | ロジック変更 |
| RSR Exit閾値 | UNKNOWN（70.0は確実に異なる時期の値） | 70.0 | パラメータ変更（基準不明） |
| ATR Trailing Exit | 無効（明示的に未呼出） | 有効（highest_close - 3×ATR20） | 追加機能 |
| Multi-layer RSR Exit | 無効（明示的に未呼出） | 有効（z-score 4層OR結合） | 追加機能 |
| MTF Filter | 無効（明示的に未呼出） | 有効（週次RSR≥75+週次MA20上） | 追加機能 |
| Position Sizing | "existing"（現金÷残候補数で動的分配） | ATR Risk Sizing（capital×1.25%÷ATR20） | ロジック変更 |
| Capital | ¥3,000,000（モジュール定数経由、現在は削除済み） | ¥3,000,000（cfg.portfolio.capital経由、動的資本連動あり） | 実装変更（値は同一） |
| Entry Filters | 全無効 | 不明（本タスク範囲外、production側未調査） | 未調査 |
| min_hold | スイープ対象(0/3/5/7/10) | 3（strategy.yaml risk.min_hold_days） | 一致（hold3d採用時） |
| max_positions | 3 | 3（PARAMS_LOCKED） | 一致 |

---

# STEP3: 変更分類（完了済みStudyのエビデンスのみ使用）

| 変更 | 分類 | エビデンス |
|---|---|---|
| 動的ユニバース(Universe) | **A: Proven Positive** | Study33: ΔCAGR+0.77pp/ΔSharpe+0.055（単独追加で改善）。Study34: DD増加の69.4%の原因だが、Study33で確認済みのリターン改善と表裏一体（トレードオフであり欠陥ではない） |
| ATR Trailing Exit | **A: Proven Positive（小）** | Study33: ΔCAGR+0.30pp/ΔSharpe+0.013（単独追加で改善、効果量は小） |
| Multi-layer RSR Exit | **C: Not Proven（実質ノイズ）** | Study33: ΔCAGR+0.00pp/ΔSharpe+0.000（単独追加効果ゼロ、既存シンプルRSR Exitと完全冗長） |
| MTF Filter | **B: Proven Negative** | Study33 IS: ΔCAGR-1.56pp/ΔSharpe-0.079。Study35 True OOS 2025: C(MTF有)Sharpe0.379 vs D(MTF無)Sharpe0.881 — IS・OOS両方で一貫して負 |
| ATR Risk Sizing | **B: Proven Negative** | Study32: 5方式アブレーションでA_EQUAL/E_NO_ATRがB(現行ATR)をCAGR/Sharpe/PF全指標で優位（confidence=HIGH、REMOVE判定）。Study35 WF: B(ATR有)はWF5fold中2fold失敗(3/5) |
| Shock Exit (full_exit→composite) | **C: Not Proven（本タスク範囲では単独検証なし）** | Study33waterfallでは"Universe"に同時バンドルされ単独分離未実施。個別の証明済みStudyなし |
| RSR Exit閾値(70.0の妥当性) | **C: Not Proven（基準点不明のため判定不可）** | 2026-03-31時点の値が不明であり、「変更」自体の内容を特定できない |
| sizing_mode (existing→equal、ATR除去後の代替) | **A: Proven Positive** | Study32: A_EQUAL(equal weight)が全指標で最良。Study33: C(equal採用)が新ベースライン確定 |

---

# STEP4: Candidate Rollback Plan（検証済みのみ・実装はASK_FIRST対象、本タスクでは未実施）

制約: execution/broker/monitoring/validation/safety/reportingは touch しない。`src/strategy/`・`src/backtest/composite_alpha_bt.py`の該当フラグ・`signal_bridge.py`の発注ロジック部分のみが対象（signal_bridge.py変更はCLAUDE.md PERMISSION上ASK_FIRST必須）。

提案された順序を、Study32-35のエビデンスで検証済みの単位に対応付ける:

| Step | 内容 | 対応Study | 分類 | Rollback Risk | Dependency Risk |
|---|---|---|---|---|---|
| 1. Remove ATR Risk Sizing | `enable_atr_risk_sizing=False`+sizing方式をequalへ | Study32(REMOVE,confidence=HIGH) | A（除去がProven Positive） | 低（バックテストでconfidence=HIGH） | sizing_modeの選択（equal）が次段に伝播、Step7と整合必須 |
| 2. Remove Multi-layer RSR Exit | `enable_multilayer_rsr=False` | Study33(ゼロ寄与) | 低リスク（効果ゼロのため除去で性能変化なし、簡素化のみ） | 極小 | 既存シンプルRSR Exitと完全冗長のため依存なし |
| 3. Remove MTF Filter | `enable_mtf_filter=False` | Study33(-1.56pp/-0.079)+Study35(OOS確認, mtf_value=NEGATIVE) | A（除去がProven Positive） | 低 | Universe(動的)との相互作用は未検証（MTFは動的ユニバースと独立に候補を絞るため、理論上独立だが交互作用テストはStudy33-35の範囲外） |
| 4. Restore Static RSR42 Universe | 動的ユニバース除去・`sym_active_df=None`化 | **未検証（本タスクで新たに提案された組合せ）** | 該当Study無し→**C** | **高**（Study33でUniverseは単独Proven Positiveであり、これを除去する方向のロールバックは性能を悪化させる可能性が高い。Step1-3と矛盾する提案） | Study34のDD分析（Universe=DD増加の69.4%）と表裏一体のCAGR改善を喪失する |
| 5. Restore Full Exit Shock | `market_shock_mode="full_exit"`に戻す | 未検証（Study33waterfallでUniverseと同時バンドル、単独効果不明） | C | 中（単独効果が未測定のため不明） | Step4のUniverse変更と同時にwaterfall内で扱われていたため、単独適用時の効果は別途検証が必要 |
| 6. Restore RSR Exit 75 | `rsr_exit_threshold=75.0` | **基準点（2026-03-31時点の真値）が不明のため「復元」ではなく「変更」** | C | 不明（当時値が70だったか75だったか不明なため、これが本当に"restore"かどうか自体が未確定） | min_rsr=75.0（エントリー専用、変更禁止）と数値が同一だが意味が異なるため混同注意 |
| 7. Restore Equal Weight Sizing | `sizing_mode="equal"` | Study32(A_EQUAL最良) | A | 低 | Step1のATR除去と一体（Step1のみではsizing_mode="existing"のままでも動作するが、Study32の"equal"はStep1のATR除去結果と紐づいたエビデンス） |

**再適用提案（Composite Shock Exit / RSR Exit 70 / ATR Trailing Exit）の評価**:
- ATR Trailing Exit再適用 → **A: Proven Positive**（Study33で確認済み、整合）
- Composite Shock Exit再適用 → **C: Not Proven単独**（Universeとバンドルされた効果のみ測定済み、単独再適用の効果は未検証）
- RSR Exit 70再適用 → **C: Not Proven**（「70」という数値の根拠はStudy33-35のCURRENT_PRODUCTION定義に過ぎず、2026-03-31基準との比較対象が不明なため「どちらが優れているか」を結論できない）

**Step4・5・6（Universe静的化・Shock full_exit化・RSR Exit 75復元）は、Study32-35で既に確立された結論（D_PROD_MINUS_ATR_MINUS_MTF = final_verdict PROMOTE_D）と直接矛盾する。** Study35のPROMOTE_D構成は「動的Universe＋composite shock＋rsr_exit=70＋ATR Trailing」を維持し「ATR Risk Sizing＋MTF Filter」のみ除去する設計であり、提案されたロールバック順序のStep1-3はこれと整合するが、Step4-6は逆方向（Study33でProven Positiveと判定済みのUniverseをむしろ除去する方向）に進む。

---

# DELIVERABLE まとめ

**1. Exact April-13 strategy definition**: STEP1参照。`min_hold_sensitivity.py`実行（`min_hold=3`, IS=2018-2024）= 静的RSR42ユニバース・SEPA≥6+RSR≥75+turtle20日エントリー・turtle_exit=55(確実)+RSR Exit閾値(**不明**)・Shock=full_exit(デフォルト推定)・Sizing="existing"動的現金分配・フィルター全無効。**コミットハッシュ特定不可（gitなし）**。

**2. Current vs April diff table**: STEP2参照。Universe/Shock/RSR Exit閾値/ATR Trailing/Multi-layer RSR/MTF/Sizingの7項目で差異。

**3. Proven-positive changes**: 動的Universe、ATR Trailing Exit、sizing_mode=equal（ATR除去後の代替として）。

**4. Proven-negative changes**: ATR Risk Sizing、MTF Filter。

**5. Candidate rollback sequence**: 提案されたStep1-3（ATR Sizing/MultiRSR/MTF除去）はエビデンスと整合し**実施候補として妥当**。提案されたStep4-6（Universe静的化/Shock full_exit化/RSR Exit75復元）は**既存Proven Positiveと矛盾する方向であり非推奨**。Step7（Equal Weight）はStep1と一体でProven Positive。再適用提案のうちATR Trailingのみ単独でProven、Composite Shock/RSR Exit70は単独未検証。

**6. Files affected**（実装する場合、本タスクでは未実施）: `src/configs/strategy.yaml`（rsr_exit等パラメータ、ASK_FIRST対象）, `src/backtest/composite_alpha_bt.py`（フラグ既存、本番側コードではない）, `src/live/signal_bridge.py`（発注ロジック本体、ASK_FIRST必須・本タスク未調査）。

**未解決事項（推測せず明示）**: (a) rsr_exit閾値の2026-03-31時点の真値、(b) Shock Exit full_exit→composite切替の単独寄与、(c) min_sepa/min_rsr/mom_period/turtle_entryが当時から不変だったかの確証 — いずれもgit履歴なしでは解決不可能。
