# 本番ロジック再現性監査（production_backtest_parity_report）

作成日: 2026-06-24
方式: コード直接比較（推測禁止）。引用は全て `file:line`。
比較対象: 本番=`src/kabusapi/signal_bridge.py`+`src/backtest/fujiko_strategy.py`（`docs/research/fujiko_strategy_spec_v202606.md`で確定済み）/ バックテスト候補=`src/backtest/composite_alpha_bt.py`（`run_scenario(scenario="BASELINE")`）。
本書で「確認済み」と記す全項目は、本セッションで実コード（`composite_alpha_bt.py`, `fujiko_strategy.py`, `mean_reversion_strategy.py`, `signal_bridge.py`の該当行）を直接読み確定。

---

## 結論（先出し）

`composite_alpha_bt.py --full-history --full-scenarios` の **BASELINE** シナリオは、Entry/Exitの中核ロジックは本番と一致するが、**サイジング方式・Exit優先順位の一部・ランキング方式・サーキットブレーカーの挙動が本番と異なる**。「本番を最大限そのまま再現」という要求水準には未達。Phase2のフルバックテストにこのまま使う場合、結果は「本番に近い近似値」であり「本番の真の実力値」ではない。

一致率（条件数ベース、後述の確定差分表 n=19項目）: **MATCH 10 / PARTIAL 1 / MISSING 8 → 52.6%**

---

## 確定差分表

| # | 項目 | 本番（出典） | バックテスト（出典） | 判定 |
|---|---|---|---|---|
| 1 | SEPAスコア≥6 | `fujiko_strategy.py:298` | `fujiko_strategy.py:298`（共通クラス使用） | ✅ MATCH |
| 2 | RSR floor≥75.0 | `fujiko_strategy.py:299,400` | 同上（共通クラス） | ✅ MATCH |
| 3 | RSRモメンタム>0かつ上昇 | `fujiko_strategy.py:300` | 同上 | ✅ MATCH |
| 4 | タートル20日高値ブレイク | `fujiko_strategy.py:301-302` | 同上 | ✅ MATCH |
| 5 | 動的ユニバース活性フィルター | `signal_bridge.py:2510-2541` | `composite_alpha_bt.py:786-789`（`sym_active_mat`, fail-open） | ✅ MATCH |
| 6 | MTFフィルター（週足RSR≥75 かつ週足Close>週足MA20） | `signal_bridge.py:2254-2280` | **実装なし**（grep確認: MTF/weekly関連コード0件） | ❌ MISSING |
| 7 | ランキング: composite=RSR+mom×weight+entry_timing_boost | `signal_bridge.py:2417-2505` | BASELINE: `rank_score=rsr_val`のみ（`composite_alpha_bt.py:798`, docstring:432「RSR降順ランキング」） | ❌ MISSING |
| 8 | Exit優先1: Composite Shock Exit（TOPIX≤-5%かつ個別≤-8%） | `signal_bridge.py:2155-2177` | `composite_alpha_bt.py:701,725-734`（`market_shock_mode="composite"`、閾値同一） | ✅ MATCH |
| 9 | Exit優先2: ATRトレーリング（保有来高値-3.0×ATR20） | `signal_bridge.py:2064-2082` | **BASELINE未実装**（`use_trail_exit`はSTEP5/6のみtrue, `composite_alpha_bt.py:441,761`。さらにSTEP5/6でも参照は`high50_close`＝50日高値で「保有来高値」ではなく、倍率も`trail_atr_mult`で3.0固定か要再検証） | ❌ MISSING |
| 10 | Exit優先3: 時間ストップ（保有≥60営業日） | `signal_bridge.py:2055-2062` | `composite_alpha_bt.py:737-739`（`max_hold_days`同一） | ✅ MATCH |
| 11 | Exit優先4: RSR単純閾値<70 | `signal_bridge.py:2118` | `composite_alpha_bt.py:444,741`（`rsr_exit_threshold=cfg.fujiko.rsr_exit`で70と一致） | ✅ MATCH |
| 12 | Exit優先4: 多層RSR z-score判定（OR結合） | `signal_bridge.py:2114-2123`(`compute_multilayer_rsr_exit`) | **実装なし** | ❌ MISSING |
| 13 | Exit優先5: mean_rev反発失敗（4日以内bounce未達） | `signal_bridge.py:2125-2153` | **実装なし**（`mean_reversion_strategy.py`は内部のRSI>65/-7%/10日のみ、bounce-fail判定なし） | ❌ MISSING |
| 14 | 緊急Exit（含み損≤-8%、min_hold無視） | `signal_bridge.py:2099-2112` | **実装なし** | ❌ MISSING |
| 15 | Exit優先6（fallback）: RSRモメンタム下降または55日安値割れ | `fujiko_strategy.py:296,386-392` | 同一クラス使用のため一致（`composite_alpha_bt.py:776-777`の`STRATEGY_EXIT`） | ✅ MATCH |
| 16 | サイジング: qty_risk=capital×0.0125÷ATR20 | `signal_bridge.py:3510-3589` | **実装なし**（ATR risk式が一切存在しない） | ❌ MISSING |
| 17 | サイジング: max_single_weight=25%キャップ | `strategy.yaml:39` | `composite_alpha_bt.py:991`（`capital*max_single_weight`） | ✅ MATCH |
| 18 | リーダースロット（RSR≥85→35%キャップ） | `signal_bridge.py:3404-3408` | **実装なし** | ❌ MISSING |
| 19 | セクター/クラスターキャップ（bull/bear別） | `strategy.yaml:67-70` | `composite_alpha_bt.py:897-898,921-957`（同一値使用） | ✅ MATCH |

---

## 追加で確認した非対応・逆方向の差異（上記n=19外）

| 項目 | 内容 | 影響方向 |
|---|---|---|
| サーキットブレーカー挙動 | 本番: DD≤-15%で新規BUY**全停止**（`circuit_breaker.py`, spec§7）。BT: `CB_SCALE=0.35`でサイズ縮小のみ、BUY自体は継続（`composite_alpha_bt.py:77,893`） | BTの方が本番よりリスクを取る（過大評価方向） |
| Gross Exposure Control | 本番: **DEAD（未実装、spec§6注記2で確認済み）**。BT: `gross_exposure_enabled=true`で実際にcapを適用し新規BUYを抑制（`composite_alpha_bt.py:608-613,959-969`） | BTの方が本番より保守的（過小評価方向、本番には存在しない制約） |
| 最大新規BUY/日=2件 | 本番: `signal_bridge.py: max_new_positions_per_day` | BT: 実装なし（1日に複数新規でも制限なし） | BTの方が緩い |
| Entry Timing Boost | 本番: `boost_weight=0.06`でcomposite scoreに加算（`strategy.yaml:214`, LIVE-ACTIVE） | BT: 未実装 | ランキング選抜結果が変わる可能性 |

---

## 解釈

1. **Entry中核（FujikoStrategy本体）は完全一致**: `fujiko_strategy.py`を両者が共有クラスとして使用しているため、SEPA/RSR floor/momentum/turtle breakoutの4条件と、turtle_exit=55のfallback exitは100%再現性がある。
2. **Exit優先順位は本番6段中、BTは実質3段のみ再現**（shock/time_stop/RSR単純閾値）。ATRトレーリング・多層RSR・mean_rev bounce-fail・緊急exitの4機構が欠落。これらは「より早く・より多く決済する」方向の機構が多く、欠落により**BTはホールド期間が本番より長くなる方向のバイアス**を持つ可能性が高い。
3. **サイジング方式が根本的に異なる**: 本番はATRに応じて縮小するリスクベース方式（ボラが高い銘柄ほど小さく買う）。BTは現金等分＋25%キャップのみ。個別トレードのPnL・ポートフォリオのDDプロファイルが本番と系統的に異なる可能性がある。
4. **サーキットブレーカーとGross Exposureは逆方向に効く2つの誤差**: CBはBTの方がリスク許容（過大評価方向）、Gross ExposureはBTの方が保守的（過小評価方向、しかも本番に存在しない制約をBTだけが持つ）。両者が部分的に相殺する可能性があり、最終的な方向性は実測しないと不明。

---

## production_status

`production_status=PARTIAL_REPRODUCTION_ONLY`

`composite_alpha_bt.py`(BASELINE)は「本番に最も近い既存ツール」だが、「本番を最大限そのまま再現」という今回のタスク要件を満たさない。これをそのままPhase2フルバックテストの基盤として使う場合、得られるKPIは**本番の真の実力値ではなく近似値**であり、最終結論フォーマットの`confidence`欄はこの事実を反映して引き下げる必要がある。

完全再現エンジンの新規構築（ATR risk sizing / MTF filter / 多層RSR exit / mean_rev bounce-fail / 緊急exit / リーダースロット / CB全停止化 / Gross Exposure除去 / Entry Timing Boost）はCLAUDE.md PERMISSIONの「新規スクリプト作成・既存スクリプト大規模改修」に該当し、着手前にASK_FIRST必須。
