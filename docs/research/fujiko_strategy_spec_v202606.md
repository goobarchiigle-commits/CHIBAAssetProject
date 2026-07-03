# フジコ法 現行版仕様書（v202606）

作成日: 2026-06-24
方式: コードベース直接抽出（推測禁止）。引用は全て `file:line` 形式。
対象: **本番実行パス**（`src/run_live_signal.py` → `SignalBridge.run()` → `_build_orders()`）。
研究専用コード（`src/backtest/study9〜29*.py` 等）はこの仕様書の対象外 → 差分は `docs/research/production_research_diff_v202606.md` を参照。

既存の自動生成スペック `src/generate_strategy_spec.py`（出力例: `docs/research/strategy_spec_2026-04-13.md`）と本書の内容は概ね一致するが、**§4 RSR Exit閾値に既知のバグがある**（後述、注記1）。本書はそのバグを修正した値を記載する。

---

## 0. 実行エントリポイントと全体フロー

```
run_live_signal.py (main)
  └─ SignalBridge(...) 構築                         signal_bridge.py:703
       └─ bridge.run()                              signal_bridge.py:4326
            ├─ データ取得（yfinance/kabu API）
            ├─ RSR計算 + 動的ユニバース活性リスト    src/strategy/universe.py
            ├─ 銘柄別シグナル決定ループ              signal_bridge.py:2040-2390
            │    （Exit優先判定 → Entry判定）
            ├─ BUY候補ランキング（composite score）  signal_bridge.py:2410-2541
            ├─ top_k選定（4スロット目ゲート）        signal_bridge.py:2542-2561
            └─ _build_orders()                       signal_bridge.py:3283-3607
                 ├─ CB状態チェック（BUY全停止 or 通常）
                 ├─ SELL注文構築
                 ├─ Adaptive Allocator（sector/cluster cap）
                 ├─ リスクベース・サイジング（ATR + alloc cap）
                 └─ ロット丸め・配分上限ゲート
  └─ (run_live_signal.py に戻る) ポスト処理オーバーレイ
       ├─ Runtime Exit Orchestrator hook            run_live_signal.py:2847-2874
       ├─ Portfolio Intelligence Engine hook        run_live_signal.py:2965-2988
       └─ Exit Intelligence（観測専用）              run_live_signal.py:2990-3058
  └─ dry-run 表示 / --live --yes 時のみ実発注
```

詳細な依存関係図は `docs/research/production_dependency_graph_v202606.md` を参照。

---

## 1. 資金・コスト設定

| パラメータ | 値 | 出典 |
|---|---|---|
| 初期資本 | ¥3,000,000 | `strategy.yaml:36` (`portfolio.capital`) |
| max_positions | 3（PARAMS_LOCKED） | `strategy.yaml:37` |
| max_single_weight | 0.25（CIRCUIT） | `strategy.yaml:39` |
| スリッページ | 0.1% | `strategy.yaml:46` (`costs.slippage_rate`) — **バックテストのみ適用、ライブ注文サイズには未適用**（§5 注記2） |
| 手数料 | 0.055% | `strategy.yaml:46` (`costs.commission_rate`) — 同上 |
| 注文単位 | 100株 | `src/live/capital_deployment_os.py:47` (`LOT_SIZE`) |

---

## 2. ユニバース・動的ユニバース選定

- 固定プール: RSR42（42銘柄, `src/configs/rsr_universe_42.csv`）
- 動的選定モジュール: `src/strategy/universe.py`
- ライブ呼び出し: `signal_bridge.py:1677-1690`（active syms 取得） / `signal_bridge.py:2510-2541`（BUY候補へのフィルター適用、fail-open＝空集合ならフォールバックして全候補使用）
- 設定: `strategy.yaml:101-117`（`dynamic_universe.enabled=true`, `pool=rsr42`, `bull_active_n=30`, `bear_active_n=20`, `bear_rs_filter=true`）
- Bear時除外セクター: `strategy.yaml:136-143`（機械・鉄鋼・銀行業・保険業・輸送用機器・海運業・化学）

ステータス: **LIVE-ACTIVE**（post-RSRフィルター、fail-open）。`wf_dyn_rsr42_2026-04-05.json` で WF 5/5 検証済み（`generate_strategy_spec.py` §14出力参照）。

---

## 3. エントリー条件（`FujikoStrategy`、本番共有クラス）

実装: `src/backtest/fujiko_strategy.py:45-` (`class FujikoStrategy`)。このクラスは**バックテストと本番ライブの両方で共有**される（`signal_bridge.py:2085`: `fujiko_strat = FujikoStrategy(... **self._fujiko_params_live)`）。

全条件AND（`fujiko_strategy.py:297-302`, `:394-409`）:

| # | 条件 | 閾値/ロジック | 出典 |
|---|---|---|---|
| 1 | SEPAスコア | ≥6（8点中） | `strategy.yaml:8` (`min_sepa`) / `fujiko_strategy.py:298` |
| 2 | RSR（ユニバース内パーセンタイルランク） | ≥75.0 | `strategy.yaml:8` (`min_rsr`) / `fujiko_strategy.py:299,400` |
| 3 | RSRモメンタム（21日差分） | >0 かつ前日比上昇 | `strategy.yaml:10` (`mom_period=21`) / `fujiko_strategy.py:300` |
| 4 | タートルズS1ブレイクアウト | 前日までの20日高値超え（`use_turtle_entry=true`時） | `strategy.yaml:13-14` / `fujiko_strategy.py:301-302` |
| 5 | 動的ユニバース活性 | 当月active listに含まれる（fail-open） | `signal_bridge.py:2510-2541` |
| 6 | MTFフィルター（BUYのみ） | 週足RSR≥75.0 かつ 週足終値>週足MA20 | `signal_bridge.py:2254-2278` |

SEPA 8条件の内訳（`fujiko_strategy.py: _calc_sepa_score_array`、`generate_strategy_spec.py`出力§5で確認済み）:
Close>MA150&MA200 / MA150>MA200 / MA200上向き / MA50上向き / Close>MA50 / 52週安値+30%以上 / 52週高値-25%以内 / RSR≥70.0。

### ランキング（composite score、top_k選定）

```
composite = RSR + RSR_momentum × MOM_WEIGHT_ADJ + Entry_Timing_Boost
```
- `MOM_WEIGHT_ADJ`: 通常0.3 / トレンドクラスター level1=0.5 / level2=0.7（`signal_bridge.py:2417-2421`）
- `Entry_Timing_Boost` = `(et_score-50) × boost_weight`、`boost_weight=0.06`（`strategy.yaml:214`）。`entry_timing.enabled=true`時のみ加算。`block_low_confidence=false`（デフォルト）のためLOW判定銘柄もブロックされない。（`signal_bridge.py:2423-2505`）
- 4スロット目ゲート: 保有数が3（max_positions上限）の時、4枚目候補はRSR≥80.0必須（`signal_bridge.py:2542-2560`）

ステータス: 全項目 **LIVE-ACTIVE**。

---

## 4. エグジット条件（判定順、`signal_bridge.py:2055-2214`）

優先順位（最初にTrueになった条件が即時SELLを生成、`signal_bridge.py:2171-2214`）:

| 優先 | 条件 | ロジック | 閾値 | 出典 |
|---|---|---|---|---|
| 1 | Composite Shock Exit | 保有中 かつ ベンチ前日比≤-5%（shock day） かつ 個別株前日比≤-8% | `shock_exit_mode="composite"` | `strategy.yaml:59` / `signal_bridge.py:2155-2177` |
| 2 | トレーリングストップ | 終値 < 保有来高値 − 3.0×ATR20 | 倍率3.0はハードコード（config外） | `signal_bridge.py:2064-2082` |
| 3 | 時間ストップ | 保有営業日数 ≥ 60 | `strategy.yaml:26` (`max_hold_days`) | `signal_bridge.py:2055-2062, 2189-2196` |
| 4 | RSR系エグジット（**多層**+**単純閾値**のOR） | 後述 | — | `signal_bridge.py:2114-2123, 2198-2205` |
| 5 | mean_rev反発失敗（mean_rev保有のみ） | エントリー後4営業日以内にHigh が entry×1.01未達 かつ Close<entry×0.995 | `MEANREV_FAIL_DAYS=4` / `MEANREV_MIN_BOUNCE=0.01`（環境変数デフォルト） | `signal_bridge.py:2125-2153, 2207-2214` |
| — | 緊急エグジット（min_hold無視） | 含み損 ≤ -8% | `strategy.yaml:27` (`emergency_exit_pct`) | `signal_bridge.py:2099-2112` — 上記条件4のmin_hold判定をバイパスする修飾子（独立した優先順位ではない） |
| 6（fallback） | 戦略シグナル（`FujikoStrategy.generate_signal`内部exit） | RSRモメンタム<0かつ下降 **または** 終値<前日までの55日安値 | `turtle_exit=55` | `strategy.yaml:12` / `fujiko_strategy.py:296,386-392` — 1〜5のいずれも発火しない場合のみ到達 |

### 注記1: RSR Exit閾値の確定（既存自動生成docのバグ訂正）

production の RSR Exitには**2つの独立した数値**が存在し、混同してはならない:

1. **エントリー閾値 `min_rsr=75.0`**（`strategy.yaml:8`）— エントリー専用。
2. **エグジット閾値 `rsr_exit=70.0`**（`strategy.yaml:9`）— `signal_bridge.py:780-782`: `self.rsr_exit_threshold = float(fujiko_params.get("rsr_exit", fujiko_params.get("min_rsr", 75.0)))`。`strategy.yaml`に`fujiko.rsr_exit=70.0`が定義されているため、フォールバック値75.0は使われず、**実際の閾値は70.0**。コード中のコメント（`signal_bridge.py:777`）も明示: 「min_rsr はエントリー専用（変更禁止）。exit 閾値は rsr_exit で分離。」

   `signal_bridge.py:2118`: `_simple_rsr_exit = (rsr_now < self.rsr_exit_threshold)` → **RSR<70.0**。

   既存の `src/generate_strategy_spec.py:549` は出力テキスト中でこの2つを取り違え、「RSR低下エグジット」の閾値ラベルに `{min_rsr}`（75.0）を誤って使用している（生成スクリプト側のバグ。コード自体は正しく70.0で動作している）。**`generate_strategy_spec.py`の該当箇所修正を推奨**（§D参照）。

3. **多層RSRエグジット（z-score方式、`compute_multilayer_rsr_exit`, `signal_bridge.py:196-243`）**: 上記1.2.とは別の、RSRのz-score化系列を使う4条件OR判定。
   - exit_1: RSR_Z<1.1（RSR≈55相当）
   - exit_2: RSR速度<-1.5σ かつ z<1.3（崩壊検知、RSR≈65相当）
   - exit_3: ピークからの乖離>0.6 かつ z<1.6（ピーク乖離、RSR≈80相当）
   - exit_4: 保有≥5日 かつ z<1.2（停滞、RSR≈60相当）

   `_ml_exit`（多層判定）と`_simple_rsr_exit`（RSR<70）は**OR結合**（`signal_bridge.py:2121-2123`）→ どちらか一方でも成立すれば exit_4のRSR系エグジットが発火。

ステータス: 全項目 **LIVE-ACTIVE**。`turtle_exit=55`（フォールバック枠6）は「デッドパラメータ」ではなく、優先順位最下位として現在も到達可能（ただし優先順位1-5のいずれかが通常先に発火するため、実際の発火頻度は低い）。

---

## 5. 平均回帰サブ戦略（非フジコ法セクター）

実装: `src/backtest/mean_reversion_strategy.py`、ライブ呼び出し `signal_bridge.py:2093-2094`。
セクター割当: `SECTOR_STRATEGY`（ガス・保険・化学・小売・輸送機器・鉄鋼・銀行 → mean_rev、他はfujiko、`dynamic`ルールはどちらか発火した方を採用）。

| エントリー | エグジット |
|---|---|
| RSI(5)<25 かつ Close>MA200 かつ Close>MA50×0.85 | RSI(5)>65 **または** Close<entry×0.93（-7%） **または** 保有≥10営業日 |

クラスターブロック: `_pre_cluster_mode and not market_trend` の時、mean_rev BUYをブロック（`signal_bridge.py:2235-2242`）。

---

## 6. ポジションサイジング（`_build_orders`, `signal_bridge.py:3283-3607`）

```
qty_risk = (capital × 0.0125 ÷ ATR20) を100株単位に丸め（0なら100株フォールバック）
qty_cap  = effective_alloc_cap ÷ (price×100) を100株単位に丸め
qty      = min(qty_risk, qty_cap)
effective_alloc_cap = capital × max_single_weight(0.25)
                       × regime_scale(bear時 bear_scale, デフォルト1.0)
                       （RSR≥85かつ未使用のリーダースロットなら capital×0.35）
qty_cap==0 → 当該銘柄はBUYスキップ（配分上限キャップ未達 = 1単元すら買えない）
```

出典: `signal_bridge.py:3362-3363`（alloc cap） / `3406-3407`（リーダースロット, RSR≥85 → 0.35） / `3510-3589`（ATR risk/cap計算） / `src/live/capital_deployment_os.py:172-188`（同種の `target_position_size()` 補助関数、ただし実発注パスはsignal_bridge.py内で独自に再計算。後者は診断用途）。

セクター/クラスター制約は Adaptive Allocator（`src/execution/adaptive_alloc.py`）が事前ランキングと事後縮小を担当（`signal_bridge.py:3366-3463`）。

| 制約 | Bull時 | Bear時（TOPIX<MA200） | 出典 |
|---|---|---|---|
| sector_cap | 25% | 18% | `strategy.yaml:67,69` |
| cluster_cap | 35% | 25% | `strategy.yaml:68,70` |
| symbol_cap | 40%（固定, `dynamic_cap=false`） | 同 | `strategy.yaml:64` |
| 同一セクター同時保有 | ≤1銘柄 | 同 | `strategy.yaml:78-81` |
| セクター合計ウェイト | ≤35% | 同 | 同上 |

### 注記2: ライブで未適用のオーバーレイ（観測専用 / DEAD）

| 機能 | 状態 | 出典 | 理由 |
|---|---|---|---|
| Capital Scaling scalars（volatility/liquidity/execution_scalar） | OBSERVATION-ONLY | `strategy.yaml:121-136`、`signal_bridge.py`に`volatility_scalar`等の参照なし（grep確認済み） | テレメトリ用途のみ。実発注サイジングは§6の独立した式を使用 |
| Position Sizing Intelligence | OBSERVATION-ONLY | `strategy.yaml:223-224` (`auto_apply=false`) / `signal_bridge.py:2838-2882` | virtual_weight計算済みだがqty計算に未反映。`auto_apply=true`化はASK_FIRST対象 |
| Gross Exposure Control | **DEAD（本番未実装）** | `strategy.yaml:72-75` (`gross_exposure_enabled=true`) | `signal_bridge.py`/`run_live_signal.py`に実装なし（grep確認済み, 0件）。`src/backtest/composite_alpha_bt.py`等の研究/バックテストコードにのみ実装あり |
| slippage_rate / commission_rate | バックテストのみ適用 | `strategy.yaml:46` / `src/backtest/composite_alpha_bt.py` (`SLIPPAGE`,`COMMISSION`) | ライブ注文の`qty`計算には未反映。CLAUDE.md「slippage_required/commission_required」はバックテストP&L側でのみ満たされている |
| Predictive Expansion 実発注オーバーレイ | INERT | `strategy.yaml:230-240` (`predictive_entry_enabled=false`) | analytics/shadow watchlistのみ稼働 |

---

## 7. サーキットブレーカー

実装: `src/risk/circuit_breaker.py`。

| 状態遷移 | 条件 |
|---|---|
| NORMAL → ENTRY_STOP_ONLY (CB_ACTIVE) | DD ≤ -15%（`max_dd_limit`, `strategy.yaml:40`） |
| ENTRY_STOP_ONLY → NORMAL | DD ≥ -5% 回復 **または** 30営業日経過 |

CB_ACTIVE中: 新規BUYは全停止、SELLは常に許可（`signal_bridge.py:3104, 3311-3345`）。CLAUDE.md CIRCUITの記述（「自動撤退はしない・警告のみ」）と一致 — **CBはエグジットを強制しない**。

---

## 8. 実行制御（ライブ専用）

| パラメータ | 値 | 出典 |
|---|---|---|
| 最大新規BUY/日 | 2件 | `signal_bridge.py: max_new_positions_per_day` |
| 発注レート制限 | 5件/秒（CLAUDE.md指定）、コード内実装は要個別確認 | CLAUDE.md `rate_limit=5/s` |
| dynamic_max_positions（CDOS） | tier+2 を `min(..., PARAMS_LOCKED=3)` で三重クランプ | `src/live/capital_deployment_os.py:159-169`、`signal_bridge.py:4658-4670` |
| Runtime Exit Orchestrator | suppress_exit / force_exit（fail-open、bridge.run()後に適用） | `run_live_signal.py:2843-2874`、`src/runtime/policy/runtime_exit_orchestrator.py` |
| Portfolio Intelligence Engine | 8 bounded policies（fail-open、bridge.run()後に適用） | `run_live_signal.py:2965-2988`、`src/runtime/policy/portfolio_intelligence_engine.py` |
| Idempotency | SHA256(day\|symbol\|side\|qty\|strategy) | `src/live/client_order_id.py:24-50` |
| 重複検知 | `InflightRegistry.is_duplicate()` | `src/live/inflight_registry.py:332-341` |

`dynamic_max_positions`はCLAUDE.md PARAMS_LOCKED=3を実行時に上書きしない（過去メモリで指摘されたバグは現在のコードでは三重クランプにより解消済み — `capital_deployment_os.py:169`の`min(result, PARAMS_LOCKED_MAX_POSITIONS)` + `signal_bridge.py:4664,4670`の2段防御）。

---

## 9. この仕様書の射程外（別文書）

- 本番コードと `Study9`〜`Study29` 研究系列との不一致 → `docs/research/production_research_diff_v202606.md`
- 依存関係の詳細図 → `docs/research/production_dependency_graph_v202606.md`
- 再ベースラインに必要なバックテストコマンド → `docs/research/rebaseline_backtest_commands_v202606.md`
