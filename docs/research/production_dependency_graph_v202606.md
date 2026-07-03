# 本番ロジック依存関係図（v202606）

作成日: 2026-06-24
方式: `src/run_live_signal.py` から実際にimport/呼び出しされているモジュールのみを逆追跡（コードベース直接抽出）。

---

## 1. 注文生成フロー（実行順）

```
run_live_signal.py (main)
│
├─[1] 起動前ガード（EMERGENCY_STOP系）
│     api_unreachable / position_sync_fail / duplicate_order_detected
│     / portfolio_state_missing → abort （CLAUDE.md EMERGENCY_STOP）
│
├─[2] 状態ファイル読込
│     portfolio_state.json, CAPITAL_STATE_FILE, DEPLOYMENT_RAMP_STATE_FILE,
│     CAPITAL_FREEZE_STATE_FILE  ※[2]の一部は読込のみで未使用（§3 DEAD参照）
│
├─[3] SignalBridge 構築                                    signal_bridge.py:703
│     入力: fujiko_params(strategy.yaml), capital, max_positions,
│           max_single_weight, rsr_exit, min_hold_days, max_hold_days,
│           emergency_exit_pct, shock_exit_mode, regime_sizing, bear_scale
│
├─[4] bridge.run()  ※ThreadPoolExecutorで150sタイムアウト          signal_bridge.py:4326
│     │
│     ├─[4a] データ取得: yfinance + ローカルキャッシュ
│     │      MIN_DAILY_VALUE_YEN 流動性フィルター
│     │
│     ├─[4b] RSR計算                                        src/backtest/rsr.py
│     │      calc_composite_return (IBD式12ヶ月加重) → calc_universe_rsr (pct rank)
│     │
│     ├─[4c] 動的ユニバース活性リスト                          src/strategy/universe.py
│     │      is_sustained_bear() → Bull/Bear分岐スコアリング → active syms
│     │      呼び出し: signal_bridge.py:1677-1690
│     │
│     ├─[4d] kabu API: ポジション・余力取得                    signal_bridge.py:1475-1507
│     │
│     ├─[4e] 銘柄別シグナル決定ループ（全銘柄を順次評価）         signal_bridge.py:2040-2390
│     │      ├─ FujikoStrategy.generate_signal()              src/backtest/fujiko_strategy.py:339
│     │      ├─ MeanReversionStrategy.generate_signal()       src/backtest/mean_reversion_strategy.py
│     │      ├─ compute_multilayer_rsr_exit()                 signal_bridge.py:196-243
│     │      ├─ トレーリングストップ計算（calc_atr）            src/portfolio/volatility_allocator.py
│     │      └─ 優先順位判定（§フジコ法spec §4参照）→ signal_int確定
│     │
│     ├─[4f] BUY候補ランキング（composite score）              signal_bridge.py:2410-2541
│     │      ├─ Entry Timing Engine                           src/entry/__init__.py
│     │      │    compute_entry_timing_for_candidates / apply_entry_timing_boost
│     │      ├─ Entry Timing Promotion（auto_apply_boost=false時は未使用）
│     │      │                                                src/entry/entry_timing_promotion.py
│     │      └─ 動的ユニバースフィルター適用（fail-open）
│     │
│     ├─[4g] top_k選定 + 4スロット目ゲート                     signal_bridge.py:2542-2561
│     │
│     └─[4h] _build_orders()                                  signal_bridge.py:3283-3607
│            ├─ CB状態チェック（cb_state: NORMAL/CB_ACTIVE/RECOVERY）
│            │                                                src/risk/circuit_breaker.py
│            ├─ SELL注文構築（既存ポジション分）
│            ├─ AdaptiveAllocator                              src/execution/adaptive_alloc.py
│            │    set_existing_exposure / forecast / pre_rank_candidates
│            ├─ Deployability pre-ranking                      src/execution/deployability.py
│            ├─ リスクベース・サイジング（ATR + alloc cap, §フジコ法spec §6）
│            └─ ロット丸め・配分上限ゲート → OrderInstruction[] 確定
│
├─[5] ポスト処理オーバーレイ（bridge.run() の戻り値を加工、fail-open）
│     │
│     ├─[5a] Runtime Exit Orchestrator hook                   run_live_signal.py:2843-2874
│     │      run_exit_orchestration_hook()                    src/runtime/policy/runtime_exit_orchestrator.py
│     │      → suppress_exit / force_exit（ExitAdjustmentConstraints frozen内で変調）
│     │      → EXIT_POLICY_DECISIONS_FILE に決定ログ append-only
│     │
│     ├─[5b] Portfolio Intelligence Engine hook                run_live_signal.py:2965-2988
│     │      run_portfolio_intelligence_hook()                 src/runtime/policy/portfolio_intelligence_engine.py
│     │      → 8 policies（PIDTYPE_SLOT_SCALE 等）でsignals/order_objects変調
│     │      → PI_DECISIONS_FILE に決定ログ
│     │
│     └─[5c] Exit Intelligence hook（観測専用、orderは変更しない）  run_live_signal.py:2990-3058
│            src/analytics/exit_intelligence/hook.py
│
├─[6] dry-run表示 / サマリーレポート（日本語要約, MORNING_ROUTINE準拠）
│
└─[7] --live --yes 指定時のみ実発注
      ├─ client_order_id 生成（idempotency）                  src/live/client_order_id.py
      ├─ InflightRegistry.is_duplicate() チェック              src/live/inflight_registry.py
      ├─ broker_worker / kabu API 発注                        src/live/broker_worker.py
      ├─ ExecutionJournal 記録（append-only）                 src/live/execution_journal.py
      └─ position_sync / reconciliation                       src/live/position_sync.py, reconciliation_engine.py
```

---

## 2. モジュール別 LIVE-ACTIVE / OBSERVATION-ONLY / DEAD 判定

| モジュール | 役割 | 状態 | 根拠 |
|---|---|---|---|
| `src/backtest/fujiko_strategy.py` | Entry/Exit共有戦略クラス | LIVE-ACTIVE | `signal_bridge.py:2085` で直接インスタンス化 |
| `src/backtest/mean_reversion_strategy.py` | 非フジコ法セクター戦略 | LIVE-ACTIVE | `signal_bridge.py:2093` |
| `src/backtest/rsr.py` | RSR計算 | LIVE-ACTIVE | RSRユニバース計算の基盤（全エントリー/エグジット判定の入力） |
| `src/strategy/universe.py` | 動的ユニバース選定 | LIVE-ACTIVE | `signal_bridge.py:1677-1690, 2510-2541` |
| `src/entry/__init__.py`, `entry_timing_promotion.py` | Entry Timing Engine | LIVE-ACTIVE（ソフトブースト） | `signal_bridge.py:2423-2505` |
| `src/execution/adaptive_alloc.py` | セクター/クラスターcap・事前ランキング | LIVE-ACTIVE | `signal_bridge.py:3366-3463` |
| `src/execution/deployability.py` | デプロイ可能性メトリクス | LIVE-ACTIVE（ランキングに影響） | `signal_bridge.py:3455-3478` |
| `src/risk/circuit_breaker.py` | DD≤-15%でBUY全停止 | LIVE-ACTIVE | `signal_bridge.py:3104, 3311-3345` |
| `src/portfolio/volatility_allocator.py` | ATR計算（トレーリングストップ/サイジング） | LIVE-ACTIVE | `signal_bridge.py:2074, 3519-3526` |
| `src/runtime/policy/runtime_exit_orchestrator.py` | Exit抑制/強制オーバーレイ | LIVE-ACTIVE（fail-open） | `run_live_signal.py:2847-2874` |
| `src/runtime/policy/portfolio_intelligence_engine.py` | ポートフォリオレベル8ポリシー | LIVE-ACTIVE（fail-open） | `run_live_signal.py:2965-2988` |
| `src/live/capital_deployment_os.py` | dynamic_max_positions（PARAMS_LOCKED=3に三重クランプ） | LIVE-ACTIVE（クランプとして機能、拡張は実質発動せず） | `signal_bridge.py:4658-4670` |
| `src/live/client_order_id.py`, `inflight_registry.py` | 冪等性・重複防止 | LIVE-ACTIVE（安全装置） | 発注経路で必須 |
| `src/analytics/exit_intelligence/hook.py` | Exit velocity score記録 | OBSERVATION-ONLY | `run_live_signal.py:2990-3058`、orderを変更しない |
| `position_sizing`（`signal_bridge.py:2838-2882`内のPSIロジック, モジュール名は外部ファイル化されていない可能性あり） | virtual weight計算 | OBSERVATION-ONLY | `auto_apply=false`（`strategy.yaml:224`） |
| `strategy.yaml: capital_scaling.*`（volatility/liquidity/execution scalar） | 資金スケーリング | OBSERVATION-ONLY/未配線 | `signal_bridge.py`に参照なし（grep 0件） |
| `strategy.yaml: gross_exposure_enabled` | Gross exposure制御 | **DEAD（本番未実装）** | `signal_bridge.py`/`run_live_signal.py`に実装なし。`src/backtest/composite_alpha_bt.py`等の研究コードのみに実装あり |
| `src/addon/winner_confirmation.py` | Winner add-on確認 | **DEAD/未統合** | `run_live_signal.py`からの参照なし（grep 0件） |
| `DEPLOYMENT_RAMP_STATE_FILE`, `CAPITAL_FREEZE_STATE_FILE`の読込部 | ガバナンス状態 | ロードのみ、シグナル経路への分岐未確認 | `run_live_signal.py:1293-1294` — 個別の追加調査が必要（本書では「未確認」として記録、断定しない） |

---

## 3. 注記

- 「fail-open」とは: try/except で囲まれ、内部エラー時は無変更でフローを継続する設計（[5a][5b][5c]）。これは EMERGENCY_STOP の「fail-closed」原則（CLAUDE.md `Execution must fail-closed on: reconciliation failure...`）とは**異なる層**であることに注意 — fail-openは「観測/政策オーバーレイ」層、fail-closedは「ブローカー整合性・状態同期」層に適用される。両者は別レイヤーであり矛盾しない。
- `src/live/*.py`（約28モジュール）のうち、本図に明示していないもの（`broker_truth_snapshot.py`, `execution_integrity_validator.py`, `replay_consistency_validator.py`, `quarantine_governance.py`, `staged_supervisor.py` 等）は実行管理・整合性検証層であり、シグナル生成ロジック（Entry/Exit/Sizing）には影響しない。これらは「研究コードに対応物がない」（`docs/research/production_research_diff_v202606.md` §7参照）。
