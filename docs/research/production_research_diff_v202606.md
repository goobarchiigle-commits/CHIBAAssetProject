# 本番ロジック vs 研究コード 差分一覧（v202606）

作成日: 2026-06-24
方式: コードベース直接比較（推測禁止）。本書は `fujiko_strategy_spec_v202606.md`（本番仕様）と `src/research_state.md` の Study9〜29 系列を対比する。

---

## 🔴 中心的発見: Study9〜29系列は本番コードを再現していない

`src/research_state.md` のStudy9（2026-06-20）以降、Study20〜29（2026-06-21〜06-24）に至る研究系列は、一貫して **「Study9 Case B」をFROZENベースラインとして引用**している（例: `research_state.md:52` 「Strategy=Study9 Case B (FROZEN)」）。このCase Bの実装は次の通り:

```python
# src/backtest/study9_standalone_validation.py:52-54
RSR_LO       = 92.0    # entry RSR lower bound (inclusive)
RSR_HI       = 95.0    # entry RSR upper bound (exclusive)
D90_MAX      = 5       # max days since RSR crossed 90
```
（エントリー条件: `study9_standalone_validation.py:219-220` `RSR∈[92,95) かつ d90≤5`。slope5≤5条件も同系列スクリプトに実装。Exit条件: `RSR<90`、`research_state.md:259,34`。）

**このロジックは `src/backtest/fujiko_strategy.py`（本番共有クラス）を一切importしていない**（`study9_standalone_validation.py:25-38` のimport文を確認、`FujikoStrategy`への参照なし）。エントリー/エグジット計算は同スクリプト内に独立して再実装されている。

一方、本番（`signal_bridge.py:2085`, `fujiko_strategy.py:298-302,400`）は:
```python
entry_mask &= sepa_score_arr >= 6           # SEPA>=6
entry_mask &= rsr_arr        >= 75.0        # RSR floor のみ（上限なし）
entry_mask &= (mom_arr > 0) & (mom_arr > mom_prev)
entry_mask &= close_arr > turtle_high_arr   # 20日高値ブレイク
```
exit閾値は `rsr_exit=70.0`（`strategy.yaml:9`, `signal_bridge.py:2118`）。

**`d90`・`slope5`という語は本番コード（`run_live_signal.py`, `src/kabusapi/signal_bridge.py`）に一件も出現しない**（grep確認済み、0件）。

### 確定差分表

| 要素 | Study9-29研究（Case B） | 本番（FujikoStrategy/signal_bridge.py） | 一致 |
|---|---|---|---|
| エントリーRSR | [92, 95) の狭帯域 | ≥75.0（floor、上限なし） | ❌ |
| d90（RSR90超え後の日数） | ≤5必須 | 該当ロジックなし | ❌ |
| slope5（傾き条件） | ≤5必須 | 該当ロジックなし | ❌ |
| エグジットRSR | <90 | <70（+多層z-score判定, §フジコ法spec §4） | ❌ |
| SEPA条件 | 言及なし（Case B独自実装に含まれず） | ≥6/8点必須 | ❌（研究側に存在しない） |
| タートルエントリー（20日高値） | 言及なし | 必須条件 | ❌（研究側に存在しない） |
| 資本/最大保有数 | Study20-27=¥1.8M/max_pos=1、Study28-29=¥3M/max_pos=3 | ¥3M/max_pos=3（PARAMS_LOCKED） | △（Study28以降のみ一致） |

### 含意

Study20〜29で確定した以下の研究判定は、**いずれも本番コードが生成するトレードを評価したものではない**:
- Study20 GO_HOLD / Study21 KEEP_MONITOR / Study22 RESEARCH_ENTRY / Study23 PARTIALLY_EXPLAINABLE / Study24 ALPHA_COMPONENT / Study25 PORTFOLIO_GEOMETRY_EXHAUSTED / Study27 EXHAUSTED / Study28 EXHAUSTED / Study29 RESEARCH_EXIT

これらは「RSR[92,95) d90≤5 slope5≤5 entry / RSR<90 exit」という**本番未実装の仮想戦略バリアント**に対する評価であり、本番のFujikoStrategy（SEPA+RSR floor+turtle breakout entry, RSR<70+多層z-score exit）の実績を表すものではない。研究結論自体（"EXHAUSTED"等)の論理は妥当だが、**適用対象が本番と異なる**。

---

## 1. エントリーRSR帯（§上記参照、再掲なし）

## 2. エグジットRSR閾値（§上記参照）

## 3. 資本・最大保有数

| 項目 | Study9-27 | Study28-29 | 本番 | 一致 |
|---|---|---|---|---|
| 資本 | ¥1,800,000 | ¥3,000,000 | ¥3,000,000 (`strategy.yaml:36`) | Study28-29のみ一致 |
| max_positions | 1 | 3 | 3 (`strategy.yaml:37`, PARAMS_LOCKED) | Study28-29のみ一致 |

ただし上記の通りEntry/Exitロジック自体が異なるため、資本/枠数の一致のみでは再現性は確保されない。

---

## 4. Entry Timing Engine（boost_weight=0.06）

- 本番: `strategy.yaml:212-216` で有効（`entry_timing.enabled=true`）。`signal_bridge.py:2423-2505`でcomposite scoreに加算。
- 研究: `research_state.md`のStudy9-29系列にEntry Timing Engineへの言及・検証記録は見当たらない（同系列はRSR[92,95)固定帯のみで候補を抽出するロジックのため、ランキングブースト自体が無関係)。別系列スクリプト`entry_timing_ab_test.py`が存在し、そこでA/B比較（boost_weight=0.0 vs 0.06）が行われている可能性があるが、これはStudy9-29のFROZENベースラインの検証対象には含まれていない。
- 結論: Entry Timing Engineは本番でLIVE-ACTIVEだが、Study9-29の評価対象には一切含まれていない（独立した未検証要素）。

---

## 5. 動的ユニバース（dyn_rsr42_bear_rs0）

- 本番: `strategy.yaml:101-117`で有効。`src/strategy/universe.py`実装。
- 研究: `research_state.md`内の別系列（Project Dynamic Universe, 2026-04-05採用、`wf_dyn_rsr42.py`でWF5/5 PASS）で検証済み。この検証はStudy9-29系列とは別の独立した研究トラックであり、Study9のCase B構築時には動的ユニバースの活性フィルターも適用されていた可能性が高いが、`study9_standalone_validation.py`が`src/strategy/universe.py`をimportしているかどうかは本書執筆時点で個別確認が必要（未確認事項として記録）。
- 結論: 動的ユニバース自体は本番と整合する形で検証済みだが、Study9-29のEntry/Exit本体とは別軸の検証。

---

## 6. 現在TRUE/FALSEのオーバーレイフラグと本番への影響

`strategy.yaml`で現在 **true** になっており、Study9-29のいずれの検証にも含まれていない本番固有の機構:

| フラグ | 値 | 影響 |
|---|---|---|
| `entry_timing.enabled` | true | composite rankingにブースト加算（§4） |
| `dynamic_universe.enabled` | true | post-RSRフィルター（別系列で検証済み、§5） |
| `risk_controls.sector_concentration.enabled` | true | 同一セクター同時1銘柄まで（Study9-29では言及なし） |
| `risk_controls.gross_exposure_enabled` | true（configのみ） | **本番未実装（DEAD）**。実害なし |
| `bear_universe_filter.enabled` | true | Bear時セクター除外（Study9-29では単一銘柄スロットのため影響度不明） |

現在 **false** で安全に不活性（Study9-29との整合性に影響しない）:

| フラグ | 値 |
|---|---|
| `entry_timing.block_low_confidence` | false |
| `entry_timing.auto_apply_boost` | false |
| `position_sizing.auto_apply` | false |
| `predictive_expansion.predictive_entry_enabled` | false |
| `bear_dynamic_filter.enabled` | false |

---

## 7. `src/live/*.py`（実行時インフラ）と研究コードの対応

| モジュール | 研究コードに対応物あり? |
|---|---|
| `capital_deployment_os.py`（dynamic_max_positions） | 一部（`src/backtest/capital_allocation_abc.py`等に概念的に類似するアロケーションロジックはあるが、PARAMS_LOCKEDクランプ自体は研究側に存在しない） |
| `client_order_id.py`, `inflight_registry.py` | なし（純粋な実行時冪等性インフラ） |
| `position_sync.py`, `reconciliation_engine.py`, `broker_truth_snapshot.py` | なし |
| `execution_integrity_validator.py`, `replay_consistency_validator.py` | なし |
| `runtime_integration.py`, `staged_supervisor.py`, `process_supervisor.py` | なし |
| `predictive_candidate_ranker.py`, `future_leader_screener.py` | 部分的（`src/backtest/`にfuture leader系の研究スクリプトが別途存在するが、Study9-29とは無関係の別研究トラック） |

これらは「シグナル生成ロジック」ではなく「実行管理・整合性検証」層であるため、Study9-29のCalmar/CAGR評価対象には本質的に含まれない（含まれるべきでもない）。ただしこれらの層が誤動作した場合、シグナル自体が正しくても実発注が阻害/重複/遅延するリスクがあり、別軸でのリリース前検証が必要（バックテストでは検証不可能な領域）。

---

## 8. 優先度付きサマリー

| 重大度 | 項目 | 内容 |
|---|---|---|
| 🔴 CRITICAL | Entry/Exitロジックの不一致 | Study9-29の全結論が本番未実装の仮想戦略を評価している |
| 🟡 MEDIUM | Entry Timing Engine未検証 | LIVE-ACTIVEだがStudy9-29評価対象外 |
| 🟡 MEDIUM | `generate_strategy_spec.py`のバグ | RSR Exit閾値ラベルが`min_rsr`(75)を誤表示（実際は`rsr_exit`=70） |
| 🟢 LOW | Gross Exposure Control | config上はtrueだが本番未実装のため実害なし（ただし「有効」という誤った認識を生むリスク） |
| 🟢 LOW | 動的ユニバースは別系列で検証済み | Study9-29と無関係だが本番との整合性自体は問題なし |
