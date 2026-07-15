# Study76 実行チェックリスト

**作成日**: 2026-07-04
**性格**: Study75完了後、Study76を「誰が実行しても同じ結果になる」ために辿る手順チェックリスト。本書自体はチェックリストの提示のみ・実行はしない。
**参照**: `study76_execution_plan.md`（目的・差分・基準）/ `study76_dependency_matrix.md`（依存関係）/ roadmap §0.6（Study実行標準11手順）。

---

## Phase 0: 前提解除確認（着手前・全て✅必須）

- [ ] J-Quants APIプラン契約完了（ユーザー実施・上場廃止込み株価取得可能なプランであることを確認済み）
- [ ] `src/.env` に `JQUANTS_MAIL_ADDRESS` / `JQUANTS_API_PASSWORD` 設定済み
- [ ] Study75完了（`study75_report.md` / `universe_diff.json` / `architecture_impact.md` 提出済み）
- [ ] Study75の「Core Closureを変更する必要があるか」判定がNOであること（YESの場合はStudy76着手前にCore Architecture側の再判定が必要 — 本チェックリストの前提が崩れる）
- [ ] Study75規則ユニバースのバイアス実測値が確定している（成功=バイアス≤-1.5pp∧Baseline CAGR≥9% / 失敗=バイアス>-3pp、いずれの場合もStudy76起案は可能だが解釈に影響）

## Phase 1: 仕様確認（ASK_FIRST・実行前必須）

- [ ] **Universe統制ポリシー確認（2026-07-04ユーザー決裁済み・再検討ではなく実行着手の確認ゲート）**: D_ATR_EQを含む全比較対象をStudy75新Universe上でfresh run再測定すること・旧Universe値との比較は禁止であることをASK_FIRSTで再確認してから着手
- [ ] D_ATR_EQのStudy75 Universe上での再測定スクリプト新規作成についてASK_FIRST承認取得
- [ ] Study76純正構成の新規スクリプト作成についてASK_FIRST承認取得（`src/backtest/study76_clenow_benchmark_wf.py`相当）
- [ ] パラメータ表（<10個）を事前固定し、ユーザーに提示（起案書内で明記 — roadmap §0.6手順②に相当）
- [ ] 成功条件（ΔCAGR≥-2pp）・失敗条件（ΔCAGR<-4pp）・未定義域（-4pp〜-2pp）の扱いをユーザーに再確認（正典に判定手順なしのため）

## Phase 2: 実装（ASK_FIRST承認後）

- [ ] 新規スクリプト作成: `src/backtest/study76_clenow_benchmark_wf.py`（既存`study73_*.py`構造踏襲）
- [ ] `composite_alpha_bt.py`をライブラリとして使用（shift(1)/翌日寄付/コスト計算を継承）。ただしStudy76最簡構成用に以下を全てOFF/バイパスするランナー分岐を実装:
  - [ ] `dynamic_universe.enabled` → 不使用（規則ユニバース固定リストに置換）
  - [ ] `bear_universe_filter.enabled` → 不使用
  - [ ] `risk_controls.gross_exposure_enabled` → 不使用
  - [ ] `risk_controls.shock_exit_mode` → 不使用
  - [ ] `fraction.bull` / `fraction.bear` → 不使用（発注比率制御なし）
  - [ ] Exit判定を「ランク脱落回転のみ」に単純化（RSR_EXIT/ATR_TRAIL/ATR_TRAILING/RUNNER_TRAIL_EXIT/TIME_STOP/TURTLE_EXIT/FIXED_PCT_TRAILING/MARKET_SHOCK_EXIT系列を無効化）
  - [ ] Entryを`(slope90d × R²)`ランクのみに単純化（RSR42/composite alpha/entry_timing boost無効化）
  - [ ] サイジングを均等ウェイト固定（eq_scale_addon/position_sizing/adaptive_growth無効化）
  - [ ] 週次リバランス処理（日次シグナルではなく週次判定サイクル）を実装
  - [ ] Turtleブレイクアウト・トリガー有無の2アーム切替フラグを実装
- [ ] `src/backtest/archive/portfolio_clenow.py` / `clenow_momentum.py`のロジックを参照実装として確認（そのまま流用不可・Study75ユニバース対応が必要な点に注意）
- [ ] コスト前提（slippage=0.001/commission=0.00055）・capital=¥3,000,000がPARAMS_LOCKED通りであることをgrep確認

## Phase 3: 実行（fresh run必須）

- [ ] IS 2018-2024（またはStudy75データ利用可能期間）fresh run × 2アーム（純正構成）
- [ ] OOS 2025 fresh run × 2アーム（純正構成）
- [ ] WF5fold fresh run × 2アーム（純正構成）
- [ ] **D_ATR_EQ比較値をStudy75新Universe上でfresh run再測定**（同一期間・同一コスト前提。旧RSR42ベースの公式値は使用禁止 — Universe統制ポリシー）
- [ ] Bootstrap（N=500・seed=42・Study73/78方式踏襲）— 純正構成・D_ATR_EQ再測定値の両方
- [ ] mom_period=21相当パラメータ（Study76側の対応パラメータ）の感度チェック（OQ3申し送り — スイープではなく単発確認）

## Phase 4: 判定・報告

- [ ] ΔCAGR算出 → 0.4ゲート相当（成功/失敗/未定義域）で機械判定
- [ ] WF5/5・2022非悪化の確認（Production採用条件の一部）
- [ ] 未定義域の場合はユーザー決裁を仰ぐ（裁量判断禁止）
- [ ] 結果を`reports/study76_clenow_benchmark_wf.md`として提出
- [ ] `backtests/study76_clenow_benchmark_wf_YYYY-MM-DD.json`として結果保存
- [ ] `research_state.md`先頭セクション追記
- [ ] `docs/research/YYYY-MM-DD.md`日次ログ追記
- [ ] メモリindex更新
- [ ] git commit（`research update: YYYY-MM-DD`。pushはASK_FIRST）

## Phase 5: Study77への引き継ぎ

- [ ] Study76成功時: 「勝者構成（純正 or Turtleトリガー有無いずれか）」を確定しStudy77起案書のインプットとして`study76_execution_plan.md`同様の計画書を新規作成
- [ ] Study76失敗時: Study77は正典定義のままでは起案不可。「対象をD_ATR_EQに差し替えるか」「Exit領域研究自体を終了するか」をユーザー決裁
- [ ] 未定義域時: Study77着手判断もユーザー決裁待ちとして保留

---

## 禁止事項（本チェックリスト全フェーズ共通）

- パラメータ追加・アーム追加・閾値スイープ（恒久閉鎖14項#9「検証独立前のハイブリッド化」等に抵触するため）
- Study75未完了段階でのfresh run実行（本チェックリストPhase 2以降は全てBlock）
- キャッシュ値・過去JSON流用によるProduction判定（fresh_run_required=true、Study52汚染事件の再発防止）
- 「惜しい」「もう少しで」等の裁量的解釈によるゲート判定の緩和
