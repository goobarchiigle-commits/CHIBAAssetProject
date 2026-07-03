# CLAUDE.md — CHIBAAssetProject

# MODE
GENSHIJIN+CAVEMAN
MINTOK=ON
PRECISION=MAX
AMBIG=OFF

# ROLE
CLD=design,logic,RCA,risk,hypothesis
CDX=impl,test,patch,batch,diff

# STYLE
noun-phrase
no filler
no honorific
no prose
jp-short
exact code/path/var preserve
NO_FLUFF=true

---

# BOOT
READ_ORDER_STRICT:
1. src/research_state.md        ← Single Source of Truth
2. src/configs/strategy.yaml
3. backtests/backtest_summary.json
4. docs/research/latest
FAIL_ON_MISSING=true
NO_MEMORY_TRUST=true           ← 会話履歴は信用しない。必ずファイルから復元

# PROJECT
phase=2
cwd=C:/ai-trading
remote=git@github.com:goobarchiigle-commits/CHIBAAssetProject.git
branch=main
account=auカブコム証券（特定口座）
api=kabuステーション REST API localhost:18080
env_file=src/.env              ← AI_TRADING_HOME / LIVE_UNIVERSE_FILE
data_gitignore=true            ← data/ 以下は絶対コミット禁止

---

# PARAMS_LOCKED
turtle_exit=55d
min_hold=3d
min_rsr=75.0
max_positions=3
capital=3_000_000
slippage=0.001
commission=0.00055
ASK_FIRST_ON_CHANGE=true      ← 上記すべて変更前に必ずユーザー確認

# CIRCUIT                      ← コード組み込み済み・変更禁止
max_dd_limit=0.15              → BUY_STOP（自動撤退はしない・警告のみ）
max_single_weight=0.25
max_positions=3

---

# GUARD_CRITICAL               ← 研究Bias詳細は /backtest-research を参照
lookahead=forbid
data_leak=forbid
auth_leak=forbid
file_rule=mandatory
permission_rule=mandatory
morning_routine=mandatory
reporting=mandatory
env_only=secret

---

# PERMISSION
AUTO_OK:
- run_live_signal.py
- run_live_signal.py --live --yes
- run_morning_signal.py
- backtest読み取り実行

ASK_FIRST:
- PARAMS_LOCKED 内の任意パラメータ変更
- 銘柄ユニバース（RSR42）追加・削除
- signal_bridge.py 発注ロジック変更
- git push / GitHub送信
- 新規スクリプト作成・既存スクリプト大規模改修

# TRADE
dup_order=block
over_order=block
same_symbol_same_side=block
same_day_duplicate=block
state_sync=exact
position_reconcile_before_live=true
sl_logic=exact
dd_warn=-0.15
exchange_rule=comply
rate_limit=5/s
cooldown=60s
idempotency=required
client_order_id=required
retry_max=1
slippage_required=true        ← 注文ロジックに必ず含める（0.1%）
commission_required=true      ← 注文ロジックに必ず含める（0.055%）

---

# MORNING_ROUTINE              ← 詳細手順は /live-signal を参照
1 api_port_check=18080
2 dry_run=required
3 signal_summary_jp=required
4 live_exec=allowed
5 execution_log_report=required
6 dd_monitor=required
FAIL_IF_SKIP=true
REPORT_ALWAYS=true

---

# SAVE
UPDATE_ATOMIC:
- src/research_state.md       ← 研究状態更新
- backtests/*.json            ← 分析結果JSON
- docs/research/YYYY-MM-DD.md ← 日次研究ログ
REQUIRE_COMMIT=true
COMMIT_MSG="research update: YYYY-MM-DD"

FILE_MAP:
research_state  → src/research_state.md          (Markdown)
backtest_result → backtests/                      (JSON)
strategy_param  → src/configs/strategy.yaml       (YAML)
universe        → src/configs/universe.yaml        (YAML)
daily_log       → docs/research/YYYY-MM-DD.md     (Markdown)
exec_log        → logs/research/                   (.log)
portfolio       → runtime/portfolio_state.json     (JSON)

---

# VALIDATION                   ← 詳細基準は /backtest-research を参照
sharpe_max=3.0
dd_max=0.5
trade_min=5
new_strategy_dir=backtest/

# EXECUTION
cwd=C:/ai-trading
run_required=true            ← 「動きそう」禁止。「動いた」を提出
no_assumption=true
no_hardcode_path=true        ← パス直書き禁止。from paths import RESULTS_DIR 等を使う

# ENV_WIN
encoding=sys.stdout.reconfigure(encoding='utf-8')  ← 日本語print冒頭必須
matplotlib_jp=rcParams['font.family']='MS Gothic'
path_sep=forward_slash_only
run_from=C:/ai-trading

---

# REVIEW
CDX_REVIEW=feasible,consistent,edge,test,regression,safe

# OUTPUT
FMT=concl,fatal,top3step,codex_task,risk_test

# SHORTHAND
DR=dry run
LV=live
PS=position sync
ATRX=entry atr cache
SL=stop loss
TP=take profit
RCA=root cause analysis
REG=regression
PATCH=min diff
SAFE=order safety

# TASK_DEFAULT
direct answer
3-step priority
codex-ready
min token

---

# VERSION
claude_md_version=2026-04-16
project_phase=2

# EMERGENCY_STOP
api_unreachable=abort
position_sync_fail=abort
duplicate_order_detected=abort
portfolio_state_missing=abort
ABORT_ACTION=stop_execution_and_report_user

# OVERFIT_GUARD                ← 詳細プロトコルは /backtest-research を参照
oos_is_ratio_min=0.7
walkforward_required=true
param_sweep_limit=bounded
single_metric_optimization=forbid
min_trade_required=5
stability_check=required
fresh_run_required=true        ← Production判定にキャッシュ値使用禁止（Study52汚染事件再発防止 / 2026-07 M3）

---

# Autonomous Runtime Rules

Infrastructure is NOT complete unless integrated into the live execution path.
Modules, tests, logs, or manual workflows alone are insufficient.
Normal runtime operation must not require manual: recovery, reconciliation, restart, lock cleanup, state repair, deployment validation.
Broker reality is authoritative.
Execution must fail-closed on: reconciliation failure, stale market data, runtime integrity violations, duplicate runtime detection.
Telemetry and analytics may fail-open only if execution integrity remains preserved.
All runtime decisions and state transitions must be: deterministic, replay-compatible, append-only logged, reproducible across restart/recovery.
