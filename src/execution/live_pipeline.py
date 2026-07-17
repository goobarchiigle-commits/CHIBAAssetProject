"""
live_pipeline.py — deterministic live trading pipeline with execution + risk control.

Pipeline order (strict, no deviation per strategy):
    1. validate_and_decompose
    2. detect_strategy_fragility
    3. evaluate_strategy_allocation
    4. check_kill_conditions
    5. generate_orders
    6. execute_orders
    7. log_all

Persisted state schema
----------------------
positions       : dict[str, int]  — current holdings (qty per symbol)
last_decision   : str             — decision from previous run
cumulative_pnl  : float           — cumulative sum of net daily returns (fraction)
cumulative_cost : float           — total realized cost paid (positive, fraction)
hwm_pnl         : float           — high-water mark of cumulative_pnl  [default 0.0]
max_drawdown    : float           — worst (cumulative_pnl - hwm_pnl)   [default 0.0]
sent_order_keys : list[str]       — idempotency record: "YYYY-MM-DD:symbol"

All new keys use .get(key, default) for backward compatibility.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
import requests

sys.stdout.reconfigure(encoding="utf-8")

from src.diagnostics.detect_strategy_fragility import detect_strategy_fragility
from src.diagnostics.evaluate_strategy_allocation import (
    EPS,
    evaluate_strategy_allocation,
)
from src.strategy.score import compute_score, enrich_with_z

# ── Project constants (CLAUDE.md PARAMS_LOCKED — do not change without user confirmation) ──
CAPITAL: float        = 3_000_000.0
MAX_POSITIONS: int    = 3
SLIPPAGE: float       = 0.001       # 0.1%
COMMISSION: float     = 0.00055     # 0.055%
DD_LIMIT: float       = -0.15       # circuit breaker threshold
COST_PER_TRADE: float = SLIPPAGE + COMMISSION

KABU_API_BASE: str = "http://localhost:18080"
RETRY_MAX: int     = 3

_EXEC_LOG_COLS = [
    "date", "symbol", "side", "qty",
    "expected_price", "filled_price", "slippage",
    "decision", "kill_state",
]
_PIPE_LOG_COLS = [
    "date", "strategy_id",
    "mean_net", "stability_score", "top3_share", "efficiency",
    "decision", "kill_state",
]

# decision → base position multiplier
_DECISION_MULT: dict[str, float] = {
    "ALLOCATE":         1.0,
    "ALLOCATE_LIMITED": 0.5,
    "DISCARD":          0.0,
}

# kill state → additional multiplier applied on top of decision multiplier
_KILL_MULT: dict[str, float] = {
    "STOP":       0.0,
    "REDUCE":     0.5,
    "REALLOCATE": 1.0,
    "KEEP":       1.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. Validator / Decomposer
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_decompose(
    df: pd.DataFrame,
    min_cs: int,
    cost_per_trade: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Open-based return decomposition. Only place where returns are computed.

    Parameters
    ----------
    df            : DataFrame with columns ['date','symbol','open','position']
    min_cs        : minimum active symbols per date (dates below threshold dropped)
    cost_per_trade: combined slippage + commission rate (fraction)

    Returns
    -------
    (daily_decomp, symbol_df, daily_ret, daily_ret_net)

    Raises
    ------
    ValueError : missing columns or fewer than 2 dates available
    """
    required = {"date", "symbol", "open", "position"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in strategy df: {missing}")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    # ── Pivot: date × symbol ──────────────────────────────────────────────────
    open_w = df.pivot(index="date", columns="symbol", values="open").sort_index()
    pos_w  = (
        df.pivot(index="date", columns="symbol", values="position")
        .sort_index()
        .fillna(0.0)
    )

    # Drop symbols where all open prices are NaN
    open_w = open_w.dropna(axis=1, how="all")
    pos_w  = pos_w.reindex(columns=open_w.columns).fillna(0.0)

    if len(open_w) < 2:
        raise ValueError("Insufficient data: need ≥ 2 dates for return computation")

    # ── Forward returns (open-based): ret[t] = (open[t+1] - open[t]) / open[t] ──
    # Position at t earns return from open[t] → open[t+1].
    # Dividing by open[t]=0 → NaN (no return if price missing).
    fwd_open  = open_w.shift(-1)
    safe_open = open_w.replace(0.0, float("nan"))
    raw_ret   = (fwd_open - open_w) / safe_open

    contrib  = raw_ret * pos_w   # per-symbol contribution
    contrib  = contrib.iloc[:-1] # drop last row (no forward open)
    pos_lag  = pos_w.iloc[:-1]   # position that earned the return at each date

    # ── min_cs filter ─────────────────────────────────────────────────────────
    active_cs  = (pos_lag.abs() > 0).sum(axis=1)
    valid_mask = active_cs >= min_cs
    contrib    = contrib[valid_mask]
    pos_lag    = pos_lag[valid_mask]

    # ── Long / short decomposition ────────────────────────────────────────────
    long_c  = contrib.clip(lower=0).sum(axis=1)
    short_c = contrib.clip(upper=0).sum(axis=1)
    gross   = contrib.sum(axis=1)

    # ── Trade cost: per-symbol position change via groupby diff ─────────────
    df["turnover_i"] = (
        df.groupby("symbol")["position"]
          .diff()
          .abs()
    )
    df["turnover_i"] = df["turnover_i"].where(
        df.groupby("symbol").cumcount() > 0,
        0.0
    )

    n_trades = (
        df.groupby("date")["turnover_i"]
        .apply(lambda x: (x > 0).sum())
        .reindex(contrib.index)
        .fillna(0.0)
    )
    cost = -(n_trades * cost_per_trade)
    net  = gross + cost

    daily_decomp = pd.DataFrame({
        "long_contribution":  long_c,
        "short_contribution": short_c,
        "gross":              gross,
        "cost":               cost,
        "net":                net,
    })

    # ── symbol_df ─────────────────────────────────────────────────────────────
    cum_pnl = contrib.sum(axis=0)
    avg_pos = pos_lag.mean(axis=0)

    valid_dates = set(contrib.index)
    sym_turn = (
        df[df["date"].isin(valid_dates)]
        .groupby("symbol")["turnover_i"]
        .sum()
        .reindex(cum_pnl.index)
        .fillna(0.0)
    )
    total_turn = float(sym_turn.sum())
    turn_contrib = sym_turn / max(abs(total_turn), EPS)

    symbol_df = pd.DataFrame({
        "cumulative_pnl":        cum_pnl,
        "avg_position":          avg_pos,
        "turnover_contribution": turn_contrib,
    })

    daily_ret     = gross.rename("daily_ret")
    daily_ret_net = net.rename("daily_ret_net")

    return daily_decomp, symbol_df, daily_ret, daily_ret_net


# ─────────────────────────────────────────────────────────────────────────────
# 2. Kill switch
# ─────────────────────────────────────────────────────────────────────────────

def check_kill_conditions(state: dict[str, Any]) -> str:
    """
    Evaluate kill conditions against the augmented state.

    Expected keys in state
    ----------------------
    max_drawdown    : float  — worst (cumulative_pnl - hwm_pnl); 0.0 or negative
    realized_cost   : float  — total cost paid so far (positive fraction)
    expected_cost   : float  — expected total cost from strategy history (positive fraction)
    decision_changed: bool   — True if allocation decision changed vs last run

    Returns
    -------
    "STOP" | "REDUCE" | "REALLOCATE" | "KEEP"
    """
    # DD_LIMIT * 1.5 = -0.225 → triggers at 22.5% drawdown from peak
    if state["max_drawdown"] < DD_LIMIT * 1.5:
        return "STOP"

    if state["realized_cost"] > state["expected_cost"] * 2:
        return "REDUCE"

    if state["decision_changed"]:
        return "REALLOCATE"

    return "KEEP"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Order generation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_full_target_sizes(
    strategy_df: pd.DataFrame,
    market_data: pd.DataFrame,
) -> dict[str, int]:
    """
    Floor allocation per symbol: floor(CAPITAL / MAX_POSITIONS / open_price).
    Only symbols with non-zero position on the latest signal date are included.
    """
    per_pos_capital = CAPITAL / MAX_POSITIONS

    # Enrich full history to compute cross-sectional z / z_regime,
    # then filter to the latest date for scoring and selection.
    enriched       = enrich_with_z(strategy_df)
    latest_date    = enriched["date"].max()
    latest_signals = enriched[enriched["date"] == latest_date].copy()

    # Score-based ranking: positive edge only, top MAX_POSITIONS.
    # Falls back to position-based selection when all scores ≤ 0.
    latest_signals = compute_score(latest_signals)
    positive = latest_signals[latest_signals["score"] > 0]
    if not positive.empty:
        latest_signals = positive.nlargest(MAX_POSITIONS, "score")

    market_open: dict[str, float] = (
        market_data.set_index("symbol")["open"].to_dict()
        if "symbol" in market_data.columns
        else {}
    )

    result: dict[str, int] = {}
    for _, row in latest_signals.iterrows():
        sym = row["symbol"]
        if float(row.get("position", 0)) != 0.0:
            price = float(market_open.get(sym, 0))
            if price > 0:
                result[sym] = int(per_pos_capital // price)

    return result


def generate_orders(
    allocation_result: dict[str, Any],
    kill_state:        str,
    full_target_sizes: dict[str, int],
    current_positions: dict[str, int],
) -> list[dict[str, Any]]:
    """
    Deterministic rebalance orders.

    target = floor(full_size * decision_mult * kill_mult)
    rebalance = target - current

    No partial logic, no optimization, no VWAP, no slicing.
    Symbols sorted for deterministic output order.

    Entry Freeze Mode（資産保全・2026-07-17）: cfg.entry_freeze.enabled=True の間、
    新規BUY方向のrebalance（現在ゼロ保有からの買い、または買い増し）は生成しない。
    保有縮小・手仕舞い方向（SELL）は無停止（signal_bridge.py::_build_ordersと同じ設計）。
    """
    from src.config_loader import load_strategy_config
    entry_frozen = False
    freeze_reason = ""
    try:
        _cfg = load_strategy_config()
        entry_frozen  = bool(_cfg.entry_freeze.enabled)
        freeze_reason = _cfg.entry_freeze.reason
    except Exception:
        pass  # 設定読込失敗時はfreeze適用なし（既存挙動を維持・fail-open）

    decision  = allocation_result["decision"]
    base_mult = _DECISION_MULT[decision]
    kill_mult = _KILL_MULT[kill_state]
    mult      = base_mult * kill_mult

    orders: list[dict[str, Any]] = []
    all_symbols = sorted(set(full_target_sizes) | set(current_positions))

    for symbol in all_symbols:
        full_size = full_target_sizes.get(symbol, 0)
        target    = int(full_size * mult)        # truncate toward zero
        current   = current_positions.get(symbol, 0)
        rebalance = target - current

        if rebalance == 0:
            continue

        side = "BUY" if rebalance > 0 else "SELL"
        if entry_frozen and side == "BUY":
            print(f"ENTRY_FROZEN: symbol={symbol} reason={freeze_reason}")
            continue

        orders.append({
            "symbol":  symbol,
            "side":    side,
            "qty":     abs(rebalance),
            "target":  target,
        })

    return orders


# ─────────────────────────────────────────────────────────────────────────────
# 4. Execution (kabuステーション REST API)
# ─────────────────────────────────────────────────────────────────────────────

def execute_orders(
    orders:       list[dict[str, Any]],
    date:         str,
    market_data:  pd.DataFrame,
    sent_keys:    set[str],
    decision:     str,
    kill_state:   str,
    strategy_id:  str,
) -> list[dict[str, Any]]:
    """
    POST orders to kabuステーション API.

    - Idempotency key: SHA256("{strategy_id}:{date}:{symbol}:{target_qty}:{current_qty}")
      checked BEFORE API call; prevents duplicates across retries and reruns
    - Retry: up to RETRY_MAX=3 with exponential backoff (1 s, 2 s, 4 s)
    - Failure after retries → raises RuntimeError (caller aborts strategy)
    - Partial fill → logged with actual fill (Option A) or NaN (Option B); no reattempt

    FrontOrderType=120 (成行), Price=0, ExecuteTiming=next open assumed.
    """
    api_key = os.environ["KABU_API_KEY"]
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}

    market_open: dict[str, float] = (
        market_data.set_index("symbol")["open"].to_dict()
        if "symbol" in market_data.columns
        else {}
    )

    executed: list[dict[str, Any]] = []

    for order in orders:
        symbol      = order["symbol"]
        target_qty  = order["target"]
        order_qty   = order["qty"]
        current_qty = target_qty - (order_qty if order["side"] == "BUY" else -order_qty)

        # Strong idempotency: SHA256 of full order identity
        idem_payload = f"{strategy_id}:{date}:{symbol}:{target_qty}:{current_qty}"
        idem_key     = hashlib.sha256(idem_payload.encode()).hexdigest()

        if idem_key in sent_keys:
            continue  # duplicate prevention before API call

        payload = {
            "Symbol":             symbol,
            "Exchange":           1,       # 東証
            "SecurityType":       1,       # 株式
            "Side":               "2" if order["side"] == "BUY" else "1",
            "CashMargin":         1,       # 現物
            "MarginTradeType":    0,
            "DelivType":          0,
            "FundType":           "  ",
            "AccountType":        4,       # 特定口座
            "Qty":                order_qty,
            "ClosePositionOrder": 0,
            "Price":              0,       # 成行
            "ExpireDay":          0,       # 当日
            "FrontOrderType":     120,     # 成行
        }

        last_exc: Exception | None = None
        success = False

        for attempt in range(RETRY_MAX):
            try:
                resp = requests.post(
                    f"{KABU_API_BASE}/kabusapi/sendorder",
                    json=payload,
                    headers=headers,
                    timeout=10,
                )
                resp.raise_for_status()
                result   = resp.json()
                order_id = result.get("OrderId", "")

                expected_price = float(market_open.get(symbol, 0))

                # Option A: fetch actual fill from kabuステーション API
                filled_price: float = float("nan")
                slippage_val: float = float("nan")
                try:
                    fill_resp = requests.get(
                        f"{KABU_API_BASE}/kabusapi/orders",
                        params={"product": 0, "id": order_id},
                        headers=headers,
                        timeout=5,
                    )
                    fill_resp.raise_for_status()
                    fill_data = fill_resp.json()
                    if isinstance(fill_data, list) and fill_data:
                        raw_price = fill_data[0].get("Price", 0)
                        if raw_price and float(raw_price) > 0:
                            filled_price = float(raw_price)
                            slippage_val = filled_price - expected_price
                except Exception:
                    pass  # Option B fallback: filled_price and slippage remain NaN

                executed.append({
                    "date":           date,
                    "symbol":         symbol,
                    "side":           order["side"],
                    "qty":            order_qty,
                    "expected_price": expected_price,
                    "filled_price":   filled_price,
                    "slippage":       slippage_val,
                    "decision":       decision,
                    "kill_state":     kill_state,
                    "order_id":       order_id,
                    "result_code":    result.get("Result", -1),
                })

                sent_keys.add(idem_key)
                success = True
                break

            except requests.exceptions.RequestException as exc:
                last_exc = exc
                if attempt < RETRY_MAX - 1:
                    time.sleep(2 ** attempt)   # 1 s → 2 s → 4 s

        if not success:
            raise RuntimeError(
                f"Order failed after {RETRY_MAX} attempts: {symbol}"
            ) from last_exc

    return executed


# ─────────────────────────────────────────────────────────────────────────────
# 5. Logging (append-only CSV)
# ─────────────────────────────────────────────────────────────────────────────

def log_all(
    date:              str,
    strategy_id:       str,
    executed_orders:   list[dict[str, Any]],
    decision:          str,
    kill_state:        str,
    metrics:           dict[str, Any],
    exec_log_path:     Path,
    pipeline_log_path: Path,
) -> None:
    """
    Append-only writes to execution_log.csv and pipeline_log.csv.
    Creates file with header on first write; never overwrites existing rows.
    """
    exec_log_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_log_path.parent.mkdir(parents=True, exist_ok=True)

    # ── execution_log ─────────────────────────────────────────────────────────
    if executed_orders:
        exec_rows = [
            {k: r.get(k, "") for k in _EXEC_LOG_COLS}
            for r in executed_orders
        ]
        exec_df = pd.DataFrame(exec_rows, columns=_EXEC_LOG_COLS)
        write_header = not exec_log_path.exists()
        exec_df.to_csv(exec_log_path, mode="a", header=write_header, index=False)

    # ── pipeline_log ──────────────────────────────────────────────────────────
    pipe_row = pd.DataFrame(
        [{
            "date":            date,
            "strategy_id":     strategy_id,
            "mean_net":        metrics.get("mean_net",        float("nan")),
            "stability_score": metrics.get("stability_score", float("nan")),
            "top3_share":      metrics.get("top3_share",      float("nan")),
            "efficiency":      metrics.get("efficiency",      float("nan")),
            "decision":        decision,
            "kill_state":      kill_state,
        }],
        columns=_PIPE_LOG_COLS,
    )
    write_header = not pipeline_log_path.exists()
    pipe_row.to_csv(pipeline_log_path, mode="a", header=write_header, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Structured event logger
# ─────────────────────────────────────────────────────────────────────────────

_EVENTS_LOG = Path("logs/live_events.jsonl")


def log_event(event: str, payload: dict[str, Any]) -> None:
    _EVENTS_LOG.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts":      datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "event":   event,
        "payload": payload,
    }
    with _EVENTS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_live_pipeline(
    market_data:       pd.DataFrame,
    strategies:        list[dict[str, Any]],
    state:             dict[str, Any],
    date:              str,
    *,
    min_cs:            int   = 1,
    cost_per_trade:    float = COST_PER_TRADE,
    exec_log_path:     Path  = Path("logs/execution_log.csv"),
    pipeline_log_path: Path  = Path("logs/pipeline_log.csv"),
) -> dict[str, Any]:
    """
    Execute the full live trading pipeline for all strategies.

    Parameters
    ----------
    market_data       : DataFrame with columns ['symbol','open',...] — today's OHLC
    strategies        : list of dicts, each:
                        {
                          "strategy_id": str,
                          "df":          DataFrame['date','symbol','open','position'],
                          "min_cs":      int   (optional, overrides default),
                          "cost_per_trade": float (optional, overrides default),
                        }
    state             : persisted state dict (mutated in-place and returned)
    date              : ISO date string for today (e.g. "2026-05-04")
    min_cs            : default minimum cross-section (overridable per strategy)
    cost_per_trade    : default cost rate (overridable per strategy)
    exec_log_path     : append-only execution log CSV path
    pipeline_log_path : append-only pipeline log CSV path

    Returns
    -------
    {
        "executed_orders":   list[dict]          — all executed orders across strategies
        "updated_state":     dict                — mutated state (same object as input)
        "strategy_results":  dict[str, dict]     — per-strategy status and metadata
    }
    """
    print("[DEBUG] run_live_pipeline entered")
    log_event("pipeline_start", {"n_strategies": len(strategies)})
    # ── State defaults for new fields ─────────────────────────────────────────
    state.setdefault("positions",       {})
    state.setdefault("last_decision",   "")
    state.setdefault("cumulative_pnl",  0.0)
    state.setdefault("cumulative_cost", 0.0)
    state.setdefault("hwm_pnl",         0.0)
    state.setdefault("max_drawdown",    0.0)
    state.setdefault("sent_order_keys", [])

    sent_keys: set[str] = set(state["sent_order_keys"])

    all_executed:     list[dict[str, Any]]       = []
    strategy_results: dict[str, dict[str, Any]]  = {}

    for strat in strategies:
        log_event("strategy_enter", {"strategy_id": strat.get("strategy_id")})

        for _key in ("strategy_id", "df"):
            if _key not in strat:
                raise KeyError(f"strategy dict missing required key: {_key!r}")

        sid          = strat["strategy_id"]
        strat_df     = strat["df"]
        strat_min_cs = strat.get("min_cs",         min_cs)
        strat_cpt    = strat.get("cost_per_trade", cost_per_trade)

        # ── Step 1: validate_and_decompose ────────────────────────────────────
        try:
            daily_decomp, symbol_df, daily_ret, daily_ret_net = validate_and_decompose(
                strat_df, strat_min_cs, strat_cpt
            )
        except (ValueError, KeyError) as exc:
            traceback.print_exc()
            strategy_results[sid] = {"status": "SKIPPED", "reason": str(exc)}
            continue

        # ── Step 2: detect_strategy_fragility ─────────────────────────────────
        try:
            detect_strategy_fragility(
                daily_decomp, symbol_df, daily_ret, daily_ret_net
            )
        except Exception as exc:
            traceback.print_exc()
            strategy_results[sid] = {"status": "SKIPPED", "reason": f"fragility: {exc}"}
            continue

        # ── Step 3: evaluate_strategy_allocation ──────────────────────────────
        try:
            alloc = evaluate_strategy_allocation(
                daily_decomp, symbol_df, daily_ret
            )
        except Exception as exc:
            traceback.print_exc()
            strategy_results[sid] = {"status": "SKIPPED", "reason": f"allocation: {exc}"}
            continue

        decision = alloc["decision"]
        metrics  = alloc["metrics"]

        # ── Step 4: check_kill_conditions ─────────────────────────────────────
        current_pnl = float(state["cumulative_pnl"])
        hwm         = float(state["hwm_pnl"])
        hwm         = max(hwm, current_pnl)
        current_dd  = current_pnl - hwm
        max_dd      = min(float(state["max_drawdown"]), current_dd)

        expected_cost = abs(float(daily_decomp["cost"].sum()))
        realized_cost = float(state["cumulative_cost"])

        kill_state_input = {
            "max_drawdown":    max_dd,
            "realized_cost":   realized_cost,
            "expected_cost":   expected_cost,
            "decision_changed": decision != state["last_decision"],
        }
        kill_state = check_kill_conditions(kill_state_input)

        # ── Step 5: generate_orders ───────────────────────────────────────────
        full_target_sizes = _compute_full_target_sizes(strat_df, market_data)
        orders = generate_orders(
            alloc,
            kill_state,
            full_target_sizes,
            state["positions"],
        )

        log_event("orders_generated", {"strategy_id": strat.get("strategy_id"), "n_orders": len(orders)})
        log_event("orders_detail", {"orders": [
            {"symbol": o["symbol"], "side": o["side"], "qty": o["qty"], "score": o.get("score")}
            for o in orders[:5]
        ]})

        # ── Step 6: execute_orders ────────────────────────────────────────────
        log_event("before_execute", {"n_orders": len(orders)})
        if os.environ.get("DRY_RUN") == "1":
            log_event("dry_run_skip_execute", {"n_orders": len(orders)})
            executed = []
        else:
            try:
                executed = execute_orders(
                    orders, date, market_data, sent_keys, decision, kill_state, sid
                )
            except RuntimeError as exc:
                # API failure → abort this strategy's execution
                log_all(
                    date, sid, [], decision, kill_state, dict(metrics),
                    exec_log_path, pipeline_log_path,
                )
                strategy_results[sid] = {
                    "status":     "STOPPED",
                    "reason":     str(exc),
                    "kill_state": kill_state,
                    "decision":   decision,
                }
                continue

        all_executed.extend(executed)

        # ── Step 7: log_all ───────────────────────────────────────────────────
        log_all(
            date, sid, executed, decision, kill_state, dict(metrics),
            exec_log_path, pipeline_log_path,
        )

        # ── State update (positions, cost, decision, DD tracking) ─────────────
        for ord_rec in executed:
            sym  = ord_rec["symbol"]
            qty  = int(ord_rec["qty"])
            side = ord_rec["side"]
            state["positions"][sym] = (
                state["positions"].get(sym, 0)
                + (qty if side == "BUY" else -qty)
            )
            if state["positions"][sym] == 0:
                del state["positions"][sym]

        # Cost: commission + slippage on filled orders (fraction of capital)
        trade_cost_fraction = (
            sum(
                ord_rec["qty"] * ord_rec["expected_price"]
                for ord_rec in executed
                if ord_rec["expected_price"] > 0
            )
            * COST_PER_TRADE
            / max(CAPITAL, EPS)
        )
        state["cumulative_cost"] = realized_cost + trade_cost_fraction

        state["last_decision"] = decision
        state["hwm_pnl"]       = hwm
        state["max_drawdown"]  = max_dd

        strategy_results[sid] = {
            "status":     "EXECUTED",
            "orders":     executed,
            "kill_state": kill_state,
            "decision":   decision,
            "metrics":    dict(metrics),
        }

    # ── Persist idempotency keys ──────────────────────────────────────────────
    state["sent_order_keys"] = sorted(sent_keys)

    log_event("pipeline_end", {})
    return {
        "executed_orders":  all_executed,
        "updated_state":    state,
        "strategy_results": strategy_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7. CLI entrypoint
# ─────────────────────────────────────────────────────────────────────────────

def _load_default() -> pd.DataFrame:
    raise ValueError("No --input provided and no default data source configured")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",   type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    try:
        print("LIVE PIPELINE START")

        df = pd.read_csv(args.input) if args.input else _load_default()

        if "date" not in df.columns:
            raise ValueError("missing date column")
        df["date"] = pd.to_datetime(df["date"])

        if df["date"].nunique() < 2:
            past_df = pd.read_csv(Path("data/signals_history.csv"))
            if "date" not in past_df.columns:
                raise ValueError("missing date column in signals_history.csv")
            past_df["date"] = pd.to_datetime(past_df["date"])
            common_cols = [c for c in df.columns if c in past_df.columns]
            past_df = past_df[common_cols]
            df = df[common_cols]
            df = pd.concat([past_df, df], ignore_index=True)

        df = df.dropna(subset=["date", "symbol"])
        if df["date"].nunique() < 2:
            raise ValueError("still insufficient dates after merge")

        df = df.sort_values(["symbol", "date"])
        df = df.drop_duplicates(subset=["symbol", "date"], keep="last")

        market_data = df
        strategies = [{"strategy_id": "debug", "df": df}]
        state: dict[str, Any] = {}
        date = df["date"].max()

        os.environ["DRY_RUN"]   = "1" if args.dry_run else "0"
        os.environ["LOG_LEVEL"] = "DEBUG" if args.verbose else "INFO"

        print(f"Loaded rows: {len(df)}")

        run_live_pipeline(market_data, strategies, state, date)

    except Exception as e:
        traceback.print_exc()
        print("[FATAL]", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
