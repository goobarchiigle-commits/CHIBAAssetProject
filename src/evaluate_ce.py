"""evaluate_ce.py — CE compare evaluation engine (stateless-safe, production-grade)"""
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(os.environ.get("AI_TRADING_HOME", "C:/ai-trading"))
LOGS = BASE / "logs"
DAILY_CSV = LOGS / "ce_compare_daily.csv"
TRADES_JSONL = LOGS / "ce_trades.jsonl"
STATE_JSON = LOGS / "ce_state.json"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROMOTE_DAYS = 20
MIN_TRADES = 15
TSTAT_THRESH = 2.0
SIGN_MATCH_PROMOTE = 0.55
TRADE_SIGN_PROMOTE = 0.52
DD_LIMIT = -0.3
CUM_DELTA_KILL = 0.0
SIGN_MATCH_KILL = 0.5
EARLY_STOP_DAYS = 5
EARLY_STOP_DELTA = -0.05
RECENT_WINDOW = 5
RECENT_DELTA_MIN = 0.005
RECENT_SIGN_MIN = 0.52
AUTOCORR_WINDOW_SEC = 3600
AUTOCORR_RATIO_LIMIT = 0.5
RETURN_ANOMALY_THRESH = 0.2
DIVERSITY_THRESH = 0.3
RE_EVAL_DAYS = 60
STABILITY_DEGRADE = 0.3

# ---------------------------------------------------------------------------
# 1. State management
# ---------------------------------------------------------------------------

def _initial_state() -> dict:
    return {"finalized": False, "decision": None}


def load_state() -> dict:
    init = _initial_state()
    try:
        if STATE_JSON.exists():
            raw = STATE_JSON.read_text(encoding="utf-8")
            s = json.loads(raw)
            if isinstance(s, dict) and "finalized" in s:
                return s
    except (json.JSONDecodeError, ValueError):
        # Corrupted JSON — overwrite with initial state for natural recovery
        save_state(init)
    except Exception:
        pass
    return init


def save_state(state: dict) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=LOGS, suffix=".json.tmp", delete=False
        ) as tf:
            tmp_path = tf.name
            json.dump(state, tf, ensure_ascii=False, indent=2)
        os.replace(tmp_path, STATE_JSON)
    except Exception:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

# ---------------------------------------------------------------------------
# JSONL safe loader
# ---------------------------------------------------------------------------

def load_jsonl_safe(path: Path) -> pd.DataFrame:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass  # skip corrupted line
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

# ---------------------------------------------------------------------------
# Debug log
# ---------------------------------------------------------------------------

DEBUG_LOG = LOGS / "ce_debug.log"


def write_debug_log(result: dict) -> None:
    try:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "decision": result.get("decision"),
            "days": result.get("days"),
            "cum_delta": result.get("cum_delta"),
            "tstat": result.get("tstat"),
            "data_issues": result.get("data_issues", []),
        }
        LOGS.mkdir(parents=True, exist_ok=True)
        with open(DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 7. Data validation
# ---------------------------------------------------------------------------

def validate_trades(trades: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if trades.empty:
        issues.append("EMPTY_DATA")
        return issues
    if "forward_return" in trades.columns:
        mean_abs = trades["forward_return"].abs().mean()
        if pd.notna(mean_abs) and mean_abs > RETURN_ANOMALY_THRESH:
            issues.append("RETURN_ANOMALY")
        uniq = trades["forward_return"].nunique()
        total = len(trades)
        if total > 0 and (uniq / total) < DIVERSITY_THRESH:
            issues.append("LOW_DIVERSITY")
    if "timestamp" in trades.columns:
        trades.sort_values("timestamp", inplace=True)
    return issues

# ---------------------------------------------------------------------------
# 8. Autocorrelation check
# ---------------------------------------------------------------------------

def check_autocorr(trades: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    if trades.empty or "timestamp" not in trades.columns:
        return issues
    try:
        ts = pd.to_datetime(trades["timestamp"], errors="coerce").dropna().sort_values()
        if len(ts) < 2:
            return issues
        diffs = ts.diff().dt.total_seconds().dropna()
        close_ratio = (diffs < AUTOCORR_WINDOW_SEC).mean()
        if close_ratio > AUTOCORR_RATIO_LIMIT:
            issues.append("HIGH_AUTOCORR")
    except Exception:
        pass
    return issues

# ---------------------------------------------------------------------------
# 2. Significance (statistical)
# ---------------------------------------------------------------------------

def corr_tstat(corr: float, n: int) -> float:
    if n < 3:
        return 0.0
    corr_c = float(np.clip(corr, -0.9999, 0.9999))
    denom = max(1e-12, 1 - corr_c ** 2)
    t = corr_c * np.sqrt((n - 2) / denom)
    return float(np.clip(t, -100, 100))

# ---------------------------------------------------------------------------
# 3. Stability check
# ---------------------------------------------------------------------------

def stability_check_v2(daily: pd.DataFrame) -> bool:
    if "cum_delta" not in daily.columns or len(daily) < 4:
        return False
    try:
        half = len(daily) // 2
        first = daily["cum_delta"].iloc[:half]
        second = daily["cum_delta"].iloc[half:]
        first_gain = first.iloc[-1] - first.iloc[0]
        second_gain = second.iloc[-1] - second.iloc[0]
        if first_gain <= 0 or second_gain <= 0:
            return False
        if first_gain > 1e-9:
            degrade = (first_gain - second_gain) / abs(first_gain)
            if degrade > STABILITY_DEGRADE:
                return False
        return True
    except Exception:
        return False

# ---------------------------------------------------------------------------
# 4. Drawdown
# ---------------------------------------------------------------------------

def max_drawdown_ratio(daily: pd.DataFrame) -> float:
    if "cum_delta" not in daily.columns or daily.empty:
        return 0.0
    try:
        s = daily["cum_delta"].fillna(method="ffill").values.astype(float)
        peak = np.maximum.accumulate(s)
        dd = np.where(peak > 0, (s - peak) / (np.abs(peak) + 1e-12), 0.0)
        return float(np.min(dd))
    except Exception:
        return 0.0

# ---------------------------------------------------------------------------
# 5. Early stop
# ---------------------------------------------------------------------------

def early_stop_condition(days: int, cum_delta: float) -> bool:
    return days >= EARLY_STOP_DAYS and cum_delta < EARLY_STOP_DELTA

# ---------------------------------------------------------------------------
# 6. Recent regime filter
# ---------------------------------------------------------------------------

def recent_performance_filter(daily: pd.DataFrame) -> bool:
    if daily.empty or "cum_delta" not in daily.columns:
        return False
    try:
        recent = daily.tail(RECENT_WINDOW)
        if len(recent) < RECENT_WINDOW:
            return False
        delta_inc = recent["cum_delta"].iloc[-1] - recent["cum_delta"].iloc[0]
        if delta_inc <= RECENT_DELTA_MIN:
            return False
        if "sign_match" in recent.columns:
            sign_avg = recent["sign_match"].mean()
            if pd.isna(sign_avg) or sign_avg <= RECENT_SIGN_MIN:
                return False
        return True
    except Exception:
        return False


def recent_vs_total_consistency(daily: pd.DataFrame) -> bool:
    if daily.empty or "cum_delta" not in daily.columns or len(daily) < RECENT_WINDOW + 1:
        return False
    try:
        total_vals = daily["cum_delta"].values.astype(float)
        if len(total_vals) < 2:
            return True
        total_slope = (total_vals[-1] - total_vals[0]) / max(len(total_vals) - 1, 1)
        recent_vals = total_vals[-RECENT_WINDOW:]
        recent_slope = (recent_vals[-1] - recent_vals[0]) / max(len(recent_vals) - 1, 1)
        if abs(total_slope) > 1e-9 and abs(recent_slope) > 2.0 * abs(total_slope):
            return False
        return True
    except Exception:
        return True

# ---------------------------------------------------------------------------
# 9. Trade metrics
# ---------------------------------------------------------------------------

def compute_trade_metrics(trades: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"corr": 0.0, "trade_sign_match": 0.0, "n_trades": 0, "tstat": 0.0}
    if trades.empty:
        return result
    if "ea" not in trades.columns or "forward_return" not in trades.columns:
        return result
    try:
        df = trades[["ea", "forward_return"]].dropna()
        n = len(df)
        result["n_trades"] = n
        if n < 5:
            return result
        corr = float(df["ea"].corr(df["forward_return"]))
        if pd.isna(corr):
            corr = 0.0
        result["corr"] = corr
        result["tstat"] = corr_tstat(corr, n)
        sign_match = float((np.sign(df["ea"].values) == np.sign(df["forward_return"].values)).mean())
        result["trade_sign_match"] = sign_match
    except Exception:
        pass
    return result

# ---------------------------------------------------------------------------
# 10 + 11. Decision logic
# ---------------------------------------------------------------------------

def robust_decision_final(
    days: int,
    cum_delta: float,
    sign_match: float,
    trade_metrics: dict,
    stability_ok: bool,
    dd_ratio: float,
    recent_ok: bool,
    consistency_ok: bool,
) -> str:
    n_trades = trade_metrics.get("n_trades", 0)
    tstat = trade_metrics.get("tstat", 0.0)
    trade_sign = trade_metrics.get("trade_sign_match", 0.0)

    # Early stop (overrides all)
    if early_stop_condition(days, cum_delta):
        return "KILL_EARLY"

    # Not enough data
    if days < PROMOTE_DAYS:
        return "HOLD"

    # PROMOTE conditions
    if (
        sign_match > SIGN_MATCH_PROMOTE
        and trade_sign > TRADE_SIGN_PROMOTE
        and tstat > TSTAT_THRESH
        and n_trades >= MIN_TRADES
        and stability_ok
        and dd_ratio > DD_LIMIT
        and recent_ok
        and consistency_ok
        and cum_delta > 0
    ):
        return "PROMOTE"

    # KILL conditions
    if cum_delta < CUM_DELTA_KILL and sign_match < SIGN_MATCH_KILL:
        return "KILL"

    return "HOLD"

# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate() -> dict[str, Any]:
    result: dict[str, Any] = {
        "days": 0,
        "cum_delta": 0.0,
        "sign_match": 0.0,
        "trade_sign_match": 0.0,
        "corr": 0.0,
        "tstat": 0.0,
        "n_trades": 0,
        "decision": "HOLD",
        "finalized": False,
        "data_issues": [],
    }

    # --- Load state ---
    state = load_state()
    result["finalized"] = state["finalized"]

    # --- Load daily CSV (always needed for days count) ---
    daily = pd.DataFrame()
    try:
        if DAILY_CSV.exists() and DAILY_CSV.stat().st_size > 0:
            daily = pd.read_csv(DAILY_CSV, parse_dates=True)
            if "date" in daily.columns:
                daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
                daily.sort_values("date", inplace=True)
                daily.reset_index(drop=True, inplace=True)
    except Exception:
        pass

    days = len(daily)
    result["days"] = days

    # --- Finalized: complete skip of normal evaluation ---
    if state["finalized"]:
        result["decision"] = state["decision"]
        if days > RE_EVAL_DAYS:
            # Re-evaluation only — state is never updated here
            cum_delta = 0.0
            sign_match = 0.0
            if not daily.empty:
                if "cum_delta" in daily.columns:
                    v = daily["cum_delta"].iloc[-1]
                    cum_delta = float(v) if pd.notna(v) else 0.0
                if "sign_match" in daily.columns:
                    v = daily["sign_match"].mean()
                    sign_match = float(v) if pd.notna(v) else 0.0
            result["cum_delta"] = cum_delta
            result["sign_match"] = sign_match
            trades = load_jsonl_safe(TRADES_JSONL)
            tm = compute_trade_metrics(trades)
            result["corr"] = tm["corr"]
            result["tstat"] = tm["tstat"]
            result["trade_sign_match"] = tm["trade_sign_match"]
            result["n_trades"] = tm["n_trades"]
            stability_ok = stability_check_v2(daily)
            dd_ratio = max_drawdown_ratio(daily)
            recent_ok = recent_performance_filter(daily)
            consistency_ok = recent_vs_total_consistency(daily)
            re_dec = robust_decision_final(
                days, cum_delta, sign_match, tm,
                stability_ok, dd_ratio, recent_ok, consistency_ok,
            )
            result["re_evaluation"] = {
                "decision": re_dec,
                "stability_ok": stability_ok,
                "dd_ratio": dd_ratio,
                "recent_ok": recent_ok,
                "consistency_ok": consistency_ok,
            }
        write_debug_log(result)
        return result

    # --- Normal evaluation path ---
    cum_delta = 0.0
    sign_match = 0.0
    if not daily.empty:
        if "cum_delta" in daily.columns:
            v = daily["cum_delta"].iloc[-1]
            cum_delta = float(v) if pd.notna(v) else 0.0
        if "sign_match" in daily.columns:
            v = daily["sign_match"].mean()
            sign_match = float(v) if pd.notna(v) else 0.0

    result["cum_delta"] = cum_delta
    result["sign_match"] = sign_match

    # --- Load trades ---
    trades = load_jsonl_safe(TRADES_JSONL)
    data_issues: list[str] = []
    data_issues.extend(validate_trades(trades))
    data_issues.extend(check_autocorr(trades))
    result["data_issues"] = data_issues

    # Trade metrics
    tm = compute_trade_metrics(trades)
    result["corr"] = tm["corr"]
    result["tstat"] = tm["tstat"]
    result["trade_sign_match"] = tm["trade_sign_match"]
    result["n_trades"] = tm["n_trades"]

    # --- Compute decision ---
    stability_ok = stability_check_v2(daily)
    dd_ratio = max_drawdown_ratio(daily)
    recent_ok = recent_performance_filter(daily)
    consistency_ok = recent_vs_total_consistency(daily)

    decision = robust_decision_final(
        days, cum_delta, sign_match, tm,
        stability_ok, dd_ratio, recent_ok, consistency_ok,
    )

    result["decision"] = decision
    result["anomaly"] = {
        "stability_ok": stability_ok,
        "dd_ratio": dd_ratio,
        "recent_ok": recent_ok,
        "consistency_ok": consistency_ok,
    }

    # 11. Finalize on terminal decisions
    if decision in ("PROMOTE", "KILL", "KILL_EARLY"):
        state["finalized"] = True
        state["decision"] = decision
        save_state(state)
        result["finalized"] = True

    write_debug_log(result)
    return result


if __name__ == "__main__":
    r = evaluate()
    print(json.dumps(r, ensure_ascii=False, indent=2))
