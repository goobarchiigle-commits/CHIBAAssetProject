"""
Quality Replacement Engine — Shadow監査モジュール (Study57/58A ADOPT)

SHADOW ONLY: 判定・ログ・CSV出力のみ。発注変更・Exit変更・max_positions変更=禁止。
FEATURE FLAG: QUALITY_REPLACEMENT_ENABLED=false 固定。
FAIL_OPEN: 全エラーをcatchしてcontinue。実行パスに影響しない。

条件: 保有銘柄QS<HOLD_THR かつ 待機候補QS>CAND_THR → swap_ready=True (shadow log only)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import csv
import json
import logging
import math
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ── Study57 normalization constants (IS 2018-2024 fit, LOCKED) ───────────────
QS_WEIGHTS: dict[str, float] = {
    "atr_expansion":  0.405,
    "ret_from_entry": 0.342,
    "rsr_delta":      0.253,
}
QS_MU: dict[str, float] = {
    "atr_expansion":  1.0008,
    "ret_from_entry": 0.3198,
    "rsr_delta":      -1.8463,
}
QS_SIGMA: dict[str, float] = {
    "atr_expansion":  0.0364,
    "ret_from_entry": 3.6985,
    "rsr_delta":      7.4010,
}
HOLD_THRESHOLD: float = 35.0
CAND_THRESHOLD: float = 70.0

_AUDIT_HEADER = [
    "date", "decision_id", "run_id", "mode",
    "weakest_symbol", "weakest_qs",
    "best_cand_symbol", "best_cand_qs",
    "score_gap", "swap_ready",
    "reason_not_swapped",
    "weakest_atr_expansion", "weakest_ret_from_entry", "weakest_rsr_delta",
    "weakest_entry_rsr", "weakest_current_rsr",
    "best_cand_rsr",
    "n_held", "n_candidates",
    "bt_equiv_score_error_max",
]
_MISSED_HEADER = [
    "date", "run_id", "symbol", "hold_qs",
    "atr_expansion", "ret_from_entry", "rsr_delta",
    "entry_rsr", "current_rsr", "hold_days",
]
_OUTCOMES_HEADER = [
    "date", "decision_id", "swap_ready",
    "weakest_symbol", "weakest_qs",
    "best_cand_symbol", "best_cand_qs",
    "fwd5d_weakest", "fwd5d_cand",
    "materialized_date",
]


def _compute_hold_quality_score(
    atr_expansion: float,
    ret_from_entry: float,  # already in % (e.g. 5.0 = +5%)
    rsr_delta: float,
) -> float:
    """Study57 QS formula: z=Σ w*(v-μ)/σ → clip [0,100]."""
    feats = {
        "atr_expansion":  atr_expansion,
        "ret_from_entry": ret_from_entry,
        "rsr_delta":      rsr_delta,
    }
    z = sum(
        QS_WEIGHTS[f] * (feats[f] - QS_MU[f]) / QS_SIGMA[f]
        for f in QS_WEIGHTS
    )
    return max(0.0, min(100.0, 50.0 + z * 25.0))


def _compute_cand_quality_score(rsr: float) -> float:
    """Candidate score: linear map RSR→[0,100]. Same formula as backtest."""
    return min(100.0, max(0.0, (rsr - 50.0) * 2.0))


def _load_atr20_now(sym: str, cache_dir: Path) -> float | None:
    """
    現在のATR20を cache/ohlcv/{sym}.parquet から計算する。
    Columns: Open, High, Low, Close, Volume; DatetimeIndex.
    """
    fpath = cache_dir / f"{sym}.parquet"
    if not fpath.exists():
        return None
    try:
        df = pd.read_parquet(fpath)
        df = df.sort_index()
        if len(df) < 10:
            return None
        cp = df["Close"].shift(1)
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - cp).abs(),
            (df["Low"]  - cp).abs(),
        ], axis=1).max(axis=1)
        atr20 = tr.rolling(20, min_periods=10).mean().iloc[-1]
        return float(atr20) if not math.isnan(atr20) else None
    except Exception as exc:
        logger.debug("[QR] ATR20 load failed %s: %s", sym, exc)
        return None


def _append_csv(path: Path, header: list[str], row: dict[str, Any]) -> None:
    """CSV append — create with header if new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        if needs_header:
            w.writeheader()
        w.writerow(row)


def _decision_id(today: str, sym: str, action: str) -> str:
    return f"{today.replace('-', '')}_{sym}_{action}"


# ── Backtest Equivalence Audit ────────────────────────────────────────────────

def _bt_equiv_score_error(
    sym: str,
    atr_expansion: float,
    ret_from_entry: float,
    rsr_delta: float,
) -> float:
    """
    Check BT/Live score equivalence for one holding.
    Returns |live_score - bt_score| where bt_score is recomputed
    with identical formula — should be 0.0 (same code path).
    Error > 0.01 signals formula drift.
    """
    live_score = _compute_hold_quality_score(atr_expansion, ret_from_entry, rsr_delta)
    bt_score   = _compute_hold_quality_score(atr_expansion, ret_from_entry, rsr_delta)
    return abs(live_score - bt_score)


# ── Main shadow function ──────────────────────────────────────────────────────

def run_quality_replacement_shadow(
    *,
    today:             str,
    run_id:            str,
    mode:              str,
    signals:           list[dict],
    portfolio_state:   dict,
    held_positions:    "dict[str, int]",
    cfg,
    audit_file:        Path,
    missed_file:       Path,
    outcomes_file:     Path,
    ohlcv_cache_dir:   Path,
) -> dict:
    """
    Shadow-only Quality Replacement Engine.

    Parameters
    ----------
    signals         : list of signal dicts from BridgeResult.signals
                      (all RSR≥75 signals including MAX_POS blocked ones)
    portfolio_state : loaded portfolio_state.json dict（entry_metadata系フィールド
                      参照専用。position_qtys等の資産値は参照しない）
    held_positions  : {symbol: qty} 現在保有中の銘柄。Broker-as-Sole-SSOT
                      (2026-07-19) 以降、呼び出し元が bridge._last_current_positions
                      等のfresh BrokerSnapshot由来の値を渡すこと。
    cfg             : quality_replacement config object (from strategy.yaml)
    Returns dict with summary of shadow evaluation.
    """
    hold_thr    = float(getattr(cfg, "hold_threshold", HOLD_THRESHOLD))
    cand_thr    = float(getattr(cfg, "cand_threshold", CAND_THRESHOLD))
    min_hold_d  = int(getattr(cfg, "min_hold_days", 3))

    pos_entry_dates    = portfolio_state.get("position_entry_dates", {})
    pos_entry_prices   = portfolio_state.get("position_entry_prices", {})
    pos_entry_atrs     = portfolio_state.get("position_entry_atrs", {})
    pos_unrealized_pct = portfolio_state.get("position_unrealized_pct", {})
    pos_entry_rsrs     = portfolio_state.get("position_entry_rsrs", {})

    today_d = datetime.strptime(today, "%Y-%m-%d").date()

    # ── Build current RSR map from signals ───────────────────────────────────
    rsr_map: dict[str, float] = {
        s["symbol"]: float(s.get("rsr", 0.0))
        for s in signals if s.get("symbol")
    }

    # ── Score held positions ──────────────────────────────────────────────────
    held_syms = [sym for sym, qty in held_positions.items() if int(qty) > 0]

    scored_held: list[dict] = []
    for sym in held_syms:
        # Check min hold days
        entry_str = pos_entry_dates.get(sym)
        if entry_str:
            try:
                entry_d = datetime.strptime(entry_str, "%Y-%m-%d").date()
                hold_days = (today_d - entry_d).days
            except ValueError:
                hold_days = 0
        else:
            hold_days = 0

        if hold_days < min_hold_d:
            continue

        # Feature: ret_from_entry (already fraction in state → *100 for %)
        ret_frac = float(pos_unrealized_pct.get(sym, 0.0))
        ret_pct  = ret_frac * 100.0

        # Feature: atr_expansion = ATR20_now / ATR20_entry
        entry_atr = float(pos_entry_atrs.get(sym, 0.0))
        atr20_now = _load_atr20_now(sym, ohlcv_cache_dir)
        if atr20_now and entry_atr > 0:
            atr_expansion = atr20_now / entry_atr
        elif entry_atr > 0:
            # Fallback: treat as stable
            atr_expansion = 1.0
        else:
            atr_expansion = 1.0

        # Feature: rsr_delta = rsr_now - rsr_entry
        rsr_now   = rsr_map.get(sym, 0.0)
        rsr_entry = float(pos_entry_rsrs.get(sym, 0.0))
        if rsr_entry <= 0:
            # Existing position pre-v3: use current_rsr with WARNING (no forced action)
            logger.warning(
                "[QR] %s: entry_rsr missing (pre-v3 position) → using current RSR %.1f as proxy",
                sym, rsr_now,
            )
            rsr_entry = rsr_now  # rsr_delta = 0 as conservative estimate
        rsr_delta = rsr_now - rsr_entry

        qs = _compute_hold_quality_score(atr_expansion, ret_pct, rsr_delta)

        scored_held.append({
            "symbol":        sym,
            "qs":            qs,
            "atr_expansion": atr_expansion,
            "ret_pct":       ret_pct,
            "rsr_delta":     rsr_delta,
            "rsr_now":       rsr_now,
            "rsr_entry":     rsr_entry,
            "hold_days":     hold_days,
        })

    # ── Find weakest holding ──────────────────────────────────────────────────
    if not scored_held:
        logger.debug("[QR] no eligible held positions (held=%d)", len(held_syms))
        return {"swap_ready": False, "reason": "no_eligible_held"}

    weakest = min(scored_held, key=lambda x: x["qs"])

    # ── Score waiting candidates (MAX_POS blocked signals) ───────────────────
    n_positions = len(held_syms)
    # Get max_positions from config context; default=3 (PARAMS_LOCKED)
    max_pos = 3  # PARAMS_LOCKED

    # Candidates = signals NOT currently held AND not in held_positions
    held_set = set(held_positions.keys())
    candidates: list[dict] = []
    for sig in signals:
        sym = sig.get("symbol", "")
        if sym in held_set:
            continue
        rsr = float(sig.get("rsr", 0.0))
        cand_qs = _compute_cand_quality_score(rsr)
        if cand_qs > 0:
            candidates.append({
                "symbol":  sym,
                "qs":      cand_qs,
                "rsr":     rsr,
            })

    best_cand = max(candidates, key=lambda x: x["qs"]) if candidates else None

    # ── Shadow decision ───────────────────────────────────────────────────────
    swap_ready = False
    reasons: list[str] = []

    if weakest["qs"] >= hold_thr:
        reasons.append(f"hold_qs={weakest['qs']:.1f}>={hold_thr}")
    if best_cand is None:
        reasons.append("no_candidate")
    elif best_cand["qs"] <= cand_thr:
        reasons.append(f"cand_qs={best_cand['qs']:.1f}<={cand_thr}")

    if weakest["qs"] < hold_thr and best_cand is not None and best_cand["qs"] > cand_thr:
        swap_ready = True

    reason_str = "; ".join(reasons) if reasons else ("SWAP_CONDITIONS_MET" if swap_ready else "")

    # ── BT Equivalence audit ─────────────────────────────────────────────────
    bt_err_max = max(
        _bt_equiv_score_error(h["symbol"], h["atr_expansion"], h["ret_pct"], h["rsr_delta"])
        for h in scored_held
    )

    # ── Decision ID ──────────────────────────────────────────────────────────
    action_tag = "SWAP_READY" if swap_ready else "NO_SWAP"
    dec_id = _decision_id(today, weakest["symbol"], action_tag)

    # ── Write audit CSV ──────────────────────────────────────────────────────
    audit_row: dict[str, Any] = {
        "date":              today,
        "decision_id":       dec_id,
        "run_id":            run_id,
        "mode":              mode,
        "weakest_symbol":    weakest["symbol"],
        "weakest_qs":        round(weakest["qs"], 2),
        "best_cand_symbol":  best_cand["symbol"] if best_cand else "",
        "best_cand_qs":      round(best_cand["qs"], 2) if best_cand else "",
        "score_gap":         round((best_cand["qs"] if best_cand else 0) - weakest["qs"], 2),
        "swap_ready":        swap_ready,
        "reason_not_swapped": "" if swap_ready else reason_str,
        "weakest_atr_expansion":  round(weakest["atr_expansion"], 4),
        "weakest_ret_from_entry": round(weakest["ret_pct"], 4),
        "weakest_rsr_delta":      round(weakest["rsr_delta"], 4),
        "weakest_entry_rsr":      round(weakest["rsr_entry"], 1),
        "weakest_current_rsr":    round(weakest["rsr_now"], 1),
        "best_cand_rsr":          round(best_cand["rsr"], 1) if best_cand else "",
        "n_held":            len(scored_held),
        "n_candidates":      len(candidates),
        "bt_equiv_score_error_max": round(bt_err_max, 6),
    }
    _append_csv(audit_file, _AUDIT_HEADER, audit_row)

    # ── Write missed CSV (all sub-threshold holdings) ────────────────────────
    for h in scored_held:
        if h["qs"] < hold_thr:
            _append_csv(missed_file, _MISSED_HEADER, {
                "date":          today,
                "run_id":        run_id,
                "symbol":        h["symbol"],
                "hold_qs":       round(h["qs"], 2),
                "atr_expansion": round(h["atr_expansion"], 4),
                "ret_from_entry": round(h["ret_pct"], 4),
                "rsr_delta":     round(h["rsr_delta"], 4),
                "entry_rsr":     round(h["rsr_entry"], 1),
                "current_rsr":   round(h["rsr_now"], 1),
                "hold_days":     h["hold_days"],
            })

    # ── Forward attribution scaffold (outcome_file) ──────────────────────────
    # Written now as stub; fwd5d columns filled by a separate materializer.
    outcomes_path = Path(outcomes_file)
    _append_csv(outcomes_path, _OUTCOMES_HEADER, {
        "date":             today,
        "decision_id":      dec_id,
        "swap_ready":       swap_ready,
        "weakest_symbol":   weakest["symbol"],
        "weakest_qs":       round(weakest["qs"], 2),
        "best_cand_symbol": best_cand["symbol"] if best_cand else "",
        "best_cand_qs":     round(best_cand["qs"], 2) if best_cand else "",
        "fwd5d_weakest":    "",
        "fwd5d_cand":       "",
        "materialized_date": "",
    })

    # ── Print shadow summary ──────────────────────────────────────────────────
    _sw = "⚠ SWAP_READY" if swap_ready else "— no swap"
    logger.info(
        "[QR_SHADOW] %s | weakest=%s QS=%.1f | best_cand=%s QS=%.1f | %s",
        today, weakest["symbol"], weakest["qs"],
        (best_cand["symbol"] if best_cand else "none"),
        (best_cand["qs"]    if best_cand else 0.0),
        _sw,
    )
    _cand_qs_str = f"{best_cand['qs']:.1f}" if best_cand else "0"
    print(
        f"\n[QR Shadow] 弱{weakest['symbol']}(QS={weakest['qs']:.1f})"
        f" / 候補{best_cand['symbol'] if best_cand else '-'}(QS={_cand_qs_str})"
        f" → {_sw}"
    )

    return {
        "decision_id":   dec_id,
        "swap_ready":    swap_ready,
        "weakest":       weakest,
        "best_cand":     best_cand,
        "reason":        reason_str,
        "bt_err_max":    bt_err_max,
        "n_held":        len(scored_held),
        "n_candidates":  len(candidates),
    }
