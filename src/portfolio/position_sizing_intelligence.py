"""
position_sizing_intelligence.py — Position Sizing Intelligence (Phase 1)

Observation-only layer. Computes virtual conviction weights for BUY candidates.
Does NOT modify actual order sizes in Phase 1.

Conviction components (weights sum to 1.0):
  RSR percentile   0.35  — cross-sectional strength (rsr_pct_smooth × 100, else rsr)
  Entry timing     0.25  — entry quality score (0-100, FAIL_OPEN → 50)
  SEPA             0.20  — trend structure integrity (sepa_score/8 × 100)
  RSR momentum     0.10  — momentum direction (sigmoid-normalized)
  Future leader    0.10  — emerging strength indicator (FAIL_OPEN → 50)

Virtual weight: softmax-normalized with T=30 across BUY candidates.

Future production path (Phase 2+):
  Phase 2 activation: strategy.yaml → position_sizing.auto_apply: true  (ASK_FIRST)

  Code path virtual_weight → actual order qty:
    signal_bridge.py:~2374  _ps_results = compute_virtual_weights(psi_inputs)
    signal_bridge.py:4167   _build_orders(..., psi_scores=_ps_results, psi_enabled=True)
    signal_bridge.py:3224   [CHANGE] _effective_alloc_cap = deployable × virtual_weight
                            [GUARD]  min(_alloc_from_psi, capital × MAX_SINGLE_WEIGHT)
                            [CURRENT] _effective_alloc_cap = max_alloc_cap  (equal weight)
    signal_bridge.py:3227   qty_cap = int(_effective_alloc_cap // lot_cost) × 100
    signal_bridge.py:3249   qty = min(qty_risk, qty_cap)
    signal_bridge.py:3291   OrderInstruction(qty=qty)
    signal_bridge.py:3341   _send_orders(orders)
    signal_bridge.py:3495   qty = int(o.qty)
    signal_bridge.py:3500   self._client.send_order(symbol, exchange, side, qty, ...)
    kabusapi/client.py:345  KabusClient.send_order()  ← kabu ステーション REST API

Feature flag: strategy.yaml → position_sizing.enabled (default: true)
FAIL_OPEN throughout. Telemetry failure must NEVER halt execution.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_JST = timezone(timedelta(hours=9))

# ── Component weights (must sum to 1.0) ───────────────────────────────────────
W_RSR_PERCENTILE: float = 0.35
W_ENTRY_TIMING:   float = 0.25
W_SEPA:           float = 0.20
W_RSR_MOMENTUM:   float = 0.10
W_FUTURE_LEADER:  float = 0.10

# ── Softmax temperature ────────────────────────────────────────────────────────
# T=30: 50pt conviction spread → ~2x weight ratio; matches design example
SOFTMAX_TEMPERATURE: float = 30.0
NEUTRAL_SCORE:       float = 50.0   # fallback when component unavailable

# ── Reason codes ──────────────────────────────────────────────────────────────
RC_RSR_SMOOTH_USED: str = "RSR_SMOOTH"
RC_RSR_FALLBACK:    str = "RSR_FALLBACK"
RC_ET_MISSING:      str = "ET_MISSING"
RC_FL_MISSING:      str = "FL_MISSING"
RC_FAIL_OPEN:       str = "FAIL_OPEN"


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PositionSizingInput:
    symbol:              str
    rsr:                 float              # 0-100 RSR score
    sepa_score:          int                # 0-8 SEPA conditions met
    rsr_momentum:        float              # raw momentum (typically -20 to +20)
    rsr_pct_smooth:      Optional[float] = None  # 0-1 percentile; preferred over rsr
    entry_timing_score:  Optional[float] = None  # 0-100; FAIL_OPEN if None → neutral
    future_leader_score: Optional[float] = None  # 0-100; FAIL_OPEN if None → neutral


@dataclass
class PositionSizingSignal:
    symbol:           str
    conviction_score: float              # 0-100 composite conviction
    virtual_weight:   float              # 0-1; sums to 1.0 across candidates
    component_scores: Dict[str, float]  # per-component breakdown
    reason_codes:     List[str]         # missing/used component flags
    computed_at:      str               # ISO timestamp


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rsr_component(inp: PositionSizingInput) -> Tuple[float, str]:
    """RSR strength 0-100. Uses rsr_pct_smooth*100 if available."""
    if inp.rsr_pct_smooth is not None:
        return max(0.0, min(100.0, inp.rsr_pct_smooth * 100.0)), RC_RSR_SMOOTH_USED
    return max(0.0, min(100.0, float(inp.rsr))), RC_RSR_FALLBACK


def _momentum_component(mom: float) -> float:
    """Sigmoid normalization: 0→50, +5→73, +10→88, -5→27, -10→12."""
    try:
        return round(100.0 / (1.0 + math.exp(-mom / 5.0)), 2)
    except OverflowError:
        return 0.0 if mom < 0 else 100.0


def _compute_conviction(
    inp: PositionSizingInput,
) -> Tuple[float, Dict[str, float], List[str]]:
    """
    Returns (conviction_score, component_scores, reason_codes).
    FAIL_OPEN: returns (NEUTRAL_SCORE, {}, [RC_FAIL_OPEN]) on exception.
    """
    try:
        reason_codes: List[str] = []

        rsr_comp, rsr_rc = _rsr_component(inp)
        reason_codes.append(rsr_rc)

        if inp.entry_timing_score is not None:
            et_comp = max(0.0, min(100.0, float(inp.entry_timing_score)))
        else:
            et_comp = NEUTRAL_SCORE
            reason_codes.append(RC_ET_MISSING)

        sepa_comp = max(0.0, min(100.0, (inp.sepa_score / 8.0) * 100.0))
        mom_comp  = _momentum_component(inp.rsr_momentum)

        if inp.future_leader_score is not None:
            fl_comp = max(0.0, min(100.0, float(inp.future_leader_score)))
        else:
            fl_comp = NEUTRAL_SCORE
            reason_codes.append(RC_FL_MISSING)

        conviction = (
            W_RSR_PERCENTILE * rsr_comp +
            W_ENTRY_TIMING   * et_comp   +
            W_SEPA           * sepa_comp +
            W_RSR_MOMENTUM   * mom_comp  +
            W_FUTURE_LEADER  * fl_comp
        )
        components: Dict[str, float] = {
            "rsr":           round(rsr_comp,  2),
            "entry_timing":  round(et_comp,   2),
            "sepa":          round(sepa_comp, 2),
            "momentum":      round(mom_comp,  2),
            "future_leader": round(fl_comp,   2),
        }
        return max(0.0, min(100.0, round(conviction, 2))), components, reason_codes

    except Exception as exc:
        logger.warning("[PSI] conviction failed for %s: %s", getattr(inp, "symbol", "?"), exc)
        return NEUTRAL_SCORE, {}, [RC_FAIL_OPEN]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_conviction_score(inp: PositionSizingInput) -> float:
    """Single-symbol conviction 0-100. FAIL_OPEN."""
    conv, _, _ = _compute_conviction(inp)
    return conv


def compute_virtual_weights(
    inputs:      List[PositionSizingInput],
    temperature: float = SOFTMAX_TEMPERATURE,
) -> List[PositionSizingSignal]:
    """
    Compute conviction scores and softmax-normalized virtual weights.
    virtual_weight sums to 1.0 across all inputs.
    Order of output matches order of inputs.
    FAIL_OPEN: returns [] on any unhandled exception.
    """
    if not inputs:
        return []
    try:
        now_str = datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S+09:00")

        scored: List[Tuple[PositionSizingInput, float, Dict, List]] = []
        for inp in inputs:
            conv, comps, reasons = _compute_conviction(inp)
            scored.append((inp, conv, comps, reasons))

        convictions = [c for _, c, _, _ in scored]
        T = max(float(temperature), 1.0)
        max_c = max(convictions)  # numerical stability (subtract max before exp)
        exp_vals: List[float] = []
        for c in convictions:
            try:
                exp_vals.append(math.exp((c - max_c) / T))
            except OverflowError:
                exp_vals.append(0.0)
        sum_exp = sum(exp_vals) or 1.0

        results: List[PositionSizingSignal] = []
        for (inp, conv, comps, reasons), exp_v in zip(scored, exp_vals):
            vw = round(exp_v / sum_exp, 6)
            results.append(PositionSizingSignal(
                symbol           = inp.symbol,
                conviction_score = conv,
                virtual_weight   = vw,
                component_scores = comps,
                reason_codes     = reasons,
                computed_at      = now_str,
            ))
        return results

    except Exception as exc:
        logger.warning("[PSI] virtual weight computation failed: %s", exc)
        return []


def append_ps_telemetry(
    signals:              List[PositionSizingSignal],
    actual_weights:       Dict[str, float],
    rsr_scores:           Dict[str, Optional[float]],
    entry_timing_scores:  Dict[str, Optional[float]],
    future_leader_scores: Dict[str, Optional[float]],
    telemetry_path:       Path,
    date_str:             str,
) -> None:
    """Append one JSONL record per signal. FAIL_OPEN."""
    try:
        if not signals:
            return
        telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        now_str = datetime.now(_JST).strftime("%Y-%m-%dT%H:%M:%S+09:00")
        with telemetry_path.open("a", encoding="utf-8") as f:
            for sig in signals:
                record = {
                    "timestamp":           now_str,
                    "date":                date_str,
                    "symbol":              sig.symbol,
                    "conviction_score":    sig.conviction_score,
                    "virtual_weight":      sig.virtual_weight,
                    "actual_weight":       actual_weights.get(sig.symbol),
                    "rsr_score":           rsr_scores.get(sig.symbol),
                    "entry_timing_score":  entry_timing_scores.get(sig.symbol),
                    "future_leader_score": future_leader_scores.get(sig.symbol),
                    "reason_codes":        sig.reason_codes,
                    "components":          sig.component_scores,
                }
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        logger.warning("[PSI] telemetry append failed: %s", exc)


def _load_telemetry_window(
    telemetry_path: Path,
    window_days:    int = 30,
    today:          Optional[date] = None,
) -> List[dict]:
    """Load records within window_days. FAIL_OPEN → []."""
    try:
        if not telemetry_path.exists():
            return []
        if today is None:
            today = datetime.now(_JST).date()
        cutoff = today - timedelta(days=window_days)
        records: List[dict] = []
        for line in telemetry_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                rec_date = datetime.strptime(r["date"], "%Y-%m-%d").date()
                if rec_date >= cutoff:
                    records.append(r)
            except Exception:
                continue
        return records
    except Exception as exc:
        logger.warning("[PSI] telemetry load failed: %s", exc)
        return []


def compute_ps_kpis(
    telemetry_path: Path,
    window_days:    int = 30,
    today:          Optional[date] = None,
) -> Dict[str, Any]:
    """
    KPIs from telemetry for weekly review.
    Keys: average_conviction, top_conviction_symbols, weight_dispersion_cv,
          virtual_vs_equal_diff, high_conviction_rate, sample_count.
    FAIL_OPEN: returns {} on failure.
    """
    try:
        records = _load_telemetry_window(telemetry_path, window_days, today)
        if not records:
            return {}

        convictions: List[float] = [
            float(r["conviction_score"]) for r in records
            if r.get("conviction_score") is not None
        ]
        vweights: List[float] = [
            float(r["virtual_weight"]) for r in records
            if r.get("virtual_weight") is not None
        ]
        if not convictions:
            return {}

        avg_conviction  = round(sum(convictions) / len(convictions), 2)
        high_conv_rate  = round(sum(1 for c in convictions if c >= 75) / len(convictions), 4)

        by_symbol: Dict[str, List[float]] = {}
        for r in records:
            sym = r.get("symbol", "")
            c   = r.get("conviction_score")
            if sym and c is not None:
                by_symbol.setdefault(sym, []).append(float(c))
        top_symbols = sorted(
            [(sym, round(max(vals), 1)) for sym, vals in by_symbol.items()],
            key=lambda x: -x[1],
        )[:5]

        # Coefficient of variation of virtual weights
        cv = 0.0
        if len(vweights) > 1:
            mean_w = sum(vweights) / len(vweights)
            var_w  = sum((w - mean_w) ** 2 for w in vweights) / (len(vweights) - 1)
            cv     = round((var_w ** 0.5) / mean_w if mean_w > 0 else 0.0, 4)

        # Average abs diff between virtual_weight and equal actual_weight
        diffs: List[float] = []
        for r in records:
            vw = r.get("virtual_weight")
            aw = r.get("actual_weight")
            if vw is not None and aw is not None:
                try:
                    diffs.append(abs(float(vw) - float(aw)))
                except Exception:
                    pass
        avg_diff = round(sum(diffs) / len(diffs), 6) if diffs else None

        return {
            "average_conviction":     avg_conviction,
            "top_conviction_symbols": top_symbols,
            "weight_dispersion_cv":   cv,
            "virtual_vs_equal_diff":  avg_diff,
            "high_conviction_rate":   high_conv_rate,
            "sample_count":           len(records),
        }

    except Exception as exc:
        logger.warning("[PSI] KPI computation failed: %s", exc)
        return {}


def format_ps_report_section(
    signals: List[PositionSizingSignal],
    kpis:    Dict[str, Any],
) -> str:
    """
    Format [POSITION SIZING] section for DRY/LIVE reports.
    FAIL_OPEN: returns "" on error.
    """
    try:
        if not signals and not kpis:
            return ""

        lines = [
            "",
            "━━ [POSITION SIZING — 仮想コンビクション配分] ━━━━━━━━━━━━━━",
        ]

        if signals:
            equal_w = 1.0 / len(signals)
            lines.append(
                f"  候補数: {len(signals)}銘柄  等配分比較: {equal_w:.1%}/銘柄"
                "  [観測専用 — 実発注サイズ未変更]"
            )
            lines.append(
                f"  {'銘柄':<8} {'コンビクション':>13} {'仮想ウェイト':>12} "
                f"{'等配分':>8} {'差分':>8}"
            )
            lines.append("  " + "─" * 55)
            for sig in sorted(signals, key=lambda s: -s.conviction_score):
                delta    = sig.virtual_weight - equal_w
                delta_s  = f"{delta:+.1%}"
                lines.append(
                    f"  {sig.symbol:<8} "
                    f"{sig.conviction_score:>13.1f} "
                    f"{sig.virtual_weight:>12.1%} "
                    f"{equal_w:>8.1%} "
                    f"{delta_s:>8}"
                )

        if kpis:
            n = kpis.get("sample_count", 0)
            lines.append(f"  ── 過去{n}件 累積KPI ─────────────────────────────────")
            if (v := kpis.get("average_conviction"))    is not None:
                lines.append(f"  平均コンビクション     : {v:.1f}")
            if (v := kpis.get("high_conviction_rate"))  is not None:
                lines.append(f"  高信頼率(≥75)        : {v:.1%}")
            if (v := kpis.get("weight_dispersion_cv"))  is not None:
                lines.append(f"  ウェイト分散(CV)      : {v:.4f}")
            if (v := kpis.get("virtual_vs_equal_diff")) is not None:
                lines.append(f"  仮想vs等配分 平均差    : {v:.4f}")
            top5 = kpis.get("top_conviction_symbols", [])
            if top5:
                sym_str = "  ".join(f"{s}({c})" for s, c in top5)
                lines.append(f"  高信頼TOP銘柄         : {sym_str}")

        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return "\n".join(lines)

    except Exception as exc:
        logger.warning("[PSI] report format failed: %s", exc)
        return ""
