"""
expansion_state_engine.py — Intraday Expansion State Engine

Classifies post-open price action within 5-30 minutes as:
  "continuation" — sustained directional move, volume persisting
  "spike"        — 寄り天パターン: early peak then reversal
  "undetermined" — insufficient bars or mixed signals

PRIMARY CONSTRAINT: 寄り天 spike を continuation と誤認することを最優先で防ぐ。
  → spike_confidence が高いほど expansion_probability を強制的に抑制する。
  → current_close > early_peak (新高値) のみがスパイク判定を完全解除する。

Components:
  OpeningDriveEfficiency    compute_opening_drive()
  IntradayRVOLCalculator    compute_rvol()
  OpeningRangePersistence   compute_persistence()
  IntradayRelativeStrength  compute_rs()
  GapAcceptanceAnalyzer     compute_gap_acceptance()

Input:
  minute_bars                list[MinuteBar] — open 以降の1分足、時系列昇順
  prev_close                 float           — 前日終値
  universe_intraday_returns  dict[str,float] — {symbol: 当日騰落率} (クロスセクションRS用)
  eval_ts                    str             — 評価タイムスタンプ (呼び出し元が提供、決定論的)

Output:
  ExpansionState dataclass
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Trading session ──────────────────────────────────────────────────────────
TOTAL_TRADING_MINUTES: int      = 330   # 09:00-15:30 JST (除昼休み)

# ── Spike detection ──────────────────────────────────────────────────────────
SPIKE_EARLY_WINDOW: int         = 5     # 最初N本でearly_peak検出
MIN_BARS_FOR_VERDICT: int       = 10    # 10本未満はスパイク/継続判定なし
SPIKE_REVERSAL_THRESHOLD: float = 0.45  # early_peak から 45% 以上戻し → スパイク一次シグナル
SPIKE_PERSISTENCE_CUTOFF: float = 0.40  # OR内でのpersistence_ratio < 40% → スパイク補助
SPIKE_RVOL_ACCEL_CUTOFF: float  = 0.60  # rvol_acceleration < 60% → 出来高急落補助
SPIKE_CONFIDENCE_THRESHOLD: float = 0.50  # spike_confidence > 50% → type="spike"

# ── Continuation ─────────────────────────────────────────────────────────────
CONT_DRIVE_MIN: float           = 0.65
CONT_PERSISTENCE_MIN: float     = 0.55
CONT_RVOL_ACCEL_MIN: float      = 0.65
CONT_RS_PCT_MIN: float          = 50.0
CONT_GAP_ACCEPT_MIN: float      = 0.70  # gap_acceptance [0,1] scale

# ── Composite weights: expansion_probability ──────────────────────────────────
W_DRIVE: float      = 0.30
W_PERSIST: float    = 0.25
W_RVOL_ACCEL: float = 0.20
W_RS: float         = 0.15
W_GAP: float        = 0.10

# ── Spike confidence weights ──────────────────────────────────────────────────
W_SPIKE_REVERSAL: float = 0.45   # 最大ウェイト: early reversal が主因
W_SPIKE_PERSIST: float  = 0.35
W_SPIKE_RVOL: float     = 0.20

# expansion_probability *= (1 - SPIKE_SUPPRESSION * spike_confidence)
SPIKE_SUPPRESSION: float = 0.90

# ── Gap ──────────────────────────────────────────────────────────────────────
MIN_GAP_PCT: float = 0.003       # |gap_pct| < 0.3% → gap_active=False


# ─────────────────────────────────────────────────────────────────────────────
# Input / output dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MinuteBar:
    """1分足OHLCVバー。dtは呼び出し元が提供するタイムスタンプ文字列（順序のみ使用）。"""
    dt: str       # e.g. "09:05" or "2026-05-23T09:05:00+09:00"
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class OpeningDriveResult:
    efficiency: float          # [0,1] 高い = 方向性維持、低い = 戻り発生
    early_reversal_pct: float  # [0,1] early_peak からどれだけ戻したか (戻し率)
    early_peak: float          # 最初 SPIKE_EARLY_WINDOW 本での最高値
    new_high_confirmed: bool   # current_close > early_peak → 新高値で spike 解除


@dataclass
class RVOLResult:
    rvol_ratio: float        # 出来高ペース / ADV ペース (0 = ADV不明で中立=1.0)
    rvol_acceleration: float # 後半出来高レート / 前半出来高レート (1.0 = 中立)
    total_volume: int
    elapsed_minutes: int


@dataclass
class PersistenceResult:
    persistence_ratio: float  # [0,1] OR内でのcurrent位置 (高い = 上限付近)
    or_high: float
    or_low: float
    or_midpoint: float
    above_midpoint: bool


@dataclass
class RelativeStrengthResult:
    rs_percentile: float      # [0,100] ユニバース内のパーセンタイル順位


@dataclass
class GapResult:
    gap_pct: float            # (open - prev_close) / prev_close
    gap_acceptance: float     # [-1,1]: 1=gap維持/拡張, 0=gap消滅(prev_close), -1=gap以下
    gap_active: bool          # |gap_pct| >= MIN_GAP_PCT


@dataclass
class ExpansionState:
    """
    単一銘柄の intraday expansion 分類結果。
    全フィールドは入力に対して決定論的。
    """
    symbol: str
    eval_ts: str
    minutes_elapsed: int

    # Component metrics
    drive_efficiency: float      # [0,1]
    rvol_ratio: float            # ≥0; 1.0 = 中立(ADV不明時)
    rvol_acceleration: float     # ≥0; 1.0 = 前後半出来高均等
    persistence_ratio: float     # [0,1]
    rs_percentile: float         # [0,100]
    gap_acceptance: float        # [-1,1]

    # Composite
    expansion_probability: float  # [0,1]
    spike_confidence: float       # [0,1]

    # Classification
    expansion_type: str           # "continuation" | "spike" | "undetermined"
    is_spike_pattern: bool

    # Diagnostics
    new_high_confirmed: bool      # current > early_peak → spike 解除フラグ
    early_peak: float
    early_reversal_pct: float
    or_high: float
    or_low: float
    spike_reasons: list[str] = field(default_factory=list)
    continuation_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "eval_ts": self.eval_ts,
            "minutes_elapsed": self.minutes_elapsed,
            "drive_efficiency": self.drive_efficiency,
            "rvol_ratio": self.rvol_ratio,
            "rvol_acceleration": self.rvol_acceleration,
            "persistence_ratio": self.persistence_ratio,
            "rs_percentile": self.rs_percentile,
            "gap_acceptance": self.gap_acceptance,
            "expansion_probability": self.expansion_probability,
            "spike_confidence": self.spike_confidence,
            "expansion_type": self.expansion_type,
            "is_spike_pattern": self.is_spike_pattern,
            "new_high_confirmed": self.new_high_confirmed,
            "early_peak": self.early_peak,
            "early_reversal_pct": self.early_reversal_pct,
            "or_high": self.or_high,
            "or_low": self.or_low,
            "spike_reasons": self.spike_reasons,
            "continuation_reasons": self.continuation_reasons,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _float_safe(x) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else 0.0
    except Exception:
        return 0.0


def _clamp01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def _validate_bars(bars: list[MinuteBar]) -> list[MinuteBar]:
    """不正バー (価格 ≤ 0 / NaN / high < low) を除外して返す。"""
    valid: list[MinuteBar] = []
    for b in bars:
        try:
            o = _float_safe(b.open)
            h = _float_safe(b.high)
            lo = _float_safe(b.low)
            c = _float_safe(b.close)
            if o > 0 and h > 0 and lo > 0 and c > 0 and h >= lo:
                valid.append(MinuteBar(
                    dt=b.dt,
                    open=o, high=h, low=lo, close=c,
                    volume=max(0, int(b.volume)) if b.volume is not None else 0,
                ))
        except Exception:
            pass
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# Component: OpeningDriveEfficiency
# ─────────────────────────────────────────────────────────────────────────────

def compute_opening_drive(
    bars: list[MinuteBar],
    open_price: float,
    spike_window: int = SPIKE_EARLY_WINDOW,
) -> OpeningDriveResult:
    """
    方向性効率とearly reversal を計算する。

    drive_efficiency = net_move / max(peak_deviation, ε)
      - net_move = current_close - open
      - peak_deviation = max(|overall_peak - open|, |overall_trough - open|)
      → 1.0 = ピークで留まっている, 0 = 完全戻し

    early_reversal_pct:
      - 上昇スパイク: (early_peak - current_close) / (early_peak - open)
      - 0 以下 (新高値) → 0.0 にクランプ

    new_high_confirmed = current_close > early_peak
      → True なら spike_confidence を 0 に強制 (spike 解除)
    """
    if not bars or open_price <= 0:
        return OpeningDriveResult(0.5, 0.0, open_price, False)

    current_close = _float_safe(bars[-1].close)
    early_bars = bars[:spike_window]

    early_high = max((_float_safe(b.high) for b in early_bars), default=open_price)
    early_low  = min((_float_safe(b.low)  for b in early_bars), default=open_price)
    early_peak = early_high  # 上昇方向を主として採用

    overall_high = max((_float_safe(b.high) for b in bars), default=open_price)
    overall_low  = min((_float_safe(b.low)  for b in bars), default=open_price)

    net_move = current_close - open_price
    up_range   = abs(overall_high - open_price)
    down_range = abs(overall_low  - open_price)
    peak_deviation = max(up_range, down_range, 1e-8)

    efficiency = _clamp01(net_move / peak_deviation)

    # Early reversal (上昇スパイク方向)
    early_peak_range = abs(early_peak - open_price)
    if early_peak_range > open_price * 0.001 and early_peak > open_price:
        reversal_raw = (early_peak - current_close) / early_peak_range
        reversal_pct = _clamp01(reversal_raw)
    else:
        reversal_pct = 0.0

    new_high = current_close > early_peak

    return OpeningDriveResult(efficiency, reversal_pct, early_peak, new_high)


# ─────────────────────────────────────────────────────────────────────────────
# Component: IntradayRVOLCalculator
# ─────────────────────────────────────────────────────────────────────────────

def compute_rvol(
    bars: list[MinuteBar],
    avg_daily_volume: int,
    total_trading_minutes: int = TOTAL_TRADING_MINUTES,
    early_window: int = SPIKE_EARLY_WINDOW,
) -> RVOLResult:
    """
    RVOL比率と出来高加速度を計算する。

    rvol_ratio      = actual_vol / expected_vol_at_elapsed_time
                      ADV不明(=0)のとき 1.0 (中立)
    rvol_acceleration = 後半N本の出来高レート / 前半N本のレート
                      < 1 = 出来高急落 (スパイク補助シグナル)
                      > 1 = 出来高継続 (継続シグナル)
    """
    if not bars:
        return RVOLResult(1.0, 1.0, 0, 0)

    elapsed = len(bars)
    total_vol = sum(max(0, b.volume) for b in bars)

    if avg_daily_volume > 0 and elapsed > 0:
        expected = avg_daily_volume * (elapsed / total_trading_minutes)
        rvol_ratio = _float_safe(total_vol / max(expected, 1.0))
        rvol_ratio = float(np.clip(rvol_ratio, 0.0, 10.0))
    else:
        rvol_ratio = 1.0

    # Acceleration: early window vs most-recent window
    if len(bars) <= early_window:
        return RVOLResult(rvol_ratio, 1.0, total_vol, elapsed)

    early_bars  = bars[:early_window]
    recent_bars = bars[-early_window:]

    early_vol  = sum(max(0, b.volume) for b in early_bars)
    recent_vol = sum(max(0, b.volume) for b in recent_bars)

    early_rate  = early_vol  / early_window
    recent_rate = recent_vol / early_window

    if early_rate > 0:
        accel = _float_safe(recent_rate / early_rate)
        accel = float(np.clip(accel, 0.0, 5.0))
    else:
        accel = 1.0 if recent_rate == 0 else 2.0  # early zero but recent nonzero

    return RVOLResult(rvol_ratio, accel, total_vol, elapsed)


# ─────────────────────────────────────────────────────────────────────────────
# Component: OpeningRangePersistence
# ─────────────────────────────────────────────────────────────────────────────

def compute_persistence(
    bars: list[MinuteBar],
    or_window: int,
) -> PersistenceResult:
    """
    Opening Range (OR) 内でのcurrent価格位置を計算する。

    persistence_ratio = clamp((current_close - OR_low) / OR_range, 0, 1)
    1.0 = OR上限付近 (継続), 0.0 = OR下限付近 (スパイク後退)
    """
    if not bars:
        return PersistenceResult(0.5, 0.0, 0.0, 0.0, True)

    or_bars  = bars[:or_window] or bars
    or_high  = max((_float_safe(b.high) for b in or_bars), default=0.0)
    or_low   = min((_float_safe(b.low)  for b in or_bars), default=0.0)
    or_range = or_high - or_low

    current_close = _float_safe(bars[-1].close)

    if or_range < 1e-8:
        ratio = 0.5
    else:
        ratio = _clamp01((current_close - or_low) / or_range)

    midpoint = (or_high + or_low) / 2.0 if (or_high > 0 and or_low > 0) else current_close
    above_mid = current_close >= midpoint

    return PersistenceResult(ratio, or_high, or_low, midpoint, above_mid)


# ─────────────────────────────────────────────────────────────────────────────
# Component: IntradayRelativeStrength
# ─────────────────────────────────────────────────────────────────────────────

def compute_rs(
    symbol: str,
    universe_returns: dict[str, float],
) -> RelativeStrengthResult:
    """
    ユニバース内でのパーセンタイル順位 [0,100]。
    高い = 市場リード (継続)、低い = 市場アンダーパフォーム (スパイク後)。
    universe_returns が空またはこのsymbolが含まれない → 50.0 (中立)。
    """
    if not universe_returns or symbol not in universe_returns:
        return RelativeStrengthResult(50.0)

    sym_ret  = _float_safe(universe_returns[symbol])
    all_rets = [_float_safe(v) for v in universe_returns.values()]
    valid    = [r for r in all_rets if np.isfinite(r)]

    if len(valid) < 2:
        return RelativeStrengthResult(50.0)

    n_below = sum(1 for r in valid if r <= sym_ret)
    pct = float(n_below) / float(len(valid)) * 100.0
    return RelativeStrengthResult(round(pct, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Component: GapAcceptanceAnalyzer
# ─────────────────────────────────────────────────────────────────────────────

def compute_gap_acceptance(
    open_price: float,
    prev_close: float,
    current_close: float,
    min_gap_pct: float = MIN_GAP_PCT,
) -> GapResult:
    """
    ギャップの維持度を計算する。

    gap_acceptance = (current_close - prev_close) / |gap|
      1.0 → open 価格で維持 (ギャップ完全維持)
      > 1.0 → open 超えて上昇 (強い継続、1.0 にクランプ)
      0.0 → prev_close まで戻し (ギャップ消滅)
      < 0 → prev_close 割れ (ギャップ以上の下落)
    gap_active=False の場合は 0.5 (中立) を返す。
    """
    if prev_close <= 0 or open_price <= 0:
        return GapResult(0.0, 0.5, False)

    gap     = open_price - prev_close
    gap_pct = gap / prev_close if prev_close > 0 else 0.0

    if abs(gap_pct) < min_gap_pct or abs(gap) < 1e-8:
        return GapResult(gap_pct, 0.5, False)

    raw = (current_close - prev_close) / abs(gap)
    acceptance = float(np.clip(raw, -2.0, 2.0))  # 2x extensionまで考慮
    acceptance = float(np.clip(acceptance, -1.0, 1.0))  # 表示レンジ [-1,1]

    return GapResult(gap_pct, acceptance, True)


# ─────────────────────────────────────────────────────────────────────────────
# Spike / Continuation classification
# ─────────────────────────────────────────────────────────────────────────────

def _compute_spike_confidence(
    drive: OpeningDriveResult,
    rvol: RVOLResult,
    persist: PersistenceResult,
) -> tuple[float, list[str]]:
    """
    スパイク確信度 [0,1] と理由リストを返す。

    Primary trigger: early_reversal_pct > SPIKE_REVERSAL_THRESHOLD が必須。
    Primary なしでは spike_confidence = 0 (false positive 抑制)。

    new_high_confirmed (current > early_peak) のとき呼び出し元が 0 に強制。
    """
    reasons: list[str] = []

    has_primary = drive.early_reversal_pct > SPIKE_REVERSAL_THRESHOLD
    if not has_primary:
        return 0.0, []

    # 主シグナル: 戻し率
    reversal_signal = _clamp01(
        (drive.early_reversal_pct - SPIKE_REVERSAL_THRESHOLD) / (1.0 - SPIKE_REVERSAL_THRESHOLD)
    )
    reasons.append(
        f"early_reversal={drive.early_reversal_pct:.2f}"
        f">thr={SPIKE_REVERSAL_THRESHOLD:.2f}"
    )

    # 補助シグナル1: persistence 不足
    if persist.persistence_ratio < SPIKE_PERSISTENCE_CUTOFF:
        persist_fail = _clamp01(
            (SPIKE_PERSISTENCE_CUTOFF - persist.persistence_ratio) / SPIKE_PERSISTENCE_CUTOFF
        )
        reasons.append(f"persist={persist.persistence_ratio:.2f}<{SPIKE_PERSISTENCE_CUTOFF:.2f}")
    else:
        persist_fail = 0.0

    # 補助シグナル2: 出来高急落
    if rvol.rvol_acceleration < SPIKE_RVOL_ACCEL_CUTOFF:
        rvol_fail = _clamp01(
            (SPIKE_RVOL_ACCEL_CUTOFF - rvol.rvol_acceleration) / SPIKE_RVOL_ACCEL_CUTOFF
        )
        reasons.append(f"rvol_accel={rvol.rvol_acceleration:.2f}<{SPIKE_RVOL_ACCEL_CUTOFF:.2f}")
    else:
        rvol_fail = 0.0

    raw = (
        W_SPIKE_REVERSAL * reversal_signal
        + W_SPIKE_PERSIST * persist_fail
        + W_SPIKE_RVOL    * rvol_fail
    )
    confidence = _clamp01(raw)
    return confidence, reasons


def _classify(
    drive: OpeningDriveResult,
    rvol: RVOLResult,
    persist: PersistenceResult,
    rs: RelativeStrengthResult,
    gap: GapResult,
    spike_confidence: float,
    minutes_elapsed: int,
) -> tuple[str, list[str]]:
    """
    expansion_type と continuation_reasons を返す。

    is_spike:
      - spike_confidence > SPIKE_CONFIDENCE_THRESHOLD
      - AND minutes_elapsed >= MIN_BARS_FOR_VERDICT
      - AND NOT new_high_confirmed (呼び出し元で処理済み)

    is_continuation (AND):
      - drive_efficiency > CONT_DRIVE_MIN
      - persistence_ratio > CONT_PERSISTENCE_MIN
      - rvol_acceleration > CONT_RVOL_ACCEL_MIN
      OR new_high_confirmed alone
    """
    reasons: list[str] = []

    # Continuation check
    cont_drive   = drive.efficiency > CONT_DRIVE_MIN
    cont_persist = persist.persistence_ratio > CONT_PERSISTENCE_MIN
    cont_rvol    = rvol.rvol_acceleration > CONT_RVOL_ACCEL_MIN

    if drive.new_high_confirmed:
        reasons.append("new_high_above_early_peak")
        return "continuation", reasons

    if cont_drive:
        reasons.append(f"drive={drive.efficiency:.2f}>{CONT_DRIVE_MIN:.2f}")
    if cont_persist:
        reasons.append(f"persist={persist.persistence_ratio:.2f}>{CONT_PERSISTENCE_MIN:.2f}")
    if cont_rvol:
        reasons.append(f"rvol_accel={rvol.rvol_acceleration:.2f}>{CONT_RVOL_ACCEL_MIN:.2f}")

    is_continuation = cont_drive and cont_persist and cont_rvol

    is_spike = (
        minutes_elapsed >= MIN_BARS_FOR_VERDICT
        and spike_confidence > SPIKE_CONFIDENCE_THRESHOLD
    )

    if is_spike and not is_continuation:
        return "spike", []

    if is_continuation and not is_spike:
        return "continuation", reasons

    return "undetermined", reasons


def _expansion_probability(
    drive: OpeningDriveResult,
    rvol: RVOLResult,
    persist: PersistenceResult,
    rs: RelativeStrengthResult,
    gap: GapResult,
    spike_confidence: float,
) -> float:
    """
    継続確率 [0,1]。

    各コンポーネントを [0,1] にマップして加重平均。
    spike_confidence が高いほど強制的に抑制。
    """
    drive_s   = _clamp01(drive.efficiency)
    persist_s = _clamp01(persist.persistence_ratio)
    rvol_s    = _clamp01(rvol.rvol_acceleration / 2.0)  # 2x = 満点
    rs_s      = _clamp01(rs.rs_percentile / 100.0)
    gap_s     = _clamp01((gap.gap_acceptance + 1.0) / 2.0) if gap.gap_active else 0.5

    raw = (
        W_DRIVE      * drive_s
        + W_PERSIST  * persist_s
        + W_RVOL_ACCEL * rvol_s
        + W_RS       * rs_s
        + W_GAP      * gap_s
    )
    suppressed = raw * (1.0 - SPIKE_SUPPRESSION * spike_confidence)
    return round(_clamp01(suppressed), 4)


# ─────────────────────────────────────────────────────────────────────────────
# Safe default
# ─────────────────────────────────────────────────────────────────────────────

def _safe_default(symbol: str, eval_ts: str, n_bars: int) -> ExpansionState:
    return ExpansionState(
        symbol=symbol,
        eval_ts=eval_ts,
        minutes_elapsed=n_bars,
        drive_efficiency=0.5,
        rvol_ratio=1.0,
        rvol_acceleration=1.0,
        persistence_ratio=0.5,
        rs_percentile=50.0,
        gap_acceptance=0.5,
        expansion_probability=0.5,
        spike_confidence=0.0,
        expansion_type="undetermined",
        is_spike_pattern=False,
        new_high_confirmed=False,
        early_peak=0.0,
        early_reversal_pct=0.0,
        or_high=0.0,
        or_low=0.0,
        spike_reasons=[],
        continuation_reasons=["safe_default"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_expansion_state(
    symbol: str,
    minute_bars: list[MinuteBar],
    prev_close: float,
    universe_intraday_returns: dict[str, float],
    eval_ts: str,
    *,
    avg_daily_volume: int = 0,
    opening_range_minutes: int = 15,
    spike_detection_window: int = SPIKE_EARLY_WINDOW,
) -> ExpansionState:
    """
    intraday expansion state を計算する。

    Args:
        symbol:                    銘柄コード
        minute_bars:               09:00以降の1分足バー (昇順)。欠けていても可。
        prev_close:                前日終値
        universe_intraday_returns: {symbol: 当日騰落率} クロスセクションRS計算用
        eval_ts:                   評価タイムスタンプ (呼び出し元提供 = 決定論的)
        avg_daily_volume:          日次平均出来高 (0 = 不明 → rvol_ratio=1.0)
        opening_range_minutes:     Opening Range 窓幅 (分)
        spike_detection_window:    early_peak 検出窓幅 (本)

    Returns:
        ExpansionState — 例外時はsafe default を返す (fail-safe)。
    """
    try:
        return _inner(
            symbol=symbol,
            minute_bars=minute_bars,
            prev_close=prev_close,
            universe_intraday_returns=universe_intraday_returns,
            eval_ts=eval_ts,
            avg_daily_volume=avg_daily_volume,
            opening_range_minutes=opening_range_minutes,
            spike_detection_window=spike_detection_window,
        )
    except Exception as exc:
        logger.error("[EXPANSION_STATE] %s compute error: %s", symbol, exc)
        return _safe_default(symbol, eval_ts, len(minute_bars))


def _inner(
    symbol: str,
    minute_bars: list[MinuteBar],
    prev_close: float,
    universe_intraday_returns: dict[str, float],
    eval_ts: str,
    avg_daily_volume: int,
    opening_range_minutes: int,
    spike_detection_window: int,
) -> ExpansionState:
    bars = _validate_bars(minute_bars)
    n    = len(bars)

    if n == 0 or prev_close <= 0:
        return _safe_default(symbol, eval_ts, n)

    open_price    = _float_safe(bars[0].open)
    current_close = _float_safe(bars[-1].close)

    if open_price <= 0:
        return _safe_default(symbol, eval_ts, n)

    # Component computations
    drive   = compute_opening_drive(bars, open_price, spike_window=spike_detection_window)
    rvol    = compute_rvol(bars, avg_daily_volume)
    persist = compute_persistence(bars, opening_range_minutes)
    rs      = compute_rs(symbol, universe_intraday_returns)
    gap     = compute_gap_acceptance(open_price, _float_safe(prev_close), current_close)

    # Spike confidence
    if drive.new_high_confirmed or n < MIN_BARS_FOR_VERDICT:
        spike_conf   = 0.0
        spike_reasons: list[str] = []
    else:
        spike_conf, spike_reasons = _compute_spike_confidence(drive, rvol, persist)

    # Classification
    exp_type, cont_reasons = _classify(drive, rvol, persist, rs, gap, spike_conf, n)
    is_spike = exp_type == "spike"

    # Expansion probability (spike-suppressed)
    exp_prob = _expansion_probability(drive, rvol, persist, rs, gap, spike_conf)

    return ExpansionState(
        symbol=symbol,
        eval_ts=eval_ts,
        minutes_elapsed=n,
        drive_efficiency=round(drive.efficiency, 4),
        rvol_ratio=round(rvol.rvol_ratio, 4),
        rvol_acceleration=round(rvol.rvol_acceleration, 4),
        persistence_ratio=round(persist.persistence_ratio, 4),
        rs_percentile=rs.rs_percentile,
        gap_acceptance=round(gap.gap_acceptance, 4),
        expansion_probability=exp_prob,
        spike_confidence=round(spike_conf, 4),
        expansion_type=exp_type,
        is_spike_pattern=is_spike,
        new_high_confirmed=drive.new_high_confirmed,
        early_peak=round(drive.early_peak, 4),
        early_reversal_pct=round(drive.early_reversal_pct, 4),
        or_high=round(persist.or_high, 4),
        or_low=round(persist.or_low, 4),
        spike_reasons=spike_reasons,
        continuation_reasons=cont_reasons,
    )
