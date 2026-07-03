"""
predictive_entry_overlay.py — Shadow Predictive Entry Observation Layer

実注文ルーティングなし。
予測スコアが高い銘柄について「シャドウエントリー」を生成し、
実際の forward returns / MAE / MFE を観測する。

目的:
  - entry timing lag の定量化
  - false expansion (寄り天誤認) 率の継続的モニタリング
  - スキップ理由別のアルファ漏洩追跡

スキップ理由カテゴリ:
  slot_blocked       エントリーシグナルなし (メイン戦略がシグナルを出さなかった)
  capital_blocked    資金満杯
  risk_blocked       リスク上限
  spread_blocked     スプレッドが広い
  weak_score         predictive score < min_score_threshold
  cooldown_blocked   クールダウン期間中
  persistence_short  OR窓より前の評価 (bars 不足)

JSONL ログ (追記専用):
  record_type = "ENTRY"       : シャドウエントリー生成時
  record_type = "FORWARD_OBS" : forward return 観測時 (後続バーで更新)
  ← (symbol, entry_ts) でリンク

設計原則:
  - JSONL append-only: クラッシュセーフ
  - 全タイムスタンプは呼び出し元が提供 (決定論的)
  - fail-closed: JSONL 書き込み失敗時は WARNING ログのみ (実行は継続)
  - 統計計算は JSONL の全記録を読み直して算出 (状態ファイルなし)
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np

from src.intraday.expansion_state_engine import ExpansionState, MinuteBar, _float_safe

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()

# ── Skip reason constants ────────────────────────────────────────────────────
SKIP_SLOT_BLOCKED       = "slot_blocked"
SKIP_CAPITAL_BLOCKED    = "capital_blocked"
SKIP_RISK_BLOCKED       = "risk_blocked"
SKIP_SPREAD_BLOCKED     = "spread_blocked"
SKIP_WEAK_SCORE         = "weak_predictive_score"
SKIP_COOLDOWN_BLOCKED   = "cooldown_blocked"
SKIP_PERSISTENCE_SHORT  = "persistence_short"

ALL_SKIP_REASONS: frozenset[str] = frozenset({
    SKIP_SLOT_BLOCKED,
    SKIP_CAPITAL_BLOCKED,
    SKIP_RISK_BLOCKED,
    SKIP_SPREAD_BLOCKED,
    SKIP_WEAK_SCORE,
    SKIP_COOLDOWN_BLOCKED,
    SKIP_PERSISTENCE_SHORT,
})

# ── Forward observation thresholds ────────────────────────────────────────────
DEFAULT_MIN_SCORE: int          = 60    # このスコア以上のみ観測対象
DEFAULT_CONT_THRESHOLD: float   = 0.005 # forward +0.5% 以上 = 継続
DEFAULT_REVERSAL_THRESHOLD: float = 0.005  # forward -0.5% 以下 = 誤認

# ── Forward return offsets (bars from entry) ─────────────────────────────────
FORWARD_5M:  int = 5
FORWARD_15M: int = 15
FORWARD_30M: int = 30


# ─────────────────────────────────────────────────────────────────────────────
# Input dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShadowEntryContext:
    """
    PredictiveEntryOverlay.observe() への入力。1銘柄 × 1評価タイミング。
    """
    symbol: str
    eval_ts: str                        # 評価タイムスタンプ (呼び出し元提供)
    expansion_state: ExpansionState
    predictive_score: int               # 0-100; -1=判定不能
    entry_price: float                  # 現在値 (シャドウエントリー価格)
    skip_reasons: list[str]             # なぜ実エントリーしなかったか


# ─────────────────────────────────────────────────────────────────────────────
# JSONL record dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ShadowEntryRecord:
    """
    JSONL に書き込む ENTRY レコード。
    (symbol, entry_ts) がリンクキー。
    """
    record_type: str          # "ENTRY"
    symbol: str
    entry_ts: str
    entry_price: float
    predictive_score: int
    expansion_type: str
    expansion_probability: float
    spike_confidence: float
    drive_efficiency: float
    persistence_ratio: float
    rvol_acceleration: float
    rs_percentile: float
    gap_acceptance: float
    minutes_elapsed: int
    skip_reasons: list[str]
    run_ts: str               # このrunのタイムスタンプ


@dataclass
class ForwardObsRecord:
    """
    JSONL に書き込む FORWARD_OBS レコード。
    (symbol, entry_ts) で ENTRY レコードにリンク。
    """
    record_type: str          # "FORWARD_OBS"
    symbol: str
    entry_ts: str
    obs_ts: str               # 観測タイムスタンプ
    forward_5m: Optional[float]
    forward_15m: Optional[float]
    forward_30m: Optional[float]
    forward_eod: Optional[float]
    mae_pct: Optional[float]  # Max Adverse Excursion
    mfe_pct: Optional[float]  # Max Favorable Excursion
    breakout_continued: Optional[bool]   # forward_30m > cont_threshold
    is_false_expansion: Optional[bool]   # expansion_type="continuation" かつ実際は逆行


@dataclass
class ShadowStatistics:
    """compute_statistics() の出力。"""
    total_observed: int
    total_with_forward_obs: int
    breakout_continuation_probability: float   # P(continued | observed)
    false_expansion_failure_rate: float        # P(false|expansion_type=continuation)
    avg_forward_5m: Optional[float]
    avg_forward_15m: Optional[float]
    avg_forward_30m: Optional[float]
    skip_reason_counts: dict[str, int]
    score_distribution: dict[str, int]         # "STRONG"/"MODERATE"/"WEAK" 件数


# ─────────────────────────────────────────────────────────────────────────────
# JSONL I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _append_record(path: Path, record: dict) -> None:
    """JSONL に 1 行追記する。クラッシュセーフ (fail-open)。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
    except Exception as exc:
        logger.warning("[SHADOW_OVERLAY] JSONL append failed (%s): %s", path.name, exc)


def _load_all_records(path: Path) -> list[dict]:
    """
    JSONL を全行読み込む。
    末尾の不完全行はスキップする (クラッシュセーフ)。
    """
    if not path.exists():
        return []
    records: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("[SHADOW_OVERLAY] malformed line skipped")
    except Exception as exc:
        logger.warning("[SHADOW_OVERLAY] JSONL read failed: %s", exc)
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Forward observation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _forward_return_at(
    entry_price: float,
    bars: list[MinuteBar],
    offset_bars: int,
) -> Optional[float]:
    """
    entry_price から offset_bars 本後のclose騰落率を返す。
    バーが足りない場合は None。
    """
    if not bars or offset_bars <= 0 or len(bars) < offset_bars:
        return None
    target_bar = bars[offset_bars - 1]
    close = _float_safe(target_bar.close)
    if entry_price <= 0 or close <= 0:
        return None
    return round((close - entry_price) / entry_price, 6)


def _compute_mae_mfe(
    entry_price: float,
    bars: list[MinuteBar],
) -> tuple[Optional[float], Optional[float]]:
    """
    エントリー後バー群の MAE / MFE を計算する。
    MAE: エントリー価格からの最大不利方向 (正値、例: 0.01 = -1%)
    MFE: エントリー価格からの最大有利方向 (正値、例: 0.03 = +3%)
    """
    if not bars or entry_price <= 0:
        return None, None
    lows   = [_float_safe(b.low)  for b in bars if _float_safe(b.low)  > 0]
    highs  = [_float_safe(b.high) for b in bars if _float_safe(b.high) > 0]
    if not lows or not highs:
        return None, None
    mae = round((entry_price - min(lows))  / entry_price, 6)
    mfe = round((max(highs)  - entry_price) / entry_price, 6)
    return max(0.0, mae), max(0.0, mfe)


# ─────────────────────────────────────────────────────────────────────────────
# PredictiveEntryOverlay
# ─────────────────────────────────────────────────────────────────────────────

class PredictiveEntryOverlay:
    """
    シャドウエントリーを観測・記録するオーバーレイレイヤー。

    実注文ルーティングは一切行わない。
    全ての書き込みは JSONL への追記のみ (状態ファイルなし)。
    """

    def __init__(
        self,
        log_path: Path,
        min_score_threshold: int = DEFAULT_MIN_SCORE,
        continuation_threshold: float = DEFAULT_CONT_THRESHOLD,
        reversal_threshold: float = DEFAULT_REVERSAL_THRESHOLD,
    ) -> None:
        self.log_path             = log_path
        self.min_score_threshold  = min_score_threshold
        self.cont_threshold       = continuation_threshold
        self.reversal_threshold   = reversal_threshold

    # ── Public API ────────────────────────────────────────────────────────────

    def observe(
        self,
        contexts: list[ShadowEntryContext],
        run_ts: str,
    ) -> list[ShadowEntryRecord]:
        """
        評価タイミングでシャドウエントリーを生成・記録する。

        - predictive_score >= min_score_threshold の銘柄のみ記録対象
        - WEAK_SCORE はスキップ理由として記録される
        - 実注文ルーティングは行わない

        Args:
            contexts: 評価対象銘柄リスト
            run_ts:   このrunのタイムスタンプ (呼び出し元提供)

        Returns:
            ログに書き込んだ ShadowEntryRecord のリスト
        """
        logged: list[ShadowEntryRecord] = []

        for ctx in contexts:
            skip_reasons = list(ctx.skip_reasons)

            # スコア閾値チェック
            if ctx.predictive_score < self.min_score_threshold:
                if SKIP_WEAK_SCORE not in skip_reasons:
                    skip_reasons.append(SKIP_WEAK_SCORE)
                # 閾値未満はログしない (観測対象外)
                continue

            rec = ShadowEntryRecord(
                record_type="ENTRY",
                symbol=ctx.symbol,
                entry_ts=ctx.eval_ts,
                entry_price=round(_float_safe(ctx.entry_price), 4),
                predictive_score=ctx.predictive_score,
                expansion_type=ctx.expansion_state.expansion_type,
                expansion_probability=ctx.expansion_state.expansion_probability,
                spike_confidence=ctx.expansion_state.spike_confidence,
                drive_efficiency=ctx.expansion_state.drive_efficiency,
                persistence_ratio=ctx.expansion_state.persistence_ratio,
                rvol_acceleration=ctx.expansion_state.rvol_acceleration,
                rs_percentile=ctx.expansion_state.rs_percentile,
                gap_acceptance=ctx.expansion_state.gap_acceptance,
                minutes_elapsed=ctx.expansion_state.minutes_elapsed,
                skip_reasons=skip_reasons,
                run_ts=run_ts,
            )
            _append_record(self.log_path, asdict(rec))
            logged.append(rec)

        return logged

    def update_forward_observations(
        self,
        symbol: str,
        entry_ts: str,
        subsequent_bars: list[MinuteBar],
        entry_price: Optional[float] = None,
        eod_close: Optional[float] = None,
        obs_ts: str = "",
        expansion_type: str = "",
    ) -> Optional[ForwardObsRecord]:
        """
        過去のシャドウエントリーに forward returns を追記する。

        Args:
            symbol:          銘柄コード
            entry_ts:        対応する ENTRY レコードの entry_ts
            subsequent_bars: エントリー後の1分足バー (昇順)
            entry_price:     エントリー価格 (None → JSONL から検索)
            eod_close:       当日引け値 (None → 最終バーのcloseを使用)
            obs_ts:          観測タイムスタンプ (呼び出し元提供)
            expansion_type:  ENTRY 記録時の expansion_type (false expansion 判定用)

        Returns:
            書き込んだ ForwardObsRecord、または None (entry_price が不明時)
        """
        bars = [b for b in subsequent_bars if _float_safe(b.close) > 0]

        # entry_price が未提供なら JSONL から取得
        if entry_price is None or entry_price <= 0:
            entry_price = self._find_entry_price(symbol, entry_ts)
        if not entry_price or entry_price <= 0:
            logger.warning(
                "[SHADOW_OVERLAY] entry_price unknown: symbol=%s entry_ts=%s",
                symbol, entry_ts,
            )
            return None

        fwd5  = _forward_return_at(entry_price, bars, FORWARD_5M)
        fwd15 = _forward_return_at(entry_price, bars, FORWARD_15M)
        fwd30 = _forward_return_at(entry_price, bars, FORWARD_30M)

        # EOD return
        if eod_close is not None and eod_close > 0:
            fwd_eod = round((eod_close - entry_price) / entry_price, 6)
        elif bars:
            last_close = _float_safe(bars[-1].close)
            fwd_eod = round((last_close - entry_price) / entry_price, 6) if last_close > 0 else None
        else:
            fwd_eod = None

        mae, mfe = _compute_mae_mfe(entry_price, bars)

        # 継続 / 誤認判定
        breakout = None
        if fwd30 is not None:
            breakout = fwd30 > self.cont_threshold

        is_false = None
        if expansion_type == "continuation" and fwd30 is not None:
            is_false = fwd30 < -self.reversal_threshold

        obs = ForwardObsRecord(
            record_type="FORWARD_OBS",
            symbol=symbol,
            entry_ts=entry_ts,
            obs_ts=obs_ts,
            forward_5m=fwd5,
            forward_15m=fwd15,
            forward_30m=fwd30,
            forward_eod=fwd_eod,
            mae_pct=mae,
            mfe_pct=mfe,
            breakout_continued=breakout,
            is_false_expansion=is_false,
        )
        _append_record(self.log_path, asdict(obs))
        return obs

    def compute_statistics(self) -> ShadowStatistics:
        """
        JSONL 全記録から統計を計算する。

        breakout_continuation_probability = P(continued | has_forward_obs)
        false_expansion_failure_rate = P(is_false_expansion | expansion_type=continuation)
        """
        records = _load_all_records(self.log_path)

        entries:  dict[tuple[str, str], dict] = {}
        fwd_obs:  dict[tuple[str, str], dict] = {}

        for rec in records:
            rt  = rec.get("record_type", "")
            key = (rec.get("symbol", ""), rec.get("entry_ts", ""))
            if rt == "ENTRY":
                entries[key] = rec
            elif rt == "FORWARD_OBS":
                fwd_obs[key] = rec  # 最後の観測で上書き (最新を使用)

        total_obs = len(entries)
        total_fwd = sum(1 for k in entries if k in fwd_obs)

        # Breakout continuation probability
        cont_count = 0
        for key, obs in fwd_obs.items():
            if obs.get("breakout_continued") is True:
                cont_count += 1
        bc_prob = (cont_count / total_fwd) if total_fwd > 0 else 0.0

        # False expansion rate (continuation type only)
        cont_entries = [
            key for key, e in entries.items()
            if e.get("expansion_type") == "continuation"
        ]
        false_count = 0
        for key in cont_entries:
            obs = fwd_obs.get(key)
            if obs and obs.get("is_false_expansion") is True:
                false_count += 1
        false_rate = (false_count / len(cont_entries)) if cont_entries else 0.0

        # Forward return averages
        fwd5s  = [obs["forward_5m"]  for obs in fwd_obs.values() if obs.get("forward_5m")  is not None]
        fwd15s = [obs["forward_15m"] for obs in fwd_obs.values() if obs.get("forward_15m") is not None]
        fwd30s = [obs["forward_30m"] for obs in fwd_obs.values() if obs.get("forward_30m") is not None]

        avg5  = float(np.mean(fwd5s))  if fwd5s  else None
        avg15 = float(np.mean(fwd15s)) if fwd15s else None
        avg30 = float(np.mean(fwd30s)) if fwd30s else None

        # Skip reason distribution
        skip_counts: dict[str, int] = {r: 0 for r in ALL_SKIP_REASONS}
        for e in entries.values():
            for reason in e.get("skip_reasons", []):
                skip_counts[reason] = skip_counts.get(reason, 0) + 1

        # Score distribution
        score_dist: dict[str, int] = {"STRONG": 0, "MODERATE": 0, "WEAK": 0}
        for e in entries.values():
            sc = e.get("predictive_score", 0)
            if sc >= 70:
                score_dist["STRONG"] += 1
            elif sc >= 45:
                score_dist["MODERATE"] += 1
            else:
                score_dist["WEAK"] += 1

        return ShadowStatistics(
            total_observed=total_obs,
            total_with_forward_obs=total_fwd,
            breakout_continuation_probability=round(bc_prob, 4),
            false_expansion_failure_rate=round(false_rate, 4),
            avg_forward_5m=round(avg5, 6) if avg5 is not None else None,
            avg_forward_15m=round(avg15, 6) if avg15 is not None else None,
            avg_forward_30m=round(avg30, 6) if avg30 is not None else None,
            skip_reason_counts=skip_counts,
            score_distribution=score_dist,
        )

    def load_all_records(self) -> list[dict]:
        """JSONL の全記録を返す (テスト・デバッグ用)。"""
        return _load_all_records(self.log_path)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_entry_price(self, symbol: str, entry_ts: str) -> Optional[float]:
        """JSONL から (symbol, entry_ts) の entry_price を取得する。"""
        for rec in _load_all_records(self.log_path):
            if (
                rec.get("record_type") == "ENTRY"
                and rec.get("symbol") == symbol
                and rec.get("entry_ts") == entry_ts
            ):
                return _float_safe(rec.get("entry_price", 0.0)) or None
        return None
