"""
predictive_expansion_report.py — Predictive Expansion Analytics Report

shadow_entries.jsonl + candidate_rankings.jsonl を読み込み、
週次/日次レポートを生成する。

出力セクション:
  1. Score Decile Performance    — 0-9, 10-19, ..., 90-100 別の forward return 実績
  2. Continuation by Bucket      — STRONG/MODERATE/WEAK/BLOCKED 別の継続確率
  3. False Expansion Analysis    — is_false_expansion 率 + score 相関
  4. Skipped High-Score          — score>=60 で slot/capital/risk blocked の機会損失
  5. Top Predictive Leaders      — 銘柄別 avg_score + sector_ignition + continuation

設計原則:
  - deterministic: 同一入力 → 同一出力
  - fail-open: データ不足でも空レポートを返す (I/O 失敗でも例外を上げない)
  - append-only JSONL summary (既存ファイルを破壊しない)
  - no ML, no live order routing
"""
from __future__ import annotations

import json
import logging
import statistics
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
JST = timezone(timedelta(hours=9))
_LOCK = threading.Lock()

# ── レポート定数 ──────────────────────────────────────────────────────────────
HIGH_SCORE_SKIP_THRESHOLD: int = 60   # この値以上を「high-score skipped」とみなす
N_TOP_LEADERS: int = 10               # top leader 表示数
SCORE_LABEL_STRONG: int = 70
SCORE_LABEL_MODERATE: int = 45
SCORE_LABEL_WEAK: int = 20

# decile 幅
DECILE_WIDTH: int = 10   # 0-9, 10-19, ..., 90-100

# ── Skip reason — slot/capital/risk blocked (high-value skips) ────────────────
_SKIP_BLOCKED_REASONS: frozenset[str] = frozenset({
    "slot_blocked",
    "capital_blocked",
    "risk_blocked",
    "spread_blocked",
    "cooldown_blocked",
    "persistence_short",
})


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DecilePerformance:
    """1 decile (score 10点幅) のパフォーマンス統計。"""
    decile: int                             # 1-10 (1=score 0-9, 10=score 90-100)
    score_low: int                          # スコア下限
    score_high: int                         # スコア上限 (inclusive)
    n_entries: int                          # サンプル数
    avg_score: float                        # 平均スコア
    avg_forward_5m: Optional[float]         # 平均 forward return 5分 (None=データなし)
    avg_forward_15m: Optional[float]
    avg_forward_30m: Optional[float]
    avg_forward_eod: Optional[float]
    continuation_rate: Optional[float]      # breakout_continued 率
    false_expansion_rate: Optional[float]   # is_false_expansion 率

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BucketStats:
    """score_label 別の継続確率統計。"""
    label: str                    # STRONG / MODERATE / WEAK / BLOCKED / NO_VERDICT
    n_entries: int
    avg_score: float
    continuation_rate: Optional[float]
    false_expansion_rate: Optional[float]
    avg_forward_eod: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FalseExpansionStats:
    """False expansion 分析結果。"""
    n_total_obs: int
    n_false_expansion: int
    false_expansion_rate: Optional[float]
    by_score_range: list[dict]            # [{"range": "0-29", "n": .., "rate": ..}, ...]
    by_expansion_type: dict[str, float]   # {"continuation": 0.15, "spike": 0.80, ...}
    high_confidence_false_rate: Optional[float]  # spike_confidence > 0.7 の false rate

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class SkippedHighScoreEntry:
    """score>=60 でブロックされた shadow entry (機会損失)。"""
    symbol: str
    entry_ts: str
    predictive_score: int
    skip_reasons: list[str]
    expansion_type: str
    spike_confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TopPredictiveLeader:
    """銘柄別の predictive expansion 統計。"""
    symbol: str
    n_observations: int
    avg_score: float
    sector_ignition_rate: float           # sector_igniting==True の割合
    continuation_rate: Optional[float]
    false_expansion_rate: Optional[float]
    avg_forward_eod: Optional[float]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PredictiveExpansionReport:
    """generate_report() の完全な出力。"""
    report_date: str
    report_id: str
    generated_at: str
    n_shadow_entries: int
    n_forward_obs: int
    n_ranking_records: int
    decile_performance: list[DecilePerformance]
    continuation_by_bucket: list[BucketStats]
    false_expansion: FalseExpansionStats
    skipped_high_score: list[SkippedHighScoreEntry]
    top_leaders: list[TopPredictiveLeader]
    summary_lines: list[str]

    def to_dict(self) -> dict:
        return {
            "report_date":           self.report_date,
            "report_id":             self.report_id,
            "generated_at":          self.generated_at,
            "n_shadow_entries":      self.n_shadow_entries,
            "n_forward_obs":         self.n_forward_obs,
            "n_ranking_records":     self.n_ranking_records,
            "decile_performance":    [d.to_dict() for d in self.decile_performance],
            "continuation_by_bucket": [b.to_dict() for b in self.continuation_by_bucket],
            "false_expansion":       self.false_expansion.to_dict(),
            "skipped_high_score":    [s.to_dict() for s in self.skipped_high_score],
            "top_leaders":           [t.to_dict() for t in self.top_leaders],
            "summary_lines":         self.summary_lines,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        import math
        return v if math.isfinite(v) else default
    except Exception:
        return default


def _avg(values: list[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _rate(count: int, total: int) -> Optional[float]:
    return round(count / total, 4) if total > 0 else None


def _load_jsonl(path: Path) -> list[dict]:
    """crash-safe JSONL reader (最終行が壊れていても継続)。"""
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
                    logger.debug("[PRED_REPORT] malformed line skipped")
    except Exception as exc:
        logger.warning("[PRED_REPORT] JSONL read failed: %s", exc)
    return records


def _save_jsonl_line(record: dict, path: Path) -> None:
    """1行を JSONL に追記する (fail-open)。"""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with _LOCK:
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line)
                fh.flush()
    except Exception as exc:
        logger.warning("[PRED_REPORT] JSONL append failed: %s", exc)


def _score_to_decile(score: int) -> int:
    """score → decile (1-10)。score=100 は decile=10。"""
    return min(max(score, 0) // DECILE_WIDTH + 1, 10)


def _score_to_label(score: int) -> str:
    if score < 0:
        return "NO_VERDICT"
    if score == 0:
        return "BLOCKED"
    if score < SCORE_LABEL_WEAK:
        return "WEAK"
    if score < SCORE_LABEL_MODERATE:
        return "WEAK"
    if score < SCORE_LABEL_STRONG:
        return "MODERATE"
    return "STRONG"


# ─────────────────────────────────────────────────────────────────────────────
# Reporter
# ─────────────────────────────────────────────────────────────────────────────

class PredictiveExpansionReporter:
    """
    SHADOW_ENTRY_LOG_FILE + PREDICTIVE_CANDIDATE_LOG_FILE を読み込み
    週次/日次 predictive expansion レポートを生成する。
    """

    # ── データ読み込み ────────────────────────────────────────────────────────

    def _load_shadow_records(
        self, path: Path
    ) -> tuple[list[dict], list[dict]]:
        """JSONL から ENTRY と FORWARD_OBS を分離して返す。"""
        all_records = _load_jsonl(path)
        entries: list[dict] = []
        fwd_obs: list[dict] = []
        for r in all_records:
            rt = r.get("record_type", "")
            if rt == "ENTRY":
                entries.append(r)
            elif rt == "FORWARD_OBS":
                fwd_obs.append(r)
        return entries, fwd_obs

    def _build_obs_map(self, fwd_obs: list[dict]) -> dict[tuple[str, str], dict]:
        """(symbol, entry_ts) → ForwardObsRecord dict のマップを構築する。"""
        obs_map: dict[tuple[str, str], dict] = {}
        for obs in fwd_obs:
            key = (obs.get("symbol", ""), obs.get("entry_ts", ""))
            if key[0] and key[1]:
                obs_map[key] = obs
        return obs_map

    # ── Section 1: Decile Performance ─────────────────────────────────────────

    def _compute_decile_performance(
        self,
        entries: list[dict],
        obs_map: dict[tuple[str, str], dict],
    ) -> list[DecilePerformance]:
        """score decile 別の forward return 統計を計算する。"""
        buckets: dict[int, list] = {d: [] for d in range(1, 11)}

        for e in entries:
            score = int(e.get("predictive_score", -1))
            if score < 0:
                continue
            decile = _score_to_decile(score)
            obs = obs_map.get((e.get("symbol", ""), e.get("entry_ts", "")))
            buckets[decile].append((e, obs))

        result: list[DecilePerformance] = []
        for d in range(1, 11):
            items = buckets[d]
            score_low = (d - 1) * DECILE_WIDTH
            score_high = d * DECILE_WIDTH - 1 if d < 10 else 100
            if not items:
                result.append(DecilePerformance(
                    decile=d, score_low=score_low, score_high=score_high,
                    n_entries=0, avg_score=0.0,
                    avg_forward_5m=None, avg_forward_15m=None,
                    avg_forward_30m=None, avg_forward_eod=None,
                    continuation_rate=None, false_expansion_rate=None,
                ))
                continue

            scores = [int(e.get("predictive_score", 0)) for e, _ in items]
            fwd5  = [_safe_float(obs["forward_5m"])  for _, obs in items if obs and obs.get("forward_5m")  is not None]
            fwd15 = [_safe_float(obs["forward_15m"]) for _, obs in items if obs and obs.get("forward_15m") is not None]
            fwd30 = [_safe_float(obs["forward_30m"]) for _, obs in items if obs and obs.get("forward_30m") is not None]
            fwdeod= [_safe_float(obs["forward_eod"]) for _, obs in items if obs and obs.get("forward_eod") is not None]

            cont_vals = [obs["breakout_continued"]   for _, obs in items if obs and obs.get("breakout_continued")   is not None]
            fexp_vals = [obs["is_false_expansion"]   for _, obs in items if obs and obs.get("is_false_expansion")   is not None]

            n_cont = sum(1 for v in cont_vals if v is True)
            n_fexp = sum(1 for v in fexp_vals if v is True)

            result.append(DecilePerformance(
                decile=d,
                score_low=score_low,
                score_high=score_high,
                n_entries=len(items),
                avg_score=round(statistics.mean(scores), 1),
                avg_forward_5m=round(_avg(fwd5), 5) if fwd5 else None,
                avg_forward_15m=round(_avg(fwd15), 5) if fwd15 else None,
                avg_forward_30m=round(_avg(fwd30), 5) if fwd30 else None,
                avg_forward_eod=round(_avg(fwdeod), 5) if fwdeod else None,
                continuation_rate=_rate(n_cont, len(cont_vals)),
                false_expansion_rate=_rate(n_fexp, len(fexp_vals)),
            ))
        return result

    # ── Section 2: Continuation by Bucket ────────────────────────────────────

    def _compute_continuation_by_bucket(
        self,
        entries: list[dict],
        obs_map: dict[tuple[str, str], dict],
    ) -> list[BucketStats]:
        """score_label 別の継続確率を計算する。"""
        buckets: dict[str, list] = {
            "STRONG": [], "MODERATE": [], "WEAK": [], "BLOCKED": [], "NO_VERDICT": [],
        }
        for e in entries:
            score = int(e.get("predictive_score", -1))
            label = _score_to_label(score)
            obs = obs_map.get((e.get("symbol", ""), e.get("entry_ts", "")))
            buckets[label].append((e, obs))

        result: list[BucketStats] = []
        for label in ("STRONG", "MODERATE", "WEAK", "BLOCKED", "NO_VERDICT"):
            items = buckets[label]
            if not items:
                result.append(BucketStats(
                    label=label, n_entries=0, avg_score=0.0,
                    continuation_rate=None, false_expansion_rate=None,
                    avg_forward_eod=None,
                ))
                continue

            scores  = [max(0, int(e.get("predictive_score", 0))) for e, _ in items]
            fwdeod  = [_safe_float(obs["forward_eod"]) for _, obs in items if obs and obs.get("forward_eod") is not None]
            cont_vals = [obs["breakout_continued"] for _, obs in items if obs and obs.get("breakout_continued") is not None]
            fexp_vals = [obs["is_false_expansion"]  for _, obs in items if obs and obs.get("is_false_expansion")  is not None]

            n_cont = sum(1 for v in cont_vals if v is True)
            n_fexp = sum(1 for v in fexp_vals if v is True)

            result.append(BucketStats(
                label=label,
                n_entries=len(items),
                avg_score=round(statistics.mean(scores), 1),
                continuation_rate=_rate(n_cont, len(cont_vals)),
                false_expansion_rate=_rate(n_fexp, len(fexp_vals)),
                avg_forward_eod=round(_avg(fwdeod), 5) if fwdeod else None,
            ))
        return result

    # ── Section 3: False Expansion Analysis ──────────────────────────────────

    def _compute_false_expansion(
        self,
        entries: list[dict],
        obs_map: dict[tuple[str, str], dict],
    ) -> FalseExpansionStats:
        """全体 + score レンジ別の false expansion 率を計算する。"""
        total_obs  = 0
        total_fexp = 0

        score_ranges = [
            ("0-29",  0,  29),
            ("30-59", 30, 59),
            ("60-79", 60, 79),
            ("80-100",80, 100),
        ]
        range_counts: dict[str, dict] = {r[0]: {"n": 0, "false": 0} for r in score_ranges}

        by_type: dict[str, list] = {}
        high_conf_total = 0
        high_conf_false = 0

        for e in entries:
            obs = obs_map.get((e.get("symbol", ""), e.get("entry_ts", "")))
            if obs is None or obs.get("is_false_expansion") is None:
                continue
            total_obs += 1
            is_fe = bool(obs["is_false_expansion"])
            if is_fe:
                total_fexp += 1

            score = int(e.get("predictive_score", -1))
            for rname, rlo, rhi in score_ranges:
                if rlo <= score <= rhi:
                    range_counts[rname]["n"] += 1
                    if is_fe:
                        range_counts[rname]["false"] += 1

            etype = e.get("expansion_type", "undetermined")
            if etype not in by_type:
                by_type[etype] = []
            by_type[etype].append(is_fe)

            spike_conf = _safe_float(e.get("spike_confidence", 0.0))
            if spike_conf > 0.7:
                high_conf_total += 1
                if is_fe:
                    high_conf_false += 1

        by_score_range = [
            {
                "range": rname,
                "n": range_counts[rname]["n"],
                "rate": _rate(range_counts[rname]["false"], range_counts[rname]["n"]),
            }
            for rname, _, _ in score_ranges
        ]

        by_type_rates: dict[str, float] = {}
        for etype, vals in by_type.items():
            n_fe = sum(1 for v in vals if v)
            by_type_rates[etype] = round(n_fe / len(vals), 4) if vals else 0.0

        return FalseExpansionStats(
            n_total_obs=total_obs,
            n_false_expansion=total_fexp,
            false_expansion_rate=_rate(total_fexp, total_obs),
            by_score_range=by_score_range,
            by_expansion_type=by_type_rates,
            high_confidence_false_rate=_rate(high_conf_false, high_conf_total),
        )

    # ── Section 4: Skipped High-Score ─────────────────────────────────────────

    def _find_skipped_high_score(
        self,
        entries: list[dict],
        min_score: int = HIGH_SCORE_SKIP_THRESHOLD,
    ) -> list[SkippedHighScoreEntry]:
        """score>=min_score かつ slot/capital/risk でブロックされた shadow entries。"""
        result: list[SkippedHighScoreEntry] = []
        for e in entries:
            score = int(e.get("predictive_score", -1))
            if score < min_score:
                continue
            skip_reasons: list[str] = e.get("skip_reasons", [])
            if not skip_reasons:
                continue
            blocked = [r for r in skip_reasons if r in _SKIP_BLOCKED_REASONS]
            if not blocked:
                continue
            result.append(SkippedHighScoreEntry(
                symbol=e.get("symbol", ""),
                entry_ts=e.get("entry_ts", ""),
                predictive_score=score,
                skip_reasons=blocked,
                expansion_type=e.get("expansion_type", "undetermined"),
                spike_confidence=_safe_float(e.get("spike_confidence", 0.0)),
            ))
        result.sort(key=lambda x: (-x.predictive_score, x.entry_ts))
        return result

    # ── Section 5: Top Predictive Leaders ────────────────────────────────────

    def _compute_top_leaders(
        self,
        entries: list[dict],
        obs_map: dict[tuple[str, str], dict],
        n_top: int = N_TOP_LEADERS,
    ) -> list[TopPredictiveLeader]:
        """銘柄別集計 → avg_score 降順の上位 n_top。"""
        sym_data: dict[str, dict] = {}
        for e in entries:
            sym = e.get("symbol", "")
            if not sym:
                continue
            if sym not in sym_data:
                sym_data[sym] = {
                    "scores": [], "igniting": [], "cont": [], "fexp": [], "fwdeod": [],
                }
            score = int(e.get("predictive_score", -1))
            if score >= 0:
                sym_data[sym]["scores"].append(score)

            # sector_igniting is stored in the entry as part of expansion_state context
            # (it may not always be present; treat absence as False)
            sym_data[sym]["igniting"].append(bool(e.get("sector_igniting", False)))

            obs = obs_map.get((sym, e.get("entry_ts", "")))
            if obs:
                if obs.get("breakout_continued") is not None:
                    sym_data[sym]["cont"].append(bool(obs["breakout_continued"]))
                if obs.get("is_false_expansion") is not None:
                    sym_data[sym]["fexp"].append(bool(obs["is_false_expansion"]))
                if obs.get("forward_eod") is not None:
                    sym_data[sym]["fwdeod"].append(_safe_float(obs["forward_eod"]))

        leaders: list[TopPredictiveLeader] = []
        for sym, d in sym_data.items():
            if not d["scores"]:
                continue
            n_ig = sum(1 for v in d["igniting"] if v)
            n_ct = sum(1 for v in d["cont"] if v)
            n_fe = sum(1 for v in d["fexp"] if v)
            leaders.append(TopPredictiveLeader(
                symbol=sym,
                n_observations=len(d["scores"]),
                avg_score=round(statistics.mean(d["scores"]), 1),
                sector_ignition_rate=round(n_ig / len(d["igniting"]), 4) if d["igniting"] else 0.0,
                continuation_rate=_rate(n_ct, len(d["cont"])),
                false_expansion_rate=_rate(n_fe, len(d["fexp"])),
                avg_forward_eod=round(_avg(d["fwdeod"]), 5) if d["fwdeod"] else None,
            ))

        leaders.sort(key=lambda x: (-x.avg_score, x.symbol))
        return leaders[:n_top]

    # ── Markdown formatter ────────────────────────────────────────────────────

    def format_report_markdown(self, report: PredictiveExpansionReport) -> str:
        lines: list[str] = [
            f"# Predictive Expansion Report — {report.report_date}",
            f"生成: {report.generated_at}  |  ID: {report.report_id}",
            f"ShadowEntries: {report.n_shadow_entries}  |  ForwardObs: {report.n_forward_obs}"
            f"  |  RankingRecords: {report.n_ranking_records}",
            "",
        ]

        # ── Summary ──────────────────────────────────────────────────────────
        lines.append("## Summary")
        for s in report.summary_lines:
            lines.append(f"- {s}")
        lines.append("")

        # ── Section 1: Score Decile Performance ──────────────────────────────
        lines.append("## 1. Score Decile Performance")
        lines.append(
            f"{'Decile':<8} {'ScoreRange':<12} {'N':>5} {'AvgScore':>9}"
            f" {'5m%':>7} {'15m%':>7} {'30m%':>7} {'EOD%':>7}"
            f" {'ContR':>6} {'FExpR':>6}"
        )
        lines.append("-" * 80)
        for d in report.decile_performance:
            def _fmt(v: Optional[float]) -> str:
                return f"{v*100:+.2f}" if v is not None else "  N/A "

            lines.append(
                f"  D{d.decile:<6} {d.score_low:>3}-{d.score_high:<7} {d.n_entries:>5}"
                f" {d.avg_score:>9.1f}"
                f" {_fmt(d.avg_forward_5m):>7}"
                f" {_fmt(d.avg_forward_15m):>7}"
                f" {_fmt(d.avg_forward_30m):>7}"
                f" {_fmt(d.avg_forward_eod):>7}"
                f" {d.continuation_rate*100:>5.1f}%" if d.continuation_rate is not None
                else f"  D{d.decile:<6} {d.score_low:>3}-{d.score_high:<7} {d.n_entries:>5}"
                     f" {d.avg_score:>9.1f}"
                     f" {_fmt(d.avg_forward_5m):>7}"
                     f" {_fmt(d.avg_forward_15m):>7}"
                     f" {_fmt(d.avg_forward_30m):>7}"
                     f" {_fmt(d.avg_forward_eod):>7}"
                     f"   N/A "
            )
        lines.append("")

        # ── Section 2: Continuation by Bucket ────────────────────────────────
        lines.append("## 2. Continuation Probability by Score Bucket")
        lines.append(f"{'Label':<12} {'N':>5} {'AvgScore':>9} {'ContR':>7} {'FExpR':>7} {'EOD%':>8}")
        lines.append("-" * 55)
        for b in report.continuation_by_bucket:
            cont_s = f"{b.continuation_rate*100:.1f}%" if b.continuation_rate is not None else "  N/A "
            fexp_s = f"{b.false_expansion_rate*100:.1f}%" if b.false_expansion_rate is not None else "  N/A "
            eod_s  = f"{b.avg_forward_eod*100:+.2f}%" if b.avg_forward_eod is not None else "  N/A "
            lines.append(
                f"  {b.label:<10} {b.n_entries:>5} {b.avg_score:>9.1f}"
                f" {cont_s:>7} {fexp_s:>7} {eod_s:>8}"
            )
        lines.append("")

        # ── Section 3: False Expansion Analysis ──────────────────────────────
        fe = report.false_expansion
        lines.append("## 3. False Expansion Analysis")
        fe_rate_s = f"{fe.false_expansion_rate*100:.1f}%" if fe.false_expansion_rate is not None else "N/A"
        hc_s = (f"{fe.high_confidence_false_rate*100:.1f}%"
                if fe.high_confidence_false_rate is not None else "N/A")
        lines.append(f"全体偽陽性率: {fe_rate_s}  (N={fe.n_total_obs}, FE={fe.n_false_expansion})")
        lines.append(f"spike_confidence>0.7 偽陽性率: {hc_s}")
        lines.append("")
        lines.append("Scoreレンジ別:")
        for r in fe.by_score_range:
            rate_s = f"{r['rate']*100:.1f}%" if r["rate"] is not None else "N/A"
            lines.append(f"  {r['range']:<8}: N={r['n']:>4}  rate={rate_s}")
        lines.append("")
        if fe.by_expansion_type:
            lines.append("expansion_type別偽陽性率:")
            for etype, rate in sorted(fe.by_expansion_type.items()):
                lines.append(f"  {etype:<15}: {rate*100:.1f}%")
        lines.append("")

        # ── Section 4: Skipped High-Score Opportunities ───────────────────────
        lines.append(f"## 4. Skipped High-Score Opportunities (score>={HIGH_SCORE_SKIP_THRESHOLD})")
        if not report.skipped_high_score:
            lines.append("  (なし)")
        else:
            lines.append(f"  {'銘柄':<10} {'Score':>6} {'TS':<22} {'SkipReasons':<30} {'ExpType'}")
            lines.append("  " + "-" * 80)
            for s in report.skipped_high_score[:20]:
                reasons_s = ",".join(s.skip_reasons)
                lines.append(
                    f"  {s.symbol:<10} {s.predictive_score:>6}"
                    f" {s.entry_ts:<22} {reasons_s:<30} {s.expansion_type}"
                )
        lines.append("")

        # ── Section 5: Top Predictive Leaders ────────────────────────────────
        lines.append("## 5. Top Predictive Leaders")
        lines.append(f"  {'銘柄':<10} {'N':>4} {'AvgScore':>9} {'IgnRate':>8} {'ContR':>7} {'FExpR':>7} {'EOD%':>8}")
        lines.append("  " + "-" * 65)
        for t in report.top_leaders:
            cont_s = f"{t.continuation_rate*100:.1f}%" if t.continuation_rate is not None else "  N/A "
            fexp_s = f"{t.false_expansion_rate*100:.1f}%" if t.false_expansion_rate is not None else "  N/A "
            eod_s  = f"{t.avg_forward_eod*100:+.2f}%" if t.avg_forward_eod is not None else "  N/A "
            lines.append(
                f"  {t.symbol:<10} {t.n_observations:>4} {t.avg_score:>9.1f}"
                f" {t.sector_ignition_rate*100:>7.1f}%"
                f" {cont_s:>7} {fexp_s:>7} {eod_s:>8}"
            )
        lines.append("")

        return "\n".join(lines)

    # ── サマリー生成 ──────────────────────────────────────────────────────────

    def _build_summary(
        self,
        entries: list[dict],
        fwd_obs: list[dict],
        ranking_records: list[dict],
        decile_perf: list[DecilePerformance],
        false_expansion: FalseExpansionStats,
        skipped: list[SkippedHighScoreEntry],
        leaders: list[TopPredictiveLeader],
    ) -> list[str]:
        lines: list[str] = []

        # 全体エントリー数
        lines.append(f"ShadowEntries: {len(entries)}件  ForwardObs: {len(fwd_obs)}件")

        # 最高スコア decile の継続率
        top_decile = next(
            (d for d in reversed(decile_perf) if d.n_entries > 0), None
        )
        if top_decile and top_decile.continuation_rate is not None:
            lines.append(
                f"D{top_decile.decile} (score {top_decile.score_low}-{top_decile.score_high}): "
                f"継続率 {top_decile.continuation_rate*100:.1f}%  N={top_decile.n_entries}"
            )

        # 偽陽性率
        if false_expansion.false_expansion_rate is not None:
            lines.append(
                f"全体偽陽性率: {false_expansion.false_expansion_rate*100:.1f}%"
                f"  (N={false_expansion.n_total_obs})"
            )

        # スキップ機会損失
        if skipped:
            lines.append(f"高スコアスキップ: {len(skipped)}件 (score>={HIGH_SCORE_SKIP_THRESHOLD})")

        # トップリーダー
        if leaders:
            top = leaders[0]
            lines.append(
                f"Top leader: {top.symbol} avg_score={top.avg_score:.1f}"
                f" N={top.n_observations}"
            )

        return lines

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_report(
        self,
        date: str,
        run_id: str = "",
        shadow_log_path: Optional[Path] = None,
        ranking_log_path: Optional[Path] = None,
        report_dir: Optional[Path] = None,
        report_jsonl_path: Optional[Path] = None,
    ) -> tuple[PredictiveExpansionReport, Optional[Path]]:
        """
        レポートを生成し、Markdown を report_dir に保存する。

        Args:
            date:              レポート対象日 "YYYY-MM-DD"
            run_id:            実行ID
            shadow_log_path:   shadow_entries.jsonl のパス
            ranking_log_path:  candidate_rankings.jsonl のパス
            report_dir:        Markdown 出力ディレクトリ
            report_jsonl_path: summary JSONL 追記先 (None でスキップ)

        Returns:
            (PredictiveExpansionReport, saved_path_or_None)
        """
        from src.paths import (
            SHADOW_ENTRY_LOG_FILE,
            PREDICTIVE_CANDIDATE_LOG_FILE,
            PREDICTIVE_REPORTS_DIR,
        )

        shadow_log_path  = shadow_log_path  or SHADOW_ENTRY_LOG_FILE
        ranking_log_path = ranking_log_path or PREDICTIVE_CANDIDATE_LOG_FILE
        report_dir       = report_dir       or PREDICTIVE_REPORTS_DIR

        generated_at = datetime.now(JST).strftime("%Y-%m-%dT%H:%M:%S%z")
        report_id    = run_id or generated_at

        entries, fwd_obs    = self._load_shadow_records(shadow_log_path)
        ranking_records     = _load_jsonl(ranking_log_path)
        obs_map             = self._build_obs_map(fwd_obs)

        decile_perf         = self._compute_decile_performance(entries, obs_map)
        cont_by_bucket      = self._compute_continuation_by_bucket(entries, obs_map)
        false_expansion     = self._compute_false_expansion(entries, obs_map)
        skipped             = self._find_skipped_high_score(entries)
        leaders             = self._compute_top_leaders(entries, obs_map)

        summary = self._build_summary(
            entries, fwd_obs, ranking_records,
            decile_perf, false_expansion, skipped, leaders,
        )

        report = PredictiveExpansionReport(
            report_date=date,
            report_id=report_id,
            generated_at=generated_at,
            n_shadow_entries=len(entries),
            n_forward_obs=len(fwd_obs),
            n_ranking_records=len(ranking_records),
            decile_performance=decile_perf,
            continuation_by_bucket=cont_by_bucket,
            false_expansion=false_expansion,
            skipped_high_score=skipped,
            top_leaders=leaders,
            summary_lines=summary,
        )

        # ── Markdown 保存 ──────────────────────────────────────────────────────
        saved_path: Optional[Path] = None
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            md_path = report_dir / f"predictive_expansion_{date}.md"
            md_content = self.format_report_markdown(report)
            tmp = md_path.with_suffix(".md.tmp")
            tmp.write_text(md_content, encoding="utf-8")
            tmp.replace(md_path)
            saved_path = md_path
            logger.info("[PRED_REPORT] saved: %s", md_path)
        except Exception as exc:
            logger.warning("[PRED_REPORT] Markdown save failed: %s", exc)

        # ── Summary JSONL 追記 ─────────────────────────────────────────────────
        if report_jsonl_path is not None:
            _save_jsonl_line(report.to_dict(), report_jsonl_path)

        return report, saved_path
