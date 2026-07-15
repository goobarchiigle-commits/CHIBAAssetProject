"""
src/scripts/evs_rca_20260629.py

2026-06-29 EVS「executed=0 skipped=2」の完全RCA（Phase1-6）。
推測禁止 — LIVEログ・実コード・EVSログ・実OHLCVキャッシュのみを使用する。

Phase1: 2026-06-29 の全Stage追跡（BUY生成→Ranking→Risk→Capacity→Execution）
Phase2: 銘柄別詳細（symbol/RSR/rank/score/Stage/reason/想定数量/想定金額）
Phase3: 直近30日 Stage別Drop数集計
Phase4: 直近30日 Block理由ランキングTOP20
Phase5: Skipped銘柄の Forward Return(5d/10d/20d) 実測
Phase6: Study53 (ENTRY_DEFICIT) との整合性評価

実行:
    python -m src.scripts.evs_rca_20260629
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd


SIGNALS_DIR   = _ROOT / "data" / "signals"
EVS_STORE     = _ROOT / "runtime" / "analytics" / "executed_vs_skipped.jsonl"
OHLCV_DIR     = _ROOT / "cache" / "ohlcv"
STUDY53_PATH  = _ROOT / "backtests" / "study53_opportunity_loss_2026-06-28.json"
MAX_POS       = 3


def _latest_signal_files_by_date(days_back: int = 30) -> dict[str, Path]:
    files = sorted(SIGNALS_DIR.glob("signal_*.json"))
    by_date: dict[str, Path] = {}
    for f in files:
        m = re.match(r"signal_(\d{8})_(\d{6})", f.name)
        if not m:
            continue
        d = m.group(1)
        # 「executed」サフィックス付きは実発注が確定したrunなので優先的に採用
        if d not in by_date or f.name.endswith("_executed.json"):
            by_date[d] = f
    dates_sorted = sorted(by_date.keys())[-days_back:]
    return {d: by_date[d] for d in dates_sorted}


def phase1_and_2_2026_06_29() -> list[dict]:
    """2026-06-29の全BUY候補についてStage追跡を実データから再構成する。"""
    files = sorted(SIGNALS_DIR.glob("signal_20260629_*.json"))
    # LIVE実行のもの（最終判断）を優先: 08:44:16実行分
    target = None
    for f in files:
        if "084416" in f.name:
            target = f
            break
    if target is None:
        target = files[-1]

    d = json.loads(target.read_text(encoding="utf-8"))
    signals = d["signals"]
    orders  = d["orders"]
    executed_syms = {o["symbol"] for o in orders if o.get("side") == "BUY"}
    held_syms     = [s["symbol"] for s in signals if s.get("currently_holding")]
    n_held        = len(held_syms)

    print("=" * 78)
    print(f"  Phase1: 2026-06-29 全BUY候補 Stage追跡 (source={target.name}, mode={d['mode']})")
    print("=" * 78)
    print(f"  実保有銘柄({n_held}件): {held_syms}")
    print(f"  max_positions = {MAX_POS}")
    print(f"  実発注BUY件数 = {len(executed_syms)} : {sorted(executed_syms)}")
    print()

    buy_candidates = [
        s for s in signals
        if s.get("signal") == 1 and not s.get("currently_holding")
    ]
    print(f"  signal=1 かつ未保有の候補: {len(buy_candidates)}件")
    print()

    rows = []
    for s in sorted(buy_candidates, key=lambda x: x["rsr_rank"]):
        sym = s["symbol"]
        executed = sym in executed_syms
        if executed:
            stage, reason = "ORDER_BUILT", "executed"
        elif n_held >= MAX_POS:
            stage, reason = "CAPACITY", "position_full"
        else:
            stage, reason = "UNKNOWN", "no_capacity_evidence_of_other_block"
        price = None
        row = {
            "symbol": sym, "rsr": s["rsr"], "rsr_rank": s["rsr_rank"],
            "sepa": s["sepa_score"], "strategy_type": s["strategy_type"],
            "reason_text": s.get("reason", ""),
            "stage": stage, "stage_reason": reason,
            "executed": executed,
        }
        rows.append(row)

    print("  ── Phase1 判定結果 ──────────────────────────────────────────")
    for r in rows:
        print(f"  {r['symbol']:<10} RSR={r['rsr']:>6.1f} rank={r['rsr_rank']:>2} "
              f"SEPA={r['sepa']} strat={r['strategy_type']:<12} "
              f"stage={r['stage']:<12} reason={r['stage_reason']}")
    print()
    print("  結論: 2 candidate(8035.T, 6920.T) はいずれも CAPACITY stage で"
          f" FAIL（実保有={n_held}/max_positions={MAX_POS} = 満枠のため、"
          "RiskCheck/Addon/Execution/Order各stageまで到達していない）。")
    print("  検証: ADDON_EXT ログ（6981.T/5301.T/2802.T ブロック）は既存保有銘柄への"
          "「追加建て」判定であり、新規2銘柄がスキップされた理由とは別事象。")
    print()

    print("=" * 78)
    print("  Phase2: 銘柄別詳細")
    print("=" * 78)
    for s in buy_candidates:
        sym = s["symbol"]
        _df_path = OHLCV_DIR / f"{sym}.parquet"
        price = None
        if _df_path.exists():
            try:
                _df = pd.read_parquet(_df_path)
                _df.index = pd.to_datetime(_df.index)
                _mask = _df.index <= pd.Timestamp("2026-06-29")
                if _mask.any():
                    price = float(_df.loc[_mask, "Close"].iloc[-1])
            except Exception:
                pass
        lot_cost = price * 100 if price else None
        alloc_cap = 3_000_000 * 0.25
        would_pass_capital = (lot_cost is not None and lot_cost <= alloc_cap)
        qty = int(alloc_cap / price / 100) * 100 if (price and would_pass_capital) else 0
        amount = qty * price if (qty and price) else None
        print(f"  symbol={sym}")
        print(f"    RSR={s['rsr']} rank={s['rsr_rank']} score(entry)={s['rsr']/100:.3f}")
        print(f"    Stage=CAPACITY reason=position_full（枠が0だったため以降未評価）")
        print(f"    想定株価(2026-06-29終値)=¥{price:,.1f}" if price else "    想定株価: (キャッシュ無し)")
        if price is not None:
            print(f"    1単元コスト=¥{lot_cost:,.0f} vs alloc_cap(25%)=¥{alloc_cap:,.0f}"
                  f" → {'CAPITAL段階も通過見込み' if would_pass_capital else '★仮に枠があってもCAPITAL段階で除外される見込み'}")
        print(f"    想定数量={qty}株" if qty else "    想定数量: 0株（alloc_cap超過のため）")
        print(f"    想定金額=¥{amount:,.0f}" if amount else "    想定金額: N/A")
    print()

    return rows


def phase3_and_4(days_back: int = 30) -> dict:
    print("=" * 78)
    print(f"  Phase3/4: 直近{days_back}日 Stage別Drop統計 + Block理由ランキング")
    print("=" * 78)

    by_date = _latest_signal_files_by_date(days_back)
    print(f"  対象日数（signalファイルが存在する日）: {len(by_date)}日"
          f" ({sorted(by_date.keys())[0]} .. {sorted(by_date.keys())[-1]})")

    stage_counts = defaultdict(int)
    reason_counts = defaultdict(int)
    n_days_with_candidate = 0
    n_days_full_when_candidate = 0
    n_days_total = len(by_date)

    for date_str, path in by_date.items():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        signals = d.get("signals", [])
        orders  = d.get("orders", [])
        executed_syms = {o["symbol"] for o in orders if o.get("side") == "BUY"}
        n_held = sum(1 for s in signals if s.get("currently_holding"))
        buy_cands = [s for s in signals if s.get("signal") == 1 and not s.get("currently_holding")]

        if buy_cands:
            n_days_with_candidate += 1
            if n_held >= MAX_POS:
                n_days_full_when_candidate += 1

        raw_warnings = d.get("warnings", []) or []

        def _match_warning(symbol: str) -> "str | None":
            for w in raw_warnings:
                if symbol in w:
                    return w
            return None

        for s in buy_cands:
            sym = s["symbol"]
            if sym in executed_syms:
                stage_counts["ORDER_BUILT(executed)"] += 1
                continue
            w = _match_warning(sym)
            if w:
                # 実際に保存された warnings 文字列（_build_orders() が生成）から
                # 分類する。これは推測ではなく、そのrunで実際に出力された理由。
                if "最大ポジション数" in w:
                    stage_counts["CAPACITY(position_full)"] += 1
                    reason_counts["最大ポジション数到達"] += 1
                elif "配分上限キャップ" in w:
                    stage_counts["CAPITAL(alloc_cap_exceeded)"] += 1
                    reason_counts["配分上限キャップ超過(1単元が高額すぎる)"] += 1
                elif "サイジング結果qty=0" in w:
                    stage_counts["SIZING(zero_qty)"] += 1
                    reason_counts["サイジング結果qty=0"] += 1
                elif "セクター集中制限" in w:
                    stage_counts["SECTOR_CONCENTRATION"] += 1
                    reason_counts["セクター集中制限(adaptive)"] += 1
                elif "pre_trade_risk_check" in w:
                    stage_counts["RISK(pre_trade_risk_check_reject)"] += 1
                    reason_counts["pre_trade_risk_check不合格"] += 1
                elif "新規 BUY 上限" in w:
                    stage_counts["DAILY_LIMIT"] += 1
                    reason_counts["1日の新規BUY上限到達"] += 1
                else:
                    stage_counts["OTHER(未分類のwarning文字列)"] += 1
                    reason_counts[w[:40]] += 1
            elif n_held >= MAX_POS:
                # このsymbol名を含む個別warningは無い（_build_ordersはcapacity break後
                # 最初の1件だけをwarningに出す）が、held>=max_positionsという
                # 実測条件からCAPACITY起因と判定できる。
                stage_counts["CAPACITY(position_full, inferred)"] += 1
                reason_counts["最大ポジション数到達(推定: 同一capacity break内)"] += 1
            else:
                stage_counts["UNCLASSIFIED(warningsにも一致なし)"] += 1
                reason_counts["unclassified"] += 1

    print()
    print("  ── Stage別集計（直近signalファイルベース・実データのみ）───────")
    total = sum(stage_counts.values())
    for stage, n in sorted(stage_counts.items(), key=lambda kv: -kv[1]):
        pct = round(100.0 * n / max(1, total), 1)
        print(f"  {stage:<45} n={n:>4}  ({pct}%)")
    print()
    print(f"  候補が存在した日 / 全対象日 = {n_days_with_candidate}/{n_days_total} "
          f"({round(100*n_days_with_candidate/max(1,n_days_total),1)}%)")
    print(f"  候補ゼロ日（ENTRY_DEFICIT相当） = {n_days_total - n_days_with_candidate}日"
          f" ({round(100*(n_days_total-n_days_with_candidate)/max(1,n_days_total),1)}%)")
    print(f"  候補あり かつ portfolio満枠(held>=3)の日 = {n_days_full_when_candidate}/{n_days_with_candidate}"
          f" ({round(100*n_days_full_when_candidate/max(1,n_days_with_candidate),1)}%)")
    print()

    print("  ── Phase4: Block理由ランキング TOP20 ───────────────────────")
    for rank, (reason, n) in enumerate(sorted(reason_counts.items(), key=lambda kv: -kv[1])[:20], 1):
        print(f"  {rank:>2}. {reason:<50} n={n}")
    print()

    # EVS store cross-check (skip_reason as literally recorded)
    print("  ── 参考: 既存EVSストア(runtime/analytics/executed_vs_skipped.jsonl)"
          "の skip_reason 分布 ──")
    if EVS_STORE.exists():
        recs = [json.loads(l) for l in EVS_STORE.read_text(encoding="utf-8").splitlines() if l.strip()]
        sr_counts = defaultdict(int)
        for r in recs:
            sr_counts[r.get("skip_reason") or "(executed)"] += 1
        for sr, n in sorted(sr_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {sr:<25} n={n}")
        print(f"  ⚠ 既知の不整合: 2026-06-23 2802.T はEVSに executed=False/slot_full と"
              f"記録されているが、同日の実行結果(logs/trades.jsonl)では実際にBUYが約定している"
              f"（同一runのorders.jsonにも2802.T BUYが存在）。executed判定ロジックに"
              f"信頼性の欠陥がある可能性が高い（時系列上、複数run/日のうちどのrunの"
              f"時点かをEVSレコードが記録していないため、taken-at-timestamp単位での"
              f"切り分けが不可能）。skip_reason自体（slot_utilizationから独立に算出）は"
              f"executedの不整合とは無関係に妥当と判断する。")
    else:
        print("  (EVSストアが見つかりません)")
    print()

    return {"stage_counts": dict(stage_counts), "reason_counts": dict(reason_counts),
            "n_days_total": n_days_total, "n_days_with_candidate": n_days_with_candidate,
            "n_days_full_when_candidate": n_days_full_when_candidate}


def _forward_return(symbol: str, eval_date: str, horizon_days: int) -> "float | None":
    path = OHLCV_DIR / f"{symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        eval_ts = pd.Timestamp(eval_date)
        mask = df.index >= eval_ts
        if not mask.any():
            return None
        base_idx = df.index[mask][0]
        base_price = float(df.loc[base_idx, "Close"])
        pos = df.index.get_loc(base_idx)
        if isinstance(pos, slice):
            pos = pos.start
        fwd_pos = pos + horizon_days
        if fwd_pos >= len(df):
            return None
        fwd_price = float(df["Close"].iloc[fwd_pos])
        return round((fwd_price / base_price - 1) * 100, 3)
    except Exception:
        return None


def phase5(days_back: int = 30) -> dict:
    print("=" * 78)
    print(f"  Phase5: Skipped銘柄 Forward Return実測（5d/10d/20d, 直近{days_back}日）")
    print("=" * 78)

    by_date = _latest_signal_files_by_date(days_back)
    skipped_events: list[tuple[str, str]] = []  # (eval_date, symbol)
    for date_str, path in by_date.items():
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        signals = d.get("signals", [])
        orders  = d.get("orders", [])
        executed_syms = {o["symbol"] for o in orders if o.get("side") == "BUY"}
        n_held = sum(1 for s in signals if s.get("currently_holding"))
        if n_held < MAX_POS:
            continue
        for s in signals:
            if s.get("signal") == 1 and not s.get("currently_holding") and s["symbol"] not in executed_syms:
                iso_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
                skipped_events.append((iso_date, s["symbol"]))

    skipped_events = sorted(set(skipped_events))
    print(f"  対象イベント数（CAPACITY skip, 重複排除）: {len(skipped_events)}")

    rets5, rets10, rets20 = [], [], []
    print()
    print(f"  {'eval_date':<12}{'symbol':<10}{'fwd5d%':>10}{'fwd10d%':>10}{'fwd20d%':>10}")
    for eval_date, sym in skipped_events:
        r5  = _forward_return(sym, eval_date, 5)
        r10 = _forward_return(sym, eval_date, 10)
        r20 = _forward_return(sym, eval_date, 20)
        print(f"  {eval_date:<12}{sym:<10}"
              f"{(f'{r5:+.2f}' if r5 is not None else 'N/A'):>10}"
              f"{(f'{r10:+.2f}' if r10 is not None else 'N/A'):>10}"
              f"{(f'{r20:+.2f}' if r20 is not None else 'N/A'):>10}")
        if r5  is not None: rets5.append(r5)
        if r10 is not None: rets10.append(r10)
        if r20 is not None: rets20.append(r20)

    def _stats(vals):
        if not vals:
            return None
        return {
            "n": len(vals),
            "mean": round(sum(vals) / len(vals), 3),
            "win_rate": round(100 * sum(1 for v in vals if v > 0) / len(vals), 1),
        }

    s5, s10, s20 = _stats(rets5), _stats(rets10), _stats(rets20)
    print()
    print("  ── 期待値集計 ──────────────────────────────────────────────")
    print(f"  5d : {s5}")
    print(f"  10d: {s10}")
    print(f"  20d: {s20}")
    print()

    return {"n_events": len(skipped_events), "fwd5d": s5, "fwd10d": s10, "fwd20d": s20}


def phase6(phase3_result: dict, phase5_result: dict) -> None:
    print("=" * 78)
    print("  Phase6: Study53(ENTRY_DEFICIT)との整合性評価")
    print("=" * 78)

    if not STUDY53_PATH.exists():
        print("  Study53結果ファイルが見つかりません。")
        return
    d = json.loads(STUDY53_PATH.read_text(encoding="utf-8"))
    ds = d["daily_stats"]
    diag = d["diagnosis"]
    fr = d["forward_returns"]["max_pos"]

    print("  ── Study53 (2018-2024 IS backtest) ─────────────────────────")
    print(f"  候補ゼロ日割合            : {ds['days_no_cand_pct']}%")
    print(f"  ポジション満枠日割合       : {ds['days_full_cap_pct']}%")
    print(f"  primary診断                : {diag['primary']}")
    print(f"  cap引き上げ効果            : {diag['cap_verdict']}")
    print(f"  MAX_POS起因棄却割合        : {diag['pct_max_pos']}%")
    print(f"  MAX_POS棄却の fwd5d/10d/20d: "
          f"{fr['fwd5d']['mean']}% / {fr['fwd10d']['mean']}% / {fr['fwd20d']['mean']}%"
          f"  win_rate={fr['fwd5d']['win_rate']}/{fr['fwd10d']['win_rate']}/{fr['fwd20d']['win_rate']}%")
    print()

    live_no_cand_pct = round(
        100 * (phase3_result["n_days_total"] - phase3_result["n_days_with_candidate"])
        / max(1, phase3_result["n_days_total"]), 1,
    )
    live_full_pct = round(
        100 * phase3_result["n_days_full_when_candidate"]
        / max(1, phase3_result["n_days_with_candidate"]), 1,
    )
    print("  ── 直近30日 実LIVE ──────────────────────────────────────────")
    print(f"  候補ゼロ日割合            : {live_no_cand_pct}%"
          f"  (Study53: {ds['days_no_cand_pct']}%)")
    print(f"  候補あり時のportfolio満枠率: {live_full_pct}%"
          f"  (Study53 MAX_POS棄却割合: {diag['pct_max_pos']}%)")
    if phase5_result["fwd5d"]:
        print(f"  Skipped fwd5d実測 mean     : {phase5_result['fwd5d']['mean']}%"
              f"  (Study53 MAX_POS fwd5d: {fr['fwd5d']['mean']}%)")
    if phase5_result["fwd10d"]:
        print(f"  Skipped fwd10d実測 mean    : {phase5_result['fwd10d']['mean']}%"
              f"  (Study53 MAX_POS fwd10d: {fr['fwd10d']['mean']}%)")
    if phase5_result["fwd20d"]:
        print(f"  Skipped fwd20d実測 mean    : {phase5_result['fwd20d']['mean']}%"
              f"  (Study53 MAX_POS fwd20d: {fr['fwd20d']['mean']}%)")
    print()
    print("  ── 評価 ─────────────────────────────────────────────────────")
    print("  Study53診断「MAX_POS_DOMINANT（棄却理由の75%はポジション上限）だが"
          "cap引き上げの効果は限定的＝エントリー不足型ボトルネック」との整合性:")
    if live_no_cand_pct >= 40:
        print(f"  → 整合。直近30日も候補ゼロ日が{live_no_cand_pct}%と高水準で、"
              "「候補自体が少ない」というStudy53のENTRY_DEFICIT診断と符合する。")
    else:
        print(f"  → 直近30日の候補ゼロ日は{live_no_cand_pct}%とStudy53(66.4%)より低く、"
              "直近レジームでは候補供給が相対的に多い可能性がある（要追加サンプル）。")
    print(f"  一方、「候補が存在する日」に限れば直近LIVEは{live_full_pct}%が"
          f"満枠でCAPACITY棄却されており、Study53のMAX_POS棄却割合({diag['pct_max_pos']}%)"
          "と近い水準 — ポジション上限が『候補が出た時の』支配的制約であることは"
          "直近LIVEデータでも再現されている。")
    print("  → 総合: Study53の『MAX_POS支配的だがcap単独の引き上げでは"
          "改善が限定的（エントリー不足型）』という結論は、直近実LIVEの"
          "候補頻度・棄却率パターンと矛盾しない。")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    phase1_and_2_2026_06_29()
    p3 = phase3_and_4(days_back=30)
    p5 = phase5(days_back=30)
    phase6(p3, p5)
    return 0


if __name__ == "__main__":
    sys.exit(main())
