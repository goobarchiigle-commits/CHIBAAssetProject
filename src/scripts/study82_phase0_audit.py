"""
src/scripts/study82_phase0_audit.py
Study82 Phase0 — 決算発表日時精度監査（実データ検証・小サンプル）

正典: reports/study82_phase0_design.md v1.3 §2-3（Audit1-6・PASS/FAIL/UNKNOWN判定基準）
      reports/roadmap_v15_governance_layer.md §8A-1A（FAIL時決定木）

目的（狭く固定・厳守）:
  Can PEAD be researched WITHOUT look-ahead bias or timestamp leakage?
  "PEADは機能するか" ではない。アルファ・イベントリターン・サプライズ率は一切計算しない。

禁止（本スクリプトの範囲外・実装しない）:
  - alpha calculation / PEAD backtest / event return measurement
  - portfolio optimization / Study83 implementation / Route ranking
  - assumption tuning（本監査の結果を見てからの閾値調整）

サンプル: 小規模のみ（20-50レコード目標）。大規模データ収集はしない。
出力: PASS / FAIL / UNKNOWN の三値のみ。
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

try:
    from src.paths import REPORTS_DIR, RESULTS_DIR
    from src.jquants.provider import JQuantsProvider
    from src.jquants.exceptions import JQuantsAPIError, JQuantsConfigError
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.paths import REPORTS_DIR, RESULTS_DIR
    from src.jquants.provider import JQuantsProvider
    from src.jquants.exceptions import JQuantsAPIError, JQuantsConfigError

RUN_DATE = "2026-07-20"
OUT_RAW_JSON = RESULTS_DIR / f"study82_phase0_raw_sample_{RUN_DATE}.json"
OUT_AUDIT_JSON = RESULTS_DIR / f"study82_phase0_audit_{RUN_DATE}.json"
OUT_REPORT_MD = REPORTS_DIR / "study82_phase0_report.md"

# 流動性の高い既知大型株のみ（監査目的=「読めるか」であり銘柄選定にアルファ的意図はない）
LIQUID_SAMPLE_CODES = ["86970", "72030", "99840", "67580", "83060"]
LIQUID_DATE_FROM, LIQUID_DATE_TO = "2025-01-01", "2025-12-31"

# 廃止銘柄カバレッジ監査(Audit6)専用: data/jquants/metadata/universe_events.parquet の
# REMOVEイベントより実測抽出した実在の廃止銘柄コード（2026-07-20時点最新に近い1件）
DELISTED_TEST_CODE = "44490"
DELISTED_DATE_FROM, DELISTED_DATE_TO = "2024-01-01", "2026-06-01"

DISC_DATE_KEYS = ("DiscDate", "DisclosedDate", "DisclosureDate", "disc_date")
DISC_TIME_KEYS = ("DiscTime", "DisclosedTime", "DisclosureTime", "disc_time")
# 実データ確認済み（2026-07-20実行）: v2実フィールド名は略記（DocType/DiscNo）。
# v1想定のTypeOfDocument/DisclosureNumberは実在しない — 他エンドポイント（daily_quotes等）
# と同様のv1ドキュメント/v2実装乖離パターン（research_state.md 2026-07-10ログと同型）。
DOC_TYPE_KEYS = ("DocType", "TypeOfDocument", "DocumentType", "type_of_document")
DISC_NUM_KEYS = ("DiscNo", "DisclosureNumber", "DiscNumber", "disclosure_number")


def _first_present(record: dict, candidates: tuple[str, ...]) -> str | None:
    for k in candidates:
        if k in record:
            return k
    return None


def main() -> None:
    print("Study82 Phase0 — PEAD発表日時精度監査（実データ・小サンプル）")
    print("目的: Can PEAD be researched without leakage? （アルファ測定は行わない）\n")

    audit: dict = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "scope": "small_sample_audit_only",
        "prohibited_not_performed": [
            "alpha_calculation", "pead_backtest", "event_return_measurement",
            "portfolio_optimization", "study83_implementation",
            "route_ranking_discussion", "assumption_tuning",
        ],
        "audits": {}, "raw_sample_meta": {}, "final_status": None, "notes": [],
    }

    # ---- Audit1: Endpoint availability ----
    try:
        provider = JQuantsProvider()
    except JQuantsConfigError as e:
        audit["audits"]["Audit1_endpoint_availability"] = {"result": "FAIL", "reason": str(e)}
        audit["final_status"] = "FAIL"
        _finalize(audit, [])
        return

    all_records: list[dict] = []
    resolved_path = None
    endpoint_errors: list[str] = []
    for code in LIQUID_SAMPLE_CODES:
        try:
            recs, path = provider.get_fins_summary(code=code, date_from=LIQUID_DATE_FROM, date_to=LIQUID_DATE_TO)
            resolved_path = resolved_path or path
            for r in recs:
                r["_sample_code"] = code
                r["_sample_group"] = "liquid_largecap"
            all_records.extend(recs)
            print(f"  code={code}: {len(recs)}件取得 (path={path})")
        except JQuantsAPIError as e:
            endpoint_errors.append(f"code={code}: {e}")
            print(f"  code={code}: 取得失敗 ({e})")
        if len(all_records) >= 50:
            break

    delisted_records: list[dict] = []
    try:
        recs, _ = provider.get_fins_summary(code=DELISTED_TEST_CODE, date_from=DELISTED_DATE_FROM, date_to=DELISTED_DATE_TO)
        for r in recs:
            r["_sample_code"] = DELISTED_TEST_CODE
            r["_sample_group"] = "delisted_probe"
        delisted_records = recs
        print(f"  廃止銘柄probe code={DELISTED_TEST_CODE}: {len(recs)}件取得")
    except JQuantsAPIError as e:
        endpoint_errors.append(f"delisted_probe code={DELISTED_TEST_CODE}: {e}")
        print(f"  廃止銘柄probe code={DELISTED_TEST_CODE}: 取得失敗 ({e})")

    endpoint_ok = resolved_path is not None
    audit["audits"]["Audit1_endpoint_availability"] = {
        "result": "PASS" if endpoint_ok else "FAIL",
        "resolved_path": resolved_path,
        "errors": endpoint_errors,
        "note": "エンドポイントが1件でも200を返せばPASS。全滅ならFAIL（Phase0.1のCONNECTABLE判定を実データで再確認）。",
    }

    if not endpoint_ok:
        audit["final_status"] = "FAIL"
        _finalize(audit, [])
        return

    sample_all = all_records + delisted_records
    audit["raw_sample_meta"] = {
        "liquid_sample_codes": LIQUID_SAMPLE_CODES, "liquid_date_range": [LIQUID_DATE_FROM, LIQUID_DATE_TO],
        "delisted_test_code": DELISTED_TEST_CODE, "delisted_date_range": [DELISTED_DATE_FROM, DELISTED_DATE_TO],
        "n_liquid_records": len(all_records), "n_delisted_records": len(delisted_records),
        "n_total_records": len(sample_all),
    }
    OUT_RAW_JSON.write_text(json.dumps(sample_all, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nraw sample保存: {OUT_RAW_JSON} ({len(sample_all)}件)")

    if not sample_all:
        audit["notes"].append("エンドポイントは疎通したがレコード0件 — 期間/コード指定を要再検討")
        audit["final_status"] = "UNKNOWN"
        _finalize(audit, sample_all)
        return

    all_keys = sorted({k for r in sample_all for k in r.keys() if not k.startswith("_")})
    audit["raw_sample_meta"]["observed_field_names"] = all_keys

    # ---- Audit2: DiscDate / DiscTime existence ----
    disc_date_key = _first_present(sample_all[0], DISC_DATE_KEYS)
    disc_time_key = _first_present(sample_all[0], DISC_TIME_KEYS)
    audit["audits"]["Audit2_discdate_disctime_existence"] = {
        "result": "PASS" if (disc_date_key and disc_time_key) else ("UNKNOWN" if disc_date_key else "FAIL"),
        "disc_date_key_found": disc_date_key, "disc_time_key_found": disc_time_key,
        "candidates_checked": {"date": DISC_DATE_KEYS, "time": DISC_TIME_KEYS},
    }

    # ---- Audit3: Missing ratio ----
    def _missing_ratio(key: str | None) -> float | None:
        if not key:
            return None
        vals = [r.get(key) for r in sample_all]
        missing = sum(1 for v in vals if v in (None, "", "null"))
        return round(missing / len(vals), 4) if vals else None

    date_missing = _missing_ratio(disc_date_key)
    time_missing = _missing_ratio(disc_time_key)
    audit["audits"]["Audit3_missing_ratio"] = {
        "disc_date_missing_ratio": date_missing, "disc_time_missing_ratio": time_missing,
        "n_sample": len(sample_all),
        "result": "PASS" if (time_missing is not None and time_missing < 0.05)
                   else ("UNKNOWN" if time_missing is None else "FAIL"),
    }

    # ---- Audit4: Correction disclosure existence ----
    doc_type_key = _first_present(sample_all[0], DOC_TYPE_KEYS)
    disc_num_key = _first_present(sample_all[0], DISC_NUM_KEYS)
    doc_types_observed = sorted({str(r.get(doc_type_key)) for r in sample_all if doc_type_key and r.get(doc_type_key)}) if doc_type_key else []
    audit["audits"]["Audit4_correction_disclosure_existence"] = {
        "result": "PASS" if (doc_type_key and disc_num_key) else ("UNKNOWN" if (doc_type_key or disc_num_key) else "FAIL"),
        "doc_type_key_found": doc_type_key, "disclosure_number_key_found": disc_num_key,
        "distinct_document_types_observed": doc_types_observed,
        "note": "フィールド構造上の可否のみ判定。小サンプル中に実際の訂正レコードが含まれる保証はない。",
    }

    # ---- Audit5: Leakage possibility (intraday vs after-close timestamps) ----
    time_values = sorted({str(r.get(disc_time_key)) for r in sample_all if disc_time_key and r.get(disc_time_key)}) if disc_time_key else []
    distinct_time_count = len(time_values)
    is_placeholder_suspect = distinct_time_count <= 1 and len(sample_all) > 3
    intraday_count = 0
    afterclose_count = 0
    for r in sample_all:
        t = r.get(disc_time_key) if disc_time_key else None
        if not t or not isinstance(t, str) or ":" not in t:
            continue
        try:
            hh, mm = int(t.split(":")[0]), int(t.split(":")[1])
        except (ValueError, IndexError):
            continue
        minutes = hh * 60 + mm
        if 9 * 60 <= minutes < 15 * 60:
            intraday_count += 1
        else:
            afterclose_count += 1
    audit["audits"]["Audit5_leakage_possibility"] = {
        "result": "FAIL" if is_placeholder_suspect else ("PASS" if disc_time_key else "UNKNOWN"),
        "distinct_time_values_observed": distinct_time_count,
        "sample_time_values": time_values[:10],
        "placeholder_suspected": is_placeholder_suspect,
        "intraday_9to15_count": intraday_count, "after_close_or_premarket_count": afterclose_count,
        "note": "distinct_time_values_observedが極端に少ない場合（定型値の疑い）はFAIL。"
                "場中/引後の分類はDiscTimeの実測値から機械的に導出（9:00-15:00=場中）。",
    }

    # ---- Audit6: Delisted stock coverage assessment ----
    audit["audits"]["Audit6_delisted_stock_coverage"] = {
        "result": "PASS" if delisted_records else ("FAIL" if endpoint_errors and any("delisted_probe" in e for e in endpoint_errors) else "UNKNOWN"),
        "delisted_test_code": DELISTED_TEST_CODE, "n_records_found": len(delisted_records),
        "note": "単一の廃止銘柄プローブのみ（小サンプル制約）。母集団規模の欠落率は未測定 — "
                "PASSは「少なくとも1件は廃止銘柄の決算データが取得できた」ことのみを意味する。",
    }

    # ---- 総合判定（study82_phase0_design.md §3準拠） ----
    def _r(name: str) -> str:
        return audit["audits"][name]["result"]

    core_results = [_r("Audit1_endpoint_availability"), _r("Audit5_leakage_possibility"),
                    _r("Audit6_delisted_stock_coverage")]
    disc_result = "PASS" if (disc_date_key and disc_time_key) else ("FAIL" if not disc_date_key else "UNKNOWN")
    core_results.append(disc_result)

    if any(r == "FAIL" for r in core_results):
        final = "FAIL"
    elif any(r == "UNKNOWN" for r in core_results):
        final = "UNKNOWN"
    else:
        final = "PASS"
    audit["final_status"] = final
    audit["final_status_basis"] = (
        "PASS/FAIL/UNKNOWN は Audit1(endpoint)・Audit2(DiscDate/DiscTime実在)・"
        "Audit5(leakage)・Audit6(delisted coverage) の4項目から機械算出。"
        "Audit3(missing ratio)・Audit4(correction handling)は付随所見（study82_phase0_design.md §3）。"
    )

    _finalize(audit, sample_all)


def _finalize(audit: dict, sample_all: list[dict]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"\n===== Study82 Phase0 監査結果 =====")
    for name, res in audit["audits"].items():
        print(f"  {name}: {res.get('result')}")
    print(f"\n最終判定: {audit['final_status']}")
    print(f"監査JSON: {OUT_AUDIT_JSON}")


if __name__ == "__main__":
    main()
