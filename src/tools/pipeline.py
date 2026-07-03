# === pipeline.py（最終安定版） ===
from __future__ import annotations

import sys
sys.stdout.reconfigure(encoding="utf-8")

import argparse
import csv
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent.parent))

from src.paths import BASE_DIR
from src.tools.trade_prompt_gen import (
    build_prompt,
    detect_schema,
    load_file,
    STAT_FN_MAP,
)
from src.tools.strategy_backtest import run as backtest_run

# =========================
# 設定
# =========================
DEEPSEEK_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
TIMEOUT = 60
RETRY = 2

CACHE_PATH = BASE_DIR / "research" / "cache.json"

# rejection_history は oos_stability_eval.py が書き込む
_REJECTION_HISTORY_PATH = BASE_DIR / "logs" / "rejection_history.json"

# rejection条件ラベル（プロンプト注入用）
_REJ_LABEL_MAP: dict[str, str] = {
    "low_sharpe":               "低Sharpe",
    "high_variance":            "高分散Sharpe",
    "low_trades":               "取引数不足",
    "insufficient_oos_periods": "OOS期間不足",
}


def _load_rej_history() -> dict[str, dict]:
    """rejection_history.json を読み込む。存在しなければ空dict。"""
    if _REJECTION_HISTORY_PATH.exists():
        try:
            return json.loads(_REJECTION_HISTORY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _inject_rejection_feedback(
    payload:     dict,
    rej_history: dict[str, dict],
    weight:      float = 0.5,   # 0.3〜0.7 の範囲で固定
) -> None:
    """
    smoothed rejection ratios を「弱制約」としてプロンプトへ注入する。
    effective_ratio = weight * smoothed_ratio
    「回避せよ」ではなく「過度に繰り返さない」という表現を使う。
    eligible=False 戦略の傾向を探索から徐々に排除する。
    """
    lines: list[str] = ["", "【過去の失敗傾向（参考・完全回避は不要）】"]
    has_signal = False
    for k, v in rej_history.items():
        eff = round(weight * v.get("ratio", 0.0), 3)
        if eff > 0.01:   # 影響が小さい条件はプロンプトに載せない
            label = _REJ_LABEL_MAP.get(k, k)
            lines.append(f"- {label}: {eff:.2f}")
            has_signal = True
    if not has_signal:
        return
    lines.append("これらを「過度に繰り返さない」戦略を生成せよ")
    payload["messages"][1]["content"] += "\n".join(lines)

# =========================
# 確率的成功パターン注入（過剰収束防止）
# =========================
_TOP_K                 = 5      # スコア上位K件
_PATTERN_PROB          = 0.4    # 注入確率 30〜50%
_MIN_TRADES_FOR_SCORE  = 30     # スコア計算の最低取引数フィルタ
_MIN_VALID_FOR_PATTERN = 2      # パターン生成に必要な最低eligible件数


def _classify_template(strategy_json: dict) -> str:
    s     = strategy_json.get("strategy", strategy_json)
    entry = s.get("entry", "").lower()
    if "rsi_14" in entry:
        return "rsi"
    ma_n = re.findall(r'ma_(\d+)', entry)
    if "atr_20" in entry and not ma_n:
        return "atr"
    if len(ma_n) >= 2:
        return "ma_cross"
    if len(ma_n) == 1:
        return "breakout"
    return "other"


def _extract_numeric_params(strategy_json: dict) -> dict[str, list]:
    """戦略JSON から正規表現で数値パラメータを抽出する。"""
    s        = strategy_json.get("strategy", strategy_json)
    combined = " ".join(str(v) for v in s.values())
    return {
        "ma_periods":     [int(x)   for x in re.findall(r'ma_(\d+)',                              combined)],
        "rsi_thresholds": [float(x) for x in re.findall(r'rsi_14\s*[<>]=?\s*([\d.]+)',            combined)],
        "hold_days":      [int(x)   for x in re.findall(r'hold_days\s*>=\s*(\d+)',                combined)],
        "sl_mult":        [float(x) for x in re.findall(r'entry_price\s*\*\s*([\d.]+)',           combined)],
        "size_coeff":     [float(x) for x in re.findall(r'capital\s*\*\s*([\d.]+)\s*/\s*entry',  combined)],
    }


def _score_entry(entry: dict) -> float:
    """score = oos_sharpe * log(trades)  ← ranking指標。trades < 30 は -inf 除外。"""
    sharpe = entry.get("sharpe") or 0.0
    trades = entry.get("trades") or 0
    if trades < _MIN_TRADES_FOR_SCORE:
        return float("-inf")
    log_trades = np.log(max(trades, 1))
    return float(sharpe) * log_trades


def _dist_desc(vals: list, name: str, unit: str = "") -> str:
    """
    「集中範囲 + ばらつき（std）」の両方を含む分布記述を返す。
    単一値への収束を防止するため、std情報を必ず明記する。
    """
    if not vals:
        return ""
    lo, hi = min(vals), max(vals)
    if len(vals) < 2:
        return f"{name}: {vals[0]}{unit}（1件のみ参考・単一値への収束禁止）"
    mean = sum(vals) / len(vals)
    var  = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    std  = var ** 0.5
    if std < 0.5:
        return (
            f"{name}: {lo}{unit}〜{hi}{unit}に集中するが近傍の値でも成立例あり"
            f"（std={std:.1f}、分散あり・単一値への収束禁止）"
        )
    return (
        f"{name}: {lo}{unit}〜{hi}{unit}（std={std:.1f}、"
        f"ばらつきあり・明確な最適値なし）"
    )


def _build_success_pattern_text(valid_results: list) -> str:
    """
    eligible な上位K件（score = sharpe * log(trades)）から成功パターンテキストを生成する。

    分散維持のため「集中範囲」と「ばらつき（std）」を両方記述する。
    最低 _MIN_VALID_FOR_PATTERN 件未満なら空文字を返す（早期ラウンドは無注入）。
    """
    eligible = [e for e in valid_results if e.get("eligible", True)]
    if len(eligible) < _MIN_VALID_FOR_PATTERN:
        return ""

    # score でランキング、trades < 30 は除外
    scored = [(e, _score_entry(e)) for e in eligible]
    scored = [(e, s) for e, s in scored if s > float("-inf")]
    if not scored:
        return ""
    scored.sort(key=lambda x: -x[1])
    top = [e for e, _ in scored[:_TOP_K]]

    # テンプレート種別の分布
    tpl_cnt: dict[str, int] = {}
    for e in top:
        t = _classify_template(e["strategy"])
        tpl_cnt[t] = tpl_cnt.get(t, 0) + 1

    # パラメータを全上位戦略から収集
    all_ma:   list[int]   = []
    all_rsi:  list[float] = []
    all_hold: list[int]   = []
    all_sl:   list[float] = []
    all_sz:   list[float] = []
    for e in top:
        p = _extract_numeric_params(e["strategy"])
        all_ma.extend(p["ma_periods"])
        all_rsi.extend(p["rsi_thresholds"])
        all_hold.extend(p["hold_days"])
        all_sl.extend(p["sl_mult"])
        all_sz.extend(p["size_coeff"])

    trades_vals = [
        e.get("trades") or 0 for e in top
        if (e.get("trades") or 0) >= _MIN_TRADES_FOR_SCORE
    ]

    lines: list[str] = [
        "",
        "【成功戦略パターン（参考のみ・コピー禁止）】",
        "- テンプレート分布: "
            + ", ".join(f"{k}={v}件" for k, v in sorted(tpl_cnt.items())),
    ]
    for desc in (
        _dist_desc(all_ma,   "MA期間"),
        _dist_desc(all_rsi,  "RSI閾値"),
        _dist_desc(all_hold, "保有日数", "日"),
        _dist_desc(all_sl,   "SL係数"),
        _dist_desc(all_sz,   "ポジサイズ係数"),
    ):
        if desc:
            lines.append(f"- {desc}")

    if trades_vals:
        lines.append(
            f"- tradesレンジ: {min(trades_vals)}〜{max(trades_vals)}件"
            f"（上位{len(top)}戦略・trades>={_MIN_TRADES_FOR_SCORE}件のみ対象）"
        )

    # 必須制約: 分散維持・コピー・収束を明示的に禁止
    lines += [
        "- 成功傾向は参考情報であり必須ではない",
        "- 同一戦略の再生成は禁止",
        "- パターンのコピーは禁止",
        "- 分散を維持すること",
    ]

    return "\n".join(lines)

# =========================
# フォールバック戦略（多様性付き）
# =========================
_MA_PAIRS   = [(5, 20), (5, 50), (10, 20), (10, 50), (20, 50)]
_SL_RANGE   = (0.91, 0.97)
_SIZE_RANGE = (0.10, 0.25)

def generate_fallback_strategy() -> dict:
    """API失敗時に呼ぶ多様な戦略ジェネレータ。4テンプレートをランダム選択。"""
    t = random.choice(["ma_cross", "breakout", "rsi", "atr"])
    sl   = round(random.uniform(*_SL_RANGE),   2)
    size = round(random.uniform(*_SIZE_RANGE), 2)

    if t == "ma_cross":
        short, long = random.choice(_MA_PAIRS)
        return {"strategy": {
            "entry":         f"ma_{short} > ma_{long}",
            "exit":          f"ma_{short} < ma_{long}",
            "stop_loss":     f"entry_price * {sl}",
            "position_size": f"capital * {size} / entry_price",
        }}

    elif t == "breakout":
        ma_n = random.choice([20, 50])
        hold = random.randint(10, 30)
        return {"strategy": {
            "entry":         f"close > ma_{ma_n}",
            "exit":          f"hold_days >= {hold}",
            "stop_loss":     f"entry_price * {sl}",
            "position_size": f"capital * {size} / entry_price",
        }}

    elif t == "rsi":
        low  = random.randint(25, 40)
        high = random.randint(60, 75)
        return {"strategy": {
            "entry":         f"rsi_14 < {low}",
            "exit":          f"rsi_14 > {high}",
            "stop_loss":     f"entry_price * {sl}",
            "position_size": f"capital * {size} / entry_price",
        }}

    else:  # atr
        k = round(random.uniform(1.0, 3.0), 2)
        return {"strategy": {
            "entry":         f"close > ma_20 + {k} * atr_20",
            "exit":          f"close < ma_20 - {k} * atr_20",
            "stop_loss":     f"entry_price * {sl}",
            "position_size": f"capital * {size} / entry_price",
        }}

# =========================
# キャッシュ
# =========================
def load_cache():
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}

def save_cache(cache):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(cache, indent=2), encoding="utf-8")

# =========================
# OHLCV ローダー
# =========================
_OHLCV_REQUIRED = {"open", "high", "low", "close", "volume"}

def load_ohlcv(symbol: str) -> pd.DataFrame:
    path = BASE_DIR / "data" / "ohlcv" / f"{symbol}.csv"
    if not path.exists():
        raise FileNotFoundError(f"OHLCVファイルが見つかりません: {path}")
    df = pd.read_csv(path, parse_dates=["date"])
    missing = _OHLCV_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"必須列不足 [{symbol}]: {sorted(missing)}")
    df = df.sort_values("date").set_index("date")
    return df

# =========================
# API
# =========================
def call_api(payload, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(RETRY + 1):
        try:
            resp = requests.post(
                DEEPSEEK_URL,
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            resp.raise_for_status()

            try:
                content = resp.json()["choices"][0]["message"]["content"]
            except Exception:
                content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")

            if not content:
                raise ValueError("empty")

            return json.loads(content)

        except Exception as e:
            if attempt < RETRY:
                print("[WARN] retry...")
                time.sleep(2)
            else:
                print(f"[FALLBACK] {e}")
                return generate_fallback_strategy()

# =========================
# 相関フィルタ
# =========================
def filter_correlated_strategies(results, threshold=0.9):
    selected = []

    for r in results:
        eq = r.get("equity_curve")
        if eq is None or len(eq) < 50:
            selected.append(r)
            continue

        eq = np.array(eq)
        keep = True

        for s in selected:
            eq2 = s.get("equity_curve")
            if eq2 is None or len(eq2) != len(eq):
                continue

            corr = np.corrcoef(eq, eq2)[0, 1]
            if corr > threshold:
                keep = False
                break

        if keep:
            selected.append(r)

    return selected

# =========================
# メイン
# =========================
def run_pipeline(df_prompt, source_name, api_key, n=50, ohlcv_df=None):

    schema = detect_schema(df_prompt, Path(source_name))
    stats = STAT_FN_MAP[schema](df_prompt)

    cache = load_cache()

    # rejection履歴をループ外で一度だけ読み込む（eligible=False傾向の弱制約注入用）
    rej_history = _load_rej_history()
    if rej_history:
        print(f"[INFO] rejection_history ロード済み ({len(rej_history)} 条件)")

    results = []
    valid_results = []
    seen_keys: set[str] = set()  # 同一戦略ループ排除用

    date_str = datetime.now().strftime("%Y-%m-%d")
    out_csv = BASE_DIR / f"research/deepseek_results_{date_str}.csv"
    out_json = BASE_DIR / f"research/valid_strategies_{date_str}.json"

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["round", "valid", "sharpe", "error"])
        writer.writeheader()

        for i in range(1, n + 1):
            row = {"round": i, "valid": False, "sharpe": "", "error": ""}

            try:
                payload = build_prompt(schema, stats, source_name, date_str)
                payload["model"] = "deepseek-ai/deepseek-v4-pro"
                payload["max_tokens"] = 500

                # 弱制約注入: eligible=False 傾向を「繰り返さない」指示として埋め込む
                # weight=0.5 固定（強制回避にならないよう傾向として注入）
                if rej_history:
                    _inject_rejection_feedback(payload, rej_history, weight=0.5)

                # 確率的成功パターン注入 (prob=0.4): 30〜50%のラウンドのみ注入
                # 非注入ラウンド → 完全探索（rejectionフィードバックのみ）で多様性を保持
                use_pattern = np.random.rand() < _PATTERN_PROB
                if use_pattern:
                    _pat = _build_success_pattern_text(valid_results)
                    if _pat:
                        payload["messages"][1]["content"] += _pat
                        print(f"[PATTERN] round {i}: injected ({len(valid_results)} eligible so far)")
                    else:
                        print(f"[PATTERN] round {i}: no eligible history yet → full exploration")
                else:
                    print(f"[PATTERN] round {i}: full exploration (no pattern)")

                key = json.dumps(payload, sort_keys=True)

                if key in cache:
                    print("[CACHE HIT]")
                    strategy_json = cache[key]
                else:
                    print("[CACHE MISS]")
                    strategy_json = call_api(payload, api_key)
                    cache[key] = strategy_json
                    save_cache(cache)

                # 同一戦略ループ排除: 既出ならランダム多様化に差し替え
                strat_key = json.dumps(strategy_json, sort_keys=True)
                if strat_key in seen_keys:
                    strategy_json = generate_fallback_strategy()
                    print("[DIVERSITY] duplicate → random fallback")
                else:
                    seen_keys.add(strat_key)

                # OOS: OHLCV を日付でスライス（trades_log は統計用途のみ）
                split = int(len(df_prompt) * 0.7)
                oos_start = pd.to_datetime(df_prompt.iloc[split]["entry_date"])
                if ohlcv_df is None:
                    raise RuntimeError("ohlcv_df が未指定です。--symbol を指定してください")
                df_test = ohlcv_df.loc[oos_start:]
                print(f"OOS: {split}/{len(df_prompt)-split} (OHLCV rows: {len(df_test)})")

                result = backtest_run(
                    strategy_json,
                    df_test,
                    stability_splits=5
                )

                sharpe = result["metrics"].get("sharpe", 0)
                trades = result["metrics"].get("n_trades", 0) or 0
                valid  = result["constraints"]["valid"]

                row["valid"]  = valid
                row["sharpe"] = round(sharpe, 4)

                entry = {
                    "strategy":     strategy_json,
                    "equity_curve": result.get("equity_curve"),
                    "valid":        valid,
                    "eligible":     valid,          # eligible=False → 再利用禁止
                    "sharpe":       float(sharpe) if sharpe is not None else 0.0,
                    "trades":       int(trades),    # score = sharpe * log(trades) 計算用
                }

                results.append(entry)

                # eligible=False 戦略は valid_results に積まない（再利用禁止）
                if valid and entry["eligible"]:
                    valid_results.append(entry)
                    print(f"[OK] {i} sharpe={sharpe:.2f}")
                else:
                    print(f"[NG] {i}")

            except Exception as e:
                row["error"] = str(e)
                print(f"[SKIP] {i}: {e}")

            writer.writerow(row)
            f.flush()

    valid_results = filter_correlated_strategies(valid_results)

    out_json.write_text(json.dumps(valid_results, indent=2), encoding="utf-8")

    print("\n完了")
    print(f"valid: {len(valid_results)} / {n}")

# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--symbol", default="7203.T", help="OHLCVシンボル（例: 7203.T）")
    args = parser.parse_args()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("APIキー未設定")
        return

    df = load_file(Path(args.data))
    ohlcv_df = load_ohlcv(args.symbol)
    print(f"[OHLCV] {args.symbol}: {len(ohlcv_df)}行 ({ohlcv_df.index[0].date()} 〜 {ohlcv_df.index[-1].date()})")
    run_pipeline(df, args.data, api_key, args.n, ohlcv_df=ohlcv_df)

if __name__ == "__main__":
    main()