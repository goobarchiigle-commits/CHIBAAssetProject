"""
src/research/evo_search.py
進化的戦略探索（パラメータ最適化）。

契約:
  DETERMINISTIC  : 固定シード。同一入力 → 同一出力。ランダム性ゼロ（外部漏れなし）
  PURE FUNCTION  : 副作用なし。入力 DataFrame を変更しない
  NO LOOKAHEAD   : 全シグナルは shift(1) 適用。TimeSeriesSplit のみ使用
  POST-COST ONLY : コスト・スリッページ・充足確率適用後のリターンのみ評価
  PANDAS/NUMPY   : 外部ライブラリ依存なし

実行方法:
    python -m src.research.evo_search
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

# ── 定数 ─────────────────────────────────────────────────────────────────────
_PERIODS_PER_YEAR: int   = 252
_EPS:              float = 1e-12

_CFG_DEFAULTS: dict[str, Any] = {
    # 母集団
    "n_variants":             50,
    "n_folds":                 5,
    "n_generations":          10,
    "top_k_frac":            0.2,    # 上位何割を親として残すか
    "seed":                   42,
    # 執行モデル
    "exec_cost":          0.00055,
    "exec_slippage":       0.001,
    "exec_fill_prob":        1.0,
    "exec_opportunity":      1.0,    # バー全体に対するポジション保有上限フラクション
    # ハードフィルター
    "max_dd_threshold":      0.30,
    "tail_return_threshold": -0.02,
    "min_trades":              5,
    # 突然変異
    "mutation_sigma":         0.10,  # パラメータレンジに対する Gaussian sigma 比率
}

# デフォルト探索空間: タートル型ブレイクアウト戦略パラメータ
_DEFAULT_PARAM_SPACE: dict[str, tuple[float, float, str]] = {
    "entry_window": ( 5.0, 100.0, "int"),
    "exit_window":  ( 5.0, 100.0, "int"),
    "atr_window":   ( 5.0,  30.0, "int"),
    "atr_mult":     ( 0.5,   5.0, "float"),
}


# ── ユーティリティ ────────────────────────────────────────────────────────────

def _safe(v: Any, fallback: float = 0.0) -> float:
    """NaN / inf / None → fallback。"""
    try:
        f = float(v)
        return f if math.isfinite(f) else fallback
    except (TypeError, ValueError):
        return fallback


def _cagr(returns: pd.Series) -> float:
    """年率換算リターン（CAGR）。空 / ゼロ → 0.0。全損 → -1.0。"""
    n = len(returns)
    if n == 0:
        return 0.0
    cum = float((1.0 + returns.clip(lower=-0.99)).prod())
    if cum <= 0.0 or not math.isfinite(cum):
        return -1.0
    n_years = n / _PERIODS_PER_YEAR
    return _safe(cum ** (1.0 / n_years) - 1.0) if n_years > 0 else 0.0


def _sharpe(returns: pd.Series) -> float:
    """年率換算シャープ比（無リスク金利=0）。"""
    if len(returns) < 2:
        return 0.0
    std = float(returns.std(ddof=1))
    if std < _EPS:
        return 0.0
    return _safe(float(returns.mean()) / std * math.sqrt(_PERIODS_PER_YEAR))


def _max_drawdown(returns: pd.Series) -> float:
    """最大ドローダウン（正値 ∈ [0, 1]）。"""
    if returns.empty:
        return 0.0
    cum = (1.0 + returns.clip(lower=-0.99)).cumprod()
    roll_max = cum.cummax()
    dd = (cum - roll_max) / roll_max.replace(0.0, np.nan)
    raw = float(-dd.min()) if not dd.dropna().empty else 0.0
    return min(1.0, max(0.0, _safe(raw)))


def _capacity(
    price_df: pd.DataFrame,
    returns: pd.Series,
    exec_cfg: dict,
) -> float:
    """
    容量プロキシ（スコアリング用。対数スケール入力として使用）。
    volume × close が利用可能: 日次出来高金額 × fill_prob
    不可:取引回数をプロキシ（ゼロでも log1p(1) = 0.69 を保証）
    """
    fill_prob = max(0.0, min(1.0, _safe(exec_cfg.get("exec_fill_prob", 1.0), 1.0)))
    if "volume" in price_df.columns and "close" in price_df.columns:
        sub = price_df.reindex(returns.index)
        daily_val = _safe(float((sub["volume"] * sub["close"]).mean()))
        return max(1.0, daily_val * fill_prob)
    n_trades = int((returns.diff().fillna(0.0) != 0.0).sum())
    return max(1.0, float(n_trades))


# ── 時系列フォールド分割 ──────────────────────────────────────────────────────

def _fold_indices(n: int, n_folds: int) -> list[tuple[int, int]]:
    """
    N バーを n_folds 等分したクロノロジカルスプリットの (start, end) リスト。
    シャッフルなし。先読みなし。
    """
    fold_size = max(1, n // n_folds)
    folds: list[tuple[int, int]] = []
    for i in range(n_folds):
        start = i * fold_size
        end = (start + fold_size) if i < n_folds - 1 else n
        if start < n:
            folds.append((start, min(end, n)))
    return folds


# ── バックテスト（タートル型ブレイクアウト）─────────────────────────────────

def _run_strategy(
    price_df: pd.DataFrame,
    params: dict,
    exec_cfg: dict,
) -> pd.Series:
    """
    タートル型ブレイクアウト戦略のバックテスト（純粋関数）。

    先読みなし: 全シグナルは shift(1) 適用済みデータのみ参照。
    執行モデル: コスト・スリッページ・充足確率・機会制約を適用。

    Args:
        price_df : OHLCV DataFrame（必須列: open, high, low, close）
        params   : {entry_window, exit_window, atr_window, atr_mult, ...}
        exec_cfg : {exec_cost, exec_slippage, exec_fill_prob, exec_opportunity}

    Returns:
        日次ネットリターン pd.Series（price_df と同インデックス）
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(price_df.columns) or price_df.empty:
        return pd.Series(0.0, index=price_df.index)

    close = price_df["close"].astype(float)
    high  = price_df["high"].astype(float)
    low   = price_df["low"].astype(float)
    n     = len(close)

    entry_n = max(2, int(_safe(params.get("entry_window", 20), 20)))
    exit_n  = max(2, int(_safe(params.get("exit_window",  55), 55)))

    # シグナル（shift(1) = 前日までのデータのみ参照 → 先読みなし）
    high_roll  = high.rolling(entry_n, min_periods=entry_n).max().shift(1)
    low_roll   = low.rolling(exit_n,  min_periods=exit_n).min().shift(1)
    entry_arr  = (close.values > high_roll.values).astype(np.int8)
    exit_arr   = (close.values < low_roll.values).astype(np.int8)

    # ポジション状態機械 O(N)（ベクトル化困難な状態依存ロジック）
    pos = np.zeros(n, dtype=np.int8)
    in_trade = False
    for i in range(n):
        if not in_trade:
            if entry_arr[i]:
                in_trade = True
        else:
            if exit_arr[i]:
                in_trade = False
        pos[i] = in_trade

    # 日次グロスリターン: 前日ポジション × 当日価格変化率（先読みなし）
    price_ret = close.pct_change().fillna(0.0).values
    prev_pos  = np.concatenate([[0], pos[:-1]])      # shift(1)
    gross_ret = prev_pos.astype(float) * price_ret

    # 執行コスト: ポジション変化時（エントリー/エグジット）に控除
    pos_diff      = np.diff(pos.astype(float), prepend=0.0)
    trade_events  = np.abs(pos_diff)                 # 1.0 on entry or exit
    cost          = _safe(exec_cfg.get("exec_cost",      0.00055))
    slippage      = _safe(exec_cfg.get("exec_slippage",  0.001))
    fill_prob     = max(0.0, min(1.0, _safe(exec_cfg.get("exec_fill_prob", 1.0), 1.0)))
    cost_per_side = cost + slippage
    cost_series   = trade_events * cost_per_side

    # 充足確率: 期待リターン × fill_prob（決定論的プロキシ）
    net_ret = gross_ret * fill_prob - cost_series

    # 機会制約: 保有フラクション > 上限の場合はリターンをスケールダウン
    opp_limit = max(0.0, min(1.0, _safe(exec_cfg.get("exec_opportunity", 1.0), 1.0)))
    actual_frac = float(pos.mean())
    if opp_limit < 1.0 and actual_frac > opp_limit + _EPS:
        net_ret = net_ret * (opp_limit / actual_frac)

    return pd.Series(np.nan_to_num(net_ret, nan=0.0), index=close.index)


# ── フォールド・ロバストネス計算 ─────────────────────────────────────────────

def _compute_fold_metrics(
    returns: pd.Series,
    price_df: pd.DataFrame,
    exec_cfg: dict,
) -> dict[str, float]:
    """1 フォールドのメトリクスを計算する（純粋関数）。"""
    n_trades = int((returns.diff().fillna(0.0).abs() > _EPS).sum())
    return {
        "sharpe":   _sharpe(returns),
        "cagr":     _cagr(returns),
        "max_dd":   _max_drawdown(returns),
        "capacity": _capacity(price_df, returns, exec_cfg),
        "n_trades": float(n_trades),
        "n_bars":   float(len(returns)),
    }


def _compute_robustness(
    fold_metrics: list[dict[str, float]],
    all_returns:  pd.Series,
) -> dict[str, float]:
    """
    フォールド横断のロバストネス指標を集計する（純粋関数）。

    Returns:
        sharpe, cagr, max_dd, capacity            : フォールド平均
        worst_sharpe, worst_cagr                  : フォールド最悪値
        sharpe_variance                           : フォールド間シャープ分散
        tail_return                               : 全リターンの 5th パーセンタイル
    """
    zero = {k: 0.0 for k in (
        "sharpe", "cagr", "max_dd", "capacity",
        "worst_sharpe", "sharpe_variance", "worst_cagr", "tail_return",
    )}
    if not fold_metrics:
        return zero

    sharpes  = np.array([m["sharpe"]   for m in fold_metrics], dtype=float)
    cagrs    = np.array([m["cagr"]     for m in fold_metrics], dtype=float)
    max_dds  = np.array([m["max_dd"]   for m in fold_metrics], dtype=float)
    caps     = np.array([m["capacity"] for m in fold_metrics], dtype=float)

    tail = (
        float(np.percentile(all_returns.values, 5))
        if len(all_returns) > 0 else 0.0
    )

    return {
        "sharpe":          _safe(float(np.mean(sharpes))),
        "cagr":            _safe(float(np.mean(cagrs))),
        "max_dd":          _safe(float(np.max(max_dds))),
        "capacity":        _safe(float(np.mean(caps))),
        "worst_sharpe":    _safe(float(np.min(sharpes))),
        "sharpe_variance": _safe(float(np.var(sharpes, ddof=0))),
        "worst_cagr":      _safe(float(np.min(cagrs))),
        "tail_return":     _safe(tail),
    }


# ── ハードフィルター ──────────────────────────────────────────────────────────

def _hard_filter(robustness: dict[str, float], cfg: dict) -> bool:
    """
    いずれかの基準に引っかかれば False（即時リジェクト）。

    基準:
      worst_sharpe < 0
      worst_cagr   < 0
      max_dd       > max_dd_threshold
      tail_return  < tail_return_threshold
    """
    if robustness["worst_sharpe"] < 0.0:
        return False
    if robustness["worst_cagr"] < 0.0:
        return False
    if robustness["max_dd"] > _safe(cfg.get("max_dd_threshold", 0.30)):
        return False
    if robustness["tail_return"] < _safe(cfg.get("tail_return_threshold", -0.02)):
        return False
    return True


# ── スコアリング ──────────────────────────────────────────────────────────────

def _score(robustness: dict[str, float]) -> float:
    """
    final_score = base_score × stability_penalty × tail_adjustment

    base_score        = Sharpe × (1 − max_dd) × log(1 + capacity)
    stability_penalty = clip(1 − sharpe_variance, 0, 1)
    tail_adjustment   = clip(1 + tail_return, 0, 2)

    - stability_penalty: 高分散ほどペナルティ（一貫性優先）
    - tail_adjustment:   テールリスクに明示的ペナルティ
    - capacity が大きいほど base_score 有利（対数圧縮）
    """
    sharpe   = _safe(robustness.get("sharpe",          0.0))
    max_dd   = _safe(robustness.get("max_dd",          1.0))
    capacity = _safe(robustness.get("capacity",        1.0))
    shp_var  = _safe(robustness.get("sharpe_variance", 0.0))
    tail_ret = _safe(robustness.get("tail_return",     0.0))

    base       = sharpe * max(0.0, 1.0 - max_dd) * math.log1p(max(0.0, capacity))
    stability  = max(0.0, min(1.0, 1.0 - shp_var))
    tail_adj   = max(0.0, min(2.0, 1.0 + tail_ret))
    return _safe(base * stability * tail_adj)


# ── 母集団生成 ────────────────────────────────────────────────────────────────

def _generate_initial_pop(
    base_params: dict,
    param_space: dict[str, tuple[float, float, str]],
    n_variants:  int,
    seed:        int,
) -> list[dict]:
    """
    base_params を中心に Gaussian ノイズで n_variants 個の変異体を生成する。
    variant[0] = base_params そのもの（精鋭保存）。

    RNG: np.random.Generator(PCG64(seed)) — 決定論的。
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    population: list[dict] = [dict(base_params)]

    for _ in range(n_variants - 1):
        variant = dict(base_params)
        for name, (lo, hi, dtype) in param_space.items():
            base_val = _safe(base_params.get(name, (lo + hi) / 2.0))
            sigma    = (hi - lo) * 0.20
            noisy    = float(base_val) + rng.normal(0.0, sigma)
            noisy    = max(lo, min(hi, noisy))
            variant[name] = int(round(noisy)) if dtype == "int" else float(noisy)
        population.append(variant)

    return population


# ── 突然変異 ─────────────────────────────────────────────────────────────────

def _mutate(
    params:      dict,
    param_space: dict[str, tuple[float, float, str]],
    rng:         np.random.Generator,
    sigma_frac:  float = 0.1,
) -> dict:
    """
    Gaussian（float）/ ±整数（int）ランダム変異。境界クリップ適用。

    Args:
        sigma_frac : レンジに対する Gaussian sigma の割合
    """
    mutated = dict(params)
    for name, (lo, hi, dtype) in param_space.items():
        val = _safe(params.get(name, (lo + hi) / 2.0))
        if dtype == "int":
            max_delta = max(1, int((hi - lo) * sigma_frac))
            delta     = int(rng.integers(-max_delta, max_delta + 1))
            new_val   = int(round(val)) + delta
        else:
            sigma   = (hi - lo) * sigma_frac
            new_val = float(val) + float(rng.normal(0.0, sigma))
        clipped = max(lo, min(hi, new_val))
        mutated[name] = int(round(clipped)) if dtype == "int" else float(clipped)
    return mutated


# ── 母集団評価 ────────────────────────────────────────────────────────────────

def _evaluate_population(
    population: list[dict],
    price_df:   pd.DataFrame,
    folds:      list[tuple[int, int]],
    exec_cfg:   dict,
    cfg:        dict,
) -> list[dict[str, Any]]:
    """
    全変異体 × 全フォールドを評価し、スコアリング済み結果リストを返す。

    Returns (各エントリ):
        params, robustness, per_split, final_score, passed_filter
    """
    results: list[dict[str, Any]] = []

    for params in population:
        fold_metrics: list[dict[str, float]] = []
        fold_returns:  list[pd.Series]       = []

        for start, end in folds:
            fold_df = price_df.iloc[start:end]
            ret     = _run_strategy(fold_df, params, exec_cfg)
            fm      = _compute_fold_metrics(ret, fold_df, exec_cfg)
            fold_metrics.append(fm)
            fold_returns.append(ret)

        all_ret  = pd.concat(fold_returns) if fold_returns else pd.Series(dtype=float)
        rob      = _compute_robustness(fold_metrics, all_ret)
        passed   = _hard_filter(rob, cfg)
        fs       = _score(rob) if passed else 0.0

        results.append({
            "params":        params,
            "robustness":    rob,
            "per_split":     fold_metrics,
            "final_score":   fs,
            "passed_filter": passed,
        })

    return results


# ── メイン: 進化的探索 ────────────────────────────────────────────────────────

def run_evolution(
    price_df:    pd.DataFrame,
    base_params: dict,
    param_space: dict[str, tuple[float, float, str]] | None = None,
    cfg:         dict | None = None,
) -> dict[str, Any]:
    """
    進化的戦略パラメータ探索を実行する（純粋関数）。

    パイプライン:
      1. 初期母集団生成（N variants, 固定シード）
      2. G 世代ループ:
         a. 全変異体を全フォールドで評価（クロノロジカルスプリット）
         b. ハードフィルター（worst_sharpe < 0 等）
         c. スコアリング（Sharpe × DD × capacity × stability × tail）
         d. 上位 K% を選択
         e. Gaussian / 離散突然変異で次世代を充填
      3. 最終評価 → トップ戦略と進化履歴を返す

    Args:
        price_df    : OHLCV DataFrame（必須列: open, high, low, close）
        base_params : 基準パラメータ dict
        param_space : {name: (min, max, dtype="int"|"float")}
                      None → _DEFAULT_PARAM_SPACE
        cfg         : 設定 dict（_CFG_DEFAULTS 参照）

    Returns:
        {
            "top_strategies":    [{params, robustness, per_split, final_score}],
            "evolution_history": [{generation, best_score, mean_score, n_passed}],
        }
        全値が JSON シリアライズ可能。
    """
    if param_space is None:
        param_space = dict(_DEFAULT_PARAM_SPACE)
    c = {**_CFG_DEFAULTS, **(cfg or {})}

    n_variants    = max(2,  int(_safe(c["n_variants"],    50)))
    n_folds       = max(2,  int(_safe(c["n_folds"],        5)))
    n_generations = max(1,  int(_safe(c["n_generations"], 10)))
    top_k_frac    = max(0.1, min(0.9, _safe(c["top_k_frac"], 0.2)))
    seed          = int(_safe(c["seed"], 42))
    sigma_frac    = _safe(c["mutation_sigma"], 0.1)

    exec_cfg = {k: c[k] for k in (
        "exec_cost", "exec_slippage", "exec_fill_prob", "exec_opportunity"
    )}

    folds   = _fold_indices(len(price_df), n_folds)
    top_k   = max(1, int(round(n_variants * top_k_frac)))

    population    = _generate_initial_pop(base_params, param_space, n_variants, seed)
    rng_mut       = np.random.Generator(np.random.PCG64(seed + 1))  # 突然変異専用 RNG

    evolution_history: list[dict[str, Any]] = []
    best_ever:         list[dict[str, Any]] = []

    for gen in range(n_generations):
        eval_results = _evaluate_population(population, price_df, folds, exec_cfg, c)
        eval_results.sort(key=lambda r: r["final_score"], reverse=True)

        passed = [r for r in eval_results if r["passed_filter"]]
        scores = [r["final_score"] for r in eval_results]
        evolution_history.append({
            "generation": gen,
            "best_score": _safe(scores[0] if scores else 0.0),
            "mean_score": _safe(float(np.mean(scores)) if scores else 0.0),
            "n_passed":   len(passed),
        })

        # 精鋭保存（ベスト世代を記憶）
        if passed and (
            not best_ever
            or passed[0]["final_score"] > best_ever[0]["final_score"]
        ):
            best_ever = passed[:top_k]

        # 選択: top_k を生存させる
        survivors = eval_results[:top_k]

        # 次世代: 生存者 + 突然変異で充填
        new_pop: list[dict] = [r["params"] for r in survivors]
        while len(new_pop) < n_variants:
            parent = survivors[int(rng_mut.integers(0, len(survivors)))]["params"]
            new_pop.append(_mutate(parent, param_space, rng_mut, sigma_frac))
        population = new_pop

    # 最終世代を評価し精鋭とマージ
    final_results = _evaluate_population(population, price_df, folds, exec_cfg, c)
    final_results.sort(key=lambda r: r["final_score"], reverse=True)
    final_passed  = [r for r in final_results if r["passed_filter"]]

    all_candidates = best_ever + final_passed
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for r in all_candidates:
        key = str(sorted(r["params"].items()))
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    deduped.sort(key=lambda r: r["final_score"], reverse=True)

    return {
        "top_strategies":    deduped[:top_k],
        "evolution_history": evolution_history,
    }


# ── 使用例 ───────────────────────────────────────────────────────────────────

def _example() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")

    # 人工 OHLCV データ
    # 明確な強いトレンド: 日次ドリフト 0.0012 (年率+35%) + 低ボラ 0.004
    # → 買い持ち Sharpe ≈ 0.0012/0.004 * sqrt(252) ≈ 4.8
    # → タートル戦略がフォールド横断で安定的に正Sharpeを出せる水準
    rng  = np.random.default_rng(0)
    n    = 1000
    ret  = 0.0012 + rng.normal(0.0, 0.004, n)
    price = 1000.0 * np.cumprod(1.0 + ret)
    noise = lambda s: rng.normal(0, s, n)
    price_df = pd.DataFrame({
        "open":   price * (1 + noise(0.001)),
        "high":   price * (1 + np.abs(noise(0.004))),
        "low":    price * (1 - np.abs(noise(0.004))),
        "close":  price,
        "volume": rng.integers(1000, 8000, n).astype(float),
    }, index=pd.date_range("2020-01-01", periods=n, freq="B"))

    base_params = {
        "entry_window": 20,
        "exit_window":  40,
        "atr_window":   14,
        "atr_mult":     1.0,
    }
    cfg = {
        "n_variants":    30,
        "n_folds":        5,
        "n_generations":  5,
        "top_k_frac":   0.2,
        "seed":          42,
        "exec_cost":     0.00055,
        "exec_slippage": 0.001,
    }

    # 診断: ベース戦略の各フォールドメトリクスを表示
    print("=== 診断: ベース戦略フォールドメトリクス ===")
    exec_cfg_diag = {"exec_cost": 0.00055, "exec_slippage": 0.001,
                     "exec_fill_prob": 1.0, "exec_opportunity": 1.0}
    folds_diag = _fold_indices(len(price_df), 5)
    for fi, (s, e) in enumerate(folds_diag):
        fold_df = price_df.iloc[s:e]
        ret_    = _run_strategy(fold_df, base_params, exec_cfg_diag)
        fm      = _compute_fold_metrics(ret_, fold_df, exec_cfg_diag)
        print(f"  fold {fi}: sharpe={fm['sharpe']:.3f}  cagr={fm['cagr']:.3f}  "
              f"max_dd={fm['max_dd']:.3f}  n_trades={int(fm['n_trades'])}")

    print("\n=== run_evolution ===")
    result = run_evolution(price_df, base_params, cfg=cfg)

    print(f"世代数: {len(result['evolution_history'])}")
    for h in result["evolution_history"]:
        print(
            f"  Gen {h['generation']:2d}: "
            f"best={h['best_score']:.4f}  "
            f"mean={h['mean_score']:.4f}  "
            f"n_passed={h['n_passed']}"
        )

    top = result["top_strategies"]
    print(f"\nトップ戦略数: {len(top)}")
    for i, s in enumerate(top[:5]):
        r = s["robustness"]
        print(
            f"  [{i+1}] score={s['final_score']:.4f}  "
            f"sharpe={r['sharpe']:.3f}  "
            f"cagr={r['cagr']:.3f}  "
            f"max_dd={r['max_dd']:.3f}  "
            f"worst_s={r['worst_sharpe']:.3f}  "
            f"tail={r['tail_return']:.4f}"
        )
        print(f"       params={s['params']}")
        print(f"       per_split sharpes={[round(f['sharpe'],2) for f in s['per_split']]}")

    # 決定論チェック: 同一入力 → 同一出力
    result2 = run_evolution(price_df, base_params, cfg=cfg)
    scores1 = [s["final_score"] for s in result["top_strategies"]]
    scores2 = [s["final_score"] for s in result2["top_strategies"]]
    print(f"\n決定論チェック: {'PASS' if scores1 == scores2 else 'FAIL'}")


if __name__ == "__main__":
    _example()
