"""
execution/execution_simulator.py
kabu API の発注制約をバックテスト内で完全再現する Execution Simulator

【再現する制約】
  1. LOT_SIZE          : 東証標準単元株（100株）の倍数チェック
  2. REJECT_CASH       : 発注額 > 利用可能資金
  3. REJECT_PRICE      : 1単元コスト > 配分上限（資金 × max_single_weight）
  4. REJECT_MIN_LIQUIDITY: 流動性不足（1日売買代金 < 閾値）
  5. REJECT_DUPLICATE  : 当日すでに同銘柄に発注済み
  6. PARTIAL_FILL      : 資金は足りるが複数単元の一部しか買えない場合

【戻り値】
  ExecutionResult dataclass:
    status   : "FILLED" / "REJECT_LOT" / "REJECT_CASH" / "REJECT_PRICE" /
               "REJECT_LIQUIDITY" / "REJECT_DUPLICATE" / "PARTIAL"
    filled_qty  : 実際に約定した株数（0 の場合は失注）
    filled_amt  : 約定金額
    slippage    : スリッページ込みの実約定単価
    reject_reason: 失注理由（FILLED の場合は空文字）

【バックテストへの組み込み例】

    from execution.execution_simulator import ExecutionSimulator

    sim = ExecutionSimulator(
        capital          = 2_000_000,
        max_single_weight= 0.25,
        slippage_rate    = 0.001,   # 0.1%
        commission_rate  = 0.00055, # 0.055%
    )

    for date, orders in signal_orders:
        sim.reset_day()   # 1日の開始時に呼ぶ（重複ロックをリセット）

        for order in orders:
            result = sim.simulate(
                symbol      = order.symbol,
                price       = order.price,
                target_qty  = order.qty,
                available_cash = current_cash,
                daily_volume_yen = order.daily_volume_yen,  # オプション
            )
            if result.is_filled:
                current_cash  -= result.filled_amt
                position[order.symbol] += result.filled_qty

            # DailyStateLogger に skipped 理由を記録
            if result.status.startswith("REJECT"):
                skip_counter[result.status] += 1
"""

from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from dataclasses import dataclass, field


# ------------------------------------------------------------------ #
# 定数
# ------------------------------------------------------------------ #
LOT_SIZE          = 100     # 東証標準単元株数
MIN_LIQUIDITY_YEN = 5_000_000_000  # 流動性フィルター（5B 円/日）


# ------------------------------------------------------------------ #
# 結果データクラス
# ------------------------------------------------------------------ #
@dataclass
class ExecutionResult:
    status:        str    # "FILLED" / "PARTIAL" / "REJECT_*"
    filled_qty:    int    = 0
    filled_amt:    float  = 0.0
    slippage:      float  = 0.0    # スリッページ額（円）
    commission:    float  = 0.0    # 手数料（円）
    reject_reason: str    = ""

    @property
    def is_filled(self) -> bool:
        return self.status in ("FILLED", "PARTIAL")

    @property
    def total_cost(self) -> float:
        """総コスト（約定額 + スリッページ + 手数料）"""
        return self.filled_amt + self.slippage + self.commission


# ------------------------------------------------------------------ #
# シミュレーター本体
# ------------------------------------------------------------------ #
class ExecutionSimulator:
    """
    kabu API の発注制約をバックテスト内で再現するシミュレーター。

    Args:
        capital:           初期資本（配分上限の基準）
        max_single_weight: 1銘柄への最大配分比率（デフォルト 0.25 = 25%）
        slippage_rate:     スリッページ率（デフォルト 0.001 = 0.1%）
        commission_rate:   手数料率（デフォルト 0.00055 = 0.055%）
        lot_size:          単元株数（東証標準は 100）
        min_liquidity_yen: 流動性フィルター（1日売買代金の下限、円）
    """

    def __init__(
        self,
        capital:            float = 2_000_000,
        max_single_weight:  float = 0.25,
        slippage_rate:      float = 0.001,
        commission_rate:    float = 0.00055,
        lot_size:           int   = LOT_SIZE,
        min_liquidity_yen:  float = MIN_LIQUIDITY_YEN,
    ):
        self.capital           = capital
        self.max_single_weight = max_single_weight
        self.slippage_rate     = slippage_rate
        self.commission_rate   = commission_rate
        self.lot_size          = lot_size
        self.min_liquidity_yen = min_liquidity_yen

        # 当日の発注済み銘柄セット（reset_day() でリセット）
        self._ordered_today: set[str] = set()

    def reset_day(self) -> None:
        """1 営業日の開始時に呼ぶ。当日発注ロックをリセットする。"""
        self._ordered_today.clear()

    def simulate(
        self,
        symbol:           str,
        price:            float,
        target_qty:       int,
        available_cash:   float,
        daily_volume_yen: float | None = None,
    ) -> ExecutionResult:
        """
        1 銘柄の発注をシミュレートして ExecutionResult を返す。

        Args:
            symbol:           銘柄コード（例: "8035.T"）
            price:            参照株価（前日終値など）
            target_qty:       発注したい株数
            available_cash:   現在の利用可能資金（円）
            daily_volume_yen: 1日売買代金（円）。None の場合は流動性チェックをスキップ。

        Returns:
            ExecutionResult
        """
        # ── 制約1: 重複発注チェック ────────────────────────────────
        if symbol in self._ordered_today:
            return ExecutionResult(
                status="REJECT_DUPLICATE",
                reject_reason=f"{symbol}: 当日すでに発注済み",
            )

        # ── 制約2: 単元株チェック ──────────────────────────────────
        if target_qty <= 0 or target_qty % self.lot_size != 0:
            return ExecutionResult(
                status="REJECT_LOT",
                reject_reason=(
                    f"{symbol}: 発注数量 {target_qty} が単元株({self.lot_size})の倍数でない"
                ),
            )

        # ── 制約3: 流動性チェック ──────────────────────────────────
        if daily_volume_yen is not None and daily_volume_yen < self.min_liquidity_yen:
            return ExecutionResult(
                status="REJECT_LIQUIDITY",
                reject_reason=(
                    f"{symbol}: 1日売買代金 ¥{daily_volume_yen:,.0f} "
                    f"< 下限 ¥{self.min_liquidity_yen:,.0f}"
                ),
            )

        # ── 制約4: 1単元コスト > 配分上限 ─────────────────────────
        alloc_cap   = self.capital * self.max_single_weight
        one_lot_cost = price * self.lot_size
        if one_lot_cost > alloc_cap:
            return ExecutionResult(
                status="REJECT_PRICE",
                reject_reason=(
                    f"{symbol}: 1単元 ¥{one_lot_cost:,.0f} "
                    f"> 配分上限 ¥{alloc_cap:,.0f} "
                    f"（株価 ¥{price:,.0f}）"
                ),
            )

        # ── 制約5: 資金チェック（全量）───────────────────────────
        order_value = price * target_qty
        if order_value > available_cash:
            # 一部だけ買えるか試みる（PARTIAL FILL）
            affordable_lots = int(available_cash // one_lot_cost)
            if affordable_lots <= 0:
                return ExecutionResult(
                    status="REJECT_CASH",
                    reject_reason=(
                        f"{symbol}: 発注額 ¥{order_value:,.0f} "
                        f"> 利用資金 ¥{available_cash:,.0f}。"
                        f"1単元も買えない（¥{one_lot_cost:,.0f}/単元）"
                    ),
                )
            # PARTIAL: 買える単元数まで減らして発注
            actual_qty = affordable_lots * self.lot_size
            return self._make_filled(
                symbol, price, actual_qty, status="PARTIAL",
                reason=f"{target_qty}株→{actual_qty}株（資金制約で減量）",
            )

        # ── FILLED（全量約定）────────────────────────────────────
        self._ordered_today.add(symbol)
        return self._make_filled(symbol, price, target_qty, status="FILLED")

    def _make_filled(
        self,
        symbol:    str,
        price:     float,
        qty:       int,
        status:    str = "FILLED",
        reason:    str = "",
    ) -> ExecutionResult:
        """約定結果を組み立てる（スリッページ・手数料を計算）。"""
        self._ordered_today.add(symbol)
        filled_amt = price * qty
        slippage   = filled_amt * self.slippage_rate
        commission = filled_amt * self.commission_rate
        return ExecutionResult(
            status        = status,
            filled_qty    = qty,
            filled_amt    = filled_amt,
            slippage      = slippage,
            commission    = commission,
            reject_reason = reason,
        )

    # ------------------------------------------------------------------ #
    # バッチ処理ユーティリティ
    # ------------------------------------------------------------------ #
    def simulate_batch(
        self,
        orders: list[dict],
        available_cash: float,
    ) -> tuple[list[ExecutionResult], dict[str, int]]:
        """
        複数銘柄の注文を一括シミュレートする。

        Args:
            orders: [{"symbol": str, "price": float, "qty": int,
                       "daily_volume_yen": float | None}, ...]
            available_cash: 当初利用可能資金

        Returns:
            (results, skip_counts):
                results     : 各注文の ExecutionResult リスト
                skip_counts : {"REJECT_PRICE": n, "REJECT_CASH": n, ...}
        """
        results:     list[ExecutionResult] = []
        skip_counts: dict[str, int]        = {}
        cash = available_cash

        for order in orders:
            result = self.simulate(
                symbol           = order["symbol"],
                price            = order["price"],
                target_qty       = order["qty"],
                available_cash   = cash,
                daily_volume_yen = order.get("daily_volume_yen"),
            )
            results.append(result)

            if result.is_filled:
                cash -= result.total_cost
            else:
                skip_counts[result.status] = skip_counts.get(result.status, 0) + 1

        return results, skip_counts

    # ------------------------------------------------------------------ #
    # 診断ユーティリティ
    # ------------------------------------------------------------------ #
    @staticmethod
    def max_buyable_qty(

        price:            float,
        available_cash:   float,
        max_alloc:        float,
        lot_size:         int = LOT_SIZE,
    ) -> int:
        """
        価格・資金・配分上限から買える最大株数（単元株単位）を返す。

        バックテスト側でターゲット数量を決める前に呼んで、
        発注前に REJECT_PRICE / REJECT_CASH を防ぐ。
        """
        one_lot_cost = price * lot_size
        if one_lot_cost > max_alloc or one_lot_cost > available_cash:
            return 0
        effective_cap = min(available_cash, max_alloc)
        return int(effective_cap // one_lot_cost) * lot_size


# ------------------------------------------------------------------ #
# FillRateTracker — バックテスト全体の約定率を集計
# ------------------------------------------------------------------ #
class FillRateTracker:
    """
    ExecutionSimulator の結果を累積して fill_rate / effective_exposure を測定する。

    【使い方】
        tracker = FillRateTracker(capital=2_000_000)

        for date, orders, cash in backtest_loop:
            sim.reset_day()
            results, skips = sim.simulate_batch(orders, cash)
            tracker.record(date, results, equity=current_equity)

        report = tracker.report()
        tracker.print_report()
    """

    def __init__(self, capital: float = 2_000_000):
        self.capital = capital
        self._days:          list[dict]              = []
        self._all_results:   list[ExecutionResult]   = []

    def record(
        self,
        date:    object,
        results: list[ExecutionResult],
        equity:  float,
    ) -> None:
        """1 営業日分の結果を記録する。"""
        total   = len(results)
        filled  = sum(1 for r in results if r.is_filled)
        amt     = sum(r.filled_amt for r in results if r.is_filled)

        reject_counts: dict[str, int] = {}
        for r in results:
            if not r.is_filled:
                reject_counts[r.status] = reject_counts.get(r.status, 0) + 1

        self._days.append({
            "date":              str(date),
            "equity":            equity,
            "total_orders":      total,
            "filled_orders":     filled,
            "fill_rate":         filled / total if total > 0 else float("nan"),
            "filled_amount":     amt,
            "effective_exposure": amt / equity if equity > 0 else 0.0,
            **reject_counts,
        })
        self._all_results.extend(results)

    def report(self) -> dict:
        """集計レポートを dict で返す。"""
        if not self._days:
            return {}

        df      = pd.DataFrame(self._days)
        total   = len(self._all_results)
        filled  = sum(1 for r in self._all_results if r.is_filled)
        partial = sum(1 for r in self._all_results if r.status == "PARTIAL")

        reject_types = [
            "REJECT_PRICE", "REJECT_CASH", "REJECT_LOT",
            "REJECT_LIQUIDITY", "REJECT_DUPLICATE",
        ]
        reject_counts = {
            rt: sum(1 for r in self._all_results if r.status == rt)
            for rt in reject_types
        }

        # target exposure = capital の 1/3 ずつ 3 ポジション想定
        target_exp = 1.0 / 3 * 3   # ≒ 1.0 に近いが max 3 ポジション前提では変動する

        avg_eff_exp  = float(df["effective_exposure"].mean())
        fill_rate    = filled / total if total > 0 else 0.0

        return {
            "period_start":          df["date"].iloc[0],
            "period_end":            df["date"].iloc[-1],
            "total_days":            len(df),
            "total_orders":          total,
            "filled_orders":         filled,
            "partial_orders":        partial,
            "fill_rate":             round(fill_rate, 4),
            "avg_effective_exposure": round(avg_eff_exp, 4),
            "reject_counts":         reject_counts,
            "reject_rate_price":     round(reject_counts["REJECT_PRICE"] / total, 4) if total else 0,
            "reject_rate_cash":      round(reject_counts["REJECT_CASH"]  / total, 4) if total else 0,
            "reject_rate_lot":       round(reject_counts["REJECT_LOT"]   / total, 4) if total else 0,
        }

    def print_report(self) -> None:
        """レポートをコンソールに出力する。"""
        r   = self.report()
        SEP = "=" * 50
        print()
        print(SEP)
        print("  Fill Rate Report")
        print(SEP)
        print(f"  期間: {r['period_start']} 〜 {r['period_end']}")
        print(f"  総発注数: {r['total_orders']} 件")
        print()
        fill_flag = "✅" if r["fill_rate"] >= 0.85 else "❌"
        print(f"  {fill_flag} fill_rate            : {r['fill_rate']:.1%}")
        print(f"     ├ FILLED              : {r['filled_orders']} 件")
        print(f"     ├ PARTIAL             : {r['partial_orders']} 件")
        print(f"     └ REJECTED            : {r['total_orders'] - r['filled_orders']} 件")
        print()
        print("  REJECT 内訳:")
        for k, n in r["reject_counts"].items():
            if n > 0:
                pct = n / r["total_orders"]
                print(f"     {k:<25}: {n:4d}件 ({pct:.1%})")
        print()
        print(f"  avg_effective_exposure : {r['avg_effective_exposure']:.1%}")
        if r["fill_rate"] < 0.85:
            dominant = max(r["reject_counts"], key=r["reject_counts"].get)
            print(f"\n  ⚠ fill_rate < 85%: 主因 = {dominant}")
        print(SEP)
        print()
