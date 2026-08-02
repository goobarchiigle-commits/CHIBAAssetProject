# Observation Schema — Study80A 観測基盤スキーマ定義

**日付**: 2026-07-04
**目的**: Study74〜74B-RCAで発生した「見送り候補の個別レコードが永続化されておらず、後から検証できない」問題を恒久的に解消する。今後のStudy81以降がこのスキーマに従って生成されたJSONを読むだけで、追加BTなしに全解析を実行できるようにする。

---

## エンジン変更概要（`composite_alpha_bt.py`）

すべて**観測専用・既存dict literalへのキー追加または新規1リスト追加**。制御フロー・トレード判定ロジックは一切変更していない（Parity検証済み・`reports/parity_report.md`参照）。

| 変更箇所 | 内容 |
|---|---|
| 日次コンテキストスナップショット（新規計算・候補ループ直前） | `_obs_selected_syms` / `_obs_selected_scores` / `_obs_position_weights` / `_obs_cash_before` / `_obs_used_slots` / `_obs_max_slots` / `_obs_regime` / `_obs_candidate_count` を1日1回計算 |
| 候補ごとの追加算出 | `momentum_63d_pct`（63日リターン・Close比較）/ `sector`（trade_symsから引当） |
| `_missed_cands`（CAP_MISS） | 上記全フィールド + `skip_reason="CAP_MISS"` を追加 |
| `_skip_detail`（SECTOR_CAP/CLUSTER_CAP/GROSS_EXPOSURE） | 同上 + `skip_reason`を各理由に応じて追加 |
| `_rejected_by_lot_detail`（LOT_REJECT） | 同上 + `skip_reason="LOT_REJECT"` |
| `_admitted_by_ratio_detail`（lot cost ratio救済） | 同上 + `skip_reason="ADMITTED_BY_RATIO"`（実際には約定成立・見送りではない点に注意） |
| `_selected_cands`（**新規リスト**） | SELECTEDされた候補を同一スキーマで記録（見送り側との直接比較用） |

---

## JSONファイル別スキーマ

### 1. `backtests/trade_dataset_v2.json`

採用トレード（SELL確定分）+ 候補時点コンテキスト。

```
{
  "date": "YYYY-MM-DD",
  "n_trades": int,
  "trades": [
    {
      "symbol": str, "sector": str,
      "entry_date": str, "exit_date": str,
      "entry_idx": int, "exit_idx": int,
      "entry_price": float, "exit_price": float, "qty": int,
      "return_pct": float, "r_multiple": float, "pnl": float,
      "holding_days": int,
      "entry_atr_pct": float, "entry_rsr": float, "entry_type": "fujiko"|"mean_rev",
      "exit_policy": "A", "addon_received": bool, "exit_reason": str,
      // ── v2拡張フィールド（_selected_candsから結合）──
      "candidate_rank": int,           // 当日の候補内順位(0=最上位)
      "candidate_count": int,          // 当日の全候補数
      "cash_before_entry": float,      // エントリー前の現金残高
      "slots_used": int,               // エントリー前の使用中スロット数
      "portfolio_exposure": float,     // エントリー前のポートフォリオ総エクスポージャー比率
      "entry_cluster_id": str,         // 同日シグナル群ID（=entry_date）
      "entry_sector": str,
      "entry_regime": "risk_off"|"normal"
    }, ...
  ]
}
```

### 2. `backtests/missed_candidates_full.json`

見送り候補の**全件**（集計ではなく生データ）。

```
{
  "date": "YYYY-MM-DD", "n_missed": int, "n_rescued_admitted_by_ratio": int,
  "missed_candidates": [
    {
      "date": str, "symbol": str, "rsr": float, "composite_score": float,
      "alpha": float, "atr_pct": float, "rank": int, "skip_reason": str,
      "momentum_63d_pct": float, "sector": str, "market_regime": str,
      "cash_before_entry": float, "used_slots": int, "max_slots": int,
      "selected_symbols": [str], "selected_scores": {symbol: float},
      "position_weights": {symbol: float}, "candidate_count_today": int,
      // ── Forward Return拡張（Part2で付与）──
      "forward_5": float, "forward_10": float, "forward_20": float,
      "forward_40": float, "forward_60": float,
      "mfe_pct": float, "mae_pct": float, "max_dd_pct": float,
      "holding_days_equivalent": int
    }, ...
  ],
  "rescued_candidates": [ ... 同スキーマ（ADMITTED_BY_RATIOのみ） ]
}
```

`skip_reason`の値: `CAP_MISS` / `SECTOR_CAP` / `CLUSTER_CAP` / `GROSS_EXPOSURE` / `LOT_REJECT`。

### 3. `backtests/forward_return_dataset.json`

見送り候補（607件）のForward Return専用ビュー（`missed_candidates_full.json`のサブセット、解析利便性のため独立ファイル化）。

```
{
  "date": "YYYY-MM-DD", "horizons": [5, 10, 20, 40, 60],
  "records": [
    {"date": str, "symbol": str, "skip_reason": str,
     "forward_5": float, "forward_10": float, "forward_20": float,
     "forward_40": float, "forward_60": float,
     "mfe_pct": float, "mae_pct": float, "max_dd_pct": float,
     "holding_days_equivalent": int}, ...
  ]
}
```

**forward_N定義**: 候補発生日のClose終値を基準(0%)として、N営業日後のCloseまでの騰落率。
**mfe_pct/mae_pct**: 発生日〜60営業日後までの窓における最大含み益/含み損（Highの最大値/Lowの最小値ベース）。
**holding_days_equivalent**: 窓内でCloseが最高値を付けた経過営業日数（「もし利確するなら何日目が最適だったか」の近似指標）。

### 4. `backtests/opportunity_cost_dataset.json`

```
{
  "date": "YYYY-MM-DD",
  "adopted_baseline": {"n": int, "mean_return_pct": float, "profit_factor": float,
                          "expectancy_yen": float, "win_rate_pct": float},
  "missed_candidates_forward20_overall": {"n": int, "mean_forward_20_pct": float},
  "opportunity_cost_by_sector": {sector: {"n": int, "mean_forward_20_pct": float, "median_forward_20_pct": float}},
  "opportunity_cost_by_regime": {"risk_off"|"normal": {...}},
  "opportunity_cost_by_rank": {"rank0"|"rank1-2"|"rank3+": {...}},
  "opportunity_cost_by_skip_reason": {skip_reason: {...}}
}
```

### 5. `backtests/correlation_dataset.json`

同日候補の集中度観測（新規ロジックなし・既存データの集計のみ）。

```
{
  "date": "YYYY-MM-DD", "n_multi_candidate_days": int,
  "avg_max_sector_concentration_pct": float,
  "avg_momentum_same_direction_pct": float,
  "daily_records": [
    {"date": str, "n_candidates": int, "distinct_sectors": int,
     "max_sector_concentration_pct": float, "momentum_same_direction_pct": float}, ...
  ]
}
```

---

## 未収録・既知の限界（Study81への申し送り）

- **portfolio_beta**: 今回は未実装（推定にはTOPIXとの回帰計算が必要で、単純な観測を超えるため本タスクのスコープ外と判断）。必要ならStudy81で`universe_raw`+TOPIX価格から追加BTなしに算出可能（設計のみ残す）。
- **Momentumの定義**: RSR自体が既にモメンタム由来の指標であり、エンジンには別建ての「momentum」列が存在しなかったため、新たに63日リターン（`momentum_63d_pct`）を観測用に追加した。Study56/74B等で使われた「alpha_score」は現行Production設定（`alpha_df=None`）では常に0であり、実質的に無効な特徴量である点に注意（Study81で他のmomentum定義に差し替える場合はこの列を上書きすること）。
- **entry_cluster_id**: 「同日シグナル群」の単純な代理として発生日そのものを使用。より精緻なクラスタリング（相関ベース等）はStudy81のスコープ。
