---
document: Common Execution & Validation Conventions
version: "1.0"
status: Frozen
created: 2026-07-24
origin: research/studies/Study082/pead_v1_spec_3proposals_2026-07-24.md §1
---

# 共通執行・検証規約 v1.0

全Strategy Specificationが `conventions: common_conventions_v1.0` で参照する共通層。変更禁止（変更は v1.1 新版）。

## 1. データソース（J-Quants Standard・疎通確認済み）

| 項目 | エンドポイント | フィールド（実名・Study82 Phase0実測） |
|---|---|---|
| 決算開示 | `/v2/fins/summary` | `Code, DiscDate, DiscTime, DiscNo, DocType, CurPerType, EPS, OP, Sales, NP, FOP, FEPS, FSales, FNP, Eq, ShOutFY, AvgSh` |
| 株価 | `/v2/equities/daily_quotes` | 調整後OHLCV |
| 市場区分 | `/v2/listed/info` | `MarketCode`（0111=プライム, 0112=スタンダード, 0113=グロース） |
| 指数 | TOPIX日次 | ベンチマーク調整用 |

収集規約: 1.2秒/リクエストスロットリング・「失敗」と「空データ」の区別必須（Study82 PhaseD 429事故再発防止）。

## 2. イベント台帳規則

- イベント日 T0 = `DiscDate`。`DiscTime`によらず**エントリーは常にT0翌営業日以降**（lookahead安全側）。
- 訂正開示は新規イベント不採用。`DiscNo`で重複排除。
- 同一銘柄・保有期間内の後続イベント: 初回のみ採用（ピラミッディング禁止）。
- PIT: ユニバース所属判定 = Study75 PITユニバースのイベント月時点所属。

## 3. 執行・コスト（PARAMS_LOCKED準拠）

```
entry      = 指定営業日の寄付成行（BT=調整後始値）
exit       = 保有満了営業日の寄付成行
slippage   = 0.001   (0.1% 片道・往復適用)
commission = 0.00055 (0.055% 片道・往復適用)
sizing     = イベント等ウェイト（研究）/ live: max_positions=3・max_single_weight=0.25（CIRCUIT）
損切り      = 仕様書に明記なき限りタイムイグジットのみ（SL付与=別版起案）
```

## 4. 検証プロトコル（共通ゲート）

```
fresh_run必須（キャッシュ判定禁止・Study52規則）
効果量表記 = Spread / 95%CI / Newey-West t / n（Study82F様式）
IS = 2016-04〜2022-12 / OOS = 2023-01〜2026-06、oos_is_ratio ≥ 0.7
Right-censoring: 有効標本比率 ≥ 70%（Study82F 70%ルール）
n_min = 1,000イベント（全体）・サブ群 ≥ 200（仕様書側で上書き可・明記必須）
sanity: sharpe ≤ 3.0 / パラメータスイープ禁止（全パラメータ事前固定）
Multiple Testing: Primary判定は各仕様書1個のみ。Secondary以下は診断扱い
```
