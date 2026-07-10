# Study75 Point-in-Time (PIT) Audit

**日付**: 2026-07-10
**目的**: Study75A（Universe Generator）着手前に、lookahead bias・survivorship leakageの
混入経路を体系的に洗い出し、各項目についてSAFE/PARTIAL/UNSAFEを明示的に判定する。

---

## 1. リバランス日 vs ユニバーススナップショット日

### 設計

`rebalance_date` と `snapshot_date` を明確に分離した別概念として定義する:

- **`snapshot_date`**: ユニバース判定（ADV20・lot feasibility・listing_age）に使うデータの
  基準日。`rebalance_date` の**直前営業日（T-1）**に固定する。
- **`rebalance_date`**: 新しいユニバース構成が発効する日（月次第1営業日）。この日自体の終値・
  出来高は一切参照しない。

```
snapshot_date = prev_trading_day(rebalance_date)     # T-1
ADV20window   = trading_days in (snapshot_date - 20営業日, snapshot_date]  # snapshot_date含む
listing_age   = (snapshot_date - first_seen).days
lot_cost      = Close[snapshot_date] × LOT_SIZE
```

この設計により、`rebalance_date`当日の始値で執行するという既存プロジェクトの取引執行規約
（"翌日寄付執行"・composite_alpha_bt.pyの`shift(1)`規律）と対称的に、
**ユニバース判定自体もT-1情報のみで完結**する。既存の`JPXCalendar.prev_trading_day()`
（`src/market/jpx_calendar.py`）をそのまま再利用できる。

### 判定: **SAFE**（設計として同日終値情報を一切使わない。実装での遵守はStudy75A実装で保証）

---

## 2. 新規上場（IPO）の取り扱い

### 設計

`min_history_days = 60` を導入。`snapshot_date` 時点で `listing_age < 60` の銘柄は
候補から除外する。

- `first_seen`: Universeイベントログ（`universe_events.parquet`）のADDイベント日、または
  日次価格データでの初出現日のいずれか早い方（イベントログが正・価格データは検証用）。
- `listing_age = snapshot_date - first_seen`（日数）。
- 除外理由は `excluded_reason="insufficient_history"` として診断ファイルに必ず記録する
  （黙って除外しない）。

**根拠**: IPO直後は値幅制限緩和・初値形成の歪み・出来高の一時的異常（初日出来高が
その後の実態と乖離）が典型的に発生するため、60営業日程度のバーンイン期間を置くことは
モメンタム系文献・既存プロジェクトのentry filter設計思想（Study22-24 Entry Causality Gate系）
とも整合する一般的な実務慣行。

### 判定: **SAFE**（機械的なT-1判定・除外理由を必ず記録するため見えない除外がない）

---

## 3. Lot Feasibility PIT監査（最重要・要結論）

### 調査内容

「単元株数（trading unit / lot size）」「上場株式数」「株式分割履歴」が、本データセット上で
**時点ごとに**取得可能かを実データで確認した。

**確認結果**:

| データ項目 | 保有状況 |
|---|---|
| 単元株数（trading unit size） | **保有していない**。`/v2/equities/bars/daily`（raw列: `Date,Code,O,H,L,C,UL,LL,Vo,Va,AdjFactor,AdjO,AdjH,AdjL,AdjC,AdjVo`）・`/v2/equities/master`（`CoName,S17,S17Nm,S33,S33Nm,ScaleCat,Mkt,MktNm,Mrgn,MrgnNm,ProdCat`）のいずれにも単元株数フィールドは存在しない。 |
| 上場株式数（listed shares outstanding） | 保有していない（同上・価格・出来高・調整係数のみ）。 |
| 株式分割履歴 | **部分的に保有**。`AdjustmentFactor`（日次）が変化した日を株式分割・併合等の企業アクション日として検出できる（比率そのものも取得可能）。ただし「単元株数の変更」という行為自体を直接示すフィールドではない（分割と単元変更は別の企業アクションだが、2007-2018年のTSE単元株数統一キャンペーンでは併合により単元変更を伴うケースが多かった）。 |

**実データでの検証**（`AdjustmentFactor`変化イベント数・年別・対象銘柄数）:

| 年 | 変化イベント数 | 対象銘柄数（ユニーク） |
|---|---|---|
| 2016 | 416 | 207 |
| 2017 | 1,296 | **635** |
| 2018 | 752 | **366** |
| 2019 | 314 | 156 |
| 2020 | 264 | 130 |

2017-2018年に企業アクション頻度が明確に急増している（635件・366件 vs 前後の年130-207件）。
これは東証が2007年から進め2018年10月に完了した「単元株数統一（100株への集約）」キャンペーンの
最終フェーズと時期的に一致する（周知の市場構造上の事実。本データはこの事実を追加で裏付ける）。

### 結論: **PARTIAL**

- **2018年10月以降**: 東証が単元株数を全銘柄100株に統一完了しているため、
  `LOT_SIZE=100`（`composite_alpha_bt.py`の既存デフォルトと同一）を全銘柄一律で適用することは
  **SAFE**（歴史的事実として単元株数のばらつきが存在しない）。
- **2016年7月〜2018年10月（本データセットの最初の約2年3ヶ月）**: 単元株数統一キャンペーンの
  終盤にあたり、一部銘柄では単元株数が100株以外（1株・10株・1000株等）だった可能性を
  **本データセットだけでは否定できない**。この期間について`LOT_SIZE=100`を一律適用することは
  厳密には未検証の仮定である。

### 対応方針（Study75A実装への反映）

1. `LOT_SIZE=100`を全期間で一律使用する（データ上の代替手段がなく、`composite_alpha_bt.py`の
   既存規約とも整合するため）。
2. **既知の限界として明示的に記録する**: 診断ファイル（`study75_universe_diagnostics.parquet`）に
   `lot_feasible`列とは別に、2018-10-01以前のスナップショット日については
   `lot_size_verified=False`（暗黙のフラグとしてレポートに明記。列追加は本タスクでは行わない・
   フォローアップ候補として記録）。
3. Study75Bのbias measurement解釈時、2016-07〜2018-10期間のlot関連の差分（bias attribution内訳）は
   相対的に信頼度が低いことをレポート内で必ず注記する。
4. **フォローアップ推奨（本タスクのスコープ外）**: J-Quants以外のソース（東証公式の単元株数変更
   履歴等）で2016-2018年の単元株数を補完できれば、この限界は解消できる。

---

## 4. ADV20 PIT監査

### 確認内容

Study75A（§2実装）でのADV20計算が、`snapshot_date`より未来のデータを一切参照しないことを
設計・実装の両面で確認する。

**設計**:
```
window = daily_bars[(daily_bars.Date <= snapshot_date) の直近20営業日]
adv20  = mean(Close × Volume) over window
```

`snapshot_date`自体は§1の通り`rebalance_date`のT-1に固定されるため、`window`の右端は
常に`rebalance_date`より過去に位置する。`.Date <= snapshot_date`という不等号の向きを
実装レビューで必ず確認する（`< `ではなく`<=`＝snapshot_date当日を含めるのは正しい。
snapshot_date自体は既にrebalance_dateより過去の確定済み情報であるため）。

**Study75 Universe Design（記述分析・2026-07-10・前回タスク）での先行実装との整合性**:
前回の記述分析スクリプト（`src/scripts/study75_universe_design_analysis.py`）でも同様の
trailing window設計（`window_start = year_end - 40日, window <= year_end`）を採用しており、
本監査の設計はその実装パターンを踏襲・一般化したものである。

### 判定: **SAFE**（`<=snapshot_date`のtrailing windowのみを参照する設計。Study75A実装時に
コードレビューで`<=`の向きを再確認することを実装チェックリストに追加する）。

---

## 5. 監査結果まとめ

| 項目 | 判定 | 備考 |
|---|---|---|
| 1. rebalance_date vs snapshot_date分離 | SAFE | T-1固定・同日終値不参照 |
| 2. IPO最小上場期間（60営業日） | SAFE | 除外理由を必ず記録 |
| 3. Lot feasibility 履歴データ | **PARTIAL** | 2018-10以降SAFE・以前は単元株数統一キャンペーン中で未検証 |
| 4. ADV20 trailing window | SAFE | `<=snapshot_date`のみ参照 |

**総合判定**: Study75A/Bの実施は上記PARTIAL項目（§3）を明示的な既知の限界として記録した上で
進めて問題ない。lookahead bias・survivorship leakageの新規混入経路は特定されなかった。
