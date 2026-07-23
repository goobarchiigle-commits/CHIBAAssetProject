# Study82E Proposal — PEAD Reverse Root Cause Audit（起案書・実装なし）

**日付**: 2026-07-22
**性格**: 起案書のみ。実装・BT実行・新規データ取得・ロードマップ変更は一切行わない。
```
Proposal only.
× implementation
× backtest
× portfolio optimization
× parameter tuning
× new data collection
× roadmap modification
× new alpha proposal（root cause未確定の間は禁止）
```
**正典**: Study82 Phase D実測（`reports/study82_phase_d_pead_alpha.md` / `backtests/study82_phase_d_pead_events_2026-07-21.csv`・30,952件イベント台帳）/ PEAD文献シンセシス（`reports/pead_literature_synthesis_2026-07-22.md`）。
**位置づけ**: Study82続篇。Phase D「FAIL」の原因分解。新規アルファ探索ではなくpost-mortem decomposition。

---

## 1. 背景

Study82 Phase D実測（PIT-safe・全銘柄・40d固定保有・コスト調整後）:

```
Positive群: n=17,959, cost-adj return(40d)=+0.619%, t=5.3
Negative群: n=12,993, cost-adj return(40d)=+1.157%, t=7.9
Spread(Positive − Negative) = -0.538%   ← 古典的PEAD理論と符号が逆転
```

両群とも単独では強く有意（t=5.3 / 7.9）——シグナル不在ではなく、**符号・構造の誤読**の可能性を示唆。Study95（CSモメンタムKILL）・Study99（Sector×Fujiko filter未確認）・Study83（TSMOM REJECT）でも同系統（モメンタム系構築での負の結果）が繰り返し観測されており、単発の実装ミスでなく構造的パターンの疑いがある。

---

## 2. 目的

```
Determine: WHY did PEAD reverse?
NOT: alpha search
IS : post-mortem decomposition
```

---

## 3. 厳格ルール

- Proposal only。実装禁止。
- 新規データ取得禁止。
- 既存30,952件イベント台帳のみ再利用。
- 新規ポートフォリオ最適化禁止。
- パラメータチューニング禁止。
- ロードマップ変更禁止。
- 明確なroot causeが特定されない限り、新規アルファ提案禁止。

---

## 4. 分解軸 実現可能性監査（新規データ取得ゼロ制約下）

既存キャッシュ資産を実地確認した結果（`data/jquants/cache/daily/*.parquet`＝Vo/Va/AdjC等、`data/jquants/cache/fins_summary/*.json`＝ShOutFY/TA/Eq/CFO/NP等、`data/jquants/processed/TOPIX.parquet`）、8軸の実現可能性は以下の通り不均一。**Tier1は全軸FEASIBLE（新規取得ゼロ）。Tier2品質軸は正式指標が算出不可＝近似のみ**。

### Tier1（必須）

| # | 軸 | データ源 | 判定 | 備考 |
|---|---|---|---|---|
| 1 | Size | daily cache AdjC(disc_date近傍) × fins_summary ShOutFY | **FEASIBLE**（結合のみ） | 時価総額proxy。ShOutFYは会計期末時点値でイベント時点とのズレ（最大1年）あり＝誤差要因として明記必須。真のTOPIX規模区分（ScaleCat）は`/v2/equities/master`未キャッシュ＝新規API呼び出しに該当し**BLOCKED**（ルール違反）。proxyで代替する。 |
| 2 | Liquidity(ADV) | daily cache Vo/Va、event前60営業日平均 | **FEASIBLE**（ゼロコスト） | |
| 3 | Time period | 既存ledger `disc_date`列 | **FEASIBLE**（結合不要） | 2016-19/2020-22/2023-26の3区分でそのままスライス可 |
| 4 | Market regime | `TOPIX.parquet`（既存）から200MA算出、disc_date時点判定 | **FEASIBLE**（ゼロコスト） | |

### Tier2（条件付き）

| # | 軸 | データ源 | 判定 | 備考 |
|---|---|---|---|---|
| 5a | F-Score | fins_summary: NP/TA(ROA)・CFO・ΔROA・CFO>NP(accrual proxy)・Sales/TA(回転率) | **PARTIAL**（5-6/9項目相当） | LT負債内訳・流動資産負債内訳・COGSが元データに無く、レバレッジΔ・流動比率Δ・売上総利益率Δは**算出不可**。公式Piotroski(2000)と非同一の簡易代理指標——明記必須。 |
| 5b | GP/A | fins_summary: COGS列なし | **BLOCKED**（正式版） | 代替＝OP/TA（営業利益/資産）で近似する場合は「Novy-Marx GP/Aの代理・非同一指標」と明記必須。 |
| 6 | Surprise strength | 既存ledger `surprise_pct`列（連続値） | **FEASIBLE**（ゼロコスト） | Top30/10/5/1%はそのままpercentile区分可 |

**結論（Tier2）**: 品質条件軸は近似指標のみ利用可能。CASE4（Quality×Event proposal）判定の主根拠には使わず、参考情報止まりとする。正式F-Score/GP/A取得は新規データ収集に該当しBLOCKED。

### Tier3（診断のみ）

| # | 軸 | データ源 | 判定 | 備考 |
|---|---|---|---|---|
| 7 | Quarter type | 既存ledger `cur_per_type`列（1Q/2Q/3Q/FY） | **FEASIBLE**（そのまま利用） | |
| 8 | Earnings gap | fins_summary `DiscTime`（寄付前/引け後判定） × daily cache OHLC | **FEASIBLE**（結合のみ） | 分単位の発表時刻判定ロジックは既存`study82_phase_d_pead_alpha.py`内`compute_event_return`を再利用想定。新規ロジック追加は「実装」に該当するため、本Proposalでは設計提示のみ——反映は別途承認するStudy82E実装フェーズで行う。 |

---

## 5. 評価指標（各スライスで報告）

- Positive return / Negative return / Spread
- t統計量
- サンプル数
- 勝率（hit ratio）
- IC（feasible な場合のみ）
- Monotonicity（分位間の単調性）

## 6. ガバナンス指標（新規必須）

1. Portfolio spread
2. Stock-level hit ratio
3. Decile monotonicity
4. Regime stability

---

## 7. 判定木

```
CASE 1: 全スライスで逆転が残存
        → PEAD family = TERMINAL

CASE 2: 特定条件下でのみ逆転が消失
        → 単一の狭義後継仮説を1件のみ定式化

CASE 3: 大型株のみで逆転消失
        → Satellite candidate

CASE 4: 品質条件付きでのみ逆転消失
        → Quality × Event proposal
          （§4の通りTier2は近似指標——判定確度を割り引くこと明記）

CASE 5: 時期・レジーム依存でのみ逆転消失
        → 構造的市場説明（Structural market explanation）
```

## 8. 成果物

1. Root-cause ranking
2. 逆転の説明（Explanation of reversal）
3. 推奨区分:
   - TERMINAL
   - CONDITIONAL SURVIVAL
   - SUCCESSOR PROPOSAL

**禁止**: 実装提案・Study103再実行・新規momentum study提案。

---

## 9. 未解決・リスク

- Tier2品質軸は正式指標と非同一（近似）——CASE4判定は確度を割り引いて解釈すること。
- Size軸はShOutFY（会計期末時点）とイベント時点のタイムラグ誤差あり。
- 8軸×4-5区分の細分化でセル数が爆発しうる——`min_trade_required=5`未満のセルはt検定を報告せず「サンプル不足」表記に留める。
- 本Proposal自体は新規Study番号のロードマップ登録前——登録はASK_FIRST対象、本書では未実施。

## 10. 次アクション

承認 → Study82E実装起案（ASK_FIRST）へ。実装時も「既存イベントCSV読み込み＋既存キャッシュ（daily/fins_summary/TOPIX.parquet）結合のみ、新規API呼び出しなし」の制約を維持する。

*作成: CLD (Fable 5)・2026-07-22。BT・コード変更・データ取得なし。*
