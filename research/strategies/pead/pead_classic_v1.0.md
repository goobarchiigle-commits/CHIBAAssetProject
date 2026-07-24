---
strategy: PEAD Classic
version: "1.0"
status: Frozen
verdict: UNTESTED
role: Calibration Benchmark
parent: none
derived_from: [Bernard&Thomas1989, FOS1984, CJL1996]
rgp: post-earnings underreaction / drift
conventions: common_conventions_v1.0
created: 2026-07-24
origin: research/studies/Study082/pead_v1_spec_3proposals_2026-07-24.md（1案）
---

# PEAD-J Classic v1.0 — 古典ベンチマーク（Bernard-Thomas/FOS直系）

**仮説**: 日本株に古典的SUE型PEADが存在する（H0: D10−D1スプレッド ≤ 0）。
**事前予測（登録済み）**: Study82F Case C確定に基づき **FAIL（spread ≤ 0）を予測**。本仕様書の価値は勝つことではなく、全改善研究の**較正原点**となること。FAILでも永続（削除・改変禁止）。

## 仕様（固定値）

| 項目 | 仕様 |
|---|---|
| Universe | Study75 PITユニバース全銘柄 + 株価 ≥ 100円・ADV20 ≥ 5,000万円（執行可能性最低限のみ。文献同様、市場区分・時価総額フィルタなし） |
| Event | 四半期決算短信（`DocType`が`FinancialStatements`を含む・訂正除外）。`CurPerType ∈ {1Q,2Q,3Q,FY}` |
| Surprise | **SUE = (EPS_q − EPS_{q−4}) / σ(EPS_q − EPS_{q−4})**、σ=直近8四半期の季節差分標準偏差（CJL1996式・コンセンサス不要）。8四半期未満は除外。EPS=`/fins/summary`の`EPS`（連結優先・単体のみはNCEPS） |
| Ranking | **前四半期のSUE分布**で十分位カットオフ確定（当四半期分布使用=look-ahead・禁止）。D10=最上位 |
| Signal | 研究: D10ロング−D1ショートのスプレッド。live検証: D10ロングオンリー |
| Entry | T0+1営業日 寄付 |
| Exit | エントリーから**60営業日**後 寄付（FOS/B&T標準） |
| Benchmark調整 | CAR = 個別リターン − TOPIX同期間リターン（B&Tサイズ十分位マッチングは日本で再現困難のためTOPIX単純化・v1.0固定） |
| Primary判定 | D10−D1スプレッド（60d・コスト後）> 0 かつ NW-t ≥ 2.0 → PASS |

## 再現手順（API）

1. `/fins/summary`全銘柄収集（conventions §1収集規約）
2. 銘柄×四半期EPS系列 → SUE計算（≥8四半期）
3. 各暦四半期末に前四半期SUE分布から十分位境界を保存 → 当四半期イベントに適用
4. T0+1寄付〜T0+61寄付のコスト後リターン → D10/D1群平均・スプレッド・NW-t
