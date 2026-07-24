---
strategy: Quality MF Practical
version: "1.0"
status: Frozen
verdict: UNTESTED
role: Production Candidate
parent: quality_mf_classic_v1.0
derived_from: [FamaFrench2015(RMW=operating profitability), Ball et al.(cash-based operating profitability・Berkin&Swedroe経由), Piotroski2000, Sloan1996, HXZ2015(四半期更新)]
rgp: quality premium / junk avoidance（クロスセクション・四半期更新）
conventions: common_conventions_v1.0
created: 2026-07-24
origin: research/strategies/quality/literature_evidence_2026-07-24.md
---

# Quality Multi-Factor Practical v1.0 — 日本市場実戦型（`/v2/fins/summary`のみで完全再現可能）

**仮説**: J-Quants Standardで取得可能なフィールドのみで構成した品質コンポジットが日本株で超過リターンを持つ。
**親版とのdiff**: ①GP/A→**OP/TA**（営業利益収益性。RMW=Fama-French 2015系の文献支持・データギャップ解消）②F-Score 9項目→**F-Score-J 8項目**（取得不能2項目を除外し1項目を代替）③年次→**四半期リバランス**（HXZ四半期更新の文献支持・短信データ鮮度活用）。初版のためv1.0として独立起案・parentは系譜表示用。

## 採用ファクター（全て`/v2/fins/summary`実在フィールドのみ）

| # | ファクター | 定義（完全定量・フィールド実名） | 方向 |
|---|---|---|---|
| F1 | 営業収益性 | **OP / TA**（直近開示の本決算またはFY通期換算: 四半期累計OP×(4/経過四半期数)/TA） | 高いほど良 |
| F2 | 会計発生高 | **ACC = (NP − CFO) / TA**（CFO開示のある直近決算。CFO空欄期はスキップし直近CFO開示期を使用） | 低いほど良 |
| F3 | F-Score-J（0-8点） | ① NP/TA > 0 ② CFO > 0 ③ Δ(NP/TA) > 0 ④ CFO/TA > NP/TA ⑤ ΔEqAR > 0（自己資本比率上昇=レバレッジ低下代替）⑥ Δ(Sales/TA) > 0 ⑦ Δ(OP/Sales) > 0（営業利益率・売上総利益率代替）⑧ ShOutFY 前期比増加なし（希薄化なし）。Δは前年同期本決算比 | 高いほど良 |

除外した原典項目: 流動比率（データなし・代替なしで削除）。長期負債比率→⑤EqARで代替。売上総利益率→⑦営業利益率で代替。

## 仕様（固定値）

| 項目 | 仕様 |
|---|---|
| Universe | 東証プライム+スタンダード（`MarketCode ∈ {0111,0112}`）− 金融4業種 + 時価総額 ≥ 50億円（`ShOutFY`×前日終値）・ADV20 ≥ 1億円・株価 ≥ 100円・Eq > 0 |
| Scoring | F1・F2（符号反転）のパーセンタイルランク平均=コンポジット。**F3はフィルタ**: F-Score-J ≥ 6 のみ採用（Piotroskiの高スコア選抜を8点制に比例換算） |
| Selection | フィルタ通過銘柄のコンポジット上位五分位を**等ウェイト**ロング |
| 形成日 | **2月・5月・8月・11月の最終営業日**（年4回。決算短信集中月の翌月末=データ出揃い後）。使用データ=形成日時点で開示済み（`DiscDate` ≤ 形成日）の最新値のみ（PIT厳守） |
| Entry | 形成日翌営業日 寄付 |
| Exit | 次回形成日翌営業日 寄付でリバランス（**保有3ヶ月**・ランク外銘柄売却・回転コストは片道ごとにconventions §3適用） |
| 欠損規則 | F1-F2計算不能は除外。F3は判定可能項目のみで採点し**分母を判定可能項目数に縮小、6/8相当の比率(0.75)以上**を通過とする |
| Primary判定 | EWポートフォリオのコスト後月次超過リターン（vs TOPIX）> 0 かつ NW-t ≥ 2.0 → PASS。月次 n ≥ 60・平均保有銘柄数 ≥ 30（conventions上書き・明記） |
| 禁止事項（RGP越境・governance §9準拠） | RSRランキング追加・決算イベント/サプライズ条件追加・テーマ株条件追加・出来高急増条件追加・ブレイクアウト/トレンドフォロー条件追加 |

## 留保（事前登録)

- 平均保有銘柄数 < 30 の期が全体の20%超なら INCONCLUSIVE（分散不足・Study110A集中性知見への防御）。
- 四半期OPのFY換算（F1）は季節性歪みを持つ——本仕様は換算式を上記に固定し、歪みは判定に織り込む（補正禁止）。
- 実装・バックテストはユーザー承認後（ASK_FIRST）。
